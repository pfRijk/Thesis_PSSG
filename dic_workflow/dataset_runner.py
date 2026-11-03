"""
Shared per-dataset pipeline: RAW -> TIFF conversion, all manual-selection
steps (level, crop, ROI polygon, edge refinement, ROI shrink, grid axis,
exclusion, scale), and (optionally) the DIC correlation + output plots.

Used by:
  - run_dic/dic_real_images.py   -- interactive, runs DIC
  - validation/validate_dic.py   -- noise-floor and convergence tests; prepares
                                    the POIs and runs its own correlations

Every manual-selection step caches its result to a JSON file in
result_folder (crop.json, level.json, scale.json, poi_polygon.json,
exclusion.json, grid_axis.json, align_markers.json). With interactive=True,
a missing JSON opens the corresponding click UI and saves the result. With
interactive=False, a missing JSON raises RuntimeError instead of prompting
-- run a dataset with interactive=True first to produce it.

With do_dic=False, the function stops after saving frame_00_POIs.png (i.e.
it resolves and previews every manual selection but never runs the
correlation itself). With do_dic=True it also runs bisection_search and
saves all the result plots/GIFs/npz.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass

import numpy as np
import imageio
import matplotlib.path as mpath

from dic_workflow.importing import scan_raw_folder, load_raw
from dic_workflow.pipeline import (
    select_crop, convert_frames,
    select_level_points, select_grid_axis, select_scale_points, select_point,
    select_alignment_markers, align_frames,
    select_poi_region, select_exclusion_region,
    export_grid_arrays,
)
from dic_workflow.plotting import (
    save_input_gif, save_poi_plot, save_frame_plots,
    save_strain_map, save_displacement_map,
    save_displacement_grid, save_strain_grid,
    save_overlay_gif,
)


@dataclass
class DatasetConfig:
    """Same knobs as dic_real_images.py's PARAMETERS block, applied to
    whichever dataset run_dataset() is called for."""
    START_INDEX: int = 0
    END_INDEX: int = None   # None = use all images

    SUBSET_LEVEL: int = 40
    STEP: int = 20
    ZNSSD_LIMIT: float = 0.15
    LOST_LEVEL: float = 0.30
    SUBPIXEL_METHOD: str = 'fagn'

    N_ZERO_PRESSURE_IMAGES: int = 1
    PRESSURE_INCREMENT_N: float = None
    PRESSURE_VALUES_N: object = None  # explicit per-frame force [N]; overrides the increment above

    BIT_DEPTH: int = 16

    ALIGN_TEMPLATE_HALF: int = 0
    ALIGN_SEARCH_RADIUS: int = 80

    SCALE_CALIBRATION_METHOD: str = 'aruco'
    ARUCO_CALIBRATION_IMAGE: str = None
    ARUCO_MARKER_ID: int = 0
    ARUCO_HINT_RADIUS: int = 250
    ARUCO_HINT_UPSCALE: float = 3.0
    ARUCO_BRIGHTNESS_STEPS: tuple = (1.5, 2.0, 3.0)

    RESIZE_TO: object = None
    FLATTEN_SIGMA: float = 0
    REFLECTION_CLIP_PCT: float = 100

    SAVE_OVERLAY_GIF: bool = True

    REFINE_ROI_WITH_EDGES: bool = True
    EDGE_BAND_WIDTH: int = 30
    EDGE_BLUR_SIGMA: float = 2

    ROI_SHRINK_PX: int = 15

    LENS_PARAMS: dict = None


class PrepIncomplete(RuntimeError):
    """Raised by run_dataset(interactive=False) when a required setup JSON
    is missing -- run the dataset with interactive=True first."""


def _require_interactive(interactive, json_path, what):
    if not interactive:
        raise PrepIncomplete(
            f"Missing {json_path} ({what}) — run this dataset with "
            f"interactive=True first (run_dic/dic_real_images.py)."
        )


def _resolve_scale(result_folder, cfg, get_ref_frame, interactive):
    """Resolve px -> mm scale, caching to result_folder/scale.json."""
    scale_json = os.path.join(result_folder, 'scale.json')

    if os.path.isfile(scale_json):
        with open(scale_json) as f:
            mm_per_px = json.load(f)['mm_per_px']
        if mm_per_px is not None:
            print(f"Using saved scale: {1/mm_per_px:.3f} px/mm  "
                  "— delete scale.json to re-calibrate.")
        else:
            print("Scale calibration disabled (scale.json) — displacement stays in px.")
        return mm_per_px

    _require_interactive(interactive, scale_json, 'px -> mm scale calibration')

    mm_per_px = None
    px_dist   = None

    if cfg.SCALE_CALIBRATION_METHOD == 'aruco':
        from dic_workflow.aruco_calibration import compute_mm_per_pixel
        aruco_source = cfg.ARUCO_CALIBRATION_IMAGE or get_ref_frame()
        try:
            mm_per_px = compute_mm_per_pixel(aruco_source, marker_id=cfg.ARUCO_MARKER_ID)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ArUco calibration failed ({e})")
            hint_img = aruco_source if not isinstance(aruco_source, str) else get_ref_frame()
            print("Click the marker's approximate location, then close window "
                  "(detection is retried there, brightening the image a few "
                  "steps if needed; close without clicking = give up and fall "
                  "back to manual click calibration) ...")
            hint = select_point(hint_img, "Click the ArUco marker's approximate location")
            if hint is not None:
                try:
                    mm_per_px = compute_mm_per_pixel(
                        aruco_source, marker_id=cfg.ARUCO_MARKER_ID,
                        hint_xy=hint, hint_radius=cfg.ARUCO_HINT_RADIUS,
                        hint_upscale=cfg.ARUCO_HINT_UPSCALE,
                        brightness_steps=cfg.ARUCO_BRIGHTNESS_STEPS)
                except (FileNotFoundError, RuntimeError) as e2:
                    print(f"ArUco calibration still failed ({e2}) — "
                          "falling back to manual click calibration.")
            else:
                print("No hint given — falling back to manual click calibration.")

    if mm_per_px is None:
        print("Click 2 points a known real-world distance apart, then close window "
              "(close without clicking = skip mm calibration) ...")
        px_dist = select_scale_points(get_ref_frame())
        if px_dist > 0:
            mm_dist   = float(input("Real-world distance between those 2 points, in mm: "))
            mm_per_px = mm_dist / px_dist
            print(f"Scale calibrated: {px_dist:.1f} px = {mm_dist:g} mm  "
                  f"({1/mm_per_px:.3f} px/mm)")
        else:
            print("Scale calibration skipped — displacement stays in px.")

    with open(scale_json, 'w') as f:
        json.dump({'pixel_distance': px_dist, 'mm_per_px': mm_per_px}, f, indent=2)
    return mm_per_px


def run_dataset(input_folder, result_folder, cfg, dic_module,
                *, interactive, do_dic, return_data=False):
    """Run the pipeline for one dataset.

    input_folder  : folder of .ARW (or already-converted .tif) frames.
    result_folder : output folder for this dataset (created if missing).
    cfg           : DatasetConfig.
    dic_module    : the `dic` package, already imported with torch_device
                    (and barron filters, if do_dic=True) configured by the
                    caller.
    interactive   : True opens click UIs for any missing setup JSON and
                    saves the result; False raises PrepIncomplete instead.
    do_dic        : True also runs bisection_search and saves all result
                    plots/GIFs/npz; False stops after frame_00_POIs.png.
    return_data   : True returns a dict with the in-memory pipeline state
                    (reader, POIs, pressures, mm_per_px, output_frame_list,
                    and -- if do_dic -- result_graph / POI_graph_list) for
                    callers that need direct access, e.g. a noise-floor
                    analysis over the raw displacements. False (default)
                    returns None.
    """
    os.makedirs(result_folder, exist_ok=True)

    # ============================================================
    #  STEP 1 — COLLECT INPUT FILES
    # ============================================================

    input_tifs = sorted(f for f in os.listdir(input_folder) if f.lower().endswith('.tif'))

    if input_tifs:
        # ── TIF input: copy to converted/ so measure scripts can find them ──
        converted_folder = os.path.join(result_folder, 'converted')
        os.makedirs(converted_folder, exist_ok=True)
        raw_files      = input_tifs
        n_images       = len(raw_files)
        align_markers  = np.empty((0, 2))
        print(f"Found {n_images} TIF images in {input_folder!r} — skipping preprocessing.")
        for fname in raw_files:
            dst = os.path.join(converted_folder, fname)
            if not os.path.isfile(dst):
                shutil.copy2(os.path.join(input_folder, fname), dst)

        mm_per_px = _resolve_scale(
            result_folder, cfg,
            lambda: imageio.imread(os.path.join(converted_folder, raw_files[0])),
            interactive)

    else:
        # ── RAW input: crop, convert, align ─────────────────────────────
        converted_folder = os.path.join(result_folder, 'converted')
        crop_json_path   = os.path.join(result_folder, 'crop.json')
        os.makedirs(converted_folder, exist_ok=True)

        raw_files = scan_raw_folder(input_folder, start=cfg.START_INDEX, end=cfg.END_INDEX)
        if not raw_files:
            raise PrepIncomplete(f"No RAW or TIF files found in {input_folder!r}")

        n_images = len(raw_files)
        print(f"Found {n_images} RAW images  "
              f"(index {cfg.START_INDEX}..{cfg.START_INDEX + n_images - 1})")

        def _load_raw_frame(path):
            return load_raw(path, resize_to=cfg.RESIZE_TO)

        # STEP 1.5 — SCALE CALIBRATION (before cropping, so a marker outside
        # the crop region can still be detected)
        mm_per_px = _resolve_scale(
            result_folder, cfg,
            lambda: _load_raw_frame(os.path.join(input_folder, raw_files[0])),
            interactive)

        # STEP 2/3 — LEVEL + CROP SELECTION, CONVERSION
        # (skipped entirely if converted TIFFs + crop.json already exist)
        tifs_exist = all(
            os.path.isfile(os.path.join(converted_folder, f'frame_{i:03d}.tif'))
            for i in range(n_images)
        )

        if tifs_exist and os.path.isfile(crop_json_path):
            with open(crop_json_path) as f:
                crop_box = tuple(json.load(f)['crop_box'])
            x1, y1, x2, y2 = crop_box
            print(f"Using saved crop  x=[{x1}:{x2}]  y=[{y1}:{y2}]  "
                  f"({x2-x1}x{y2-y1} px)  — delete crop.json to re-select.")
        else:
            _require_interactive(interactive, crop_json_path,
                                 'level + crop selection / RAW conversion')

            print("Reading reference frame for crop selection ...")
            gray_ref = _load_raw_frame(os.path.join(input_folder, raw_files[0]))
            if cfg.LENS_PARAMS is not None:
                import cv2
                from dic_workflow.processing import build_lens_corrector, apply_vignetting
                ref_coords, ref_mod = build_lens_corrector(gray_ref.shape, **cfg.LENS_PARAMS)
                gray_ref = apply_vignetting(ref_mod, gray_ref)
                gray_ref = cv2.remap(gray_ref, ref_coords, None, cv2.INTER_LANCZOS4)
                print("Lens correction applied to reference frame for crop selection.")

            # STEP 1.9 — LEVEL SELECTION (rotate to horizontal, optional)
            level_json_path = os.path.join(result_folder, 'level.json')
            if os.path.isfile(level_json_path):
                with open(level_json_path) as f:
                    level_angle = json.load(f)['angle_deg']
                print(f"Using saved level angle: {level_angle:+.3f}°  "
                      "— delete level.json to re-select.")
            else:
                print("Click 2 points that should be level, then close window "
                      "(close without clicking = skip leveling) ...")
                level_angle = select_level_points(gray_ref)
                with open(level_json_path, 'w') as f:
                    json.dump({'angle_deg': level_angle}, f, indent=2)
                if level_angle != 0.0:
                    print(f"Level angle selected: {level_angle:+.3f}°")
                else:
                    print("Leveling skipped.")

            if level_angle != 0.0:
                from dic_workflow.processing import rotate_to_level
                gray_ref = rotate_to_level(gray_ref, level_angle)

            # STEP 2 — CROP SELECTION
            x1, y1, x2, y2 = select_crop(gray_ref)
            crop_box = (x1, y1, x2, y2)
            print(f"Crop selected:  x=[{x1}:{x2}]  y=[{y1}:{y2}]  ({x2-x1}x{y2-y1} px)")
            with open(crop_json_path, 'w') as f:
                json.dump({'crop_box': list(crop_box)}, f, indent=2)

            # STEP 3 — CONVERT ALL FRAMES
            convert_frames(raw_files, input_folder, converted_folder, crop_box,
                           cfg.FLATTEN_SIGMA, cfg.REFLECTION_CLIP_PCT, _load_raw_frame,
                           lens_params=cfg.LENS_PARAMS, bit_depth=cfg.BIT_DEPTH,
                           level_angle=level_angle)
            print(f"\nConversion done -> {converted_folder}")

        # STEP 3.5 — IMAGE ALIGNMENT
        align_json = os.path.join(result_folder, 'align_markers.json')
        ref_gray   = imageio.imread(os.path.join(converted_folder, 'frame_000.tif'))

        if cfg.ALIGN_TEMPLATE_HALF <= 0:
            align_markers = np.empty((0, 2))
            print("Alignment skipped (ALIGN_TEMPLATE_HALF = 0).")
        elif os.path.isfile(align_json):
            with open(align_json) as f:
                align_markers = np.array(json.load(f)['markers'])
            print(f"Loaded {len(align_markers)} alignment markers — "
                  "delete align_markers.json to re-select.")
        else:
            _require_interactive(interactive, align_json, 'alignment marker selection')
            align_markers = select_alignment_markers(ref_gray)
            if len(align_markers) >= 2:
                with open(align_json, 'w') as f:
                    json.dump({'markers': align_markers.tolist()}, f, indent=2)
                print(f"Saved {len(align_markers)} alignment markers.")
            else:
                print(f"Alignment skipped ({len(align_markers)} marker(s) — need >= 2).")

        if len(align_markers) >= 2:
            align_frames(converted_folder, n_images, align_markers,
                        cfg.ALIGN_TEMPLATE_HALF, cfg.ALIGN_SEARCH_RADIUS)

    # ============================================================
    #  STEP 3.9 — PER-FRAME PRESSURE LABELS (optional)
    # ============================================================

    if cfg.PRESSURE_VALUES_N is not None:
        if len(cfg.PRESSURE_VALUES_N) != n_images:
            raise ValueError(
                f"PRESSURE_VALUES_N has {len(cfg.PRESSURE_VALUES_N)} entries "
                f"but there are {n_images} frames.")
        pressures = list(cfg.PRESSURE_VALUES_N)
    else:
        def _pressure_for_frame(i):
            if cfg.PRESSURE_INCREMENT_N is None:
                return None
            if i < cfg.N_ZERO_PRESSURE_IMAGES:
                return 0
            return (i - cfg.N_ZERO_PRESSURE_IMAGES + 1) * cfg.PRESSURE_INCREMENT_N

        pressures = [_pressure_for_frame(i) for i in range(n_images)]

    # ============================================================
    #  STEP 4 — LOAD IMAGES FOR DIC
    # ============================================================

    reader            = dic_module.image_folder_loader(converted_folder, suffix='tif')
    start_index       = 0
    end_index         = n_images - 1
    output_frame_list = np.arange(1, end_index + 1, dtype=np.int64)

    # ============================================================
    #  STEP 5 — INPUT FRAMES GIF
    # ============================================================

    save_input_gif(reader, raw_files, n_images,
                   os.path.join(result_folder, 'input_frames.gif'),
                   pressures=pressures)

    # ============================================================
    #  STEP 6 — POI SELECTION (polygon + edge refine + shrink, cached)
    # ============================================================

    poly_json = os.path.join(result_folder, 'poi_polygon.json')

    if os.path.isfile(poly_json):
        with open(poly_json) as f:
            vertices = json.load(f)['vertices']
        print(f"Using saved ROI polygon from {poly_json} ({len(vertices)} vertices) "
              "— delete it to re-draw.")
    else:
        _require_interactive(interactive, poly_json,
                             'ROI polygon selection (+ edge refinement / shrink)')

        vertices = select_poi_region(reader, start_index)
        if len(vertices) < 3:
            raise PrepIncomplete('Need at least 3 vertices to define an ROI.')

        print('Vertices:')
        for v in vertices:
            print(f'  [{v[0]:.1f}, {v[1]:.1f}],')

        # STEP 6.1 — EDGE DETECTION ROI REFINEMENT (optional)
        if cfg.REFINE_ROI_WITH_EDGES:
            from dic_workflow.edge_detection import refine_roi_to_edges
            ref_img  = np.asarray(reader[start_index])
            edge_png = os.path.join(result_folder, 'edge_refinement.png')
            vertices, edge_accepted = refine_roi_to_edges(
                ref_img, vertices,
                band_width=cfg.EDGE_BAND_WIDTH,
                blur_sigma=cfg.EDGE_BLUR_SIGMA,
                result_path=edge_png,
            )
            if edge_accepted:
                print(f'ROI refined: {len(vertices)} vertices.')
                print('Refined vertices:')
                for v in vertices:
                    print(f'  [{v[0]:.1f}, {v[1]:.1f}],')
            else:
                print('Edge refinement skipped — using original polygon.')

        # STEP 6.2 — SHRINK ROI INWARD (keep subsets off the physical edge)
        if cfg.ROI_SHRINK_PX > 0:
            from dic_workflow.edge_detection import shrink_polygon
            vertices = shrink_polygon(vertices, cfg.ROI_SHRINK_PX)
            print(f'ROI shrunk inward by {cfg.ROI_SHRINK_PX}px on all sides:')
            for v in vertices:
                print(f'  [{v[0]:.1f}, {v[1]:.1f}],')

        with open(poly_json, 'w') as f:
            json.dump({'vertices': vertices}, f, indent=2)

    # ============================================================
    #  STEP 6.5 — EXCLUSION REGION (slit / hole, optional)
    # ============================================================

    excl_json = os.path.join(result_folder, 'exclusion.json')

    if os.path.isfile(excl_json):
        with open(excl_json) as f:
            excl_data = json.load(f)
        exclusion_vertices = excl_data.get('vertices', [])
        if exclusion_vertices:
            print(f'Loaded exclusion polygon ({len(exclusion_vertices)} vertices) '
                  '— delete exclusion.json to re-draw.')
        else:
            print('exclusion.json present but empty — no exclusion applied.')
    else:
        _require_interactive(interactive, excl_json, 'exclusion region selection')
        exclusion_vertices = select_exclusion_region(reader, start_index,
                                                     exterior_vertices=vertices)
        with open(excl_json, 'w') as f:
            json.dump({'vertices': exclusion_vertices}, f, indent=2)
        if exclusion_vertices:
            print(f'Exclusion polygon saved ({len(exclusion_vertices)} vertices).')
        else:
            print('No exclusion region drawn — skipped.')

    # ============================================================
    #  STEP 6.6 — GRID AXIS SELECTION (local "horizontal" for the POI grid)
    # ============================================================
    # The level angle above (if any) squares up the whole photographed
    # setup. This is separate: it only reorients the POI lattice, for a
    # sample whose tracked segment sits at its own angle relative to the
    # leveled image (e.g. past a bend point in a bending test) — no pixel
    # data is resampled.

    axis_json = os.path.join(result_folder, 'grid_axis.json')

    if os.path.isfile(axis_json):
        with open(axis_json) as f:
            grid_axis_angle = json.load(f)['angle_deg']
        print(f"Using saved grid axis angle from {axis_json}: {grid_axis_angle:+.3f}°  "
              "— delete grid_axis.json to re-select.")
    else:
        _require_interactive(interactive, axis_json, 'grid axis selection')
        print("Click 2 points along the sample's own axis (its local 'horizontal' "
              "for the DIC grid), then close window "
              "(close without clicking = grid stays image-aligned) ...")
        grid_axis_angle = select_grid_axis(np.asarray(reader[start_index]))
        with open(axis_json, 'w') as f:
            json.dump({'angle_deg': grid_axis_angle}, f, indent=2)
        if grid_axis_angle != 0.0:
            print(f"Grid axis angle selected: {grid_axis_angle:+.3f}°")
        else:
            print("Grid axis selection skipped — grid stays image-aligned.")

    # Build the POI lattice in a coordinate frame rotated by grid_axis_angle,
    # so grid rows/columns run parallel/perpendicular to the sample's own
    # axis instead of the image's raw x/y — then map back to image (pixel)
    # coords. Rotation is an isometry so STEP still gives exact px spacing.
    # Points don't need to land on any particular pixel grid: the mesh
    # (build_grid_edges / build_triangles) is a Delaunay triangulation of
    # the actual POI positions, not an index lookup.
    theta = np.radians(grid_axis_angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def _to_axis_frame(xy):
        x, y = xy[..., 0], xy[..., 1]
        u = x * cos_t + y * sin_t
        v = -x * sin_t + y * cos_t
        return np.stack([u, v], axis=-1)

    def _from_axis_frame(uv):
        u, v = uv[..., 0], uv[..., 1]
        x = u * cos_t - v * sin_t
        y = u * sin_t + v * cos_t
        return np.stack([x, y], axis=-1)

    verts_uv = _to_axis_frame(np.asarray(vertices, dtype=float))
    u1, v1 = verts_uv.min(axis=0)
    u2, v2 = verts_uv.max(axis=0)
    uu, vv = np.meshgrid(np.arange(u1, u2 + cfg.STEP, cfg.STEP),
                         np.arange(v1, v2 + cfg.STEP, cfg.STEP))
    POIs = _from_axis_frame(np.column_stack([uu.ravel(), vv.ravel()])).astype(dic_module.floattype)

    # Filter to only points inside the selected polygon (not just its bounding box).
    roi_verts = np.asarray(vertices, dtype=float)
    if not np.array_equal(roi_verts[0], roi_verts[-1]):
        roi_verts = np.vstack([roi_verts, roi_verts[0]])
    POIs = POIs[mpath.Path(roi_verts).contains_points(POIs)]

    if len(exclusion_vertices) >= 3:
        ev = np.asarray(exclusion_vertices, dtype=float)
        if not np.array_equal(ev[0], ev[-1]):
            ev = np.vstack([ev, ev[0]])
        POIs = POIs[~mpath.Path(ev).contains_points(POIs)]
    print(f'POIs: {len(POIs)} points,  step={cfg.STEP} px,  '
          f'subset={2*cfg.SUBSET_LEVEL+1}x{2*cfg.SUBSET_LEVEL+1}')

    save_poi_plot(reader[start_index], POIs,
                  os.path.join(result_folder, 'frame_00_POIs.png'), dic_module,
                  pressure=pressures[start_index])

    state = dict(
        reader=reader, POIs=POIs, vertices=vertices,
        exclusion_vertices=exclusion_vertices, pressures=pressures,
        mm_per_px=mm_per_px, converted_folder=converted_folder,
        n_images=n_images, raw_files=raw_files,
        start_index=start_index, end_index=end_index,
        output_frame_list=output_frame_list,
        result_graph=None, POI_graph_list=None,
    )

    if not do_dic:
        print(f'Prep done (DIC not run) -> {result_folder}')
        return state if return_data else None

    # ============================================================
    #  STEP 7 — RUN DIC
    # ============================================================

    t1 = time.time()
    result_graph, POI_graph_list = dic_module.bisection_search(
        reader, start_index, end_index, POIs, output_frame_list,
        subset_level=cfg.SUBSET_LEVEL, ZNSSD_limit=cfg.ZNSSD_LIMIT,
        subpixel_method=cfg.SUBPIXEL_METHOD, lost_level=cfg.LOST_LEVEL,
    )
    print(f'DIC calculation time: {time.time() - t1:.1f}s')

    # ============================================================
    #  STEP 8 — SAVE FRAME PLOTS + STRAIN/DISPLACEMENT MAPS
    # ============================================================

    save_frame_plots(reader, result_graph, output_frame_list,
                     raw_files, result_folder, dic_module, pressures=pressures)
    save_strain_map(reader, result_graph, output_frame_list, result_folder,
                    step=cfg.STEP, pressures=pressures)
    save_displacement_map(reader, result_graph, output_frame_list, result_folder,
                          POIs=POIs, step=cfg.STEP, pressures=pressures,
                          mm_per_px=mm_per_px)

    # ============================================================
    #  STEP 9 — DISPLACEMENT & STRAIN GRID VISUALIZATIONS
    # ============================================================

    save_displacement_grid(POIs, result_graph, end_index, cfg.STEP, result_folder,
                           exclusion_verts=exclusion_vertices, pressures=pressures)
    save_strain_grid(POIs, result_graph, end_index, cfg.STEP, result_folder,
                     exclusion_verts=exclusion_vertices, pressures=pressures)

    grid_data = export_grid_arrays(POIs, result_graph, end_index)
    npz_path = os.path.join(result_folder, 'deformation_grids.npz')
    np.savez(npz_path, **grid_data)
    print(f'Saved: {npz_path}')

    if cfg.SAVE_OVERLAY_GIF:
        save_overlay_gif(reader, POIs, result_graph, end_index, cfg.STEP,
                         result_folder, n_images=n_images,
                         exclusion_verts=exclusion_vertices, pressures=pressures)

    print(f'\nAll done.  Results -> {result_folder}')

    if return_data:
        state['result_graph']    = result_graph
        state['POI_graph_list']  = POI_graph_list
        return state
