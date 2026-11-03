"""
DIC workflow helpers: crop selection, frame conversion, image alignment,
interactive POI region selection, and grid geometry utilities.

Interactive functions (select_crop, select_alignment_markers,
select_poi_region) open matplotlib windows and block until the window
is closed.  Call matplotlib.use(...) in the main script before importing
this module.
"""

import os
import numpy as np


# ── Crop selection ────────────────────────────────────────────────────────

def select_crop(gray_ref):
    """Show reference frame and let user draw a crop rectangle.

    Returns (x1, y1, x2, y2) clamped to image bounds.
    Closing the window without drawing returns the full image extent.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    h_full, w_full = gray_ref.shape
    crop_box = [0, 0, w_full, h_full]

    def _on_select(eclick, erelease):
        crop_box[:] = [
            int(round(min(eclick.xdata,   erelease.xdata))),
            int(round(min(eclick.ydata,   erelease.ydata))),
            int(round(max(eclick.xdata,   erelease.xdata))),
            int(round(max(eclick.ydata,   erelease.ydata))),
        ]

    fig, ax = plt.subplots()
    ax.imshow(gray_ref, cmap='gray', vmin=0, vmax=np.percentile(gray_ref, 99))
    ax.set_title('Draw crop rectangle, then close window\n'
                 '(close without drawing = use full image)')
    # Keep a reference: an unreferenced selector is garbage-collected and stops responding.
    _rs = RectangleSelector(ax, _on_select, useblit=True, button=[1],
                            interactive=True,
                            props=dict(edgecolor='red', linewidth=2, fill=False))
    plt.tight_layout()
    plt.show()

    x1, y1, x2, y2 = crop_box
    x1 = max(0, min(x1, w_full - 1))
    y1 = max(0, min(y1, h_full - 1))
    x2 = max(x1 + 1, min(x2, w_full))
    y2 = max(y1 + 1, min(y2, h_full))
    return x1, y1, x2, y2


# ── Shared 2-point click widget ──────────────────────────────────────────

def _select_two_points(gray_ref, title, color='c'):
    """Click up to 2 points on an image; used by select_level_points and
    select_scale_points.

    Returns [(x1, y1), (x2, y2)], or None if fewer than 2 points were
    clicked before the window was closed.
    """
    import matplotlib.pyplot as plt

    pts, point_hdls = [], []
    line_hdl = [None]

    fig, ax = plt.subplots()
    ax.imshow(gray_ref, cmap='gray', vmin=0, vmax=np.percentile(gray_ref, 99))
    ax.set_title(title)

    def _redraw_line():
        if line_hdl[0] is not None:
            line_hdl[0].remove()
            line_hdl[0] = None
        if len(pts) == 2:
            xs = [pts[0][0], pts[1][0]]
            ys = [pts[0][1], pts[1][1]]
            line_hdl[0], = ax.plot(xs, ys, color + '--', lw=1.2)

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1 and len(pts) < 2:
            x, y = event.xdata, event.ydata
            pts.append((x, y))
            h_, = ax.plot(x, y, color + '+', ms=14, mew=2)
            point_hdls.append(h_)
            _redraw_line()
            fig.canvas.draw()
        elif event.button == 3 and pts:
            pts.pop()
            point_hdls.pop().remove()
            _redraw_line()
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.tight_layout()
    plt.show()

    return pts if len(pts) == 2 else None


# ── Level (rotate-to-horizontal) point selection ────────────────────────────

def select_level_points(gray_ref):
    """Click 2 points that should be level (e.g. a horizontal edge or two
    markers at the same height).

    Returns the rotation angle in degrees that levels the image (pass to
    processing.rotate_to_level), or 0.0 if fewer than 2 points are clicked
    before the window is closed (leveling skipped).
    """
    pts = _select_two_points(
        gray_ref,
        'Click 2 points that should be level (horizontal)\n'
        'Right-click: undo  |  Close: confirm  '
        '(close without clicking to skip leveling)',
        color='c')
    if pts is None:
        return 0.0
    (x1, y1), (x2, y2) = pts
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


# ── DIC grid axis (local "horizontal" for the POI lattice) ──────────────────

def select_grid_axis(gray_ref):
    """Click 2 points along the sample's own axis, to use as the local
    "horizontal" for laying out the DIC POI grid.

    Distinct from select_level_points: that rotates the whole image (e.g.
    to square up the camera/setup), while this only reorients the POI
    lattice -- useful when the tracked segment of the sample sits at an
    angle relative to the leveled image (e.g. past a bend point in a
    bending test), without resampling any pixel data.

    Returns the angle in degrees, or 0.0 if fewer than 2 points are clicked
    before the window is closed (grid stays aligned to the image axes).
    """
    pts = _select_two_points(
        gray_ref,
        "Click 2 points along the sample's axis "
        "(this becomes the DIC grid's local horizontal)\n"
        'Right-click: undo  |  Close: confirm  '
        '(close without clicking to keep the grid image-aligned)',
        color='g')
    if pts is None:
        return 0.0
    (x1, y1), (x2, y2) = pts
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


# ── Scale (px -> mm) point selection ─────────────────────────────────────

def select_scale_points(gray_ref):
    """Click 2 points a known real-world distance apart (e.g. a ruler or a
    fixture of known size), for pixel -> mm calibration.

    Returns the pixel distance between the two points, or 0.0 if fewer than
    2 points are clicked before the window is closed (calibration skipped).
    """
    pts = _select_two_points(
        gray_ref,
        'Click 2 points a KNOWN real-world distance apart (e.g. a ruler)\n'
        'Right-click: undo  |  Close: confirm  '
        '(close without clicking to skip mm calibration)',
        color='y')
    if pts is None:
        return 0.0
    (x1, y1), (x2, y2) = pts
    return float(np.hypot(x2 - x1, y2 - y1))


# ── Single-point click (e.g. ArUco marker location hint) ─────────────────

def select_point(gray_ref, title, color='m'):
    """Click a single point on an image (right-click to move it).

    Returns (x, y), or None if the window is closed without clicking.
    """
    import matplotlib.pyplot as plt

    pt  = [None]
    hdl = [None]

    fig, ax = plt.subplots()
    ax.imshow(gray_ref, cmap='gray', vmin=0, vmax=np.percentile(gray_ref, 99))
    ax.set_title(title)

    def _on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        pt[0] = (event.xdata, event.ydata)
        if hdl[0] is not None:
            hdl[0].remove()
        hdl[0], = ax.plot(event.xdata, event.ydata, color + '+', ms=18, mew=2.5)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.tight_layout()
    plt.show()

    return pt[0]


# ── Frame conversion ──────────────────────────────────────────────────────

def convert_frames(raw_files, input_folder, converted_folder, crop_box,
                   flatten_sigma, reflection_clip_pct, load_fn, lens_params=None,
                   bit_depth=8, level_angle=0.0):
    """Convert RAW files to leveled, cropped, pre-processed TIFFs.

    load_fn   : callable(path) -> float32 grayscale (from importing.load_raw).
    lens_params: dict with keys (cam_maker, cam_model, lens_maker, lens_model,
                 focal_length, aperture, distance) to enable geometry distortion
                 and vignetting correction via lensfunpy + cv2.remap.
                 Pass None (default) to skip lens correction entirely.
    bit_depth : 8 or 16. 16-bit preserves more of the sensor's dynamic range and
                improves subpixel accuracy by giving the B-spline interpolator a
                finer intensity grid to work with.
    level_angle: rotation in degrees applied about the image center before
                 cropping (from select_level_points). 0.0 = no rotation.

    Returns (ref_min, ref_max) taken from the reference frame (index 0).
    """
    import imageio
    from dic_workflow.processing import (flatten_illumination, clip_reflections,
                                         normalize_8bit, normalize_16bit,
                                         rotate_to_level)

    normalize = normalize_16bit if bit_depth == 16 else normalize_8bit

    x1, y1, x2, y2 = crop_box
    ref_min = ref_max = None
    _undist_coords = None   # built lazily from first frame's dimensions
    _lens_mod = None

    for i, fname in enumerate(raw_files):
        out_path = os.path.join(converted_folder, f'frame_{i:03d}.tif')
        print(f'  [{i:03d}] converting {fname} ...', end=' ', flush=True)

        gray = load_fn(os.path.join(input_folder, fname))

        if lens_params is not None:
            import cv2
            from dic_workflow.processing import build_lens_corrector, apply_vignetting
            if _undist_coords is None:
                _undist_coords, _lens_mod = build_lens_corrector(gray.shape, **lens_params)
                print(f'lens correction map built ({gray.shape[1]}x{gray.shape[0]}) ...', end=' ', flush=True)
            gray = apply_vignetting(_lens_mod, gray)
            gray = cv2.remap(gray, _undist_coords, None, cv2.INTER_LANCZOS4)

        gray = rotate_to_level(gray, level_angle)

        gray = gray[y1:y2, x1:x2]

        if flatten_sigma > 0:
            gray = flatten_illumination(gray, flatten_sigma)
        if reflection_clip_pct < 100:
            gray = clip_reflections(gray, reflection_clip_pct)

        if i == 0:
            ref_min = float(gray.min())
            ref_max = float(gray.max())

        imageio.imwrite(out_path, normalize(gray, ref_min, ref_max))
        print(f'{gray.shape[1]}x{int(y2-y1)} px saved.')

    return ref_min, ref_max


# ── Alignment marker selection ────────────────────────────────────────────

def select_alignment_markers(ref_gray):
    """Click ≥2 fixed markers on the reference frame.

    Returns an (N, 2) float array of (x, y) marker positions,
    or an empty (0, 2) array when fewer than 2 points are clicked.
    """
    import matplotlib.pyplot as plt

    pts, hdls = [], []
    fig, ax   = plt.subplots()
    ax.imshow(ref_gray, cmap='gray')
    ax.set_title(
        'Click ≥2 FIXED markers (rigid fixtures, outside the deforming area)\n'
        'Right-click: undo  |  Close: confirm  '
        '(close without clicking to skip alignment)')

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            pts.append([x, y])
            h_, = ax.plot(x, y, 'g+', ms=14, mew=2)
            ax.annotate(str(len(pts)), (x, y),
                        color='lime', fontsize=9, ha='center', va='bottom')
            hdls.append(h_)
            fig.canvas.draw()
        elif event.button == 3 and pts:
            pts.pop()
            hdls.pop().remove()
            for ann in list(ax.texts):
                ann.remove()
            for n, v in enumerate(pts):
                ax.annotate(str(n + 1), (v[0], v[1]),
                            color='lime', fontsize=9, ha='center', va='bottom')
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.show()

    if len(pts) >= 2:
        return np.array(pts)
    return np.empty((0, 2))


# ── Frame alignment ───────────────────────────────────────────────────────

def align_frames(converted_folder, n_images, align_markers,
                 template_half, search_radius):
    """Template-match each frame to the reference and apply a rigid correction.

    Modifies the TIFF files in converted_folder in-place.
    """
    import imageio
    from skimage.feature import match_template
    from scipy.ndimage import affine_transform

    ref_gray  = imageio.imread(os.path.join(converted_folder, 'frame_000.tif'))
    _bit16    = ref_gray.dtype == np.uint16
    _clip_max = 65535 if _bit16 else 255
    _save_dt  = np.uint16 if _bit16 else np.uint8
    th        = template_half
    sr        = search_radius
    Pswap    = np.array([[0., 1.], [1., 0.]])

    tmpls = []
    for mx, my in align_markers:
        cx_, cy_ = int(round(mx)), int(round(my))
        patch = ref_gray[cy_ - th: cy_ + th + 1,
                         cx_ - th: cx_ + th + 1].astype(np.float32)
        tmpls.append((cx_, cy_, patch))

    print(f"Aligning {n_images - 1} frame(s) to reference ...")
    for fi in range(1, n_images):
        tif_path = os.path.join(converted_folder, f'frame_{fi:03d}.tif')
        img_f    = imageio.imread(tif_path).astype(np.float32)

        ref_xy, found_xy = [], []
        for cx_, cy_, tmpl in tmpls:
            wy0 = max(0, cy_ - sr - th)
            wy1 = min(img_f.shape[0], cy_ + sr + th + 1)
            wx0 = max(0, cx_ - sr - th)
            wx1 = min(img_f.shape[1], cx_ + sr + th + 1)
            win = img_f[wy0:wy1, wx0:wx1]
            if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
                continue
            res  = match_template(win, tmpl, pad_input=False)
            r, c = np.unravel_index(res.argmax(), res.shape)
            ref_xy.append([cx_, cy_])
            found_xy.append([wx0 + c + th, wy0 + r + th])

        if len(ref_xy) < 2:
            print(f"  frame {fi:03d}: <2 markers located — skipped")
            continue

        src = np.array(found_xy, dtype=np.float64)
        dst = np.array(ref_xy,   dtype=np.float64)
        sc, dc = src.mean(0), dst.mean(0)
        U, _, Vt = np.linalg.svd((src - sc).T @ (dst - dc))
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = dc - R @ sc

        M = Pswap @ R.T @ Pswap
        b = -(Pswap @ (R.T @ t))
        aligned = affine_transform(img_f, M, offset=b, order=3, mode='nearest')
        imageio.imwrite(tif_path, np.clip(aligned, 0, _clip_max).astype(_save_dt))

        ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"  frame {fi:03d}: tx={t[0]:+.1f}px  ty={t[1]:+.1f}px  "
              f"θ={ang:+.4f}°  ({len(ref_xy)}/{len(tmpls)} markers matched)")

    print("Alignment done.\n")


# ── Interactive POI region selection ─────────────────────────────────────

def select_poi_region(reader, start_index):
    """Click polygon vertices to define the ROI on the reference frame.

    Returns a list of [x, y] vertices, or [] when closed without drawing.
    Left-click: add vertex  |  Right-click: undo last  |  Close: confirm.
    """
    import matplotlib.pyplot as plt

    vertices, markers, annotations = [], [], []
    poly_line = [None]

    fig, ax = plt.subplots()
    ax.imshow(np.asarray(reader[start_index]), cmap='gray')
    ax.set_title(
        'Draw polygon ROI\n'
        'Left-click: add vertex  |  Right-click: undo  |  Close: confirm\n'
        '(close without drawing to abort)'
    )

    def _redraw_poly():
        if poly_line[0] is not None:
            poly_line[0].remove()
            poly_line[0] = None
        if len(vertices) >= 2:
            pts = vertices + [vertices[0]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            poly_line[0], = ax.plot(xs, ys, 'r-', lw=1.5, alpha=0.85)

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            vertices.append([x, y])
            pt, = ax.plot(x, y, 'rs', markersize=5)
            ann = ax.annotate(str(len(vertices)), (x, y),
                              color='yellow', fontsize=8, ha='center', va='bottom')
            markers.append(pt)
            annotations.append(ann)
            _redraw_poly()
            fig.canvas.draw()
        elif event.button == 3 and vertices:
            vertices.pop()
            markers.pop().remove()
            annotations.pop().remove()
            _redraw_poly()
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.tight_layout()
    plt.show()
    return vertices


# ── Interactive exclusion region selection ────────────────────────────────

def select_exclusion_region(reader, start_index, exterior_vertices=None):
    """Click polygon vertices to define a region to exclude from the ROI.

    Typically used to mask a slit or hole.  Pass exterior_vertices to show
    the already-selected ROI outline as context.

    Returns a list of [x, y] vertices, or [] when closed without drawing
    (meaning: no exclusion).
    """
    import matplotlib.pyplot as plt

    vertices, markers, annotations = [], [], []
    poly_line = [None]

    fig, ax = plt.subplots()
    ax.imshow(np.asarray(reader[start_index]), cmap='gray')

    if exterior_vertices and len(exterior_vertices) >= 3:
        ext = np.array(exterior_vertices + [exterior_vertices[0]])
        ax.plot(ext[:, 0], ext[:, 1], 'g--', lw=1.2, alpha=0.6)

    ax.set_title(
        'Draw polygon around region to EXCLUDE (slit / hole)\n'
        'Left-click: add vertex  |  Right-click: undo  |  Close: confirm\n'
        '(close without drawing to skip exclusion)'
    )

    def _redraw_poly():
        if poly_line[0] is not None:
            poly_line[0].remove()
            poly_line[0] = None
        if len(vertices) >= 2:
            pts = vertices + [vertices[0]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            poly_line[0], = ax.plot(xs, ys, 'b-', lw=1.5, alpha=0.85)

    def _on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            vertices.append([x, y])
            pt, = ax.plot(x, y, 'bs', markersize=5)
            ann = ax.annotate(str(len(vertices)), (x, y),
                              color='cyan', fontsize=8, ha='center', va='bottom')
            markers.append(pt)
            annotations.append(ann)
            _redraw_poly()
            fig.canvas.draw()
        elif event.button == 3 and vertices:
            vertices.pop()
            markers.pop().remove()
            annotations.pop().remove()
            _redraw_poly()
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', _on_click)
    plt.show()
    return vertices


# ── Grid geometry helpers ─────────────────────────────────────────────────

def _excl_path(exclusion_verts):
    """Build an mpl.path.Path from exclusion vertices, or return None."""
    if exclusion_verts is None or len(exclusion_verts) < 3:
        return None
    import matplotlib.path as mpath
    verts = np.asarray(exclusion_verts, dtype=float)
    if not np.array_equal(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])
    return mpath.Path(verts)


def _triangulate(pts, step_px, exclusion_verts=None, max_edge_factor=1.8):
    """Delaunay-triangulate pts, for a mesh that conforms to the true POI
    region instead of assuming a square/rectangular boundary.

    A plain Delaunay triangulation fills the convex hull, which would bridge
    any concave notch in the ROI (or a hole cut out by exclusion_verts) with
    spurious long triangles. Since POIs sit on a ~step_px lattice, a genuine
    mesh edge is never much longer than step_px*sqrt(2) (a lattice
    diagonal) -- so triangles with any edge longer than
    max_edge_factor * step_px are dropped as hull-bridging artifacts, and
    remaining triangles whose centroid falls inside exclusion_verts are
    dropped too.

    Returns the (M, 3) simplices array (rows of POI indices).
    """
    from scipy.spatial import Delaunay
    simplices = Delaunay(pts).simplices

    max_edge = max_edge_factor * step_px
    d01 = np.linalg.norm(pts[simplices[:, 0]] - pts[simplices[:, 1]], axis=1)
    d12 = np.linalg.norm(pts[simplices[:, 1]] - pts[simplices[:, 2]], axis=1)
    d20 = np.linalg.norm(pts[simplices[:, 2]] - pts[simplices[:, 0]], axis=1)
    simplices = simplices[(d01 <= max_edge) & (d12 <= max_edge) & (d20 <= max_edge)]

    excl = _excl_path(exclusion_verts)
    if excl is not None and len(simplices):
        centroids = pts[simplices].mean(axis=1)
        simplices = simplices[~excl.contains_points(centroids)]

    return simplices


def build_grid_edges(pts, step_px, exclusion_verts=None):
    """Return list of (i, j) index pairs for the triangulated mesh edges.

    Derived from a Delaunay triangulation of pts (see _triangulate) rather
    than an axis-aligned lattice lookup, so the mesh boundary follows the
    true (possibly non-square) POI region instead of stair-stepping.
    """
    simplices = _triangulate(pts, step_px, exclusion_verts=exclusion_verts)
    edge_set  = set()
    for i0, i1, i2 in simplices:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            edge_set.add((a, b) if a < b else (b, a))
    return list(edge_set)


def build_triangles(pts, step_px, exclusion_verts=None):
    """Return list of (i, j, k) triangle index tuples for strain faces.

    Triangles overlapping the exclusion region, or bridging a gap in the
    POI cloud, are omitted (see _triangulate).
    """
    return [tuple(s) for s in _triangulate(pts, step_px, exclusion_verts=exclusion_verts)]


def make_segs(pts, edgelist):
    """Convert edge index pairs to matplotlib LineCollection segment format."""
    return [[[pts[i, 0], pts[i, 1]], [pts[j, 0], pts[j, 1]]] for i, j in edgelist]


def frame_pts(k, result_graph, POIs):
    """Return tracked POI positions at frame k; lost points remain NaN."""
    if k == 0:
        return POIs.copy()
    return result_graph.nodes[k]['POIs'].copy()


def frame_strain(k, result_graph):
    """Return equivalent Green strain array at frame k, or None if unavailable."""
    node = result_graph.nodes[k]
    if 'green_strain' not in node:
        return None
    return node['green_strain'][..., 3]


def export_grid_arrays(POIs, result_graph, end_index):
    """Extract deformation grid data as numpy arrays for downstream calculations.

    Returns a dict with:
      'POIs'          : (N, 2)       reference POI positions [x, y]
      'positions'     : (F+1, N, 2)  tracked positions per frame; index 0 = reference
      'displacements' : (F+1, N, 2)  displacement [u, v] per POI; zero at reference
      'strain'        : (F+1, N)     equivalent Green strain; NaN at reference/missing
    where F = end_index (so the array covers frames 0 .. end_index inclusive).
    """
    n        = len(POIs)
    n_frames = end_index + 1

    positions     = np.full((n_frames, n, 2), np.nan, dtype=np.float32)
    displacements = np.full((n_frames, n, 2), np.nan, dtype=np.float32)
    strain        = np.full((n_frames, n),    np.nan, dtype=np.float32)

    positions[0]     = POIs.astype(np.float32)
    displacements[0] = 0.0

    for k in range(1, n_frames):
        cur                = frame_pts(k, result_graph, POIs).astype(np.float32)
        positions[k]       = cur
        displacements[k]   = cur - POIs.astype(np.float32)
        s                  = frame_strain(k, result_graph)
        if s is not None:
            strain[k] = s.astype(np.float32)

    return {
        'POIs':          POIs.astype(np.float32),
        'positions':     positions,
        'displacements': displacements,
        'strain':        strain,
    }
