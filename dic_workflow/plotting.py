"""
DIC result visualization: frame overlays, displacement grids,
strain maps, and animated GIFs.

The dic argument of save_frame_plots / save_poi_plot is the dic module from
the dic/ package, passed in by the caller.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PolyCollection

from dic_workflow.pipeline import (
    build_grid_edges, build_triangles, make_segs, frame_pts, frame_strain,
)


# ── Pressure labelling helper ─────────────────────────────────────────────

def _pressure_suffix(pressures, i):
    """Return '  [P N]' for frame i, or '' if pressure tracking is disabled."""
    if pressures is None:
        return ''
    p = pressures[i]
    if p is None:
        return ''
    return f'  [{p:.0f} N]'


# ── POI plot ──────────────────────────────────────────────────────────────

def save_poi_plot(image, POIs, output_path, dic, pressure=None):
    plt.figure()
    dic.plot_img_points(image, POIs, size=1.2)
    if pressure is not None:
        plt.title(f'{pressure:.0f} N')
    plt.savefig(output_path, pad_inches=0, bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Saved: {output_path}')


# ── Input frames GIF ──────────────────────────────────────────────────────

def save_input_gif(reader, raw_files, n_images, output_path, pressures=None):
    img0      = np.asarray(reader[0])
    h_i, w_i  = img0.shape[:2]
    asp        = w_i / max(h_i, 1)
    fw         = max(5, min(10, 6 * asp))
    fh         = max(3, fw / max(asp, 0.5))

    _vmax = 65535 if img0.dtype == np.uint16 else 255
    fig, ax   = plt.subplots(figsize=(fw, fh), facecolor='black')
    ax.axis('off')
    im_obj    = ax.imshow(img0, cmap='gray', vmin=0, vmax=_vmax)
    ttl       = ax.set_title(f'frame 0  ({raw_files[0]}){_pressure_suffix(pressures, 0)}',
                             color='white', fontsize=9)
    fig.tight_layout(pad=0.2)

    def _upd(i):
        im_obj.set_data(np.asarray(reader[i]))
        ttl.set_text(f'frame {i}  ({raw_files[i]}){_pressure_suffix(pressures, i)}')
        return im_obj, ttl

    ani = animation.FuncAnimation(fig, _upd, frames=range(n_images),
                                  interval=250, blit=False)
    ani.save(output_path, writer='pillow', dpi=100)
    plt.close(fig)
    print(f'Saved: {output_path}')


# ── Per-frame POI overlay plots ───────────────────────────────────────────

def save_frame_plots(reader, result_graph, output_frame_list,
                     raw_files, result_folder, dic, pressures=None):
    for fi in output_frame_list:
        plt.figure()
        dic.plot_img_points(reader[fi], result_graph.nodes[fi]['POIs'], size=1.2)
        plt.title(f'frame {fi}  ({raw_files[fi]}){_pressure_suffix(pressures, fi)}')
        plt.savefig(os.path.join(result_folder, f'frame_{fi:02d}.png'),
                    pad_inches=0, bbox_inches='tight', dpi=200)
        plt.close()


# ── Scalar field maps (last frame) ────────────────────────────────────────

def _save_field_map(img, cur, values, step, cmap, label, title, out):
    """Colour the tracked POIs' triangles by `values` over the frame image.

    Delaunay triangulation of the valid points, with triangles whose longest
    edge exceeds 1.8 * step dropped, so notches and gaps in the POI region
    aren't spanned by long triangles. The colour scale runs from 0 to the
    98th percentile of `values`.
    """
    from scipy.spatial import Delaunay

    vmax  = float(np.nanpercentile(values, 98))
    valid = ~np.any(np.isnan(cur), axis=1)
    cur_v, values_v = cur[valid], values[valid]

    verts, colors = [], []
    for simplex in Delaunay(cur_v).simplices:
        pts = cur_v[simplex]
        edges = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[0] - pts[2]),
        ]
        if max(edges) > 1.8 * step:
            continue
        verts.append(pts)
        colors.append(float(np.nanmean(values_v[simplex])))

    img      = np.asarray(img)
    h_i, w_i = img.shape[:2]
    asp      = w_i / max(h_i, 1)
    fw       = max(6, min(14, 8 * asp))
    fh       = max(3, fw / max(asp, 0.5))
    _vmax_img = 65535 if img.dtype == np.uint16 else 255

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.imshow(img, cmap='gray', vmin=0, vmax=_vmax_img,
              extent=[0, w_i, h_i, 0], aspect='equal')

    if verts:
        pc = PolyCollection(verts, array=np.array(colors), cmap=cmap,
                            clim=(0.0, vmax), edgecolors='none', alpha=0.75)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label=label, fraction=0.03, pad=0.02)

    ax.set_xlim(0, w_i)
    ax.set_ylim(h_i, 0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, pad_inches=0, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved: {out}')


def save_strain_map(reader, result_graph, output_frame_list, result_folder,
                    step, pressures=None):
    """Save the equivalent Green strain map for the last frame."""
    last = output_frame_list[-1]
    _save_field_map(reader[last], result_graph.nodes[last]['POIs'],
                    result_graph.nodes[last]['green_strain'][..., 3], step,
                    cmap='jet', label='equiv. Green strain',
                    title=f'Equivalent Green strain — frame {last}'
                          f'{_pressure_suffix(pressures, last)}',
                    out=os.path.join(result_folder, 'strain.png'))


def save_displacement_map(reader, result_graph, output_frame_list, result_folder,
                          POIs, step, pressures=None, mm_per_px=None):
    """Save the displacement-magnitude map for the last frame, relative to
    each POI's frame-0 position; in mm when mm_per_px is given, else px."""
    last = output_frame_list[-1]
    cur  = result_graph.nodes[last]['POIs']
    disp = np.linalg.norm(cur - POIs, axis=1)
    unit = 'mm' if mm_per_px else 'px'
    _save_field_map(reader[last], cur, disp * mm_per_px if mm_per_px else disp, step,
                    cmap='viridis', label=f'displacement ({unit})',
                    title=f'Displacement relative to frame 0 — frame {last}'
                          f'{_pressure_suffix(pressures, last)}',
                    out=os.path.join(result_folder, 'displacement.png'))


# ── Internal layout helpers ───────────────────────────────────────────────

def _view_bounds(POIs, result_graph, end_index, step):
    pad    = step
    all_xy = np.vstack([frame_pts(k, result_graph, POIs)
                        for k in range(0, end_index + 1)])
    return (np.nanmin(all_xy[:, 0]) - pad, np.nanmax(all_xy[:, 0]) + pad,
            np.nanmin(all_xy[:, 1]) - pad, np.nanmax(all_xy[:, 1]) + pad)


def _fig_sizes(x0, x1, y0, y1):
    asp = (x1 - x0) / max(y1 - y0, 1)
    fw  = max(7, min(12, 5 * asp))
    fh  = max(3, fw / max(asp, 0.5))
    pw  = max(2.8, min(3.5, 2.8 * asp))
    ph  = max(2.5, pw / max(asp, 0.5))
    return fw, fh, pw, ph


def _keyframes(frames_all):
    pool   = [0] + list(frames_all)
    stride = max(1, (len(pool) - 1) // 4)
    kf     = pool[::stride]
    if pool[-1] not in kf:
        kf.append(pool[-1])
    return kf


# ── Displacement grid ─────────────────────────────────────────────────────

def save_displacement_grid(POIs, result_graph, end_index, step, result_folder,
                           exclusion_verts=None, pressures=None):
    """Save displacement grid multi-panel PNG and animated GIF."""
    edges      = build_grid_edges(POIs, step, exclusion_verts=exclusion_verts)
    ref_segs   = make_segs(POIs, edges)
    frames_all = list(range(1, end_index + 1))
    x0, x1, y0, y1 = _view_bounds(POIs, result_graph, end_index, step)
    fw, fh, pw, ph  = _fig_sizes(x0, x1, y0, y1)
    kf = _keyframes(frames_all)

    # Multi-panel static image
    fig, axes = plt.subplots(1, len(kf), figsize=(pw * len(kf), ph), facecolor='white')
    axes = np.atleast_1d(axes)
    for idx, (ax, k) in enumerate(zip(axes, kf)):
        ax.add_collection(LineCollection(ref_segs, colors='#d0d0d0', linewidths=0.6))
        ax.scatter(POIs[:, 0], POIs[:, 1], s=1, c='#d0d0d0')
        cur = frame_pts(k, result_graph, POIs)
        ax.add_collection(LineCollection(make_segs(cur, edges),
                                         colors='#1f77b4', linewidths=0.9))
        ax.scatter(cur[:, 0], cur[:, 1], s=3, c='#1f77b4')
        ax.set_xlim(x0, x1);  ax.set_ylim(y1, y0)
        ax.set_aspect('equal')
        _lbl = 'reference' if k == 0 else f'frame {k}'
        ax.set_title(f'{_lbl}{_pressure_suffix(pressures, k)}', fontsize=9)
        ax.set_xlabel('x (px)', fontsize=8)
        ax.tick_params(labelsize=7)
        if idx > 0:
            ax.set_yticklabels([])
    axes[0].set_ylabel('y (px)', fontsize=8)
    fig.suptitle('POI grid — gray: reference  |  blue: tracked', fontsize=10)
    fig.tight_layout()
    path = os.path.join(result_folder, 'grid_deformation_panels.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')

    # Animated GIF
    fig_a, ax_a = plt.subplots(figsize=(fw, fh), facecolor='white')
    ax_a.add_collection(LineCollection(ref_segs, colors='#d0d0d0', linewidths=0.7))
    ax_a.scatter(POIs[:, 0], POIs[:, 1], s=2, c='#d0d0d0')
    lc  = LineCollection([], colors='#1f77b4', linewidths=1.1)
    sc  = ax_a.scatter([], [], s=5, c='#1f77b4')
    ax_a.add_collection(lc)
    ax_a.set_xlim(x0, x1);  ax_a.set_ylim(y1, y0)
    ax_a.set_aspect('equal')
    ax_a.set_xlabel('x (px)');  ax_a.set_ylabel('y (px)')
    ttl = ax_a.set_title('reference')
    fig_a.tight_layout()

    def _upd_disp(k):
        cur = frame_pts(k, result_graph, POIs)
        lc.set_segments(make_segs(cur, edges))
        sc.set_offsets(cur)
        _lbl = 'reference' if k == 0 else f'frame {k}'
        ttl.set_text(f'{_lbl}{_pressure_suffix(pressures, k)}')
        return lc, sc, ttl

    ani = animation.FuncAnimation(fig_a, _upd_disp, frames=[0] + frames_all,
                                  interval=250, blit=False)
    path_gif = os.path.join(result_folder, 'grid_deformation.gif')
    ani.save(path_gif, writer='pillow', dpi=100)
    plt.close(fig_a)
    print(f'Saved: {path_gif}')


# ── Overlay GIF (image + tracked grid) ───────────────────────────────────

def save_overlay_gif(reader, POIs, result_graph, end_index, step, result_folder,
                     n_images=None, exclusion_verts=None, pressures=None):
    """Animate input frames with the tracked POI grid overlaid.

    Gray lines/dots = reference positions.
    Blue lines/dots = tracked (deformed) positions for each frame.
    """
    edges    = build_grid_edges(POIs, step, exclusion_verts=exclusion_verts)
    ref_segs = make_segs(POIs, edges)

    frames_all = list(range(0, end_index + 1))
    n_frames   = n_images if n_images is not None else len(frames_all)

    img0      = np.asarray(reader[0])
    h_i, w_i  = img0.shape[:2]
    asp        = w_i / max(h_i, 1)
    fw         = max(5, min(12, 7 * asp))
    fh         = max(3, fw / max(asp, 0.5))

    fig, ax = plt.subplots(figsize=(fw, fh), facecolor='black')
    ax.axis('off')

    _vmax = 65535 if img0.dtype == np.uint16 else 255
    im_obj = ax.imshow(img0, cmap='gray', vmin=0, vmax=_vmax,
                       extent=[0, w_i, h_i, 0], aspect='equal')

    # Reference grid (drawn once, stays static)
    ax.add_collection(LineCollection(ref_segs, colors='#d0d0d0',
                                     linewidths=0.7, alpha=0.55))
    ax.scatter(POIs[:, 0], POIs[:, 1], s=2, c='#d0d0d0', alpha=0.55)

    # Deformed grid (updated each frame)
    lc_d = LineCollection([], colors='#1f77b4', linewidths=1.1)
    sc_d = ax.scatter([], [], s=5, c='#1f77b4')
    ax.add_collection(lc_d)

    ax.set_xlim(0, w_i)
    ax.set_ylim(h_i, 0)
    ttl = ax.set_title(f'frame 0{_pressure_suffix(pressures, 0)}', color='white', fontsize=9)
    fig.tight_layout(pad=0.2)

    def _upd(k):
        frame_idx = min(k, n_frames - 1)
        im_obj.set_data(np.asarray(reader[frame_idx]))
        cur = frame_pts(k, result_graph, POIs)
        lc_d.set_segments(make_segs(cur, edges))
        sc_d.set_offsets(cur)
        _lbl = 'reference' if k == 0 else f'frame {k}'
        ttl.set_text(f'{_lbl}{_pressure_suffix(pressures, k)}')
        return im_obj, lc_d, sc_d, ttl

    ani = animation.FuncAnimation(fig, _upd, frames=frames_all,
                                  interval=250, blit=False)
    path_gif = os.path.join(result_folder, 'overlay.gif')
    ani.save(path_gif, writer='pillow', dpi=100)
    plt.close(fig)
    print(f'Saved: {path_gif}')


# ── Strain grid ───────────────────────────────────────────────────────────

def save_strain_grid(POIs, result_graph, end_index, step, result_folder,
                     exclusion_verts=None, pressures=None):
    """Save strain grid multi-panel PNG and animated GIF."""
    edges       = build_grid_edges(POIs, step, exclusion_verts=exclusion_verts)
    triangles   = build_triangles(POIs, step, exclusion_verts=exclusion_verts)
    frames_all  = list(range(1, end_index + 1))
    x0, x1, y0, y1 = _view_bounds(POIs, result_graph, end_index, step)
    fw, fh, pw, ph  = _fig_sizes(x0, x1, y0, y1)
    kf = _keyframes(frames_all)

    strain_frames = [k for k in frames_all if frame_strain(k, result_graph) is not None]
    if not strain_frames:
        print('No green_strain data in result_graph — strain grid skipped.')
        return

    all_s   = np.concatenate([frame_strain(k, result_graph).ravel()
                               for k in strain_frames])
    vmin_s  = 0.0
    vmax_s  = float(np.nanpercentile(all_s, 98))
    kf_s    = [k for k in kf if k in strain_frames]

    # Multi-panel static image
    fig2, axes2 = plt.subplots(1, len(kf_s), figsize=(pw * len(kf_s), ph),
                                facecolor='white')
    axes2 = np.atleast_1d(axes2)
    for idx, (ax, k) in enumerate(zip(axes2, kf_s)):
        cur    = frame_pts(k, result_graph, POIs)
        sdat   = frame_strain(k, result_graph)
        verts  = [cur[[i0, i1, i2]] for i0, i1, i2 in triangles]
        colors = np.array([np.nanmean(sdat[[i0, i1, i2]])
                           for i0, i1, i2 in triangles])
        pc = PolyCollection(verts, array=colors, cmap='hot_r',
                            clim=(vmin_s, vmax_s), edgecolors='none', alpha=0.9)
        ax.add_collection(pc)
        ax.add_collection(LineCollection(make_segs(cur, edges),
                                         colors='k', linewidths=0.4, alpha=0.35))
        ax.set_xlim(x0, x1);  ax.set_ylim(y1, y0)
        ax.set_aspect('equal')
        ax.set_title(f'frame {k}{_pressure_suffix(pressures, k)}', fontsize=9)
        ax.set_xlabel('x (px)', fontsize=8)
        ax.tick_params(labelsize=7)
        if idx > 0:
            ax.set_yticklabels([])
        if idx == len(axes2) - 1:
            fig2.colorbar(pc, ax=ax, label='equiv. Green strain',
                          fraction=0.046, pad=0.04)
    axes2[0].set_ylabel('y (px)', fontsize=8)
    fig2.suptitle('Equivalent Green strain on deformed grid', fontsize=10)
    fig2.tight_layout()
    path2 = os.path.join(result_folder, 'strain_grid_panels.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {path2}')

    # Animated GIF
    k0      = strain_frames[0]
    cur0    = frame_pts(k0, result_graph, POIs)
    s0      = frame_strain(k0, result_graph)
    verts0  = [cur0[[i0, i1, i2]] for i0, i1, i2 in triangles]
    colors0 = np.array([np.nanmean(s0[[i0, i1, i2]])
                        for i0, i1, i2 in triangles])

    fig_b, ax_b = plt.subplots(figsize=(fw, fh), facecolor='white')
    pc2  = PolyCollection(verts0, array=colors0, cmap='hot_r',
                          clim=(vmin_s, vmax_s), edgecolors='none', alpha=0.9)
    lc2  = LineCollection(make_segs(cur0, edges), colors='k', linewidths=0.4, alpha=0.35)
    ax_b.add_collection(pc2)
    ax_b.add_collection(lc2)
    ax_b.set_xlim(x0, x1);  ax_b.set_ylim(y1, y0)
    ax_b.set_aspect('equal')
    ax_b.set_xlabel('x (px)');  ax_b.set_ylabel('y (px)')
    ttl2 = ax_b.set_title(f'frame {k0} — equiv. strain{_pressure_suffix(pressures, k0)}')
    fig_b.colorbar(pc2, ax=ax_b, label='equiv. Green strain')
    fig_b.tight_layout()

    def _upd_strain(k):
        cur    = frame_pts(k, result_graph, POIs)
        sdat   = frame_strain(k, result_graph)
        verts  = [cur[[i0, i1, i2]] for i0, i1, i2 in triangles]
        colors = np.array([np.nanmean(sdat[[i0, i1, i2]])
                           for i0, i1, i2 in triangles])
        pc2.set_verts(verts)
        pc2.set_array(colors)
        lc2.set_segments(make_segs(cur, edges))
        ttl2.set_text(f'frame {k} — equiv. strain{_pressure_suffix(pressures, k)}')
        return pc2, lc2, ttl2

    ani2 = animation.FuncAnimation(fig_b, _upd_strain, frames=strain_frames,
                                   interval=250, blit=False)
    path_gif2 = os.path.join(result_folder, 'strain_grid.gif')
    ani2.save(path_gif2, writer='pillow', dpi=100)
    plt.close(fig_b)
    print(f'Saved: {path_gif2}')
