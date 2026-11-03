"""
Edge detection module for ROI boundary refinement.

Given a rough polygon drawn by the user, detects the sample's real edges
within a search band around each polygon edge and returns a refined set
of vertices that better follows the physical boundary.

Algorithm
---------
For each edge segment of the input polygon:
  1. Sample N perpendicular intensity profiles within ``band_width`` pixels
     on both sides of the segment.
  2. The dominant intensity gradient peak in each profile locates the real
     physical boundary for that sample position.
  3. The median peak offset across all samples gives the refined position
     of the whole edge segment.
Adjacent shifted edges are intersected to compute new vertex positions.

Usage (in the main script)
--------------------------
    from dic_workflow.edge_detection import refine_roi_to_edges

    vertices, accepted = refine_roi_to_edges(
        ref_image, vertices,
        band_width=30,     # px search radius each side of each edge
        blur_sigma=2,      # pre-smoothing before gradient
        result_path=os.path.join(RESULT_FOLDER, 'edge_refinement.png'),
    )
"""

import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────


def _to_float32(image):
    """Normalise any grayscale array to [0, 1] float32."""
    img = image.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img


def _edge_offset_for_segment(img_float, p1, p2, band_width, n_samples):
    """
    Find the dominant intensity gradient perpendicular to edge p1 -> p2.

    Samples ``n_samples`` perpendicular profiles, each spanning
    [-band_width, +band_width] pixels along the edge's left-hand normal.
    The peak gradient position in each profile is collected and the median
    is returned as the signed offset (positive = toward the polygon interior
    when the polygon is wound counter-clockwise).

    Returns 0.0 when not enough valid pixels are found.
    """
    from scipy.ndimage import map_coordinates

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    tangent = p2 - p1
    length = np.linalg.norm(tangent)
    if length < 2:
        return 0.0
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])  # 90° CCW (left-hand)

    h, w = img_float.shape

    # (n_samples,) sample positions along the edge
    t_vals = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    centers = p1[None, :] + t_vals[:, None] * (p2 - p1)[None, :]  # (n, 2) [x, y]

    # (2*band_width+1,) offsets along the normal
    ss = np.arange(-band_width, band_width + 1, dtype=float)

    # Pixel coordinates for all (sample, offset) pairs
    pts_x = centers[:, 0:1] + ss[None, :] * normal[0]  # (n_samples, n_ss)
    pts_y = centers[:, 1:2] + ss[None, :] * normal[1]

    in_bounds = (
        (pts_y >= 0) & (pts_y < h - 1) &
        (pts_x >= 0) & (pts_x < w - 1)
    )

    # map_coordinates uses (row, col) = (y, x) convention
    coords = np.array([
        np.clip(pts_y, 0, h - 1).ravel(),
        np.clip(pts_x, 0, w - 1).ravel(),
    ])
    profiles = (map_coordinates(img_float, coords, order=1, mode='nearest')
                .reshape(n_samples, len(ss)))
    profiles[~in_bounds] = np.nan

    # Gradient magnitude along the normal direction for each profile
    grad = np.abs(np.gradient(profiles, axis=1))
    grad[~in_bounds] = 0.0

    # Collect peak offsets from profiles with enough in-bounds pixels
    peak_idx = np.argmax(grad, axis=1)
    good = in_bounds.sum(axis=1) >= max(5, band_width)
    if not good.any():
        return 0.0

    return float(np.median(ss[peak_idx[good]]))


def _shifted_line_points(p1, p2, offset):
    """Return (q1, q2) — the edge (p1, p2) shifted by offset along its left normal."""
    p1, p2 = np.array(p1, float), np.array(p2, float)
    tangent = p2 - p1
    length = np.linalg.norm(tangent)
    if length < 1e-6:
        return p1, p2
    tangent /= length
    shift = offset * np.array([-tangent[1], tangent[0]])
    return p1 + shift, p2 + shift


def _intersect_infinite_lines(a1, a2, b1, b2):
    """
    Intersection point of the infinite line through (a1, a2) and (b1, b2).
    Falls back to the midpoint of a2 and b1 when lines are (nearly) parallel.
    """
    d1 = np.array(a2, float) - np.array(a1, float)
    d2 = np.array(b2, float) - np.array(b1, float)
    a1, b1 = np.array(a1, float), np.array(b1, float)

    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return (np.array(a2, float) + np.array(b1, float)) / 2

    t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / denom
    return a1 + t * d1


# ── Public API ────────────────────────────────────────────────────────────────


def _polygon_signed_area(vertices):
    """Shoelace signed area. The sign tells offset_polygon which normal
    direction (left- vs right-hand) points into the polygon interior."""
    verts = np.asarray(vertices, dtype=float)
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def offset_polygon(vertices, offsets, image_shape=None):
    """Shift each polygon edge along its left-hand normal by a per-edge
    offset, then re-intersect adjacent shifted edges for the new vertices.

    offsets[i] applies to the edge vertices[i] -> vertices[(i+1) % n].
    Shallow corners (where the miter intersection would shoot far away)
    fall back to a bevel joint, same as refine_roi_to_edges.

    image_shape: optional (h, w) to clip the result into image bounds.
    """
    n = len(vertices)
    verts = [np.array(v, float) for v in vertices]

    shifted = [
        _shifted_line_points(verts[i], verts[(i + 1) % n], offsets[i])
        for i in range(n)
    ]

    max_offset = max((abs(o) for o in offsets), default=0.0)
    max_miter_dist = 4.0 * max(max_offset, 1.0)

    refined = []
    for i in range(n):
        prev = (i - 1) % n
        prev_shifted = shifted[prev]
        cur_shifted  = shifted[i]
        pt = _intersect_infinite_lines(
            prev_shifted[0], prev_shifted[1],
            cur_shifted[0],  cur_shifted[1],
        )
        if not np.all(np.isfinite(pt)) or np.linalg.norm(pt - verts[i]) > max_miter_dist:
            pt = (prev_shifted[1] + cur_shifted[0]) / 2.0
        if image_shape is not None:
            h, w = image_shape
            pt = np.array([np.clip(pt[0], 0, w - 1), np.clip(pt[1], 0, h - 1)])
        refined.append(pt.tolist())
    return refined


def shrink_polygon(vertices, shrink_px):
    """Uniformly shrink a polygon inward by shrink_px along every edge.

    Handles either winding order (the inward sign is resolved from the
    polygon's signed area) and concave polygons the same way
    refine_roi_to_edges handles its corners (miter joint, bevel fallback).
    """
    sign = 1.0 if _polygon_signed_area(vertices) > 0 else -1.0
    offsets = [sign * shrink_px] * len(vertices)
    return offset_polygon(vertices, offsets)


def refine_roi_to_edges(image, vertices, band_width=30, blur_sigma=2,
                        n_samples=60, result_path=None, interactive=True):
    """Snap each polygon edge to the nearest strong image gradient.

    Parameters
    ----------
    image       : 2-D grayscale ndarray (uint8 or uint16), the reference frame.
    vertices    : list of [x, y] points from ``select_poi_region``.
    band_width  : search radius in pixels on each side of each polygon edge.
    blur_sigma  : Gaussian pre-smoothing sigma (suppresses noise-induced peaks).
    n_samples   : number of perpendicular profiles sampled per polygon edge.
    result_path : path to save the verification PNG; ``None`` skips saving.
    interactive : if True, open an interactive matplotlib window — the user
                  closes it to **accept** or presses **r** to **reject**.

    Returns
    -------
    refined_vertices : list of [x, y] — same vertex count as the input.
    accepted         : bool — True when refinement succeeded and was accepted.
                       When False the original vertices are returned unchanged.
    """
    from scipy.ndimage import gaussian_filter

    img_f = _to_float32(image)
    if blur_sigma > 0:
        img_f = gaussian_filter(img_f, sigma=float(blur_sigma))

    n = len(vertices)
    verts = [np.array(v, float) for v in vertices]

    print(f"Edge detection: {n} edges, band_width={band_width}px, "
          f"blur_sigma={blur_sigma}, n_samples={n_samples}")

    offsets = []
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        off = _edge_offset_for_segment(img_f, p1, p2, band_width, n_samples)
        offsets.append(off)
        print(f"  edge {i:2d} -> {(i+1)%n:2d}:  offset = {off:+.1f} px")

    # Shift each edge along its normal and intersect adjacent pairs for vertices.
    # Shallow corners (near-parallel adjacent edges) fall back to a bevel
    # joint inside offset_polygon rather than an arbitrarily far miter point.
    h, w = img_f.shape
    refined = offset_polygon(verts, offsets, image_shape=(h, w))

    accepted = _show_verification(
        image, vertices, refined, offsets,
        img_float=img_f,
        result_path=result_path,
        interactive=interactive,
    )

    if not accepted:
        print("Edge refinement rejected — keeping original polygon.")
        return vertices, False

    max_shift = max(abs(o) for o in offsets)
    print(f"Edge refinement accepted  (max edge shift = {max_shift:.1f} px)")
    return refined, True


# ── Verification figure ───────────────────────────────────────────────────────


def _show_verification(image, orig_verts, ref_verts, offsets,
                       img_float=None, result_path=None, interactive=True):
    """
    Display a two-panel comparison figure and return True if the user accepts.

    Left  — original polygon on the reference image.
    Right — gradient magnitude overlay with both polygons; refined in green.

    Close the window to accept.  Press **r** before closing to reject.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import sobel

    # Build gradient magnitude for overlay (uses pre-smoothed float if available)
    _img = img_float if img_float is not None else _to_float32(image)
    Gx = sobel(_img, axis=1)
    Gy = sobel(_img, axis=0)
    grad_mag = np.hypot(Gx, Gy)
    grad_norm = grad_mag / max(float(grad_mag.max()), 1e-9)

    orig_arr = np.array(orig_verts + [orig_verts[0]], float)
    ref_arr  = np.array(ref_verts  + [ref_verts[0]],  float)

    vmin = float(np.percentile(image, 1))
    vmax = float(np.percentile(image, 99))

    accepted = [True]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), tight_layout=True)

    for ax in axes:
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')

    # ── Left: original polygon ──────────────────────────────────────────────
    axes[0].plot(orig_arr[:, 0], orig_arr[:, 1], color='tomato', lw=2)
    axes[0].scatter(orig_arr[:-1, 0], orig_arr[:-1, 1],
                    c='tomato', s=50, zorder=5)
    for i, v in enumerate(orig_verts):
        axes[0].annotate(str(i), (v[0], v[1]),
                         color='yellow', fontsize=9,
                         ha='center', va='bottom',
                         fontweight='bold')
    axes[0].set_title('Original polygon  (user-drawn)', fontsize=11)

    # ── Right: gradient overlay + both polygons ─────────────────────────────
    axes[1].imshow(grad_norm, cmap='hot', alpha=0.45,
                   vmin=0, vmax=float(np.percentile(grad_norm, 99.5)))

    axes[1].plot(orig_arr[:, 0], orig_arr[:, 1],
                 color='tomato', lw=1.5, ls='--', alpha=0.6, label='Original')
    axes[1].plot(ref_arr[:, 0], ref_arr[:, 1],
                 color='limegreen', lw=2.5, label='Refined')
    axes[1].scatter(ref_arr[:-1, 0], ref_arr[:-1, 1],
                    c='limegreen', s=50, zorder=5)

    n = len(orig_verts)
    for i in range(n):
        v = ref_verts[i]
        # annotation: vertex index + offset of the edge *starting* here
        axes[1].annotate(
            f"{i}\n{offsets[i]:+.0f}px",
            (v[0], v[1]),
            color='cyan', fontsize=7, ha='center', va='bottom',
        )

    axes[1].legend(loc='upper right', fontsize=9, framealpha=0.75)
    axes[1].set_title(
        'Gradient magnitude (bright = strong edge)  +  refined polygon\n'
        'Vertex labels: index / edge offset',
        fontsize=10,
    )

    if interactive:
        fig.suptitle(
            'Edge-refined ROI   ·   Close window to ACCEPT   ·   Press  r  to REJECT',
            fontsize=11, fontweight='bold',
        )

        def _on_key(event):
            if event.key == 'r':
                accepted[0] = False
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', _on_key)
    else:
        fig.suptitle('Edge-refined ROI — verification', fontsize=11)

    if result_path is not None:
        fig.savefig(result_path, dpi=130, bbox_inches='tight')
        print(f"Verification image saved: {result_path}")

    if interactive:
        plt.show()
    else:
        plt.close(fig)

    return accepted[0]
