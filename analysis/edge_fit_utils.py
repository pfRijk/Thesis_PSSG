"""
edge_fit_utils.py
==================
Shared edge detection/fitting primitives, used by
measure_vertical_angles_edge_ref.py and measure_block_tilt.py.

Edge LOCATION per column is gradient-based (column_edge_gradient), not a
fixed-brightness-threshold crossing: a photographed edge isn't uniformly
sharp everywhere along a sample (grazing light gives a hard 2-3 row
transition in some places, a rounded corner or softer lighting a 15-30 row
ramp elsewhere), and a single absolute brightness cutoff crosses those two
kinds of ramp at different relative points — not reliably "the same physical
feature" in both. Locating each column's own steepest rise sidesteps that.
"""
import numpy as np


def estimate_min_gradient(band, max_ramp_rows=40):
    """A minimum-gradient cutoff for column_edge_gradient, derived from the
    search band's own contrast (1st-99th percentile spread, robust against a
    couple of extreme pixels) divided by an assumed worst-case transition
    width. Any real background-to-sample transition, however gradual, should
    complete within max_ramp_rows — so its steepest single-row rise must
    exceed contrast / max_ramp_rows somewhere in the ramp; texture noise
    within the material or within the background shouldn't."""
    band = np.asarray(band, dtype=float)
    contrast = float(np.percentile(band, 99) - np.percentile(band, 1))
    return contrast / max_ramp_rows


def robust_poly_fit(indep, dep, degree=2, n_iter=5, k=2.5):
    """Fit dep = poly(indep) of the given degree, iteratively dropping
    points whose residual exceeds k * MAD (median absolute deviation) —
    rejects outliers regardless of cause (texture/noise/stray contamination/
    a tapered end) rather than relying on pre-filtering, so the final curve
    matches the bulk of the points, not the extremes. Returns
    (coeffs, kept_indep, kept_dep, residual_std)."""
    xs, ys = np.asarray(indep, dtype=float), np.asarray(dep, dtype=float)
    min_pts = degree + 3
    for _ in range(n_iter):
        if len(xs) < min_pts:
            break
        coeffs = np.polyfit(xs, ys, degree)
        resid = ys - np.polyval(coeffs, xs)
        mad = max(np.median(np.abs(resid - np.median(resid))) * 1.4826, 0.05)
        keep = np.abs(resid) <= k * mad
        if keep.all() or keep.sum() < min_pts:
            break
        xs, ys = xs[keep], ys[keep]
    coeffs = np.polyfit(xs, ys, degree)
    resid_std = float(np.std(ys - np.polyval(coeffs, xs)))
    return coeffs, xs, ys, resid_std


def despike_by_x(x, y, window_px=60, max_dev_px=15):
    """Rejects points whose y deviates from the LOCAL median of neighbouring
    points (within window_px in x, on both sides, excluding itself) by more
    than max_dev_px. Targets narrow contamination — e.g. a marker peg or
    applicator rod standing on the sample, whose top gets picked up by
    column_edge_gradient instead of the sample's own surface at that column
    — that a single global polynomial fit's residual-based outlier rejection
    (robust_poly_fit) can miss: a polynomial is least constrained at the ends
    of its x-range, so a cluster of contaminated columns right at one edge
    can bias the fit there rather than get flagged as outliers. A local
    neighbour comparison doesn't have that blind spot — it only asks "does
    this point agree with what's immediately next to it", regardless of
    where in the domain it sits. Returns (kept_x, kept_y, keep_mask) where
    keep_mask aligns with the ORIGINAL input order (not x-sorted)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    keep_sorted = np.ones(len(xs), dtype=bool)
    for i in range(len(xs)):
        lo = np.searchsorted(xs, xs[i] - window_px)
        hi = np.searchsorted(xs, xs[i] + window_px)
        neighborhood = np.concatenate([ys[lo:i], ys[i + 1:hi]])
        if len(neighborhood) < 3:
            continue  # too few neighbours nearby to judge — leave it in
        if abs(ys[i] - np.median(neighborhood)) > max_dev_px:
            keep_sorted[i] = False
    keep = np.empty(len(x), dtype=bool)
    keep[order] = keep_sorted
    return x[keep], y[keep], keep


def column_edge_gradient(img, x_range, y_range, min_gradient, smooth_px=5,
                         min_gradient_frac=0.0):
    """For each column in x_range, the sub-pixel row of the FIRST prominent
    rise in intensity (row-to-row gradient exceeding min_gradient), scanning
    down from y_range[0] — i.e. the local sample top-surface edge, located
    by where the intensity is rising FASTEST rather than by where it crosses
    one fixed absolute brightness value.

    This matters because a photographed edge isn't uniformly sharp: a
    grazing-light hard edge might transition in 2-3 rows, while a rounded
    corner or a softer-lit stretch ramps up over 15-30 rows. A fixed
    brightness threshold crosses a GRADUAL ramp at a different relative
    point than a SHARP one — nowhere near guaranteed to be "the same
    physical feature" in both cases — so the detected edge drifts depending
    on local blur, not on the sample's actual shape. Gradient location doesn't
    have that dependency: each column's OWN steepest-rise point is used,
    whatever the ramp's width, so the result is consistent across columns
    regardless of how sharp or soft the transition happens to be there.

    Each column is smoothed first (a smooth_px-row box filter) to suppress
    pixel-level speckle-texture noise, which would otherwise register its
    own small gradients. Only the FIRST rising run (from the top down) is
    used — not the single steepest gradient anywhere in the column — so a
    second, unrelated dark/bright feature further down the column (e.g. a
    groove or shadow line lower on the sample) doesn't get picked over the
    true top edge.

    min_gradient_frac > 0 additionally requires the rise to be at least that
    fraction of the column's OWN steepest rise, so a faint halo in front of
    the real edge (a side face turning into view, a blurred reflection) is
    skipped instead of taken as the first rise. Returns (col_xs, edge_rows),
    both in full-image coordinates."""
    y0, y1 = y_range
    col_xs, edge_rows = [], []
    kernel = np.ones(smooth_px) / smooth_px
    pad = smooth_px // 2
    for x in range(*x_range):
        col = img[y0:y1, x].astype(float)
        if smooth_px > 1:
            # Pad by replicating the edge values, not np.convolve's own
            # implicit zero-padding (mode='same'): zero-padding pulls the
            # smoothed value sharply toward 0 right at the window boundary
            # whenever the true pixel value there is bright (e.g. the top of
            # the search band sits near a peg, reflection, or brighter
            # background), fabricating a steep "rise" in the first couple of
            # rows that has nothing to do with any real edge — and that
            # fake rise, being first, gets picked over the true edge further
            # down.
            col_padded = np.pad(col, pad, mode='edge')
            col_s = np.convolve(col_padded, kernel, mode='valid')
        else:
            col_s = col
        grad = np.diff(col_s)
        idxs = np.nonzero(grad > max(min_gradient, min_gradient_frac * grad.max()))[0]
        if len(idxs) == 0:
            continue  # no rise steep enough anywhere in this column
        # Walk forward from the first steep-enough row through the rest of
        # this rising run, to its peak — the true centre of the transition,
        # not just the first row that happened to cross min_gradient.
        i = idxs[0]
        while i + 1 < len(grad) and grad[i + 1] >= grad[i]:
            i += 1
        if i <= 0 or i >= len(grad) - 1:
            col_xs.append(float(x))
            edge_rows.append(y0 + i + 0.5)
            continue
        # Sub-pixel refinement: parabolic interpolation on the 3 gradient
        # samples around the peak.
        g0, g1, g2 = grad[i - 1], grad[i], grad[i + 1]
        denom = g0 - 2 * g1 + g2
        delta = float(np.clip(0.5 * (g0 - g2) / denom, -1, 1)) if denom != 0 else 0.0
        col_xs.append(float(x))
        edge_rows.append(y0 + i + 0.5 + delta)
    return np.array(col_xs), np.array(edge_rows)
