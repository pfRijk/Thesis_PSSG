"""
measure_block_tilt.py
=====================
Measures the tilt of the upright sawguide block in the middle of an SG
sample, per load step, from the converted frames of a DIC result folder. The
DIC itself is not used or rerun: the POI grid does not cover the block, so
its tilt is read straight from the block's own outline in each image.

What is measured
----------------
The converted frames are already rotated by level.json, the calibrated
horizon, so the image axes ARE level and vertical. In every frame the
block's three outer edges above the guide body are located to sub-pixel
accuracy (edge_fit_utils.column_edge_gradient) and each one is fitted with a
straight line (robust_poly_fit, degree 1):

  top edge   scanned top-down per column, its angle to the horizontal
  left side  scanned left-to-right per row, its angle to the vertical
  right side scanned right-to-left per row, its angle to the vertical

The top and the sides are 90° apart on the part, so for a rigid block all of
them turn by the same amount and give the same tilt. They are independent
measurements of it, which is what the check columns use:

  top_tilt_deg        tilt read from the top edge
  side_tilt_deg       mean of left_tilt_deg and right_tilt_deg
  block_tilt_deg      mean of top_tilt_deg and side_tilt_deg — the result
  top_minus_side_deg  the check: ~0 when the top and the sides agree. A
                      CONSTANT offset means the corner is not exactly 90° in
                      the image (machining, rounded corner, camera view) and
                      does not affect the change in tilt; an offset that
                      CHANGES over the frames means one edge is mis-detected
                      (check block_edge_check.png).
  delta_block_tilt_deg  block_tilt_deg minus its frame-0 value

Sign convention, as in measure_vertical_angles_edge_ref.py: 0° =
level/vertical, positive = the top leans RIGHT (a clockwise turn in the
image), negative = the top leans LEFT.

Each fit only uses the straight part of an edge: CORNER_MARGIN_PX is kept
clear of the rounded corners, and the sides stop at the bottom of the box,
where the block disappears into the guide body. The box drawn on frame 0 is
moved along with the block from frame to frame (by the previous frame's
detected position), so it does not have to allow for the block's travel.

The applied force per frame comes from FORCE_PER_FRAME_CANDIDATES in
compute_lay_flat_force.py (frame 0 = 0 N).

Usage:
    python analysis/measure_block_tilt.py <result_folder>
    python analysis/measure_block_tilt.py --all
--all runs every DATASET_PATTERN folder under results/ and then writes the
summary. A box is asked for only where block_box.json is missing; delete
that file to mark a block again.

Window (once per dataset)
-------------------------
Block box, on frame 0. A suggested box is pre-drawn; drag its edges/corners
to adjust it, or drag a new one.
    Top    : a little above the block's top edge
    Bottom : just ABOVE where the block meets the guide body
    Sides  : a little outside the block, with dark background on both sides
    Close  : confirm (closing untouched accepts the suggestion)

Output, written to the result folder:
    block_box.json            the box, in frame-0 pixels
    block_tilt_per_frame.csv  frame | force_N | block_tilt_deg |
                              delta_block_tilt_deg | top_tilt_deg |
                              side_tilt_deg | left_tilt_deg | right_tilt_deg |
                              top_minus_side_deg | top_se_deg | side_se_deg |
                              top/left/right_n_points | top/left/right_resid_px
                              | box_dx_px | box_dy_px
                              (*_se_deg = standard error of the fitted angle,
                              *_resid_px = rms distance of the edge points to
                              their line)
    block_tilt.png            tilt per edge and the check, per load step
    block_edge_check.png      edge points and fitted lines on the first and
                              last frame
With --all, also in results/sg_block_tilt/:
    block_tilt_summary.csv    one row per dataset (incl. tilt and change at
                              SUMMARY_FORCE_N)
    block_tilt_overview.png   block tilt per load step, one panel per thickness
"""

# ── PARAMETERS ────────────────────────────────────────────────────────────────
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

DATASET_PATTERN       = 'ART_SG_*_results'   # datasets processed by --all
SUMMARY_FOLDER        = os.path.join(_ROOT, 'results', 'sg_block_tilt')

# Distance (px) kept clear of the block's rounded corners: the top edge is
# fitted between the two sides minus this margin, the sides from the top edge
# plus this margin down to the bottom of the box.
CORNER_MARGIN_PX = 20

# A real edge completes its dark-to-bright rise within this many px (see
# edge_fit_utils.estimate_min_gradient).
EDGE_MAX_RAMP_PX = 40

# Per scan line, a rise only counts as the edge if it is at least this
# fraction of the line's steepest rise. As the block turns, a faint grey halo
# (its side face coming into view) can appear just outside the bright front
# face; without this the scan stops at the halo instead of the face edge.
# 0 = take the first rise.
EDGE_MIN_RISE_FRAC = 0.5

# Edge points further than this from the median of their neighbours (within
# the window, along the edge) are dropped before the fit — a speck of dust or
# a reflection on the edge. robust_poly_fit then rejects any remaining outliers.
EDGE_DESPIKE_WINDOW_PX = 40
EDGE_DESPIKE_MAX_DEV_PX = 4

# An edge with fewer usable points than this is not fitted (NaN for that frame).
EDGE_FIT_MIN_POINTS = 30

# |top_minus_side_deg| above this is flagged in the console and the plot.
SQUARENESS_TOL_DEG = 0.5

# Load step reported in its own columns of block_tilt_summary.csv (tilt and
# change since frame 0 at this force; blank for a dataset without it).
SUMMARY_FORCE_N = 100

# Load steps labelled on the x-axis of the plots.
FORCE_TICKS_N = [0, 10, 20, 50, 100, 200]

# Plot colours (the categorical order of the shared palette).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, MUTED, GRID = '#52514e', '#898781', '#e1e0d9'

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import csv
import glob
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

from edge_fit_utils import (
    estimate_min_gradient as _estimate_min_gradient,
    robust_poly_fit as _robust_poly_fit,
    column_edge_gradient as _column_edge_gradient,
    despike_by_x as _despike_by_x,
)
from compute_lay_flat_force import FORCE_PER_FRAME_CANDIDATES


# ── IMAGES AND FORCES ─────────────────────────────────────────────────────────
def _frame_paths(result_folder):
    folder = os.path.join(result_folder, 'converted')
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.tif')]


def _forces(n_frames):
    """Force per frame (frame 0 = 0 N), or None without a matching ladder."""
    matches = [c for c in FORCE_PER_FRAME_CANDIDATES if len(c) == n_frames - 1]
    return [0.0] + [float(f) for f in matches[0]] if len(matches) == 1 else None


def _img_vmax(img):
    return float(np.percentile(img, 99.5))


# ── BOX SELECTION ─────────────────────────────────────────────────────────────
def _suggest_box(img):
    """A starting box for the window: the widest bright column run in the
    upper part of the image is the block; the box bottom stops where the
    background beside the block turns into the (bright) guide body."""
    f = img.astype(float)
    h = f.shape[0]
    upper = f[: int(h * 0.45)]
    thr = 0.5 * (np.percentile(upper, 5) + np.percentile(upper, 99))
    cols = np.nonzero((upper > thr).mean(axis=0) > 0.5)[0]
    if len(cols) == 0:
        return None
    runs = np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1)
    run = max(runs, key=len)
    bx0, bx1 = int(run[0]), int(run[-1])
    top = int(np.nonzero((upper[:, bx0:bx1] > thr).mean(axis=1) > 0.8)[0][0])
    beside = np.concatenate([f[:, max(bx0 - 60, 0):max(bx0 - 20, 1)],
                             f[:, bx1 + 20:bx1 + 60]], axis=1).mean(axis=1)
    below = np.nonzero(beside[top:] > thr)[0]
    guide_top = top + int(below[0]) if len(below) else int(h * 0.6)
    return (max(bx0 - 30, 0), min(bx1 + 30, f.shape[1] - 1),
            max(top - 30, 0), guide_top - 10)


def _select_box(img, dataset):
    suggestion = _suggest_box(img)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(img, cmap='gray', vmin=0, vmax=_img_vmax(img))
    sel = RectangleSelector(ax, lambda *a: None, useblit=True, button=[1],
                            minspanx=10, minspany=10, spancoords='pixels',
                            interactive=True)
    if suggestion is not None:
        sel.extents = suggestion
        sel.set_visible(True)
        cx = 0.5 * (suggestion[0] + suggestion[1])
        ax.set_xlim(cx - 350, cx + 350)
    ax.set_title(f'{dataset}: BLOCK BOX — top above the block, bottom just above the '
                 'guide body,\nsides on the dark background. Drag to adjust, close to confirm.',
                 fontsize=9)
    plt.tight_layout()
    plt.show()
    x0, x1, y0, y1 = sel.extents
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return tuple(int(round(v)) for v in (x0, x1, y0, y1))


def _load_or_select_box(result_folder, img, interactive):
    path = os.path.join(result_folder, 'block_box.json')
    if os.path.isfile(path):
        with open(path) as f:
            return tuple(json.load(f)['box_x0_x1_y0_y1_px'])
    if not interactive:
        return None
    box = _select_box(img, os.path.basename(result_folder))
    if box is not None:
        with open(path, 'w') as f:
            json.dump({'box_x0_x1_y0_y1_px': list(box)}, f, indent=2)
        print(f'  Saved: {path}')
    return box


# ── EDGE DETECTION ────────────────────────────────────────────────────────────
# column_edge_gradient finds the first dark-to-bright rise scanning DOWN each
# image column. The top edge is exactly that. The left side is the same scan
# on the transposed image (rows become columns, scanning right), the right
# side on the mirrored, transposed image (scanning left).
def _top_points(f, x_range, y_range):
    mg = _estimate_min_gradient(f[y_range[0]:y_range[1], x_range[0]:x_range[1]], EDGE_MAX_RAMP_PX)
    return _column_edge_gradient(f, x_range, y_range, mg,
                                 min_gradient_frac=EDGE_MIN_RISE_FRAC)   # (x, y)


def _left_points(f, row_range, x_range):
    mg = _estimate_min_gradient(f[row_range[0]:row_range[1], x_range[0]:x_range[1]], EDGE_MAX_RAMP_PX)
    return _column_edge_gradient(f.T, row_range, x_range, mg,
                                 min_gradient_frac=EDGE_MIN_RISE_FRAC)   # (y, x)


def _right_points(f, row_range, x_range):
    w = f.shape[1]
    mg = _estimate_min_gradient(f[row_range[0]:row_range[1], x_range[0]:x_range[1]], EDGE_MAX_RAMP_PX)
    rows, x_mirrored = _column_edge_gradient(
        f[:, ::-1].T, row_range, (w - x_range[1], w - x_range[0]), mg,
        min_gradient_frac=EDGE_MIN_RISE_FRAC)
    return rows, (w - 1) - x_mirrored                                    # (y, x)


def _fit_line(indep, dep):
    """Straight-line fit dep = a * indep + b after despiking. Returns None
    with too few points."""
    if len(indep) < EDGE_FIT_MIN_POINTS:
        return None
    indep, dep, _ = _despike_by_x(indep, dep, EDGE_DESPIKE_WINDOW_PX, EDGE_DESPIKE_MAX_DEV_PX)
    if len(indep) < EDGE_FIT_MIN_POINTS:
        return None
    coeffs, kept_i, kept_d, resid = _robust_poly_fit(indep, dep, degree=1)
    slope = float(coeffs[0])
    spread = float(np.sum((kept_i - kept_i.mean()) ** 2))
    se_slope = resid / np.sqrt(spread) if spread > 0 else np.nan
    return {'coeffs': coeffs, 'slope': slope,
            'se_deg': float(np.degrees(se_slope / (1 + slope ** 2))),
            'resid_px': resid, 'n': int(len(kept_i)), 'kept': (kept_i, kept_d)}


def measure_frame(img, box):
    """Tilt of the block in one frame, from the edges inside box
    (x0, x1, y0, y1). Returns a dict; angles are NaN where an edge could not
    be fitted."""
    f = img.astype(float)
    h, w = f.shape
    x0, x1 = max(box[0], 0), min(box[1], w - 1)
    y0, y1 = max(box[2], 0), min(box[3], h - 1)
    xc = (x0 + x1) // 2
    nan = float('nan')
    out = {'top': None, 'left': None, 'right': None}

    # Pass 1: rough top height from the middle half of the box, to know
    # where the straight part of the sides starts.
    tx, ty = _top_points(f, (x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4), (y0, y1))
    if len(tx) < 5:
        return {**out, 'top_tilt': nan, 'left_tilt': nan, 'right_tilt': nan, 'centre': None}
    top_rough = float(np.median(ty))
    side_rows = (int(top_rough) + CORNER_MARGIN_PX, y1)

    ly, lx = _left_points(f, side_rows, (x0, xc))
    ry, rx = _right_points(f, side_rows, (xc, x1))
    out['left'] = _fit_line(ly, lx)       # x = a * y + b
    out['right'] = _fit_line(ry, rx)

    # Pass 2: the top edge between the two sides, clear of the corners.
    xl = np.polyval(out['left']['coeffs'], top_rough) if out['left'] else x0
    xr = np.polyval(out['right']['coeffs'], top_rough) if out['right'] else x1
    top_cols = (int(np.ceil(xl)) + CORNER_MARGIN_PX, int(np.floor(xr)) - CORNER_MARGIN_PX)
    if top_cols[1] > top_cols[0]:
        tx, ty = _top_points(f, top_cols, (y0, y1))
        out['top'] = _fit_line(tx, ty)    # y = a * x + b

    # Image y points down. A clockwise turn (top leaning right) lowers the top
    # edge's right end (dy/dx > 0) and moves the sides' upper ends right
    # (dx/dy < 0), hence the opposite signs.
    out['top_tilt'] = np.degrees(np.arctan(out['top']['slope'])) if out['top'] else nan
    out['left_tilt'] = -np.degrees(np.arctan(out['left']['slope'])) if out['left'] else nan
    out['right_tilt'] = -np.degrees(np.arctan(out['right']['slope'])) if out['right'] else nan

    # Block position (centre between the sides at the top-edge height, top
    # edge height at that x), used to carry the box along to the next frame.
    xm = 0.5 * (xl + xr)
    ym = np.polyval(out['top']['coeffs'], xm) if out['top'] else top_rough
    out['centre'] = (float(xm), float(ym))
    return out


# ── ONE DATASET ───────────────────────────────────────────────────────────────
CSV_FIELDS = ['frame', 'force_N', 'block_tilt_deg', 'delta_block_tilt_deg',
              'top_tilt_deg', 'side_tilt_deg', 'left_tilt_deg', 'right_tilt_deg',
              'top_minus_side_deg', 'top_se_deg', 'side_se_deg',
              'top_n_points', 'left_n_points', 'right_n_points',
              'top_resid_px', 'left_resid_px', 'right_resid_px',
              'box_dx_px', 'box_dy_px']


def _row(i, force, m, dx, dy):
    side = np.nanmean([m['left_tilt'], m['right_tilt']]) if (
        np.isfinite(m['left_tilt']) or np.isfinite(m['right_tilt'])) else np.nan
    se = lambda k: m[k]['se_deg'] if m[k] else np.nan
    side_se = 0.5 * np.hypot(se('left'), se('right'))
    return {
        'frame': i, 'force_N': force,
        'block_tilt_deg': 0.5 * (m['top_tilt'] + side),
        'top_tilt_deg': m['top_tilt'], 'side_tilt_deg': side,
        'left_tilt_deg': m['left_tilt'], 'right_tilt_deg': m['right_tilt'],
        'top_minus_side_deg': m['top_tilt'] - side,
        'top_se_deg': se('top'), 'side_se_deg': side_se,
        **{f'{k}_n_points': (m[k]['n'] if m[k] else 0) for k in ('top', 'left', 'right')},
        **{f'{k}_resid_px': (m[k]['resid_px'] if m[k] else np.nan) for k in ('top', 'left', 'right')},
        'box_dx_px': dx, 'box_dy_px': dy,
    }


def process_dataset(result_folder, interactive=True):
    """Measures every frame of one dataset and writes its outputs. Returns
    (rows, None) or (None, reason)."""
    dataset = os.path.basename(result_folder).replace('_results', '')
    paths = _frame_paths(result_folder)
    if not paths:
        return None, 'no converted frames'
    img0 = imageio.imread(paths[0])
    box0 = _load_or_select_box(result_folder, img0, interactive)
    if box0 is None:
        return None, 'no block box'
    forces = _forces(len(paths))

    rows, measured = [], {}
    centre0, dx, dy = None, 0, 0
    for i, p in enumerate(paths):
        img = img0 if i == 0 else imageio.imread(p)
        box = (box0[0] + dx, box0[1] + dx, box0[2] + dy, box0[3] + dy)
        m = measure_frame(img, box)
        rows.append(_row(i, forces[i] if forces else None, m, dx, dy))
        if i in (0, len(paths) - 1):
            measured[i] = (img, box, m)
        if m['centre'] is not None:
            centre0 = centre0 or m['centre']
            dx = int(round(m['centre'][0] - centre0[0]))
            dy = int(round(m['centre'][1] - centre0[1]))
        else:
            print(f'  frame {i}: block not found in the box')

    ref = rows[0]['block_tilt_deg']
    for r in rows:
        r['delta_block_tilt_deg'] = r['block_tilt_deg'] - ref

    csv_path = os.path.join(result_folder, 'block_tilt_per_frame.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})

    _plot_dataset(rows, dataset, os.path.join(result_folder, 'block_tilt.png'))
    _plot_edge_check(measured, dataset, os.path.join(result_folder, 'block_edge_check.png'))

    diff = np.array([r['top_minus_side_deg'] for r in rows], dtype=float)
    print(f"{dataset}: block tilt {rows[0]['block_tilt_deg']:+.2f}° -> "
          f"{rows[-1]['block_tilt_deg']:+.2f}° (change {rows[-1]['delta_block_tilt_deg']:+.2f}°), "
          f"top-side {np.nanmean(diff):+.2f} ± {np.nanstd(diff):.2f}°")
    if np.nanmax(np.abs(diff - np.nanmean(diff))) > SQUARENESS_TOL_DEG:
        print(f'  CHECK: top and sides disagree by a varying amount (> {SQUARENESS_TOL_DEG}°) '
              '— look at block_edge_check.png')
    print(f'  Saved: {csv_path}')
    return rows, None


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def _style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=8, labelcolor=INK, color=MUTED)


def _load_axis(ax, all_forces):
    """x = load step index on a shared, evenly spaced ladder (all_forces,
    sorted), labelled with the forces in FORCE_TICKS_N. Evenly spaced
    because the ladder runs from 1 N steps to 25 N steps."""
    ticks = [all_forces.index(f) for f in FORCE_TICKS_N if f in all_forces]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{all_forces[t]:g}' for t in ticks])
    ax.set_xlabel('Force (N)', fontsize=9, color=INK)


def _x_values(rows, all_forces):
    if rows[0]['force_N'] is None:
        return [r['frame'] for r in rows]
    return [all_forces.index(r['force_N']) for r in rows]


def _plot_dataset(rows, dataset, out_path):
    forces = None if rows[0]['force_N'] is None else sorted({r['force_N'] for r in rows})
    x = _x_values(rows, forces)
    col = lambda k: np.array([r[k] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                   gridspec_kw={'height_ratios': [2.2, 1]})
    for key, label, c, lw in [('top_tilt_deg', 'Top edge', SERIES[0], 1.5),
                              ('left_tilt_deg', 'Left side', SERIES[1], 1.5),
                              ('right_tilt_deg', 'Right side', SERIES[2], 1.5),
                              ('block_tilt_deg', 'Block (mean of top and sides)', INK, 2.2)]:
        ax1.plot(x, col(key), color=c, lw=lw, marker='o', ms=3.5, label=label)
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_ylabel('Tilt from level (°)\n+ = top leaning right', fontsize=9, color=INK)
    ax1.set_title(f'{dataset}: sawguide block tilt', fontsize=10, color=INK, loc='left')
    ax1.legend(fontsize=8, frameon=False)

    diff = col('top_minus_side_deg')
    ax2.axhspan(np.nanmean(diff) - SQUARENESS_TOL_DEG, np.nanmean(diff) + SQUARENESS_TOL_DEG,
                color=GRID, alpha=0.6, lw=0, label=f'mean ± {SQUARENESS_TOL_DEG}°')
    ax2.errorbar(x, diff, yerr=np.hypot(col('top_se_deg'), col('side_se_deg')),
                 color=INK, lw=1.2, marker='o', ms=3.5, capsize=2, label='top − sides (± 1 s.e.)')
    ax2.set_ylabel('Check:\ntop − sides (°)', fontsize=9, color=INK)
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        _style(ax)
    if forces:
        _load_axis(ax2, forces)
    else:
        ax2.set_xlabel('Frame', fontsize=9, color=INK)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_edge_check(measured, dataset, out_path):
    fig, axes = plt.subplots(1, len(measured), figsize=(4.2 * len(measured), 6))
    axes = np.atleast_1d(axes)
    for ax, (i, (img, box, m)) in zip(axes, sorted(measured.items())):
        pad = 40
        x0, x1, y0, y1 = box[0] - pad, box[1] + pad, box[2] - pad, box[3] + pad
        ax.imshow(img, cmap='gray', vmin=0, vmax=_img_vmax(img))
        ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                                   fill=False, ec=MUTED, lw=0.8, ls='--'))
        for key, c in (('top', SERIES[0]), ('left', SERIES[1]), ('right', SERIES[2])):
            fit = m[key]
            if fit is None:
                continue
            ki, kd = fit['kept']
            span = np.array([ki.min() - 15, ki.max() + 15])
            if key == 'top':
                ax.plot(ki, kd, '.', color=c, ms=2)
                ax.plot(span, np.polyval(fit['coeffs'], span), color=c, lw=1.2)
            else:
                ax.plot(kd, ki, '.', color=c, ms=2)
                ax.plot(np.polyval(fit['coeffs'], span), span, color=c, lw=1.2)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"frame {i}\ntop {m['top_tilt']:+.2f}°  L {m['left_tilt']:+.2f}°  "
                     f"R {m['right_tilt']:+.2f}°", fontsize=9, color=INK)
        ax.axis('off')
    fig.suptitle(f'{dataset}: detected edges (dots) and fitted lines', fontsize=10, color=INK)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── SUMMARY OVER DATASETS ─────────────────────────────────────────────────────
def _parse(dataset):
    """'ART_SG_10mm_E20' -> (10, 'E20')"""
    parts = dataset.split('_')
    return int(parts[2].replace('mm', '')), parts[3]


def write_summary(results):
    os.makedirs(SUMMARY_FOLDER, exist_ok=True)
    csv_path = os.path.join(SUMMARY_FOLDER, 'block_tilt_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        F = SUMMARY_FORCE_N
        wr.writerow(['dataset', 'thickness_mm', 'tag', 'n_frames', 'tilt_frame0_deg',
                     f'tilt_{F:g}N_deg', f'delta_{F:g}N_deg',
                     'tilt_final_deg', 'delta_final_deg', 'max_abs_delta_deg',
                     'top_minus_side_mean_deg', 'top_minus_side_std_deg'])
        for ds, rows in sorted(results.items()):
            t, tag = _parse(ds)
            d = np.array([r['delta_block_tilt_deg'] for r in rows], dtype=float)
            diff = np.array([r['top_minus_side_deg'] for r in rows], dtype=float)
            at_f = next((r for r in rows if r['force_N'] is not None
                         and abs(r['force_N'] - F) < 1e-6), None)
            wr.writerow([ds, t, tag, len(rows), f"{rows[0]['block_tilt_deg']:.3f}",
                         f"{at_f['block_tilt_deg']:.3f}" if at_f else '',
                         f"{at_f['delta_block_tilt_deg']:.3f}" if at_f else '',
                         f"{rows[-1]['block_tilt_deg']:.3f}", f'{d[-1]:.3f}',
                         f'{np.nanmax(np.abs(d)):.3f}', f'{np.nanmean(diff):.3f}',
                         f'{np.nanstd(diff):.3f}'])
    print(f'Saved: {csv_path}')

    known = {ds: rows for ds, rows in results.items() if rows[0]['force_N'] is not None}
    if not known:
        return
    forces = sorted({r['force_N'] for rows in known.values() for r in rows})
    thicknesses = sorted({_parse(ds)[0] for ds in known})
    tags = sorted({_parse(ds)[1] for ds in known})
    fig, axes = plt.subplots(1, len(thicknesses), figsize=(3.6 * len(thicknesses), 4),
                             sharey=True, squeeze=False)
    for ax, t in zip(axes[0], thicknesses):
        for k, tag in enumerate(tags):
            ds = next((d for d in known if _parse(d) == (t, tag)), None)
            if ds is None:
                continue
            rows = known[ds]
            ax.plot(_x_values(rows, forces), [r['block_tilt_deg'] for r in rows],
                    color=SERIES[k % len(SERIES)], lw=1.5, marker='o', ms=3, label=tag)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f'{t} mm', fontsize=10, color=INK, loc='left')
        _style(ax)
        _load_axis(ax, forces)
    axes[0][0].set_ylabel('Block tilt from level (°)\n+ = top leaning right', fontsize=9, color=INK)
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.suptitle('SG sawguide block tilt per load step', fontsize=11, color=INK, x=0.01, ha='left')
    plt.tight_layout()
    out = os.path.join(SUMMARY_FOLDER, 'block_tilt_overview.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if '--all' in args:
        folders = sorted(glob.glob(os.path.join(_ROOT, 'results', DATASET_PATTERN)))
    elif args:
        folders = [args[0]]
    else:
        sys.exit('Usage: python analysis/measure_block_tilt.py <result_folder> | --all')

    results = {}
    for folder in folders:
        rows, err = process_dataset(folder)
        if rows is None:
            print(f'{os.path.basename(folder)}: skipped ({err})')
            continue
        results[os.path.basename(folder).replace('_results', '')] = rows
    if '--all' in args and results:
        write_summary(results)


if __name__ == '__main__':
    main()
