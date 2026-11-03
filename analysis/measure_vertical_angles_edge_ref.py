"""
measure_vertical_angles_edge_ref.py
====================================
Measures how far the sample's vertical grid lines lean, along its length and
per load step, from the DIC result of one dataset.

The ABSOLUTE-angle reference comes from the sample's own visible top edge, so
no physical markers on the sample are needed.

What is measured
----------------
Vertical lines are placed every LINE_SPACING_PX across the DIC field of view
(one native POI-grid column each) and each line's angle is computed per frame.

Sign convention everywhere: 0° = vertical, positive = the top leans RIGHT
(the top is right of the bottom, a clockwise turn in the image), negative =
the top leans LEFT. The same as measure_block_tilt.py, so guide and block
angles can be compared directly.

  delta_angle_deg     relative to each line's OWN frame-0 angle: 0° at frame
                      0, then the change since. Whatever curvature a line
                      started with is factored out.

  absolute_angle_deg  the true angle. The sample already rests tilted at
                      frame 0 (on an object underneath it), so the POI grid
                      alone cannot give the true starting angle. The sample's
                      top edge does: it is detected in frame 0, fitted ONCE
                      over its whole length with one robust polynomial of
                      degree EDGE_FIT_DEGREE, and each line's reference angle
                      is that curve's local slope at the line's x — what a
                      rigid peg mounted perpendicular to the sample there
                      would read. Then
                          absolute = edge angle(x) + delta_angle(x, frame)
                      i.e. the edge-measured starting angle carried forward by
                      however much the grid says that region rotated. Empty if
                      no edge box is drawn.

Also reported: the force at which the sample is closest to level, the
seating force (compute_lay_flat_force.py), and the positions of the
applicators marked by hand.

The applied force per frame is picked automatically from
FORCE_PER_FRAME_CANDIDATES (in compute_lay_flat_force.py): whichever list has
exactly n_frames - 1 entries. Add a list there for a new load ladder.

Usage:
    python analysis/measure_vertical_angles_edge_ref.py <result_folder> [--batch] [--overlay-force=<F>]

The applicators and the top-edge box are saved (applicators.json,
edge_band.json) and reused on the next run, so their windows only open when
the file is missing; delete it to mark them again. --batch also saves the
result plots without opening them and skips the manual angle check, so a
dataset with all three files saved (scale.json too) runs without any window.

Output, written to the result folder:
    vertical_angle_per_index.csv  — line_index | x_ref_px | x_ref_mm | frame |
                                    angle_deg | delta_angle_deg |
                                    absolute_angle_deg | force
                                    (x_ref_mm / force only when known)
    vertical_angle_per_index.png  — delta angle and absolute angle vs position
                                    (2 panels), coloured by force, with the
                                    marked applicators as bands
    edge_reference_check.png      — detected edge points, the fit, and the
                                    per-line reference angles (if a box was drawn)
    edge_band.json                — the drawn top-edge box (if drawn)
    final_frame_angles.png        — the last frame's image with a fitted line
                                    and angle label every
                                    FINAL_FRAME_MARK_SPACING_MM
                                    (frame_angles_<F>N.png with
                                    --overlay-force=<F>)
    level_force_result.json       — force at which the sample is closest to
                                    level (if a box was drawn)
    lay_flat_force_result.json    — force needed to lay flat
    applicators.json              — marked applicator edges (if marked)
    scale.json                    — px -> mm scale (only written if missing)

Windows, in order
-----------------
1. Calibration (only if scale.json is missing)
     Left-click  : place a point (2 points = one known distance)
     Right-click : undo            Close : skip (x-axis stays in pixels)
2. Applicators
     Left-click  : 2 points = the edges of applicator 1 (red); 2 more =
                   applicator 2 (blue, optional)
     Right-click : undo            Close : confirm / skip
3. Sample top edge
     Click + drag: ONE box loosely bracketing the top edge over the sample's
                   full width — top of the box safely above the highest point
                   of the edge, bottom safely below the lowest. Precision is
                   not needed; just keep background clutter above the sample
                   out of it.
     Close       : confirm (without a box: absolute angle is skipped)
   (Windows 2 and 3 only open when applicators.json / edge_band.json are
   missing.)
4. Edge check, final-frame overlay and angle plots: close each to continue
   (saved without opening with --batch).
5. Manual angle check (if ENABLE_ANGLE_VERIFICATION_WINDOW, not with --batch)
     Left-click  : 2 points = a baseline; 3rd point = where the perpendicular
                   measurement line goes
     Right-click : undo            Close : done
"""

# ── PARAMETERS ────────────────────────────────────────────────────────────────
import os
import sys

_ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
BATCH = '--batch' in sys.argv[1:]   # save result plots without showing them
if not _ARGS:
    sys.exit('Usage: python analysis/measure_vertical_angles_edge_ref.py <result_folder> '
             '[--batch] [--overlay-force=<F>]')
RESULT_FOLDER    = _ARGS[0]
LINE_SPACING_PX      = 50  # horizontal spacing between vertical lines (pixels)
MIN_STRIP_POIS       = 2   # minimum POIs in a column to form a valid line
ANGLE_AVG_HALF_WIDTH = 6   # average over ±N neighbouring lines (0 = no averaging)

# Line despiking, per frame and BEFORE the ANGLE_AVG_HALF_WIDTH averaging: a
# line whose angle differs from the median of its neighbours (within this many
# px either side) by more than this many degrees is dropped for that frame, so
# one bad reading can't pull its neighbours' average or corrupt that line's
# frame-to-frame unwrap. _column_angle_fit already handles a single bad POI
# inside a column; this catches the rarer case of most of a column drifting
# together. Set MAX_DEV very large to disable.
ANGLE_DESPIKE_WINDOW_PX   = 150
ANGLE_DESPIKE_MAX_DEV_DEG = 2.0

APPLICATOR_BRIGHTNESS = 3  # >1 brightens the image shown for applicator selection
EDGE_BRIGHTNESS       = 3  # >1 brightens the image in the edge-reference check plot
                            # (the box-drawing window shows the raw image)

# Final-frame overlay: a fitted line + angle label on the LAST frame's image
# at this real-world spacing. Needs a calibrated scale; without one, 20 evenly
# spaced marks are drawn instead.
FINAL_FRAME_MARK_SPACING_MM = 10

# Force (N) of the frame drawn in that overlay instead of the last one, e.g.
# 100; None = the last frame (final_frame_angles.png). Any other force is
# saved as frame_angles_<F>N.png. Needs a matching load step; also settable
# with --overlay-force=<F>. The manual angle check uses the same frame.
OVERLAY_FORCE_N = None
for _a in sys.argv[1:]:
    if _a.startswith('--overlay-force='):
        OVERLAY_FORCE_N = float(_a.split('=', 1)[1])

# Each mark reads ONE native column, without the neighbour averaging of the
# main plots. A column whose angle differs from the median of its neighbours
# (within this many px either side) by more than this many degrees is treated
# as mistracked and skipped; the mark uses the nearest remaining good column.
FINAL_FRAME_DESPIKE_WINDOW_PX  = 150
FINAL_FRAME_DESPIKE_MAX_DEV_DEG = 2.0

# The largest-magnitude angle is read from a local average over this many
# native columns on each side (after despiking), so it has to be corroborated
# by its neighbours and is not one noisy reading.
FINAL_FRAME_EXTREME_AVG_HALF_WIDTH = 2

# If True, one extra window opens at the very end on the final frame, where an
# angle can be checked by hand against the automatic measurement. Nothing
# downstream depends on it.
ENABLE_ANGLE_VERIFICATION_WINDOW = True

# Half-length (px) of the perpendicular line drawn in that window. Visual
# only; roughly the sample's on-screen thickness works well.
MANUAL_PERPENDICULAR_HALF_LENGTH_PX = 100

# Degree of the single polynomial fitted through the whole top edge.
# 1 = straight line (one angle everywhere), 2 = one gentle bend, 3 = an S-bend.
# Higher degrees start fitting noise, especially near the ends: raise it only
# as far as the visible curvature needs and check edge_reference_check.png
# (yellow fit vs orange kept points).
EDGE_FIT_DEGREE = 2

# The edge fit (and with it the absolute angle) is skipped if fewer than this
# many usable edge points are found.
EDGE_FIT_MIN_POINTS = 30

# The edge is located per column at the steepest intensity rise, not at a
# fixed brightness (see edge_fit_utils.column_edge_gradient): a sharply lit
# edge and a soft, rounded one rise over very different numbers of rows, and a
# fixed threshold crosses them at different relative points. A real edge is
# assumed to complete its rise within this many rows. Raise it for much softer
# transitions; lower it if something else bright sits within this many rows of
# the edge.
EDGE_MAX_RAMP_ROWS = 40

# Edge-point despiking, before the fit. Things standing ON the sample (a
# marker peg, an applicator rod) show up as a short, rounded blob a bit above
# the true edge, and the column scan can latch onto their top. The polynomial
# fit's own outlier rejection can miss a cluster of these, especially near the
# ends where the curve is least constrained. So any point whose y differs from
# the median of its neighbours (within EDGE_DESPIKE_WINDOW_PX either side) by
# more than EDGE_DESPIKE_MAX_DEV_PX is dropped first. In
# edge_reference_check.png: magenta x = dropped here, red x = rejected later
# by the fit. Set MAX_DEV_PX very large to disable.
EDGE_DESPIKE_WINDOW_PX  = 60
EDGE_DESPIKE_MAX_DEV_PX = 15

APPLICATOR_COLORS = ['red', 'blue']   # applicator 1, applicator 2

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore', message='CUDA path could not be detected')

import csv
import json

import imageio.v2 as imageio
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import RectangleSelector
import numpy as np

from edge_fit_utils import (
    estimate_min_gradient as _estimate_min_gradient,
    robust_poly_fit as _robust_poly_fit,
    column_edge_gradient as _column_edge_gradient,
    despike_by_x as _despike_by_x,
)
from compute_lay_flat_force import FORCE_PER_FRAME_CANDIDATES, compute_lay_flat


def _img_vmax(img):
    """Full-scale display value of an 8- or 16-bit image."""
    return 65535 if img.dtype == np.uint16 else 255


def _show_result(fig):
    """Show a result figure (already saved), or just close it with --batch."""
    if BATCH:
        plt.close(fig)
    else:
        plt.show()


# ── LOAD DIC DATA ─────────────────────────────────────────────────────────────
npz_path = os.path.join(RESULT_FOLDER, 'deformation_grids.npz')
if not os.path.isfile(npz_path):
    raise FileNotFoundError(
        f'{npz_path!r} not found — run dic_real_images.py first.')

data      = np.load(npz_path)
POIs      = data['POIs']       # (N, 2)       reference [x, y] per POI
positions = data['positions']  # (F+1, N, 2)  tracked; index 0 = reference

n_frames = positions.shape[0]
print(f'Loaded {len(POIs)} POIs, {n_frames} frames (0 = reference).')

# Applied force per deformed frame (frame 1, 2, ...): the one candidate ladder
# with exactly as many entries as there are deformed frames. With no match, or
# more than one, nothing is guessed and the plots are coloured by frame number.
n_deformed = n_frames - 1
_force_matches = [c for c in FORCE_PER_FRAME_CANDIDATES if len(c) == n_deformed]
if len(_force_matches) == 1:
    force_per_frame = np.asarray(_force_matches[0], dtype=float)
    print(f'Force per frame: {force_per_frame.tolist()}')
else:
    force_per_frame = None
    _why = 'no candidate matches' if not _force_matches else f'{len(_force_matches)} candidates match'
    print(f'FORCE_PER_FRAME_CANDIDATES: {_why} {n_deformed} deformed frame(s) — '
          f'plots are coloured by frame number instead of force.')


def _force_at(frame):
    """Applied force at a frame (frame 0 = unloaded = 0 N); None if unknown."""
    if force_per_frame is None:
        return None
    return 0.0 if frame == 0 else float(force_per_frame[frame - 1])


# ── LOAD REFERENCE IMAGE ──────────────────────────────────────────────────────
converted_folder = os.path.join(RESULT_FOLDER, 'converted')
img_files = sorted(f for f in os.listdir(converted_folder) if f.endswith('.tif'))
if len(img_files) != n_frames:
    raise RuntimeError(
        f'Expected {n_frames} TIFFs in {converted_folder!r}, '
        f'found {len(img_files)}.')
ref_img = imageio.imread(os.path.join(converted_folder, img_files[0]))
ref_h, ref_w = ref_img.shape[:2]

# ── CALIBRATION ───────────────────────────────────────────────────────────────
# Reuse the scale saved by dic_real_images.py if present, otherwise fall back
# to interactive click calibration (and save it for next time).
scale_json_path = os.path.join(RESULT_FOLDER, 'scale.json')

if os.path.isfile(scale_json_path):
    with open(scale_json_path) as _f:
        mm_per_px = json.load(_f)['mm_per_px']
    if mm_per_px is not None:
        print(f'Using saved scale from {scale_json_path!r}: '
              f'1 px = {mm_per_px:.6f} mm  ({1 / mm_per_px:.3f} px/mm)')
    else:
        print(f'{scale_json_path!r} has mm_per_px=null — x-axis will be in pixels.')
else:
    _CAL_TITLE = ('CALIBRATION: click two points with a known distance\n'
                  'Right-click: undo  |  Close: skip calibration (x-axis stays in pixels)')

    fig_cal, ax_cal = plt.subplots(figsize=(11, 7))
    ax_cal.imshow(ref_img, cmap='gray', vmin=0, vmax=_img_vmax(ref_img))
    ax_cal.scatter(POIs[:, 0], POIs[:, 1], s=6, c='cyan', alpha=0.5, zorder=3)
    ax_cal.set_title(_CAL_TITLE)
    ax_cal.set_xlabel('x (px)')
    ax_cal.set_ylabel('y (px)')

    _cal = {'pts': [], 'artists': [], 'line': None}

    def _on_cal_click(event):
        if event.inaxes != ax_cal:
            return
        if event.button == 1 and len(_cal['pts']) < 2:
            x, y = event.xdata, event.ydata
            dot, = ax_cal.plot(x, y, 'yo', ms=9, zorder=5)
            lbl  = ax_cal.annotate(
                str(len(_cal['pts']) + 1), (x, y),
                color='black', fontsize=9, ha='center', va='center',
                fontweight='bold', zorder=6)
            _cal['pts'].append([x, y])
            _cal['artists'].append((dot, lbl))
            if len(_cal['pts']) == 2:
                xs = [_cal['pts'][0][0], _cal['pts'][1][0]]
                ys = [_cal['pts'][0][1], _cal['pts'][1][1]]
                _cal['line'], = ax_cal.plot(xs, ys, 'y-', lw=2, zorder=4)
                px_dist = np.hypot(xs[1] - xs[0], ys[1] - ys[0])
                ax_cal.set_title(
                    f'Calibration line: {px_dist:.1f} px  —  '
                    'close window, then enter the real distance in the terminal')
            fig_cal.canvas.draw_idle()
        elif event.button == 3 and _cal['pts']:
            _cal['pts'].pop()
            dot_, lbl_ = _cal['artists'].pop()
            dot_.remove(); lbl_.remove()
            if _cal['line'] is not None:
                _cal['line'].remove()
                _cal['line'] = None
            ax_cal.set_title(_CAL_TITLE)
            fig_cal.canvas.draw_idle()

    fig_cal.canvas.mpl_connect('button_press_event', _on_cal_click)
    plt.tight_layout()
    plt.show()

    mm_per_px = None
    cal_px = None
    if len(_cal['pts']) == 2:
        p0c, p1c = _cal['pts']
        cal_px   = float(np.hypot(p1c[0] - p0c[0], p1c[1] - p0c[1]))
        while True:
            try:
                cal_mm = float(input(f'\nCalibration: {cal_px:.1f} px — enter real distance in mm: '))
                if cal_mm > 0:
                    break
                print('  Distance must be positive.')
            except ValueError:
                print('  Please enter a number.')
        mm_per_px = cal_mm / cal_px
        print(f'Scale: 1 px = {mm_per_px:.6f} mm  ({1 / mm_per_px:.3f} px/mm)')
    else:
        print('Calibration skipped — x-axis will be in pixels.')

    with open(scale_json_path, 'w') as _f:
        json.dump({'pixel_distance': cal_px, 'mm_per_px': mm_per_px}, _f, indent=2)

# ── APPLICATOR SELECTION ──────────────────────────────────────────────────────
# Up to two applicators can be marked: the 1st (red) and, optionally, a 2nd
# (blue) — two edge clicks each, in order. They are only drawn on the plots
# and saved; no angle depends on them. Reused from applicators.json if present.
def _select_applicators():
    """Applicator window; returns the clicked edge x positions, in order."""
    fig_app, ax_app = plt.subplots(figsize=(11, 7))
    ax_app.imshow(ref_img, cmap='gray', vmin=0, vmax=_img_vmax(ref_img) / APPLICATOR_BRIGHTNESS)
    ax_app.scatter(POIs[:, 0], POIs[:, 1], s=6, c='cyan', alpha=0.5, zorder=3)
    ax_app.set_xlabel('x (px)')
    ax_app.set_ylabel('y (px)')

    _app = {'xs': [], 'artists': []}

    def _app_title():
        n = len(_app['xs'])
        if n < 2:
            return ('APPLICATOR 1 (red): click two points to mark the edges\n'
                    'Right-click: undo  |  Close: skip')
        if n < 4:
            return (f'Applicator 1 edges: {_app["xs"][0]:.0f}, {_app["xs"][1]:.0f} px\n'
                     'APPLICATOR 2 (blue, optional): click two points, or close to confirm')
        return (f'Applicator 1: {_app["xs"][0]:.0f}, {_app["xs"][1]:.0f} px   '
                f'Applicator 2: {_app["xs"][2]:.0f}, {_app["xs"][3]:.0f} px  —  close to confirm')

    def _on_app_click(event):
        if event.inaxes != ax_app:
            return
        if event.button == 1 and len(_app['xs']) < 2 * len(APPLICATOR_COLORS):
            color = APPLICATOR_COLORS[len(_app['xs']) // 2]
            side  = 'L' if len(_app['xs']) % 2 == 0 else 'R'
            x = event.xdata
            vl = ax_app.axvline(x, color=color, lw=1.5, ls='--', zorder=5)
            lbl = ax_app.annotate(
                f'  {side}  {x:.0f} px',
                xy=(x, ax_app.get_ylim()[0]), color=color, fontsize=9, zorder=6)
            _app['xs'].append(x)
            _app['artists'].append((vl, lbl))
            ax_app.set_title(_app_title())
            fig_app.canvas.draw_idle()
        elif event.button == 3 and _app['xs']:
            _app['xs'].pop()
            vl_, lbl_ = _app['artists'].pop()
            vl_.remove(); lbl_.remove()
            ax_app.set_title(_app_title())
            fig_app.canvas.draw_idle()

    ax_app.set_title(_app_title())
    fig_app.canvas.mpl_connect('button_press_event', _on_app_click)
    plt.tight_layout()
    plt.show()
    return _app['xs']


applicators_json_path = os.path.join(RESULT_FOLDER, 'applicators.json')
if os.path.isfile(applicators_json_path):
    with open(applicators_json_path) as _f:
        _saved_app = json.load(_f)
    applicator_px  = list(_saved_app['applicator_1_edges_px'])
    applicator2_px = list(_saved_app.get('applicator_2_edges_px') or [])
    print(f'Using saved applicators from {applicators_json_path!r} (delete it to mark them again).')
else:
    _app_xs = _select_applicators()
    applicator_px  = sorted(_app_xs[0:2]) if len(_app_xs) >= 2 else []
    applicator2_px = sorted(_app_xs[2:4]) if len(_app_xs) >= 4 else []

    # Keep the marked edges with the results, so later runs can read them.
    if applicator_px:
        with open(applicators_json_path, 'w') as _f:
            json.dump({'applicator_1_edges_px': [float(x) for x in applicator_px],
                       'applicator_2_edges_px': [float(x) for x in applicator2_px] or None,
                       'mm_per_px': mm_per_px}, _f, indent=2)
        print(f'  Saved: {applicators_json_path}')

if applicator_px:
    print(f'Applicator 1 edges at x = {applicator_px[0]:.1f} px  and  {applicator_px[1]:.1f} px')
else:
    print('Applicator 1 selection skipped.')
if applicator2_px:
    print(f'Applicator 2 edges at x = {applicator2_px[0]:.1f} px  and  {applicator2_px[1]:.1f} px')

# ── SAMPLE TOP-EDGE BAND SELECTION ─────────────────────────────────────────────
# ONE box loosely bracketing the top edge over the sample's full width. Only
# its y-range is used, to keep the edge search away from background clutter
# above the sample; the edge itself is located to sub-pixel accuracy inside it.
# Reused from edge_band.json if present.
def _select_edge_band():
    """Edge-box window; returns (x0, x1, y0, y1) or None if no box was drawn."""
    fig_eb, ax_eb = plt.subplots(figsize=(15, 5))
    ax_eb.imshow(ref_img, cmap='gray', vmin=0, vmax=_img_vmax(ref_img))
    ax_eb.set_xlabel('x (px)')
    ax_eb.set_ylabel('y (px)')

    _edge_band = {'box': None}

    def _on_edge_band_select(eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        _edge_band['box'] = (x0, x1, y0, y1)

    # Keep a reference: an unreferenced selector is garbage-collected and stops responding.
    _edge_band_selector = RectangleSelector(
        ax_eb, _on_edge_band_select, useblit=True, button=[1],
        minspanx=3, minspany=3, spancoords='pixels')

    ax_eb.set_title(
        "SAMPLE TOP EDGE: drag ONE box loosely bracketing it, full width\n"
        "Close to confirm  |  Close without dragging: skip absolute angle")
    plt.tight_layout()
    plt.show()
    return _edge_band['box']


edge_band_json_path = os.path.join(RESULT_FOLDER, 'edge_band.json')
if os.path.isfile(edge_band_json_path):
    with open(edge_band_json_path) as _f:
        _edge_box = tuple(json.load(_f)['box_x0_x1_y0_y1_px'])
    print(f'Using saved top-edge box from {edge_band_json_path!r} (delete it to draw it again).')
else:
    _edge_box = _select_edge_band()
    if _edge_box is not None:
        with open(edge_band_json_path, 'w') as _f:
            json.dump({'box_x0_x1_y0_y1_px': [float(v) for v in _edge_box]}, _f, indent=2)
        print(f'  Saved: {edge_band_json_path}')

if _edge_box is None:
    print('No edge band drawn — absolute angle column will be empty.')
    edge_band_y = None
else:
    _, _, _band_y0, _band_y1 = _edge_box
    edge_band_y = (max(0, int(round(_band_y0))), min(ref_h, int(round(_band_y1))))
    print(f'Sample top-edge search band: y = [{edge_band_y[0]}, {edge_band_y[1]}] px')

# ── BUILD VERTICAL LINES FROM NATIVE GRID COLUMNS ─────────────────────────────
# The POI grid is regular, so POIs are first grouped into its native vertical
# columns (by x); a line's points then all come from the SAME column. Simply
# binning POIs into LINE_SPACING_PX-wide strips could pair the top of one
# column with the bottom of another and report a diagonal that isn't there.
order      = np.argsort(POIs[:, 0])
xs_sorted  = POIs[order, 0]
gaps       = np.diff(xs_sorted)

# Consecutive-x gaps are bimodal: small (drift within one column, from any
# slight grid/camera tilt) and large (the column pitch). Split at the biggest
# jump between consecutive sorted gap values instead of a fixed threshold,
# since the column pitch varies by dataset.
gaps_sorted   = np.sort(gaps)
ratios        = gaps_sorted[1:] / np.maximum(gaps_sorted[:-1], 1e-9)
split_i       = int(np.argmax(ratios)) + 1
break_thresh  = float(gaps_sorted[split_i])  # smallest gap classified as a column break
col_break     = np.concatenate([[True], gaps > break_thresh * 0.5])
col_id        = np.empty(len(POIs), dtype=int)
col_id[order] = np.cumsum(col_break) - 1
n_cols        = int(col_id.max()) + 1

columns = []   # one entry per native column: {top_idx, bot_idx, x_ref, idxs}
for c in range(n_cols):
    idxs = np.where(col_id == c)[0]
    if len(idxs) < MIN_STRIP_POIS:
        continue

    ys      = POIs[idxs, 1]
    top_idx = int(idxs[np.argmin(ys)])  # smallest y = top in image
    bot_idx = int(idxs[np.argmax(ys)])  # largest y  = bottom
    if top_idx == bot_idx:
        continue

    columns.append({
        'top_idx': top_idx,
        'bot_idx': bot_idx,
        'x_ref':   float(POIs[idxs, 0].mean()),
        'idxs':    idxs,  # ALL POIs of the column; the angle is fitted through them
    })
columns.sort(key=lambda d: d['x_ref'])

# LINE_SPACING_PX sets the display density: keep roughly one column every
# LINE_SPACING_PX pixels. `lines` stays sorted left -> right.
col_xs    = np.array([c['x_ref'] for c in columns])
col_pitch = float(np.median(np.diff(col_xs))) if len(col_xs) > 1 else break_thresh
cols_per_line = max(1, round(LINE_SPACING_PX / col_pitch))
lines = columns[::cols_per_line]

n_lines = len(lines)
if n_lines == 0:
    raise RuntimeError(
        f'No vertical lines found with LINE_SPACING_PX={LINE_SPACING_PX}. '
        'Try a larger value.')
line_xs = np.array([ln['x_ref'] for ln in lines])

print(f'Found {n_lines} vertical lines (spacing = {LINE_SPACING_PX} px).')
for li, ln in enumerate(lines):
    print(f'  Line {li:3d}: x_ref={ln["x_ref"]:.1f} px  '
          f'POI {ln["top_idx"]} → {ln["bot_idx"]}')

# ── SAMPLE-EDGE REFERENCE ANGLE PER LINE (automatic) ──────────────────────────
# Inside the drawn y-band, the top edge is located in every column, despiked
# (EDGE_DESPIKE_* above), and fitted ONCE over the sample's whole width with a
# robust degree-EDGE_FIT_DEGREE polynomial. A line's reference angle is the
# local slope of that curve at the line's x, rotated 90°: the direction of the
# local surface normal, i.e. what a rigid peg mounted perpendicular to the
# sample there would read.
reference_angle_by_line = np.full(n_lines, np.nan)
edge_x, edge_y = np.array([]), np.array([])
edge_x_despiked, edge_y_despiked = np.array([]), np.array([])
edge_x_kept, edge_y_kept, edge_coeffs = np.array([]), np.array([]), None
deriv_coeffs = None


def _edge_reference_angle(x):
    """Reference angle (deg) from the fitted top edge at image x; + = top
    leans right. Image y points down: turning clockwise lowers the edge's
    right end (dy/dx > 0)."""
    slope = float(np.polyval(deriv_coeffs, x))
    return float(np.degrees(np.arctan2(slope, 1.0)))


if edge_band_y is None:
    have_reference = False
else:
    ebx0 = max(0, int(round(line_xs.min() - 1)))
    ebx1 = min(ref_w, int(round(line_xs.max() + 1)))
    eby0, eby1 = edge_band_y
    min_gradient = _estimate_min_gradient(ref_img[eby0:eby1, ebx0:ebx1].astype(float),
                                           max_ramp_rows=EDGE_MAX_RAMP_ROWS)
    edge_x, edge_y = _column_edge_gradient(ref_img, (ebx0, ebx1), (eby0, eby1), min_gradient)
    print(f'Sample top edge: {len(edge_x)} usable column(s) across x=[{ebx0}, {ebx1}] '
          f'(min_gradient={min_gradient:.0f}/row).')

    edge_x_despiked, edge_y_despiked, _despike_keep = _despike_by_x(
        edge_x, edge_y, window_px=EDGE_DESPIKE_WINDOW_PX, max_dev_px=EDGE_DESPIKE_MAX_DEV_PX)
    n_despiked_out = len(edge_x) - len(edge_x_despiked)
    if n_despiked_out:
        print(f'Despiked {n_despiked_out} column(s) that disagreed with their local '
              f'neighbours by > {EDGE_DESPIKE_MAX_DEV_PX}px (likely a peg/rod top, '
              f'not the sample edge).')

    have_reference = len(edge_x_despiked) >= EDGE_FIT_MIN_POINTS
    if not have_reference:
        print(f'Only {len(edge_x_despiked)} usable edge point(s) after despiking '
              f'(< EDGE_FIT_MIN_POINTS={EDGE_FIT_MIN_POINTS}) — absolute angle column '
              f'will be empty.')
    else:
        edge_coeffs, edge_x_kept, edge_y_kept, resid_std = _robust_poly_fit(
            edge_x_despiked, edge_y_despiked, degree=EDGE_FIT_DEGREE)
        deriv_coeffs = np.polyder(edge_coeffs)
        print(f'Fit degree-{EDGE_FIT_DEGREE} curve through {len(edge_x_kept)}/{len(edge_x_despiked)} '
              f'despiked edge point(s) (residual std={resid_std:.3f} px).')

        for li, ln in enumerate(lines):
            reference_angle_by_line[li] = _edge_reference_angle(ln['x_ref'])
        for li, ln in enumerate(lines):
            print(f'  Line {li:3d} (x={ln["x_ref"]:.0f}): edge angle={reference_angle_by_line[li]:+.2f}°')

if have_reference:
    fig_chk, (ax_chk1, ax_chk2) = plt.subplots(2, 1, figsize=(15, 9))
    ax_chk1.imshow(ref_img, cmap='gray', vmin=0, vmax=_img_vmax(ref_img) / EDGE_BRIGHTNESS)
    ax_chk1.scatter(POIs[:, 0], POIs[:, 1], s=6, c='cyan', alpha=0.3, zorder=2)
    ax_chk1.axhspan(edge_band_y[0], edge_band_y[1], color='lime', alpha=0.08, zorder=1)
    despiked_out_mask = ~np.isin(edge_x, edge_x_despiked)
    if despiked_out_mask.any():
        ax_chk1.scatter(edge_x[despiked_out_mask], edge_y[despiked_out_mask], s=8, c='magenta',
                         alpha=0.6, marker='x', zorder=3, label='despiked (peg/rod top)')
    fit_rejected_mask = ~np.isin(edge_x_despiked, edge_x_kept)
    if fit_rejected_mask.any():
        ax_chk1.scatter(edge_x_despiked[fit_rejected_mask], edge_y_despiked[fit_rejected_mask], s=4, c='red',
                         alpha=0.5, marker='x', zorder=3, label='fit-rejected outlier')
    ax_chk1.scatter(edge_x_kept, edge_y_kept, s=4, c='orange', alpha=0.5, zorder=4,
                     label='kept (used in fit)')
    xs_curve = np.linspace(edge_x_despiked.min(), edge_x_despiked.max(), 400)
    ax_chk1.plot(xs_curve, np.polyval(edge_coeffs, xs_curve), color='yellow', lw=1.5, zorder=5,
                 label=f'degree-{EDGE_FIT_DEGREE} fit')
    ax_chk1.set_xlabel('x (px)')
    ax_chk1.set_ylabel('y (px)')
    ax_chk1.legend(fontsize=8, loc='lower right')
    ax_chk1.set_title(f'Sample-edge reference check — {len(edge_x_kept)}/{len(edge_x)} point(s) kept\n'
                       f'(green band = search region, magenta = despiked, red = fit-rejected, '
                       f'yellow = single global fit used for all lines)')

    ax_chk2.plot(line_xs, reference_angle_by_line, 'o-', color='orange', ms=4, zorder=3)
    ax_chk2.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax_chk2.set_xlabel('x reference position (px)')
    ax_chk2.set_ylabel('Edge reference angle (°)')
    ax_chk2.set_title('Per-line reference angle used for absolute_angle_deg '
                       '(local slope of the global fit)')
    ax_chk2.grid(True, alpha=0.35)

    plt.tight_layout()
    chk_path = os.path.join(RESULT_FOLDER, 'edge_reference_check.png')
    fig_chk.savefig(chk_path, dpi=150, bbox_inches='tight')
    print(f'Saved edge reference preview: {chk_path}')
    _show_result(fig_chk)

# ── COMPUTE ANGLES ────────────────────────────────────────────────────────────
# Angle from the image y-axis: 0° = vertical, positive = the line's top leans
# right (its top is right of its bottom), negative = leans left.
#
# A column usually has more than 2 POIs, so its angle is a straight-line fit
# x = f(y) through ALL of them at that frame, not just the top and bottom one:
# with only the two extremes, one lost or mistracked POI corrupts the whole
# angle. robust_poly_fit drops points that don't fall on the same line as the
# rest.
def _column_angle_fit(positions_frame, idxs):
    """Angle (deg) of one native column at one frame, fit through all its
    valid POIs. Returns (angle, coeffs, y0, y1): coeffs is the fitted
    x = coeffs[0]*y + coeffs[1] (None if fewer than 2 valid points), y0/y1
    the y-span of the points actually used (for drawing a line that matches
    the reported angle)."""
    pts = positions_frame[idxs]
    valid = ~np.isnan(pts).any(axis=1)
    pts = pts[valid]
    if len(pts) < 2:
        return np.nan, None, None, None
    ys, xs = pts[:, 1], pts[:, 0]
    if len(pts) >= 4:
        coeffs, kept_ys, _, _ = _robust_poly_fit(ys, xs, degree=1)
    else:
        coeffs = np.polyfit(ys, xs, 1)
        kept_ys = ys
    slope = coeffs[0]  # dx per unit dy: < 0 when the top is right of the bottom
    angle = float(-np.degrees(np.arctan2(slope, 1.0)))
    return angle, coeffs, float(kept_ys.min()), float(kept_ys.max())


angles = np.full((n_lines, n_frames), np.nan)

for li, ln in enumerate(lines):
    for k in range(n_frames):
        angles[li, k] = _column_angle_fit(positions[k], ln['idxs'])[0]

# Despike across lines, per frame (see ANGLE_DESPIKE_* above). despike_by_x's
# max_dev_px is simply "max deviation in the units of the values": degrees here.
n_angle_despiked = 0
for k in range(n_frames):
    valid_k = ~np.isnan(angles[:, k])
    if valid_k.sum() < 3:
        continue
    _, _, keep = _despike_by_x(line_xs[valid_k], angles[valid_k, k],
                                window_px=ANGLE_DESPIKE_WINDOW_PX,
                                max_dev_px=ANGLE_DESPIKE_MAX_DEV_DEG)
    if not keep.all():
        bad_li = np.where(valid_k)[0][~keep]
        angles[bad_li, k] = np.nan
        n_angle_despiked += len(bad_li)
if n_angle_despiked:
    print(f'Vertical-angle plot: excluded {n_angle_despiked} line-frame reading(s) whose angle '
          f'disagreed with local neighbours by > {ANGLE_DESPIKE_MAX_DEV_DEG}° (before averaging).')

# Unwrap each line over the frames, so a ±180° wrap can't appear as a jump.
for li in range(n_lines):
    valid = ~np.isnan(angles[li])
    if valid.sum() > 1:
        angles[li, valid] = np.degrees(np.unwrap(np.radians(angles[li, valid])))

# Average each line's angle over its ±ANGLE_AVG_HALF_WIDTH neighbours
if ANGLE_AVG_HALF_WIDTH > 0:
    W = ANGLE_AVG_HALF_WIDTH
    angles_avg = np.full_like(angles, np.nan)
    for li in range(n_lines):
        lo = max(0, li - W)
        hi = min(n_lines, li + W + 1)
        angles_avg[li] = np.nanmean(angles[lo:hi], axis=0)
    angles = angles_avg

print(f'(frame 0 mean angle overall: {np.nanmean(angles[:, 0]):+.2f}°)')

# Change of each line since ITS OWN frame-0 angle.
delta = angles - angles[:, [0]]

# Absolute angle = the edge-measured starting angle + the grid's change since
# frame 0. The two are needed together: the edge gives the true frame-0 angle
# but is measured only once, so it can't follow the motion; the grid follows
# the motion but can't tell the true starting angle. At frame 0 this is the
# edge-measured angle exactly.
if have_reference:
    absolute_angle = reference_angle_by_line[:, None] + delta
else:
    absolute_angle = np.full_like(angles, np.nan)

# ── FIND THE FORCE WHERE THE SAMPLE IS CLOSEST TO LEVEL (0° ABSOLUTE) ──────────
# "Closest to level" = the frame with the smallest mean |absolute angle| over
# ALL lines (not one line reaching 0° while others are still tilted). The
# spread across the lines at that frame is reported with it, to show how
# uneven the sample still is end to end.
level_result = None
if have_reference:
    with np.errstate(invalid='ignore'):
        mean_abs_by_frame = np.nanmean(np.abs(absolute_angle), axis=0)
    valid_frames = ~np.isnan(mean_abs_by_frame)
    if valid_frames.any():
        best_frame = int(np.nanargmin(mean_abs_by_frame))
        best_vals = absolute_angle[:, best_frame]
        best_valid = best_vals[~np.isnan(best_vals)]
        best_force = _force_at(best_frame)
        level_result = {
            'frame': best_frame,
            'force_N': best_force,
            'mean_abs_deg': float(mean_abs_by_frame[best_frame]),
            'rms_deg': float(np.sqrt(np.nanmean(best_valid ** 2))),
            'min_deg': float(best_valid.min()),
            'max_deg': float(best_valid.max()),
            'n_lines': int(len(best_valid)),
        }
        force_str = f'{best_force:.1f} N' if best_force is not None else 'n/a (no FORCE_PER_FRAME)'
        print(f"\nSample closest to level (0° absolute) at frame {best_frame} (force={force_str}):")
        print(f"  mean |angle| across {level_result['n_lines']} lines: {level_result['mean_abs_deg']:.3f}°")
        print(f"  range across lines: [{level_result['min_deg']:+.3f}°, {level_result['max_deg']:+.3f}°]  "
              f"(rms={level_result['rms_deg']:.3f}°)")

        level_json_path = os.path.join(RESULT_FOLDER, 'level_force_result.json')
        with open(level_json_path, 'w') as _f:
            json.dump(level_result, _f, indent=2)
        print(f'  Saved: {level_json_path}')

# ── SEATING FORCE ─────────────────────────────────────────────────────────────
# The sample rests tilted at frame 0 on an object under its left end; force
# presses it down until it seats, after which it only creeps. The method is
# compute_lay_flat_force.py's compute_lay_flat() (see its docstring), so this
# script and the standalone one always report the same number.
lay_flat_result, _lay_flat_err = compute_lay_flat(RESULT_FOLDER)
if lay_flat_result is None:
    print(f'\nSeating force: not computed ({_lay_flat_err})')
else:
    lay_flat_force = lay_flat_result['force_N']
    force_str = f'{lay_flat_force:.1f} N' if lay_flat_force is not None else 'n/a (no FORCE_PER_FRAME)'
    if not lay_flat_result['seated']:
        force_str = f'not seated by {force_str}'
    print(f"\n{lay_flat_result['n_pois']} POI(s) {lay_flat_result['reference']}: last load step moving more "
          f"than {lay_flat_result['step_threshold_um']:g} um ends at frame {lay_flat_result['frame']} "
          f"(seating force={force_str})")

    lay_flat_json_path = os.path.join(RESULT_FOLDER, 'lay_flat_force_result.json')
    with open(lay_flat_json_path, 'w') as _f:
        json.dump(lay_flat_result, _f, indent=2)
    print(f'  Saved: {lay_flat_json_path}')

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
# Frame 0 is written with force 0 (the unloaded reference), not a blank.
csv_path = os.path.join(RESULT_FOLDER, 'vertical_angle_per_index.csv')
x_ref_mm_col = ['x_ref_mm'] if mm_per_px is not None else []
force_col = ['force'] if force_per_frame is not None else []
with open(csv_path, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['line_index', 'x_ref_px'] + x_ref_mm_col
                     + ['frame', 'angle_deg', 'delta_angle_deg', 'absolute_angle_deg']
                     + force_col)
    for li, ln in enumerate(lines):
        x_mm_val = [f'{ln["x_ref"] * mm_per_px:.4f}'] if mm_per_px is not None else []
        for k in range(n_frames):
            force_val = [f'{_force_at(k):.4f}'] if force_per_frame is not None else []
            writer.writerow([
                li,
                f'{ln["x_ref"]:.2f}',
                *x_mm_val,
                k,
                '' if np.isnan(angles[li, k]) else f'{angles[li, k]:.6f}',
                '' if np.isnan(delta[li, k])  else f'{delta[li, k]:.6f}',
                '' if np.isnan(absolute_angle[li, k]) else f'{absolute_angle[li, k]:.6f}',
                *force_val,
            ])
print(f'\nSaved: {csv_path}')

# ── FINAL-FRAME ANGLE OVERLAY ──────────────────────────────────────────────────
# The last frame's image with, at a real-world spacing, a line fitted through
# one native column's POIs at that frame and its angle as a label. The marks
# look up native columns directly (not the display-density `lines`), so their
# spacing does not depend on LINE_SPACING_PX.
#
# A mark has no averaging over neighbouring columns. Occasionally most of a
# column's points drift together for a few frames while its neighbours track
# fine; such columns are despiked out by neighbour comparison (FINAL_FRAME_
# DESPIKE_* above), so a mark uses the nearest GOOD column instead of showing
# a mistracked one.
#
# With OVERLAY_FORCE_N set, the overlay (and everything below that says
# "final frame") uses the frame at that force instead of the last one.
final_frame = n_frames - 1
overlay_name = 'final_frame_angles.png'
if OVERLAY_FORCE_N is not None:
    _at_force = [k for k in range(n_frames)
                 if _force_at(k) is not None and abs(_force_at(k) - OVERLAY_FORCE_N) < 1e-6]
    if not _at_force:
        raise RuntimeError(f'OVERLAY_FORCE_N={OVERLAY_FORCE_N:g} N: no frame at that force '
                           f'(force per frame: {force_per_frame}).')
    final_frame = _at_force[0]
    overlay_name = f'frame_angles_{OVERLAY_FORCE_N:g}N.png'
final_img = imageio.imread(os.path.join(converted_folder, img_files[final_frame]))

ff_col_xs, ff_display_angle, ff_fit_final = [], [], []
for col in columns:
    idxs = col['idxs']
    angle_final, coeffs_final, y0_final, y1_final = _column_angle_fit(positions[final_frame], idxs)
    angle_frame0 = _column_angle_fit(positions[0], idxs)[0]
    if np.isnan(angle_final) or np.isnan(angle_frame0):
        continue
    delta_final = angle_final - angle_frame0

    # absolute angle if an edge reference exists, otherwise the change since frame 0
    display_angle = (_edge_reference_angle(col['x_ref']) + delta_final if have_reference
                     else delta_final)

    ff_col_xs.append(col['x_ref'])
    ff_display_angle.append(display_angle)
    ff_fit_final.append((coeffs_final, y0_final, y1_final))

ff_col_xs = np.array(ff_col_xs)
ff_display_angle = np.array(ff_display_angle)
_, _, ff_keep = _despike_by_x(ff_col_xs, ff_display_angle,
                               window_px=FINAL_FRAME_DESPIKE_WINDOW_PX,
                               max_dev_px=FINAL_FRAME_DESPIKE_MAX_DEV_DEG)
n_ff_despiked = int((~ff_keep).sum())
if n_ff_despiked:
    print(f'Final-frame overlay: excluded {n_ff_despiked} native column(s) whose angle '
          f'disagreed with their local neighbours by > {FINAL_FRAME_DESPIKE_MAX_DEV_DEG}° '
          f'(likely a single mistracked POI) — marks fall back to the nearest good column.')

good_col_xs = ff_col_xs[ff_keep]
good_display_angle = ff_display_angle[ff_keep]
good_fit_final = [f for f, k in zip(ff_fit_final, ff_keep) if k]

# Largest-magnitude angle (either sign) at the final frame, read from a local
# average of the despiked per-column angles (FINAL_FRAME_EXTREME_AVG_HALF_WIDTH).
extreme_x = extreme_angle = None
if len(good_col_xs) > 0:
    W = FINAL_FRAME_EXTREME_AVG_HALF_WIDTH
    smoothed_angle = np.array([
        good_display_angle[max(0, i - W):i + W + 1].mean()
        for i in range(len(good_display_angle))
    ])
    extreme_i = int(np.argmax(np.abs(smoothed_angle)))
    extreme_x = float(good_col_xs[extreme_i])
    extreme_angle = float(smoothed_angle[extreme_i])
    sign_word = 'positive' if extreme_angle >= 0 else 'negative'
    print(f'Final frame — largest-magnitude angle: {extreme_angle:+.2f}° ({sign_word}) '
          f'at x={extreme_x:.0f}px (averaged over {2 * W + 1} neighbouring native columns).')

col_x_min = min(c['x_ref'] for c in columns)
col_x_max = max(c['x_ref'] for c in columns)
if mm_per_px is not None:
    mark_step_px = FINAL_FRAME_MARK_SPACING_MM / mm_per_px
    spacing_label = f'{FINAL_FRAME_MARK_SPACING_MM:g} mm'
else:
    mark_step_px = (col_x_max - col_x_min) / 20
    spacing_label = f'{mark_step_px:.0f} px (no scale calibrated — spacing is approximate)'
    print(f'No mm_per_px scale available — final-frame overlay marks spaced every '
          f'{mark_step_px:.0f}px instead of {FINAL_FRAME_MARK_SPACING_MM}mm.')

target_xs = np.arange(col_x_min, col_x_max + 1e-6, mark_step_px)

fig_ff, ax_ff = plt.subplots(figsize=(16, 6))
ax_ff.imshow(final_img, cmap='gray', vmin=0, vmax=_img_vmax(final_img))

n_marked = 0
for tx in target_xs:
    if len(good_col_xs) == 0:
        break
    ci = int(np.argmin(np.abs(good_col_xs - tx)))
    display_angle = good_display_angle[ci]
    coeffs_final, y0_final, y1_final = good_fit_final[ci]
    # Drawn from the fitted line, so the visible line matches the reported angle.
    x0f = float(coeffs_final[0] * y0_final + coeffs_final[1])
    x1f = float(coeffs_final[0] * y1_final + coeffs_final[1])

    ax_ff.plot([x0f, x1f], [y0_final, y1_final], '-', color='red', lw=1.8, zorder=4)
    ax_ff.plot(x0f, y0_final, 'o', color='red', ms=3, zorder=5)
    mid_x, top_y = (x0f + x1f) / 2, min(y0_final, y1_final)
    ax_ff.annotate(f'{display_angle:+.1f}°', xy=(mid_x, top_y), xytext=(0, -8),
                   textcoords='offset points', color='yellow', fontsize=9,
                   ha='center', va='bottom', fontweight='bold', zorder=6,
                   bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6, ec='none'))
    n_marked += 1

extreme_line = ''
if extreme_x is not None:
    ax_ff.axvline(extreme_x, color='magenta', lw=1.2, ls='--', alpha=0.8, zorder=3)
    extreme_line = f'\nLargest-magnitude angle: {extreme_angle:+.2f}° at x={extreme_x:.0f}px (magenta dashed line)'

ax_ff.set_xlabel('x (px)')
ax_ff.set_ylabel('y (px)')
angle_kind = 'absolute' if have_reference else 'delta (no edge reference available)'
_frame_label = ('Final frame' if OVERLAY_FORCE_N is None
                else f'{OVERLAY_FORCE_N:g} N')
ax_ff.set_title(f'{_frame_label} (frame {final_frame}) — {angle_kind} angle every {spacing_label} '
                f'(+ = top leaning right)\n'
                f'{n_marked} mark(s); red line = robust fit through all of this column\'s POIs{extreme_line}')
plt.tight_layout()
final_png_path = os.path.join(RESULT_FOLDER, overlay_name)
fig_ff.savefig(final_png_path, dpi=150, bbox_inches='tight')
print(f'Saved: {final_png_path}')
_show_result(fig_ff)

# ── PLOT ──────────────────────────────────────────────────────────────────────
# x-axis: real-world position when the mm scale is known, otherwise line index.
if mm_per_px is not None:
    x_vals  = line_xs * mm_per_px
    x_label = 'x reference position (mm)'
else:
    x_vals  = np.arange(n_lines)
    x_label = 'Line index  (0 = leftmost)'

# One colour per deformed frame, by applied force if known, else frame number.
deformed_frames = list(range(1, n_frames))
if force_per_frame is not None:
    color_values = force_per_frame
    color_label  = 'Force'
else:
    color_values = np.array(deformed_frames, dtype=float)
    color_label  = 'Frame'
cmap   = cm.turbo
norm   = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
sm     = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def _draw_applicator(ax, edges_px, color):
    """Mark an applicator's two edges and the band between them on a plot."""
    if mm_per_px is not None:
        edges = [x * mm_per_px for x in edges_px]
    else:
        # x-axis is the line index: use the line closest to each edge
        edges = [float(np.argmin(np.abs(line_xs - x))) for x in edges_px]
    for edge in edges:
        ax.axvline(edge, color=color, lw=1.2, ls='--', alpha=0.8)
    ax.axvspan(edges[0], edges[1], color=color, alpha=0.07)


def _draw_applicators(ax):
    for edges_px, color in zip((applicator_px, applicator2_px), APPLICATOR_COLORS):
        if edges_px:
            _draw_applicator(ax, edges_px, color)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

# Panel 1: change since frame 0 (frame 0 itself is 0 everywhere, so not drawn).
for k in deformed_frames:
    valid = ~np.isnan(delta[:, k])
    if not valid.any():
        continue
    ax1.plot(x_vals[valid], delta[valid, k],
             'o-', color=cmap(norm(color_values[k - 1])), ms=4, lw=1.5, alpha=0.85)

_draw_applicators(ax1)

cb1 = fig.colorbar(sm, ax=ax1)
cb1.set_label(color_label, fontsize=9)
ax1.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
ax1.set_xlabel(x_label, fontsize=10)
ax1.set_ylabel('Δ angle (°)', fontsize=10)
avg_label = f', avg ±{ANGLE_AVG_HALF_WIDTH}' if ANGLE_AVG_HALF_WIDTH > 0 else ''
ax1.set_title(f'Angular change vs. position\n(spacing = {LINE_SPACING_PX} px{avg_label})', fontsize=10)
ax1.grid(True, alpha=0.35)

# Panel 2: absolute angle. Frame 0 is meaningful here (the edge-measured
# angle, not 0), so it is drawn too, as a dashed reference curve.
valid0 = ~np.isnan(absolute_angle[:, 0])
if valid0.any():
    ax2.plot(x_vals[valid0], absolute_angle[valid0, 0], 'k--', lw=1.2, alpha=0.7,
              label='Frame 0 (reference)')

for k in deformed_frames:
    valid = ~np.isnan(absolute_angle[:, k])
    if not valid.any():
        continue
    ax2.plot(x_vals[valid], absolute_angle[valid, k],
              'o-', color=cmap(norm(color_values[k - 1])), ms=4, lw=1.5, alpha=0.85)

# Highlight the frame closest to level.
if level_result is not None:
    bf = level_result['frame']
    valid_bf = ~np.isnan(absolute_angle[:, bf])
    force_lbl = f"{level_result['force_N']:.1f} N" if level_result['force_N'] is not None else f'frame {bf}'
    ax2.plot(x_vals[valid_bf], absolute_angle[valid_bf, bf], 'k-', lw=2.5, alpha=0.9, zorder=5,
             label=f"Closest to level: {force_lbl} "
                   f"(mean |angle|={level_result['mean_abs_deg']:.2f}°)")

_draw_applicators(ax2)

cb2 = fig.colorbar(sm, ax=ax2)
cb2.set_label(color_label, fontsize=9)
if have_reference:
    ax2.axhline(0, color='k', lw=0.8, ls=':', alpha=0.4)
ax2.set_xlabel(x_label, fontsize=10)
ax2.set_ylabel('Absolute angle (°)', fontsize=10)
ax2.set_title('Absolute angle vs. position\n(sample-edge baseline + grid\'s change since frame 0; '
              '0° = vertical)'
              if have_reference else
              'Absolute angle unavailable\n(no edge band was drawn)', fontsize=10)
ax2.grid(True, alpha=0.35)
if valid0.any() or level_result is not None:
    ax2.legend(fontsize=8)

fig.suptitle('Vertical line angle — left to right', fontsize=12)
fig.tight_layout()

png_path = os.path.join(RESULT_FOLDER, 'vertical_angle_per_index.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight')
print(f'Saved: {png_path}')

_show_result(fig)

# ── MANUAL ANGLE VERIFICATION (optional) ────────────────────────────────────
# One more window on the final frame for checking an angle by hand, with the
# same sign convention as the rest of the script.
#
# Clicks 1-2 set a BASELINE (e.g. along the sample's edge near the spot to
# check): a reference direction, the role the local edge-fit slope plays in
# the automatic measurement. Click 3 is a point the measurement line should
# pass through; the line itself is drawn automatically, perpendicular to the
# baseline, so the result does not depend on how precisely a perpendicular
# could be clicked by hand.
if ENABLE_ANGLE_VERIFICATION_WINDOW and not BATCH:
    fig_vf, ax_vf = plt.subplots(figsize=(16, 6))
    ax_vf.imshow(final_img, cmap='gray', vmin=0, vmax=_img_vmax(final_img))
    if extreme_x is not None:
        ax_vf.axvline(extreme_x, color='magenta', lw=1.2, ls='--', alpha=0.8, zorder=3,
                       label=f'largest-magnitude angle: {extreme_angle:+.2f}° at x={extreme_x:.0f}px')
        ax_vf.legend(fontsize=8, loc='lower right')
    ax_vf.set_xlabel('x (px)')
    ax_vf.set_ylabel('y (px)')

    _vf = {'pts': [], 'artists': [], 'baseline': None, 'perp': None, 'perp_ends': None}

    def _vf_perp_endpoints(p1, p2, p3):
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        blen = float(np.hypot(bx, by))
        if blen == 0:
            return None
        ux, uy = -by / blen, bx / blen  # baseline direction rotated 90°
        h = MANUAL_PERPENDICULAR_HALF_LENGTH_PX
        return (p3[0] + h * ux, p3[1] + h * uy), (p3[0] - h * ux, p3[1] - h * uy)

    def _vf_manual_angle():
        """Angle (deg) of the drawn perpendicular line, script sign convention."""
        end_a, end_b = _vf['perp_ends']
        top, bot = (end_a, end_b) if end_a[1] <= end_b[1] else (end_b, end_a)
        dx, dy = bot[0] - top[0], bot[1] - top[1]
        return -np.degrees(np.arctan2(dx, dy))

    def _vf_title():
        n = len(_vf['pts'])
        if n < 2:
            return ('MANUAL ANGLE CHECK: click 2 points to set a baseline direction '
                     '(e.g. along the sample edge)\n'
                     'Right-click: undo  |  Close: done')
        if n < 3:
            return ('Baseline set. Click a 3rd point — a perpendicular measurement '
                     'line is drawn through it\n'
                     'Right-click: undo  |  Close: done')
        manual_angle = _vf_manual_angle()
        compare = ''
        if extreme_x is not None:
            compare = f'  |  vs. largest-magnitude angle {extreme_angle:+.2f}° → diff {manual_angle - extreme_angle:+.2f}°'
        return (f'Manual angle (perpendicular to baseline): {manual_angle:+.2f}°{compare}\n'
                'Right-click: undo  |  Close: done')

    def _on_vf_click(event):
        if event.inaxes != ax_vf:
            return
        if event.button == 1 and len(_vf['pts']) < 3:
            x, y = event.xdata, event.ydata
            color = 'cyan' if len(_vf['pts']) < 2 else 'yellow'
            dot, = ax_vf.plot(x, y, 'o', color=color, ms=7, zorder=5)
            _vf['pts'].append((x, y))
            _vf['artists'].append(dot)
            if len(_vf['pts']) == 2:
                xs = [_vf['pts'][0][0], _vf['pts'][1][0]]
                ys = [_vf['pts'][0][1], _vf['pts'][1][1]]
                _vf['baseline'], = ax_vf.plot(xs, ys, '-', color='cyan', lw=2, zorder=4)
            elif len(_vf['pts']) == 3:
                end_a, end_b = _vf_perp_endpoints(*_vf['pts'])
                _vf['perp_ends'] = (end_a, end_b)
                _vf['perp'], = ax_vf.plot([end_a[0], end_b[0]], [end_a[1], end_b[1]],
                                           '-', color='yellow', lw=2, zorder=4)
            ax_vf.set_title(_vf_title())
            fig_vf.canvas.draw_idle()
        elif event.button == 3 and _vf['pts']:
            _vf['pts'].pop()
            _vf['artists'].pop().remove()
            if len(_vf['pts']) == 1 and _vf['baseline'] is not None:
                _vf['baseline'].remove()
                _vf['baseline'] = None
            if len(_vf['pts']) == 2 and _vf['perp'] is not None:
                _vf['perp'].remove()
                _vf['perp'] = None
                _vf['perp_ends'] = None
            ax_vf.set_title(_vf_title())
            fig_vf.canvas.draw_idle()

    ax_vf.set_title(_vf_title())
    fig_vf.canvas.mpl_connect('button_press_event', _on_vf_click)
    plt.tight_layout()
    plt.show()

    if len(_vf['pts']) == 3 and _vf['perp_ends'] is not None:
        manual_angle = _vf_manual_angle()
        print(f'\nManual angle check (perpendicular to a hand-picked baseline): {manual_angle:+.2f}°')
        if extreme_x is not None:
            print(f'  vs. largest-magnitude angle {extreme_angle:+.2f}° at x={extreme_x:.0f}px '
                  f'→ diff {manual_angle - extreme_angle:+.2f}°')
