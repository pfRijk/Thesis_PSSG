"""
Preprocessing debug tool
=========================
Isolates dic_real_images.py's preprocessing chain — RAW load -> lens
undistort -> level (rotate to horizontal) -> crop -> illumination flatten ->
reflection clip -> bit-depth normalize — so each stage can be inspected on
its own, without running crop/level selection, alignment, POI selection, or DIC.

Reuses the saved crop_box / level_angle from <RESULT_FOLDER>/crop.json and
level.json if they exist (so the panels match what the real run actually
correlates), otherwise falls back to the full frame / no rotation. This
tool never draws the level points itself — run dic_real_images.py first to
create level.json.

On first run you'll be asked to draw the zoom-inset rectangle on the
reference frame (same click-drag-close interaction as select_crop); the
choice is saved to zoom.json and reused on later runs. Delete zoom.json
to re-draw it.

Usage:
    python run_dic/debug_preprocessing.py [input_folder] [result_folder]

Use the same folders as for dic_real_images.py, so the saved crop and level
angle are found.

Output:
    <RESULT_FOLDER>/debug_preprocessing/
        zoom.json                <- saved zoom-inset rectangle
        stage_frame###.png     <- one panel row per stage, + the zoomed
                                   speckle inset, for each inspected frame
        stages/frame###/        <- each stage (and its inset) as its own PNG,
                                   numbered in pipeline order (SAVE_STAGE_IMAGES)
        flatten_sigma_sweep.png <- same crop at a few FLATTEN_SIGMA values
        stats.json               <- min/max/mean/std/contrast per stage
"""

# ============================================================
#  CONFIG  --  point these at the same run as dic_real_images.py
# ============================================================
import os

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

INPUT_FOLDER  = sys.argv[1] if len(sys.argv) > 1 else r'path/to/your/images'
if not os.path.isdir(INPUT_FOLDER):
    sys.exit(f'Input folder not found: {INPUT_FOLDER!r}\n'
             'Set INPUT_FOLDER or run: python run_dic/debug_preprocessing.py <input_folder> [result_folder]')
_dataset_name = os.path.basename(os.path.normpath(INPUT_FOLDER))
RESULT_FOLDER = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join(_ROOT, 'results', _dataset_name + '_results'))

START_INDEX = 0
END_INDEX   = None    # None = use all images (only affects which frames are indexable)

FRAME_INDICES = [1]   # None = [first, last]; or an explicit list, e.g. [0, 5, 20]

FLATTEN_SIGMA       = 200    # 0 = flattening disabled
REFLECTION_CLIP_PCT = 99.9   # 100 = clipping disabled
BIT_DEPTH           = 16
LENS_PARAMS = dict(          # keep the same as dic_real_images.py; None = disabled
    cam_maker    = 'Sony',
    cam_model    = 'ILCE-7M2',
    lens_maker   = 'Samyang',
    lens_model   = 'Samyang AF 85mm f/1.4',
    focal_length = 85,
    aperture     = 8,
    distance     = 1,
)
RESIZE_TO           = None

SAVE_STAGE_IMAGES = True   # also save every stage (+ its zoom inset) as its own PNG under stages/

FLATTEN_SIGMA_SWEEP = [0, 50, 100, 200, 300]  # 0 = flattening disabled

SAVE_DPI = 300

# ============================================================
import json
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use('TkAgg')   # needed for the interactive zoom-region selector below
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import sobel

from dic_workflow.importing import scan_raw_folder, load_raw
from dic_workflow.processing import (
    flatten_illumination, clip_reflections,
    normalize_8bit, normalize_16bit, build_lens_corrector, apply_vignetting, rotate_to_level,
)
from dic_workflow.pipeline import select_crop

OUT_DIR = os.path.join(RESULT_FOLDER, 'debug_preprocessing')
os.makedirs(OUT_DIR, exist_ok=True)

normalize = normalize_16bit if BIT_DEPTH == 16 else normalize_8bit

raw_files = scan_raw_folder(INPUT_FOLDER, start=START_INDEX, end=END_INDEX)
if not raw_files:
    sys.exit(f"ERROR: no RAW files found in {INPUT_FOLDER!r} — this tool inspects "
              f"the RAW preprocessing chain, which is skipped entirely when the "
              f"input folder already holds .tif frames.")

n_images = len(raw_files)
print(f"Found {n_images} RAW images.")

crop_json_path = os.path.join(RESULT_FOLDER, 'crop.json')
if os.path.isfile(crop_json_path):
    with open(crop_json_path) as f:
        crop_box = tuple(json.load(f)['crop_box'])
    print(f"Using saved crop from {crop_json_path}: {crop_box}")
else:
    crop_box = None   # resolved to full frame once the first image is loaded
    print("No crop.json found — using full frame.")

level_json_path = os.path.join(RESULT_FOLDER, 'level.json')
if os.path.isfile(level_json_path):
    with open(level_json_path) as f:
        LEVEL_ANGLE = json.load(f)['angle_deg']
    print(f"Using saved level angle from {level_json_path}: {LEVEL_ANGLE:+.3f}°")
else:
    LEVEL_ANGLE = 0.0
    print("No level.json found — leveling disabled (this tool doesn't draw one; "
          "run dic_real_images.py once to create it).")

if FRAME_INDICES is None:
    frame_indices = sorted({0, n_images - 1})
else:
    frame_indices = FRAME_INDICES

_undist_coords = [None]   # mutable box so run_stages/_load_and_prep share the cache
_lens_mod = [None]

# ── zoom-inset region: draw once (or reuse zoom.json), applied everywhere ──

def _load_and_prep(idx):
    """Load + (optionally) undistort + level frame `idx`. No crop applied yet."""
    g = load_raw(os.path.join(INPUT_FOLDER, raw_files[idx]), resize_to=RESIZE_TO)
    if LENS_PARAMS is not None:
        import cv2
        if _undist_coords[0] is None:
            _undist_coords[0], _lens_mod[0] = build_lens_corrector(g.shape, **LENS_PARAMS)
        g = apply_vignetting(_lens_mod[0], g)
        g = cv2.remap(g, _undist_coords[0], None, cv2.INTER_LANCZOS4)
    return rotate_to_level(g, LEVEL_ANGLE)

_ref_full = _load_and_prep(frame_indices[0])
if crop_box is None:
    h, w = _ref_full.shape
    crop_box = (0, 0, w, h)
_cx1, _cy1, _cx2, _cy2 = crop_box
_ref_cropped = _ref_full[_cy1:_cy2, _cx1:_cx2]

zoom_json_path = os.path.join(OUT_DIR, 'zoom.json')
if os.path.isfile(zoom_json_path):
    with open(zoom_json_path) as f:
        zx1, zy1, zx2, zy2 = json.load(f)['zoom_box']
    print(f"Using saved zoom region from {zoom_json_path}:  "
          f"x=[{zx1}:{zx2}]  y=[{zy1}:{zy2}]  — delete zoom.json to re-draw.")
else:
    print("Draw the region to zoom in on, then close the window "
          "(close without drawing = whole crop) ...")
    zx1, zy1, zx2, zy2 = select_crop(_ref_cropped)
    with open(zoom_json_path, 'w') as f:
        json.dump({'zoom_box': [zx1, zy1, zx2, zy2]}, f, indent=2)
    print(f"Zoom region saved:  x=[{zx1}:{zx2}]  y=[{zy1}:{zy2}]")

_ch, _cw = _ref_cropped.shape[:2]
if zx1 >= zx2 or zy1 >= zy2 or zx2 > _cw or zy2 > _ch:
    sys.exit(
        f"ERROR: zoom region x=[{zx1}:{zx2}] y=[{zy1}:{zy2}] doesn't fit the "
        f"current crop ({_cw}x{_ch} px) — crop.json was likely re-selected "
        f"since {zoom_json_path} was drawn. Delete it and re-run to re-draw."
    )


# ── stage pipeline ──────────────────────────────────────────────────────

def contrast_metric(img):
    """Mean Sobel gradient magnitude — a proxy for surviving speckle texture."""
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    return float(np.mean(np.hypot(gx, gy)))


def run_stages(path, crop_box, undist_coords, lens_mod, ref_range=None):
    """Return list of (label, array) tuples, one per preprocessing stage.

    ref_range = (min, max) to normalize with, like pipeline.convert_frames does
    with the reference frame's range; None = use this frame's own range.
    """
    stages = []

    gray = load_raw(path, resize_to=RESIZE_TO)
    stages.append(('raw', gray.copy()))

    if LENS_PARAMS is not None:
        import cv2
        if undist_coords[0] is None:
            undist_coords[0], lens_mod[0] = build_lens_corrector(gray.shape, **LENS_PARAMS)
        gray = apply_vignetting(lens_mod[0], gray)
        stages.append(('vignetting-corrected', gray.copy()))
        gray = cv2.remap(gray, undist_coords[0], None, cv2.INTER_LANCZOS4)
        stages.append(('undistorted', gray.copy()))

    if LEVEL_ANGLE != 0.0:
        gray = rotate_to_level(gray, LEVEL_ANGLE)
        stages.append(('leveled', gray.copy()))

    if crop_box is None:
        h, w = gray.shape
        crop_box = (0, 0, w, h)
    x1, y1, x2, y2 = crop_box
    gray = gray[y1:y2, x1:x2]
    stages.append(('cropped', gray.copy()))

    if FLATTEN_SIGMA > 0:
        gray = flatten_illumination(gray, FLATTEN_SIGMA)
        stages.append(('flattened', gray.copy()))

    if REFLECTION_CLIP_PCT < 100:
        gray = clip_reflections(gray, REFLECTION_CLIP_PCT)
        stages.append(('clipped', gray.copy()))

    if ref_range is None:
        ref_range = (float(gray.min()), float(gray.max()))
    normed = normalize(gray, *ref_range)
    stages.append(('normalized', normed))

    return stages, crop_box


# ── run + plot per frame ────────────────────────────────────────────────

all_stats = {}

# The real conversion normalizes every frame with the reference frame's
# (index 0) post-clip range, so take it from there.
_ref_stages, _ = run_stages(os.path.join(INPUT_FOLDER, raw_files[0]),
                            crop_box, _undist_coords, _lens_mod)
_ref_pre_norm = _ref_stages[-2][1]
REF_RANGE = (float(_ref_pre_norm.min()), float(_ref_pre_norm.max()))
del _ref_stages, _ref_pre_norm

for idx in frame_indices:
    fname = raw_files[idx]
    print(f"\nFrame {idx} ({fname}):")
    stages, resolved_crop = run_stages(
        os.path.join(INPUT_FOLDER, fname), crop_box, _undist_coords, _lens_mod, REF_RANGE)
    if crop_box is None:
        crop_box = resolved_crop  # reuse the full-frame box for later frames too

    n = len(stages)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    frame_stats = {}
    vmax_by_stage = {}
    prev_label = None
    for col, (label, arr) in enumerate(stages):
        if label == 'clipped':
            # clip_reflections only caps outliers, it can never brighten the
            # image — reuse the input stage's scale so the panel doesn't look
            # falsely brighter just because its own 99.5th percentile shrank.
            vmax = vmax_by_stage[prev_label]
        else:
            vmax = max(np.percentile(arr, 99.5), 1)
            vmax_by_stage[label] = vmax
        axes[0, col].imshow(arr, cmap='gray', vmin=0, vmax=vmax)
        axes[0, col].set_title(label, fontsize=10)
        axes[0, col].axis('off')

        # zoom inset: for 'raw'/'undistorted' the crop box doesn't apply yet,
        # so zoom on the same region relative to that stage's own shape.
        h_s, w_s = arr.shape[:2]
        if label in ('raw', 'vignetting-corrected', 'undistorted', 'leveled'):
            x1c, y1c, _, _ = crop_box
            zzx1, zzy1 = min(x1c + zx1, w_s), min(y1c + zy1, h_s)
            zzx2, zzy2 = min(x1c + zx2, w_s), min(y1c + zy2, h_s)
        else:
            zzx1, zzy1, zzx2, zzy2 = zx1, zy1, min(zx2, w_s), min(zy2, h_s)
        inset = arr[zzy1:zzy2, zzx1:zzx2]
        axes[1, col].imshow(inset, cmap='gray', vmin=0, vmax=vmax)
        axes[1, col].axis('off')

        if SAVE_STAGE_IMAGES:
            stage_dir = os.path.join(OUT_DIR, 'stages', f'frame{idx:03d}')
            os.makedirs(stage_dir, exist_ok=True)
            stem = f'{col + 1:02d}_{label}'
            plt.imsave(os.path.join(stage_dir, stem + '.png'), arr, cmap='gray', vmin=0, vmax=vmax)
            plt.imsave(os.path.join(stage_dir, stem + '_inset.png'), inset, cmap='gray', vmin=0, vmax=vmax)

        frame_stats[label] = dict(
            min=float(arr.min()), max=float(arr.max()),
            mean=float(arr.mean()), std=float(arr.std()),
            contrast=contrast_metric(inset.astype(np.float32)),
        )
        print(f"  {label:12s} min={frame_stats[label]['min']:9.2f}  "
              f"max={frame_stats[label]['max']:9.2f}  "
              f"mean={frame_stats[label]['mean']:9.2f}  "
              f"std={frame_stats[label]['std']:8.2f}  "
              f"contrast={frame_stats[label]['contrast']:6.2f}")
        prev_label = label

    fig.text(0.005, 0.73, 'full frame', rotation=90, va='center', fontsize=9)
    fig.text(0.005, 0.28, f'{zx2-zx1}x{zy2-zy1} inset', rotation=90, va='center', fontsize=9)
    fig.suptitle(f'frame {idx} ({fname})')
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'stage_frame{idx:03d}.png')
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    all_stats[f'frame_{idx:03d}'] = frame_stats

with open(os.path.join(OUT_DIR, 'stats.json'), 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f"\nSaved: {os.path.join(OUT_DIR, 'stats.json')}")


# ── FLATTEN_SIGMA sweep on the first inspected frame's crop ────────────

print(f"\nFLATTEN_SIGMA sweep on frame {frame_indices[0]} ...")
x1, y1, x2, y2 = crop_box
crop0 = _load_and_prep(frame_indices[0])[y1:y2, x1:x2]

fig, axes = plt.subplots(1, len(FLATTEN_SIGMA_SWEEP),
                         figsize=(3.2 * len(FLATTEN_SIGMA_SWEEP), 3.6))
for col, sigma in enumerate(FLATTEN_SIGMA_SWEEP):
    test = flatten_illumination(crop0, sigma) if sigma > 0 else crop0.copy()
    inset = test[zy1:zy2, zx1:zx2]
    c = contrast_metric(inset.astype(np.float32))
    axes[col].imshow(inset, cmap='gray', vmin=0, vmax=max(np.percentile(inset, 99.5), 1))
    axes[col].set_title(f'sigma={sigma}\ncontrast={c:.2f}', fontsize=9)
    axes[col].axis('off')
    print(f"  sigma={sigma:<5}  contrast={c:.2f}")
fig.suptitle(f'FLATTEN_SIGMA sweep ({zx2-zx1}x{zy2-zy1} inset)')
fig.tight_layout()
sweep_path = os.path.join(OUT_DIR, 'flatten_sigma_sweep.png')
fig.savefig(sweep_path, dpi=SAVE_DPI)
plt.close(fig)
print(f"Saved: {sweep_path}")

print(f"\nAll done. Results -> {OUT_DIR}")
