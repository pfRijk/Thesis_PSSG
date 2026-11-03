"""
DIC validation on your own image
================================
One tool for finding out what the DIC pipeline can resolve on a given speckle
pattern, with your own subset size, grid step and correlation settings. Select
an image and it imposes known deformations on it, measures them with the same
correlation engine as run_dic/dic_real_images.py and compares the result with
the known answer.

Tests on one image (warped with a quintic spline, so it does not share the
engine's cubic interpolator; every test frame gets fresh sensor noise):

  translation   Smallest detectable shift. A ladder of small shifts, plus zero
                shifts that give the noise floor:
                    sigma0_poi   pooled per-point std      (precision of one point)
                    sigma0_stat  std of the mean over points (floor of the statistic)
                A shift is detected if |mean| > DETECT_SNR * sigma0_stat and the
                mean is within DETECT_REL_TOL of the imposed value. The limit is
                the smallest shift from which every larger one is detected.
  strain        The same for a uniform strain along x; the statistic is the
                fitted slope of u against x.
  upper_bound   Largest shift that can be bridged in one step: the mean is within
                UPPER_REL_TOL and at least UPPER_MIN_VALID of the points are
                tracked, for that shift and every smaller one.
  resolution    Spatial resolution: a sinusoidal displacement along x with a
                ladder of wavelengths, all with strain amplitude EPS0. Measured
                amplitude / imposed amplitude is compared with first-order
                shape-function theory (Schreier & Sutton 2002):
                    displacement  H_u = sin z / z
                    strain        H_e = 3 (sin z - z cos z) / z^3,   z = pi L / lambda

Each measurement is an isolated pair of frames (reference + deformed), so the
correlation has no intermediate frames to bridge through.

Instead of a photo, IMAGE_PATH = 'synthetic' uses an analytic Gaussian-blob
speckle: shifts and strains are then rendered exactly, without interpolation.
(The resolution test still warps the rendered pattern.)

Tests on real data, run when their folder is given:

  noise_floor   NOISE_FOLDER: repeat photos of an undeformed sample. Nothing
                moves, so any measured displacement is noise of the whole setup
                (sensor, vibration, lighting). Opens the pipeline's ROI windows.
  convergence   RESULT_FOLDER: re-correlates a finished dic_real_images.py result
                with each subset, reusing its frames and selections. If the fields
                stop changing with subset size, the subset is not smoothing away
                real deformation. Compares against the largest subset, so give
                at least two sizes.
  frame_step    RESULT_FOLDER (one result, or a folder of results): largest shift
                between consecutive frames in the real test, to compare with the
                upper_bound result.

Usage:
    python validation/validate_dic.py                     (asks for an image)
    python validation/validate_dic.py --image photo.tif --subsets 41 81 --step 20
    python validation/validate_dic.py --image synthetic --tests translation upper_bound
    python validation/validate_dic.py --noise-folder repeat_photos/ --subsets 41 81
    python validation/validate_dic.py --result-folder results/my_test_results --subsets 41 61 81

Settings are the constants below; the options above override them for one run
(python validation/validate_dic.py --help lists them all).

Output, in validation_results/<image name>/ (or --output):
    summary.json                      settings and the limits found
    <test>.csv, <test>.png            per-measurement results and plots
    crop.json                         the crop; delete it to select again
"""

# ============================================================
#  PARAMETERS  --  defaults, overridden by the command line
# ============================================================
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

# ── Input ────────────────────────────────────────────────────
IMAGE_PATH    = None      # None = file dialog; a path (tif, png, jpg, RAW); or 'synthetic'
CROP          = 'select'  # 'select' = draw it once; None = whole image; or (x1, y1, x2, y2)
NOISE_FOLDER  = None      # repeat photos of an undeformed sample (noise_floor)
RESULT_FOLDER = None      # finished dic_real_images.py result (convergence, frame_step)
OUTPUT_FOLDER = None      # None = validation_results/<image name>

IMAGE_TESTS = ['translation', 'strain', 'upper_bound', 'resolution']

# ── DIC correlation ──────────────────────────────────────────
SUBSET_SIZES    = [81]    # full subset width [px], odd; several = compare them
STEP            = 20      # POI grid spacing [px]
ZNSSD_LIMIT     = 0.15
LOST_LEVEL      = 0.30
SUBPIXEL_METHOD = 'fagn'  # 'fagn' | 'icgn'
POI_MARGIN      = 60      # min distance of POIs to the image border [px]
TORCH_DEVICE    = 'auto'  # 'auto' | 'cuda' | 'cpu'

# ── Imposed deformation ──────────────────────────────────────
SHIFT_DIRECTIONS = ['x', 'y']
NOISE_STD        = 2.0    # sensor noise, in grey levels of 255 (scaled for 16-bit images)
SEED             = 42
N_ZERO_PAIRS     = 8      # zero-shift pairs used for the noise floor
SMALL_SHIFTS_PX  = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
STRAINS          = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
LARGE_SHIFTS_PX  = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40]
WARP_SPLINE_ORDER = 5     # differs from the engine's cubic interpolator on purpose

# ── Pass criteria ────────────────────────────────────────────
DETECT_SNR      = 3.0     # detected: statistic above this multiple of the zero floor...
DETECT_REL_TOL  = 0.25    # ...and within this relative error
UPPER_REL_TOL   = 0.05    # upper bound: mean within this relative error...
UPPER_MIN_VALID = 0.95    # ...and at least this fraction of the points tracked
MISMATCH_PX     = 0.5     # a point off by more than this counts as mismatched

# ── Resolution test ──────────────────────────────────────────
WAVELENGTHS_PX = [30, 40, 50, 60, 80, 100, 120, 160, 200, 260, 320, 400, 500, 640]
EPS0           = 5e-3     # strain amplitude, the same for every wavelength
RES_STEP_X     = 4        # POI spacing along x, to sample the sinusoid
RES_ROW_OFFSETS = [-20, 0, 20]   # POI rows relative to the image centre [px]
TRANSFER_LEVEL = 0.9      # resolution = shortest wavelength resolved to this ratio

# ── Synthetic speckle (IMAGE_PATH = 'synthetic') ─────────────
IMG_SIZE  = (512, 512)    # (height, width)
N_DOTS    = 3500
DOT_SIGMA = 2.5           # blob std [px]

# ── Preprocessing of the noise_floor photos ──────────────────
# Keep the same as run_dic/dic_real_images.py. Not used for .tif input.
BIT_DEPTH           = 16
FLATTEN_SIGMA       = 200
REFLECTION_CLIP_PCT = 99.9
LENS_PARAMS         = None

# ── Convergence test: which points count as interior ─────────
EDGE_BAND_PX      = 20    # closer than this to the ROI boundary -> 'edge'
EXCLUSION_BAND_PX = 120   # within this of the exclusion polygon -> 'near_exclusion'

ALL_TESTS = IMAGE_TESTS + ['noise_floor', 'convergence', 'frame_step']


def parse_args():
    p = argparse.ArgumentParser(description='Validate the DIC pipeline on your own image.')
    p.add_argument('--image', default=IMAGE_PATH, help="image file, or 'synthetic'; omitted = file dialog")
    p.add_argument('--crop', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'), help='crop box in px')
    p.add_argument('--no-crop', action='store_true', help='use the whole image')
    p.add_argument('--noise-folder', default=NOISE_FOLDER, help='repeat photos of an undeformed sample')
    p.add_argument('--result-folder', default=RESULT_FOLDER, help='finished dic_real_images.py result')
    p.add_argument('--output', default=OUTPUT_FOLDER, help='output folder')
    p.add_argument('--tests', nargs='+', choices=ALL_TESTS, help='tests to run (default: all that have their input)')
    p.add_argument('--subsets', type=int, nargs='+', default=SUBSET_SIZES, help='subset widths in px (odd)')
    p.add_argument('--step', type=int, default=STEP, help='POI grid spacing in px')
    p.add_argument('--znssd', type=float, default=ZNSSD_LIMIT, help='ZNSSD limit')
    p.add_argument('--directions', nargs='+', choices=['x', 'y'], default=SHIFT_DIRECTIONS, help='shift directions')
    p.add_argument('--noise-std', type=float, default=NOISE_STD, help='imposed noise, grey levels of 255')
    a = p.parse_args()
    for L in a.subsets:
        if L % 2 == 0 or L < 5:
            p.error(f'subset width must be odd and at least 5, got {L}')
    return a


ARGS = parse_args()

# ============================================================
#  IMPORTS  --  matplotlib backend must be set before pyplot
# ============================================================
import warnings
warnings.filterwarnings('ignore', message='CUDA path could not be detected')

import matplotlib
matplotlib.use('TkAgg')   # interactive windows for the crop and ROI selection

import csv
import glob
import json
import shutil
import tempfile
import time
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dic_workflow.dataset_runner import DatasetConfig, run_dataset
from dic_workflow.importing import RAW_EXTENSIONS, load_raw
from dic_workflow.pipeline import select_crop
from dic_workflow.synthetic import (
    make_speckle_dots, render_speckle, deform_translation, deform_uniform_strain_x,
)

from dic import dic
import torch
dic.torch_device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')) \
    if TORCH_DEVICE == 'auto' else torch.device(TORCH_DEVICE)
dic.barron_filter   = dic.barron_filter.to(dic.torch_device)
dic.barron_filter_x = dic.barron_filter_x.to(dic.torch_device)
dic.barron_filter_y = dic.barron_filter_y.to(dic.torch_device)
print(f'Using device: {dic.torch_device}')

# Plot style: fixed colour per subset, in the order of SUBSET_SIZES; marker as a
# second cue; direction y is dashed.
SERIES  = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
INK, MUTED, GRID = '#52514e', '#898781', '#e1e0d9'


# ============================================================
#  SHARED
# ============================================================

class PairReader:
    """Two in-memory frames, indexed like dic.image_folder_loader."""
    def __init__(self, a, b):
        self.frames = [a, b]

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return self.frames[i]


def correlate(reader, end_index, POIs, frames, L):
    """bisection_search with the selected settings; returns the result graph."""
    rg, _ = dic.bisection_search(reader, 0, end_index, np.copy(POIs), frames,
                                 subset_level=(L - 1) // 2, ZNSSD_limit=ARGS.znssd,
                                 subpixel_method=SUBPIXEL_METHOD, lost_level=LOST_LEVEL)
    return rg


def frame_fields(rg, k, POIs):
    """(u, v, exx, exy, eyy, eeq) at frame k; NaN where the POI was lost."""
    node = rg.nodes[k]
    disp = node['POIs'] - POIs
    gs   = node.get('green_strain', np.full((len(POIs), 4), np.nan))
    return (disp[:, 0].astype(float), disp[:, 1].astype(float),
            gs[:, 0].astype(float), gs[:, 1].astype(float), gs[:, 2].astype(float), gs[:, 3].astype(float))


def write_csv(rows, path):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'Saved: {path}')


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'Saved: {path}')


def style(ax):
    ax.grid(color=GRID, lw=0.6, which='major')
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(labelsize=8, labelcolor=INK, color=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def series_style(L, direction='x'):
    i = ARGS.subsets.index(L)
    return dict(color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)], ms=5, lw=1.5,
                ls='--' if direction == 'y' else '-')


# ============================================================
#  THE IMAGE THAT GETS DEFORMED
# ============================================================

class Photo:
    """A photo, deformed by spline warping."""
    def __init__(self, img):
        self.ref = img
        self.H, self.W = img.shape
        self.full_scale = 65535 if img.max() > 255 else 255
        self.yy, self.xx = np.mgrid[0:self.H, 0:self.W].astype(np.float64)
        self.gain = (230 / 255 * self.full_scale) / img.max()

    def _warp(self, src_y, src_x):
        return map_coordinates(self.ref, [src_y, src_x], order=WARP_SPLINE_ORDER, mode='reflect')

    def shifted(self, direction, s):
        if s == 0:
            return self.ref
        return self._warp(self.yy, self.xx - s) if direction == 'x' else self._warp(self.yy - s, self.xx)

    def strained(self, m):
        return self._warp(self.yy, (self.xx + m * self.W / 2.0) / (1.0 + m))

    def sinusoid(self, wavelength, amplitude):
        # Material point X moves to x = X + A sin(kX): solve for X at each pixel x.
        k = 2 * np.pi / wavelength
        X = self.xx.copy()
        for _ in range(30):
            X = self.xx - amplitude * np.sin(k * X)
        return self._warp(self.yy, X)

    def noisy(self, img, rng):
        """Frame as the camera would deliver it: anchored grey scale + fresh noise."""
        sigma = ARGS.noise_std / 255 * self.full_scale
        out = img * self.gain + (rng.normal(0, sigma, img.shape) if sigma > 0 else 0)
        dtype = np.uint16 if self.full_scale == 65535 else np.uint8
        return np.clip(np.round(out), 0, self.full_scale).astype(dtype)


class SyntheticSpeckle(Photo):
    """Analytic speckle: shifts and strains are re-rendered, not interpolated."""
    def __init__(self):
        H, W = IMG_SIZE
        self.centers, self.amps = make_speckle_dots(H, W, N_DOTS, SEED)
        super().__init__(render_speckle(H, W, self.centers, self.amps, DOT_SIGMA))
        self.full_scale = 255
        self.gain = 230.0 / self.ref.max()

    def shifted(self, direction, s):
        tx, ty = (s, 0.0) if direction == 'x' else (0.0, s)
        return render_speckle(self.H, self.W, deform_translation(self.centers, tx, ty), self.amps, DOT_SIGMA)

    def strained(self, m):
        return render_speckle(self.H, self.W, deform_uniform_strain_x(self.centers, m, self.W / 2.0),
                              self.amps, sigma_x=(1.0 + m) * DOT_SIGMA, sigma_y=DOT_SIGMA)


def ask_path(kind, title):
    """File or folder dialog; exits if cancelled."""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    if kind == 'file':
        exts = ' '.join(f'*{e}' for e in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', *sorted(RAW_EXTENSIONS)])
        path = filedialog.askopenfilename(title=title, filetypes=[('Images', exts), ('All files', '*.*')])
    else:
        path = filedialog.askdirectory(title=title)
    root.destroy()
    if not path:
        sys.exit('Nothing selected.')
    return path


def load_gray(path):
    if os.path.splitext(path)[1].lower() in RAW_EXTENSIONS:
        return load_raw(path).astype(np.float64)
    img = np.asarray(imageio.imread(path))
    if img.ndim == 3:
        img = img[..., :3].astype(np.float64) @ [0.299, 0.587, 0.114]
    return img.astype(np.float64)


def load_source(path, out):
    if path == 'synthetic':
        return SyntheticSpeckle(), None
    if not os.path.isfile(path):
        sys.exit(f'Image not found: {path!r}')
    img = load_gray(path)
    print(f'Loaded {path!r} ({img.shape[1]}x{img.shape[0]} px)')

    crop_json = os.path.join(out, 'crop.json')
    if ARGS.no_crop or (CROP is None and ARGS.crop is None):
        box = None
    elif ARGS.crop is not None:
        box = ARGS.crop
    elif CROP != 'select':
        box = list(CROP)
    elif os.path.isfile(crop_json):
        with open(crop_json) as f:
            box = json.load(f)['crop_box']
        print(f'Using saved crop {box}; delete {crop_json!r} to select again.')
    else:
        print('Draw the speckle-covered region, then close the window ...')
        box = [int(v) for v in select_crop(img)]
        with open(crop_json, 'w') as f:
            json.dump({'crop_box': box}, f, indent=2)
    if box is not None:
        x1, y1, x2, y2 = box
        img = img[y1:y2, x1:x2]
        print(f'Using region x=[{x1}:{x2}] y=[{y1}:{y2}] ({img.shape[1]}x{img.shape[0]} px)')
    return Photo(img), box


# ============================================================
#  TRANSLATION, STRAIN, UPPER BOUND
# ============================================================

def build_pois(src, L, extra_x=0, extra_y=0):
    """Regular POI grid whose subsets stay inside the image, also after a shift
    of up to extra_x / extra_y. None if the image is too small for that."""
    margin = max(POI_MARGIN, (L - 1) // 2 + 20)
    xs = np.arange(margin + extra_x, src.W - margin - extra_x + 1, ARGS.step)
    ys = np.arange(margin + extra_y, src.H - margin - extra_y + 1, ARGS.step)
    if len(xs) == 0 or len(ys) == 0:
        return None
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()]).astype(dic.floattype)


def measure_pairs(test, src, L, direction, values, POIs, deformed, truth, statistic):
    """One isolated frame pair per imposed value; returns a row per value."""
    x_poi = POIs[:, 0].astype(float)
    # Same seed per test and direction, so every subset sees the same noisy frames.
    rng = np.random.default_rng([SEED, ALL_TESTS.index(test), 0 if direction == 'x' else 1])
    rows = []
    for m in values:
        t0 = time.time()
        a, b = src.noisy(src.ref, rng), src.noisy(deformed(m), rng)
        u, v, *_ = frame_fields(correlate(PairReader(a, b), 1, POIs, None, L), 1, POIs)
        along, across = (v, u) if direction == 'y' else (u, v)
        ok = ~np.isnan(along)
        err = along - truth(m, x_poi)
        stat = statistic(x_poi, along, ok)
        rows.append({
            'test': test, 'subset_px': L, 'direction': direction, 'imposed': float(m),
            'n_poi': len(POIs), 'valid_frac': float(ok.mean()), 'stat': stat,
            'std_along': float(np.nanstd(along)), 'rmse_along': float(np.sqrt(np.nanmean(err ** 2))),
            'mean_across': float(np.nanmean(across)), 'std_across': float(np.nanstd(across)),
            'mismatch_frac': float(np.mean(np.abs(err[ok]) > MISMATCH_PX)) if ok.any() else float('nan'),
            'rel_err': float(abs(stat - m) / m) if m > 0 else float('nan'),
        })
        print(f'  [{test}] L={L} {direction} imposed={m:g}  stat={stat:+.3e}  '
              f'valid {ok.mean():.3f}  {time.time() - t0:.1f}s')
    return rows


def mean_stat(x, along, ok):
    return float(np.nanmean(along))


def slope_stat(x, along, ok):
    return float(np.polyfit(x[ok], along[ok], 1)[0]) if ok.sum() >= 2 else float('nan')


def detection_limit(rows):
    """Noise floors from the zero pairs + the smallest reliably detected value.
    Adds snr and detected to the ladder rows."""
    zero = [r for r in rows if r['imposed'] == 0]
    ladder = sorted((r for r in rows if r['imposed'] > 0), key=lambda r: r['imposed'])
    sigma0_stat = float(np.std([r['stat'] for r in zero]))
    for r in ladder:
        r['snr'] = abs(r['stat']) / sigma0_stat if sigma0_stat > 0 else float('inf')
        r['detected'] = bool(r['snr'] > DETECT_SNR and r['rel_err'] < DETECT_REL_TOL)
    limit = next((r['imposed'] for i, r in enumerate(ladder)
                  if all(q['detected'] for q in ladder[i:])), None)
    return {'sigma0_poi': float(np.sqrt(np.mean([r['std_along'] ** 2 for r in zero]))),
            'sigma0_stat': sigma0_stat,
            'bias0_stat': float(np.mean([r['stat'] for r in zero])),
            'detection_limit': limit}


def plot_detection(rows, summaries, unit, title, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, s in summaries.items():
        L, direction = s['subset_px'], s['direction']
        ladder = sorted((r for r in rows if r['imposed'] > 0 and r['subset_px'] == L and r['direction'] == direction),
                        key=lambda r: r['imposed'])
        x = [r['imposed'] for r in ladder]
        label = f'{L} px, {direction}'
        a1.loglog(x, [abs(r['stat']) for r in ladder], label=label, **series_style(L, direction))
        a2.semilogx(x, [(r['stat'] - r['imposed']) / r['imposed'] * 100 for r in ladder],
                    label=label, **series_style(L, direction))
    ideal = sorted({r['imposed'] for r in rows if r['imposed'] > 0})
    a1.loglog(ideal, ideal, color=INK, lw=1, ls=':', label='ideal (y = x)')
    a1.set(xlabel=f'Imposed ({unit})', ylabel=f'|Measured| ({unit})', title=f'{title}: measured vs imposed')
    for tol in (DETECT_REL_TOL, -DETECT_REL_TOL):
        a2.axhline(tol * 100, color=MUTED, lw=1, ls=':')
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylim(-100, 100)
    a2.set(xlabel=f'Imposed ({unit})', ylabel='Relative error of the statistic (%)',
           title=f'Error (dotted = ±{DETECT_REL_TOL:.0%} tolerance, axis cut at ±100%)')
    a1.legend(fontsize=8, frameon=False)
    style(a1), style(a2)
    save_fig(fig, path)


def run_translation(src, out):
    rows, summaries = [], {}
    for L in ARGS.subsets:
        POIs = build_pois(src, L)
        if POIs is None:
            print(f'  L={L}: image too small for this subset + margin; skipped.')
            continue
        for d in ARGS.directions:
            rr = measure_pairs('translation', src, L, d, [0.0] * N_ZERO_PAIRS + SMALL_SHIFTS_PX, POIs,
                               lambda m, d=d: src.shifted(d, m), lambda m, x: m, mean_stat)
            summaries[f'{d}_s{L}'] = {'subset_px': L, 'direction': d, 'n_poi': len(POIs),
                                      'unit': 'px', **detection_limit(rr)}
            rows += rr
    if rows:
        write_csv(rows, os.path.join(out, 'translation.csv'))
        plot_detection(rows, summaries, 'px', 'Shift', os.path.join(out, 'translation.png'))
    return summaries


def run_strain(src, out):
    rows, summaries = [], {}
    for L in ARGS.subsets:
        POIs = build_pois(src, L)
        if POIs is None:
            print(f'  L={L}: image too small for this subset + margin; skipped.')
            continue
        rr = measure_pairs('strain', src, L, 'x', [0.0] * N_ZERO_PAIRS + STRAINS, POIs,
                           src.strained, lambda m, x: m * (x - src.W / 2.0), slope_stat)
        summaries[f's{L}'] = {'subset_px': L, 'direction': 'x', 'n_poi': len(POIs),
                              'unit': 'strain', **detection_limit(rr)}
        rows += rr
    if rows:
        write_csv(rows, os.path.join(out, 'strain.csv'))
        plot_detection(rows, summaries, 'strain', 'Uniform strain', os.path.join(out, 'strain.png'))
    return summaries


def run_upper_bound(src, out):
    rows, summaries = [], {}
    extra = int(np.ceil(max(LARGE_SHIFTS_PX)))
    for L in ARGS.subsets:
        for d in ARGS.directions:
            POIs = build_pois(src, L, extra_x=extra if d == 'x' else 0, extra_y=extra if d == 'y' else 0)
            if POIs is None:
                print(f'  L={L} {d}: image too small to fit a {extra} px shift; skipped. '
                      'Use a larger crop or lower LARGE_SHIFTS_PX.')
                continue
            rr = measure_pairs('upper_bound', src, L, d, LARGE_SHIFTS_PX, POIs,
                               lambda m, d=d: src.shifted(d, m), lambda m, x: m, mean_stat)
            limit = None
            for r in sorted(rr, key=lambda r: r['imposed']):
                r['reliable'] = bool(r['rel_err'] < UPPER_REL_TOL and r['valid_frac'] >= UPPER_MIN_VALID)
                if not r['reliable']:
                    break
                limit = r['imposed']
            summaries[f'{d}_s{L}'] = {'subset_px': L, 'direction': d, 'n_poi': len(POIs),
                                      'upper_bound_px': limit, 'tested_up_to_px': max(LARGE_SHIFTS_PX),
                                      'max_mismatch_frac': float(np.nanmax([r['mismatch_frac'] for r in rr]))}
            rows += rr
    if not rows:
        return summaries
    write_csv(rows, os.path.join(out, 'upper_bound.csv'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for s in summaries.values():
        L, d = s['subset_px'], s['direction']
        rr = sorted((r for r in rows if r['subset_px'] == L and r['direction'] == d), key=lambda r: r['imposed'])
        x = [r['imposed'] for r in rr]
        a1.semilogy(x, [max(r['rel_err'] * 100, 1e-3) for r in rr], label=f'{L} px, {d}', **series_style(L, d))
        a2.plot(x, [r['valid_frac'] * 100 for r in rr], label=f'{L} px, {d}', **series_style(L, d))
    a1.axhline(UPPER_REL_TOL * 100, color=MUTED, lw=1, ls=':')
    a2.axhline(UPPER_MIN_VALID * 100, color=MUTED, lw=1, ls=':')
    a1.set(xlabel='Imposed shift (px)', ylabel='|Relative error of the mean| (%)',
           title=f'Accuracy (dotted = {UPPER_REL_TOL:.0%} limit)')
    a2.set(xlabel='Imposed shift (px)', ylabel='Points tracked (%)',
           title=f'Tracking (dotted = {UPPER_MIN_VALID:.0%} limit)')
    a2.set_ylim(0, 102)
    a1.legend(fontsize=8, frameon=False)
    style(a1), style(a2)
    save_fig(fig, os.path.join(out, 'upper_bound.png'))
    return summaries


# ============================================================
#  SPATIAL RESOLUTION
# ============================================================

def theory_u(z):
    return np.sinc(z / np.pi)                        # sin z / z


def theory_e(z):
    z = np.asarray(z, float)
    return 3 * (np.sin(z) - z * np.cos(z)) / z ** 3


def lambda_at(H, L):
    """Wavelength where theory transfer H(pi L / lambda) equals TRANSFER_LEVEL."""
    lam = np.linspace(L * 0.6, L * 20, 20000)
    return float(lam[np.argmax(H(np.pi * L / lam) >= TRANSFER_LEVEL)])


def run_resolution(src, out):
    rows, summaries = [], {}
    for L in ARGS.subsets:
        level = (L - 1) // 2
        margin = max(POI_MARGIN, level + 20)
        xs = np.arange(margin, src.W - margin + 1, RES_STEP_X)
        ys = [src.H // 2 + o for o in RES_ROW_OFFSETS if level + 5 <= src.H // 2 + o < src.H - level - 5]
        lams = [lam for lam in WAVELENGTHS_PX if len(xs) and lam <= xs[-1] - xs[0]]
        if not ys or not lams:
            print(f'  L={L}: image too small for this subset; skipped.')
            continue
        gx, gy = np.meshgrid(xs, ys)
        POIs = np.column_stack([gx.ravel(), gy.ravel()]).astype(dic.floattype)
        X0 = POIs[:, 0].astype(float)
        rng = np.random.default_rng([SEED, ALL_TESTS.index('resolution')])

        rr = []
        for lam in lams:
            t0 = time.time()
            k, A = 2 * np.pi / lam, EPS0 * lam / (2 * np.pi)
            a, b = src.noisy(src.ref, rng), src.noisy(src.sinusoid(lam, A), rng)
            u, _, exx, *_ = frame_fields(correlate(PairReader(a, b), 1, POIs, None, L), 1, POIs)
            ok = ~np.isnan(u) & ~np.isnan(exx)
            if ok.sum() < 4:
                print(f'  L={L} lambda={lam}: too few points tracked; skipped.')
                continue
            S, C, one = np.sin(k * X0[ok]), np.cos(k * X0[ok]), np.ones(ok.sum())
            cu, *_ = np.linalg.lstsq(np.column_stack([S, C, one]), u[ok], rcond=None)
            ce, *_ = np.linalg.lstsq(np.column_stack([C, S, one]), exx[ok], rcond=None)
            z = np.pi * L / lam
            rr.append({'subset_px': L, 'lambda_px': lam, 'lambda_over_L': lam / L, 'A_px': A,
                       'u_transfer': cu[0] / A, 'e_transfer': ce[0] / EPS0,
                       'u_transfer_theory': float(theory_u(z)), 'e_transfer_theory': float(theory_e(z)),
                       'n_valid': int(ok.sum()), 'n_poi': len(POIs)})
            print(f'  [resolution] L={L} lambda={lam}  u ratio {rr[-1]["u_transfer"]:+.3f} '
                  f'(theory {rr[-1]["u_transfer_theory"]:+.3f})  strain ratio {rr[-1]["e_transfer"]:+.3f} '
                  f'(theory {rr[-1]["e_transfer_theory"]:+.3f})  {time.time() - t0:.1f}s')

        def first_ok(key):
            # shortest wavelength from which every longer one is resolved
            for r in rr:
                if all(q[key] >= TRANSFER_LEVEL for q in rr if q['lambda_px'] >= r['lambda_px']):
                    return r['lambda_px']
            return None
        summaries[f's{L}'] = {'disp_resolution_px': first_ok('u_transfer'),
                              'disp_resolution_px_theory': lambda_at(theory_u, L),
                              'strain_resolution_px': first_ok('e_transfer'),
                              'strain_resolution_px_theory': lambda_at(theory_e, L)}
        rows += rr
    if not rows:
        return summaries
    write_csv(rows, os.path.join(out, 'resolution.csv'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    lam_t = np.geomspace(min(r['lambda_px'] for r in rows) * 0.8, max(r['lambda_px'] for r in rows) * 1.2, 400)
    for ax, key, H_, title in ((axes[0], 'u_transfer', theory_u, 'Displacement'),
                               (axes[1], 'e_transfer', theory_e, 'Strain')):
        for L in sorted({r['subset_px'] for r in rows}):
            rr = [r for r in rows if r['subset_px'] == L]
            st = series_style(L)
            ax.plot(lam_t, H_(np.pi * L / lam_t), color=st['color'], lw=1, alpha=0.6)
            ax.plot([r['lambda_px'] for r in rr], [r[key] for r in rr], ls='none', marker=st['marker'],
                    ms=st['ms'], color=st['color'], label=f'{L} px subset')
        ax.axhline(TRANSFER_LEVEL, color=MUTED, ls=':', lw=1)
        ax.axhline(1, color=MUTED, lw=0.6)
        ax.set_xscale('log')
        ax.set_ylim(-0.3, 1.1)
        ax.set(xlabel='Wavelength of the imposed field (px)', title=f'{title}: markers = measured, lines = theory')
        style(ax)
    axes[0].set_ylabel('Measured / imposed amplitude')
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    save_fig(fig, os.path.join(out, 'resolution.png'))
    return summaries


# ============================================================
#  NOISE FLOOR OF THE REAL SETUP
# ============================================================

def run_noise_floor(folder, out):
    prep = os.path.join(out, 'noise_floor_prep')
    os.makedirs(prep, exist_ok=True)
    scale_json = os.path.join(prep, 'scale.json')
    if not os.path.isfile(scale_json):      # noise is reported in px: no calibration needed
        with open(scale_json, 'w') as f:
            json.dump({'pixel_distance': None, 'mm_per_px': None}, f)

    cfg = DatasetConfig(STEP=ARGS.step, N_ZERO_PRESSURE_IMAGES=0, BIT_DEPTH=BIT_DEPTH,
                        FLATTEN_SIGMA=FLATTEN_SIGMA, REFLECTION_CLIP_PCT=REFLECTION_CLIP_PCT,
                        LENS_PARAMS=LENS_PARAMS)
    st = run_dataset(folder, prep, cfg, dic, interactive=True, do_dic=False, return_data=True)
    POIs, frames = st['POIs'], st['output_frame_list']
    if len(frames) < 2:
        print('  noise_floor needs at least 3 photos; skipped.')
        return {}

    rows = []
    for L in ARGS.subsets:
        t0 = time.time()
        rg = correlate(st['reader'], st['end_index'], POIs, frames, L)
        f = [frame_fields(rg, k, POIs) for k in frames]
        u, v = np.array([x[0] for x in f]), np.array([x[1] for x in f])
        exx, eyy = np.array([x[2] for x in f]), np.array([x[4] for x in f])
        rows.append({
            'subset_px': L, 'n_poi': len(POIs), 'n_frames': len(frames),
            'valid_frac': float(np.mean(~np.isnan(u))),
            'sigma0_poi_u': float(np.sqrt(np.nanmean(np.nanvar(u, axis=1)))),
            'sigma0_poi_v': float(np.sqrt(np.nanmean(np.nanvar(v, axis=1)))),
            'sigma0_stat_u': float(np.std(np.nanmean(u, axis=1))),
            'sigma0_stat_v': float(np.std(np.nanmean(v, axis=1))),
            'bias0_stat_u': float(np.mean(np.nanmean(u, axis=1))),
            'bias0_stat_v': float(np.mean(np.nanmean(v, axis=1))),
            'strain_noise_exx': float(np.sqrt(np.nanmean(np.nanvar(exx, axis=1)))),
            'strain_noise_eyy': float(np.sqrt(np.nanmean(np.nanvar(eyy, axis=1)))),
        })
        r = rows[-1]
        print(f'  [noise_floor] L={L}  per-point u {r["sigma0_poi_u"]:.4f} v {r["sigma0_poi_v"]:.4f} px  '
              f'mean u {r["sigma0_stat_u"]:.5f} v {r["sigma0_stat_v"]:.5f} px  '
              f'valid {r["valid_frac"]:.3f}  {time.time() - t0:.0f}s')
    write_csv(rows, os.path.join(out, 'noise_floor.csv'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    Ls = [r['subset_px'] for r in rows]
    for ax, series, ylabel in ((a1, [('sigma0_poi_u', 'x (u)'), ('sigma0_poi_v', 'y (v)')], 'Per-point displacement noise (px)'),
                               (a2, [('strain_noise_exx', 'exx'), ('strain_noise_eyy', 'eyy')], 'Strain noise (-)')):
        for i, (key, label) in enumerate(series):
            ax.plot(Ls, [r[key] for r in rows], color=SERIES[i], marker=MARKERS[i], ms=5, lw=1.5, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(Ls, [str(L) for L in Ls])
        ax.minorticks_off()
        ax.set(xlabel='Subset size (px)', ylabel=ylabel)
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    fig.suptitle('Repeat photos of an undeformed sample', fontsize=10, color=INK)
    save_fig(fig, os.path.join(out, 'noise_floor.png'))
    return {f's{r["subset_px"]}': r for r in rows}


# ============================================================
#  CONVERGENCE ON A FINISHED RESULT
# ============================================================

def poly_distance(P, vertices):
    """Distance from each point to the nearest edge of a closed polygon."""
    V = np.asarray(vertices, float)
    P = np.asarray(P, float)
    d = np.full(len(P), np.inf)
    for i in range(len(V)):
        A, B = V[i], V[(i + 1) % len(V)]
        AB = B - A
        t = np.clip(((P - A) @ AB) / max(AB @ AB, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(P - (A + t[:, None] * AB), axis=1))
    return d


def run_convergence(folder, out):
    if not os.path.isdir(os.path.join(folder, 'converted')):
        print(f'  {folder!r} has no converted/ frames; skipped.')
        return {}
    scratch = tempfile.mkdtemp(prefix='dic_convergence_')
    try:
        for fn in ['scale.json', 'poi_polygon.json', 'exclusion.json', 'grid_axis.json']:
            if os.path.isfile(os.path.join(folder, fn)):
                shutil.copy2(os.path.join(folder, fn), os.path.join(scratch, fn))
        cfg = DatasetConfig(STEP=ARGS.step, N_ZERO_PRESSURE_IMAGES=0)
        st = run_dataset(os.path.join(folder, 'converted'), scratch, cfg, dic,
                         interactive=False, do_dic=False, return_data=True)
        POIs, frames, end = st['POIs'], st['output_frame_list'], st['end_index']

        d_edge = poly_distance(POIs, st['vertices'])
        d_excl = (poly_distance(POIs, st['exclusion_vertices']) if len(st['exclusion_vertices']) >= 3
                  else np.full(len(POIs), np.inf))
        region = np.where(d_excl <= EXCLUSION_BAND_PX, 'near_exclusion',
                          np.where(d_edge < EDGE_BAND_PX, 'edge', 'interior'))

        fields = {}
        for L in ARGS.subsets:
            t0 = time.time()
            rg = correlate(st['reader'], end, POIs, frames, L)
            disp = np.full((end + 1, len(POIs), 2), np.nan)
            disp[0] = 0
            eeq = np.full((end + 1, len(POIs)), np.nan)
            for k in frames:
                u, v, _, _, _, e = frame_fields(rg, k, POIs)
                disp[k] = np.column_stack([u, v])
                eeq[k] = e
            fields[L] = (disp, eeq)
            print(f'  [convergence] L={L} done in {time.time() - t0:.0f}s')
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # A rerun with the original settings must reproduce the saved result.
    saved_disp = None
    saved_npz = os.path.join(folder, 'deformation_grids.npz')
    if os.path.isfile(saved_npz):
        saved_disp = np.load(saved_npz)['displacements']

    last, ref_L = int(frames[-1]), max(ARGS.subsets)
    interior = region == 'interior'
    rows = []
    for L in ARGS.subsets:
        disp, eeq = fields[L]
        du = disp[last] - fields[ref_L][0][last]
        row = {'subset_px': L, 'frame': last, 'n_poi': len(POIs), 'n_interior': int(interior.sum()),
               'valid_frac_last': float(np.mean(~np.isnan(disp[last, :, 0]))),
               f'rms_du_vs_{ref_L}_interior_px': float(np.sqrt(np.nanmean(du[interior, 0] ** 2))),
               f'rms_dv_vs_{ref_L}_interior_px': float(np.sqrt(np.nanmean(du[interior, 1] ** 2))),
               'eeq_median_interior': float(np.nanmedian(eeq[last, interior])),
               'eeq_p95_interior': float(np.nanpercentile(eeq[last, interior], 95)),
               'max_diff_vs_saved_px': (float(np.nanmax(np.abs(saved_disp - disp)))
                                        if saved_disp is not None and saved_disp.shape == disp.shape else None)}
        rows.append(row)
        print(f'    L={L}: valid {row["valid_frac_last"]:.3f}, interior rms vs {ref_L} px: '
              f'u {row[f"rms_du_vs_{ref_L}_interior_px"]:.4f}, v {row[f"rms_dv_vs_{ref_L}_interior_px"]:.4f} px, '
              f'max diff vs saved result: {row["max_diff_vs_saved_px"]}')
    write_csv(rows, os.path.join(out, 'convergence.csv'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for L in ARGS.subsets:
        disp, eeq = fields[L]
        du = disp[frames] - fields[ref_L][0][frames]
        a1.plot(frames, np.sqrt(np.nanmean(du[:, interior] ** 2, axis=(1, 2))), label=f'{L} px', **series_style(L))
        a2.plot(frames, np.nanmedian(eeq[frames][:, interior], axis=1), label=f'{L} px', **series_style(L))
    a1.set(xlabel='Frame', ylabel=f'RMS displacement difference to {ref_L} px (px)', title='Interior points: displacement')
    a2.set(xlabel='Frame', ylabel='Median equivalent strain (-)', title='Interior points: strain')
    a1.legend(fontsize=8, frameon=False)
    style(a1), style(a2)
    save_fig(fig, os.path.join(out, 'convergence.png'))
    return {f's{r["subset_px"]}': r for r in rows}


# ============================================================
#  LARGEST FRAME-TO-FRAME SHIFT
# ============================================================

def run_frame_step(folder, out):
    paths = (glob.glob(os.path.join(folder, 'deformation_grids.npz'))
             or sorted(glob.glob(os.path.join(folder, '*', 'deformation_grids.npz'))))
    if not paths:
        print(f'  No deformation_grids.npz in or directly under {folder!r}; skipped.')
        return {}
    rows = []
    for path in paths:
        disp = np.load(path)['displacements']          # (F+1, N, 2), px
        inc = np.diff(disp, axis=0)                    # frame-to-frame increment
        step = np.hypot(inc[..., 0], inc[..., 1])      # (F, N)
        p99 = np.nanpercentile(step, 99, axis=1)
        rows.append({'dataset': os.path.basename(os.path.dirname(path)), 'n_frames': disp.shape[0] - 1,
                     'n_poi': disp.shape[1], 'max_step_px': float(np.nanmax(step)),
                     'p99_step_px': float(p99.max()),
                     'median_step_px': float(np.nanmedian(step, axis=1).max()),
                     'at_frame': int(np.nanargmax(p99)) + 1,
                     'lost_pct_last': float(np.mean(np.isnan(disp[-1, :, 0])) * 100)})
        r = rows[-1]
        print(f'  [frame_step] {r["dataset"]}: max {r["max_step_px"]:.1f} px, p99 {r["p99_step_px"]:.1f} px, '
              f'median {r["median_step_px"]:.1f} px')
    write_csv(rows, os.path.join(out, 'frame_step.csv'))
    worst = max(rows, key=lambda r: r['max_step_px'])
    return {'max_step_px': worst['max_step_px'], 'max_step_dataset': worst['dataset'],
            'largest_p99_step_px': max(r['p99_step_px'] for r in rows)}


# ============================================================
#  MAIN
# ============================================================

def fmt(v, unit=''):
    return 'not reached' if v is None else f'{v:g}{unit}'


def print_summary(results):
    print('\n' + '=' * 72 + '\nVALIDATION SUMMARY\n' + '=' * 72)
    for key, s in results.get('translation', {}).items():
        print(f"  translation {key:<8} per-point noise {s['sigma0_poi']:.2e} px, "
              f"floor of the mean {s['sigma0_stat']:.2e} px, "
              f"smallest detected shift {fmt(s['detection_limit'], ' px')}")
    for key, s in results.get('strain', {}).items():
        print(f"  strain      {key:<8} floor {s['sigma0_stat']:.2e}, "
              f"smallest detected strain {fmt(s['detection_limit'])}")
    for key, s in results.get('upper_bound', {}).items():
        print(f"  upper bound {key:<8} {fmt(s['upper_bound_px'], ' px')} "
              f"(tested up to {s['tested_up_to_px']:g} px)")
    for key, s in results.get('resolution', {}).items():
        print(f"  resolution  {key:<8} displacement {fmt(s['disp_resolution_px'], ' px')} "
              f"(theory {s['disp_resolution_px_theory']:.0f}), "
              f"strain {fmt(s['strain_resolution_px'], ' px')} (theory {s['strain_resolution_px_theory']:.0f})")
    for key, s in results.get('noise_floor', {}).items():
        print(f"  noise floor {key:<8} per-point u {s['sigma0_poi_u']:.2e}, v {s['sigma0_poi_v']:.2e} px")
    if results.get('frame_step'):
        s = results['frame_step']
        print(f"  frame step           largest {s['max_step_px']:.1f} px ({s['max_step_dataset']})")


def main():
    tests = ARGS.tests
    folders_only = ARGS.image is None and (ARGS.noise_folder or ARGS.result_folder)
    if tests is None:
        tests = [] if folders_only else list(IMAGE_TESTS)
        if ARGS.noise_folder:
            tests.append('noise_floor')
        if ARGS.result_folder:
            tests += ['convergence', 'frame_step']

    image = ARGS.image
    if any(t in IMAGE_TESTS for t in tests) and image is None:
        image = ask_path('file', 'Select an image of the speckle pattern')
    noise_folder = ARGS.noise_folder
    if 'noise_floor' in tests and noise_folder is None:
        noise_folder = ask_path('folder', 'Select the folder of repeat photos of an undeformed sample')
    result_folder = ARGS.result_folder
    if {'convergence', 'frame_step'} & set(tests) and result_folder is None:
        result_folder = ask_path('folder', 'Select a finished DIC result folder')

    named = image or noise_folder or result_folder
    name = os.path.splitext(os.path.basename(os.path.normpath(named)))[0]
    out = ARGS.output or os.path.join(_ROOT, 'validation_results', name)
    os.makedirs(out, exist_ok=True)

    src, box = None, None
    if any(t in IMAGE_TESTS for t in tests):
        src, box = load_source(image, out)

    print(f'Tests: {tests}\nSubsets: {ARGS.subsets} px, step {ARGS.step} px, ZNSSD limit {ARGS.znssd}')
    runners = {
        'translation': lambda: run_translation(src, out),
        'strain':      lambda: run_strain(src, out),
        'upper_bound': lambda: run_upper_bound(src, out),
        'resolution':  lambda: run_resolution(src, out),
        'noise_floor': lambda: run_noise_floor(noise_folder, out),
        'convergence': lambda: run_convergence(result_folder, out),
        'frame_step':  lambda: run_frame_step(result_folder, out),
    }
    results = {}
    for test in tests:
        print(f'\n===== {test} =====')
        results[test] = runners[test]()

    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump({'config': {'image': image, 'crop_box': box, 'noise_folder': noise_folder,
                              'result_folder': result_folder, 'tests': tests,
                              'subset_sizes': ARGS.subsets, 'step': ARGS.step,
                              'znssd_limit': ARGS.znssd, 'lost_level': LOST_LEVEL,
                              'subpixel_method': SUBPIXEL_METHOD, 'noise_std': ARGS.noise_std,
                              'seed': SEED, 'detect_snr': DETECT_SNR, 'detect_rel_tol': DETECT_REL_TOL,
                              'upper_rel_tol': UPPER_REL_TOL, 'upper_min_valid': UPPER_MIN_VALID},
                   'results': results}, f, indent=2)
    print_summary(results)
    print(f'\nAll outputs -> {out}')


if __name__ == '__main__':
    main()
