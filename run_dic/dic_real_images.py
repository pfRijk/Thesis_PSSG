"""
DIC on real photographs
=======================
Converts a folder of RAW photos (.ARW, .CR2, .NEF, .DNG, ...) to leveled,
cropped grayscale TIFFs, then runs the DIC correlation on them and saves all
outputs.

This is a thin driver over dic_workflow/dataset_runner.py's run_dataset(),
which asks for any manual selection that is not saved yet.

Pipeline (see run_dataset() for the details):

Step 1 — scale:      px -> mm calibration (optional), run on the raw
                     reference frame before cropping so an ArUco marker
                     outside the crop region is still detected: auto-detected
                     ArUco marker or interactive 2-point click
                     (SCALE_CALIBRATION_METHOD)
Step 2 — level:      interactive 2-point rotate-to-horizontal (optional)
Step 3 — crop:       interactive rectangle crop on the (leveled) reference frame
Step 4 — conversion: every RAW -> lens-corrected, leveled, cropped, flattened
                     grayscale TIFF, normalised to BIT_DEPTH using the
                     reference frame's intensity range
Step 5 — alignment:  marker-based frame alignment (optional, off by default)
Step 6 — ROI:        interactive POI polygon (+ edge refinement and inward
                     shrink), exclusion polygon, and POI-grid axis
Step 7 — DIC:        bisection_search over all frames
Step 8 — outputs:    per-frame plots, strain/displacement maps, grids, GIFs

If INPUT_FOLDER already contains .tif frames, steps 2-5 are skipped and the
TIFFs are used as-is.

Every manual selection is saved to JSON in RESULT_FOLDER (scale.json,
level.json, crop.json, align_markers.json, poi_polygon.json, exclusion.json,
grid_axis.json), so re-runs skip the interactive steps once they exist.
Delete the relevant .json to re-select; level and crop are only re-asked if
crop.json or the converted/ TIFFs are missing.

Usage:
    python run_dic/dic_real_images.py [input_folder] [result_folder]

input_folder defaults to INPUT_FOLDER below, result_folder to
results/<name of input_folder>_results.

Output:
    <RESULT_FOLDER>/
        converted/                  <- leveled, cropped grayscale TIFFs (frame_000.tif ...)
        scale.json                  <- saved px -> mm calibration
        level.json                  <- saved level angle
        crop.json                   <- saved crop box
        align_markers.json          <- saved alignment markers (only if alignment is enabled)
        poi_polygon.json            <- saved ROI polygon (post edge-refine/shrink)
        exclusion.json              <- saved exclusion polygon
        grid_axis.json              <- saved POI-grid axis angle
        edge_refinement.png         <- drawn vs edge-refined ROI polygon (if REFINE_ROI_WITH_EDGES)
        input_frames.gif
        frame_00_POIs.png
        frame_01.png ... frame_NN.png
        strain.png
        displacement.png            <- displacement relative to frame 0 (mm if calibrated)
        grid_deformation_panels.png
        grid_deformation.gif
        strain_grid_panels.png
        strain_grid.gif
        deformation_grids.npz       <- gridded results, read by the analysis/ scripts
        overlay.gif                 <- only if SAVE_OVERLAY_GIF
"""

import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

# ============================================================
#  PARAMETERS  --  edit here, nowhere else
# ============================================================

# ── Dataset ──────────────────────────────────────────────────
# Folder of RAW images or .tif frames, or pass it on the command line.
INPUT_FOLDER  = r'path/to/your/images'
if len(sys.argv) > 1:
    INPUT_FOLDER = sys.argv[1]
if not os.path.isdir(INPUT_FOLDER):
    sys.exit(f'Input folder not found: {INPUT_FOLDER!r}\n'
             'Set INPUT_FOLDER or run: python run_dic/dic_real_images.py <input_folder> [result_folder]')

_dataset_name = os.path.basename(os.path.normpath(INPUT_FOLDER))
RESULT_FOLDER = (sys.argv[2] if len(sys.argv) > 2
                 else os.path.join(_ROOT, 'results', _dataset_name + '_results'))

# Range of images to use, as indices into the sorted file list (inclusive).
START_INDEX = 0
END_INDEX   = None   # None = up to the last image

# ── DIC correlation ──────────────────────────────────────────
SUBSET_LEVEL    = 40      # subset half-width [px]; subset = (2*SUBSET_LEVEL + 1)^2 px. 40 for real images
STEP            = 20      # POI grid spacing [px]. 20 for real images
ZNSSD_LIMIT     = 0.15    # ZNSSD threshold passed to bisection_search
LOST_LEVEL      = 0.30    # lost-POI threshold passed to bisection_search (fraction of the frame range)
SUBPIXEL_METHOD = 'fagn'  # 'fagn' | 'icgn'

# ── Force labelling (optional) ───────────────────────────────
# Only used to label the plots/GIFs with the force applied in each frame.
# The first N_ZERO_PRESSURE_IMAGES frames are taken at 0 N (setup check);
# every frame after that adds another PRESSURE_INCREMENT_N.
# Leave PRESSURE_INCREMENT_N = None to disable force labelling.
N_ZERO_PRESSURE_IMAGES = 1     # number of 0 N images taken to verify the setup
PRESSURE_INCREMENT_N   = None  # force increase [N] per image after those

# Explicit per-frame force [N], for ramps that aren't a constant step size.
# Must have exactly one entry per image. Overrides PRESSURE_INCREMENT_N above
# when set; leave as None to use the constant-increment behaviour instead.
PRESSURE_VALUES_N = None

# ── Scale (px -> mm) calibration ─────────────────────────────
#   'click' -> interactive 2-point click on a known real-world distance
#   'aruco' -> auto-detect a printed ArUco marker (falls back to 'click' if not found)
SCALE_CALIBRATION_METHOD = 'aruco'
ARUCO_CALIBRATION_IMAGE  = None  # path to a photo with the marker; None = use the raw reference frame (before crop)
ARUCO_MARKER_ID          = 0     # 0 = 10 mm marker, 1 = 20 mm marker (see dic_workflow/aruco_calibration.py)

# If full-frame auto-detection misses the marker, you're prompted to click its
# approximate location; detection is retried on a cropped, upscaled window
# around that click (much more reliable for a small marker in a big frame).
ARUCO_HINT_RADIUS  = 250   # px half-width of the search window around the click (250 -> a 500x500 px box centred on it)
ARUCO_HINT_UPSCALE = 3.0   # upscale factor applied to the cropped window before re-detecting
# If the marker still isn't found in that window, it's retried brightened by
# each of these gamma steps in turn (likely cause of a miss: too dark). If
# every step fails, calibration falls back to the manual 2-point click.
# Use () to disable brightness retries.
ARUCO_BRIGHTNESS_STEPS = (1.5, 2.0, 3.0)

# ── Image conversion / preprocessing ─────────────────────────
# Only applied when the TIFFs are (re)converted — delete converted/ after
# changing any of these. Use debug_preprocessing.py to inspect each stage.
RESIZE_TO           = None   # (height, width) to resample each raw frame to; None = full resolution
BIT_DEPTH           = 16     # 8 or 16; bit depth of the converted TIFFs (16 preserves sensor dynamic range)
FLATTEN_SIGMA       = 200    # Gaussian sigma [px] of the background divided out to flatten illumination; 0 = disabled
REFLECTION_CLIP_PCT = 99.9   # intensity percentile above which pixels are clamped (specular highlights); 100 = disabled

# Lens distortion + vignetting correction (lensfun profile).
# These values describe the camera and lens of the original experiments: set
# them to your own, or to None to disable, or your images get the wrong correction.
LENS_PARAMS = dict(
    cam_maker    = 'Sony',
    cam_model    = 'ILCE-7M2',
    lens_maker   = 'Samyang',
    lens_model   = 'Samyang AF 85mm f/1.4',
    focal_length = 85,    # actual mm, not 35 mm equivalent
    aperture     = 8,
    distance     = 1,     # focus distance [m]
)

# ── Frame alignment (optional) ───────────────────────────────
# Registers every frame to the reference using >= 2 clicked fixed markers.
ALIGN_TEMPLATE_HALF = 0    # template half-width [px]; 0 disables alignment — camera setup is now physically fixed
ALIGN_SEARCH_RADIUS = 80   # search radius [px] around each marker

# ── ROI ──────────────────────────────────────────────────────
# Edge-detection ROI refinement (optional).
# After you draw the rough polygon, each edge is snapped to the strongest
# image gradient within EDGE_BAND_WIDTH pixels on both sides.
# Set REFINE_ROI_WITH_EDGES = False to skip and use the drawn polygon as-is.
REFINE_ROI_WITH_EDGES = True
EDGE_BAND_WIDTH       = 30   # px search radius each side of each polygon edge
EDGE_BLUR_SIGMA       = 2    # Gaussian pre-smoothing sigma before gradient

# Shrink the (edge-refined) ROI polygon inward by this many px on every side,
# so subset windows near the boundary don't sample right up to the physical
# edge. Set to 0 to disable.
ROI_SHRINK_PX = 15

# ── Outputs / runtime ────────────────────────────────────────
SAVE_OVERLAY_GIF = True     # also generate overlay.gif
TORCH_DEVICE     = 'auto'   # 'auto' | 'cuda' | 'cpu'

# ============================================================
#  SETUP  --  matplotlib backend must be set before pyplot is imported
# ============================================================

warnings.filterwarnings('ignore', message='CUDA path could not be detected')

import matplotlib
matplotlib.use('TkAgg')   # interactive windows for the click-selection steps

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from dic import dic
from dic_workflow.dataset_runner import DatasetConfig, run_dataset

# The dic package keeps its device and gradient filters as module globals.
if TORCH_DEVICE == 'auto':
    dic.torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    dic.torch_device = torch.device(TORCH_DEVICE)
dic.barron_filter   = dic.barron_filter.to(dic.torch_device)
dic.barron_filter_x = dic.barron_filter_x.to(dic.torch_device)
dic.barron_filter_y = dic.barron_filter_y.to(dic.torch_device)
print(f'Using device: {dic.torch_device}')

# ============================================================
#  RUN
# ============================================================

cfg = DatasetConfig(
    START_INDEX=START_INDEX,
    END_INDEX=END_INDEX,

    SUBSET_LEVEL=SUBSET_LEVEL,
    STEP=STEP,
    ZNSSD_LIMIT=ZNSSD_LIMIT,
    LOST_LEVEL=LOST_LEVEL,
    SUBPIXEL_METHOD=SUBPIXEL_METHOD,

    N_ZERO_PRESSURE_IMAGES=N_ZERO_PRESSURE_IMAGES,
    PRESSURE_INCREMENT_N=PRESSURE_INCREMENT_N,
    PRESSURE_VALUES_N=PRESSURE_VALUES_N,

    SCALE_CALIBRATION_METHOD=SCALE_CALIBRATION_METHOD,
    ARUCO_CALIBRATION_IMAGE=ARUCO_CALIBRATION_IMAGE,
    ARUCO_MARKER_ID=ARUCO_MARKER_ID,
    ARUCO_HINT_RADIUS=ARUCO_HINT_RADIUS,
    ARUCO_HINT_UPSCALE=ARUCO_HINT_UPSCALE,
    ARUCO_BRIGHTNESS_STEPS=ARUCO_BRIGHTNESS_STEPS,

    RESIZE_TO=RESIZE_TO,
    BIT_DEPTH=BIT_DEPTH,
    FLATTEN_SIGMA=FLATTEN_SIGMA,
    REFLECTION_CLIP_PCT=REFLECTION_CLIP_PCT,
    LENS_PARAMS=LENS_PARAMS,

    ALIGN_TEMPLATE_HALF=ALIGN_TEMPLATE_HALF,
    ALIGN_SEARCH_RADIUS=ALIGN_SEARCH_RADIUS,

    REFINE_ROI_WITH_EDGES=REFINE_ROI_WITH_EDGES,
    EDGE_BAND_WIDTH=EDGE_BAND_WIDTH,
    EDGE_BLUR_SIGMA=EDGE_BLUR_SIGMA,
    ROI_SHRINK_PX=ROI_SHRINK_PX,

    SAVE_OVERLAY_GIF=SAVE_OVERLAY_GIF,
)

run_dataset(INPUT_FOLDER, RESULT_FOLDER, cfg, dic,
            interactive=True, do_dic=True)
