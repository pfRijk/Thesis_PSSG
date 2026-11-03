# DIC for samples under stepped loading

A Python pipeline that measures how a speckled sample moves and rotates under load, from a series of photographs. It uses 2D digital image correlation (DIC) and was built for long, thin guide samples photographed from the side.

It does four things:

1. **Preprocesses** RAW photos into flat, level, cropped 16-bit frames.
2. **Runs DIC** on the frames and saves the tracked points.
3. **Measures angles**: the lean of the sample along its length per load step, the force at which it seats, and the tilt of an upright block.
4. **Validates** the measurement on your own image: smallest detectable shift, largest trackable shift, spatial resolution and noise floor, for the subset size you choose.

The correlation engine is [DIC-demo](https://github.com/vincentme/DIC-demo) by Zhang et al.

## Install

Python 3.9 was used for testing.

```
git clone <this repository>
cd <this repository>
pip install -r requirements.txt
```

A GPU is optional. For GPU use, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org). The scripts open interactive matplotlib windows, so they need a display and Tkinter.

## Quick start

```
# 1. Convert the photos and run DIC (opens windows for the manual selections)
python run_dic/dic_real_images.py path/to/images

# 2. Measure the angles from the result
python analysis/measure_vertical_angles_edge_ref.py results/images_results

# 3. Validate on one of your own images, with your own settings
python validation/validate_dic.py --subsets 41 81 --step 20
```

Try the validation without any data of your own:

```
python validation/validate_dic.py --image synthetic --tests translation upper_bound
```

## Before you use your own setup

Some settings describe the original experiments. Check these first:

| Setting | Where | What to do |
|---|---|---|
| `LENS_PARAMS` | `run_dic/dic_real_images.py`, `run_dic/debug_preprocessing.py` | Set your camera and lens, or `None`. **Otherwise your images get the wrong lens correction, with no warning.** |
| `SCALE_CALIBRATION_METHOD`, `ARUCO_MARKER_ID` | `run_dic/dic_real_images.py` | The script looks for a 10 mm (ID 0) or 20 mm (ID 1) ArUco marker, as defined in `dic_workflow/aruco_calibration.py`. Use `'click'` to click a known distance instead. |
| `FORCE_PER_FRAME_CANDIDATES` | `analysis/compute_lay_flat_force.py` | Add your load steps. Without a matching list, plots use frame numbers and no force is reported. |
| `SEATING_STEP_UM` | `analysis/compute_lay_flat_force.py` | The 39.3 µm threshold was measured on the original rig. Measure your own creep per load step (mean + 3 SD). |
| `SUBSET_LEVEL`, `STEP` | `run_dic/dic_real_images.py` | Half-width of the subset and the grid spacing. Use `validate_dic.py` to choose them. |

Settings are constants at the top of each script.

## 1. Preprocessing and DIC

```
python run_dic/dic_real_images.py [input_folder] [result_folder]
```

The input is a folder of RAW photos (`.ARW`, `.CR2`, `.NEF`, `.DNG` and others). Results go to `results/<folder name>_results` unless you name a result folder.

**Steps for RAW input**

| Step | What happens | Setting |
|---|---|---|
| Scale | px to mm from an ArUco marker or two clicks | `SCALE_CALIBRATION_METHOD` |
| Lens correction | Distortion and vignetting, from a lensfun profile | `LENS_PARAMS` |
| Level | Rotate so two clicked points are horizontal | window |
| Crop | Draw a rectangle | window |
| Flatten | Divide out the large-scale illumination | `FLATTEN_SIGMA` |
| Clip | Limit specular highlights | `REFLECTION_CLIP_PCT` |
| Normalise | Scale to 8 or 16 bit, using the range of the reference frame | `BIT_DEPTH` |
| ROI | Draw a polygon; it snaps to the sample edges and shrinks inward | `REFINE_ROI_WITH_EDGES`, `ROI_SHRINK_PX` |
| Exclusion, grid axis | Optional hole or slit to skip, and the grid direction | window |
| DIC | Tracks the grid through all frames | `SUBSET_LEVEL`, `STEP`, `ZNSSD_LIMIT` |

A folder of `.tif` frames (not `.tiff`) skips the steps from lens correction to normalise and is used as it is. Frames are read in name order and the first one is the reference.

**Re-running.** Every selection is saved as JSON in the result folder, so a second run opens no windows. Delete a file to redo that selection. After changing a preprocessing setting, delete `converted/` and `crop.json`.

**Main outputs**

| File | Content |
|---|---|
| `deformation_grids.npz` | `POIs`, and per frame `positions`, `displacements` and `strain`. Input for the angle scripts. |
| `converted/` | The preprocessed frames |
| `displacement.png`, `strain.png` | Field maps |
| `grid_deformation.gif`, `overlay.gif` | Animations of the deforming grid |

**Tuning the preprocessing.** This shows every stage for a few frames, plus a sweep of `FLATTEN_SIGMA`:

```
python run_dic/debug_preprocessing.py [input_folder] [result_folder]
```

## 2. Angle measurements

**Sign convention for all angles:** 0° = vertical, positive = the top leans right (clockwise in the image).

### Lean along the sample

```
python analysis/measure_vertical_angles_edge_ref.py <result_folder> [--batch] [--overlay-force=<F>]
```

Vertical lines are fitted through the columns of the DIC grid, along the length of the sample, for every frame.

- `delta_angle_deg` is the change since frame 0.
- `absolute_angle_deg` is the true angle. The grid only knows how much the sample turned, so the starting angle comes from the sample's top edge, which is detected in frame 0 and fitted with one polynomial. A line's starting angle is the local slope of that fit.

You mark the applicators and drag one box around the top edge. `--batch` saves the plots without opening them, and needs no windows once the selections are saved.

Outputs: `vertical_angle_per_index.csv` and `.png`, `final_frame_angles.png`, `edge_reference_check.png`, `level_force_result.json`, `lay_flat_force_result.json`.

Check `edge_reference_check.png` first: all absolute angles depend on that fit.

### Seating force

```
python analysis/compute_lay_flat_force.py <result_folder>
python analysis/compute_lay_flat_force.py --all [results_root]
```

This is the force at the end of the last load step in which a tracked group of points still moves more than `SEATING_STEP_UM`. Smaller steps count as creep, and an isolated spike is ignored. The script needs no windows and the angle script above runs the same calculation. A result folder whose name starts with `Ramp_` tracks the points next to applicator 1 instead of the left end.

### Tilt of an upright block

```
python analysis/measure_block_tilt.py <result_folder>
python analysis/measure_block_tilt.py --all
```

This is for samples with an upright block outside the DIC grid. It reads the block's top edge and both sides from the images themselves, so DIC is not used. Top and sides are independent measurements of the same tilt, and their difference is a built-in check. If that difference changes over the frames, look at `block_edge_check.png`.

`--all` processes every `results/ART_SG_*_results` folder and expects names like `ART_SG_10mm_E20_results`.

Outputs: `block_tilt_per_frame.csv`, `block_tilt.png`, `block_edge_check.png`.

## 3. Validation on your own image

`validation/validate_dic.py` imposes known deformations on your image, measures them with the same engine and compares the result with the known answer. Run it without `--image` to pick a file in a dialog, then draw the crop.

```
python validation/validate_dic.py --image photo.tif --subsets 41 61 81 --step 20
```

**Tests on one image**

| Test | Question | Result |
|---|---|---|
| `translation` | What is the smallest shift I can detect? | Noise floors and detection limit per subset, in x and y |
| `strain` | What is the smallest uniform strain I can detect? | Noise floor and detection limit |
| `upper_bound` | What is the largest shift between two frames? | Largest shift that stays accurate, with enough points tracked |
| `resolution` | How small a feature can I resolve? | Measured against imposed amplitude per wavelength, compared with theory |

**Tests on your data**

| Test | Input | Result |
|---|---|---|
| `noise_floor` | `--noise-folder`: repeat photos of a sample that does not move | Real noise of the whole setup, per subset |
| `convergence` | `--result-folder`: a finished result from step 1 | Whether the fields stop changing with subset size. Give at least two sizes. |
| `frame_step` | `--result-folder`: a result, or a folder of results | Largest shift between frames in your test, to compare with `upper_bound` |

**Options**

| Option | Meaning | Default |
|---|---|---|
| `--image` | Image file, or `synthetic` for a generated speckle pattern with an exact known answer | file dialog |
| `--subsets` | Subset widths in px, odd | `81` |
| `--step` | Grid spacing in px | `20` |
| `--znssd` | ZNSSD limit | `0.15` |
| `--directions` | Shift directions | `x y` |
| `--noise-std` | Imposed sensor noise, in grey levels of 255 | `2` |
| `--crop X1 Y1 X2 Y2`, `--no-crop` | Set the crop without the window | draw once |
| `--tests` | Run only these tests | all that have their input |
| `--output` | Output folder | `validation_results/<name>` |

The shift ladders and pass criteria are constants at the top of the script.

**Pass criteria**

- *Detected:* the mean is above 3× the noise floor from zero-shift pairs and within 25% of the imposed value, for that value and every larger one.
- *Upper bound:* the mean is within 5% and at least 95% of the points are tracked, for that shift and every smaller one.

**Reading the results**

- The imposed noise is a model. Warped images contain no vibration or lighting changes, so these limits are best-case values. `noise_floor` gives the real figure for your setup.
- Each shift is an isolated pair of frames. `upper_bound` therefore tests a single step, as between two photos in an experiment.
- Larger images take much longer: the initial guess runs on the CPU for every pair. Crop to the speckled region. A full run on several thousand by several hundred pixels can take well over 10 minutes.
- "not reached" means the limit lies outside what was tested, for example when the image is too narrow for the longest wavelengths.

Outputs: `summary.json`, plus a `.csv` and usually a `.png` per test.

## Repository layout

```
run_dic/
    dic_real_images.py      preprocessing and DIC on one dataset
    debug_preprocessing.py  inspect the preprocessing stages
analysis/
    measure_vertical_angles_edge_ref.py  lean along the sample
    compute_lay_flat_force.py            seating force, load steps per frame
    measure_block_tilt.py                tilt of an upright block
    edge_fit_utils.py                    edge detection and robust fits
validation/
    validate_dic.py         all validation tests
dic_workflow/               support code for the pipeline
    dataset_runner.py       the pipeline for one dataset
    importing.py            RAW loading
    processing.py           flatten, clip, normalise, level, lens correction
    pipeline.py             selection windows, frame conversion, alignment
    edge_detection.py       ROI edge snapping and shrinking
    aruco_calibration.py    px to mm from an ArUco marker
    plotting.py             plots and GIFs
    synthetic.py            analytic speckle for the validation
dic/                        correlation engine (LGPL-3.0)
```

## The correlation engine

`dic/dic.py` is included unchanged from DIC-demo. The original package can get its initial guess from the LiteFlowNet neural network. That part is left out here, so the engine uses its built-in dense optical flow (`Pixel_DenseFlow`, from scikit-image) instead. The sub-pixel correlation is the same and runs on the GPU if one is available.

The theory curves in the resolution test follow Schreier & Sutton (2002), first-order shape functions.

## Licence

**`dic/` (correlation engine):** this code is not my work and remains the property of the authors of [DIC-demo](https://github.com/vincentme/DIC-demo). It is included unchanged under the LGPL-3.0, see [dic/LICENSE](dic/LICENSE). Those terms apply to this folder, not the terms below.

**Everything else in this repository:** free to use, copy, adapt and share, including commercially. The code comes as it is, without any warranty. If you publish work that uses this code, you must cite both:

1. This repository: Philip de Rijk (2026). *MECHANICAL PERFORMANCE OF PATIENT-SPECIFIC OSTEOTOMY GUIDES*. http://repository.tudelft.nl/ 
2. The DIC-demo paper: Zhang, Yunlu, et al. "Bisection searching based reference frame update strategy for digital image correlation." *SN Applied Sciences* 1.6 (2019): 588.
