"""
compute_lay_flat_force.py
=========================
Seating force of one dataset (or, with --all, of every *_results folder in a
results folder), without opening any windows. The file and JSON names call
it "lay flat".

Needs deformation_grids.npz, scale.json (the threshold is in um),
FORCE_PER_FRAME_CANDIDATES below to know the force per frame, and for Ramp
tests applicators.json.

Method: the mean downward displacement of a small group of POIs (see
"Which POIs") is followed over the whole test, and its movement in each load
step (frame to frame) is compared with SEATING_STEP_UM:
  - A step larger than that is real movement; a smaller one is creep (after
    seating, the guide keeps settling by a roughly constant amount per frame,
    whatever the size of the load step).
  - A single step above the threshold with quiet steps on both sides is
    ignored as a spike (e.g. a tracking jump). The first step is always
    counted, as it has no step before it.
  - Seating force = the force at the end of the last step that is still real
    movement. Looking back from the last frame means a guide that barely
    moves at first and more at higher forces still counts as moving.
  - If the last step is still real movement, the guide had not seated by the
    highest force: 'seated' is False and frame = last_frame.

The threshold is absolute rather than a share of each guide's own travel
(e.g. "the first frame at 90 % of the total"): the total includes the creep,
so a relative rule lands near the end of the test whenever creep dominates,
and it cannot report "not seated".

Which POIs (only POIs tracked in EVERY frame, so the mean doesn't jump when
one is lost):
  - Default: the sample's tilted left end — the leftmost native grid
    columns, added one at a time until at least LAY_FLAT_MIN_POIS POIs are
    in (left_end_displacement).
  - Result folders whose name starts with 'Ramp_': the guide's left end rests
    on a bump and barely moves, while applicator 1 presses the guide down to
    the right of it. The tracked POIs are the RAMP_N_POIS POIs closest to the
    right of applicator 1's right edge, as marked in
    measure_vertical_angles_edge_ref.py and saved in applicators.json
    (right_of_applicator_displacement).

SEATING_STEP_UM is mean + 3 SD of the creep per load step in a test where
nothing can seat (like a detection limit over a blank). On the original rig:
Ramp position 1 (applicator on the bump), 10 POIs right of applicator 1,
every load step after the first two, 4/6/8/10 mm samples — n = 48, mean
10.3 um, SD 9.7 um -> 39.3 um. Measure it again for another setup.

measure_vertical_angles_edge_ref.py imports compute_lay_flat() and
FORCE_PER_FRAME_CANDIDATES from here, so the method and the load ladders
each live in one place.

Usage:
    python analysis/compute_lay_flat_force.py <result_folder>
    python analysis/compute_lay_flat_force.py --all [results_root]

Output:
    <result_folder>/lay_flat_force_result.json
"""
import glob
import json
import os
import sys

import numpy as np

LAY_FLAT_MIN_POIS = 5      # leftmost columns are added until this many POIs are in
RAMP_N_POIS       = 10     # POIs right of applicator 1 tracked in the Ramp tests
SEATING_STEP_UM   = 39.3   # mean + 3 SD of creep per load step (Ramp position 1); above = real movement

# Candidate force-per-deformed-frame ladders (frame 1, 2, ... — frame 0 is the
# unloaded reference). The one with exactly as many entries as the dataset has
# deformed frames is used; with no match, or more than one, the force is left
# unknown instead of guessed. Add a list here for a new load ladder.
FORCE_PER_FRAME_CANDIDATES = [
    [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 125, 150, 175, 200],     # standard, first frame missed
    [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 125, 150, 175, 200],  # standard
    [1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200],                                   # ramp tests
    [1, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200],                                              # lower-res, small-displacement tests
]


def left_end_displacement(result_folder):
    """Downward displacement of the sample's left end, per frame.

    Returns (dict, None) or (None, reason_str). The dict holds:
      'disp_px'         (n_frames,) mean y-displacement of the selected POIs
                        relative to frame 0, in px (+ = down in the image)
      'force_per_frame' (n_frames - 1,) applied force per deformed frame, or
                        None if no FORCE_PER_FRAME_CANDIDATES entry matches
      'n_columns', 'n_pois', 'x_ref_px_range'   which POIs were used
    """
    npz_path = os.path.join(result_folder, 'deformation_grids.npz')
    if not os.path.isfile(npz_path):
        return None, 'no deformation_grids.npz'

    data = np.load(npz_path)
    POIs = data['POIs']
    positions = data['positions']
    n_frames = positions.shape[0]
    n_deformed = n_frames - 1

    matches = [c for c in FORCE_PER_FRAME_CANDIDATES if len(c) == n_deformed]
    force_per_frame = np.asarray(matches[0], dtype=float) if len(matches) == 1 else None

    order = np.argsort(POIs[:, 0])
    xs_sorted = POIs[order, 0]
    gaps = np.diff(xs_sorted)
    if len(gaps) == 0:
        return None, 'not enough POIs to form columns'
    gaps_sorted = np.sort(gaps)
    ratios = gaps_sorted[1:] / np.maximum(gaps_sorted[:-1], 1e-9)
    split_i = int(np.argmax(ratios)) + 1
    break_thresh = float(gaps_sorted[split_i])
    col_break = np.concatenate([[True], gaps > break_thresh * 0.5])
    col_id = np.empty(len(POIs), dtype=int)
    col_id[order] = np.cumsum(col_break) - 1
    n_cols = int(col_id.max()) + 1

    columns = []
    for c in range(n_cols):
        idxs = np.where(col_id == c)[0]
        if len(idxs) < 2:
            continue
        columns.append({'x_ref': float(POIs[idxs, 0].mean()), 'idxs': idxs})
    columns.sort(key=lambda d: d['x_ref'])
    if not columns:
        return None, 'no valid native columns'

    # Leftmost columns, added until LAY_FLAT_MIN_POIS POIs that are tracked in
    # every frame are included.
    always_valid = ~np.isnan(positions).any(axis=(0, 2))
    left_cols, left_idxs = [], []
    for col in columns:
        idxs = col['idxs'][always_valid[col['idxs']]]
        if len(idxs) == 0:
            continue
        left_cols.append(col)
        left_idxs.extend(idxs)
        if len(left_idxs) >= LAY_FLAT_MIN_POIS:
            break
    if not left_idxs:
        return None, 'no POI in the leftmost columns is tracked in every frame'
    left_idxs = np.asarray(left_idxs)

    left_disp = positions[:, left_idxs, 1].mean(axis=1)
    return {
        'disp_px': left_disp - left_disp[0],
        'force_per_frame': force_per_frame,
        'n_columns': len(left_cols),
        'n_pois': int(len(left_idxs)),
        'x_ref_px_range': [float(left_cols[0]['x_ref']), float(left_cols[-1]['x_ref'])],
        'reference': 'left end',
    }, None


def right_of_applicator_displacement(result_folder):
    """Downward displacement of the guide just right of applicator 1, per
    frame: the mean over the RAMP_N_POIS POIs closest to the right of the
    applicator's right edge that are tracked in every frame. Same return
    format as left_end_displacement."""
    npz_path = os.path.join(result_folder, 'deformation_grids.npz')
    app_path = os.path.join(result_folder, 'applicators.json')
    if not os.path.isfile(npz_path):
        return None, 'no deformation_grids.npz'
    if not os.path.isfile(app_path):
        return None, 'no applicators.json (mark them in measure_vertical_angles_edge_ref.py)'
    with open(app_path) as f:
        edges = json.load(f).get('applicator_1_edges_px')
    if not edges:
        return None, 'applicator 1 not marked in applicators.json'
    x_edge = max(edges)

    data = np.load(npz_path)
    POIs = data['POIs']
    positions = data['positions']
    n_deformed = positions.shape[0] - 1
    matches = [c for c in FORCE_PER_FRAME_CANDIDATES if len(c) == n_deformed]
    force_per_frame = np.asarray(matches[0], dtype=float) if len(matches) == 1 else None

    always_valid = ~np.isnan(positions).any(axis=(0, 2))
    right = np.nonzero((POIs[:, 0] >= x_edge) & always_valid)[0]
    idxs = right[np.argsort(POIs[right, 0])[:RAMP_N_POIS]]
    if len(idxs) < RAMP_N_POIS:
        return None, (f'only {len(idxs)} POI(s) right of applicator 1 are tracked in every frame '
                      f'(need {RAMP_N_POIS})')

    disp = positions[:, idxs, 1].mean(axis=1)
    return {
        'disp_px': disp - disp[0],
        'force_per_frame': force_per_frame,
        'n_columns': None,
        'n_pois': int(len(idxs)),
        'x_ref_px_range': [float(POIs[idxs, 0].min()), float(POIs[idxs, 0].max())],
        'reference': 'right of applicator 1',
    }, None


def seating_frame(step_um):
    """Frame at which the tracked POIs have seated, from their movement per
    load step (um, frame i-1 -> i at index i-1). Returns (frame, seated):
    the frame at the end of the last step above SEATING_STEP_UM that is not
    an isolated spike (0 if there is none); seated is False if that is the
    last step, i.e. still moving at the highest force."""
    hot = np.abs(step_um) > SEATING_STEP_UM
    paired = np.zeros_like(hot)
    paired[1:] |= hot[:-1]
    paired[:-1] |= hot[1:]
    moving = hot & paired
    moving[0] = hot[0]
    steps = np.nonzero(moving)[0]
    if len(steps) == 0:
        return 0, True
    frame = int(steps[-1]) + 1
    return frame, frame < len(step_um)


def compute_lay_flat(result_folder):
    """Seating force for one dataset (see the module docstring). Returns
    (result_dict, None) or (None, reason_str) if it couldn't be computed for
    this folder."""
    is_ramp = os.path.basename(os.path.normpath(result_folder)).startswith('Ramp_')
    end, err = (right_of_applicator_displacement if is_ramp else left_end_displacement)(result_folder)
    if end is None:
        return None, err
    scale_path = os.path.join(result_folder, 'scale.json')
    if not os.path.isfile(scale_path):
        return None, 'no scale.json (the step threshold is in um)'
    with open(scale_path) as f:
        mm_per_px = json.load(f).get('mm_per_px')
    if not mm_per_px:
        return None, 'no mm_per_px in scale.json'

    disp_px, force_per_frame = end['disp_px'], end['force_per_frame']
    step_um = np.diff(disp_px) * mm_per_px * 1000
    frame, seated = seating_frame(step_um)

    force = None
    if force_per_frame is not None:
        force = 0.0 if frame == 0 else float(force_per_frame[frame - 1])

    result = {
        'frame': frame,
        'force_N': force,
        'seated': seated,
        'last_frame': len(disp_px) - 1,
        'reference': end['reference'],
        'n_columns': end['n_columns'],
        'n_pois': end['n_pois'],
        'x_ref_px_range': end['x_ref_px_range'],
        'travel_px': float(disp_px[-1]),
        'step_threshold_um': SEATING_STEP_UM,
        'step_um': [round(float(s), 1) for s in step_um],
        'last_step_um': round(float(step_um[-1]), 1),
    }
    return result, None


def main():
    args = sys.argv[1:]
    do_all = '--all' in args
    args = [a for a in args if a != '--all']
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

    if do_all:
        results_root = args[0] if args else os.path.join(root, 'results')
        folders = sorted(glob.glob(os.path.join(results_root, '*_results')))
    elif args:
        folders = [args[0]]
    else:
        sys.exit('Usage: python analysis/compute_lay_flat_force.py <result_folder> | --all [results_root]')

    for folder in folders:
        dataset = os.path.basename(folder)
        result, err = compute_lay_flat(folder)
        if result is None:
            print(f'{dataset}: skipped ({err})')
            continue
        out_path = os.path.join(folder, 'lay_flat_force_result.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        force_str = (f"{result['force_N']:.1f} N" if result['force_N'] is not None
                     else 'n/a (no FORCE_PER_FRAME match)')
        if not result['seated']:
            force_str = f'not seated by {force_str} (last step {result["last_step_um"]:.0f} um)'
        print(f"{dataset}: {result['reference']}, seating force {force_str}, "
              f"frame {result['frame']} -> {out_path}")


if __name__ == '__main__':
    main()
