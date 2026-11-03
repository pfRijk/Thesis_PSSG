"""
Auto pixel-to-mm calibration using a printed ArUco fiducial marker.

Marker spec (must match the printed target; print it at exactly this size
and measure it after printing):
    Dictionary : DICT_4X4_50
    id 0 -> 10.000 mm side
    id 1 -> 20.000 mm side
For another size, edit MARKER_SIZES_MM.

Standalone usage:
    python dic_workflow/aruco_calibration.py path/to/calibration_image.jpg
    python dic_workflow/aruco_calibration.py path/to/calibration_image.jpg --id 1
"""

import numpy as np

MARKER_SIZES_MM = {0: 10.0, 1: 20.0}   # marker id -> printed side length [mm]
MARKER_ID = 0  # default; override with --id N or pass marker_id= directly


def _load_gray(image, cv2):
    """Accept either an image file path (str) or an already-loaded array
    (e.g. a raw reference frame straight from the DIC pipeline, before it's
    written to disk) and return an 8-bit single-channel array for detection.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _brighten(gray, gamma, cv2):
    """Gamma-lift an 8-bit image (gamma > 1 brightens). A gamma curve lifts
    the dark end far more than the bright end, so a marker in a dim scene
    becomes readable without the white parts clipping the way a plain
    multiplicative gain would."""
    lut = (255.0 * (np.arange(256) / 255.0) ** (1.0 / gamma)).astype(np.uint8)
    return cv2.LUT(gray, lut)


def compute_mm_per_pixel(image, marker_id=MARKER_ID, marker_size_mm=None, show=False,
                         hint_xy=None, hint_radius=250, hint_upscale=3.0,
                         brightness_steps=(1.5, 2.0, 3.0)):
    """Detect an ArUco marker in `image` and return the mm/px scale.

    `image` may be a file path (str) or an already-loaded grayscale/BGR array.

    hint_xy : optional (x, y) pixel location (e.g. user-clicked) near where
              the marker actually is. When given, detection is first tried
              on a cropped, upscaled window around that point — this finds
              markers the full-frame detector misses because they're small
              relative to the whole image. Falls back to a full-frame search
              if nothing is found in the window.
    hint_radius  : half-width in px (in the *original* image) of the search
                   window around hint_xy.
    hint_upscale : factor the cropped window is upscaled by before detection
                   (helps the detector resolve a small marker's bit pattern).
                   If that fails, 1x and 0.5x are tried too, since a large
                   marker only reads at native size or smaller.
    brightness_steps : gamma values tried, in order, on the hinted window
                   AFTER the unmodified window fails (a marker missed
                   because the frame is too dark often reads fine once
                   brightened). Each step brightens more than the last;
                   empty/None = unmodified window only. Ignored without
                   hint_xy.

    Raises FileNotFoundError if the image can't be read, or RuntimeError if
    the marker isn't found.
    """
    import cv2

    if marker_size_mm is None:
        if marker_id not in MARKER_SIZES_MM:
            raise ValueError(
                f"Unknown marker_id {marker_id}; pass marker_size_mm explicitly."
            )
        marker_size_mm = MARKER_SIZES_MM[marker_id]

    gray = _load_gray(image, cv2)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    marker_corners = None

    if hint_xy is not None:
        h, w = gray.shape
        hx, hy = hint_xy
        x1 = int(max(0, hx - hint_radius))
        y1 = int(max(0, hy - hint_radius))
        x2 = int(min(w, hx + hint_radius))
        y2 = int(min(h, hy + hint_radius))
        crop = gray[y1:y2, x1:x2]

        # Unmodified window first, then progressively brighter versions.
        crops = [(None, crop)]
        for gamma in (brightness_steps or ()):
            crops.append((gamma, _brighten(crop, gamma, cv2)))

        # Scales: the requested upscale first (small markers), then 1x and
        # 0.5x. A LARGE marker (a couple hundred px across) fails when
        # upscaled — its modules become far bigger than the detector's
        # fixed-size adaptive-threshold window can read — so it needs the
        # window at native size or shrunk.
        scales = [hint_upscale] + [s for s in (1.0, 0.5) if s != hint_upscale]

        for scale in scales:
            for gamma, candidate in crops:
                search = (cv2.resize(candidate, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC if scale > 1
                                     else cv2.INTER_AREA)
                          if scale != 1.0 else candidate)
                corners, ids, _ = detector.detectMarkers(search)
                if ids is not None and marker_id in ids.flatten():
                    idx = list(ids.flatten()).index(marker_id)
                    marker_corners = corners[idx].reshape(4, 2) / scale + [x1, y1]
                    how = ("" if gamma is None
                           else f", brightened (gamma {gamma:g})")
                    print(f"Marker found near click hint ({int(hx)}, {int(hy)}) "
                          f"at window scale {scale:g}x{how}.")
                    break
            if marker_corners is not None:
                break
            print(f"  hinted window at scale {scale:g}x: marker not detected "
                  "at any brightness.")

        if marker_corners is None:
            print("Marker not found in the hinted search window at any "
                 "scale/brightness — falling back to full-frame search.")

    if marker_corners is None:
        # Plain full-frame search. When a click hint was given (i.e. the
        # plain search already failed once and the hinted window didn't
        # catch it either — e.g. the click was a bit off and the marker fell
        # partly outside the window), also retry the whole frame brightened,
        # so an imprecise click doesn't sink a marker that is findable.
        full_attempts = [(None, gray)]
        if hint_xy is not None:
            for gamma in (brightness_steps or ()):
                full_attempts.append((gamma, _brighten(gray, gamma, cv2)))

        for gamma, candidate in full_attempts:
            corners, ids, _ = detector.detectMarkers(candidate)
            if ids is not None and marker_id in ids.flatten():
                if gamma is not None:
                    print(f"Marker found in full-frame search after brightening "
                          f"(gamma {gamma:g}).")
                break
        else:
            raise RuntimeError(
                "Marker not found. Check lighting/focus, that the marker isn't "
                "cropped, or that marker_id matches the printed target."
            )
        idx = list(ids.flatten()).index(marker_id)
        marker_corners = corners[idx].reshape(4, 2)  # top-left, top-right, bottom-right, bottom-left

    # Average side length in pixels across all 4 edges
    side_lengths = [
        np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
        for i in range(4)
    ]
    mean_px_per_side = float(np.mean(side_lengths))

    mm_per_pixel = marker_size_mm / mean_px_per_side
    px_per_mm = 1.0 / mm_per_pixel

    print(f"Detected marker side (avg): {mean_px_per_side:.3f} px")
    print(f"Scale                     : {mm_per_pixel:.6f} mm/px")
    print(f"Scale                     : {px_per_mm:.6f} px/mm")

    if show:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        cv2.imshow("Detected marker", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mm_per_pixel


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python aruco_calibration.py <image_path> [--id 0|1] [--show]")
        sys.exit(1)

    chosen_id = MARKER_ID
    if "--id" in sys.argv:
        chosen_id = int(sys.argv[sys.argv.index("--id") + 1])

    compute_mm_per_pixel(sys.argv[1], marker_id=chosen_id, show="--show" in sys.argv)
