"""
Image pre-processing: illumination flattening, reflection clipping,
bit-depth normalisation, and optional lens distortion correction.
"""

import numpy as np


def flatten_illumination(img, sigma):
    """Divide image by heavily blurred background to remove slow-varying gradients."""
    from scipy.ndimage import gaussian_filter
    background = gaussian_filter(img, sigma=sigma)
    background = np.maximum(background, background.max() * 0.01)
    return img / background


def clip_reflections(img, percentile):
    """Clamp bright pixels at given percentile to suppress specular hot-spots."""
    return np.clip(img, None, np.percentile(img, percentile))


def normalize_8bit(img, ref_min=None, ref_max=None):
    """Scale float image to uint8, optionally anchored to a reference min/max."""
    mn = float(img.min()) if ref_min is None else ref_min
    mx = float(img.max()) if ref_max is None else ref_max
    scaled = (img - mn) / max(mx - mn, 1.0) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def normalize_16bit(img, ref_min=None, ref_max=None):
    """Scale float image to uint16, optionally anchored to a reference min/max."""
    mn = float(img.min()) if ref_min is None else ref_min
    mx = float(img.max()) if ref_max is None else ref_max
    scaled = (img - mn) / max(mx - mn, 1.0) * 65535.0
    return np.clip(scaled, 0, 65535).astype(np.uint16)


def rotate_to_level(img, angle_deg):
    """Rotate image about its center by angle_deg so a previously-tilted
    line becomes horizontal. Output has the same shape as the input.

    angle_deg is the tilt returned by pipeline.select_level_points
    (atan2 of the two clicked points, in degrees). Pass 0.0 to skip.
    """
    if angle_deg == 0.0:
        return img
    import cv2
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REPLICATE)


def build_lens_corrector(shape, cam_maker, cam_model, lens_maker, lens_model,
                          focal_length, aperture=2.8, distance=10.0):
    """Build the lensfunpy Modifier for geometry distortion + vignetting correction.

    Call once per image series (same camera / lens / focal length). Use the
    returned coords with cv2.remap() for geometry, and pass mod to
    apply_vignetting() per frame.

    Parameters
    ----------
    shape        : (H, W) of the full-sensor image before any crop.
    cam_maker    : camera body maker string as recognised by lensfunpy (e.g. 'Sony').
    cam_model    : camera body model string (e.g. 'ILCE-7M2').
    lens_maker   : lens maker string (e.g. 'Sony').
    lens_model   : lens model string (e.g. 'FE 28-70mm f/3.5-5.6 OSS').
    focal_length : actual focal length in mm (not 35 mm equivalent).
    aperture     : actual f-number of the shot (drives vignetting strength).
    distance     : focus distance in metres.

    Returns
    -------
    coords : float32 ndarray usable as the first map argument of cv2.remap,
             for geometry distortion correction.
    mod    : lensfunpy.Modifier, reusable for per-frame vignetting correction
             via apply_vignetting(mod, gray).
    """
    import lensfunpy
    db   = lensfunpy.Database()
    cam  = db.find_cameras(cam_maker, cam_model)[0]
    lens = db.find_lenses(cam, lens_maker, lens_model)[0]
    h, w = shape
    mod  = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal_length, aperture, distance, pixel_format=np.float32)
    return mod.apply_geometry_distortion(), mod


def apply_vignetting(mod, gray):
    """Apply lensfun vignetting correction to a single-channel image.

    lensfun's apply_color_modification requires a 3-channel (H, W, 3) array,
    so the grayscale frame is tiled across channels, corrected in place, and
    the (possibly per-channel) gains are averaged back down to grayscale.

    Must run before geometry distortion is remapped, since the vignetting
    model is defined in as-shot (distorted) pixel coordinates.
    """
    rgb = np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32))
    mod.apply_color_modification(rgb)
    return rgb.mean(axis=2)
