"""
RAW image loading and folder scanning.
"""

import os
import numpy as np

RAW_EXTENSIONS = {'.arw', '.cr2', '.cr3', '.nef', '.nrw', '.dng',
                  '.orf', '.rw2', '.pef', '.srw', '.raf'}


def load_raw(path, resize_to=None):
    """Read a RAW camera file via rawpy; return float32 grayscale."""
    import rawpy
    with rawpy.imread(path) as rf:
        rgb = rf.postprocess(
            use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True,
            gamma=(1, 1),
        )
    gray = (0.299 * rgb[..., 0].astype(np.float32)
            + 0.587 * rgb[..., 1].astype(np.float32)
            + 0.114 * rgb[..., 2].astype(np.float32))
    if resize_to is not None:
        from scipy.ndimage import zoom
        rh, rw = resize_to
        gray = zoom(gray, (rh / gray.shape[0], rw / gray.shape[1]), order=3)
    return gray


def scan_raw_folder(folder, start=0, end=None):
    """Return sorted list of RAW filenames in folder, sliced to [start:end+1]."""
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in RAW_EXTENSIONS
    )
    if not files:
        return []
    last = len(files) - 1 if end is None else end
    return files[start: last + 1]
