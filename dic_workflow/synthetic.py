"""
Analytic synthetic speckle generation for validation/validate_dic.py.

The speckle is modelled as a continuous sum of 2-D Gaussian blobs.  Deformed
frames are produced by transforming the blob centres (and, for strain, their
widths) in closed form and re-rendering — never by interpolating a pixel
image.  This matters for detection-power studies: warping with a spline
interpolator shares its interpolation family with the DIC engine itself,
which can mask or inflate sub-pixel interpolation bias.  Analytic rendering
makes the ground truth exact to machine precision.

Supported deformations (all exact, applied to blob centres):
  translation      centres + (tx, ty)
  uniform x-strain x' = (1+eps)*(x-cx) + cx, sigma_x' = (1+eps)*sigma
"""

import numpy as np


def make_speckle_dots(h, w, n_dots, seed, pad=12):
    """Random blob centres and amplitudes for a speckle pattern.

    Centres are continuous (sub-pixel) positions drawn uniformly over the
    image extended by `pad` px on every side, so blobs keep entering/leaving
    the frame consistently when the pattern is deformed.

    Returns (centers, amps): (n, 2) float64 [x, y] and (n,) float64 in [0.5, 1].
    """
    rng = np.random.default_rng(seed)
    centers = np.column_stack([
        rng.uniform(-pad, w + pad, n_dots),
        rng.uniform(-pad, h + pad, n_dots),
    ])
    amps = rng.uniform(0.5, 1.0, n_dots)
    return centers, amps


def render_speckle(h, w, centers, amps, sigma_x, sigma_y=None):
    """Render Gaussian blobs analytically onto an (h, w) float64 canvas.

    Each blob i contributes amps[i] * exp(-dx^2/2sx^2) * exp(-dy^2/2sy^2),
    evaluated exactly at integer pixel positions within a 4-sigma window.
    """
    if sigma_y is None:
        sigma_y = sigma_x
    rx = int(np.ceil(4 * sigma_x))
    ry = int(np.ceil(4 * sigma_y))
    dx = np.arange(-rx, rx + 1)
    dy = np.arange(-ry, ry + 1)

    x0 = np.floor(centers[:, 0]).astype(np.int64)
    y0 = np.floor(centers[:, 1]).astype(np.int64)
    xx = x0[:, None] + dx[None, :]                                   # (n, Wx)
    yy = y0[:, None] + dy[None, :]                                   # (n, Wy)
    gx = np.exp(-0.5 * ((xx - centers[:, 0:1]) / sigma_x) ** 2)      # (n, Wx)
    gy = np.exp(-0.5 * ((yy - centers[:, 1:2]) / sigma_y) ** 2)      # (n, Wy)
    vals = amps[:, None, None] * gy[:, :, None] * gx[:, None, :]     # (n, Wy, Wx)

    YY = np.broadcast_to(yy[:, :, None], vals.shape)
    XX = np.broadcast_to(xx[:, None, :], vals.shape)
    inside = (YY >= 0) & (YY < h) & (XX >= 0) & (XX < w)

    canvas = np.zeros((h, w), dtype=np.float64)
    np.add.at(canvas, (YY[inside], XX[inside]), vals[inside])
    return canvas


# ── Exact blob-centre deformations ────────────────────────────────────────

def deform_translation(centers, tx, ty=0.0):
    """u = (tx, ty) everywhere."""
    return centers + np.array([tx, ty])


def deform_uniform_strain_x(centers, eps, cx):
    """u(x) = eps * (x - cx), v = 0.  Blob sigma_x must be scaled by (1+eps)."""
    out = centers.copy()
    out[:, 0] = (1.0 + eps) * (centers[:, 0] - cx) + cx
    return out
