"""Module M1: preprocessing (CdCF 3.1).

Place at: src/morphoretina/preprocessing/preprocess.py

Steps: green channel -> CLAHE -> oriented median denoising.
Deterministic: no randomness, fixed parameters from the config.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def green_channel(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 image -> green channel (best lesion/vessel contrast)."""
    return np.ascontiguousarray(rgb[..., 1])


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    return clahe.apply(gray)


def _line_footprint(length: int, angle_deg: float) -> np.ndarray:
    """Boolean footprint: a thin line through the centre at the given angle."""
    fp = np.zeros((length, length), dtype=bool)
    c = (length - 1) / 2.0
    a = np.deg2rad(angle_deg)
    for t in np.linspace(-c, c, 4 * length):
        x = int(round(c + t * np.cos(a)))
        y = int(round(c - t * np.sin(a)))
        fp[y, x] = True
    return fp


def oriented_median(gray: np.ndarray, n_orientations: int = 4,
                    line_length: int = 5) -> np.ndarray:
    """Median along thin line windows at several orientations.

    For each pixel, keep the oriented median closest to the original value:
    noise is removed, while thin structures (vessels) survive because at
    least one orientation runs along them.
    """
    best = gray.copy()
    best_err = np.full(gray.shape, 255, dtype=np.int16)
    g16 = gray.astype(np.int16)
    for k in range(n_orientations):
        fp = _line_footprint(line_length, 180.0 * k / n_orientations)
        med = ndi.median_filter(gray, footprint=fp, mode="reflect")
        err = np.abs(med.astype(np.int16) - g16)
        better = err < best_err
        best[better] = med[better]
        best_err[better] = err[better]
    return best


def preprocess(rgb: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Run M1. Returns intermediate images (useful for notebooks and overlay)."""
    p = cfg["preprocessing"]
    green = green_channel(rgb)
    eq = apply_clahe(green, p["clahe"]["clip_limit"], p["clahe"]["tile_grid"])
    den = oriented_median(eq, p["denoise"]["n_orientations"],
                          p["denoise"]["line_length"])
    return {"green": green, "clahe": eq, "denoised": den}