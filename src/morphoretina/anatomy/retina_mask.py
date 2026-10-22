"""Retina field-of-view mask (CdCF 3.2) and scale normalisation (CdCF 3).

Place at: src/morphoretina/anatomy/retina_mask.py
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def retina_mask(rgb: np.ndarray, closing_size: int = 15) -> np.ndarray:
    """Otsu on the red channel + global morphological closing -> bool mask."""
    red = np.ascontiguousarray(rgb[..., 0])
    thr, _ = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (red > thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    lab, n = ndi.label(mask)
    if n == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = lab == (1 + int(np.argmax(sizes)))
    return ndi.binary_fill_holes(mask)


def fov_diameter(mask: np.ndarray) -> int:
    """Horizontal extent of the mask (robust to vertically truncated fields)."""
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[-1] - cols[0] + 1) if cols.size else 0


def estimate_dfov(rgb: np.ndarray, probe_width: int = 512) -> float:
    """Field-of-view diameter in original pixels, measured on a thumbnail."""
    h, w = rgb.shape[:2]
    f = probe_width / w
    small = cv2.resize(rgb, (probe_width, round(h * f)), interpolation=cv2.INTER_AREA)
    return fov_diameter(retina_mask(small, closing_size=5)) / f


def normalize_scale(rgb: np.ndarray, d_ref: float) -> tuple[np.ndarray, float]:
    """Resize so the field of view has diameter d_ref. Returns (image, s)."""
    s = estimate_dfov(rgb) / d_ref
    h, w = rgb.shape[:2]
    new_size = (max(1, round(w / s)), max(1, round(h / s)))
    interp = cv2.INTER_AREA if s > 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, new_size, interpolation=interp), s