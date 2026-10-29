"""Module M2a: blood vessels by directional black top-hat.

Place at: src/morphoretina/anatomy/vessels.py

Input: denoised green channel (uint8, scaled to d_ref) and the FOV mask.
Output: vessel response map and binary vessel mask.
Parameters are PROVISIONAL until checked against the CdCF values.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {
    "length": 21,       # line length in px at d_ref (must exceed the widest vessel)
    "n_orient": 12,     # 15-degree steps
    "mode": "max",      # "max" (plain) or "range" (max - min, suppresses round blobs)
    "border": 31,       # px eroded from the FOV edge to avoid edge artefacts
    "low_pct": 90,      # hysteresis low threshold, percentile inside the FOV
    "high_pct": 96,     # hysteresis high threshold
    "min_area": 40,     # px, smallest kept component
}


def line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Thin line (1 px wide) through the centre, uint8 {0,1}."""
    length = length | 1  # force odd
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * c, -np.sin(a) * c
    cv2.line(k, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return k


def vessel_response(gray: np.ndarray, mask: np.ndarray, length: int = 21,
                    n_orient: int = 12, mode: str = "max",
                    border: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Directional black top-hat. Returns (response float32, inner mask bool)."""
    resp = np.stack([
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                         line_kernel(length, i * 180.0 / n_orient))
        for i in range(n_orient)
    ]).astype(np.float32)
    r = resp.max(0) if mode == "max" else resp.max(0) - resp.min(0)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border, border))
    inner = cv2.erode(mask.astype(np.uint8), se) > 0
    r[~inner] = 0
    return r, inner


def hysteresis(r: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep connected regions of (r >= low) that contain a pixel >= high."""
    lab, n = ndi.label(r >= low, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(r.shape, bool)
    keep = np.unique(lab[r >= high])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def remove_small(seg: np.ndarray, min_area: int) -> np.ndarray:
    lab, n = ndi.label(seg, structure=np.ones((3, 3)))
    if n == 0:
        return seg
    sizes = ndi.sum(seg, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= min_area))


def segment_vessels(gray: np.ndarray, mask: np.ndarray, **params) -> dict:
    """Run M2a. Returns {'response', 'vessels', 'inner'}."""
    p = {**DEFAULTS, **params}
    r, inner = vessel_response(gray, mask, p["length"], p["n_orient"],
                               p["mode"], p["border"])
    r[bright_zone(gray, inner)] = 0
    lo, hi = np.percentile(r[inner], [p["low_pct"], p["high_pct"]])
    seg = hysteresis(r, lo, hi)
    seg = remove_small(seg, p["min_area"])
    return {"response": r, "vessels": seg, "inner": inner}


def overlay(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Red overlay on an RGB image (returns BGR, ready for cv2.imwrite)."""
    out = rgb.copy()
    out[seg] = (0.5 * out[seg] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def bright_zone(gray, inner, size=41, pct=97, grow=15):
    """Zone around bright blobs (exudates), to keep out of the vessel mask."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    wth = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se).astype(np.float32)
    thr = np.percentile(wth[inner], pct)
    zone = (wth >= thr).astype(np.uint8)
    zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
    return zone > 0