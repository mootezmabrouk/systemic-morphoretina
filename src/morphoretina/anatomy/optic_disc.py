"""Module M2b: optic disc (CdCF 3.2).

Place at: src/morphoretina/anatomy/optic_disc.py

Input: denoised green channel at d_ref scale (uint8) and the FOV mask.
DEVIATION from CdCF: the CdCF says opening, but vessels are dark in the green
channel, so a closing is needed to erase them; the opening then removes small
bright spots (exudates). The geodesic dilation of the p98 marker inside the
p92 region is done as a binary reconstruction (label + keep components that
contain a marker pixel).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

DEFAULTS = {
    "radius": 25,        # px at d_ref (CdCF: R > 25 px)
    "work_width": 355,   # processing width (d_ref / 4) for speed
    "marker_pct": 98,
    "region_pct": 92,
    "erode_fov": 30,     # px at d_ref, ignore the FOV edge
    "min_frac": 0.003,   # plausible disc area / FOV area
    "max_frac": 0.06,
}


def find_optic_disc(gray: np.ndarray, fov: np.ndarray, **params) -> dict | None:
    """Returns {'mask' (bool, full size), 'center' (x, y), 'radius'} or None."""
    p = {**DEFAULTS, **params}
    h, w = gray.shape
    f = p["work_width"] / w
    size = (p["work_width"], round(h * f))
    g = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(fov.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    ke = max(3, round(p["erode_fov"] * f)) | 1
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))) > 0

    r = max(3, round(p["radius"] * f))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)   # erase dark vessels
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, se)    # erase small bright spots
    g = g.astype(np.float32)
    g[~inner] = 0

    hi, lo = np.percentile(g[inner], [p["marker_pct"], p["region_pct"]])
    lab, n = ndi.label(g >= lo)
    keep = np.unique(lab[g >= hi])
    keep = keep[keep > 0]
    if keep.size == 0:
        return None
    sizes = ndi.sum(np.ones_like(lab), lab, keep)
    best = keep[int(np.argmax(sizes))]
    reg = ndi.binary_fill_holes(lab == best)

    frac = reg.sum() / max(1, m.sum())
    if not (p["min_frac"] <= frac <= p["max_frac"]):
        return None

    ys, xs = np.nonzero(reg)
    full = cv2.resize(reg.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_LINEAR) > 0.5
    area_full = float(full.sum())
    return {"mask": full & fov.astype(bool),
            "center": (xs.mean() / f, ys.mean() / f),
            "radius": float(np.sqrt(area_full / np.pi))}