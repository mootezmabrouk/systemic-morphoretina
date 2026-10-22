import cv2
import numpy as np
from scipy import ndimage as ndi

from morphoretina.io.loader import Diaretdb1

d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")


def width(mask):
    cols = np.where(mask.any(axis=0))[0]
    return (cols[-1] - cols[0] + 1) if cols.size else 0


def largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def est(img, method, probe=512):
    f = probe / img.shape[1]
    s = cv2.resize(img, (probe, round(img.shape[0] * f)),
                   interpolation=cv2.INTER_AREA)
    if method == "otsu_red":
        t, _ = cv2.threshold(s[..., 0], 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = s[..., 0] > t
    else:
        m = s.max(axis=2) > method  # fixed threshold on max(R, G, B)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return width(largest(m)) / f


imgs = [d.load_image(n) for n in d.all_images()]
print("target: 1410 on every image")
for method in ["otsu_red", 10, 15, 20, 30, 40]:
    v = np.array([est(i, method) for i in imgs])
    print(method, "mean %.0f min %.0f max %.0f std %.1f"
          % (v.mean(), v.min(), v.max(), v.std()))