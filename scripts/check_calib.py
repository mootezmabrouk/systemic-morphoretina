import numpy as np
from scipy import ndimage as ndi

from morphoretina.anatomy.retina_mask import estimate_dfov
from morphoretina.io.loader import Diaretdb1

d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")
names = d.all_images()

dfov = np.array([estimate_dfov(d.load_image(n)) for n in names])
print("estimate_dfov: mean %.0f min %.0f max %.0f (official mask: 1410)"
      % (dfov.mean(), dfov.min(), dfov.max()))

for k in ("MA", "HE", "EX", "SE"):
    n_img, n_obj, areas = 0, 0, []
    for n in names:
        m = d.load_raw_mask(n, k) >= 189
        lab, c = ndi.label(m)
        if c:
            n_img += 1
            n_obj += c
            areas += ndi.sum(m, lab, range(1, c + 1)).tolist()
    print(k, "images with lesion:", n_img, "objects:", n_obj,
          "median area:", np.median(areas) if areas else None)