import numpy as np

from morphoretina.anatomy.retina_mask import fov_diameter
from morphoretina.io.loader import Diaretdb1, _read_gray

d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")

tr, te = d.split("train"), d.split("test")
print("train", len(tr), tr[:3])
print("test", len(te), te[:3])
print("all images", len(d.all_images()))
print("image shape", d.load_image("image001.png").shape)

for k in ("MA", "HE", "EX", "SE"):
    vals = set()
    for n in d.all_images():
        vals |= set(np.unique(d.load_raw_mask(n, k)).tolist())
    print(k, "mask values:", sorted(vals))

fm = list((d.img_dir.parent / "ddb1_fundusmask").glob("*"))
print("fov mask files:", [p.name for p in fm])
m = _read_gray(fm[0]) > 0
print("fov mask shape", m.shape, "DFOV", fov_diameter(m), "fraction", round(float(m.mean()), 2))