"""Usage: python scripts\check_m2.py image001 image047 IDRiD_001 ...
Names starting with 'image' go to DIARETDB1, the rest to IDRiD."""
import sys
import time
from pathlib import Path

import cv2
import yaml

from morphoretina.anatomy.retina_mask import normalize_scale, retina_mask
from morphoretina.anatomy.vessels import overlay, segment_vessels
from morphoretina.io.loader import Diaretdb1, load_image
from morphoretina.preprocessing.preprocess import preprocess

cfg = yaml.safe_load(open("configs/default.yaml"))
vcfg = cfg.get("vessels", {})          # optional section, defaults used if absent
out = Path("outputs/check_m2")
out.mkdir(parents=True, exist_ok=True)
db = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")

for name in sys.argv[1:]:
    if name.startswith("image"):
        img = db.load_image(name if name.endswith(".png") else f"{name}.png")
    else:
        img = load_image(f"data/raw/idrid/Imagenes/{name}.jpg")

    small, s = normalize_scale(img, cfg["scale"]["d_ref"])
    gray = preprocess(small, cfg)["denoised"]
    mask = retina_mask(small)

    for mode in ("max", "range"):
        t = time.perf_counter()
        res = segment_vessels(gray, mask, **{**vcfg, "mode": mode})
        dt = time.perf_counter() - t
        frac = res["vessels"].sum() / res["inner"].sum()
        cv2.imwrite(str(out / f"{name}_{mode}.png"), overlay(small, res["vessels"]))
        print(f"{name} [{mode}]: M2a={dt:.2f}s vessel_fraction={frac:.3f}")