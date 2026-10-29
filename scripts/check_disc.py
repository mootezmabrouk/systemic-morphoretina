import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from morphoretina.anatomy.optic_disc import find_optic_disc
from morphoretina.anatomy.retina_mask import normalize_scale, retina_mask
from morphoretina.anatomy.vessels import overlay, segment_vessels
from morphoretina.io.loader import Diaretdb1, load_image
from morphoretina.preprocessing.preprocess import preprocess

cfg = yaml.safe_load(open("configs/default.yaml"))
d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")
out = Path("outputs/check_m2")
out.mkdir(parents=True, exist_ok=True)

for name in sys.argv[1:]:
    if name.startswith("image"):
        img = d.load_image(name + ".png")
    else:
        img = load_image(f"data/raw/idrid/Imagenes/{name}.jpg")
    small, s = normalize_scale(img, cfg["scale"]["d_ref"])
    gray = preprocess(small, cfg)["denoised"]
    fov = retina_mask(small)
    v = segment_vessels(gray, fov)
    disc = find_optic_disc(gray, fov)
    ov = overlay(small, v["vessels"])
    if disc is None:
        print(name, "optic disc NOT FOUND")
    else:
        cnts, _ = cv2.findContours(disc["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
        print(name, "disc centre", tuple(round(c) for c in disc["center"]),
              "radius", round(disc["radius"]))
    cv2.imwrite(str(out / f"{name}_m2.png"), ov)