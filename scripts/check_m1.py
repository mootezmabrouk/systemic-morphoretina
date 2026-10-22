import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from morphoretina.anatomy.retina_mask import normalize_scale, retina_mask
from morphoretina.io.loader import load_image
from morphoretina.preprocessing.preprocess import preprocess

cfg = yaml.safe_load(open("configs/default.yaml"))
out = Path("outputs/check_m1")
out.mkdir(parents=True, exist_ok=True)

for name in sys.argv[1:]:
    img = load_image(f"data/raw/idrid/Imagenes/{name}.jpg")
    t = time.perf_counter()
    small, s = normalize_scale(img, cfg["scale"]["d_ref"])
    res = preprocess(small, cfg)
    dt = time.perf_counter() - t
    mask = retina_mask(small)
    ov = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 3)
    cv2.imwrite(str(out / f"{name}_mask.png"), ov)
    cv2.imwrite(str(out / f"{name}_denoised.png"), res["denoised"])
    print(f"{name}: s={s:.2f} shape={small.shape[:2]} M1={dt:.2f}s mask_fraction={mask.mean():.2f}")