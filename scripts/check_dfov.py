import cv2
import numpy as np

from morphoretina.anatomy.retina_mask import estimate_dfov, retina_mask
from morphoretina.io.loader import Diaretdb1

d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")
res = sorted((estimate_dfov(d.load_image(n)), n) for n in d.all_images())
bad = [(round(v), n) for v, n in res if v < 1380]
print(len(bad), "images below 1380:", bad)

for v, n in res[:4]:
    img = d.load_image(n)
    small = cv2.resize(img, (512, round(img.shape[0] * 512 / img.shape[1])),
                       interpolation=cv2.INTER_AREA)
    m = retina_mask(small, closing_size=5)
    ov = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(f"outputs/check_m1/dr1_{n}", ov)
    print(n, round(v))