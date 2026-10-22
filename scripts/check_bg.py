import numpy as np

from morphoretina.io.loader import Diaretdb1

d = Diaretdb1("data/raw/diaretdb1/diaretdb1_v_1_1")
for n in ("image072.png", "image037.png", "image001.png"):
    mx = d.load_image(n).max(axis=2)
    c = np.concatenate([mx[:60, :60].ravel(), mx[:60, -60:].ravel(),
                        mx[-60:, :60].ravel(), mx[-60:, -60:].ravel()])
    print(n, "corners max(RGB): median", np.median(c),
          "p99", np.percentile(c, 99), "max", c.max())
    for t in (15, 20, 25, 30, 40):
        print("   thr", t, "fraction above:", round(float((mx > t).mean()), 3))