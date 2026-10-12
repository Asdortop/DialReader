"""Build the stress set: every test-split photo with one severe degradation applied.

The clean test split has almost no gross errors, which leaves nothing for the
abstention logic or the GenAI hooks to catch. This set supplies hard cases with
the same ground truth.

Run:  python experiments/make_stress_set.py
"""
import csv

import cv2

from common import DATA, load_synth
from dialreader import degrade, w02_attributes as attrs

STRESS = DATA / "synthetic_stress"

HARD = [
    ("gaussian blur 5 px", lambda im, s: degrade.gaussian_blur(im, 5.0)),
    ("motion blur 21 px", lambda im, s: degrade.motion_blur(im, 21, 35.0)),
    ("gaussian noise 40", lambda im, s: degrade.gaussian_noise(im, 40.0, s)),
    ("JPEG quality 5", lambda im, s: degrade.jpeg(im, 5)[0]),
    ("brightness -110", lambda im, s: degrade.brightness_contrast(im, 1.0, -110.0)),
    ("contrast x0.2", lambda im, s: degrade.brightness_contrast(im, 0.2, 0.0)),
    ("salt-and-pepper 20%", lambda im, s: degrade.salt_pepper(im, 0.20, s)),
    ("blur 3 px + noise 20", lambda im, s: degrade.gaussian_noise(degrade.gaussian_blur(im, 3.0), 20.0, s)),
]

if __name__ == "__main__":
    (STRESS / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rec in enumerate(load_synth(split="test")):
        name, fn = HARD[i % len(HARD)]
        img = fn(attrs.read_image(rec["path"]), i)
        cv2.imwrite(str(STRESS / rec["file"]), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        row = {k: v for k, v in rec.items() if k != "path"}
        row.update(set="stress", stress=name)
        rows.append(row)
    with open(STRESS / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} stress photos to {STRESS}")
