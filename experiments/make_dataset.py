"""Build the synthetic dataset that stands in for the hand-collected photo set.

Follows the collection protocol of the plan:
  base set       ~10 readings per gauge type in good conditions
  variation set  3 gauges x 10 readings x 3 tilts (0, 20, 40 deg) x 2 lightings (room, window glare)
  two phones, and two gauge types held out entirely for testing.

Run:  python experiments/make_dataset.py
"""
import csv

import cv2
import numpy as np
from joblib import Parallel, delayed

from common import SYNTH, save_table
from dialreader import synth, attributes as attrs

import pandas as pd

VARIATION_GAUGES = ("pressure_gauge", "tyre_gauge", "thermometer")
READINGS = 10
VAL_READINGS = (0, 3, 6, 9)          # reading ids that go to the validation split


def plan():
    rng = np.random.default_rng(2026)
    jobs = []
    for gi, (name, spec) in enumerate(synth.GAUGES.items()):
        for j in range(READINGS):
            v = synth.random_value(spec, rng)
            jobs.append(dict(set="base", gauge=name, reading=j, value=v, seed=1000 * gi + j, tilt=None,
                             lighting="room", phone="AB"[j % 2]))
    for gi, name in enumerate(VARIATION_GAUGES):
        spec = synth.GAUGES[name]
        for j in range(READINGS):
            v = synth.random_value(spec, rng)
            for tilt in (0.0, 20.0, 40.0):
                for lighting in ("room", "window"):
                    jobs.append(dict(set="variation", gauge=name, reading=j, value=v,
                                     seed=500000 + 10000 * gi + 100 * j + int(tilt) + (lighting == "window"),
                                     tilt=tilt, lighting=lighting, phone="AB"[j % 2]))
    return jobs


def render(job):
    spec = synth.GAUGES[job["gauge"]]
    img, truth = synth.render_photo(spec, job["value"], seed=job["seed"], tilt_deg=job["tilt"],
                                    lighting=job["lighting"], phone=job["phone"])
    tilt_tag = "free" if job["tilt"] is None else f"t{int(job['tilt']):02d}"
    name = f"{job['set']}_{job['gauge']}_r{job['reading']:02d}_{tilt_tag}_{job['lighting']}.jpg"
    (SYNTH / "images").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SYNTH / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if job["gauge"] in synth.HELD_OUT:
        split = "test"
    else:
        split = "val" if job["reading"] in VAL_READINGS else "test"
    return {
        "file": f"images/{name}", "set": job["set"], "split": split, "gauge": job["gauge"],
        "held_out": int(job["gauge"] in synth.HELD_OUT), "reading": job["reading"],
        "value": round(job["value"], 6), "unit": spec.unit, "vmin": spec.vmin, "vmax": spec.vmax,
        "scale_type": spec.scale_type, "direction": "clockwise" if spec.clockwise else "anticlockwise",
        "n_ticks": spec.n_ticks, "half_division": truth["half_division"] or "",
        "start_deg": spec.start_deg, "sweep_deg": spec.sweep_deg,
        "tilt_deg": round(truth["tilt_deg"], 2), "lighting": job["lighting"], "phone": job["phone"],
        "glare_frac": round(truth["glare_frac"], 4), "seed": job["seed"],
        # Where the dial is in the photo: ground truth for the Stage 9 detector only.
        "cx": round(truth["cx"], 1), "cy": round(truth["cy"], 1), "r_px": round(truth["r_px"], 1),
    }


if __name__ == "__main__":
    rows = Parallel(n_jobs=-1)(delayed(render)(j) for j in plan())
    with open(SYNTH / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    df = pd.DataFrame(rows)
    print(f"wrote {len(df)} photos to {SYNTH}")
    print(df.groupby(["set", "split"]).size().unstack(fill_value=0))

    # The dataset statistics table (image attributes per gauge type).
    stats = pd.DataFrame([dict(attrs.image_attributes(SYNTH / r["file"]), gauge=r["gauge"], set=r["set"],
                               split=r["split"], phone=r["phone"]) for r in rows])
    table = stats.groupby("gauge").agg(photos=("file", "count"), width=("width", "first"),
                                       height=("height", "first"), mean_grey=("mean", "mean"),
                                       std_grey=("std", "mean"), entropy_bits=("entropy_bits", "mean"),
                                       bright_fraction=("bright_fraction", "mean")).reset_index()
    save_table(table.round(3), "dataset_statistics")
    print(table.round(2).to_string(index=False))
