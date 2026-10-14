"""E6 - How does it compare with VLMs on public data?

Runs on a hand-filtered subset of the public MeasureBench real-world split
(single-needle circular dials with one scale; see data/measurebench/subset.csv).
MeasureBench scores a reading as correct when it falls inside a valid interval,
so that is the primary score here; the numeric distance to the interval is
reported too.

Run:  python experiments/e6_measurebench.py
"""
import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, FIGS, RESULTS, save_table
from dialreader import pipeline, uncertainty as unc, viz, attributes as attrs
from dialreader.hooks import scale_parser, sentinel
from dialreader.hooks.vlm import VLM

MB = DATA / "measurebench"
# Published reference points (MeasureBench paper, real-world images, 2025 models; from the project plan).
PUBLISHED = {"best of 18 VLMs, overall accuracy on real images": 30.2,
             "value accuracy on real images (best model)": 30.7,
             "Reitsma et al. 2024 pipeline, overall accuracy": 8.5,
             "Shu et al. 2023 pipeline, value accuracy on real images": 4.2}


def in_interval(value, interval):
    return value is not None and interval[0] <= value <= interval[1]


def distance_fs(value, interval, scale):
    """Distance from the reading to the valid interval, in % of full scale (0 when inside)."""
    if value is None:
        return None
    gap = max(interval[0] - value, value - interval[1], 0.0)
    return min(100.0, 100.0 * gap / (scale["vmax"] - scale["vmin"]))


if __name__ == "__main__":
    index = {r["question_id"]: r for r in csv.DictReader(open(MB / "dial" / "index.csv", encoding="utf-8"))}
    subset = list(csv.DictReader(open(MB / "subset.csv", encoding="utf-8")))
    vlm = VLM()
    thresholds = unc.load_thresholds()
    summary_path = RESULTS / "genai_summary.json"
    thr_fs = json.loads(summary_path.read_text())["sentinel_threshold_fs"] if summary_path.exists() \
        else sentinel.DEFAULT_THRESHOLD_FS

    rows, tiles = [], []
    for s in subset:
        rec = index[s["question_id"]]
        interval = json.loads(rec["interval"])
        img = attrs.read_image(MB / "dial" / rec["file"])
        scale = {"vmin": float(s["vmin"]), "vmax": float(s["vmax"]), "unit": s["unit"], "scale_type": "linear",
                 "direction": "clockwise"}
        R = pipeline.read_dial(img, scale, thresholds=thresholds)
        row = {"image": s["question_id"], "valid interval": f"{interval[0]} to {interval[1]}",
               "scale entered by hand": f"{s['vmin']} to {s['vmax']} {s['unit']}".strip(),
               "classical reading": None if not R.ok else round(R.value, 4),
               "error bar": None if not R.ok else round(R.uncertainty, 4), "status": R.status,
               "classical in interval": in_interval(R.value, interval),
               "classical distance to interval (% FS)": distance_fs(R.value, interval, scale),
               "why": "; ".join(R.reasons)[:90]}
        if vlm.available:
            v, _ = sentinel.coarse_reading(vlm, img, tag="e6_vlm_alone")
            row.update({"VLM alone": v, "VLM in interval": in_interval(v, interval),
                        "VLM distance to interval (% FS)": distance_fs(v, interval, scale)})
            gap = sentinel.check(v, R, scale, thr_fs)
            row["sentinel flags it"] = bool(gap is not None and gap > 1.0)
            # Hook 1 in place of the hand-entered scale.
            parse, _ = scale_parser.parse_scale(vlm, R.images["rectified"], "e6_scale_parser") \
                if "rectified" in R.images and R.info.get("ticks", {}).get("angles") is not None else (None, None)
            parsed, report = scale_parser.validate(parse, R.info["ticks"]) if parse else (None, {"reason": "no ticks"})
            row["hook 1 parse"] = "accepted" if parsed else f"rejected: {report['reason']}"
            row["reading with parsed scale"] = None
            row["parsed-scale reading in interval"] = False
            if parsed:
                R1 = pipeline.read_dial(img, parsed, thresholds=thresholds, keep_images=False)
                row["reading with parsed scale"] = None if not R1.ok else round(R1.value, 4)
                row["parsed-scale reading in interval"] = in_interval(R1.value, interval)
        rows.append(row)
        tile = viz.overlay_rectified(R) if "strip" in R.images else cv2.resize(R.images.get("input", img), (640, 640))
        mark = viz.bgr(viz.GOOD) if row["classical in interval"] else viz.bgr(viz.CRIT)
        tile = cv2.copyMakeBorder(cv2.resize(tile, (300, 300)), 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=mark)
        tiles.append((tile, f"{s['question_id'].replace('pressure_gauge', 'pg')}: {R.status}"))
        print(row)

    df = save_table(pd.DataFrame(rows), "e6_measurebench_per_image")
    n = len(df)
    reliable = df[df["status"] == "reliable"]
    summary = [{"method": "DialReader classical (C0), scale entered by hand", "images": n,
                "answered %": 100 * df["classical reading"].notna().mean(),
                "interval accuracy % (all images)": 100 * df["classical in interval"].mean(),
                "flagged reliable %": 100 * len(reliable) / n,
                "interval accuracy % among reliable": 100 * reliable["classical in interval"].mean() if len(reliable) else np.nan,
                "median distance to interval (% FS), answered": df["classical distance to interval (% FS)"].median()}]
    if "VLM alone" in df:
        summary.append({"method": f"VLM alone ({vlm.model})", "images": n,
                        "answered %": 100 * df["VLM alone"].notna().mean(),
                        "interval accuracy % (all images)": 100 * df["VLM in interval"].mean(),
                        "median distance to interval (% FS), answered": df["VLM distance to interval (% FS)"].median()})
        summary.append({"method": "DialReader with hook 1 (scale parsed by the VLM, validated)", "images": n,
                        "answered %": 100 * df["reading with parsed scale"].notna().mean(),
                        "interval accuracy % (all images)": 100 * df["parsed-scale reading in interval"].mean()})
    for name, value in PUBLISHED.items():
        summary.append({"method": f"Published: {name}", "interval accuracy % (all images)": value})
    out = save_table(pd.DataFrame(summary).round(1), "e6_measurebench")
    print(out.to_string(index=False))

    cols = 7
    rows_n = int(np.ceil(len(tiles) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(2.3 * cols, 2.6 * rows_n))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (tile, title) in zip(axes.ravel(), tiles):
        ax.imshow(viz.rgb(tile))
        ax.set_title(title, fontsize=8)
    fig.suptitle("E6: MeasureBench single-needle dial subset. Green border: classical reading inside the valid "
                 "interval; red: outside or no reading", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "e6_measurebench.png")
