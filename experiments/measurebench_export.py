"""Export the dial images of the public MeasureBench real-world split, with contact sheets
for hand filtering to single-needle circular dials.

MeasureBench (FlagEval/MeasureBench on Hugging Face, CC BY-SA 4.0) is downloaded data:
this script only decodes images and reads the label columns.

Run:  python -I experiments/measurebench_export.py
"""
import csv
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "measurebench" / "raw" / "data" / "real_world-00000-of-00001.parquet"
OUT = ROOT / "data" / "measurebench" / "dial"
TYPES = ("pressure_gauge", "thermometer", "hygrometer", "weighing_scale", "sphygmomanometer", "tachometer",
         "speedometer", "fuel_gauge", "voltmeter", "ammeter")

if __name__ == "__main__":
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    table = pq.read_table(RAW).to_pylist()
    rows, thumbs = [], []
    for r in table:
        if r["design"] != "dial" or r["image_type"] not in TYPES:
            continue
        img = cv2.imdecode(np.frombuffer(r["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        s = 1024.0 / max(img.shape[:2])
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"{r['question_id']}.jpg"
        cv2.imwrite(str(OUT / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        kw = json.loads(r["evaluator_kwargs"])
        rows.append({"file": f"images/{name}", "question_id": r["question_id"], "image_type": r["image_type"],
                     "question": r["question"], "evaluator": r["evaluator"],
                     "interval": json.dumps(kw.get("interval") or kw.get("intervals")),
                     "units": json.dumps(kw.get("units")), "source": (r["meta_info"] or {}).get("source", "")})
        t = cv2.resize(img, (236, 236))
        cv2.putText(t, r["question_id"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, r["question_id"], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append((r["image_type"], t))
    with open(OUT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "dial images exported")
    (OUT / "sheets").mkdir(exist_ok=True)
    per_sheet, cols = 40, 8
    for i in range(0, len(thumbs), per_sheet):
        chunk = [t for _, t in thumbs[i:i + per_sheet]]
        while len(chunk) % cols:
            chunk.append(np.zeros_like(chunk[0]))
        sheet = np.vstack([np.hstack(chunk[j:j + cols]) for j in range(0, len(chunk), cols)])
        cv2.imwrite(str(OUT / "sheets" / f"sheet_{i // per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    counts = {}
    for t, _ in thumbs:
        counts[t] = counts.get(t, 0) + 1
    print(counts)
