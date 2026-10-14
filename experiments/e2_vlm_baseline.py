"""E2, VLM-alone baseline: the same robustness sweeps, read by the VLM with no classical help.

Kept small to respect the API budget: one photo per gauge type, and three
degraded levels per factor besides the clean one. Tilt and glare reuse the
answers already cached by the sentinel experiment.

Run through:  python experiments/e2_robustness.py --vlm
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import error_fs, load_synth
from dialreader import degrade, attributes as attrs
from dialreader.hooks import sentinel
from dialreader.hooks.vlm import VLM


def _score(value, rec):
    if value is None:
        return None, False
    half = rec.get("half_division")
    within = bool(half not in (None, "") and abs(value - float(rec["value"])) <= 2 * float(half))
    return error_fs(value, rec), within


def vlm_sweep():
    vlm = VLM()
    base = load_synth(split="test", subset="base")
    picked = list({r["gauge"]: r for r in reversed(base)}.values())      # one photo per gauge type
    jobs = []
    for factor, (label, levels, fn) in degrade.SWEEPS.items():
        hard = [lv for lv in levels if lv != degrade.CLEAN[factor]]
        for level in [degrade.CLEAN[factor]] + [hard[i] for i in (0, len(hard) // 2, len(hard) - 1)]:
            jobs += [(factor, label, level, rec, fn) for rec in picked]
    variation = load_synth(split="test", subset="variation")
    for tilt in (0.0, 20.0, 40.0):
        jobs += [("tilt", "Camera tilt (degrees), room light", tilt, r, None) for r in variation
                 if r["lighting"] == "room" and abs(r["tilt_deg"] - tilt) < 1]
    for lighting in ("room", "window"):
        jobs += [("glare", "Lighting (all tilts)", lighting, r, None) for r in variation if r["lighting"] == lighting]

    def one(job):
        factor, label, level, rec, fn = job
        img = attrs.read_image(rec["path"])
        if fn is not None:
            img = fn(img, level, 7)
        try:
            value, _ = sentinel.coarse_reading(vlm, img, tag="e2_vlm_alone")
        except Exception:
            value = None
        return _score(value, rec)

    with ThreadPoolExecutor(8) as pool:
        scores = list(pool.map(one, jobs))
    rows = []
    for factor, label, level in dict.fromkeys((j[0], j[1], j[2]) for j in jobs):
        part = [s for j, s in zip(jobs, scores) if (j[0], j[2]) == (factor, level)]
        errs = np.array([e for e, _ in part if e is not None])
        rows.append({"factor": factor, "axis": label, "level": level, "n": len(part),
                     "answered %": 100.0 * len(errs) / len(part),
                     "correct % (within 1 division)": 100.0 * sum(w for _, w in part) / len(part),
                     "within 5% FS %": 100.0 * float(np.sum(errs <= 5.0)) / len(part),
                     "mean error (% FS)": float(errs.mean()) if len(errs) else np.nan,
                     "gross error % (>10% FS)": 100.0 * float(np.mean(errs > 10.0)) if len(errs) else np.nan})
    print(f"VLM baseline: {vlm.calls} new calls, {vlm.cached_calls} cached, total spend ${vlm.spent_total():.3f}")
    return pd.DataFrame(rows)
