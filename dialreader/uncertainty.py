"""Stage 8 - Uncertainty and abstention.

Turns the diagnostics of Stages 2 to 7 into an error bar and a reliable /
unreliable flag. When any signal crosses its threshold the reading is reported
as unreliable instead of being passed off as a measurement.
"""
import json
from pathlib import Path

import numpy as np

# name: (default threshold, True if larger is worse, what it means when it fires)
SIGNALS = {
    "needle_spread":  (1.0,  True,  "needle angle is uncertain"),
    "method_gap":     (1.5,  True,  "line fit and Hough line disagree on the needle"),
    "rim_support":    (0.55, False, "dial rim only partly found"),
    "rim_residual":   (1.2,  True,  "dial rim does not fit an ellipse well"),
    "comb_residual":  (0.12, True,  "tick marks are not evenly spaced"),
    "comb_missing":   (0.15, True,  "many tick marks are missing"),
    "dft_gap":        (0.06, True,  "tick spacing from the DFT disagrees with the comb"),
    "tick_snr":       (6.0,  False, "tick marks are faint"),
    "glare_overlap":  (0.25, True,  "glare covers the needle"),
    "tick_count":     (0.5,  True,  "tick count does not match the scale"),
    "out_of_range":   (1.0,  True,  "needle is outside the scale"),
    "sentinel":       (1.0,  True,  "VLM coarse reading disagrees (gross-error sentinel)"),
}

THRESHOLD_FILE = Path(__file__).resolve().parent.parent / "results" / "thresholds.json"


def default_thresholds():
    return {name: spec[0] for name, spec in SIGNALS.items()}


def load_thresholds(path=THRESHOLD_FILE):
    """Thresholds tuned on the validation split if present, otherwise the defaults."""
    thr = default_thresholds()
    if Path(path).exists():
        thr.update(json.loads(Path(path).read_text()))
    return thr


def badness(signals, thresholds):
    """Each signal divided by its threshold, oriented so that a value above 1 means 'crossed'."""
    out = {}
    for name, value in signals.items():
        if name not in SIGNALS or value is None or not np.isfinite(value):
            continue
        thr, larger_is_worse = thresholds.get(name, SIGNALS[name][0]), SIGNALS[name][1]
        out[name] = float(value / thr) if larger_is_worse else float(thr / max(value, 1e-9))
    return out


def assess(signals, thresholds=None):
    """Returns (status, reasons, badness, confidence in 0..1). Confidence is 0.5 exactly at a threshold."""
    thresholds = thresholds or default_thresholds()
    bad = badness(signals, thresholds)
    worst = max(bad.values()) if bad else 0.0
    reasons = [SIGNALS[k][2] for k, v in sorted(bad.items(), key=lambda kv: -kv[1]) if v > 1.0]
    confidence = float(0.5 ** (worst ** 2))
    return ("reliable" if not reasons else "unreliable"), reasons, bad, confidence


def reading_uncertainty(slope, sigma_theta, division):
    """u_v = |dv/dtheta| * sigma_theta + half of the smallest division."""
    return abs(slope) * sigma_theta + 0.5 * division


def tune_thresholds(rows, tolerance_fs=2.0, quantile=0.99, margin=1.5):
    """Set each threshold from the validation split.

    For every signal, take the readings that were correct (error within
    `tolerance_fs` % of full scale) and put the threshold just outside the range
    those correct readings occupy. Signals with too few examples keep their default.
    """
    thr = default_thresholds()
    good = [r for r in rows if r.get("error_fs") is not None and r["error_fs"] <= tolerance_fs]
    for name, (default, larger_is_worse, _) in SIGNALS.items():
        if name in ("tick_count", "out_of_range", "sentinel"):
            continue                      # logical checks, not tuned
        vals = np.array([r["signals"][name] for r in good
                         if r["signals"].get(name) is not None and np.isfinite(r["signals"][name])])
        if len(vals) < 20:
            continue
        if larger_is_worse:
            thr[name] = float(max(np.quantile(vals, quantile) * margin, 0.25 * default))
        else:
            thr[name] = float(min(np.quantile(vals, 1 - quantile) / margin, 4.0 * default))
    return thr
