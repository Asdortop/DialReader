"""Shared helpers for the experiment scripts: paths, evaluation, metrics, tables."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dialreader import pipeline, uncertainty as unc, w02_attributes as attrs
from dialreader.synth import GaugeSpec

DATA = ROOT / "data"
SYNTH = DATA / "synthetic"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"
for d in (FIGS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

GROSS_FS = 10.0     # an error above 10 % of full scale is a gross error
N_JOBS = -1


def scale_fraction(value, rec):
    """Position along the scale (0..1); on a log scale this is linear in log(value)."""
    vmin, vmax = float(rec["vmin"]), float(rec["vmax"])
    if (rec.get("scale_type") or "linear") == "log":
        return float(np.log(max(value, 1e-12) / vmin) / np.log(vmax / vmin))
    return float((value - vmin) / (vmax - vmin))


def error_fs(pred, rec):
    """e = |v_pred - v_true| / (v_max - v_min) * 100 %, measured along the scale.

    Capped at 100: a reading extrapolated far outside the scale is simply wrong by the
    full scale, and should not dominate a mean.
    """
    return min(100.0, abs(scale_fraction(pred, rec) - scale_fraction(float(rec["value"]), rec)) * 100.0)


def evaluate_image(img, rec, params=None, thresholds=None, scale=None):
    """Run the pipeline on one image and score it against the record's ground truth."""
    scale = scale or attrs.scale_of(rec)
    for key in ("start_deg", "sweep_deg"):
        if rec.get(key) not in (None, ""):
            scale.setdefault(key, float(rec[key]))
    try:
        R = pipeline.read_dial(img, scale, params, thresholds, keep_images=False)
    except Exception as exc:                       # a crash counts as a failed reading
        R = pipeline.Reading(reasons=[f"error: {exc!r}"])
    row = {"file": rec["file"], "gauge": rec["gauge"], "true": float(rec["value"]), "pred": R.value,
           "status": R.status, "confidence": R.confidence, "uncertainty": R.uncertainty,
           "reasons": "; ".join(R.reasons), "signals": dict(R.signals), "time": R.timings.get("total", 0.0),
           "error_fs": None, "within_division": None, "within_bar": None}
    if R.ok:
        row["error_fs"] = error_fs(R.value, rec)
        half = rec.get("half_division")
        tol = 2.0 * float(half) if half not in (None, "") else R.info["calibration"]["division"]
        row["within_division"] = bool(abs(R.value - float(rec["value"])) <= tol)
        row["within_bar"] = bool(abs(R.value - float(rec["value"])) <= R.uncertainty)
    return row


def _eval_record(rec, params, thresholds, transform, scale):
    img = attrs.read_image(rec["path"])
    if transform is not None:
        img = transform(img)
    return evaluate_image(img, rec, params, thresholds, dict(scale) if scale else None)


def run(records, params=None, thresholds=None, transform=None, scales=None):
    """Evaluate the pipeline over dataset records, in parallel. Returns a list of row dicts.

    scales: optional {gauge type: scale dict} to use instead of the hand-entered scale
            of the labels (for scales parsed by GenAI hook 1).
    """
    thresholds = thresholds or unc.load_thresholds()
    scales = scales or {}
    return Parallel(n_jobs=N_JOBS)(delayed(_eval_record)(r, params, thresholds, transform, scales.get(r["gauge"]))
                                   for r in records)


def coverage_at_risk(rows, max_risk=0.02, score_key="confidence"):
    """Largest share of images that can be answered while the gross-error rate stays within max_risk."""
    cov, risk = risk_coverage(rows, score_key)
    ok = np.nonzero(risk <= max_risk)[0]
    return float(cov[ok[-1]]) if len(ok) else 0.0


def bootstrap_ci(values, stat=np.mean, n=2000, seed=0):
    """95 % percentile bootstrap confidence interval of a statistic."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), (n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarise(rows, reliable_only=False):
    """The metrics of the plan for a set of evaluated rows.

    Error statistics are over the images that were answered (a value came out).
    Coverage is the share flagged reliable; risk is the gross-error rate among those.
    """
    n = len(rows)
    answered = [r for r in rows if r["error_fs"] is not None]
    reliable = [r for r in answered if r["status"] == "reliable"]
    use = reliable if reliable_only else answered
    e = np.array([r["error_fs"] for r in use], float)
    out = {"n": n, "answered_%": 100.0 * len(answered) / max(n, 1), "coverage_%": 100.0 * len(reliable) / max(n, 1)}
    if len(e) == 0:
        out.update({"mean_err": np.nan, "median_err": np.nan, "p90_err": np.nan, "gross_%": np.nan,
                    "within_div_%": np.nan, "mean_ci": (np.nan, np.nan), "gross_ci": (np.nan, np.nan)})
    else:
        out.update({"mean_err": float(e.mean()), "median_err": float(np.median(e)),
                    "p90_err": float(np.percentile(e, 90)), "gross_%": float(100.0 * np.mean(e > GROSS_FS)),
                    "within_div_%": float(100.0 * np.mean([bool(r["within_division"]) for r in use])),
                    "mean_ci": bootstrap_ci(e), "gross_ci": bootstrap_ci(100.0 * (e > GROSS_FS))})
    er = np.array([r["error_fs"] for r in reliable], float)
    out["risk_%"] = float(100.0 * np.mean(er > GROSS_FS)) if len(er) else float("nan")
    out["reliable_mean_err"] = float(er.mean()) if len(er) else float("nan")
    out["bar_coverage_%"] = (float(100.0 * np.mean([bool(r["within_bar"]) for r in answered]))
                             if answered else float("nan"))
    out["time_s"] = float(np.mean([r["time"] for r in rows])) if rows else float("nan")
    return out


def risk_coverage(rows, score_key="confidence"):
    """Risk-coverage curve: answer the most confident images first.

    Returns (coverage, risk) arrays, where risk is the gross-error rate among the
    answered share. Images with no reading are never answered, so coverage tops out
    below 1 when the pipeline failed on some.
    """
    scored = sorted([r for r in rows if r["error_fs"] is not None], key=lambda r: -r[score_key])
    gross = np.array([r["error_fs"] > GROSS_FS for r in scored], float)
    k = np.arange(1, len(scored) + 1)
    return k / max(len(rows), 1), np.cumsum(gross) / k


def fmt_ci(ci, digits=2):
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def to_markdown(df):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.2f}"
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join(" --- " for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + body) + "\n"


def save_table(df, name):
    """Write a table as CSV (for reuse) and Markdown (for reading)."""
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")
    return df


def load_synth(split=None, subset=None):
    records = attrs.load_dataset(SYNTH)
    if split:
        records = [r for r in records if r["split"] == split]
    if subset:
        records = [r for r in records if r["set"] == subset]
    return records
