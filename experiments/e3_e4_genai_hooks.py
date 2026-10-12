"""E3 and E4 - Do the GenAI hooks help?

E3: the five-step ablation ladder
    C0  classical pipeline, scale entered by hand
    C1  C0 + hook 1 (scale parser, validated against the detected ticks)
    C2  C1 + hook 2 (gross-error sentinel)
    C3  C2 + hook 3 (repair loop)
    B   VLM alone
E4: risk-coverage curves with and without the sentinel.

Each hook is judged by a paired bootstrap against the step below it, and is
reported as not helping when the interval includes zero.

Needs a VLM key in .env. Every call is cached, so a rerun costs nothing.
Run:  python experiments/e3_e4_genai_hooks.py
"""
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (DATA, FIGS, GROSS_FS, RESULTS, coverage_at_risk, error_fs, load_synth, risk_coverage, run,
                    save_table)
from dialreader import pipeline, uncertainty as unc, viz, w02_attributes as attrs
from dialreader.hooks import repair, scale_parser, sentinel
from dialreader.hooks.vlm import VLM

CORRECT_FS = 2.0          # a reading within 2 % of full scale counts as correct
RISK_LIMIT = 0.02         # "coverage at fixed risk": gross-error rate held at 2 %
VLM_CONF = {"high": 0.9, "medium": 0.6, "low": 0.3}


def threaded(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


# ------------------------------------------------------------------ VLM calls

def vlm_readings(vlm, records, tag):
    """Hook 2 calls (one per image). The same answers serve as the VLM-alone baseline B."""
    def one(rec):
        try:
            value, call = sentinel.coarse_reading(vlm, attrs.read_image(rec["path"]), tag)
        except Exception as exc:
            return {"value": None, "confidence": None, "note": repr(exc)[:200]}
        res = call["result"] or {}
        return {"value": value, "confidence": res.get("confidence"), "scale_min": res.get("scale_min"),
                "scale_max": res.get("scale_max"), "unit": res.get("unit")}
    return dict(zip([r["file"] for r in records], threaded(one, records)))


def same_scale(scale, rec):
    """Does a parsed scale agree with the ground-truth scale of the record?"""
    if scale is None:
        return False
    tol = 0.01 * abs(float(rec["vmax"]) - float(rec["vmin"]))
    return (abs(scale["vmin"] - float(rec["vmin"])) <= tol and abs(scale["vmax"] - float(rec["vmax"])) <= tol
            and scale["scale_type"] == (rec.get("scale_type") or "linear")
            and scale["direction"] == (rec.get("direction") or "clockwise"))


def parse_gauge_types(vlm, records, per_type=3):
    """Hook 1: parse the scale of each gauge type from a few photos and validate each parse."""
    jobs = []
    for gauge in dict.fromkeys(r["gauge"] for r in records):
        jobs += [r for r in records if r["gauge"] == gauge and r["lighting"] == "room"][:per_type]

    def one(rec):
        # Stages 1 to 5 need no scale: run them with a placeholder to get the dial face and its ticks.
        R = pipeline.read_dial(attrs.read_image(rec["path"]), {"vmin": 0.0, "vmax": 1.0, "unit": ""})
        ticks = R.info.get("ticks", {})
        if "rectified" not in R.images or ticks.get("angles") is None:
            return {"gauge": rec["gauge"], "file": rec["file"], "accepted": False, "reason": "no ticks detected",
                    "scale": None, "correct": False, "raw_correct": False, "match_rate": 0.0}
        parse, _ = scale_parser.parse_scale(vlm, R.images["rectified"], tag="hook1_scale_parser")
        scale, report = scale_parser.validate(parse, ticks)
        raw = scale_parser.unvalidated_scale(parse)
        vlm.log({"event": "validation", "hook": 1, "file": rec["file"], "accepted": report["accepted"],
                 "reason": report["reason"], "checks": report["checks"]})
        return {"gauge": rec["gauge"], "file": rec["file"], "accepted": report["accepted"],
                "reason": report["reason"], "scale": scale, "raw_scale": raw,
                "correct": same_scale(scale, rec), "raw_correct": same_scale(raw, rec),
                "match_rate": report.get("match_rate", 0.0), "n_labels": report.get("n_labels", 0)}
    return threaded(one, jobs, workers=6)


# ------------------------------------------------------------------ scoring

def apply_sentinel(rows, records, vlm_out, thr_fs, thresholds):
    """C2: add the sentinel signal to each classical row and reassess its status."""
    by_file = {r["file"]: r for r in records}
    out = []
    for row in rows:
        row = dict(row, signals=dict(row["signals"]))
        v = (vlm_out.get(row["file"]) or {}).get("value")
        if row["error_fs"] is not None and v is not None:
            row["signals"]["sentinel"] = sentinel.disagreement_fs(v, row["pred"], attrs.scale_of(by_file[row["file"]])) / thr_fs
            row["status"], reasons, _, row["confidence"] = unc.assess(row["signals"], thresholds)
            row["reasons"] = "; ".join(reasons)
        out.append(row)
    return out


def baseline_rows(records, vlm_out):
    """B: the VLM's own reading, scored like any other."""
    rows = []
    for rec in records:
        got = vlm_out.get(rec["file"]) or {}
        v = got.get("value")
        row = {"file": rec["file"], "gauge": rec["gauge"], "true": float(rec["value"]), "pred": v,
               "status": "failed", "confidence": 0.0, "error_fs": None, "within_division": None}
        if v is not None:
            half = rec.get("half_division")
            row.update(status="reliable", confidence=VLM_CONF.get(got.get("confidence"), 0.5),
                       error_fs=error_fs(v, rec),
                       within_division=bool(half not in (None, "") and abs(v - float(rec["value"])) <= 2 * float(half)))
        rows.append(row)
    return rows


def metrics(rows, calls_per_image):
    n = len(rows)
    answered = [r for r in rows if r["error_fs"] is not None]
    reliable = [r for r in answered if r["status"] == "reliable"]
    e = np.array([r["error_fs"] for r in reliable], float)
    e_all = np.array([r["error_fs"] for r in answered], float)
    return {
        "n": n, "answered %": 100.0 * len(answered) / n, "coverage % (reliable)": 100.0 * len(reliable) / n,
        "mean error (% FS)": float(e.mean()) if len(e) else np.nan,
        "90th percentile": float(np.percentile(e, 90)) if len(e) else np.nan,
        "gross-error rate %": float(100.0 * np.mean(e > GROSS_FS)) if len(e) else np.nan,
        "gross-error rate %, all answered": float(100.0 * np.mean(e_all > GROSS_FS)) if len(e_all) else np.nan,
        f"coverage % at risk <= {100 * RISK_LIMIT:.0f}%": 100.0 * coverage_at_risk(rows, RISK_LIMIT),
        "useful readings % (reliable and correct)": 100.0 * sum(r["error_fs"] <= CORRECT_FS for r in reliable) / n,
        "silent failures % (reliable but gross)": 100.0 * sum(r["error_fs"] > GROSS_FS for r in reliable) / n,
        "VLM calls per image": calls_per_image,
    }


def paired_delta(rows_a, rows_b, n_boot=2000, seed=0):
    """Paired bootstrap of (b - a) for useful readings and silent failures, in % of all images."""
    def flags(rows):
        useful = np.array([r["error_fs"] is not None and r["status"] == "reliable" and r["error_fs"] <= CORRECT_FS
                           for r in rows], float)
        silent = np.array([r["error_fs"] is not None and r["status"] == "reliable" and r["error_fs"] > GROSS_FS
                           for r in rows], float)
        return useful, silent
    (ua, sa), (ub, sb) = flags(rows_a), flags(rows_b)
    idx = np.random.default_rng(seed).integers(0, len(rows_a), (n_boot, len(rows_a)))
    out = {}
    for name, d in (("useful", ub - ua), ("silent", sb - sa)):
        boot = 100.0 * d[idx].mean(axis=1)
        out[name] = (100.0 * d.mean(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return out


def verdict(delta):
    """Helpful only if an interval excludes zero: more useful readings or fewer silent failures.

    'mixed' when useful readings rise for certain but silent failures also appeared.
    """
    (du, ulo, uhi), (ds, slo, shi) = delta["useful"], delta["silent"]
    if uhi < 0 or slo > 0:
        return "hurts"
    if ulo > 0 and ds > 0:
        return "mixed"
    if ulo > 0 or shi < 0:
        return "helps"
    return "no measurable effect"


# ------------------------------------------------------------------ hook 3

def run_repairs(vlm, records, rows, vlm_out, scales, thr_fs, thresholds):
    """Repair loop on the flagged images, with the VLM chooser and both classical baselines."""
    by_file = {r["file"]: r for r in records}
    flagged = [r for r in rows if r["status"] != "reliable"]

    def one(row):
        rec = by_file[row["file"]]
        img = attrs.read_image(rec["path"])
        scale = dict(scales.get(rec["gauge"]) or attrs.scale_of(rec))
        v = (vlm_out.get(rec["file"]) or {}).get("value")

        def post(R):
            if R.ok and v is not None:
                R.signals["sentinel"] = sentinel.disagreement_fs(v, R.value, scale) / thr_fs
                R.status, R.reasons, R.badness, R.confidence = unc.assess(R.signals, thresholds)
            return R

        R0 = post(pipeline.read_dial(img, scale, None, thresholds))
        out = {"file": rec["file"], "gauge": rec["gauge"]}
        for name, fn in (("vlm", lambda: repair.repair_loop(img, scale, R0, repair.vlm_chooser(vlm, scale, "hook3_repair"),
                                                             thresholds, post=post)),
                         ("rule", lambda: repair.repair_loop(img, scale, R0, repair.rule_chooser(), thresholds, post=post)),
                         ("grid", lambda: repair.grid_search(img, scale, R0, thresholds, post=post))):
            R, trace = fn()
            err = error_fs(R.value, rec) if R.ok else None
            out[name] = {"status": R.status, "pred": R.value, "error_fs": err, "confidence": R.confidence,
                         "reruns": sum(1 for t in trace if t["action"] != "give_up"),
                         "actions": [t["action"] for t in trace],
                         "reasons": [t.get("reason", "") for t in trace]}
        vlm.log({"event": "validation", "hook": 3, "file": rec["file"], "actions": out["vlm"]["actions"],
                 "status_after": out["vlm"]["status"]})
        return out
    return threaded(one, flagged, workers=6)


def merge_repairs(rows, repairs, which):
    """Rows after the repair loop: flagged images take the repaired reading."""
    fixed = {r["file"]: r[which] for r in repairs}
    out = []
    for row in rows:
        if row["file"] in fixed:
            f = fixed[row["file"]]
            row = dict(row, pred=f["pred"], error_fs=f["error_fs"], status=f["status"], confidence=f["confidence"])
        out.append(row)
    return out


def repair_table(repairs, data_name):
    rows = []
    n = len(repairs)
    for name, label in (("none", "No repair"), ("vlm", "Hook 3: VLM picks the action (at most 2 reruns)"),
                        ("rule", "Rule-based, fixed order (at most 2 reruns)"),
                        ("grid", "Grid search over all actions (5 reruns)")):
        if name == "none":
            rows.append({"data": data_name, "method": label, "flagged images": n, "fixed (reliable and correct)": 0,
                         "fixed %": 0.0, "made reliable but wrong": 0, "pipeline reruns per image": 0.0,
                         "VLM calls per image": 0.0})
            continue
        res = [r[name] for r in repairs]
        fixed = sum(r["status"] == "reliable" and r["error_fs"] is not None and r["error_fs"] <= CORRECT_FS for r in res)
        wrong = sum(r["status"] == "reliable" and r["error_fs"] is not None and r["error_fs"] > GROSS_FS for r in res)
        reruns = float(np.mean([r["reruns"] for r in res])) if res else 0.0
        calls = float(np.mean([len(r["actions"]) for r in res])) if (res and name == "vlm") else 0.0
        rows.append({"data": data_name, "method": label, "flagged images": n, "fixed (reliable and correct)": fixed,
                     "fixed %": 100.0 * fixed / max(n, 1), "made reliable but wrong": wrong,
                     "pipeline reruns per image": reruns, "VLM calls per image": calls})
    return rows


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    vlm = VLM()
    if not vlm.available:
        raise SystemExit("No VLM key in .env: the GenAI experiments cannot run. The classical pipeline is unaffected.")
    thresholds = unc.load_thresholds()
    print(f"VLM: {vlm.model} via {vlm.provider}; spent so far ${vlm.spent_total():.3f} of ${vlm.budget:.2f}")

    # ---- thresholds are set on the validation split only
    val = load_synth(split="val")
    val_rows = run(val, thresholds=thresholds)
    val_vlm = vlm_readings(vlm, val, "hook2_sentinel_val")
    gaps = [sentinel.disagreement_fs(val_vlm[r["file"]]["value"], r["pred"], attrs.scale_of(rec))
            for r, rec in zip(val_rows, val)
            if r["error_fs"] is not None and r["error_fs"] <= CORRECT_FS and val_vlm[r["file"]]["value"] is not None]
    thr_fs = float(np.clip(np.percentile(gaps, 97.5) * 1.1, 5.0, 30.0))
    print(f"sentinel threshold from validation: {thr_fs:.1f} % FS "
          f"(median VLM-vs-geometry gap {np.median(gaps):.2f} %, n={len(gaps)})")

    # ---- hook 1, once per gauge type
    everything = load_synth()
    parses = parse_gauge_types(vlm, everything)
    scales = {}
    for p in parses:
        if p["accepted"] and p["gauge"] not in scales:
            scales[p["gauge"]] = p["scale"]
    types = list(dict.fromkeys(p["gauge"] for p in parses))
    hook1 = pd.DataFrame([{
        "gauge type": g,
        "parses tried": sum(p["gauge"] == g for p in parses),
        "accepted by validation": sum(p["gauge"] == g and p["accepted"] for p in parses),
        "accepted and correct": sum(p["gauge"] == g and p["accepted"] and p["correct"] for p in parses),
        "VLM answer correct without validation": sum(p["gauge"] == g and p["raw_correct"] for p in parses),
        "label-to-tick match rate %": 100.0 * np.mean([p["match_rate"] for p in parses if p["gauge"] == g]),
        "rejection reasons": "; ".join(sorted({p["reason"] for p in parses if p["gauge"] == g and p["reason"]})),
    } for g in types])
    save_table(hook1.round(1), "e3_hook1_scale_parser")
    print(hook1.to_string(index=False))
    n_parse = len(parses)
    accepted = sum(p["accepted"] for p in parses)
    wrong_accepted = sum(p["accepted"] and not p["correct"] for p in parses)
    hook1_summary = {"parses": n_parse, "accepted": accepted, "accepted_wrong": wrong_accepted,
                     "raw_correct": sum(p["raw_correct"] for p in parses),
                     "types_with_scale": len(scales), "types": len(types),
                     "mean_match_rate": float(np.mean([p["match_rate"] for p in parses]))}

    ladder, repair_rows, curves, sentinel_rows, summary = [], [], {}, [], {"sentinel_threshold_fs": thr_fs,
                                                                           "hook1": hook1_summary}
    datasets = {"Test split": load_synth(split="test"),
                "Stress set": attrs.load_dataset(DATA / "synthetic_stress")}
    for data_name, records in datasets.items():
        n = len(records)
        c0 = run(records, thresholds=thresholds)
        c1 = run(records, thresholds=thresholds, scales=scales)
        vlm_out = vlm_readings(vlm, records, "hook2_sentinel")
        c2 = apply_sentinel(c1, records, vlm_out, thr_fs, thresholds)
        repairs = run_repairs(vlm, records, c2, vlm_out, scales, thr_fs, thresholds)
        c3 = merge_repairs(c2, repairs, "vlm")
        c3_rule = merge_repairs(c2, repairs, "rule")
        base = baseline_rows(records, vlm_out)

        hook1_calls = n_parse / len(everything)             # hook 1 runs per gauge type, not per image
        repair_calls = sum(len(r["vlm"]["actions"]) for r in repairs) / n
        steps = [("C0", "Classical pipeline, scale entered by hand", c0, 0.0),
                 ("C1", "C0 + hook 1 (scale parser)", c1, hook1_calls),
                 ("C2", "C1 + hook 2 (sentinel flags)", c2, hook1_calls + 1.0),
                 ("C3", "C2 + hook 3 (repair loop)", c3, hook1_calls + 1.0 + repair_calls),
                 ("B", "VLM alone, as the baseline", base, 1.0)]
        prev = None
        for cid, label, rows, calls in steps:
            m = metrics(rows, round(calls, 2))
            entry = {"data": data_name, "id": cid, "what is switched on": label, **m, "vs previous step": ""}
            if prev is not None and cid != "B":
                d = paired_delta(prev, rows)
                entry["vs previous step"] = (f"{verdict(d)}: useful {d['useful'][0]:+.1f} "
                                             f"[{d['useful'][1]:+.1f}, {d['useful'][2]:+.1f}], silent failures "
                                             f"{d['silent'][0]:+.1f} [{d['silent'][1]:+.1f}, {d['silent'][2]:+.1f}]")
            ladder.append(entry)
            if cid != "B":
                prev = rows
        d = paired_delta(c3_rule, c3)
        summary[f"{data_name}: hook 3 vs rule-based"] = {"verdict": verdict(d), **{k: list(v) for k, v in d.items()}}
        repair_rows += repair_table(repairs, data_name)

        # Hook 2 on its own terms: gross errors caught against false alarms.
        answered = [(a, b) for a, b in zip(c1, c2) if a["error_fs"] is not None and "sentinel" in b["signals"]]
        gross = [(a, b) for a, b in answered if a["error_fs"] > GROSS_FS]
        fine = [(a, b) for a, b in answered if a["error_fs"] <= CORRECT_FS]
        missed_by_stage8 = [(a, b) for a, b in gross if a["status"] == "reliable"]
        sentinel_rows.append({
            "data": data_name, "images with both readings": len(answered),
            "classical gross errors": len(gross),
            "gross errors flagged by sentinel": sum(b["signals"]["sentinel"] > 1 for _, b in gross),
            "recall %": 100.0 * np.mean([b["signals"]["sentinel"] > 1 for _, b in gross]) if gross else np.nan,
            "gross errors Stage 8 had missed": len(missed_by_stage8),
            "of those, caught by sentinel": sum(b["signals"]["sentinel"] > 1 for _, b in missed_by_stage8),
            "false alarms on correct readings %": 100.0 * np.mean([b["signals"]["sentinel"] > 1 for _, b in fine]) if fine else np.nan,
            "VLM alone mean error (% FS)": float(np.mean([r["error_fs"] for r in base if r["error_fs"] is not None])),
        })
        curves[data_name] = {"Stage 8 confidence only": risk_coverage(c1), "Stage 8 + sentinel": risk_coverage(c2),
                             "VLM alone (its own confidence)": risk_coverage(base)}
        print(f"{data_name}: done; spent so far ${vlm.spent_total():.3f}")

    ladder_df = save_table(pd.DataFrame(ladder).round(2), "e3_genai_ladder")
    save_table(pd.DataFrame(repair_rows).round(2), "e3_hook3_repair")
    save_table(pd.DataFrame(sentinel_rows).round(2), "e4_sentinel")
    summary["vlm_model"] = vlm.model
    summary["vlm_spend_usd"] = round(vlm.spent_total(), 4)
    attrs.save_json(summary, RESULTS / "genai_summary.json")
    print(ladder_df.drop(columns=["what is switched on"]).to_string(index=False))
    print(pd.DataFrame(repair_rows).to_string(index=False))
    print(pd.DataFrame(sentinel_rows).T.to_string())

    # ---- E4 figure: risk-coverage with and without the sentinel
    fig, axes = plt.subplots(1, len(curves), figsize=(6.2 * len(curves), 4.0), sharey=True)
    for ax, (data_name, series) in zip(np.atleast_1d(axes), curves.items()):
        for (label, (cov, risk)), colour, style in zip(series.items(), viz.SERIES, ("-", "--", ":")):
            ax.plot(100 * cov, 100 * risk, style, color=colour, label=label, drawstyle="steps-post")
        ax.set(title=data_name, xlabel="coverage: share of images answered (%)", xlim=(0, 100))
        ax.axhline(100 * RISK_LIMIT, color=viz.MUTED, lw=1)
        ax.text(1, 100 * RISK_LIMIT, f" risk limit {100 * RISK_LIMIT:.0f}%", va="bottom", fontsize=9, color=viz.INK2)
    np.atleast_1d(axes)[0].set_ylabel(f"risk: gross-error rate among answered (%)")
    np.atleast_1d(axes)[-1].legend(loc="upper left")
    fig.suptitle("E4: risk against coverage, with and without the sentinel", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "e4_risk_coverage.png")
    print(f"total VLM spend: ${vlm.spent_total():.3f}")
