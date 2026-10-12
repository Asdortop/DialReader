"""E2 - How does the error change with tilt, glare, blur, noise and JPEG quality?

One factor is varied at a time. Tilt and glare come from the variation set
(photographed that way); the other factors are synthetic degradations of clean
test photos. A VLM-alone baseline is added when a VLM key is configured.

Run:  python experiments/e2_robustness.py [--vlm]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, GROSS_FS, TABLES, load_synth, run, save_table, summarise
from dialreader import degrade, viz


class Degrade:
    """Picklable 'apply this factor at this level' transform."""

    def __init__(self, factor, level):
        self.factor, self.level = factor, level

    def __call__(self, img):
        return degrade.SWEEPS[self.factor][2](img, self.level, 7)


def row(factor, label, level, rows):
    s = summarise(rows)
    ok = [r for r in rows if r["error_fs"] is not None]
    correct = sum(bool(r["within_division"]) for r in ok)
    return {"factor": factor, "axis": label, "level": level, "n": s["n"], "answered %": s["answered_%"],
            "correct % (within 1 division)": 100.0 * correct / s["n"], "coverage % (reliable)": s["coverage_%"],
            "mean error (% FS)": s["mean_err"], "median error": s["median_err"], "90th pct": s["p90_err"],
            f"gross error % (>{GROSS_FS:.0f}% FS)": s["gross_%"], "gross error % among reliable": s["risk_%"]}


def classical():
    out = []
    base = load_synth(split="test", subset="base")
    for factor, (label, levels, _) in degrade.SWEEPS.items():
        for level in levels:
            out.append(row(factor, label, level, run(base, transform=Degrade(factor, level))))
            print(factor, level, {k: round(v, 2) for k, v in out[-1].items() if isinstance(v, float)})
    variation = load_synth(split="test", subset="variation")
    done = {r["file"]: r for r in run(variation)}
    for tilt in (0.0, 20.0, 40.0):
        rows = [done[r["file"]] for r in variation if r["lighting"] == "room" and abs(r["tilt_deg"] - tilt) < 1]
        out.append(row("tilt", "Camera tilt (degrees), room light", tilt, rows))
    for i, lighting in enumerate(("room", "window")):
        rows = [done[r["file"]] for r in variation if r["lighting"] == lighting]
        out.append(row("glare", "Lighting (all tilts)", lighting, rows))
    return pd.DataFrame(out)


def _key(level):
    """Levels survive a CSV round trip as strings; compare them as numbers where possible."""
    try:
        return float(level)
    except (TypeError, ValueError):
        return str(level)


def _label(level):
    k = _key(level)
    return f"{k:g}" if isinstance(k, float) else k


def plot(df, vlm=None):
    factors = list(dict.fromkeys(df["factor"]))
    specs = [("e2_robustness.png", "E2: robustness, one factor at a time (share of photos read correctly)",
              "% of images", (-4, 104),
              [("correct % (within 1 division)", "-o", viz.SERIES[0], "Classical: read within 1 division", df),
               ("coverage % (reliable)", "--s", viz.SERIES[2], "Classical: flagged reliable", df),
               ("within 5% FS %", "-^", viz.SERIES[1], "VLM alone: within 5% of full scale", vlm),
               ("correct % (within 1 division)", ":v", viz.SERIES[3], "VLM alone: read within 1 division", vlm)]),
             ("e2_robustness_error.png", "E2: mean error of the readings that were given", "mean error (% FS)", None,
              [("mean error (% FS)", "-o", viz.SERIES[0], "Classical pipeline (answered photos)", df),
               ("mean error (% FS)", "-^", viz.SERIES[1], "VLM alone", vlm)])]
    for filename, title, ylabel, ylim, series in specs:
        fig, axes = plt.subplots(3, 3, figsize=(13, 9.6))
        for ax, factor in zip(axes.ravel(), factors):
            part = df[df["factor"] == factor]
            order = [_key(v) for v in part["level"]]
            for column, style, colour, label, data in series:
                if data is None or factor not in set(data["factor"]):
                    continue
                d = data[data["factor"] == factor]
                points = sorted((order.index(_key(v)), y) for v, y in zip(d["level"], d[column]) if _key(v) in order)
                ax.plot([p for p, _ in points], [y for _, y in points], style, color=colour, label=label,
                        lw=2.0 if style[0] == "-" and style[1] != "-" else 1.5)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([_label(v) for v in part["level"]])
            ax.set(title=part["axis"].iloc[0], ylabel=ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            else:
                ax.set_ylim(bottom=0)
        handles, labels = {}, []
        for ax in axes.ravel():
            for h, l in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(l, h)
        fig.legend(handles.values(), handles.keys(), loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, fontsize=9)
        fig.suptitle(title, x=0.01, y=0.985, ha="left", fontweight="semibold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(FIGS / filename)
        plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        df = pd.read_csv(TABLES / "e2_robustness.csv")
    else:
        df = save_table(classical().round(2), "e2_robustness")
    vlm = None
    if "--vlm" in sys.argv:
        from e2_vlm_baseline import vlm_sweep
        vlm = save_table(vlm_sweep().round(2), "e2_robustness_vlm")
    elif (TABLES / "e2_robustness_vlm.csv").exists():
        vlm = pd.read_csv(TABLES / "e2_robustness_vlm.csv")
    plot(df, vlm)
    print(df.to_string(index=False))
