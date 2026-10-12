"""E5 - Does the pipeline generalise to unseen gauges?

Two gauge types were held out entirely: none of their photos were used to tune
the thresholds. This reports the error per gauge type, held-out types marked.

Run:  python experiments/e5_generalisation.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, GROSS_FS, fmt_ci, load_synth, run, save_table, summarise
from dialreader import viz

if __name__ == "__main__":
    test = load_synth(split="test", subset="base")
    rows = run(test)
    by_file = {r["file"]: r for r in rows}
    out = []
    for gauge in dict.fromkeys(r["gauge"] for r in test):
        recs = [r for r in test if r["gauge"] == gauge]
        s = summarise([by_file[r["file"]] for r in recs])
        out.append({"gauge type": gauge, "held out": "yes" if int(recs[0]["held_out"]) else "no", "n": s["n"],
                    "answered %": s["answered_%"], "mean error (% FS)": s["mean_err"],
                    "95% CI": fmt_ci(s["mean_ci"], 3), "90th pct": s["p90_err"],
                    f"gross error % (>{GROSS_FS:.0f}% FS)": s["gross_%"],
                    "within 1 division %": s["within_div_%"], "coverage % (reliable)": s["coverage_%"]})
    for name, flag in (("ALL SEEN TYPES", "0"), ("ALL HELD-OUT TYPES", "1")):
        s = summarise([by_file[r["file"]] for r in test if str(r["held_out"]) == flag])
        out.append({"gauge type": name, "held out": "yes" if flag == "1" else "no", "n": s["n"],
                    "answered %": s["answered_%"], "mean error (% FS)": s["mean_err"],
                    "95% CI": fmt_ci(s["mean_ci"], 3), "90th pct": s["p90_err"],
                    f"gross error % (>{GROSS_FS:.0f}% FS)": s["gross_%"],
                    "within 1 division %": s["within_div_%"], "coverage % (reliable)": s["coverage_%"]})
    df = save_table(pd.DataFrame(out).round(3), "e5_generalisation")
    print(df.to_string(index=False))

    part = df[~df["gauge type"].str.startswith("ALL")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    # Bars are placed by row number, so each label is tied to its own bar whatever the colour.
    for flag, colour, label in (("no", viz.SERIES[0], "seen during threshold tuning"),
                                ("yes", viz.SERIES[1], "held-out gauge type")):
        sel = part[part["held out"] == flag]
        ax.barh(sel.index, sel["mean error (% FS)"], height=0.5, color=colour, label=label)
    for i, r in part.iterrows():
        ax.text(r["mean error (% FS)"], i, f"  {r['mean error (% FS)']:.3f}", va="center", fontsize=9, color=viz.INK2)
    ax.set(xlabel="mean error on clean test photos (% of full scale)",
           title="E5: error per gauge type, held-out types included",
           xlim=(0, max(0.05, part["mean error (% FS)"].max() * 1.35)))
    ax.set_yticks(part.index)
    ax.set_yticklabels(part["gauge type"].str.replace("_", " "))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "e5_generalisation.png")
