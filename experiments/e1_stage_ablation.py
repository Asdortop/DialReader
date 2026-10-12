"""E1 - Does each classical stage matter?

Switch off one stage at a time and tabulate the error, on the test split and on a
degraded copy of it (blur plus noise), where restoration has something to do.

Run:  python experiments/e1_stage_ablation.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import FIGS, GROSS_FS, fmt_ci, load_synth, run, save_table, summarise
from dialreader import degrade, viz
from dialreader.pipeline import Params

CONFIGS = [
    ("Full pipeline", Params()),
    ("Stage 1 off (no restoration)", Params(use_restore=False)),
    ("Stage 3 off (no tilt correction)", Params(use_tilt=False)),
    ("Stage 4 off (no polar unwrap)", Params(use_unwrap=False)),
    ("Stage 5 check off (no DFT / comb fit)", Params(use_comb=False)),
]


def blur_and_noise(img):
    return degrade.gaussian_noise(degrade.gaussian_blur(img, 2.0), 12.0, seed=1)


if __name__ == "__main__":
    test = load_synth(split="test")
    out = []
    for data_name, transform in (("Test split", None), ("Test split + blur 2 px + noise 12", blur_and_noise)):
        for name, params in CONFIGS:
            rows = run(test, params, transform=transform)
            s = summarise(rows)
            out.append({"data": data_name, "configuration": name, "n": s["n"], "answered %": s["answered_%"],
                        "mean error (% FS)": s["mean_err"], "95% CI": fmt_ci(s["mean_ci"]),
                        "median": s["median_err"], "90th pct": s["p90_err"],
                        f"gross error % (>{GROSS_FS:.0f}% FS)": s["gross_%"],
                        "within 1 division %": s["within_div_%"], "coverage % (reliable)": s["coverage_%"],
                        "gross error % among reliable": s["risk_%"]})
            if name == "Full pipeline":
                # Stage 8 off: every answered image counts, flagged or not.
                out.append({**out[-1], "configuration": "Stage 8 off (no abstention)",
                            "coverage % (reliable)": s["answered_%"], "gross error % among reliable": s["gross_%"]})
            print(data_name, "|", name, "| mean", round(s["mean_err"], 2), "gross", round(s["gross_%"], 1),
                  "coverage", round(s["coverage_%"], 1), "risk", round(s["risk_%"], 2))
    df = save_table(pd.DataFrame(out).round(2), "e1_stage_ablation")

    gross_col = f"gross error % (>{GROSS_FS:.0f}% FS)"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (data_name, part) in zip(axes, df.groupby("data", sort=False)):
        part = part[part["configuration"] != "Stage 8 off (no abstention)"]
        bars = ax.barh(part["configuration"], part[gross_col], height=0.5, color=viz.SERIES[0])
        for bar, v in zip(bars, part[gross_col]):
            ax.text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9,
                    color=viz.INK2)
        ax.set(title=data_name, xlabel="gross-error rate among answered images (%)",
               xlim=(0, max(10.0, df[gross_col].max() * 1.2)))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()                 # shared axis: full pipeline on top in both panels
    fig.suptitle("E1: gross-error rate with one stage switched off", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_stage_ablation.png")
    print(df.to_string(index=False))
