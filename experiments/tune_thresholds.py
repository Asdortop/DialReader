"""Set the Stage 8 abstention thresholds on the validation split only.

The test split is not read here. Run:  python experiments/tune_thresholds.py
"""
import numpy as np
import pandas as pd

from common import load_synth, run, save_table, summarise
from dialreader import uncertainty as unc, w02_attributes as attrs

if __name__ == "__main__":
    val = load_synth(split="val")
    defaults = unc.default_thresholds()
    rows = run(val, thresholds=defaults)
    tuned = unc.tune_thresholds(rows)
    attrs.save_json(tuned, unc.THRESHOLD_FILE)

    table = pd.DataFrame([{"signal": k, "meaning when crossed": unc.SIGNALS[k][2],
                           "worse when": "larger" if unc.SIGNALS[k][1] else "smaller",
                           "default": defaults[k], "tuned on validation": tuned[k]} for k in unc.SIGNALS])
    save_table(table.round(3), "stage8_thresholds")
    print(table.round(3).to_string(index=False))

    for name, thr in (("default", defaults), ("tuned", tuned)):
        for r in rows:
            r["status"], _, _, r["confidence"] = unc.assess(r["signals"], thr) if r["error_fs"] is not None \
                else ("failed", [], {}, 0.0)
        s = summarise(rows)
        print(f"{name:8s} validation: n={s['n']} answered={s['answered_%']:.0f}% coverage={s['coverage_%']:.0f}% "
              f"mean={s['mean_err']:.2f} gross={s['gross_%']:.1f}% risk among reliable={s['risk_%']:.1f}%")
