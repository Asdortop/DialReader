"""Regenerate every result, in order.

    python experiments/run_all.py            classical experiments, lab figures, report
    python experiments/run_all.py --genai    also the GenAI experiments (needs a key in .env)

The dataset step is skipped when data/synthetic already exists; delete it to re-render.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENAI = "--genai" in sys.argv

STEPS = [
    ("make_dataset.py", [], not (HERE.parent / "data" / "synthetic" / "labels.csv").exists()),
    ("make_stress_set.py", [], not (HERE.parent / "data" / "synthetic_stress" / "labels.csv").exists()),
    ("tune_thresholds.py", [], True),          # validation split only
    ("e1_stage_ablation.py", [], True),
    ("e5_generalisation.py", [], True),
    ("e2_robustness.py", ["--vlm"] if GENAI else [], True),
    ("e3_e4_genai_hooks.py", [], GENAI),
    ("e6_measurebench.py", [], (HERE.parent / "data" / "measurebench" / "dial" / "index.csv").exists()),
    ("stage9_recognition.py", [], True),
    ("stage10_registration.py", [], True),
    ("stage11_compression.py", [], True),
    ("weekly_demos.py", [], True),
    ("build_report.py", [], True),
]

if __name__ == "__main__":
    for script, args, wanted in STEPS:
        if not wanted:
            print(f"-- skip {script}")
            continue
        t0 = time.time()
        print(f"-- {script} {' '.join(args)}", flush=True)
        code = subprocess.call([sys.executable, str(HERE / script), *args], cwd=str(HERE.parent))
        print(f"   {'ok' if code == 0 else 'FAILED'} in {time.time() - t0:.0f} s", flush=True)
