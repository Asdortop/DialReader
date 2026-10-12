"""Download the public MeasureBench real-world split (about 100 MB) for experiment E6.

MeasureBench (FlagEval/MeasureBench on Hugging Face) is licensed CC BY-SA 4.0, so it
is fetched here rather than stored in this repository.

Run:  python experiments/measurebench_download.py
then: python -I experiments/measurebench_export.py
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

RAW = Path(__file__).resolve().parent.parent / "data" / "measurebench" / "raw"

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "data/real_world-00000-of-00001.parquet"):
        print("saved", hf_hub_download("FlagEval/MeasureBench", name, repo_type="dataset", local_dir=str(RAW)))
