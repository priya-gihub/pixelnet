"""Diagnostic probe: train/evaluate on the CURRENT dataset with outputs sent to
a scratch directory, so the real checkpoints and results are never touched.

Purpose: measure what the pipeline achieves right now and locate the technical
bottleneck before the 1000-target run finishes.

Usage: python tools/probe_train.py [--epochs 40] [--models a,b,c]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C

SCRATCH = ROOT / "results" / "probe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--models", default="baseline_3dcnn,pixel_verifynet")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    ck = SCRATCH / "checkpoints"
    for sub in (ck, SCRATCH / "tables"):
        sub.mkdir(parents=True, exist_ok=True)
    C.CHECKPOINTS = ck
    C.RESULTS = SCRATCH
    C.TABLES = SCRATCH / "tables"

    import src.train as tr
    import src.evaluate as ev

    models = a.models.split(",")
    for m in models:
        print(f"\n=== probe train {m} ===", flush=True)
        tr.main(["--dataset-dir", a.dataset_dir, "--model", m,
                 "--epochs", str(a.epochs)])
    ev.main(["--dataset-dir", a.dataset_dir, "--models", *models])
    print(f"[probe] artefacts in {SCRATCH}")


if __name__ == "__main__":
    main()