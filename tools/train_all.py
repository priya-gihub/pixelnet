"""Background job: train all 5 models sequentially, then evaluate.
Logs to logs/train_all.log. Run detached AFTER the dataset is final:
    start /b cmd /c python -u tools\\train_all.py >> logs\\train_all.log 2>&1
"""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

MODELS = ["baseline_1dcnn", "baseline_2dcnn", "baseline_lstm",
          "baseline_3dcnn", "pixel_verifynet"]

def run(*args):
    print(f"[train_all] $ python {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-u", *args], cwd=ROOT)
    print(f"[train_all] exit={r.returncode}: {' '.join(args)}", flush=True)
    return r.returncode

if __name__ == "__main__":
    for m in MODELS:
        rc = run("-m", "src.train", "--dataset-dir", "data/dataset", "--model", m)
        if rc != 0:
            print(f"[train_all] FAILED on {m}, stopping", flush=True)
            break
    else:
        run("-m", "src.evaluate", "--dataset-dir", "data/dataset")
    print("[train_all] DONE", flush=True)
