"""Quick status audit: counts unique targets, splits, class balance. No quoting hell."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
print(f"ROOT={ROOT}")
for p in ["data/raw/kepler_catalog.csv", "data/raw/tess_catalog.csv",
          "data/matched/matched_all.csv", "data/processed/index.csv",
          "data/physics/physics.csv", "data/dataset/splits.csv",
          "results/final_metrics.csv"]:
    f = ROOT / p
    if not f.exists():
        print(f"MISSING {p}")
        continue
    try:
        df = pd.read_csv(f)
        cols = list(df.columns)
        line = f"OK {p}: rows={len(df)} cols={cols[:8]}"
        if "target_id" in df.columns:
            line += f" uniq_targets={df['target_id'].nunique()}"
        if "label" in df.columns:
            line += f" label_counts={df['label'].value_counts().to_dict()}"
        if "mission" in df.columns:
            line += f" mission={df['mission'].value_counts().to_dict()}"
        if "split" in df.columns:
            line += f" split={df['split'].value_counts().to_dict()}"
        print(line)
    except Exception as e:
        print(f"ERR {p}: {e!r}")

for d in ["data/raw/kepler", "data/raw/tess"]:
    p = ROOT / d
    n = len(list(p.glob("*.npz"))) if p.exists() else 0
    print(f"DIR {d}: {n} npz files")
for p in ["data/processed/tensors.npy", "data/physics/physics.npy", "data/dataset/tensors.npy"]:
    f = ROOT / p
    if f.exists():
        try:
            a = np.load(f, mmap_mode="r")
            print(f"OK {p}: shape={a.shape}")
        except Exception as e:
            print(f"ERR {p}: {e!r}")
    else:
        print(f"MISSING {p}")
ckpt = list((ROOT / "checkpoints").glob("*.pt")) if (ROOT / "checkpoints").exists() else []
print(f"CHECKPOINTS: {[x.name for x in ckpt]}")
