"""Learning curve: how much does PIXEL-VerifyNet actually gain from more targets?

Trains the real model on nested subsets of the TRAINING targets (25 / 50 / 75 /
100 %) and evaluates every one of them on the SAME held-out test split, with the
identical protocol. The slope tells us whether more downloads help, and by how
much; a flat curve is the honest answer that they would not.

Usage: python tools/learning_curve.py [--epochs 20] [--repeats 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config as C

SCRATCH = ROOT / "results" / "learning_curve"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="pixel_verifynet")
    a = ap.parse_args()

    C.ensure_dirs()
    (SCRATCH / "checkpoints").mkdir(parents=True, exist_ok=True)
    C.CHECKPOINTS = SCRATCH / "checkpoints"
    C.RESULTS = SCRATCH

    import torch
    from torch.utils.data import DataLoader

    import src.train as tr

    splits = pd.read_csv(Path(a.dataset_dir) / "splits.csv")
    tensors = np.load(Path(a.dataset_dir) / "tensors.npy", mmap_mode="r")
    masks = np.load(Path(a.dataset_dir) / "masks.npy", mmap_mode="r")
    physics = np.load(Path(a.dataset_dir) / "physics.npy")

    train_all = splits[splits.split == "train"].reset_index(drop=True)
    test = splits[splits.split == "test"].reset_index(drop=True)
    n_test = test["target_id"].nunique()
    print(f"pool: {train_all['target_id'].nunique()} train targets, "
          f"{n_test} test targets (test set never changes)\n")

    rows = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        aucs, accs = [], []
        for rep in range(a.repeats):
            if frac >= 1.0:
                sub = train_all
            else:
                sub, _ = train_test_split(
                    train_all, train_size=frac, random_state=C.SEED + rep,
                    stratify=train_all[["mission", "label"]])
                sub = sub.reset_index(drop=True)
            stats = tr.fit_physics_stats(physics[sub.row_index.values])
            dl_tr = DataLoader(
                tr.WindowDataset(tensors, masks,
                                 tr.apply_physics_stats(
                                     physics[sub.row_index.values], stats), sub),
                batch_size=C.BATCH_SIZE, shuffle=True)
            dl_va = DataLoader(
                tr.WindowDataset(tensors, masks,
                                 tr.apply_physics_stats(
                                     physics[test.row_index.values], stats), test),
                batch_size=C.BATCH_SIZE, shuffle=False)

            torch.manual_seed(C.SEED + rep)
            from src.models import build
            model = build(a.model)
            opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                                    weight_decay=C.WEIGHT_DECAY)
            lossfn = torch.nn.BCEWithLogitsLoss()
            best, wait = -1.0, 0
            for _ in range(a.epochs):
                model.train()
                for x, m, p, y in dl_tr:
                    opt.zero_grad()
                    lossfn(model(x, m, p), y.float()).backward()
                    opt.step()
                yv, pv = tr.predict(model, dl_va, torch.device("cpu"))
                auc = float(roc_auc_score(yv, pv))
                if auc > best:
                    best, wait = auc, 0
                else:
                    wait += 1
                    if wait >= C.PATIENCE:
                        break
            yv, pv = tr.predict(model, dl_va, torch.device("cpu"))
            pred = (pv >= C.THRESHOLD).astype(int)
            aucs.append(float(roc_auc_score(yv, pv)))
            accs.append(float((pred == yv).mean()))
        rows.append({"train_fraction": frac,
                     "n_train_targets": int(sub["target_id"].nunique()),
                     "test_roc_auc_mean": float(np.mean(aucs)),
                     "test_roc_auc_std": float(np.std(aucs)),
                     "test_accuracy_mean": float(np.mean(accs))})
        r = rows[-1]
        print(f"  {100 * frac:5.0f}% of pool = {r['n_train_targets']:4d} targets "
              f"-> test ROC-AUC {r['test_roc_auc_mean']:.4f} "
              f"(+/-{r['test_roc_auc_std']:.4f}), acc {r['test_accuracy_mean']:.4f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(SCRATCH / "learning_curve.csv", index=False)

    # fit AUC vs log(N) to see whether it is still climbing
    x = np.log(df["n_train_targets"].values.astype(float))
    yv = df["test_roc_auc_mean"].values
    slope, intercept = np.polyfit(x, yv, 1)
    print(f"\nfit: ROC-AUC = {intercept:.4f} + {slope:.4f} * ln(N_train)")
    for target in (0.90, 0.95, 0.98):
        if slope <= 0:
            print(f"  to reach {target:.2f}: slope <= 0 -> NOT reachable by "
                  "adding data alone")
            break
        n_needed = np.exp((target - intercept) / slope)
        print(f"  to reach {target:.2f} by this fit: N_train ~ {n_needed:,.0f} "
              f"targets (x{n_needed / df.n_train_targets.max():.1f} the current "
              "pool)")
    (SCRATCH / "learning_curve.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[learning_curve] wrote {SCRATCH / 'learning_curve.csv'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time() - t0:.0f}s")