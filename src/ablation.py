"""Ablation study - which branch actually carries the evidence?

Five configurations are trained from scratch under the same protocol:
    physics only -> pixel only -> pixel+physics -> pixel+mask -> full

Usage:
    python -m src.ablation --dataset-dir data/dataset
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config as C
from src.evaluate import metrics_from
from src.models import PIXELVerifyNet, count_parameters
from src.train import (WindowDataset, apply_physics_stats, device,
                       fit_physics_stats, predict)

CONFIGS = [
    ("physics_only",  dict(use_pixel=False, use_mask=False, use_physics=True)),
    ("pixel_only",    dict(use_pixel=True,  use_mask=False, use_physics=False)),
    ("pixel_physics", dict(use_pixel=True,  use_mask=False, use_physics=True)),
    ("pixel_mask",    dict(use_pixel=True,  use_mask=True,  use_physics=False)),
    ("full",          dict(use_pixel=True,  use_mask=True,  use_physics=True)),
]


def _train_one(model, tr_dl, va_dl, dev, epochs, lr, wd, patience, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossfn = torch.nn.BCEWithLogitsLoss()
    best, wait = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, m, p, y in tr_dl:
            opt.zero_grad()
            loss = lossfn(model(x.to(dev), m.to(dev), p.to(dev)), y.to(dev).float())
            loss.backward()
            opt.step()
        yv, pv = predict(model, va_dl, dev)
        auc = float(roc_auc_score(yv, pv)) if len(set(yv.tolist())) > 1 else float("nan")
        if np.isfinite(auc) and auc > best:
            best, wait = auc, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Branch ablation study.")
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args(argv)

    C.ensure_dirs()
    dev = device()
    d = Path(a.dataset_dir)
    splits = pd.read_csv(d / "splits.csv")
    tensors = np.load(d / "tensors.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    physics = np.load(d / "physics.npy")

    tr = splits[splits.split == "train"].reset_index(drop=True)
    va = splits[splits.split == "validation"].reset_index(drop=True)
    te = splits[splits.split == "test"].reset_index(drop=True)

    stats = fit_physics_stats(physics[tr.row_index.values])
    tr_dl = DataLoader(WindowDataset(tensors, masks,
                                     apply_physics_stats(physics[tr.row_index.values], stats), tr),
                       batch_size=a.batch_size, shuffle=True)
    va_dl = DataLoader(WindowDataset(tensors, masks,
                                     apply_physics_stats(physics[va.row_index.values], stats), va),
                       batch_size=a.batch_size, shuffle=False)
    te_dl = DataLoader(WindowDataset(tensors, masks,
                                     apply_physics_stats(physics[te.row_index.values], stats), te),
                       batch_size=a.batch_size, shuffle=False)

    rows = []
    for name, kw in CONFIGS:
        t0 = time.time()
        model = PIXELVerifyNet(**kw).to(dev)
        best = _train_one(model, tr_dl, va_dl, dev, a.epochs, C.LEARNING_RATE,
                          C.WEIGHT_DECAY, a.patience, a.seed)
        y, prob = predict(model, te_dl, dev)
        mtr = metrics_from(np.asarray(y).astype(int), np.asarray(prob))
        mtr.update({"configuration": name, "pixel": int(kw["use_pixel"]),
                    "mask": int(kw["use_mask"]), "physics": int(kw["use_physics"]),
                    "parameters": count_parameters(model),
                    "best_val_roc_auc": best, "seconds": time.time() - t0})
        rows.append(mtr)
        print(f"[ablation] {name:14s} acc={mtr['accuracy']:.4f} "
              f"f1={mtr['f1']:.4f} auc={mtr['roc_auc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "ablation.csv", index=False)
    cols = ["configuration", "pixel", "mask", "physics", "accuracy", "precision",
            "recall", "f1", "roc_auc", "pr_auc", "fpr", "parameters"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
