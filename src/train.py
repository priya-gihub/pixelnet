"""Module 7 - training loop. One identical protocol for all five models.

All models see the same targets, the same target-level split, the same
normalised inputs, the same loss, the same seed and the same early-stopping rule
(validation ROC-AUC). Physics features are imputed and standardised using
**training-split statistics only**, which is stored inside the checkpoint so the
evaluation step is guaranteed to reuse exactly the same transform.

Usage:
    python -m src.train --dataset-dir data/dataset --model pixel_verifynet
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
from torch.utils.data import DataLoader, Dataset

import config as C
from src.models import MODELS, MODEL_LABELS, build, count_parameters


# ------------------------------------------------------- physics transform ----
def fit_physics_stats(P: np.ndarray) -> dict:
    """Median-impute + standardise using TRAIN-split statistics only."""
    med = np.nanmedian(P, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float64)
    Z = np.where(np.isfinite(P), P, med)
    return {"median": med, "mean": Z.mean(axis=0), "std": Z.std(axis=0) + 1e-6}


def apply_physics_stats(P: np.ndarray, stats: dict) -> np.ndarray:
    Z = np.where(np.isfinite(P), P, stats["median"])
    return ((Z - stats["mean"]) / stats["std"]).astype(np.float32)


# ----------------------------------------------------------------- dataset ----
class WindowDataset(Dataset):
    def __init__(self, tensors, masks, physics, rows):
        self.t, self.m, self.p, self.rows = tensors, masks, physics, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        j = int(r["row_index"])
        # NOTE: self.p is ALWAYS the physics block aligned to self.rows order
        # (train.py and evaluate.py pre-slice physics with tr/va/te row_index and
        # pass a positional block), so index it with positional i, while the
        # image cubes live in whole-dataset arrays and use dataset row j.
        return (torch.tensor(np.asarray(self.t[j], np.float32)),
                torch.tensor(np.asarray(self.m[j], np.float32)),
                torch.tensor(np.asarray(self.p[i], np.float32)),
                torch.tensor(float(r["label"])))


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, m, p, y in loader:
        logit = model(x.to(dev), m.to(dev), p.to(dev))
        probs.extend(torch.sigmoid(logit).cpu().numpy().tolist())
        ys.extend(y.numpy().tolist())
    return np.asarray(ys), np.asarray(probs)


# --------------------------------------------------------------------- main ---
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train one model.")
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args(argv)

    C.ensure_dirs()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = device()
    d = Path(a.dataset_dir)

    splits = pd.read_csv(d / "splits.csv")
    tensors = np.load(d / "tensors.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    physics = np.load(d / "physics.npy")

    tr = splits[splits.split == "train"].reset_index(drop=True)
    va = splits[splits.split == "validation"].reset_index(drop=True)

    p_tr = physics[tr.row_index.values]
    stats = fit_physics_stats(p_tr)
    loaders = {
        "train": DataLoader(
            WindowDataset(tensors, masks, apply_physics_stats(p_tr, stats), tr),
            batch_size=a.batch_size, shuffle=True, num_workers=C.NUM_WORKERS),
        "validation": DataLoader(
            WindowDataset(tensors, masks,
                          apply_physics_stats(physics[va.row_index.values], stats), va),
            batch_size=a.batch_size, shuffle=False, num_workers=C.NUM_WORKERS),
    }

    model = build(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lossfn = torch.nn.BCEWithLogitsLoss()

    best, wait, history = -1.0, 0, []
    ckpt = C.CHECKPOINTS / f"{a.model}.pt"
    t0 = time.time()

    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []
        for x, m, p, y in loaders["train"]:
            x, m, p, y = x.to(dev), m.to(dev), p.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossfn(model(x, m, p), y.float())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        yv, pv = predict(model, loaders["validation"], dev)
        auc = float(roc_auc_score(yv, pv)) if len(set(yv.tolist())) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_roc_auc": auc})
        print(f"[{a.model}] epoch {epoch:03d} loss={np.mean(losses):.4f} "
              f"val_roc_auc={auc:.4f}", flush=True)

        if np.isfinite(auc) and auc > best:
            best, wait = auc, 0
            torch.save({"state": model.state_dict(), "model": a.model,
                        "physics_stats": stats, "val_roc_auc": best,
                        "epoch": epoch, "config": vars(a)}, ckpt)
        else:
            wait += 1
            if wait >= a.patience:
                print(f"[{a.model}] early stop at epoch {epoch}", flush=True)
                break

    summary = {"model": a.model, "label": MODEL_LABELS[a.model],
               "parameters": count_parameters(model), "epochs_run": len(history),
               "best_val_roc_auc": best, "train_seconds": time.time() - t0,
               "checkpoint": str(ckpt),
               "physics_scaler": "fitted on the training split only"}
    (C.RESULTS / f"{a.model}_history.json").write_text(json.dumps(history, indent=2))
    (C.RESULTS / f"{a.model}_train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
