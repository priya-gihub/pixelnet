"""MC-Dropout uncertainty + Expected Calibration Error for PIXEL-VerifyNet.

Dropout is left active at inference time (``model.train()``) so repeated forward
passes sample from the approximate posterior; the spread of those samples is the
predictive uncertainty, and the calibration of the mean is measured with ECE.

Usage:
    python -m src.uncertainty --checkpoint checkpoints/pixel_verifynet.pt \
        --dataset-dir data/dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config as C
from src.models import build
from src.train import WindowDataset, apply_physics_stats, device


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        q = (probs >= lo) & (probs < hi)
        if q.any():
            err += float(q.mean()) * abs(float(probs[q].mean()) - float(labels[q].mean()))
    return float(err)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MC-Dropout uncertainty + ECE.")
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINTS / "pixel_verifynet.pt"))
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--split", default="test")
    ap.add_argument("--mc-samples", type=int, default=C.MC_SAMPLES)
    a = ap.parse_args(argv)

    C.ensure_dirs()
    dev = device()
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = build(ck.get("model", "pixel_verifynet")).to(dev)
    model.load_state_dict(ck["state"])

    d = Path(a.dataset_dir)
    splits = pd.read_csv(d / "splits.csv")
    part = splits[splits.split == a.split].reset_index(drop=True)
    tensors = np.load(d / "tensors.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    physics = np.load(d / "physics.npy")
    dl = DataLoader(WindowDataset(
        tensors, masks,
        apply_physics_stats(physics[part.row_index.values], ck["physics_stats"]), part),
        batch_size=C.BATCH_SIZE, shuffle=False)

    model.train()                     # keep dropout ON -> MC sampling
    runs, labels = [], []
    with torch.no_grad():
        for i in range(a.mc_samples):
            probs, ys = [], []
            for x, m, p, y in dl:
                probs.extend(torch.sigmoid(model(x.to(dev), m.to(dev), p.to(dev)))
                             .cpu().numpy().tolist())
                ys.extend(y.numpy().tolist())
            runs.append(probs)
            if i == 0:
                labels = ys

    A = np.asarray(runs)
    mean, var = A.mean(0), A.var(0)
    y = np.asarray(labels).astype(int)
    out = {"split": a.split, "mc_samples": int(a.mc_samples), "n": int(len(y)),
           "mean_predictive_variance": float(var.mean()),
           "ece": expected_calibration_error(mean, y),
           "roc_auc_of_mean": (float(roc_auc_score(y, mean))
                               if len(set(y.tolist())) > 1 else None)}
    (C.RESULTS / "uncertainty_calibration.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"target_id": part["target_id"], "label": y, "prob_mean": mean,
                  "prob_std": np.sqrt(var)}).to_csv(
        C.RESULTS / f"uncertainty_{a.split}.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
