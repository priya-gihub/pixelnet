"""Module 8 - metrics on the held-out test targets.

Every model is evaluated through exactly the same code path, with the same
inputs, the same physics transform (loaded from its own checkpoint) and the same
fixed decision threshold (0.5). No metric is hard-coded.

Usage:
    python -m src.evaluate --dataset-dir data/dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

import config as C
from src.models import MODELS, MODEL_LABELS, build, count_parameters
from src.train import WindowDataset, apply_physics_stats, device

METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]
IEEE_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"]


def markdown_table(df: pd.DataFrame) -> str:
    """Minimal Markdown writer (avoids the optional ``tabulate`` dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule] + body) + "\n"


def metrics_from(y: np.ndarray, prob: np.ndarray, thr: float = C.THRESHOLD) -> dict:
    """All requested metrics from raw probabilities at the fixed threshold."""
    pred = (np.asarray(prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(set(np.asarray(y).tolist())) > 1
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if two else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if two else float("nan"),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "mcc": float(matthews_corrcoef(y, pred)) if two else float("nan"),
        "threshold": float(thr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate all models on the test split.")
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args(argv)

    C.ensure_dirs()
    dev = device()
    d = Path(a.dataset_dir)

    splits = pd.read_csv(d / "splits.csv")
    tensors = np.load(d / "tensors.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    physics = np.load(d / "physics.npy")

    te = splits[splits.split == "test"].reset_index(drop=True)
    if len(te) == 0:
        raise SystemExit("no test rows in splits.csv")

    rows, all_metrics, skipped = [], {}, []
    for name in a.models:
        ckpt_path = C.CHECKPOINTS / f"{name}.pt"
        if not ckpt_path.exists():
            skipped.append(name)
            print(f"[evaluate] SKIP {name}: no checkpoint")
            continue

        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model = build(ck.get("model", name)).to(dev)
        model.load_state_dict(ck["state"])
        model.eval()

        stats = ck["physics_stats"]
        ds = WindowDataset(tensors, masks,
                           apply_physics_stats(physics[te.row_index.values], stats), te)
        dl = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False)

        y, prob = [], []
        with torch.no_grad():
            for x, m, p, yy in dl:
                logit = model(x.to(dev), m.to(dev), p.to(dev))
                prob.extend(torch.sigmoid(logit).cpu().numpy().tolist())
                y.extend(yy.numpy().tolist())
        y = np.asarray(y).astype(int)
        prob = np.asarray(prob)

        mtr = metrics_from(y, prob)
        mtr["model"] = name
        mtr["model_label"] = MODEL_LABELS.get(name, name)
        mtr["parameters"] = count_parameters(model)
        mtr["n_test_targets"] = int(len(te))
        all_metrics[name] = mtr
        rows.append(mtr)

        pd.DataFrame({"target_id": te["target_id"], "mission": te["mission"],
                      "label": y, "probability": prob,
                      "prediction": (prob >= C.THRESHOLD).astype(int)}).to_csv(
            C.RESULTS / f"predictions_{name}.csv", index=False)
        print(f"[evaluate] {MODEL_LABELS.get(name, name):18s} "
              f"acc={mtr['accuracy']:.4f} f1={mtr['f1']:.4f} auc={mtr['roc_auc']:.4f}")

    cols = ["model_label"] + METRIC_ORDER + ["fnr", "mcc", "threshold", "tn", "fp",
                                             "fn", "tp", "parameters", "n_test_targets"]
    table = pd.DataFrame(rows)[["model"] + cols] if rows else pd.DataFrame(columns=cols)
    table.to_csv(C.RESULTS / "final_metrics.csv", index=False)
    (C.RESULTS / "final_metrics.json").write_text(
        json.dumps({"threshold": C.THRESHOLD, "split": "test",
                    "metrics": all_metrics}, indent=2))

    # IEEE-ready table (clean labels, seven required columns)
    ieee = pd.DataFrame([{"Model": r["model_label"],
                          "Accuracy": r["accuracy"], "Precision": r["precision"],
                          "Recall": r["recall"], "F1": r["f1"],
                          "ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"],
                          "FPR": r["fpr"]} for r in rows])
    ieee_round = ieee.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR"]:
        ieee_round[c] = ieee_round[c].round(4)
    ieee_round.to_csv(C.TABLES / "final_model_comparison.csv", index=False)
    (C.TABLES / "final_model_comparison.md").write_text(markdown_table(ieee_round))

    if skipped:
        print(f"[evaluate] missing checkpoints: {skipped}")
    print(f"[evaluate] wrote {C.RESULTS / 'final_metrics.csv'}")
    print(ieee_round.to_string(index=False))


if __name__ == "__main__":
    main()
