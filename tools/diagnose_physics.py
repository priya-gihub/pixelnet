"""Diagnostic: how much planet/false-positive signal do the PGSEV features carry?

Reports per-class statistics, single-feature ROC-AUC and a train-only logistic
regression ceiling. Diagnostic only - it never feeds the final models.

Usage: python tools/diagnose_physics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config as C


def main():
    from src.train import fit_physics_stats, apply_physics_stats

    idx = pd.read_csv(C.PROCESSED / "index.csv")
    P = np.load(C.PHYSICS / "physics.npy")
    splits = pd.read_csv(C.DATASET / "splits.csv")
    s = splits.merge(idx[["row_index", "period", "catalog_depth", "mission"]],
                     on="row_index", how="left")
    P = P[s["row_index"].values]
    y = s["label"].values

    print(f"targets={len(y)}  features={C.PHYSICS_NAMES}")
    print(f"epoch present: {int(idx['epoch'].notna().sum())}/{len(idx)}  "
          f"period present: {int(idx['period'].notna().sum())}/{len(idx)}  "
          f"duration present: {int(idx['duration'].notna().sum())}/{len(idx)}")
    print(f"catalog_depth (ppm) median: "
          f"{pd.to_numeric(idx['catalog_depth'], errors='coerce').median()}\n")

    hdr = f"{'feature':22s} {'NaN':>5s} {'AUC(all)':>9s} {'AUC(train)':>11s} {'AUC(test)':>10s} {'planet_med':>11s} {'fp_med':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for j, name in enumerate(C.PHYSICS_NAMES):
        v = P[:, j].astype(float)
        ok = np.isfinite(v)
        auc_all = auc_tr = auc_te = float("nan")
        if ok.sum() > 10 and len(set(y[ok].tolist())) > 1:
            auc_all = roc_auc_score(y[ok], v[ok])
        stats = {}
        for split in ("train", "test"):
            m = (s["split"].values == split) & ok
            if m.sum() > 5 and len(set(y[m].tolist())) > 1:
                a = roc_auc_score(y[m], v[m])
                if split == "train":
                    auc_tr = a
                else:
                    auc_te = a
        stats["p"] = float(np.nanmedian(v[ok & (y == 1)])) if (ok & (y == 1)).any() else np.nan
        stats["f"] = float(np.nanmedian(v[ok & (y == 0)])) if (ok & (y == 0)).any() else np.nan
        print(f"{name:22s} {int((~ok).sum()):5d} {auc_all:9.3f} {auc_tr:11.3f} "
              f"{auc_te:10.3f} {stats['p']:11.3f} {stats['f']:9.3f}")

    # how often is there NO measurable signal at all, per class?
    print("\nno-signal (all-NaN feature rows) rate per class:")
    allnan = ~np.isfinite(P).any(axis=1)
    for lab in (1, 0):
        m = y == lab
        print(f"  label={lab}: {int((allnan & m).sum())}/{int(m.sum())} "
              f"({100 * (allnan & m).sum() / max(m.sum(), 1):.1f}%)")
    anysig = np.isfinite(P).any(axis=1)
    if len(set(y[anysig].tolist())) > 1:
        print(f"  AUC['has any measurable evidence'] = "
              f"{roc_auc_score(y[anysig], ~allnan[anysig].astype(float)):.4f}")

    # train-only logistic ceiling
    tr = s["split"].values == "train"
    te = s["split"].values == "test"
    stats_ = fit_physics_stats(P[tr])
    Xtr = apply_physics_stats(P[tr], stats_)
    Xte = apply_physics_stats(P[te], stats_)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
    ptr = clf.predict_proba(Xtr)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    print(f"\nlogistic regression on the 8 PGSEV features (train-only fit):")
    print(f"  train ROC-AUC = {roc_auc_score(y[tr], ptr):.4f}")
    print(f"  test  ROC-AUC = {roc_auc_score(y[te], pte):.4f}")
    print(f"  coefficients = "
          + ", ".join(f"{n}={c:+.2f}" for n, c in
                      zip(C.PHYSICS_NAMES, clf.coef_[0])))


if __name__ == "__main__":
    main()