"""Module 5b - PGSEV: Physics-Guided Scientific Evidence Verification.

Converts the six *raw* physics features into bounded evidence contributions in
[0, 1] using **fixed, documented domain priors** - never statistics derived from
the labels or from the test split - and combines them into the PGSEV score.

Usage:
    python -m src.pgsev --processed-dir data/processed --outdir data/physics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from src.physics_features import compute_all

# feature -> (transform, lo, hi)   (fixed, documented, label-free priors in
# config.PGSEV_PRIORS; never statistics derived from labels or from the test split)
REFERENCE = dict(C.PGSEV_PRIORS)


def _norm(kind: str, lo: float, hi: float, v: float) -> float:
    if not np.isfinite(v):
        return np.nan
    if kind == "clip":
        x = (v - lo) / (hi - lo)
    elif kind == "log_ratio":
        x = ((np.log10(max(v, 1e-9)) - np.log10(lo))
             / (np.log10(hi) - np.log10(lo)))
    elif kind == "invert":
        x = 1.0 - (v - lo) / (hi - lo)
    else:
        x = v
    return float(min(1.0, max(0.0, x)))


def evidence_matrix(P: np.ndarray) -> np.ndarray:
    E = np.zeros(P.shape, np.float32)
    for j, n in enumerate(C.PHYSICS_NAMES):
        kind, lo, hi = REFERENCE[n]
        E[:, j] = [_norm(kind, lo, hi, float(v)) for v in P[:, j]]
    return E


def pgsev_score(E: np.ndarray) -> np.ndarray:
    """Weighted mean of the evidence contributions (weights sum to 1)."""
    w = np.asarray([C.PGSEV_WEIGHTS[n] for n in C.PHYSICS_NAMES], np.float64)
    w = w / w.sum()
    E = np.where(np.isfinite(E), E, 0.5)
    return (E * w).sum(axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Physics features + PGSEV evidence.")
    ap.add_argument("--processed-dir", default=str(C.PROCESSED))
    ap.add_argument("--outdir", default=str(C.PHYSICS))
    a = ap.parse_args()

    C.ensure_dirs()
    outdir = Path(a.outdir)
    P, df = compute_all(a.processed_dir, a.outdir)

    E = evidence_matrix(P)
    pg = pgsev_score(E)

    df2 = df.copy()
    for j, n in enumerate(C.PHYSICS_NAMES):
        df2["evidence_" + n] = E[:, j]
    df2["pgsev"] = pg
    df2.to_csv(outdir / "pgsev.csv", index=False)
    np.save(outdir / "pgsev.npy", pg)

    # label-separability diagnostic (reporting only - never a model input)
    from scipy.stats import spearmanr

    diag = {}
    for n in C.PHYSICS_NAMES:
        col = pd.to_numeric(df[n], errors="coerce")
        col = col.fillna(col.median() if col.notna().any() else 0.0)
        diag[n] = float(spearmanr(col, df["label"]).statistic)
    diag["pgsev"] = float(spearmanr(pg, df["label"]).statistic)

    report = {"targets": int(len(df)),
              "pgsev_mean": float(np.mean(pg)), "pgsev_std": float(np.std(pg)),
              "pgsev_min": float(np.min(pg)), "pgsev_max": float(np.max(pg)),
              "nan_per_feature": {n: int(pd.to_numeric(df[n], errors="coerce").isna().sum())
                                  for n in C.PHYSICS_NAMES},
              "spearman_with_label": diag}
    (C.RESULTS / "pgsev_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"[pgsev] wrote {outdir / 'pgsev.csv'} for {len(df)} targets")


if __name__ == "__main__":
    main()
