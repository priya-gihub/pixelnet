"""Inspect which pixel the physics stage treats as 'the target' for real data.

Usage: python tools/diagnose_target_pixel.py [n_targets]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C
from src.physics_features import target_pixel_index


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    idx = pd.read_csv(C.PROCESSED / "index.csv")
    T = np.load(C.PROCESSED / "tensors.npy", mmap_mode="r")
    M = np.load(C.PROCESSED / "masks.npy", mmap_mode="r")
    print(f"{'target':16s} {'picked(y,x)':>12s} {'brightest':>10s} "
          f"{'obs_px':>7s} {'med@pick':>9s} {'med@max':>9s} {'max/med@pick':>13s}")
    agree = 0
    for i in range(min(n, len(idx))):
        t = np.asarray(T[i], np.float32)
        m = np.asarray(M[i], np.float32) > 0
        obs = m.sum(axis=0) > 0
        ty, tx = target_pixel_index(t, m)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(m, t, np.nan), axis=0)
        med = np.where(obs, med, np.nan)
        if not np.isfinite(med).any():
            print(f"{idx.target_id.iloc[i]:16s} no observed pixel")
            continue
        flat = int(np.nanargmax(med))
        by, bx = flat // med.shape[1], flat % med.shape[1]
        same = (by, bx) == (ty, tx)
        agree += same
        mp = float(med[ty, tx])
        mm = float(med[by, bx])
        print(f"{idx.target_id.iloc[i]:16s} {str((ty, tx)):>12s} "
              f"{str((by, bx)):>10s} {int(obs.sum()):7d} {mp:9.3f} {mm:9.3f} "
              f"{mm / (abs(mp) + 1e-6):13.2f}{'  <-- same' if same else ''}")
    print(f"\npicked == brightest in {agree}/{min(n, len(idx))} targets")


if __name__ == "__main__":
    main()