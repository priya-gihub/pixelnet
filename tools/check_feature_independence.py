"""Diagnostic: are the ten physics features actually distinct, and sane?

Prints, per feature: coverage, the number of unique values, and the max
absolute Pearson correlation with every other feature. A feature that is an
exact duplicate of another (corr == 1.0) carries no extra information, which
inflates its PGSEV weight and wastes a model input.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import config as C


def main():
    P = np.load(C.PHYSICS / "physics.npy").astype(np.float64)
    print(f"physics matrix: {P.shape}\n")

    n = len(C.PHYSICS_NAMES)
    dup = []
    for j, name in enumerate(C.PHYSICS_NAMES):
        col = P[:, j]
        finite = np.isfinite(col)
        uniq = np.unique(col[finite]).size
        corrs = []
        for k, other in enumerate(C.PHYSICS_NAMES):
            if k == j:
                continue
            m = finite & np.isfinite(P[:, k])
            if m.sum() >= 8 and np.std(col[m]) > 0 and np.std(P[m, k]) > 0:
                corrs.append((abs(float(np.corrcoef(col[m], P[m, k])[0, 1])),
                              other))
        corrs.sort(reverse=True)
        top = corrs[0] if corrs else (float("nan"), "-")
        flag = "  <== EXACT DUPLICATE" if top[0] > 0.9999 else ""
        if top[0] > 0.9999:
            dup.append((name, top[1]))
        print(f"{name:22s} finite={finite.mean():6.1%} unique={uniq:5d} "
              f"max|r|={top[0]:.4f} ({top[1]}){flag}")

    print()
    if dup:
        print("FAIL: exactly duplicated feature pairs (double-counted PGSEV "
              "weight, wasted model input):")
        for a, b in dup:
            print(f"  {a} == {b}")
        return 1
    print("PASS: all ten physics features are distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())