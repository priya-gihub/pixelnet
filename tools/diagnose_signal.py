"""Diagnose why the pixel branch under-performs: temporal resolution vs transit
duration, per-target signal strength, and how much a global-average-pooled head
dilutes the transit signal.

Usage: python tools/diagnose_signal.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C


def main():
    idx = pd.read_csv(C.PROCESSED / "index.csv")
    T = np.load(C.PROCESSED / "tensors.npy", mmap_mode="r")
    M = np.load(C.PROCESSED / "masks.npy", mmap_mode="r")

    print(f"targets={len(idx)}  tensor={T.shape}")
    cad = idx["cadences"].values
    print(f"cadences: median={np.median(cad):.0f} min={cad.min()} max={cad.max()}")
    print(f"TIME_BINS={C.TIME_BINS} -> cadences/bin (median target) = "
          f"{np.median(cad) / C.TIME_BINS:.1f}")
    print(f"                    -> days/bin at 0.0204 d cadence = "
          f"{np.median(cad) / C.TIME_BINS * 0.0204:.3f} d")
    print("typical Kepler transit duration ~0.10 d, TESS ~0.10-0.30 d")
    print("=> a transit is diluted into roughly "
          f"{0.10 / (np.median(cad) / C.TIME_BINS * 0.0204):.1f} bin(s); its depth is "
          "averaged with the surrounding out-of-transit cadences.\n")

    # per-target depth in the TARGET pixel (where a real transit must appear).
    # cut_window centres the target, so the native centroid maps to the fixed
    # window centre: window_index = native_index - centroid + H//2 (W//2).
    off = (C.HEIGHT // 2, C.WIDTH // 2)
    ty, tx = [], []
    for p in idx["raw_path"]:
        try:
            with np.load(p) as z:
                pm = np.asarray(z["pipeline_mask"], bool)
            if pm.sum() > 0:
                ys, xs = np.nonzero(pm)
                cy_i, cx_i = int(round(ys.mean())), int(round(xs.mean()))
            else:
                with np.load(p) as z:
                    f0 = np.asarray(z["flux"], np.float32)
                med = np.nanmedian(f0, axis=0)
                cy_i, cx_i = (int(v) for v in
                              np.unravel_index(np.nanargmax(med), med.shape))
            ty.append(cy_i - off[0])
            tx.append(cx_i - off[1])
        except Exception:
            ty.append(None)
            tx.append(None)

    rows = []
    for i in range(len(idx)):
        if ty[i] is None:
            rows.append((np.nan, np.nan, np.nan))
            continue
        y, x = ty[i], tx[i]
        if not (0 <= y < C.HEIGHT and 0 <= x < C.WIDTH):
            rows.append((np.nan, np.nan, np.nan))
            continue
        t = np.asarray(T[i], np.float32)
        m = np.asarray(M[i], np.float32) > 0
        sel = m[:, y, x]
        v = t[sel, y, x] if sel.sum() >= 3 else np.array([], np.float32)
        if v.size == 0:
            rows.append((np.nan, np.nan, np.nan))
            continue
        rows.append((float(v.min()), float(np.median(v)), float(v.std())))

    d = pd.DataFrame(rows, columns=["tmin", "tmed", "tstd"], index=idx.index)
    d["label"] = idx["label"].values
    d["mission"] = idx["mission"].values
    print("TARGET-PIXEL signal per target (per-pixel sigma units):")
    for lab, g in d.groupby("label"):
        print(f"  label={lab} n={len(g):4d} min: median={g['tmin'].median():.3f} "
              f"p10={g['tmin'].quantile(0.10):.3f} | std median={g['tstd'].median():.3f}")

    from sklearn.metrics import roc_auc_score
    for col, name in (("tmin", "deepest dip in target pixel"),
                      ("tstd", "target-pixel scatter")):
        v = d[col].fillna(0.0).values
        try:
            print(f"  AUC[{name}] = {roc_auc_score(d.label.values, v):.4f}")
        except Exception as e:
            print(f"  AUC[{name}] failed: {e}")

    for lab, g in d.groupby("mission"):
        try:
            print(f"  AUC[{lab} deepest dip] = "
                  f"{roc_auc_score(g.label.values, g['tmin'].fillna(0.0).values):.4f}"
                  f"  (n={len(g)})")
        except Exception:
            pass

    # dilution factor if we bin to different depths
    print("\nhypothetical: how much of the deepest dip survives binning?")
    for nb in (64, 128, 192, 256, 384):
        factor = 0.10 / (np.median(cad) / nb * 0.0204)
        print(f"  TIME_BINS={nb:4d} -> {nb / C.TIME_BINS:4.1f}x finer, "
              f"transit spans ~{factor:4.1f} bins "
              f"({'RESOLVED' if factor <= 1.5 else 'DILUTED'})")


if __name__ == "__main__":
    main()