"""Audit: does single-target inference reproduce the stored training features?

Train/serve skew is the classic way a "single-target demo" silently disagrees
with the reported test metrics. This replays the inference path for real dataset
rows and compares every physics feature, the tensor and the mask against the
artifacts the model was actually trained and evaluated on.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C
from src.inference_pipeline import load_arrays
from src.physics_features import compute_features
from src.preprocessing import build_from_arrays
from pathlib import Path


def main(n=5):
    idx = pd.read_csv(C.PROCESSED / "index.csv")
    P = np.load(C.PHYSICS / "physics.npy")
    T = np.load(C.PROCESSED / "tensors.npy", mmap_mode="r")
    M = np.load(C.PROCESSED / "masks.npy", mmap_mode="r")
    rows = idx.sample(n=min(n, len(idx)), random_state=0)

    report, worst_t, worst_m, worst_p = [], 0.0, 0.0, 0.0
    for _, r in rows.iterrows():
        i = int(r["row_index"])
        flux, time, quality, pmask = load_arrays(Path(str(r["raw_path"])))
        tensor, mask, info = build_from_arrays(
            flux, time, quality, pmask,
            period=float(r["period"]), epoch=float(r["epoch"]),
            duration=float(r["duration"]))
        f = compute_features(
            tensor, mask, bin_time=info["bin_time"],
            period=float(r["period"]), epoch=float(r["epoch"]),
            duration=float(r["duration"]),
            star=tuple(info["star_pixel"]),
            raw_depth_ppm=info["raw_depth_ppm"])

        dt = float(np.nanmax(np.abs(np.asarray(tensor, np.float64)
                                  - np.asarray(T[i], np.float64))))
        dm = float(np.nanmax(np.abs(np.asarray(mask, np.float64)
                                  - np.asarray(M[i], np.float64))))
        both_finite = np.isfinite(f) & np.isfinite(P[i])
        dp = (float(np.nanmax(np.abs(f[both_finite] - P[i][both_finite])))
              if both_finite.any() else float("nan"))
        worst_t, worst_m, worst_p = max(worst_t, dt), max(worst_m, dm), max(worst_p, dp)
        report.append({"target_id": str(r["target_id"]), "row": i,
                       "max_abs_dtensor": dt, "max_abs_dmask": dm,
                       "max_abs_dphysics": dp})
        print(json.dumps(report[-1]))

    ok = (worst_t < 1e-5 and worst_m < 1e-6
          and (np.isnan(worst_p) or worst_p < 1e-3))
    print(f"\nmax |d tensor|={worst_t:.3e}  max |d mask|={worst_m:.3e}  "
          f"max |d physics|={worst_p:.3e}")
    print("PASS: inference reproduces the training artifacts" if ok
          else "FAIL: train/serve skew detected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))