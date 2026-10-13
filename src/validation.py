"""Module 3 - structural validation of the cached TPF products.

Checks every downloaded product for usable pixel data, valid cadences and a
sane time axis, and writes ``results/validation_report.json``.

Usage:
    python -m src.validation --matched-dir data/matched
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def check_product(path: str) -> dict:
    """Validate one cached ``.npz`` TPF product."""
    with np.load(path) as z:
        flux = np.asarray(z["flux"])
        time = np.asarray(z["time"])
        pmask = np.asarray(z["pipeline_mask"])
    if flux.ndim != 3:
        raise ValueError(f"flux is not 3-D: {flux.shape}")

    r = {"cadences": int(flux.shape[0]),
         "frame": [int(flux.shape[1]), int(flux.shape[2])]}
    r["finite_fraction"] = float(np.isfinite(flux).mean())
    r["valid_cadences"] = int((np.isfinite(flux).sum(axis=(1, 2)) > 0).sum())

    t = time[np.isfinite(time)]
    dt = np.diff(t)
    r["median_cadence_days"] = float(np.median(dt)) if dt.size else float("nan")
    r["time_monotonic"] = bool(dt.size > 0 and np.all(dt > 0))
    r["aperture_pixels"] = int(pmask.sum())
    r["ok"] = bool(r["finite_fraction"] > 0.5 and r["valid_cadences"] > 0
                   and min(r["frame"]) >= 3)
    return r


def main():
    ap = argparse.ArgumentParser(description="Validate cached TPF products.")
    ap.add_argument("--matched-dir", default=str(C.MATCHED))
    ap.add_argument("--max-sample", type=int, default=25)
    a = ap.parse_args()

    C.ensure_dirs()
    m = pd.read_csv(Path(a.matched_dir) / "matched_all.csv")

    rows, fails = [], []
    for _, r in m.iterrows():
        try:
            rec = check_product(str(r["raw_path"]))
            rec["target_id"] = str(r["target_id"])
            rec["mission"] = str(r["mission"])
            rows.append(rec)
        except Exception as e:
            fails.append({"target_id": str(r["target_id"]), "error": repr(e)})

    d = pd.DataFrame(rows)
    report = {"checked": int(len(d)), "failed_to_open": int(len(fails)),
              "ok": int(d["ok"].sum()) if len(d) else 0}
    if len(d):
        report["by_mission"] = {str(g): {"n": int(len(s)), "ok": int(s["ok"].sum())}
                                for g, s in d.groupby("mission")}
        report["frame_sizes"] = {k: int(v) for k, v in
                                 d["frame"].astype(str).value_counts().items()}
        report["median_cadence_days"] = {str(g): float(s["median_cadence_days"].median())
                                         for g, s in d.groupby("mission")}
        report["median_valid_cadences"] = {str(g): float(s["valid_cadences"].median())
                                           for g, s in d.groupby("mission")}

    (C.RESULTS / "validation_report.json").write_text(json.dumps(report, indent=2))
    d.head(a.max_sample).to_csv(C.RESULTS / "validation_sample.csv", index=False)
    if fails:
        pd.DataFrame(fails).to_csv(C.RESULTS / "validation_failures.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"[validation] {report['ok']}/{report['checked']} products OK")


if __name__ == "__main__":
    main()
