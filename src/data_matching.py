"""Module 2 - match the downloaded TPF products to their KOI/TOI labels.

Produces ``data/matched/matched_all.csv``: one row per unique target with the
archive label, archive metadata and the path of the cached pixel product.

Usage:
    python -m src.data_matching --raw-dir data/raw --outdir data/matched
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def load_catalogs(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for mission in C.MISSIONS:
        p = raw_dir / f"{mission}_catalog.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit(f"no *_catalog.csv under {raw_dir}: run src.data_download first")
    return pd.concat(frames, ignore_index=True)


def match(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    d = load_catalogs(raw_dir)
    n0 = len(d)

    d = d.dropna(subset=["target_id", "label", "raw_path"]).copy()
    d["label"] = d["label"].astype(int)

    present = d["raw_path"].map(lambda p: Path(str(p)).exists())
    missing = int((~present).sum())
    d = d[present].copy()

    dup = int(d["target_id"].duplicated().sum())
    d = d.drop_duplicates("target_id").reset_index(drop=True)

    d = d.sort_values(["mission", "label", "target_id"]).reset_index(drop=True)
    d["row_index"] = np.arange(len(d), dtype=int)

    report = {
        "rows_raw": int(n0),
        "targets_matched": int(len(d)),
        "unique_targets": int(d["target_id"].nunique()),
        "missing_files_dropped": missing,
        "duplicates_removed": dup,
        "by_mission": {m: int((d.mission == m).sum()) for m in C.MISSIONS},
        "by_mission_label": {f"{m}:{l}": int(((d.mission == m) & (d.label == l)).sum())
                             for m in C.MISSIONS for l in (0, 1)},
        "labels": {"planet": int((d.label == 1).sum()),
                   "false_positive": int((d.label == 0).sum())},
    }
    return d, report


def main():
    ap = argparse.ArgumentParser(description="Match TPFs to KOI/TOI labels.")
    ap.add_argument("--raw-dir", default=str(C.RAW))
    ap.add_argument("--outdir", default=str(C.MATCHED))
    a = ap.parse_args()

    C.ensure_dirs()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d, report = match(Path(a.raw_dir))
    out = outdir / "matched_all.csv"
    d.to_csv(out, index=False)
    (C.RESULTS / "matching_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"[matching] wrote {out} ({len(d)} targets)")


if __name__ == "__main__":
    main()
