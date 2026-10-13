"""Parallel stage worker: fetch TESS FALSE-POSITIVE targets while the main
fetch1000 job is still working through the TESS *planet* phase.

Why this is safe (no file races, no catalog corruption):
  * the main job's planet phase never requests an FP target, so the two workers
    fetch disjoint sets of targets;
  * this worker writes its own catalog ``data/raw/tess_stage.csv`` and never
    touches ``data/raw/tess_catalog.csv`` (which the running job owns);
  * ``tools/prune_and_refill.py`` merges the stage catalog into the main one at
    the end, de-duplicating by target_id.

Launch detached:
    start /b cmd /c python -u tools\\fetch_tess_fp_stage.py >> logs\\stage_fp.log 2>&1
"""
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C
from src import data_download as dd

MISSION = "tess"
LABEL = 0
PER_CLASS = C.TARGETS_PER_STRATUM          # 250
STAGE_CSV = ROOT / "data" / "raw" / "tess_stage.csv"
LOCK = threading.Lock()


def existing_ids() -> set:
    ids = set()
    for csv in (ROOT / "data" / "raw" / f"{MISSION}_catalog.csv", STAGE_CSV):
        if csv.exists():
            try:
                ids |= set(pd.read_csv(csv)["target_id"].astype(str))
            except Exception as e:
                print(f"[stage] warn reading {csv.name}: {e}", flush=True)
    return ids


def main():
    pool = dd._label_pool(MISSION)
    sub = pool[pool["label"] == LABEL].copy()
    sub = sub.sample(frac=1.0, random_state=C.SEED + 1).reset_index(drop=True)
    have = existing_ids()
    print(f"[stage] FP pool={len(sub)} already_present={len(have)}", flush=True)

    rows = []
    if STAGE_CSV.exists():
        try:
            rows = pd.read_csv(STAGE_CSV).to_dict("records")
            have |= {str(r["target_id"]) for r in rows}
        except Exception:
            rows = []

    outdir = C.RAW
    failures = []
    for _, r in sub.iterrows():
        with LOCK:
            got = len([x for x in rows if int(x["label"]) == LABEL])
        if got >= PER_CLASS:
            break
        tid = str(r["target_id"])
        if tid in have:
            continue
        path = outdir / MISSION / f"{dd._safe(tid)}.npz"
        meta = None
        if path.exists():
            try:
                with np.load(path) as z:
                    meta = {"n_cadences": int(z["flux"].shape[0]),
                            "exptime": float("nan"), "product": "cached",
                            "author": "cached"}
            except Exception:
                path.unlink(missing_ok=True)
        if meta is None:
            try:
                print(f"[stage] fetching {tid} ...", flush=True)
                meta = dd.fetch_tpf(tid, MISSION, path)
            except Exception as e:
                failures.append((tid, repr(e)))
                continue
        if meta is None:
            continue
        rows.append({
            "target_id": tid, "archive_id": str(r["archive_id"]),
            "mission": MISSION, "label": LABEL,
            "period": float(r["period"]) if pd.notna(r["period"]) else float("nan"),
            "depth": float(r["depth"]) if pd.notna(r["depth"]) else float("nan"),
            "duration": float(r["duration"]) if pd.notna(r["duration"]) else float("nan"),
            "ra": float(r["ra"]) if pd.notna(r["ra"]) else float("nan"),
            "dec": float(r["dec"]) if pd.notna(r["dec"]) else float("nan"),
            "raw_path": str(path), **meta})
        have.add(tid)
        with LOCK:
            if len(rows) % 10 == 0:
                pd.DataFrame(rows, columns=dd.CAT_COLUMNS).to_csv(STAGE_CSV, index=False)
        print(f"[stage] fp {got + 1}/{PER_CLASS} ok ({tid})", flush=True)

    df = pd.DataFrame(rows, columns=dd.CAT_COLUMNS)
    df.to_csv(STAGE_CSV, index=False)
    if failures:
        pd.DataFrame(failures, columns=["target_id", "error"]).to_csv(
            outdir / "tess_stage_failures.csv", index=False)
    print(f"[stage] DONE wrote {STAGE_CSV} with {len(df)} fp targets", flush=True)


if __name__ == "__main__":
    main()