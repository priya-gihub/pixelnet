"""Parallel stage worker: fetch TESS targets for one class while the main
refill job (tools/prune_and_refill.py) works on the other.

Safety:
  * products are written atomically (``dd._atomic_savez`` -> temp + os.replace),
    so two workers racing on the same target can never corrupt a file;
  * this worker owns its own stage catalog and never writes the mission
    catalog, which the running job owns;
  * every row is validated (pure pool + readable product) and the main job
    de-duplicates by target_id when it merges the stage file.

Launch detached:
    python -u tools\\fetch_tess_stage.py --label 1 >> logs\\stage_p.log 2>&1
    python -u tools\\fetch_tess_stage.py --label 0 >> logs\\stage_fp.log 2>&1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C
from src import data_download as dd

MISSION = "tess"


def existing_ids() -> set:
    ids = set()
    for csv in sorted((ROOT / "data" / "raw").glob(f"{MISSION}*.csv")):
        if "failure" in csv.name:
            continue
        try:
            ids |= set(pd.read_csv(csv, dtype={"target_id": str})["target_id"])
        except Exception as e:
            print(f"[stage] warn reading {csv.name}: {e}", flush=True)
    return ids


def record(row: dict, stage_csv: Path) -> None:
    """Append one row to the stage catalog, re-reading it each time.

    Re-reading keeps the file valid if another process (or an earlier run)
    appended rows in the meantime, and de-duplicates on target_id.
    """
    df = pd.read_csv(stage_csv, dtype={"target_id": str}).drop_duplicates(
        "target_id", keep="first") if stage_csv.exists() and \
        stage_csv.stat().st_size else pd.DataFrame(columns=dd.CAT_COLUMNS)
    new = pd.DataFrame([row])
    for c in df.columns:
        if c not in new.columns:
            new[c] = pd.NA
    for c in new.columns:
        if c not in df.columns:
            df[c] = pd.NA
    pd.concat([df, new[df.columns]], ignore_index=True).to_csv(
        stage_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=int, required=True, choices=[0, 1])
    ap.add_argument("--want", type=int, default=C.TARGETS_PER_STRATUM)
    ap.add_argument("--seed", type=int, default=C.SEED + 7)
    ap.add_argument("--tag", default="",
                    help="suffix for the stage file, so several workers can "
                         "fetch the same class without racing on one file")
    a = ap.parse_args()
    tag = "planet" if a.label == 1 else "false_positive"
    suffix = f"p{a.label}{a.tag}"
    stage_csv = ROOT / "data" / "raw" / f"{MISSION}_stage_{suffix}.csv"

    pool = dd._label_pool(MISSION)
    sub = pool[pool["label"] == a.label].copy()
    sub = sub.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    print(f"[stage] {tag} pool={len(sub)} want={a.want}", flush=True)

    rows = []
    have = existing_ids()
    print(f"[stage] already_present={len(have)}", flush=True)
    failures = []
    for _, r in sub.iterrows():
        got = len([x for x in rows if int(x["label"]) == a.label])
        if got >= a.want:
            break
        tid = str(r["target_id"])
        if tid in have:
            continue
        path = ROOT / "data" / "raw" / MISSION / f"{dd._safe(tid)}.npz"
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
                print(f"[stage] {got + 1}/{a.want} fetching {tid} ...", flush=True)
                meta = dd.fetch_tpf(tid, MISSION, path)
            except Exception as e:
                print(f"[stage] FAIL {tid}: {type(e).__name__}", flush=True)
                failures.append((tid, repr(e)))
                continue
        if meta is None:
            continue
        rows.append({"target_id": tid, "archive_id": str(r["archive_id"]),
                     "mission": MISSION, "label": a.label,
                     "period": float(r["period"]) if pd.notna(r["period"]) else float("nan"),
                     "depth": float(r["depth"]) if pd.notna(r["depth"]) else float("nan"),
                     "duration": float(r["duration"]) if pd.notna(r["duration"]) else float("nan"),
                     "ra": float(r["ra"]) if pd.notna(r["ra"]) else float("nan"),
                     "dec": float(r["dec"]) if pd.notna(r["dec"]) else float("nan"),
                     "raw_path": str(path), **meta})
        have.add(tid)
        record(rows[-1], stage_csv)          # incremental, crash-safe checkpoint

    df = pd.DataFrame(rows, columns=dd.CAT_COLUMNS)
    df.to_csv(stage_csv, index=False)
    if failures:
        pd.DataFrame(failures, columns=["target_id", "error"]).to_csv(
            ROOT / "data" / "raw" / f"{MISSION}_stage_{suffix}_failures.csv",
            index=False)
    print(f"[stage] DONE {stage_csv.name}: {len(df)} {tag} targets", flush=True)


if __name__ == "__main__":
    main()