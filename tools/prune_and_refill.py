"""Post-fetch top-up: merge stage catalogs, drop ambiguous targets, refill to the
exact 250/250/250/250 design using pure targets from the live archive pools.

Why: the first fetch pass lost targets to transient MAST/DNS failures and ran
before the purity filter existed, so this job repairs the catalogs afterwards.
It is fully resumable - progress is written after every target.

Run detached:
    start /b cmd /c python -u tools\\prune_and_refill.py >> logs\\refill.log 2>&1
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import config as C
from src import data_download as dd


def _f(v):
    """NaN-safe float for catalog columns."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if pd.notna(f) else float("nan")


def append_row(cat: Path, row: dict) -> pd.DataFrame:
    """Append one catalog row, re-reading the file each time.

    Re-reading keeps any columns added by other tools (e.g. the archive
    ephemeris merged in by tools/enrich_ephemeris.py) and guarantees the file on
    disk is never rewritten from a stale in-memory copy.
    """
    df = pd.read_csv(cat, dtype={"target_id": str, "archive_id": str})
    df = df.drop_duplicates("target_id", keep="first")
    new = pd.DataFrame([row])
    for c in df.columns:
        if c not in new.columns:
            new[c] = pd.NA
    for c in new.columns:
        if c not in df.columns:
            df[c] = pd.NA
    new = new[list(df.columns)]
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(cat, index=False)
    return df


def repair_mission(mission: str, want: int) -> int:
    cat = ROOT / "data" / "raw" / f"{mission}_catalog.csv"
    if not cat.exists():
        print(f"[{mission}] no catalog yet, skipping", flush=True)
        return 0

    df = pd.read_csv(cat, dtype={"target_id": str})
    for stage in sorted(p for p in (ROOT / "data" / "raw").glob(f"{mission}_stage*.csv")
                        if "failure" not in p.name):
        sdf = pd.read_csv(stage, dtype={"target_id": str})
        sdf = sdf[[c for c in dd.CAT_COLUMNS if c in sdf.columns]]
        before = len(df)
        df = (pd.concat([df, sdf], ignore_index=True)
                .drop_duplicates("target_id", keep="first"))
        print(f"[{mission}] merged {stage.name}: {before} -> {len(df)}", flush=True)

    # 1) drop rows whose product is missing or unreadable
    keep = []
    for _, r in df.iterrows():
        p = Path(str(r["raw_path"]))
        if not p.exists():
            continue
        try:
            with dd.np.load(p) as z:
                if "flux" not in z:
                    continue
        except Exception:
            try:
                p.unlink()
            except OSError:
                pass
            continue
        keep.append(r["target_id"])
    df = df[df["target_id"].astype(str).isin(set(keep))].reset_index(drop=True)
    print(f"[{mission}] {len(df)} rows with a readable product", flush=True)

    # 2) drop ambiguous targets (present in the raw archive query but not in the
    #    purity-filtered pool)
    pool = dd._label_pool(mission)
    ok_ids = set(pool["target_id"].astype(str))
    bad = df[~df["target_id"].astype(str).isin(ok_ids)]
    if len(bad):
        print(f"[{mission}] removing {len(bad)} ambiguous targets: "
              f"{bad['target_id'].tolist()}", flush=True)
        for _, r in bad.iterrows():
            Path(str(r["raw_path"])).unlink(missing_ok=True)
        df = df[df["target_id"].astype(str).isin(ok_ids)].reset_index(drop=True)

    df.to_csv(cat, index=False)
    df = pd.read_csv(cat, dtype={"target_id": str})
    have = set(df["target_id"].astype(str))

    # 3) top up to exactly want/2 per class, using pure, unused pool targets
    per_class = want // 2
    for label, tag in ((1, "planet"), (0, "false_positive")):
        while True:
            got = int((df["label"] == label).sum())
            need = per_class - got
            if need <= 0:
                print(f"[{mission}/{tag}] complete ({got}/{per_class})", flush=True)
                break
            sub = pool[(pool["label"] == label)
                       & (~pool["target_id"].astype(str).isin(have))]
            if sub.empty:
                print(f"[{mission}/{tag}] pool exhausted at {got}/{per_class}",
                      flush=True)
                break
            sub = sub.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
            added = 0
            for _, r in sub.iterrows():
                if added >= need:
                    break
                tid = str(r["target_id"])
                path = ROOT / "data" / "raw" / mission / f"{dd._safe(tid)}.npz"
                try:
                    print(f"  [{tag}] {got + added + 1}/{per_class} fetching {tid}",
                          flush=True)
                    meta = dd.fetch_tpf(tid, mission, path)
                except Exception as e:
                    print(f"  [{tag}] FAIL {tid}: {type(e).__name__}", flush=True)
                    continue
                if meta is None:
                    continue
                have.add(tid)
                added += 1
                row = {"target_id": tid, "archive_id": str(r["archive_id"]),
                       "mission": mission, "label": int(label),
                       "period": _f(r["period"]), "depth": _f(r["depth"]),
                       "duration": _f(r["duration"]), "ra": _f(r["ra"]),
                       "dec": _f(r["dec"]), "raw_path": str(path), **meta}
                df = append_row(cat, row)               # incremental checkpoint
                print(f"  [{tag}] {got + added}/{per_class} ok ({tid})", flush=True)
            if added == 0:
                print(f"[{mission}/{tag}] no progress this round at {got}",
                      flush=True)
                break

    n1, n0 = int((df["label"] == 1).sum()), int((df["label"] == 0).sum())
    # final safety net: one row per target, product present on disk, and exactly
    # the per_class budget per label so the dataset is exactly `want` targets
    df = (df.drop_duplicates("target_id", keep="first")
            .reset_index(drop=True))
    df = df[[Path(str(p)).exists() for p in df["raw_path"]]].reset_index(drop=True)
    df = (df.groupby("label", group_keys=False)
            .head(per_class)
            .sort_values(["mission", "label", "target_id"])
            .reset_index(drop=True))
    df.to_csv(cat, index=False)
    n1, n0 = int((df["label"] == 1).sum()), int((df["label"] == 0).sum())
    print(f"[{mission}] catalog now {len(df)} targets "
          f"({n1} planet / {n0} fp)", flush=True)
    return min(n1, n0)


def main():
    worst = 10 ** 9
    for mission, want in (("kepler", 500), ("tess", 500)):
        worst = min(worst, repair_mission(mission, want))
    print(f"PRUNE_AND_REFILL DONE (smallest stratum={worst})", flush=True)


if __name__ == "__main__":
    main()
