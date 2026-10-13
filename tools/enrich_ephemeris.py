"""Merge the public catalog ephemeris (period, epoch, duration, depth) into the
download catalogs.

The ephemeris is public, class-agnostic information: BOTH planets and false
positives were detected by the same pipeline, so both have an ephemeris. It is
used only to FOLD the light curve and to compare measured vs. catalog values,
which is standard transit-vetting practice. The disposition (the label) is never
used as a feature.

Usage: python tools/enrich_ephemeris.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config as C
from src import data_download as dd

EPHEM_COLUMNS = ["period", "epoch", "duration", "depth"]


def _clean_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Drop suffixed duplicates left by earlier merges.

    An interrupted run can leave ``period_x``/``period_y`` columns behind; the
    plain ``period`` written by this tool is the authoritative one, so the
    suffixed copies are removed instead of shadowing it downstream.
    """
    import re

    drop = [c for c in d.columns
            if re.fullmatch(r"(period|depth|duration|epoch)_[xy]", c)]
    if drop:
        print(f"  dropping duplicate columns {drop}", flush=True)
        d = d.drop(columns=drop)
    return d


def ephemeris_for(mission: str) -> pd.DataFrame:
    """One row per target: period, transit-centre epoch, duration, depth."""
    if mission == "kepler":
        q = ("select kepid,koi_period,koi_time0bk,koi_duration,koi_depth "
             "from cumulative where kepid is not null")
        df = dd.tap(q)
        df = df.rename(columns={"kepid": "archive_id",
                                "koi_period": "period",
                                "koi_time0bk": "epoch",
                                "koi_duration": "duration",
                                "koi_depth": "depth"})
    else:
        q = ("select tid,pl_orbper,pl_tranmid,pl_trandurh,pl_trandep "
             "from toi where tid is not null")
        df = dd.tap(q)
        df = df.rename(columns={"tid": "archive_id",
                                "pl_orbper": "period",
                                "pl_tranmid": "epoch",
                                "pl_trandurh": "duration",
                                "pl_trandep": "depth"})
    df["archive_id"] = df["archive_id"].astype(str)
    for c in EPHEM_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # duration is stored in HOURS in both catalogs -> convert to days
    if mission == "kepler":
        df["duration"] = df["duration"] / 24.0
    else:
        df["duration"] = df["duration"] / 24.0
    g = df.groupby("archive_id")
    out = pd.DataFrame({
        "period": g["period"].median(),
        "epoch": g["epoch"].median(),
        "duration": g["duration"].median(),
        "depth": g["depth"].max(),
    }).reset_index()
    return out


def _norm_id(v) -> str:
    """TIC/KIC ids round-trip through CSV as '12345.0'; normalise to '12345'."""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if "." in s:
        s = s.split(".")[0]
    return s


def main():
    for mission in C.MISSIONS:
        cat = ROOT / "data" / "raw" / f"{mission}_catalog.csv"
        if not cat.exists():
            print(f"[{mission}] no catalog", flush=True)
            continue
        d = pd.read_csv(cat, dtype={"target_id": str})
        d["archive_id"] = d["archive_id"].map(_norm_id)
        d["label"] = pd.to_numeric(d["label"], errors="raise").astype(int)
        d = _clean_columns(d)
        eph = ephemeris_for(mission)
        d = d.drop(columns=[c for c in EPHEM_COLUMNS if c in d.columns],
                   errors="ignore")
        d = d.merge(eph, on="archive_id", how="left")
        d.to_csv(cat, index=False)
        ok = int(d["period"].notna().sum())
        print(f"[{mission}] ephemeris merged: {ok}/{len(d)} rows have a period "
              f"(columns {EPHEM_COLUMNS})", flush=True)
    print("ENRICH DONE", flush=True)


if __name__ == "__main__":
    main()