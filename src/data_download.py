"""Module 1 - download real Kepler/TESS Target Pixel Files from MAST and real
dispositions from the NASA Exoplanet Archive.

Two stages:
  1. build a *label pool* from the archive (KOI / TOI dispositions)
  2. fetch real Target Pixel Files with ``lightkurve`` and cache them as compact
     ``.npz`` products under ``data/raw/<mission>/`` so the run is resumable.

Usage:
    python -m src.data_download --mission kepler --n-targets 500 --outdir data/raw
    python -m src.data_download --mission tess   --n-targets 500 --outdir data/raw
"""
from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")

MISSION_STR = {"kepler": "Kepler", "tess": "TESS"}
CAT_COLUMNS = ["target_id", "archive_id", "mission", "label", "period", "depth",
               "duration", "ra", "dec", "raw_path", "n_cadences", "exptime",
               "product", "author"]


# --------------------------------------------------------------------- TAP ----
def _with_retry(fn, tries: int = 6, base: float = 3.0, what: str = ""):
    """Call ``fn`` with exponential backoff.

    MAST / the Exoplanet Archive intermittently fail DNS resolution or time out
    when hit repeatedly, so every network call is retried with backoff instead
    of being counted as a lost target.
    """
    import time as _time

    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:                     # noqa: BLE001 - deliberate
            last = e
            wait = base * (1.6 ** k)
            print(f"    [retry {k + 1}/{tries}] {what or 'request'} failed "
                  f"({type(e).__name__}); sleeping {wait:.0f}s", flush=True)
            _time.sleep(wait)
    raise last


def tap(query: str, timeout: int = 240) -> pd.DataFrame:
    """Run a synchronous TAP/ADQL query and return the result as a DataFrame."""
    import requests

    def _once():
        r = requests.get(C.EXO_TAP, params={"query": query, "format": "csv"},
                         timeout=timeout)
        r.raise_for_status()
        if not r.text.strip():
            raise RuntimeError("empty TAP response")
        df = pd.read_csv(io.StringIO(r.text))
        if "QUERY_STATUS" in df.columns:
            raise RuntimeError(str(df.iloc[0].to_dict()))
        return df

    return _with_retry(_once, tries=6, base=4.0, what="exoarchive TAP")


def _label_pool(mission: str) -> pd.DataFrame:
    """One row per unique target with a binary label from real archive dispositions."""
    if mission == "kepler":
        disp = "','".join(C.KEPLER_PLANET_DISP + C.KEPLER_FP_DISP)
        q = ("select kepid,koi_disposition,koi_period,koi_depth,"
             "koi_duration,ra,dec from cumulative "
             f"where koi_disposition in ('{disp}')")
        df = tap(q)
        df["is_planet"] = df["koi_disposition"].isin(C.KEPLER_PLANET_DISP).astype(int)
        g = df.groupby("kepid")
        # Purity filter: drop the 47 stars hosting BOTH planet and FP KOIs
        # (label = max would silently mark them all PLANET).
        mixed = (g["is_planet"].max() == 1) & (g["is_planet"].min() == 0)
        pure = df[~df["kepid"].isin(mixed[mixed].index)].copy()
        g = pure.groupby("kepid")
        out = pd.DataFrame({
            "label": g["is_planet"].max(),
            "ra": g["ra"].first(),
            "dec": g["dec"].first(),
            "period": g["koi_period"].min(),
            "depth": g["koi_depth"].max(),
            "duration": g["koi_duration"].max(),
        }).reset_index()
        out["archive_id"] = out["kepid"].astype(str)
        out["target_id"] = "KIC " + out["archive_id"]
    else:
        disp = "','".join(C.TESS_PLANET_DISP + C.TESS_FP_DISP)
        q = ("select tid,tfopwg_disp,pl_orbper,pl_trandep,pl_trandurh,ra,dec "
             f"from toi where tfopwg_disp in ('{disp}')")
        df = tap(q)
        df["is_planet"] = df["tfopwg_disp"].isin(C.TESS_PLANET_DISP).astype(int)
        g = df.groupby("tid")
        # Purity filter: drop the 5 TICs with mixed planet/FP dispositions.
        mixed = (g["is_planet"].max() == 1) & (g["is_planet"].min() == 0)
        pure = df[~df["tid"].isin(mixed[mixed].index)].copy()
        g = pure.groupby("tid")
        out = pd.DataFrame({
            "label": g["is_planet"].max(),
            "ra": g["ra"].first(),
            "dec": g["dec"].first(),
            "period": g["pl_orbper"].min(),
            "depth": g["pl_trandep"].max(),
            "duration": g["pl_trandurh"].max(),
        }).reset_index()
        out["archive_id"] = out["tid"].astype(str)
        out["target_id"] = "TIC " + out["archive_id"]

    out["mission"] = mission
    return out[["target_id", "archive_id", "mission", "label",
                "period", "depth", "duration", "ra", "dec"]]


# ---------------------------------------------------------------- download ----
def _safe(target_id: str) -> str:
    return target_id.replace(" ", "_").replace("/", "_")


def _product_preference(mission: str, exptime: np.ndarray, n: int) -> list:
    """Order candidate products: long cadence for Kepler, 120 s for TESS."""
    if n == 0:
        return []
    exptime = np.asarray(exptime, float)
    if mission == "kepler":
        key = np.where(exptime >= 600, 0, 1)
    else:
        key = np.abs(np.nan_to_num(exptime, nan=1e6) - 120.0)
    return list(np.argsort(key, kind="stable"))


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an .npz through a temp file + os.replace (atomic on Windows).

    Readers therefore never observe a half-written product, and two fetchers
    racing on the same target cannot leave a corrupt file behind.
    """
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def fetch_tpf(target_id: str, mission: str, path: Path, tries: int = 5) -> dict | None:
    """Download one real TPF and cache the essential arrays to ``path`` (.npz)."""
    import lightkurve as lk

    sr = _with_retry(
        lambda: lk.search_targetpixelfile(target_id,
                                           mission=MISSION_STR[mission]),
        tries=4, base=3.0, what=f"MAST search {target_id}")
    if len(sr) == 0:
        return None
    tbl = sr.table.to_pandas()
    exptime = np.asarray(tbl.get("exptime", pd.Series([np.nan] * len(tbl))), float)
    order = _product_preference(mission, exptime, len(tbl))

    for i in order[:tries]:
        try:
            tpf = _with_retry(lambda i=i: sr[int(i)].download(), tries=3, base=2.5,
                              what=f"MAST download {target_id}")
        except Exception:
            continue
        if tpf is None:
            continue
        flux = np.asarray(tpf.flux.value, dtype=np.float32)
        if flux.ndim != 3:
            continue
        good = np.isfinite(flux).sum(axis=(1, 2))
        if int((good > 0).sum()) < 2 * C.TIME_BINS:
            continue
        time = np.asarray(tpf.time.value, dtype=np.float64)
        try:
            quality = np.asarray(tpf.quality, dtype=np.int32)
        except Exception:
            quality = np.zeros(len(time), np.int32)
        try:
            pmask = np.asarray(tpf.pipeline_mask, dtype=bool)
        except Exception:
            pmask = np.zeros(flux.shape[1:], bool)

        # memory/disk guard: uniformly thin very long light curves (a whole TESS
        # sector at 2-min cadence is ~18k cadences). Thinning keeps the gap
        # structure, so the transit signature survives.
        n = flux.shape[0]
        if n > C.MAX_CADENCES:
            keep = np.linspace(0, n - 1, C.MAX_CADENCES).astype(int)
            flux, time, quality = flux[keep], time[keep], quality[keep]

        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_savez(path, flux=flux, time=time, quality=quality,
                      pipeline_mask=pmask,
                      column=int(np.atleast_1d(getattr(tpf, "column", 0))[0]),
                      row=int(np.atleast_1d(getattr(tpf, "row", 0))[0]))
        sector = getattr(tpf, "sector", None)
        quarter = getattr(tpf, "quarter", None)
        prod = str(sector) if sector is not None else (
            str(quarter) if quarter is not None else "?")
        return {"n_cadences": int(flux.shape[0]),
                "exptime": float(np.nanmedian(exptime)) if exptime.size else float("nan"),
                "product": prod,
                "author": str(getattr(tpf, "author", ""))}
    return None


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Download real Kepler/TESS TPFs.")
    ap.add_argument("--mission", required=True, choices=list(C.MISSIONS))
    ap.add_argument("--n-targets", type=int,
                    default=C.N_TARGETS_TOTAL // len(C.MISSIONS))
    ap.add_argument("--outdir", default=str(C.RAW))
    ap.add_argument("--attempt-factor", type=float, default=4.0,
                    help="candidate-pool size multiplier (downloads can fail)")
    a = ap.parse_args()

    C.ensure_dirs()
    outdir = Path(a.outdir)
    (outdir / a.mission).mkdir(parents=True, exist_ok=True)

    pool = _label_pool(a.mission)
    print(f"[download] {a.mission}: label pool = {len(pool)} unique targets "
          f"(planet={int((pool.label == 1).sum())}, fp={int((pool.label == 0).sum())})")

    per_class = max(1, a.n_targets // 2)
    cat = outdir / f"{a.mission}_catalog.csv"
    # --- resumable: keep already-catalogued targets so a killed run loses nothing
    prev = None
    if cat.exists():
        try:
            prev = pd.read_csv(cat)
            # keep rows whose file still exists
            prev = prev[prev["raw_path"].map(lambda p: Path(str(p)).exists())]
        except Exception:
            prev = None
    have_ids = set(map(str, prev["target_id"].tolist())) if prev is not None and len(prev) else set()
    rows = prev.to_dict("records") if prev is not None and len(prev) else []
    failures = []
    for label, tag in ((1, "planet"), (0, "false_positive")):
        got = sum(1 for r in rows if int(r["label"]) == label)
        if got >= per_class:
            print(f"  [{tag}] already have {got}/{per_class}, skipping", flush=True)
            continue
        sub = pool[pool.label == label]
        sub = sub.sample(frac=1.0, random_state=C.SEED).reset_index(drop=True)
        budget = min(len(sub), int(per_class * a.attempt_factor))
        for _, r in sub.head(budget).iterrows():
            if got >= per_class:
                break
            target_id = str(r["target_id"])
            if target_id in have_ids:
                continue
            path = outdir / a.mission / f"{_safe(target_id)}.npz"
            meta = None
            if path.exists():
                try:
                    with np.load(path) as z:
                        meta = {"n_cadences": int(z["flux"].shape[0]),
                                "exptime": float("nan"),
                                "product": "cached", "author": "cached"}
                except Exception:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            if meta is None:
                try:
                    print(f"  [{tag}] fetching {target_id} ...", flush=True)
                    meta = fetch_tpf(target_id, a.mission, path)
                except Exception as e:            # network / MAST problems
                    failures.append((target_id, repr(e)))
                    meta = None
            if meta is None:
                continue
            got += 1
            have_ids.add(target_id)
            rows.append({
                "target_id": target_id,
                "archive_id": str(r["archive_id"]),
                "mission": a.mission,
                "label": int(r["label"]),
                "period": float(r["period"]) if pd.notna(r["period"]) else np.nan,
                "depth": float(r["depth"]) if pd.notna(r["depth"]) else np.nan,
                "duration": float(r["duration"]) if pd.notna(r["duration"]) else np.nan,
                "ra": float(r["ra"]) if pd.notna(r["ra"]) else np.nan,
                "dec": float(r["dec"]) if pd.notna(r["dec"]) else np.nan,
                "raw_path": str(path),
                **meta,
            })
            print(f"  [{tag}] {got}/{per_class} ok  ({target_id})", flush=True)
            if len(rows) % 10 == 0:  # incremental checkpoint every 10 targets
                pd.DataFrame(rows, columns=CAT_COLUMNS).to_csv(cat, index=False)
        if got < per_class:
            print(f"[download] WARNING: {a.mission}/{tag} only {got}/{per_class}")
    df = pd.DataFrame(rows, columns=CAT_COLUMNS)
    df.to_csv(cat, index=False)
    print(f"[download] wrote {cat} with {len(df)} targets "
          f"({int((df.label == 1).sum())} planet / {int((df.label == 0).sum())} fp)")
    if failures:
        pd.DataFrame(failures, columns=["target_id", "error"]).to_csv(
            outdir / f"{a.mission}_failures.csv", index=False)


if __name__ == "__main__":
    main()

