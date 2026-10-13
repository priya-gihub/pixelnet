"""End-to-end inference on one unseen target pixel file.

Accepts either a cached ``.npz`` product or a raw FITS/TPF file, runs the whole
chain (preprocess -> physics/PGSEV -> model) and prints a decision.

Usage:
    python -m src.inference_pipeline --tpf-path data/raw/kepler/KIC_11601584.npz \
        --checkpoint checkpoints/pixel_verifynet.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from src.models import build
from src.pgsev import REFERENCE, _norm
from src.physics_features import compute_features
from src.preprocessing import build_from_arrays, build_target
from src.train import apply_physics_stats, device


def ephemeris_from_catalogs(path: Path):
    """Look up (period, epoch, duration) for a cached product in the catalogs."""
    import pandas as pd

    stem = path.stem.replace("_", " ")  
                   # TIC_123 -> TIC 123
    for mission in C.MISSIONS:
        for f in (C.RAW / f"{mission}_catalog.csv",
                  *sorted((C.RAW).glob(f"{mission}_stage*.csv"))):
            if not f.exists() or "failure" in f.name:
                continue
            try:
                d = pd.read_csv(f, dtype={"target_id": str})
            except Exception:
                continue
            hit = d[d["target_id"].astype(str) == stem]
            if len(hit) and "period" in hit.columns:
                r = hit.iloc[0]

                def g(k):
                    try:
                        v = float(r.get(k))
                        return v if np.isfinite(v) else float("nan")
                    except (TypeError, ValueError):
                        return float("nan")
                return g("period"), g("epoch"), g("duration")
    return float("nan"), float("nan"), float("nan")


def load_arrays(path: Path):
    """Return (flux, time, quality, pipeline_mask) from .npz or FITS/TPF."""
    if path.suffix.lower() == ".npz":
        with np.load(path) as z:
            return (np.asarray(z["flux"], np.float32), np.asarray(z["time"], np.float64),
                    np.asarray(z["quality"]), np.asarray(z["pipeline_mask"], bool))
    import lightkurve as lk

    tpf = lk.read(str(path))
    try:
        q = np.asarray(tpf.quality, np.int32)
    except Exception:
        q = np.zeros(len(np.asarray(tpf.time.value)), np.int32)
    try:
        pm = np.asarray(tpf.pipeline_mask, bool)
    except Exception:
        pm = np.zeros(np.asarray(tpf.flux.value).shape[1:], bool)
    return (np.asarray(tpf.flux.value, np.float32),
            np.asarray(tpf.time.value, np.float64), q, pm)


def default_tpf_path():
    """A real held-out TEST target product (falls back to any cached product)."""
    sp = C.DATASET / "splits.csv"
    if sp.exists():
        import pandas as pd

        idx = pd.read_csv(C.PROCESSED / "index.csv")
        te = pd.read_csv(sp)
        te = te[te.split == "test"]
        if len(te):
            merged = te.merge(idx[["target_id", "raw_path"]], on="target_id")
            for p in merged["raw_path"]:
                if Path(str(p)).exists():
                    return str(p)
    for m in C.MISSIONS:
        for f in sorted((C.RAW / m).glob("*.npz")):
            return str(f)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Single-target inference.")
    ap.add_argument("--tpf-path", default=None,
                    help="cached .npz or raw FITS/TPF; defaults to a held-out test target")
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINTS / "pixel_verifynet.pt"))
    ap.add_argument("--threshold", type=float, default=C.THRESHOLD)
    # optional catalog ephemeris: when omitted it is looked up from the raw
    # catalogs, so inference folds on exactly the same evidence as training
    ap.add_argument("--period", type=float, default=float("nan"))
    ap.add_argument("--epoch", type=float, default=float("nan"))
    ap.add_argument("--duration", type=float, default=float("nan"))
    a = ap.parse_args(argv)

    C.ensure_dirs()
    dev = device()
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = build(ck.get("model", "pixel_verifynet")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    path = Path(a.tpf_path) if a.tpf_path else Path(default_tpf_path() or "")
    if not str(path) or not path.exists():
        raise SystemExit("no TPF product available for inference")

    period, epoch, duration = a.period, a.epoch, a.duration
    if not np.isfinite(period):
        period, epoch, duration = ephemeris_from_catalogs(path)
    flux, time, quality, pmask = load_arrays(path)
    # the ephemeris is folded in DURING preprocessing (raw folded depth) as well as
    # in the physics features, so it must be resolved before build_from_arrays
    tensor, mask, info = build_from_arrays(flux, time, quality, pmask,
                                           period=period, epoch=epoch,
                                           duration=duration)

    phys_raw = compute_features(
        tensor, mask,
        bin_time=info.get("bin_time"),
        period=period,
        epoch=epoch,
        duration=duration,
        star=tuple(info["star_pixel"]) if "star_pixel" in info else None,
        raw_depth_ppm=info.get("raw_depth_ppm", np.nan))
    # phys_raw is (n_features,); add the sample axis so the stored train-fitted
    # stats are applied exactly as during training -> scaled is (1, n_features)
    scaled = apply_physics_stats(phys_raw[None, :], ck["physics_stats"])

    with torch.no_grad():
        x = torch.tensor(np.asarray(tensor, np.float32), device=dev)[None]
        m = torch.tensor(np.asarray(mask, np.float32), device=dev)[None]
        p = torch.tensor(np.asarray(scaled, np.float32), device=dev)
        prob = float(torch.sigmoid(model(x, m, p)).item())

    # per-feature evidence, then the same weighted mean used everywhere else;
    # an unavailable (NaN) feature is neutral evidence (0.5), NOT zero, and is
    # reported so a missing measurement cannot silently drag the score down
    evidence = {n: float(_norm(*REFERENCE[n], float(v)))
                for n, v in zip(C.PHYSICS_NAMES, phys_raw)}
    missing = [n for n, v in evidence.items() if not np.isfinite(v)]
    ev = np.asarray([0.5 if not np.isfinite(evidence[n]) else evidence[n]
                      for n in C.PHYSICS_NAMES], np.float64)
    w = np.asarray([C.PGSEV_WEIGHTS[n] for n in C.PHYSICS_NAMES], np.float64)
    pgsev = float((ev * (w / w.sum())).sum())

    out = {"target": path.stem, "source": str(path),
           "prediction": "PLANET" if prob >= a.threshold else "FALSE POSITIVE",
           "probability": prob, "threshold": a.threshold, "pgsev": pgsev,
           "model": ck.get("model", "pixel_verifynet"),
           "ephemeris": {"period": None if not np.isfinite(period) else period,
                         "epoch": None if not np.isfinite(epoch) else epoch,
                         "duration": None if not np.isfinite(duration) else duration},
           "physics_features": {n: (None if not np.isfinite(v) else float(v))
                                for n, v in zip(C.PHYSICS_NAMES, phys_raw)},
           "physics_evidence": evidence,
           "physics_evidence_missing": missing,
           "valid_fraction": float(info["valid_fraction"]),
           "pixel_coverage": int(info["pixel_coverage"])}
    (C.RESULTS / "inference.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
