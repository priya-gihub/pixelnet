"""Module 4 - cleaning, masking and tensor building.

Turns each cached TPF product (native cadence, native frame) into a fixed
``T x H x W`` flux cube plus a matching observation-validity mask.

Steps per target
  1. locate the target pixel (pipeline aperture centroid, else brightest pixel)
  2. cut an ``H x W`` window around it (padding with invalid pixels at the edge)
  3. drop non-finite and bad-quality cadences
  4. gap-aware temporal binning into ``T`` bins (mean of the valid cadences)
  5. robust per-target normalisation computed from *valid pixels only*

Usage:
    python -m src.preprocessing --matched-dir data/matched --outdir data/processed
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


def target_pixel(flux: np.ndarray, pmask: np.ndarray) -> tuple[int, int]:
    """Return (row, col) of the target in the native frame."""
    if pmask is not None and np.asarray(pmask).sum() > 0:
        ys, xs = np.nonzero(pmask)
        return int(round(ys.mean())), int(round(xs.mean()))
    med = np.nanmedian(flux, axis=0)
    if not np.isfinite(med).any():
        return flux.shape[1] // 2, flux.shape[2] // 2
    cy, cx = np.unravel_index(np.nanargmax(med), med.shape)
    return int(cy), int(cx)


def cut_window(flux: np.ndarray, cy: int, cx: int, h: int, w: int) -> np.ndarray:
    """Vectorised H x W cut around (cy, cx); outside-frame pixels become NaN."""
    ph, pw = h // 2 + 1, w // 2 + 1
    padded = np.pad(flux.astype(np.float32), ((0, 0), (ph, ph), (pw, pw)),
                    constant_values=np.nan)
    y0 = cy - h // 2 + ph
    x0 = cx - w // 2 + pw
    return padded[:, y0:y0 + h, x0:x0 + w]


def bin_time(values: np.ndarray, valid: np.ndarray, time: np.ndarray,
             n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Gap-aware temporal binning: mean of valid samples per bin, per pixel."""
    t = np.asarray(time, float)
    finite = np.isfinite(t)
    shape = (n_bins,) + values.shape[1:]
    if finite.sum() < 2:
        return np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    t0, t1 = float(np.nanmin(t[finite])), float(np.nanmax(t[finite]))
    if t1 <= t0:
        t1 = t0 + 1.0
    idx = np.clip(((t - t0) / (t1 - t0) * n_bins).astype(int), 0, n_bins - 1)
    idx = np.where(finite, idx, 0)

    num = np.zeros(shape, np.float64)
    den = np.zeros(shape, np.float64)
    v = np.where(valid, np.nan_to_num(values, nan=0.0), 0.0)
    np.add.at(num, idx, v)
    np.add.at(den, idx, valid.astype(np.float64))

    out = np.zeros(shape, np.float32)
    msk = np.zeros(shape, np.float32)
    good = den > 0
    out[good] = (num[good] / den[good]).astype(np.float32)
    msk[good] = 1.0
    return out, msk


def bin_times(time: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    """Centre time of every temporal bin (NaN where the bin is empty).

    The physics stage needs the real time of each bin to fold the light curve on
    the catalog ephemeris; bin index alone is not a time axis.
    """
    t = np.asarray(time, float)
    finite = np.isfinite(t)
    centres = np.full(n_bins, np.nan)
    if finite.sum() < 2:
        return centres
    t0, t1 = float(np.nanmin(t[finite])), float(np.nanmax(t[finite]))
    if t1 <= t0:
        t1 = t0 + 1.0
    idx = np.clip(((t - t0) / (t1 - t0) * n_bins).astype(int), 0, n_bins - 1)
    idx = np.where(finite, idx, 0)
    v = np.where(valid.any(axis=tuple(range(1, valid.ndim))) if valid.ndim > 1
                 else valid, t, np.nan)
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    np.add.at(num, idx, np.nan_to_num(v, nan=0.0))
    np.add.at(den, idx, np.isfinite(v).astype(float))
    centres[den > 0] = num[den > 0] / den[den > 0]
    return centres.astype(np.float64)


def build_target(path: str, period: float = np.nan, epoch: float = np.nan,
                 duration: float = np.nan) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build one (tensor, mask) pair from a cached .npz product."""
    with np.load(path) as z:
        flux = np.asarray(z["flux"], np.float32)
        time = np.asarray(z["time"], np.float64)
        quality = np.asarray(z["quality"])
        pmask = np.asarray(z["pipeline_mask"], bool)
    return build_from_arrays(flux, time, quality, pmask, period, epoch, duration)


def _folded_depth_ppm(series: np.ndarray, valid: np.ndarray, time: np.ndarray,
                      period: float, epoch: float, duration: float) -> float:
    """Absolute fractional transit depth (ppm) of the RAW star channel.

    This is the measurement that separates a planet from an eclipsing binary on
    physical grounds: a planet dims its star by ~1e-4 to 1e-3 of its flux, while
    a blended or eclipsing false positive typically dims it by 1e-2 or more.
    Per-pixel sigma units cannot express this, because every pixel is normalised
    by its own noise and a faint neighbour then looks as deep as the star.

    Returns NaN when the catalog ephemeris is unusable.
    """
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration)
            and duration > 0 and np.isfinite(epoch)):
        return float("nan")
    ok = np.isfinite(series) & valid & np.isfinite(time)
    if ok.sum() < 12:
        return float("nan")
    t = np.asarray(time, float)[ok]
    y = np.asarray(series, float)[ok]
    if not np.isfinite(np.median(y)) or np.median(y) <= 0:
        return float("nan")
    phase = ((t - epoch + 0.5 * period) % period) / period - 0.5
    tr = np.abs(phase) <= 0.5 * duration / period
    guard = 3.0 * duration / period
    oot = ~tr & (np.abs(phase) <= 0.5 - guard)
    if tr.sum() < 2 or oot.sum() < 8:
        return float("nan")
    base = float(np.median(y[oot]))
    if base <= 0:
        return float("nan")
    return float((base - np.median(y[tr])) / base * 1e6)


def normalize_pixels(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robust per-pixel normalisation (median / MAD computed along TIME).

    Why per pixel: a Target Pixel File mixes channels whose flux scales differ by
    orders of magnitude (target vs. neighbouring stars vs. background). Pooling
    every pixel into one median/MAD lets the brightest channel set the scale for
    all the others, which inflates them into large outliers and buries the
    transit signal (measured: pooled-MAD tensors had sigma ~8.9 instead of ~1).
    Normalising every pixel to its own robust baseline and scatter expresses each
    channel in units of its own noise, which is what makes a shallow transit
    visible and keeps pixel-to-pixel comparisons meaningful.

    Fully vectorised over the H x W plane; invalid samples stay exactly 0.
    """
    import warnings

    valid = mask > 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.where(valid, tensor.astype(np.float64), np.nan)
        med = np.nanmedian(v, axis=0)                       # (H, W)
        mad = np.nanmedian(np.abs(v - med), axis=0)          # (H, W)
        std = np.nanstd(v, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    scale = 1.4826 * np.where(np.isfinite(mad), mad, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale,
                     np.where(np.isfinite(std) & (std > 0), std, 0.0))
    enough = valid.sum(axis=0) >= 3
    scale = np.where(enough & (scale > 0), scale, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (tensor.astype(np.float64) - med) / scale
    z = np.where(np.isfinite(z) & valid, z, 0.0)
    return np.clip(z, -50.0, 50.0).astype(np.float32)


def build_from_arrays(flux: np.ndarray, time: np.ndarray, quality: np.ndarray,
                      pmask: np.ndarray, period: float = np.nan,
                      epoch: float = np.nan,
                      duration: float = np.nan) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build one (tensor, mask) pair from raw TPF arrays (used by inference too)."""
    flux = np.asarray(flux, np.float32)
    time = np.asarray(time, np.float64)
    quality = np.asarray(quality)
    pmask = np.asarray(pmask, bool)

    cy, cx = target_pixel(flux, pmask)
    patch = cut_window(flux, cy, cx, C.HEIGHT, C.WIDTH)

    bad = np.asarray(quality) != 0
    if bad.size == 0 or bad.mean() > 0.5:
        bad = np.zeros(flux.shape[0], bool)
    valid = np.isfinite(patch) & (~bad)[:, None, None]
    valid = valid.any(axis=(1, 2))[:, None, None] & valid

    tensor, mask = bin_time(patch, valid, time, C.TIME_BINS)
    centres = bin_times(time, valid.any(axis=(1, 2)), C.TIME_BINS)

    # Identify the star channel INSIDE the cut window.
    #
    # The archive ephemeris (KOI/TOI period, epoch, depth) was derived from the
    # *pipeline aperture* light curve, so the aperture centroid is by definition
    # the channel the catalog is talking about; cut_window centres the frame on
    # it, which puts it at the window centre. Falling back to the brightest raw
    # channel would silently change the measured depth, and after per-pixel
    # normalisation every channel has median 0 so brightness cannot be recovered
    # from the tensor anyway - it must be decided here, on the raw flux.
    star_y, star_x = C.HEIGHT // 2, C.WIDTH // 2
    if mask[:, star_y, star_x].sum() == 0:
        # the padded centre is not observed: take the brightest observed channel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_med = np.nanmedian(np.where(valid, patch, np.nan), axis=0)
        raw_med = np.where(valid.any(axis=0), raw_med, -np.inf)
        if np.isfinite(raw_med).any() and np.max(raw_med) > -np.inf:
            star_y, star_x = (int(v) for v in np.unravel_index(
                int(np.argmax(raw_med)), raw_med.shape))

    tensor = normalize_pixels(tensor, mask)

    info = {"target_pixel": [cy, cx], "star_pixel": [star_y, star_x],
            "valid_fraction": float(mask.mean()),
            "pixel_coverage": int((mask.sum(axis=0) > 0).sum()),
            "native_frame": [int(flux.shape[1]), int(flux.shape[2])],
            "cadences": int(flux.shape[0]),
            "bin_time": centres,
            "raw_depth_ppm": _folded_depth_ppm(
                patch[:, star_y, star_x], valid[:, star_y, star_x], time,
                period, epoch, duration)}
    return tensor, mask.astype(np.float32), info


def _num(row, key: str) -> float:
    """NaN-safe float lookup on a pandas Series (missing column -> NaN)."""
    v = row.get(key, np.nan)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Build T x H x W tensors + masks.")
    ap.add_argument("--matched-dir", default=str(C.MATCHED))
    ap.add_argument("--outdir", default=str(C.PROCESSED))
    a = ap.parse_args()

    C.ensure_dirs()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(Path(a.matched_dir) / "matched_all.csv")
    tensors, masks, rows, bad = [], [], [], []
    bin_time_rows: list = []
    for _, r in m.iterrows():
        per = _num(r, "period")
        ep = _num(r, "epoch")
        du = _num(r, "duration")
        try:
            t, mk, info = build_target(str(r["raw_path"]), per, ep, du)
        except Exception as e:
            bad.append({"target_id": str(r["target_id"]), "error": repr(e)})
            continue
        # NOTE: Kepler native frames are 5x5 and TESS 11x11, so a fixed 11x11 cut
        # pads the edges. Accept a target when enough *real* pixels survive
        # (>= 3x3), rather than on the raw mask fraction of the padded cube.
        if info["pixel_coverage"] < 9:
            bad.append({"target_id": str(r["target_id"]),
                        "error": f"pixel_coverage={info['pixel_coverage']}"})
            continue
        tensors.append(t)
        masks.append(mk)
        rows.append({"row_index": len(rows), "target_id": str(r["target_id"]),
                     "mission": str(r["mission"]), "label": int(r["label"]),
                     "raw_path": str(r["raw_path"]),
                     "valid_fraction": float(info["valid_fraction"]),
                     "pixel_coverage": int(info["pixel_coverage"]),
                     "cadences": int(info["cadences"]),
                     "star_y": int(info["star_pixel"][0]),
                     "star_x": int(info["star_pixel"][1]),
                     # public catalog ephemeris (class-agnostic); used only to fold
                     "period": per, "epoch": ep, "duration": du,
                     "raw_depth_ppm": float(info["raw_depth_ppm"]),
                     "catalog_depth": _num(r, "depth")})
        bin_time_rows.append(np.asarray(info["bin_time"], np.float64))

    if tensors:
        T = np.stack(tensors).astype(np.float32)
        M = np.stack(masks).astype(np.float32)
        BT = (np.stack(bin_time_rows).astype(np.float64)
              if bin_time_rows else np.zeros((0, C.TIME_BINS)))
    else:
        T = np.zeros((0, C.TIME_BINS, C.HEIGHT, C.WIDTH), np.float32)
        M = np.zeros_like(T)
        BT = np.zeros((0, C.TIME_BINS))
    np.save(outdir / "tensors.npy", T)
    np.save(outdir / "masks.npy", M)
    np.save(outdir / "bin_times.npy", BT)
    pd.DataFrame(rows).to_csv(outdir / "index.csv", index=False)

    report = {"tensors": list(T.shape), "targets_kept": len(rows),
              "targets_dropped": len(bad),
              "mean_valid_fraction": float(M.mean()) if M.size else 0.0,
              "nan_in_tensor": int(np.isnan(T).sum()),
              "by_mission": (pd.DataFrame(rows).groupby("mission").size().to_dict()
                             if rows else {})}
    (C.RESULTS / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    if bad:
        pd.DataFrame(bad).to_csv(C.RESULTS / "preprocessing_dropped.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"[preprocessing] wrote {outdir / 'tensors.npy'} {T.shape}")


if __name__ == "__main__":
    main()

