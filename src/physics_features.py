"""Module 5a - physics / morphological evidence from the pixel cubes.

Evidence is computed **only from the pixel data plus the public catalog
ephemeris** (period, transit-centre epoch, duration). The ephemeris is
class-agnostic - planets *and* false positives were detected by the same
pipeline, so both have one - and the disposition (the label) is never used.

Folding the light curve on the ephemeris is the standard transit-vetting step and
is what makes this evidence measurable at all: a single Kepler quarter contains
only 2-3 transits, so an unfolded dip is indistinguishable from noise, whereas the
folded curve stacks every observed transit onto the same phase.

Eight features (``config.PHYSICS_NAMES``):
    transit_depth         folded depth in the target pixel (sigma units)
    snr                   folded transit SNR
    duration_consistency  measured duration vs. the catalog duration
    periodicity           coherence gain from folding
    symmetry              odd/even depth agreement
    centroid_stability    centroid offset of the in-transit difference image
    neighbor_contrast     target-pixel dip vs. the strongest neighbouring dip
    secondary_eclipse     depth at phase 0.5 (blended eclipsing-binary test)

They are stored *raw* (unnormalised) in ``data/physics/physics.npy``. Any
standardisation happens later, inside ``train.py``/``evaluate.py``, fitted on the
training split only, so these features can never carry the label into the model.
"""
from __future__ import annotations

import warnings

import numpy as np

import config as C


def light_curve(tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aperture light curve: mean of the valid pixels at each time bin."""
    s = (tensor * mask).sum(axis=(1, 2))
    n = mask.sum(axis=(1, 2))
    return np.where(n > 0, s / np.maximum(n, 1e-6), np.nan)


N_PHASE_BINS = 32


def pixel_series(tensor: np.ndarray, mask: np.ndarray, y: int, x: int) -> np.ndarray:
    """Per-pixel time series (NaN where the pixel is not observed)."""
    sel = mask[:, y, x] > 0
    v = np.full(tensor.shape[0], np.nan, np.float64)
    v[sel] = tensor[sel, y, x]
    return v


def _robust(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, (mad if mad > 0 else float(np.std(x)) + 1e-8)


def bin_time_axis(time: np.ndarray) -> np.ndarray:
    """Normalise a binned time axis to [-0.5, 0.5) (bin centres)."""
    t = np.asarray(time, float)
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    if not np.isfinite(t1 - t0) or t1 <= t0:
        return np.zeros_like(t)
    return (t - t0) / (t1 - t0) - 0.5


def phase_of(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase of each sample in [-0.5, 0.5), 0 = predicted transit centre."""
    if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)):
        return np.full(len(time), np.nan)
    return ((np.asarray(time, float) - epoch + 0.5 * period) % period) / period - 0.5


def in_transit(phase: np.ndarray, duration_days: float, period: float) -> np.ndarray:
    """Boolean mask of samples inside the predicted transit."""
    if not (np.isfinite(period) and period > 0
            and np.isfinite(duration_days) and duration_days > 0):
        return np.zeros(len(phase), bool)
    half = 0.5 * duration_days / period
    return np.abs(np.asarray(phase, float)) <= half


def fold(series: np.ndarray, phase: np.ndarray, n_bins: int = N_PHASE_BINS):
    """Median of the series in each phase bin (NaN where empty)."""
    out = np.full(n_bins, np.nan)
    ok = np.isfinite(series) & np.isfinite(phase)
    if ok.sum() < 8:
        return out
    idx = np.clip(((phase[ok] + 0.5) * n_bins).astype(int), 0, n_bins - 1)
    s = series[ok]
    order = np.argsort(idx, kind="stable")
    idx, s = idx[order], s[order]
    bounds = np.searchsorted(idx, np.arange(n_bins + 1))
    for b in range(n_bins):
        seg = s[bounds[b]:bounds[b + 1]]
        if seg.size:
            out[b] = float(np.median(seg))
    return out


def folded_evidence(series: np.ndarray, phase: np.ndarray, duration_days: float,
                    period: float, n_bins: int = N_PHASE_BINS) -> dict:
    """Folded depth/SNR/duration/secondary-eclipse for one pixel series.

    The depth is measured on the RAW folded samples (not on phase bins): for a
    10 d period and a 0.1 d duration the transit occupies only ~0.2% of the
    cycle, so a 32-bin phase histogram would usually contain *no* in-transit bin
    at all. The phase bins are used only to estimate a robust out-of-transit
    baseline and its scatter, which needs many samples per bin.
    """
    out = {"depth": np.nan, "snr": np.nan, "duration": np.nan,
           "secondary": np.nan}
    ok = np.isfinite(series) & np.isfinite(phase)
    if ok.sum() < 12:
        return out
    s, ph = series[ok], phase[ok]

    if np.isfinite(period) and period > 0 and np.isfinite(duration_days) \
            and duration_days > 0:
        tr = np.abs(ph) <= 0.5 * duration_days / period
        sec = np.abs(np.abs(ph) - 0.5) <= 0.5 * duration_days / period
        guard = 3.0 * duration_days / period
    else:
        tr = np.abs(ph) <= 0.025
        sec = np.abs(np.abs(ph) - 0.5) <= 0.025
        guard = 0.075
    oot = ~tr & ~sec & (np.abs(ph) <= 0.5 - guard)
    if tr.sum() < 2 or oot.sum() < 8:
        return out

    fb = fold(s, ph, n_bins)
    fb_phase = (np.arange(n_bins) + 0.5) / n_bins - 0.5
    fb_oot = np.isfinite(fb) & ~in_transit(fb_phase, duration_days, period) \
        & (np.abs(fb_phase) <= 0.5 - guard)
    if fb_oot.sum() >= 4:
        base, scatter = _robust(fb[fb_oot])
    else:                                    # fall back to the raw samples
        base, scatter = _robust(s[oot])
    if not np.isfinite(base) or not np.isfinite(scatter) or scatter <= 0:
        return out

    depth = float(base - np.median(s[tr]))
    # `depth` is the dip size in robust sigma units; `snr` is the *detection*
    # significance, which also scales with how many in-transit samples were
    # stacked. These must NOT be the same number: returning depth/scatter for
    # both made transit_depth and snr bit-identical for every target, which
    # double-counted their combined PGSEV weight and wasted a model input.
    out["depth"] = depth / scatter
    out["snr"] = depth / scatter * np.sqrt(max(int(tr.sum()), 1))

    # measured duration: contiguous in-transit phase span that reaches 1 sigma
    if tr.sum() >= 2:
        p = np.sort(ph[tr])
        span = float(p.max() - p.min())
        extra = 2.0 * (p[1] - p[0]) if len(p) > 2 else 0.0
        out["duration"] = (span + extra) * period
    if sec.sum() >= 2 and oot.sum() >= 8:
        out["secondary"] = float(base - np.median(s[sec])) / scatter
    return out


def _runs(flags: np.ndarray) -> list:
    out, c = [], 0
    for v in flags:
        if v:
            c += 1
        elif c:
            out.append(c)
            c = 0
    if c:
        out.append(c)
    return out


def transit_depth(lc: np.ndarray) -> float:
    ok = np.isfinite(lc)
    if ok.sum() < 8:
        return np.nan
    med = float(np.median(lc[ok]))
    return float(max(0.0, med - np.percentile(lc[ok], 3)))


def snr(lc: np.ndarray) -> float:
    ok = np.isfinite(lc)
    if ok.sum() < 8:
        return np.nan
    med, mad = _robust(lc[ok])
    depth = max(0.0, med - float(np.percentile(lc[ok], 3)))
    return float(depth / mad)


def periodicity(lc: np.ndarray) -> float:
    ok = np.isfinite(lc)
    if ok.sum() < 16:
        return np.nan
    try:
        from astropy.timeseries import LombScargle

        t = np.arange(len(lc), dtype=float)[ok]
        y = np.asarray(lc, float)[ok]
        y = y - np.nanmean(y)
        ls = LombScargle(t, y)
        freq, power = ls.autopower(minimum_frequency=1.0 / 32,
                                   maximum_frequency=0.5, samples_per_peak=8)
        return float(np.nanmax(power))
    except Exception:
        return np.nan


def duration_consistency(lc: np.ndarray) -> float:
    ok = np.isfinite(lc)
    if ok.sum() < 8:
        return np.nan
    med, mad = _robust(lc[ok])
    below = (lc < med - 1.5 * mad) & ok
    runs = _runs(below)
    if not runs:
        return 0.0
    return float(max(runs) / sum(runs))


def symmetry(lc: np.ndarray) -> float:
    """Ingress/egress symmetry of the deepest dip (1 = perfectly symmetric)."""
    ok = np.isfinite(lc)
    if ok.sum() < 8:
        return np.nan
    y = np.asarray(lc, float)
    imin = int(np.nanargmin(np.where(ok, y, np.nan)))
    k = 3
    lo = max(0, imin - k)
    left = y[lo:imin][ok[lo:imin]]
    right = y[imin + 1:imin + 1 + k][ok[imin + 1:imin + 1 + k]]
    if left.size == 0 or right.size == 0:
        return np.nan
    a, b = float(left.mean()), float(right.mean())
    return float(1.0 - abs(a - b) / (abs(a) + abs(b) + 1e-8))


def centroid_stability(tensor: np.ndarray, mask: np.ndarray) -> float:
    """Centroid offset (in pixels) of the in-transit difference image.

    A genuine transit dims the *same* pixels, so the difference image is centred
    on the target; a blended/eclipsing companion shifts the centroid away.
    """
    lc = light_curve(tensor, mask)
    ok = np.isfinite(lc) & (mask.sum(axis=(1, 2)) > 0)
    if ok.sum() < 8:
        return np.nan
    thr = float(np.nanpercentile(lc[ok], 10))
    in_tr, out_tr = ok & (lc <= thr), ok & (lc > thr)
    if in_tr.sum() == 0 or out_tr.sum() == 0:
        return np.nan
    diff = np.nan_to_num(np.nanmean(tensor[out_tr], axis=0)
                         - np.nanmean(tensor[in_tr], axis=0))
    tot = float(diff.sum())
    if tot <= 0:
        return np.nan
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy = float((diff * yy).sum() / tot)
    cx = float((diff * xx).sum() / tot)
    return float(np.hypot(cy - (h - 1) / 2, cx - (w - 1) / 2))


def dip_centroid(tensor: np.ndarray, mask: np.ndarray, phase: np.ndarray,
                 duration_days: float, period: float, ty: int, tx: int):
    """Significance-weighted centroid of the in-transit dip, and target share.

    A naive centroid over the raw difference image is useless: for a shallow
    transit the image is dominated by noise, ``diff.sum()`` hovers around zero
    and the "centroid" lands several pixels away for every target (measured:
    planet median 4.0 px, i.e. meaningless). Weighting each pixel by how far its
    dip exceeds the pixel-to-pixel noise floor keeps only the pixels that really
    dim, so a genuine transit is centred on the target while a blended or
    eclipsing companion shifts the weight onto a neighbour.

    Returns (offset_in_pixels_from_target, fraction_of_dip_on_target).
    """
    tr = np.isfinite(phase) & in_transit(phase, duration_days, period)
    oot = (np.isfinite(phase) & ~in_transit(phase, duration_days, period)
           & (mask.sum(axis=(1, 2)) > 0))
    if tr.sum() < 3 or oot.sum() < 5:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arr = np.where(mask > 0, tensor, np.nan)
        diff = np.nanmean(arr[oot], axis=0) - np.nanmean(arr[tr], axis=0)
    diff = np.where(mask.sum(axis=0) > 0, diff, np.nan)
    diff = np.nan_to_num(diff, nan=0.0)
    noise = float(np.median(np.abs(diff - np.median(diff)))) * 1.4826
    if not np.isfinite(noise) or noise <= 0:
        noise = float(diff.std()) + 1e-8
    w = np.clip(diff - 0.5 * noise, 0.0, None)      # significant dips only
    tot = float(w.sum())
    if not np.isfinite(tot) or tot <= 0:
        return np.nan, np.nan
    h, wd = diff.shape
    yy, xx = np.mgrid[0:h, 0:wd]
    cy = float((w * yy).sum() / tot)
    cx = float((w * xx).sum() / tot)
    share = float(w[ty, tx] / tot) if (0 <= ty < h and 0 <= tx < wd) else np.nan
    return float(np.hypot(cy - ty, cx - tx)), share


def target_pixel_index(tensor: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """The observed pixel closest to the centre of the cut window.

    cut_window() centres the frame on the target, so the target is the observed
    pixel nearest the window centre.
    """
    obs = mask.sum(axis=0) > 0
    h, w = obs.shape
    cy0, cx0 = (h - 1) / 2.0, (w - 1) / 2.0
    if not obs.any():
        return h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    d = (yy - cy0) ** 2 + (xx - cx0) ** 2
    d = np.where(obs, d, np.inf)
    flat = int(np.argmin(d))
    return flat // w, flat % w


def measure_depth(cube, times, period, epoch, duration):
    """Fold a light curve on an ephemeris and return (depth_ppm, snr)."""
    ok = np.isfinite(cube) & np.isfinite(times)
    if ok.sum() < 12:
        return float("nan"), float("nan")
    t, y = np.asarray(times)[ok], np.asarray(cube)[ok]
    if not (np.isfinite(period) and period > 0 and np.isfinite(duration)
            and duration > 0 and np.isfinite(epoch)):
        return float("nan"), float("nan")
    ph = ((t - epoch + 0.5 * period) % period) / period - 0.5
    tr = np.abs(ph) <= 0.5 * duration / period
    guard = 3.0 * duration / period
    oot = ~tr & (np.abs(ph) <= 0.5 - guard)
    if tr.sum() < 2 or oot.sum() < 8:
        return float("nan"), float("nan")
    base = float(np.median(y[oot]))
    _, mad = _robust(y[oot])
    if base <= 0 or not np.isfinite(mad) or mad <= 0:
        return float("nan"), float("nan")
    d = base - float(np.median(y[tr]))
    return float(d / base * 1e6), float(d / mad)


def compute_features(tensor: np.ndarray, mask: np.ndarray, bin_time: np.ndarray,
                     period: float = np.nan, epoch: float = np.nan,
                     duration: float = np.nan,
                     star: tuple | None = None,
                     raw_depth_ppm: float = np.nan) -> np.ndarray:
    """Ten physics features in ``config.PHYSICS_NAMES`` order (label-free).

    ``bin_time`` is the real centre time of every temporal bin and
    ``period``/``epoch``/``duration`` come from the public catalog ephemeris.
    They are used only to fold the light curve and to compare the measurement
    with the catalog; the disposition is never used.

    ``raw_depth_ppm`` is the absolute folded depth measured on the unnormalised
    star channel (computed once during preprocessing, where the raw flux is
    still available).
    """
    ty, tx = target_pixel_index(tensor, mask)
    if star is not None:
        sy, sx = int(star[0]), int(star[1])
        if 0 <= sy < mask.shape[1] and 0 <= sx < mask.shape[2] \
                and (mask[:, sy, sx] > 0).sum() >= 8:
            ty, tx = sy, sx          # the star pixel found from the RAW flux
    t = np.asarray(bin_time, float) if bin_time is not None else \
        np.arange(tensor.shape[0], dtype=float)

    usable = (np.isfinite(period) and period > 0 and np.isfinite(duration)
              and duration > 0 and np.isfinite(t).sum() >= 8)
    if usable:
        phase = phase_of(t, period, epoch if np.isfinite(epoch) else float(np.nanmin(t)))
        dur, per = duration, period
    else:
        # no usable ephemeris -> fold on a nominal 1-cycle axis so the features
        # stay finite (they simply carry less information)
        span = float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else 1.0
        per = max(span, 1e-3)
        phase = bin_time_axis(t)
        dur = 0.05 * per

    tgt = pixel_series(tensor, mask, ty, tx)
    ev = folded_evidence(tgt, phase, dur, per)

    # strongest neighbouring dip: the deepest folded dip on the neighbouring
    # pixels. Limited to the best-observed candidates so the cost stays bounded
    # (folding all 121 channels for 1000 targets is minutes of pure Python).
    cand = [(int((mask[:, y, x] > 0).sum()), y, x)
            for y in range(mask.shape[1]) for x in range(mask.shape[2])
            if (y, x) != (ty, tx)]
    cand.sort(reverse=True)
    best_nb = np.nan
    for n_obs, y, x in cand[:C.MAX_NEIGHBOUR_PIXELS]:
        if n_obs < 8:
            break
        d = folded_evidence(pixel_series(tensor, mask, y, x), phase, dur,
                            per)["depth"]
        if np.isfinite(d) and (not np.isfinite(best_nb) or d > best_nb):
            best_nb = d
    neighbour_contrast = (ev["depth"] - best_nb) if np.isfinite(best_nb) else np.nan

    # odd/even agreement of the raw samples inside the predicted transit
    odd_even = np.nan
    if usable:
        tr = np.isfinite(tgt) & in_transit(phase, dur, per)
        if tr.sum() >= 4:
            t0 = float(np.nanmin(t))
            n_orb = np.floor((np.nan_to_num(t, nan=t0) - t0) / per).astype(np.int64)
            odd = (n_orb % 2) == 0
            if (tr & odd).sum() >= 2 and (tr & ~odd).sum() >= 2:
                odd_even = float(abs(np.median(tgt[tr & odd])
                                     - np.median(tgt[tr & ~odd])))

    dur_consistency = np.nan
    if np.isfinite(ev["duration"]) and np.isfinite(dur) and dur > 0:
        dur_consistency = float(1.0 - abs(ev["duration"] - dur) / dur)

    # folding coherence: folded SNR relative to the strongest single-sample dip
    periodicity = np.nan
    ok = np.isfinite(tgt)
    # compare like with like: both sides are sigma-unit dip sizes (ev["snr"]
    # now also carries the sqrt(n_in_transit) stacking gain, so it must not be
    # used as the per-sample reference here)
    if ok.sum() >= 8 and np.isfinite(ev["depth"]):
        _, scatter = _robust(tgt[ok])
        if np.isfinite(scatter) and scatter > 0:
            raw_best = float(np.nanmax((np.nanmedian(tgt[ok]) - tgt[ok]) / scatter))
            periodicity = float(ev["depth"] / max(raw_best, 1e-6))

    cen_off, target_share = dip_centroid(tensor, mask, phase, dur, per, ty, tx)

    return np.asarray([ev["depth"], dur_consistency, periodicity, odd_even,
                       cen_off, ev["snr"], neighbour_contrast, ev["secondary"],
                       target_share, float(raw_depth_ppm)], dtype=np.float32)


def compute_all(processed_dir, outdir):
    """Build ``physics.npy`` (raw, NaNs preserved) + ``physics.csv``."""
    from pathlib import Path

    import pandas as pd

    processed_dir, outdir = Path(processed_dir), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    T = np.load(processed_dir / "tensors.npy")
    M = np.load(processed_dir / "masks.npy")
    idx = pd.read_csv(processed_dir / "index.csv")
    bt_path = processed_dir / "bin_times.npy"
    BT = np.load(bt_path) if bt_path.exists() else None

    def _num(row, key):
        v = row.get(key, np.nan)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return f if np.isfinite(f) else float("nan")

    feats = []
    for i in range(len(T)):
        try:
            bt = BT[i] if BT is not None and len(BT) > i else None
            row = idx.iloc[i]
            star = None
            if "star_y" in idx.columns and pd.notna(row.get("star_y")):
                star = (int(row["star_y"]), int(row["star_x"]))
            feats.append(compute_features(
                T[i], M[i], bt,
                period=_num(row, "period"),
                epoch=_num(row, "epoch"),
                duration=_num(row, "duration"),
                star=star,
                raw_depth_ppm=_num(row, "raw_depth_ppm")))
        except Exception:
            feats.append(np.full(C.N_PHYSICS_FEATURES, np.nan, np.float32))
    P = np.asarray(feats, np.float32)
    np.save(outdir / "physics.npy", P)

    df = idx.copy()
    for j, n in enumerate(C.PHYSICS_NAMES):
        df[n] = P[:, j]
    df.to_csv(outdir / "physics.csv", index=False)
    return P, df


def main():
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Extract the ten physics / PGSEV features.")
    ap.add_argument("--processed-dir", default=str(C.PROCESSED))
    ap.add_argument("--outdir", default=str(C.PHYSICS))
    a = ap.parse_args()
    C.ensure_dirs()
    P, _ = compute_all(a.processed_dir, a.outdir)
    n_nan = int(np.isnan(P).sum())
    print(f"[physics] physics.npy {P.shape} (NaN cells: {n_nan})")


if __name__ == "__main__":
    main()

