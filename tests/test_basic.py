"""Unit + end-to-end smoke tests. No network access required.

Run with:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from src import models as M
from src.dataset_builder import target_level_split
from src.evaluate import metrics_from
from src.physics_features import in_transit, phase_of
from src.pgsev import evidence_matrix
from src.preprocessing import (bin_time, bin_times, build_from_arrays,
                               cut_window, target_pixel)


# ------------------------------------------------------------------- models ---
def test_model_shapes():
    x = torch.randn(2, C.TIME_BINS, C.HEIGHT, C.WIDTH)
    m = torch.ones_like(x)
    p = torch.randn(2, C.N_PHYSICS_FEATURES)
    for name, cls in M.MODELS.items():
        out = cls()(x, m, p)
        assert out.shape == (2,), name


def test_pixel_verifynet_branches_are_independent():
    x = torch.randn(3, C.TIME_BINS, C.HEIGHT, C.WIDTH)
    m = torch.ones_like(x)
    p = torch.randn(3, C.N_PHYSICS_FEATURES)
    full = M.PIXELVerifyNet()(x, m, p)
    pixel_only = M.PIXELVerifyNet(use_mask=False, use_physics=False)(x, m, p)
    assert full.shape == pixel_only.shape == (3,)


# ------------------------------------------------------------ preprocessing ---
def test_bin_time_gap_aware():
    values = np.ones((20, 2, 2), np.float32)
    valid = np.ones((20, 2, 2), bool)
    valid[5:15] = False                       # a 10-cadence gap
    out, msk = bin_time(values, valid, np.arange(20, dtype=float), 8)
    assert out.shape == (8, 2, 2) and msk.shape == (8, 2, 2)
    assert msk.mean() < 1.0                   # the gap must appear in the mask
    assert np.allclose(out[msk > 0], 1.0)


def test_cut_window_pads_out_of_frame():
    img = np.ones((3, 5, 5), np.float32)
    patch = cut_window(img, 2, 2, 11, 11)
    assert patch.shape == (3, 11, 11)
    assert int(np.isfinite(patch).sum()) == 3 * 25


def test_build_from_arrays_with_ephemeris_measures_depth():
    """The full per-target build must accept the ephemeris and measure ppm depth.

    This is the path every real target takes; it previously raised a NameError
    that silently dropped every target (preprocessing kept 0 of 636).
    """
    rng = np.random.default_rng(11)
    n, h, w = 600, 7, 7
    period, dur, t0 = 5.0, 0.4, 100.0
    t = np.linspace(t0, t0 + 12 * period, n)
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    flux = np.full((n, h, w), 1.0e5, np.float32)
    intr = np.abs(phase) <= 0.5 * dur / period
    flux[intr, 3, 3] *= (1.0 - 1.0e-3)          # 1000 ppm dip on the centre
    flux += rng.normal(0, 30, flux.shape).astype(np.float32)
    quality = np.zeros(n, np.int32)
    pmask = np.zeros((h, w), bool)
    pmask[2:5, 2:5] = True

    tensor, mask, info = build_from_arrays(flux, t, quality, pmask,
                                           period=period, epoch=t0, duration=dur)
    assert tensor.shape == (C.TIME_BINS, C.HEIGHT, C.WIDTH)
    # the aperture centroid is the window centre, and the ephemeris refers to it
    assert info["star_pixel"] == [C.HEIGHT // 2, C.WIDTH // 2]
    assert np.isfinite(info["raw_depth_ppm"]), info
    assert 700.0 < info["raw_depth_ppm"] < 1300.0, info["raw_depth_ppm"]
    # per-pixel normalisation leaves valid samples centred and sigma-scaled
    sel = mask > 0
    assert abs(float(np.median(tensor[sel]))) < 0.2
    assert 0.3 < float(tensor[sel].std()) < 3.0
    assert np.isfinite(tensor).all() and (tensor[~sel] == 0).all()


def test_bin_times_track_the_time_axis():
    from src.preprocessing import bin_times

    t = np.arange(100, dtype=float)
    valid = np.ones(100, bool)
    valid[40:60] = False                  # a gap must not shift the centres
    centres = bin_times(t, valid, 10)
    assert centres.shape == (10,)
    ok = np.isfinite(centres)
    assert ok.sum() == 8, centres          # the gap occupies two empty bins
    assert centres[ok].min() < 40 and centres[ok].max() > 60
    assert np.all(np.diff(centres[ok]) > 0)          # centres stay ordered


def test_target_pixel_prefers_pipeline_mask():
    flux = np.zeros((2, 5, 5), np.float32)
    pmask = np.zeros((5, 5), bool)
    pmask[1, 2] = True                       # single-pixel aperture -> unambiguous
    assert target_pixel(flux, pmask) == (1, 2)


# ------------------------------------------------------------------- pgsev ----
def test_pgsev_evidence_bounded():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(64, C.N_PHYSICS_FEATURES)).astype(np.float32)
    P[0, :] = np.nan                      # NaNs must stay bounded, not explode
    E = evidence_matrix(P)
    assert E.shape == P.shape
    finite = np.isfinite(E)
    assert finite.any()
    assert np.all((E[finite] >= 0.0) & (E[finite] <= 1.0))
    assert np.all(np.isnan(E[np.isnan(P)]))       # missing -> missing, not 0/1


def test_pgsev_priors_cover_every_feature():
    assert set(C.PGSEV_PRIORS) == set(C.PHYSICS_NAMES)
    assert set(C.PGSEV_WEIGHTS) == set(C.PHYSICS_NAMES)
    assert abs(sum(C.PGSEV_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------- physics -----
def test_folded_depth_detects_a_planted_transit():
    """A clean periodic transit must produce a large folded depth + SNR."""
    from src.physics_features import compute_features, fold

    n, period, dur, depth = 400, 10.0, 0.6, 2.5
    t = np.linspace(0.0, 4 * period, n)
    phase = phase_of(t, period, 0.0)
    lc = rng_noise(n, 0.3) - depth * in_transit(phase, dur, period)
    tensor = np.zeros((n, 3, 3), np.float32)
    tensor[:, 1, 1] = lc
    tensor[:, 0, 1] = rng_noise(n, 0.3)      # neighbour stays flat
    mask = np.ones((n, 3, 3), np.float32)
    feats = compute_features(tensor, mask, t, period=period, epoch=0.0,
                             duration=dur)
    d = dict(zip(C.PHYSICS_NAMES, feats))
    assert d["transit_depth"] > 2.0, d
    assert d["snr"] > 2.0, d
    assert d["neighbor_contrast"] > 0.5, d
    assert phase_of(np.array([0.0, 5.0, 10.0]), 10.0, 0.0).round(3).tolist() \
        == [-0.0, -0.5, 0.0]


def rng_noise(n, sigma, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def test_depth_and_snr_are_not_the_same_measurement():
    """transit_depth and snr must be distinct features.

    Regression test: folded_evidence() once returned depth/scatter for BOTH,
    making the two features bit-identical for every real target. That silently
    double-counted their combined PGSEV weight (0.12 + 0.06) on a single
    measurement and wasted one of the ten model inputs.
    """
    from src.physics_features import compute_features

    n, period, dur, depth = 400, 10.0, 0.6, 2.5
    t = np.linspace(0.0, 4 * period, n)
    tensor = np.zeros((n, 3, 3), np.float32)
    tensor[:, 1, 1] = rng_noise(n, 0.3) - depth * in_transit(
        phase_of(t, period, 0.0), dur, period)
    mask = np.ones((n, 3, 3), np.float32)

    feats = compute_features(tensor, mask, t, period=period, epoch=0.0,
                             duration=dur)
    d = dict(zip(C.PHYSICS_NAMES, feats))
    assert d["transit_depth"] != d["snr"], d
    # snr stacks n in-transit samples, so it must exceed the single-sample depth
    assert d["snr"] > d["transit_depth"] > 0.0, d
    # and the ratio must be the sqrt(n_in) stacking gain, not 1.0
    assert abs(d["snr"] / d["transit_depth"] - 1.0) > 0.1, d


def test_physics_matrix_has_no_exactly_duplicated_features():
    """No two physics features may be exact duplicates of each other.

    Guards the whole feature vector at once, on the real 1000-target matrix
    when it is available.
    """
    f = C.PHYSICS / "physics.npy"
    if not f.exists():
        pytest.skip("physics matrix not built")
    P = np.load(f).astype(np.float64)
    assert P.shape[1] == len(C.PHYSICS_NAMES)
    for j, a in enumerate(C.PHYSICS_NAMES):
        for k, b in enumerate(C.PHYSICS_NAMES):
            if k <= j:
                continue
            m = np.isfinite(P[:, j]) & np.isfinite(P[:, k])
            if m.sum() < 8:
                continue
            assert not np.allclose(P[m, j], P[m, k], rtol=0, atol=0), \
                f"{a} and {b} are identical for every target"


# ------------------------------------------------------------------- split ----
def test_split_is_target_level():
    rng = np.random.default_rng(0)
    n = 400
    targets = pd.DataFrame({"target_id": [f"T{i}" for i in range(n)],
                            "mission": rng.choice(["kepler", "tess"], n),
                            "label": rng.integers(0, 2, n)})
    s = target_level_split(targets, C.SEED)
    assert len(s) == n and s.index.is_unique
    counts = s.value_counts()
    assert abs(counts["train"] / n - 0.70) < 0.05
    assert abs(counts["validation"] / n - 0.15) < 0.05
    assert abs(counts["test"] / n - 0.15) < 0.05


# ----------------------------------------------------------------- metrics ----
def test_metrics_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    perfect = metrics_from(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["accuracy"] == 1.0 and perfect["fpr"] == 0.0
    assert perfect["roc_auc"] == 1.0
    worst = metrics_from(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert worst["accuracy"] == 0.0 and worst["fpr"] == 1.0


# ------------------------------------------------------- end-to-end smoke -----
def _write_fake_dataset(processed: Path, n: int = 60):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, n)
    tensors = rng.normal(0, 1, (n, C.TIME_BINS, C.HEIGHT, C.WIDTH)).astype(np.float32)
    for i in range(n):
        if labels[i] == 1:                    # learnable transit-like dip
            tensors[i, 10:14, 4:7, 4:7] -= 2.5
    processed.mkdir(parents=True, exist_ok=True)
    np.save(processed / "tensors.npy", tensors)
    np.save(processed / "masks.npy", np.ones_like(tensors))
    physics = rng.normal(0, 1, (n, C.N_PHYSICS_FEATURES)).astype(np.float32)
    physics[:, 0] += labels * 1.5
    np.save(processed / "physics.npy", physics)
    pd.DataFrame({"row_index": range(n),
                  "target_id": [f"T{i:03d}" for i in range(n)],
                  "mission": ["kepler"] * n, "label": labels}).to_csv(
        processed / "index.csv", index=False)


def test_end_to_end_train_then_evaluate(tmp_path, monkeypatch):
    import src.dataset_builder as db
    import src.evaluate as ev
    import src.train as tr

    processed = tmp_path / "processed"
    _write_fake_dataset(processed)
    physics = tmp_path / "physics"
    physics.mkdir()
    np.save(physics / "physics.npy", np.load(processed / "physics.npy"))
    dataset = tmp_path / "dataset"
    results = tmp_path / "results"
    tables = tmp_path / "tables"
    ckpts = tmp_path / "checkpoints"
    for p in (results, tables, ckpts):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(C, "RESULTS", results)
    monkeypatch.setattr(C, "TABLES", tables)
    monkeypatch.setattr(C, "CHECKPOINTS", ckpts)

    db.main(["--processed-dir", str(processed), "--physics-dir", str(physics),
             "--outdir", str(dataset)])
    splits = pd.read_csv(dataset / "splits.csv")
    sets = {s: set(splits.loc[splits.split == s, "target_id"]) for s in
            ("train", "validation", "test")}
    assert not (sets["train"] & sets["test"])
    assert not (sets["train"] & sets["validation"])
    assert not (sets["validation"] & sets["test"])

    tr.main(["--dataset-dir", str(dataset), "--model", "pixel_verifynet",
             "--epochs", "3", "--patience", "3", "--batch-size", "8"])
    assert (ckpts / "pixel_verifynet.pt").exists()

    ev.main(["--dataset-dir", str(dataset), "--models", "pixel_verifynet"])
    table = pd.read_csv(results / "final_metrics.csv")
    assert len(table) == 1
    for col in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"):
        assert col in table.columns
        assert 0.0 <= float(table.loc[0, col]) <= 1.0
    assert (tables / "final_model_comparison.csv").exists()
    assert (results / "predictions_pixel_verifynet.png").exists() is False
    assert (results / "predictions_pixel_verifynet.csv").exists()


def test_cli_help_and_empty_test_guard(tmp_path):
    import src.evaluate as ev
    import src.train as tr

    for mod, args in ((tr, ["--help"]), (ev, ["--help"])):
        try:
            mod.main(args)
        except SystemExit as e:  # argparse --help exits 0
            assert e.code == 0, e.code

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame({"dataset_row": [], "row_index": [], "target_id": [],
                  "mission": [], "label": [], "window_index": [],
                  "split": []}).to_csv(dataset / "splits.csv", index=False)
    np.save(dataset / "tensors.npy",
            np.zeros((0, C.TIME_BINS, C.HEIGHT, C.WIDTH), np.float32))
    np.save(dataset / "masks.npy",
            np.zeros((0, C.TIME_BINS, C.HEIGHT, C.WIDTH), np.float32))
    np.save(dataset / "physics.npy",
            np.zeros((0, C.N_PHYSICS_FEATURES), np.float32))
    try:
        ev.main(["--dataset-dir", str(dataset)])
    except SystemExit as e:
        assert "no test rows" in str(e)
    else:  # pragma: no cover
        raise AssertionError("evaluate must refuse a dataset with no test rows")

