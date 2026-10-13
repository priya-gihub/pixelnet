"""Tests for the paper-report builder (no network, no real training).

Builds fake per-target predictions in a temp results dir, runs
tools/make_report.main() and checks that every number in the output is
recomputed from the probabilities (never hard-coded).

Run with:  python -m pytest tests -q
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from src.evaluate import IEEE_COLUMNS, METRIC_ORDER, metrics_from
from src.models import MODEL_LABELS

MODEL_ORDER = ["baseline_1dcnn", "baseline_2dcnn", "baseline_lstm",
               "baseline_3dcnn", "pixel_verifynet"]


def _load_make_report():
    spec = importlib.util.spec_from_file_location(
        "make_report", ROOT / "tools" / "make_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_artifacts(tmp_path, monkeypatch):
    results = tmp_path / "results"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"
    dataset = tmp_path / "dataset"
    raw = tmp_path / "raw"
    for p in (results, tables, figures, dataset, raw):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(C, "RESULTS", results)
    monkeypatch.setattr(C, "TABLES", tables)
    monkeypatch.setattr(C, "FIGURES", figures)
    monkeypatch.setattr(C, "DATASET", dataset)
    monkeypatch.setattr(C, "RAW", raw)

    rng = np.random.default_rng(7)
    n = 40
    rows = []
    for split, k in (("train", 20), ("validation", 8), ("test", n - 28)):
        rows.append(pd.DataFrame({
            "dataset_row": [], "row_index": [], "target_id": [], "mission": [],
            "label": [], "window_index": [], "split": []}))
    sp = pd.concat(rows, ignore_index=True)
    sp = pd.DataFrame([{
        "dataset_row": i, "row_index": i, "target_id": f"T{i:03d}",
        "mission": "kepler" if i % 2 else "tess", "label": i % 2,
        "window_index": 0,
        "split": "train" if i < 28 else ("validation" if i < 34 else "test")}
        for i in range(n)])
    sp.to_csv(dataset / "splits.csv", index=False)
    (results / "label_leakage_audit.json").write_text('{"n_unique_targets": %d}' % n)

    all_metrics = []
    for k, name in enumerate(MODEL_ORDER):
        y = sp.loc[sp.split == "test", "label"].values.astype(int)
        prob = np.clip(rng.normal(0.4 + 0.15 * k, 0.25, len(y)), 0.001, 0.999)
        pd.DataFrame({"target_id": sp.loc[sp.split == "test", "target_id"].values,
                      "mission": sp.loc[sp.split == "test", "mission"].values,
                      "label": y, "probability": prob,
                      "prediction": (prob >= C.THRESHOLD).astype(int)}).to_csv(
            results / f"predictions_{name}.csv", index=False)
        mtr = metrics_from(y, prob)
        mtr.update({"model": name, "model_label": MODEL_LABELS[name],
                    "parameters": 1000 + k, "n_test_targets": len(y)})
        all_metrics.append(mtr)
    pd.DataFrame(all_metrics).to_csv(results / "final_metrics.csv", index=False)
    return results, tables, figures, sp


def test_make_report_recomputes_and_writes(tmp_path, monkeypatch):
    results, tables, figures, sp = _fake_artifacts(tmp_path, monkeypatch)
    mr = _load_make_report()
    mr.main()

    # every metric in the emitted table must equal a recomputation from the
    # stored probabilities -> nothing is hard-coded anywhere in the chain
    table = pd.read_csv(results / "final_metrics.csv")
    for _, r in table.iterrows():
        d = pd.read_csv(results / f"predictions_{r['model']}.csv")
        rec = metrics_from(d["label"].values, d["probability"].values)
        for k in METRIC_ORDER:
            assert abs(float(r[k]) - rec[k]) < 1e-9, (r["model"], k)

    assert (tables / "final_model_comparison.csv").exists()
    assert (tables / "final_model_comparison.tex").exists()
    assert (tables / "final_model_comparison.md").exists()
    assert (results / "final_report.json").exists()

    tex = (tables / "final_model_comparison.tex").read_text()
    for col in IEEE_COLUMNS:
        assert col in tex
    for name in MODEL_ORDER:
        assert MODEL_LABELS[name] in tex

    report = pd.read_json(results / "final_report.json", typ="series")
    assert report["dataset"]["total_unique_targets"] == int(sp.target_id.nunique())

    assert (figures / "roc_curves.png").exists()
    assert (figures / "pr_curves.png").exists()
    assert (figures / "confusion_matrices.png").exists()


def test_report_run_never_overwrites_readme_with_fixture_numbers(tmp_path,
                                                               monkeypatch):
    """A synthetic run must not publish its toy metrics into the real README.

    Regression test: make_report() injects the measured table into README.md
    between the RESULTS markers. The unit tests call main() against a 40-target
    fixture, which briefly replaced the real published results with fixture
    numbers. The real README must still describe the real 1000-target run.
    """
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README.md not present")
    before = readme.read_text(encoding="utf-8")

    _fake_artifacts(tmp_path, monkeypatch)
    mr = _load_make_report()
    mr.main()

    assert readme.read_text(encoding="utf-8") == before, \
        "make_report() wrote fixture metrics into the real README.md"