# PIXEL-VerifyNet — Project Completion Guide

> **What this file is:** a working document (not the public README) that explains *what* the
> project is, *where it stands right now*, and *exactly how it will be finished end-to-end*.
> `README.md` is the public-facing description; this file is the engineering roadmap.

---

## 1. What this project is

**PIXEL-VerifyNet** is a *Physics-Guided Scientific Evidence Verification Framework for
Spatio-Temporal Target Pixel Sequence Analysis in Kepler and TESS Exoplanet Detection.*

It decides whether an exoplanet candidate is a real **PLANET** or a **FALSE POSITIVE** using the
observatory's *raw pixel data* (Target Pixel Files), not a pre-extracted light curve.

Each target becomes a flux cube of shape **`T × H × W`** (T = 64 binned cadences, H × W = 11 × 11
pixels) plus a matching observation-validity **mask**. The network fuses three independent
evidence branches into one classifier:

| Branch | Input | Network | Purpose |
|---|---|---|---|
| Pixel branch | `T × H × W` flux cube | 3D CNN | transit morphology in space *and* time |
| Mask branch | `T × H × W` validity mask | 3D CNN | teaches the model to distrust gaps/artefacts |
| Physics / PGSEV branch | 10 physical features | MLP | folded depth, SNR, duration consistency, periodicity, odd/even symmetry, centroid stability, neighbour contrast, secondary eclipse, target-dip share, raw absolute depth |

**PGSEV** = *Pixel-Grounded Statistical Evidence Vector*: the physical evidence score computed
**only from the pixels**, never from the label.

---

## 2. The deliverable (why any of this matters)

One final, real, reproducible comparison table:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Baseline 1D CNN | | | | | | | |
| Baseline 2D CNN | | | | | | | |
| Baseline LSTM | | | | | | | |
| Baseline 3D CNN | | | | | | | |
| PIXEL-VerifyNet | | | | | | | |

…evaluated on **exactly 1,000 unique Kepler + TESS targets** with a **target-level split**, under
**one identical protocol** for all five models. Outputs: `results/final_metrics.csv`,
`results/final_metrics.json`, `results/predictions_<model>.csv`, and the IEEE-ready
`tables/final_model_comparison.csv`.

---

## 3. Current build status (as of this file)

Environment verified on this machine: **Python 3.14.6**, `torch 2.14.0+cpu` (no CUDA),
`numpy 2.5.2`, `pandas 2.3.3`, `scikit-learn 1.9.0`, `scipy 1.18.1`, `lightkurve 2.6.0`,
`astropy 8.0.1`, `astroquery 0.4.11`, `shap 0.52.0`, `captum 0.9.0`.
Missing and still to install: **seaborn, plotly, streamlit**.

**Network access is confirmed working here** (this is the important difference from a locked
sandbox): a live test returned **43 real Kepler quarter products** for `search_targetpixelfile('11442793')`
and **61 real TESS sector products** for `TIC 261136679`; the NASA Exoplanet Archive TAP service
returns the KOI table (`cumulative`, 9,564 rows) and the TOI table (`toi`, 8,148 rows).

| # | Module | File | Status |
|---|---|---|---|
| — | Central config | `config.py` | ✅ done |
| — | Dependencies | `requirements.txt` | ✅ done (+ pytest, seaborn, plotly, streamlit installed) |
| — | Package bootstrap | `src/__init__.py` | ✅ done |
| 1 | MAST download (real TPFs + real labels) | `src/data_download.py` | ✅ done, verified on live MAST |
| 2 | KIC↔KOI / TIC↔TOI matching | `src/data_matching.py` | ✅ done |
| 3 | FITS/TPF structural validation | `src/validation.py` | ✅ done |
| 4 | Cleaning, masking, tensor building | `src/preprocessing.py` | ✅ done |
| 5a | Physics features | `src/physics_features.py` | ✅ done |
| 5b | PGSEV | `src/pgsev.py` | ✅ done |
| 6 | Windowing + target-level split | `src/dataset_builder.py` | ✅ done |
| — | Baselines + PIXEL-VerifyNet | `src/models.py` | ✅ done (LSTM included) |
| 7 | Training loop | `src/train.py` | ✅ done |
| 8 | Metrics on held-out test set | `src/evaluate.py` | ✅ done |
| — | Grad-CAM + SHAP | `src/explainability.py` | ✅ written |
| — | MC Dropout + ECE | `src/uncertainty.py` | ✅ written |
| — | 5-configuration ablation | `src/ablation.py` | ✅ written |
| — | Single-TPF inference | `src/inference_pipeline.py` | ✅ written |
| — | Unit + end-to-end tests | `tests/test_basic.py` | ✅ 9/9 passing |
| — | Dashboard | `dashboard/app.py` | ✅ rewired to the new layout |
| — | Background runners | `tools/runpy.bat`, `tools/pip_install.bat` | ✅ done |
| — | Public description | `README.md` | ✅ done |

**Status: COMPLETE and verified.** The 1,000-target run finished end-to-end; every artifact below is
regenerated from real held-out predictions and re-checked by `tools/verify_final.py` (29/29 checks)
and `python -m pytest tests -q` (18/18 tests).

Final held-out test results (150 targets, threshold `0.5` fixed for all models):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---|---|---|---|---|---|
| Baseline 1D CNN | 0.5133 | 0.5043 | 0.7838 | 0.6138 | 0.4984 | 0.4926 | 0.7500 |
| Baseline 2D CNN | 0.5067 | 0.0000 | 0.0000 | 0.0000 | 0.5580 | 0.5695 | 0.0000 |
| Baseline LSTM | 0.5067 | 0.0000 | 0.0000 | 0.0000 | 0.4496 | 0.4549 | 0.0000 |
| Baseline 3D CNN | 0.5733 | 0.6250 | 0.3378 | 0.4386 | 0.6129 | 0.5640 | 0.1974 |
| **PIXEL-VerifyNet** | **0.6267** | **0.6552** | 0.5135 | **0.5758** | **0.6725** | **0.6349** | 0.2632 |

Ablation (same protocol, ROC-AUC): `pixel_physics` 0.6709 > `physics_only` 0.6487 > `pixel_only`
0.6133 > `pixel_mask` 0.6051 — i.e. fusing pixels with physics beats either branch alone, which is
the architectural claim the paper needs.

**Two bugs found and fixed during finalisation, both of which had corrupted earlier numbers:**

1. `transit_depth` and `snr` were computed from the identical expression, making two of the ten
   physics features bit-identical for all 1,000 targets and double-counting their combined PGSEV
   weight (0.12 + 0.06). Before the fix PIXEL-VerifyNet scored acc `0.7067` / AUC `0.7891`; the
   honest corrected figures are acc `0.6267` / AUC `0.6725`. **The earlier headline number was
   inflated by this bug and must not be quoted.** Guards added:
   `tools/check_feature_independence.py` and `tests/test_basic.py::test_physics_matrix_has_no_exactly_duplicated_features`.
2. `src/inference_pipeline.py` rebuilt tensors without the catalog ephemeris, so single-target
   inference folded on a nominal axis and emitted a NaN physics vector. Guard added:
   `tools/check_inference_consistency.py`, which replays inference on real rows and asserts the
   tensors, masks and features match the training artifacts bit-for-bit (currently max |Δ| = 0.0).

Also fixed: `make_report()` was injecting its own numbers into `README.md`, so the 40-target unit-test
fixture briefly overwrote the published results. It is now guarded to the real directories, with
`tests/test_report.py::test_report_run_never_overwrites_readme_with_fixture_numbers` preventing
regression.

---

## 4. Pipeline data flow

```
NASA Exoplanet Archive (TAP)          MAST (lightkurve)
   KOI / TOI dispositions   +    real Target Pixel Files
                │                         │
                └────────► src/data_download.py ────────► data/raw/<mission>/TIC_*.npz
                                                          data/raw/<mission>_catalog.csv
                                        │
                                        ▼
                            src/data_matching.py ─────────► data/matched/matched_all.csv
                                        │
                                        ▼
                            src/validation.py ────────────► results/validation_report.json
                                        │
                                        ▼
                            src/preprocessing.py ─────────► data/processed/{tensors,masks}.npy
                                                            data/processed/index.csv
                                        │
                                        ▼
        src/physics_features.py + src/pgsev.py ───────────► data/physics/{physics.npy,pgsev.csv}
                                        │
                                        ▼
                            src/dataset_builder.py ───────► data/dataset/{tensors,masks,physics}.npy
                                                            data/dataset/splits.csv
                                                            results/label_leakage_audit.json
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            src/train.py (×5 models)              src/models.py (architecture)
                    │                       baseline_1dcnn · baseline_2dcnn ·
                    ▼                       baseline_lstm · baseline_3dcnn ·
          checkpoints/<model>.pt            pixel_verifynet
                    │
                    ▼
            src/evaluate.py ─────────────► results/final_metrics.{csv,json}
                                           results/predictions_<model>.csv
                                           tables/final_model_comparison.csv
                    │
     ┌──────────────┼───────────────┬──────────────────┐
     ▼              ▼               ▼                  ▼
explainability  uncertainty     ablation        inference_pipeline
(Grad-CAM+SHAP) (MC Dropout+ECE) (5 configs)    (single TPF → decision)
```

---

## 5. Completion roadmap (do these in order)

### Step 0 — Environment
```bash
pip install seaborn plotly streamlit
```
Fill the remaining gaps in `requirements.txt` and record versions.

### Step 1 — Acquire the 1,000 real targets (RUNNING NOW, ~1-3 h total)
```bash
python -m src.data_download --mission kepler --n-targets 500 --outdir data/raw
python -m src.data_download --mission tess   --n-targets 500 --outdir data/raw
```
Balanced strata: 250 Kepler planets, 250 Kepler FPs, 250 TESS planets, 250 TESS FPs.
Running now as detached job `tools/fetch1000.py -> logs/fetch1000.log`
(resumable, per-10-target catalog checkpoints; status 2026-10-03: Kepler ~160/250
planets, TESS queued next). Poll with `python tools/status.py`.

> **Long jobs:** this machine's command runner stops after 30 s, so run these in the background:
> `start /b tools\runpy.bat -m src.data_download --mission kepler --n-targets 500 --outdir data/raw`
> then poll progress with `type logs\runner.log`.

**Gate:** `data/raw/kepler_catalog.csv` + `data/raw/tess_catalog.csv` together list **1,000**
targets with `label in {0,1}`. Then run `python tools/prune_and_refill.py` to drop
the 47 Kepler + 5 TESS ambiguous mixed-disposition stars the running job may have
picked up pre-purity-filter, and refill pure replacements (pool headroom verified:
Kepler 1928/4666, TESS 1274/1405).

### Step 2 — Match, validate, tensorise
```bash
python -m src.data_matching --raw-dir data/raw --outdir data/matched
python -m src.validation     --matched-dir data/matched
python -m src.preprocessing  --matched-dir data/matched --outdir data/processed
```
**Gate:** every target in `matched_all.csv` has one valid `256 × 11 × 11` tensor + mask, no NaN, and
appears exactly once. `256` temporal bins (not 64) because a Kepler long-cadence light curve spans
~33 d over ~1626 valid cadences: at 64 bins each bin is ~0.52 d and a ~0.1 d transit is averaged with
~4× more out-of-transit flux, which dilutes the very signal the pixel branch must see.

### Step 3 — Physics features + PGSEV
```bash
python -m src.pgsev --processed-dir data/processed --outdir data/physics
```
(`physics_features.py` is imported by `pgsev.py`; PGSEV is a documented weighted evidence score
computed only from pixels.)

**Gate:** `data/physics/physics.npy` has shape `(1000, 6)`, all finite, and the leakage audit shows
no single feature alone reproduces the label.

### Step 4 — Build the dataset and the target-level split
```bash
python -m src.dataset_builder --processed-dir data/processed \
    --physics-dir data/physics --outdir data/dataset
```
**Gate:** `splits.csv` has 700 / 150 / 150 rows; the three target-ID sets are **pairwise disjoint**;
`results/label_leakage_audit.json` records the proof.

### Step 5 — Train all five models (identical protocol)
```bash
python -m src.train --dataset-dir data/dataset --model baseline_1dcnn
python -m src.train --dataset-dir data/dataset --model baseline_2dcnn
python -m src.train --dataset-dir data/dataset --model baseline_lstm
python -m src.train --dataset-dir data/dataset --model baseline_3dcnn
python -m src.train --dataset-dir data/dataset --model pixel_verifynet
```
Same split, same preprocessing, same loss, same seed, same early-stopping rule (val ROC-AUC), same
fixed threshold (0.5). Checkpoints land in `checkpoints/<model>.pt`.

### Step 6 — Evaluate → the final table
```bash
python -m src.evaluate --dataset-dir data/dataset
```
Writes `results/final_metrics.csv`, `results/final_metrics.json`,
`results/predictions_<model>.csv`, `tables/final_model_comparison.csv`.

### Step 7 — Secondary analyses
```bash
python -m src.explainability     --checkpoint checkpoints/pixel_verifynet.pt --sample-idx 0
python -m src.uncertainty        --checkpoint checkpoints/pixel_verifynet.pt --dataset-dir data/dataset
python -m src.ablation           --dataset-dir data/dataset
python -m src.inference_pipeline --tpf-path data/raw/<mission>/<TARGET>.npz \
    --checkpoint checkpoints/pixel_verifynet.pt
```

### Step 8 — Tests + final verification
```bash
python -m pytest tests -q
```
Then verify the numbers in the summary by hand against the written artefacts.

---

## 6. Definition of done

1. `data/raw/*_catalog.csv` together describe **exactly 1,000 unique** targets
   (no duplicated `target_id`, balanced 250/250/250/250).
2. `data/dataset/splits.csv` shows a **target-level** 70 / 15 / 15 split with **no target in two
   splits** (asserted in code, not just observed).
3. Five models trained from the same split with the same protocol.
4. `results/final_metrics.{csv,json}` contain all seven required metrics for all five models,
   computed from **real forward passes** on the held-out test targets.
5. `results/predictions_<model>.csv` preserves the raw probabilities (audit trail).
6. `tables/final_model_comparison.csv` is the clean IEEE-ready table.
7. `python -m pytest tests -q` passes and the whole pipeline runs without manual steps.

## 7. Honesty rules (non-negotiable)

- **No hard-coded metrics.** Every number is produced by the evaluation code.
- **No threshold tuning** to inflate scores. Accuracy/Precision/Recall/F1/FPR use the fixed 0.5
  threshold; ROC-AUC and PR-AUC are threshold-free.
- **No label manipulation.** Labels come from real archive dispositions and are never edited
  during evaluation.
- **No leakage.** Splits happen at the star/target level; features never see the label; any
  dataset-wide normalisation statistic is computed on the training split only.
- **PGSEV/physics features must not copy the label.** They are computed from pixels only.

## 8. Risks and known constraints

| Risk | Mitigation |
|---|---|
| MAST download slow or partially failing | cache + resume; `--attempt-factor` oversamples the candidate pool; failures logged to `*_failures.csv` |
| CPU-only training | compact networks, modest epochs, early stopping |
| TESS has fewer usable targets than Kepler | balanced 250-per-stratum design; caveat reported, never hidden |
| Some TPFs have too few valid cadences | rejected at download time (`≥ 2 × TIME_BINS` cadences required) |
| Feature/label leakage in physics | automatic leakage audit written to `results/` |
| Command runner 30 s limit | long jobs run through `tools/runpy.bat` into `logs/runner.log` |

## 9. Files created by a full run

```
data/raw/kepler_catalog.csv, data/raw/tess_catalog.csv, data/raw/<mission>/*.npz
data/matched/matched_all.csv
data/processed/tensors.npy, masks.npy, index.csv
data/physics/physics.npy, physics.csv, pgsev.csv
data/dataset/tensors.npy, masks.npy, physics.npy, splits.csv, meta.json
results/final_metrics.csv, results/final_metrics.json
results/predictions_<model>.csv, results/<model>_history.json
results/label_leakage_audit.json, results/validation_report.json
results/uncertainty_calibration.json, results/ablation.csv
tables/final_model_comparison.csv
checkpoints/<model>.pt
logs/runner.log
```

