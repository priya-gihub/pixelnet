# PIXEL-VerifyNet

A Physics-Guided Scientific Evidence Verification Framework for Spatio-Temporal
Target Pixel Sequence Analysis in Kepler and TESS Exoplanet Detection.

## IMPORTANT — Read First

This codebase is a **complete, runnable implementation**. It downloads real Kepler
and TESS Target Pixel Files (TPFs) from NASA's MAST archive and real dispositions from
the NASA Exoplanet Archive, then runs the full pipeline:

download → match → validate → preprocess → physics / PGSEV features → dataset build →
train (baselines + PIXEL-VerifyNet) → evaluate → explainability → uncertainty →
ablation → inference.

**Network (verified on this machine).** The archives are reachable — a live test returned
 real Kepler quarter products for `search_targetpixelfile('11442793')` and 61 real TESS
sector products for `TIC 261136679`, and NASA Exoplanet Archive TAP queries return the KOI
and TOI tables. The download step therefore runs for real here, not against samples.

**Runtime.** Python 3.14.6, `torch` 2.14.0 **+cpu** (no CUDA). All networks are kept compact
so the 1,000-target experiment trains in a reasonable time on CPU.

**Scientific rule.** Every reported number comes from a real forward pass on the held-out
test targets. No metric is hard-coded, and no threshold, label or prediction is altered to
inflate scores.

## Setup

```bash
# Python 3.11+ works; this machine uses the system interpreter (3.14.6)
python -m venv venv
venv\Scripts\activate            # Windows      (bash: source venv/bin/activate)
pip install -r requirements.txt
```

## Pipeline (run in order)

```bash
# 1. Download Kepler + TESS catalogs and TPFs (real MAST; cached + resumable)
python -m src.data_download --mission kepler --n-targets 500 --outdir data/raw
python -m src.data_download --mission tess   --n-targets 500 --outdir data/raw

# 2. Match KIC<->KOI and TIC<->TOI
python -m src.data_matching --raw-dir data/raw --outdir data/matched

# 3. Validate a sample TPF from each mission
python -m src.validation --matched-dir data/matched

# 4. Preprocess into tensors + validity masks
python -m src.preprocessing --matched-dir data/matched --outdir data/processed

# 5. Extract physics features + run PGSEV
python -m src.pgsev --processed-dir data/processed --outdir data/physics

# 6. Build final windowed dataset (target-level split)
python -m src.dataset_builder --processed-dir data/processed \
    --physics-dir data/physics --outdir data/dataset

# 7. Train baselines + PIXEL-VerifyNet
python -m src.train --dataset-dir data/dataset --model baseline_1dcnn
python -m src.train --dataset-dir data/dataset --model baseline_2dcnn
python -m src.train --dataset-dir data/dataset --model baseline_lstm
python -m src.train --dataset-dir data/dataset --model baseline_3dcnn
python -m src.train --dataset-dir data/dataset --model pixel_verifynet

# 8. Evaluate every model on the held-out test targets
python -m src.evaluate --dataset-dir data/dataset

# 9. Explainability (Grad-CAM + SHAP) and uncertainty (MC Dropout + ECE)
python -m src.explainability --checkpoint checkpoints/pixel_verifynet.pt --sample-idx 0
python -m src.uncertainty --checkpoint checkpoints/pixel_verifynet.pt --dataset-dir data/dataset

# 10. Ablation study (Pixel-only -> +Physics -> +Mask -> Full)
python -m src.ablation --dataset-dir data/dataset

# 11. Run the full inference pipeline on one new/unseen TPF
python -m src.inference_pipeline --tpf-path path/to/new_target.fits \
    --checkpoint checkpoints/pixel_verifynet.pt
```

## Final comparison (the deliverable)



Written to `results/final_metrics.csv` and `results/final_metrics.json`, with raw per-target
probabilities in `results/predictions_<model>.csv` and an IEEE-ready table in
`tables/final_model_comparison.csv`. All five models share one protocol: same targets, same
target-level split, same preprocessing, same loss, same seed and the same fixed threshold (0.5).

## Directory Layout

```
.
├── config.py                 # central config: paths, hyperparameters
├── requirements.txt
├── src/
│   ├── data_download.py      # Module 1: MAST download (lightkurve/astroquery)
│   ├── data_matching.py      # Module 2: KIC<->KOI / TIC<->TOI matching
│   ├── validation.py         # Module 3: FITS/TPF structural validation
│   ├── preprocessing.py      # Module 4: cleaning, masking, tensor building
│   ├── physics_features.py   # Module 5a: depth/duration/periodicity/etc.
│   ├── pgsev.py              # Module 5b: Physics-Guided Evidence Verification
│   ├── dataset_builder.py    # Module 6: windowing + target-level split
│   ├── models.py             # baselines + PIXEL-VerifyNet architecture
│   ├── train.py              # Module 7: training loop
│   ├── evaluate.py           # Module 8: metrics on held-out test set
│   ├── explainability.py     # Grad-CAM + SHAP
│   ├── uncertainty.py        # MC Dropout + Expected Calibration Error
│   ├── ablation.py           # 5-configuration ablation study
│   └── inference_pipeline.py # single-TPF end-to-end inference
├── dashboard/app.py          # Streamlit dashboard
├── tests/
│   └── test_basic.py         # unit tests (synthetic data, no network needed)
└── data/  results/  tables/  figures/  checkpoints/  logs/   # pipeline outputs
```

## Dataset protocol

- **Exactly 1,000 unique targets**, selected from real NASA Exoplanet Archive dispositions:
  - Kepler — `cumulative.koi_disposition`: `CONFIRMED` → PLANET, `FALSE POSITIVE` → FP
  - TESS — `toi.tfopwg_disp`: `CP`/`KP` → PLANET, `FP`/`FA` → FP
- Balanced design (250 + 250 + 250 + 250): Kepler PLANET, Kepler FP, TESS PLANET, TESS FP.
- Ambiguous classes (`CANDIDATE`, `PC`, `APC`) are excluded from the headline result to avoid label noise.
- Configurable in `config.py` (`N_TARGETS_TOTAL`, `TARGETS_PER_STRATUM`, `MISSIONS`).

## Notes on honesty / scale

- All splits are done at the **star/target level**, never the window level, so windows from the
  same star can never appear in both train and test. `dataset_builder.py` asserts the three ID
  sets are pairwise disjoint and writes the evidence to `results/label_leakage_audit.json`.
- Physics features and PGSEV are derived **only from the pixel data**, never from the label.
- Downloads are cached and resumable: an interrupted run never re-downloads a target.
- TESS has fewer suitable targets than Kepler, so the balanced 250-per-stratum design is what keeps
  the comparison fair; cross-mission generalisation is reported with an explicit caveat rather
  than hidden.
- This README documents the target design; section-level results are filled in by the pipeline
  itself, never typed by hand.

## Verified results

Produced by the full pipeline on the **150 held-out test targets** (never trained on), at the fixed
threshold shared by every model. Every number below is recomputed by `tools/make_report.py`
from the per-target prediction CSVs in `results/`; `tools/verify_final.py` re-derives them
independently and asserts agreement.

<!-- RESULTS:BEGIN (generated by tools/make_report.py - do not edit by hand) -->

**Dataset.** 1000 unique targets, split 700 train / 150 validation / 150 test at the **target** level (zero target overlap).
Test split: 74 planets / 76 false positives, 75 Kepler / 75 TESS.
Tensors are `256 × 11 × 11`; threshold `0.5`, seed `42`, fixed for every model.



<!-- RESULTS:END -->

## Reproducing / auditing

```bash
python -m pytest tests -q                    # unit + regression tests
python tools/verify_final.py                 # 29 leakage / integrity / metric checks
python tools/check_feature_independence.py   # no two physics features are duplicates
python tools/check_inference_consistency.py 5  # single-target inference reproduces training
```

The last two guards exist because two real bugs slipped through earlier in this project and would
have silently degraded the published numbers:

- `transit_depth` and `snr` were computed from the identical expression, so two of the ten physics
  features were bit-identical for all 1,000 targets. That double-counted their combined PGSEV
  weight (0.12 + 0.06) on a single measurement and wasted one model input. `snr` now carries the
  `sqrt(n_in_transit)` stacking gain that makes it a genuine detection significance, and
  `tools/check_feature_independence.py` fails the build if any two features ever duplicate again.
- `src/inference_pipeline.py` rebuilt the tensor without the catalog ephemeris, so single-target
  inference folded on a nominal axis and reported a NaN physics vector. `tools/check_inference_consistency.py`
  replays the inference path on real dataset rows and asserts it reproduces the stored tensors,
  masks and features exactly.

## Honest limitations

- **Scale.** One quarter (Kepler) or one sector (TESS) per target, 1,000 targets total. The measured
  learning curve (`tools/learning_curve.py`) indicates that data quantity — not model capacity — is
  the binding constraint; reaching the ~0.98 accuracy implied by earlier drafts would require
  millions of targets. The multi-quarter / multi-sector ingestion needed to make that claim is
  future work and is deliberately not simulated.
- **Ablation ordering.** The fused `pixel_physics` variant now beats both
  single-branch variants (`physics_only` & `pixel_only` ), which supports the fusion
  claim. Note this was *not* true before the `transit_depth`/`snr` bug was fixed — the duplicated
  feature had been propping up the physics-only branch — so any ablation table from before that fix
  is not comparable and should be regenerated.
- **Checkpoint freshness.** `tools/finalize_all.py` invalidates checkpoints whenever the dataset,
  physics matrix or feature code changes, so a model can never be evaluated against features it was
  not trained on.

"# pixelnet" 
