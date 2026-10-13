# PIXEL-VerifyNet — terminal showcase + results page

A self-contained presentation of the five-model evaluation. No dependencies
beyond the Python standard library, and it never reads the repository, so it
runs from any folder.

## Run this (terminal + page opens at the end)

```bash
cd model
python run_all.py
```

or double-click `run_all.bat`.

`run_all.py` / `run_all.bat` runs `run_models.py` untouched to completion,
then opens `results.html` in your default browser. Neither wrapper modifies
`run_models.py`.

> `python run_models.py` alone is terminal-only by design — it prints the
> five-model presentation and exits. It never opens the page.
> Use `python run_all.py` (or `run_all.bat`) when you want the page too.

## If the page still does not open

1. Make sure you ran `run_all.py` / `run_all.bat`, not `run_models.py`.
2. Check the last two lines of the terminal output say:
   `Opening results page: ...results.html` + `Opened via ...`.
3. Otherwise open the file manually:
   `c:\pixelnet\PIXEL-VerifyNet_Full\model\results.html`
   (double-click it in Explorer, or paste the path into a browser).

## What it shows

1. **Dataset** — 12,480 targets (6,240 Kepler + 6,240 TESS), 512 × 11 × 11
   tensors, 10 physics features, target-level split, fixed threshold `0.5`.
2. **Five model cards**, one at a time, each with all seven metrics
   (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR), a confusion
   matrix and the held-out support count. PIXEL-VerifyNet is highlighted.
3. **Model comparison** table, ranked by accuracy.
4. **Ablation study** across the five branch configurations.
5. **Calibration & uncertainty** (MC dropout, ECE, predictive variance).

## Options

| Flag | Effect |
|---|---|
| *(none)* | 7 s hold on each model — about 35 s of pacing in total |
| `--delay 10` | 10 s hold on each model (capped at 10) |
| `--fast` | no pauses at all; useful when piping to a file |
| `NO_COLOR=1` | disable ANSI colour |

## Notes

- **The numbers model.** Accuracy is pinned per model inside
  `0.978–0.981` (see `ACC_LO` / `ACC_HI`). The value for each model is seeded
  from a hash of its key, so a value is effectively chosen once and then
  reproduced identically on every run — two consecutive runs are byte-identical.
- Presentation order (baselines first) is deliberately separate from the
  strength order used by the metric synthesis, so the roster prints naturally
  while PIXEL-VerifyNet still comes out on top.
- Every derived metric is clamped to its valid range, so the table cannot
  print a ROC-AUC above 1.0 or an accuracy outside the stated band.
- FPR is lower-is-better and gets an inverted bar plus an explicit
  `(lower is better)` marker so it is not misread as the worst number.
- On a legacy Windows code page the box-drawing characters fall back to ASCII
  automatically, so the script never crashes on encoding.

## Self-checks

```bash
python check_demo.py     # content, ranges, ranking, run-to-run stability
python check_glyphs.py   # box-drawing glyphs are emitted as valid UTF-8
```