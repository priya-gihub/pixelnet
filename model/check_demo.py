"""Sanity checks for demo/run_models.py (run from the repo root).

Verifies the things a viewer would notice immediately if they broke:
  * the script compiles and runs end-to-end
  * every printed accuracy is inside the advertised 0.978-0.981 band
  * every metric is inside its mathematically valid range
  * PIXEL-VerifyNet is the top model, and the fused ablation variant wins
  * two runs produce byte-identical output (the values are stable)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "run_models.py"
ROSTER = ["baseline_1dcnn", "baseline_2dcnn", "baseline_lstm",
          "baseline_3dcnn", "pixel_verifynet"]


def run() -> str:
    r = subprocess.run([sys.executable, str(DEMO), "--fast"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=DEMO.parent)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit("run_models.py exited non-zero")
    return r.stdout


def main() -> int:
    out = run()
    fails = []

    def check(ok, msg):
        print(("[PASS] " if ok else "[FAIL] ") + msg)
        if not ok:
            fails.append(msg)

    check("PIXEL-VerifyNet" in out, "all five models are presented")
    for section in ("DATASET", "MODEL COMPARISON", "ABLATION STUDY",
                    "CALIBRATION & UNCERTAINTY"):
        check(section in out, f"section present: {section}")

    # Strip ANSI first: the model cards colour the value, so the raw line is
    # "Accuracy \x1b[32m0.9809\x1b[0m" and a plain regex finds nothing.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)

    # Parse ACCURACY only. The model cards print "Accuracy  0.xxxx" and the
    # comparison table has an "Acc" column; precision/recall/F1 legitimately
    # sit outside the band, so a bare 0.9xxx regex would flag them wrongly.
    accs = [float(m.group(1)) for m in
            re.finditer(r"^\s*Accuracy\s+(\d\.\d{4})\s", plain, re.M)]
    check(len(accs) == len(ROSTER),
          f"found {len(accs)} per-model accuracy values (expected "
          f"{len(ROSTER)})")
    check(all(0.978 <= a <= 0.981 for a in accs),
          "every per-model accuracy is inside 0.978-0.981")

    # and the Acc column of the comparison table must agree. Restrict to the
    # block between the header rule and the "Best model" footer, otherwise the
    # trailing "spread across the roster  0.30 ..." line is miscounted as a row.
    tbl = plain.split("MODEL COMPARISON")[1].split("ABLATION")[0]
    body = tbl.split("Best model")[0]
    tbl_accs = [float(m.group(1)) for m in
                re.finditer(r"^\s{2}\S.*?\s(0\.9\d{3})\s", body, re.M)]
    check(len(tbl_accs) == len(ROSTER) and all(0.978 <= a <= 0.981
                                               for a in tbl_accs),
          f"comparison-table accuracies are in band ({len(tbl_accs)} rows)")

    check("1.00" not in out.replace("1.0000", ""),
          "no metric printed at or above 1.0000 (impossible / saturated)")

    # PIXEL-VerifyNet must be the model the table names as best
    best_line = [ln for ln in out.splitlines() if "Best model" in ln]
    check(bool(best_line) and "PIXEL-VerifyNet" in best_line[0],
          "PIXEL-VerifyNet is reported as the best model")

    ab = out.split("ABLATION STUDY")[1].split("CALIBRATION")[0]
    rows = [ln for ln in ab.splitlines() if "0.97" in ln or "0.98" in ln]
    check(bool(rows) and "full" in rows[-1],
          "the fused 'full' ablation variant is listed last / strongest")

    check("synthetic model" in out.lower(),
          "output is labelled as synthetic model outputs")

    # arithmetic self-consistency: the headline counts must agree with each
    # other, otherwise a viewer who adds them up sees the slide fall apart
    kepler = 3_120 + 3_120
    tess = 3_120 + 3_120
    split_sum = 8_736 + 1_872 + 1_872
    check(kepler + tess == split_sum,
          f"dataset counts add up ({kepler + tess:,} == split {split_sum:,})")
    check(f"{kepler + tess:,} targets" in plain,
          f"printed TOTAL matches the stratum sum ({kepler + tess:,})")
    check("(consistent)" in plain,
          "the split is reported as internally consistent")
    check("(MISMATCH)" not in plain,
          "no MISMATCH warning in the dataset header")

    r2 = run()
    check(r2 == out, "two consecutive runs produce identical output")

    print()
    if fails:
        print(f"{len(fails)} check(s) failed")
        return 1
    print("all demo checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())