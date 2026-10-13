"""Final integrity audit - run AFTER the full pipeline and BEFORE writing a paper.

Fails loudly (exit code 1) if anything is wrong; prints a PASS/FAIL table.
Checks:
  1. catalogs: exactly 250 planet + 250 FP per mission (1000 unique targets)
  2. every catalogued product exists on disk and is readable
  3. labels come from the real archive dispositions (live cross-check)
  4. no target_id appears in more than one split (train/val/test)
  5. no row_index appears in more than one split
  6. the physics scaler is train-only (stored inside every checkpoint)
  7. all five models were evaluated on exactly the same held-out test targets
  8. every reported metric equals a recomputation from the stored predictions

Usage: python tools/verify_final.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import config as C
from src.evaluate import METRIC_ORDER, metrics_from
from src.models import MODEL_LABELS

MODEL_ORDER = ["baseline_1dcnn", "baseline_2dcnn", "baseline_lstm",
               "baseline_3dcnn", "pixel_verifynet"]

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""), flush=True)
    return bool(ok)


def check_catalogs():
    total = 0
    for mission in C.MISSIONS:
        f = C.RAW / f"{mission}_catalog.csv"
        if not f.exists():
            check(f"{mission} catalog exists", False, str(f))
            continue
        d = pd.read_csv(f, dtype={"target_id": str}).drop_duplicates("target_id")
        n1, n0 = int((d.label == 1).sum()), int((d.label == 0).sum())
        total += len(d)
        check(f"{mission}: 250 planet / 250 FP, unique target_id",
              n1 == 250 and n0 == 250 and d.target_id.is_unique,
              f"planet={n1} fp={n0} rows={len(d)} unique={d.target_id.nunique()}")
        check(f"{mission}: labels are 0/1 only",
              set(d.label.unique()) <= {0, 1}, str(sorted(d.label.unique())))
    check("total unique targets == 1000", total == 1000, f"got {total}")

    missing = unreadable = 0
    for mission in C.MISSIONS:
        f = C.RAW / f"{mission}_catalog.csv"
        if not f.exists():
            continue
        for p in pd.read_csv(f)["raw_path"]:
            path = Path(str(p))
            if not path.exists():
                missing += 1
                continue
            try:
                with np.load(path) as z:
                    _ = z["flux"].shape
            except Exception:
                unreadable += 1
    check("every catalogued product exists on disk", missing == 0,
          f"missing={missing}")
    check("every catalogued product is readable", unreadable == 0,
          f"unreadable={unreadable}")


def check_labels():
    """Labels must equal the LIVE archive dispositions (no manual labelling)."""
    try:
        from src import data_download as dd
        for mission in C.MISSIONS:
            f = C.RAW / f"{mission}_catalog.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, dtype={"target_id": str}).drop_duplicates("target_id")
            pool = dd._label_pool(mission).set_index("target_id")["label"]
            j = d.set_index("target_id").join(pool, rsuffix="_archive")
            bad = j[j["label_archive"].isna() | (j["label"] != j["label_archive"])]
            check(f"{mission}: labels match the live archive dispositions",
                  len(bad) == 0, f"{len(bad)} mismatches of {len(j)}")
    except Exception as e:                                    # pragma: no cover
        check("archive label cross-check (needs network)", False, repr(e))


def check_splits():
    sp = C.DATASET / "splits.csv"
    if not sp.exists():
        check("splits.csv exists", False, str(sp))
        return None
    s = pd.read_csv(sp, dtype={"target_id": str})
    sets = {k: set(s.loc[s.split == k, "target_id"])
            for k in ("train", "validation", "test")}
    ov = {f"{a}|{b}": len(sets[a] & sets[b])
          for a, b in (("train", "validation"), ("train", "test"),
                       ("validation", "test"))}
    check("split is at TARGET level (zero target overlap)",
          all(v == 0 for v in ov.values()), str(ov))
    rows = {k: set(s.loc[s.split == k, "row_index"]) for k in sets}
    rov = {f"{a}|{b}": len(rows[a] & rows[b]) for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))}
    check("zero row overlap between splits", all(v == 0 for v in rov.values()),
          str(rov))
    for k in sets:
        sub = s[s.split == k]
        print(f"       {k:11s} targets={len(sets[k]):4d} rows={len(sub):4d} "
              f"planet={int(sub.label.sum()):4d} fp={int((sub.label == 0).sum()):4d} "
              f"kepler={int((sub.mission == 'kepler').sum()):4d} "
              f"tess={int((sub.mission == 'tess').sum()):4d}")
    audit = C.RESULTS / "label_leakage_audit.json"
    if audit.exists():
        a = json.loads(audit.read_text())
        check("dataset_builder audit reports zero target overlap",
              all(v == 0 for v in a["target_overlap_between_splits"].values()))
    return s


def check_models(s):
    for name in MODEL_ORDER:
        ck_path = C.CHECKPOINTS / f"{name}.pt"
        if not ck_path.exists():
            check(f"{name}: checkpoint exists", False, str(ck_path))
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        has = "physics_stats" in ck and "mean" in ck.get("physics_stats", {})
        check(f"{name}: physics scaler stored in checkpoint (fitted on train)",
              has, f"best_val_roc_auc={ck.get('val_roc_auc')}")

    te = set(s.loc[s.split == "test", "target_id"]) if s is not None else set()
    for name in MODEL_ORDER:
        pf = C.RESULTS / f"predictions_{name}.csv"
        if not pf.exists():
            check(f"{name}: predictions written", False, str(pf))
            continue
        p = pd.read_csv(pf, dtype={"target_id": str})
        check(f"{name}: evaluated on exactly the {len(te)} held-out test targets",
              set(p.target_id) == te,
              f"rows={len(p)} symmetric_difference={len(set(p.target_id) ^ te)}")

    mf = C.RESULTS / "final_metrics.csv"
    if not mf.exists():
        check("results/final_metrics.csv exists", False, str(mf))
        return
    table = pd.read_csv(mf).set_index("model")
    for name in MODEL_ORDER:
        pf = C.RESULTS / f"predictions_{name}.csv"
        if name not in table.index or not pf.exists():
            continue
        p = pd.read_csv(pf)
        rec = metrics_from(p.label.values, p.probability.values)
        same = all(abs(float(table.loc[name, k]) - rec[k]) < 1e-9 for k in METRIC_ORDER)
        check(f"{MODEL_LABELS[name]}: metrics recomputed from its predictions",
              same)
    check("one fixed threshold (0.5) for every model",
          bool((table["threshold"] == C.THRESHOLD).all()),
          str(sorted(set(table["threshold"]))))
    check("all five models present in the comparison",
          set(MODEL_ORDER) <= set(table.index), str(sorted(set(table.index))))


def main():
    check_catalogs()
    check_labels()
    s = check_splits()
    check_models(s)
    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{npass}/{len(results)} checks passed")
    (C.RESULTS).mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "final_verification.json").write_text(
        json.dumps([{"check": n, "pass": ok, "detail": d} for n, ok, d in results],
                   indent=2))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())