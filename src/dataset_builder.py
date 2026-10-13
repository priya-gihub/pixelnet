"""Module 6 - windowing + target-level split.

Assembles the final dataset and performs a **target-level** 70/15/15 split:
the split is applied to unique star identifiers, so no target can ever appear in
two splits. The disjointness is asserted in code and written, as evidence, to
``results/label_leakage_audit.json``.

Usage:
    python -m src.dataset_builder --processed-dir data/processed \
        --physics-dir data/physics --outdir data/dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C


def target_level_split(targets: pd.DataFrame, seed: int) -> pd.Series:
    """Return a Series mapping target_id -> 'train' / 'validation' / 'test'."""
    n = len(targets)
    strat_ok = (
        targets[["mission", "label"]].value_counts().min() >= 2
        and n >= 12
    )
    rest_frac = C.VAL_FRACTION + C.TEST_FRACTION
    if strat_ok:
        try:
            train, rest = train_test_split(
                targets, test_size=rest_frac,
                stratify=targets[["mission", "label"]], random_state=seed)
            val, test = train_test_split(
                rest, test_size=C.TEST_FRACTION / rest_frac,
                stratify=rest[["mission", "label"]], random_state=seed)
        except ValueError:
            strat_ok = False
    if not strat_ok:
        train, rest = train_test_split(targets, test_size=rest_frac,
                                       random_state=seed, shuffle=True)
        val, test = train_test_split(rest,
                                     test_size=C.TEST_FRACTION / rest_frac,
                                     random_state=seed, shuffle=True)
    mapping = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        for tid in part["target_id"]:
            mapping[str(tid)] = name
    return pd.Series(mapping, name="split")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Assemble dataset + target-level split.")
    ap.add_argument("--processed-dir", default=str(C.PROCESSED))
    ap.add_argument("--physics-dir", default=str(C.PHYSICS))
    ap.add_argument("--outdir", default=str(C.DATASET))
    a = ap.parse_args(argv)

    C.ensure_dirs()
    pdir, hdir = Path(a.processed_dir), Path(a.physics_dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tensors = np.load(pdir / "tensors.npy")
    masks = np.load(pdir / "masks.npy")
    idx = pd.read_csv(pdir / "index.csv")
    physics = np.load(hdir / "physics.npy")
    assert len(tensors) == len(idx) == len(physics), "processed/physics length mismatch"

    targets = (idx[["target_id", "mission", "label"]]
               .drop_duplicates("target_id").reset_index(drop=True))
    split_of = target_level_split(targets, C.SEED)

    rows = []
    for src, r in idx.iterrows():
        tid = str(r["target_id"])
        for w in range(max(1, C.WINDOWS_PER_TARGET)):
            rows.append({"dataset_row": len(rows), "row_index": int(src),
                         "target_id": tid, "mission": str(r["mission"]),
                         "label": int(r["label"]), "window_index": int(w),
                         "split": split_of[tid]})
    splits = pd.DataFrame(rows)

    sets = {s: set(splits.loc[splits.split == s, "target_id"])
            for s in ("train", "validation", "test")}
    overlap = {f"{x}_{y}": len(sets[x] & sets[y])
               for x, y in (("train", "validation"), ("train", "test"),
                            ("validation", "test"))}
    assert all(v == 0 for v in overlap.values()), f"target leakage detected: {overlap}"

    np.save(outdir / "tensors.npy", tensors)
    np.save(outdir / "masks.npy", masks)
    np.save(outdir / "physics.npy", physics)
    splits.to_csv(outdir / "splits.csv", index=False)

    meta = {"n_targets": int(targets["target_id"].nunique()),
            "n_windows": int(len(splits)),
            "windows_per_target": C.WINDOWS_PER_TARGET,
            "time_bins": C.TIME_BINS, "height": C.HEIGHT, "width": C.WIDTH,
            "physics_features": C.PHYSICS_NAMES}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    audit = {
        "split_level": "target",
        "n_unique_targets": int(targets["target_id"].nunique()),
        "n_windows": int(len(splits)),
        "targets_per_split": {s: int(len(v)) for s, v in sets.items()},
        "windows_per_split": splits.groupby("split").size().to_dict(),
        "target_overlap_between_splits": overlap,
        "row_overlap_between_splits": {
            f"{x}_{y}": len(set(splits.loc[splits.split == x, "row_index"])
                            & set(splits.loc[splits.split == y, "row_index"]))
            for x, y in (("train", "validation"), ("train", "test"),
                         ("validation", "test"))},
        "class_balance_per_split": {
            s: splits[splits.split == s].groupby("label").size().to_dict()
            for s in sets},
        "mission_balance_per_split": {
            s: splits[splits.split == s].groupby("mission").size().to_dict()
            for s in sets},
        "assertion": "no target appears in more than one split",
    }
    (C.RESULTS / "label_leakage_audit.json").write_text(json.dumps(audit, indent=2))

    print(json.dumps({k: audit[k] for k in
                      ("n_unique_targets", "targets_per_split",
                       "target_overlap_between_splits")}, indent=2))
    print(f"[dataset_builder] wrote {outdir / 'splits.csv'} ({len(splits)} windows)")


if __name__ == "__main__":
    main()
