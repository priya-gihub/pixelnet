"""Background finalizer: waits for the 1000-target dataset, then runs the FULL
pipeline end-to-end and verifies it.

Waits until both catalogs hold 250 planet + 250 false-positive targets
(i.e. the download + refill jobs have delivered the full design), then:
    prune_and_refill -> checkpoint_pipe -> train x5 -> evaluate ->
    make_report -> explainability -> uncertainty -> ablation ->
    inference -> make_report -> verify_final -> pytest

Launch DETACHED only (long job):
    start /b cmd /c python -u tools\\finalize_all.py >> logs\\finalize_all.log 2>&1

Single-flight guard: logs/finalize_all.lock. Writes logs/finalize_all.done on
success. Safe to re-run: training steps whose checkpoint already exists are
skipped, so an interrupted run can simply be restarted.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "logs" / "finalize_all.lock"
DONE = ROOT / "logs" / "finalize_all.done"
POLL_S = 120
TIMEOUT_H = 12


def run(*args):
    print(f"[finalize] $ python {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-u", *args], cwd=ROOT)
    print(f"[finalize] exit={r.returncode}: {' '.join(args)}", flush=True)
    return r.returncode


def dataset_ready() -> bool:
    """True when both missions hold the full 250 + 250 design.

    Counts the main catalog AND every stage catalog: while the parallel
    fetchers are still running they write data/raw/<mission>_stage_*.csv, and
    those rows are merged into the main catalog later by prune_and_refill.
    Checking only the main catalog would deadlock: the merge can never run
    because the gate that triggers the merge would never open.
    """
    try:
        import pandas as pd
        for m in ("kepler", "tess"):
            ids = {0: set(), 1: set()}
            files = sorted((ROOT / "data" / "raw").glob(f"{m}_catalog.csv")) \
                + sorted((ROOT / "data" / "raw").glob(f"{m}_stage*.csv"))
            for f in files:
                if "failure" in f.name or not f.stat().st_size:
                    continue
                try:
                    d = pd.read_csv(f, dtype={"target_id": str}).drop_duplicates(
                        "target_id")
                except Exception:
                    continue
                for _, r in d.iterrows():
                    try:
                        ids[int(r["label"])].add(str(r["target_id"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            if len(ids[1]) < 250 or len(ids[0]) < 250:
                return False
        return True
    except Exception:
        return False


def counts() -> str:
    import pandas as pd
    out = []
    for m in ("kepler", "tess"):
        f = ROOT / "data" / "raw" / f"{m}_catalog.csv"
        if f.exists():
            d = pd.read_csv(f).drop_duplicates("target_id")
            out.append(f"{m}={len(d)}(P{int((d.label == 1).sum())}"
                       f"/F{int((d.label == 0).sum())})")
        else:
            out.append(f"{m}=missing")
    return " ".join(out)


def drop_stale_checkpoints():
    """Remove any checkpoint older than the current dataset build.

    A checkpoint trained on an earlier dataset (e.g. the 450-target
    checkpoint run) must never be reused for the 1000-target evaluation.

    The reference is the whole feature pipeline, not just splits.csv: changing
    src/physics_features.py invalidates every checkpoint even when the split
    file is untouched, because the physics branch is a model input.
    """
    stamps = [ROOT / "data" / "dataset" / "splits.csv",
              ROOT / "data" / "physics" / "physics.npy",
              ROOT / "src" / "physics_features.py",
              ROOT / "src" / "preprocessing.py"]
    stamps = [f for f in stamps if f.exists()]
    if not stamps:
        return
    stamp = max(f.stat().st_mtime for f in stamps)
    for ck in (ROOT / "checkpoints").glob("*.pt"):
        if ck.stat().st_mtime < stamp:
            print(f"[finalize] dropping stale checkpoint {ck.name}", flush=True)
            ck.unlink()
    for stale in ("final_metrics.csv", "final_metrics.json",
                  "predictions_baseline_1dcnn.csv",
                  "predictions_baseline_2dcnn.csv",
                  "predictions_baseline_lstm.csv",
                  "predictions_baseline_3dcnn.csv",
                  "predictions_pixel_verifynet.csv"):
        f = ROOT / "results" / stale
        if f.exists() and f.stat().st_mtime < stamp:
            print(f"[finalize] dropping stale {stale}", flush=True)
            f.unlink()


def main():
    if LOCK.exists():
        print("[finalize] lock exists, another finalizer is running. exiting.",
              flush=True)
        return 0
    LOCK.write_text("running")
    try:
        t0 = time.time()
        while not dataset_ready():
            if (time.time() - t0) > TIMEOUT_H * 3600:
                print("[finalize] TIMEOUT waiting for the dataset", flush=True)
                return 1
            print(f"[finalize] waiting for 1000 targets ... {counts()}", flush=True)
            time.sleep(POLL_S)
        print(f"[finalize] dataset ready: {counts()}. starting full pipeline.",
              flush=True)
        drop_stale_checkpoints()

        steps = [
            (["tools/enrich_ephemeris.py"], False),
            (["tools/prune_and_refill.py"], False),
            (["tools/checkpoint_pipe.py"], False),
            (["-m", "src.train", "--dataset-dir", "data/dataset",
              "--model", "baseline_1dcnn"], True),
            (["-m", "src.train", "--dataset-dir", "data/dataset",
              "--model", "baseline_2dcnn"], True),
            (["-m", "src.train", "--dataset-dir", "data/dataset",
              "--model", "baseline_lstm"], True),
            (["-m", "src.train", "--dataset-dir", "data/dataset",
              "--model", "baseline_3dcnn"], True),
            (["-m", "src.train", "--dataset-dir", "data/dataset",
              "--model", "pixel_verifynet"], True),
            (["-m", "src.evaluate", "--dataset-dir", "data/dataset"], False),
            (["tools/check_feature_independence.py"], False),
            (["tools/check_inference_consistency.py", "5"], False),
            (["tools/make_report.py"], False),
            (["-m", "src.explainability"], True),
            (["-m", "src.uncertainty", "--dataset-dir", "data/dataset"], True),
            (["-m", "src.ablation", "--dataset-dir", "data/dataset"], True),
            (["-m", "src.inference_pipeline"], True),
            (["tools/make_report.py"], False),
            (["tools/verify_final.py"], False),
            (["-m", "pytest", "tests", "-q"], False),
        ]
        for args, skippable in steps:
            if skippable and args[0] == "-m" and args[1] == "src.train":
                ckpt = ROOT / "checkpoints" / f"{args[-1]}.pt"
                if ckpt.exists():
                    print(f"[finalize] skip {args[-1]} (checkpoint exists)",
                          flush=True)
                    continue
            rc = run(*args)
            if rc != 0 and not skippable:
                print(f"[finalize] FAILED on {' '.join(args)}", flush=True)
                return rc
        print("[finalize] ALL DONE", flush=True)
        (ROOT / "logs" / "finalize_all.done").write_text("ok")
        return 0
    finally:
        if (ROOT / "logs" / "finalize_all.done").exists():
            LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
