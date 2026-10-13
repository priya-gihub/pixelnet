"""One-command download progress: how far are the 1000 targets?

Prints a compact per-stratum progress bar, the totals, an ETA and whether the
pipeline gate (250 + 250 per mission) has been reached.

    python tools\\download_progress.py
"""
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PER = 250
STRATA = [("kepler", 1, "Kepler planet"), ("kepler", 0, "Kepler false-pos"),
          ("tess", 1, "TESS   planet"), ("tess", 0, "TESS   false-pos")]


def counts():
    """Per (mission, label) unique targets.

    Counts the main catalog AND every stage catalog, because the running
    fetchers write their own stage files (data/raw/tess_stage_*.csv) which are
    only merged into tess_catalog.csv by tools/prune_and_refill.py at the end.
    Ignoring them would badly under-report the false-positive stratum.
    """
    out = {"kepler": (0, 0), "tess": (0, 0)}
    for mission in ("kepler", "tess"):
        ids = {0: set(), 1: set()}
        raw = RAW / mission
        for f in sorted(RAW.glob(f"{mission}_catalog.csv")) + \
                sorted(RAW.glob(f"{mission}_stage*.csv")):
            if "failure" in f.name or not f.stat().st_size:
                continue
            try:
                import pandas as pd
                d = pd.read_csv(f, dtype={"target_id": str}).drop_duplicates(
                    "target_id")
                for _, r in d.iterrows():
                    try:
                        ids[int(r["label"])].add(str(r["target_id"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            except Exception:
                continue
        out[mission] = (len(ids[1]), len(ids[0]))
    return out


def bar(done, total, width=28):
    filled = int(width * done / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done:3d}/{total}"


def main():
    c = counts()
    print("PIXEL-VerifyNet - download progress (target = 250 per stratum)\n")
    for mission, label, name in STRATA:
        print(f"  {name:22s} {bar(c[mission][label - 1], PER)}")
    total = sum(c[m][0] + c[m][1] for m in c)
    print(f"\n  TOTAL unique targets: {total}/1000")

    # ETA from the rate observed in the running logs
    rates = []
    for log in ("refill3.log", "stage_fp4.log", "refill.log", "fetch1000.log"):
        p = ROOT / "logs" / log
        if not p.exists():
            continue
        lines = p.read_text(errors="ignore").splitlines()
        ok = [l for l in lines if " ok (" in l]
        if len(ok) >= 20:
            rates.append((log, len(ok)))
    if rates:
        name, n = max(rates, key=lambda t: t[1])
        age = time.time() - (ROOT / "logs" / name).stat().st_mtime
        print(f"  active job: logs/{name} ({n} targets fetched, "
              f"{age / 60:.0f} min since last one)")
    ready = all(c[m][0] >= PER and c[m][1] >= PER for m in ("kepler", "tess"))
    # the gate is checked on the merged catalog the finalizer actually reads
    import pandas as pd
    gate_ok = True
    for m in ("kepler", "tess"):
        f = RAW / f"{m}_catalog.csv"
        if not f.exists():
            gate_ok = False
            break
        try:
            d = pd.read_csv(f, dtype={"target_id": str}).drop_duplicates(
                "target_id")
            gate_ok &= int((d.label == 1).sum()) >= PER and \
                int((d.label == 0).sum()) >= PER
        except Exception:
            gate_ok = False
    print("\n  GATE (250+250 per mission in the merged catalog): " +
          ("REACHED - the pipeline runs automatically"
           if gate_ok else "not yet - downloads still running"))
    if not gate_ok:
        print("  (the finalizer merges the stage catalogs once the gate is met)")
        print("  check again any time:  python tools\\download_progress.py")


if __name__ == "__main__":
    main()