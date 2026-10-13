"""Pipeline checkpoint: matching -> validation -> preprocessing -> pgsev -> dataset.
Safe to run WHILE fetch1000 is downloading (reads only files+catalog rows whose
.npz exists; the final rerun after 1000 targets overwrites these outputs).
Usage: python tools/checkpoint_pipe.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as C

def main():
    C.ensure_dirs()
    from src import data_matching as dm, validation as va
    from src import preprocessing as pp
    from src import pgsev as pg, dataset_builder as db

    m, report = dm.match(C.RAW)
    (C.MATCHED / "matched_all.csv").write_text("")  # placeholder guard
    m.to_csv(C.MATCHED / "matched_all.csv", index=False)
    import json
    (C.RESULTS / "matching_report.json").write_text(json.dumps(report, indent=2))
    print(f"[ckpt] matched {len(m)} targets: "
          f"{m['mission'].value_counts().to_dict()} "
          f"{m['label'].value_counts().to_dict()}", flush=True)

    va.main()
    pp.main()
    pg.main()
    try:
        db.main([])
    except SystemExit as e:
        print(f"[ckpt] dataset_builder: {e}", flush=True)
    print("[ckpt] DONE", flush=True)

if __name__ == "__main__":
    main()
