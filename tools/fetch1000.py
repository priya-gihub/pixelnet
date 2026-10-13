"""Background job: fetch the full 1000-target set (500 Kepler + 500 TESS), resumable."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_download as dd


def run(mission, n):
    sys.argv = ["data_download", "--mission", mission, "--n-targets", str(n),
                "--outdir", str(ROOT / "data" / "raw")]
    dd.main()


if __name__ == "__main__":
    print("[fetch1000] kepler 500 ...", flush=True)
    run("kepler", 500)
    print("[fetch1000] tess 500 ...", flush=True)
    run("tess", 500)
    print("[fetch1000] DONE", flush=True)
