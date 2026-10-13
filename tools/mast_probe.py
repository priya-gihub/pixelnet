"""Probe the MAST search API for one target id (diagnostic helper).

Usage: python tools/mast_probe.py "TIC 137906197"
"""
import sys

import lightkurve as lk

tid = sys.argv[1] if len(sys.argv) > 1 else "TIC 137906197"
mission = "TESS" if tid.upper().startswith("TIC") else "Kepler"

trials = [
    ("no cadence", dict(mission=mission)),
    ("cadence=long", dict(mission=mission, cadence="long")),
    ("cadence=10min", dict(mission=mission, cadence="10min")),
    ("cadence=2min", dict(mission=mission, cadence="2min")),
]
for name, kw in trials:
    try:
        r = lk.search_targetpixelfile(tid, **kw)
        print(f"{name:16s} -> {len(r)} products")
        if len(r):
            t = r.table.to_pandas()
            cols = [c for c in ("exptime", "cadence", "productFilename", "tess_sector")
                    if c in t.columns]
            print(t[cols].head(5).to_string())
            break
    except Exception as e:
        print(f"{name:16s} -> {type(e).__name__}: {e}")