"""Disk footprint of the raw pixel cache (per mission and per target).

Usage: python tools/disk_report.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

total = 0.0
for mission in ("kepler", "tess"):
    d = RAW / mission
    files = list(d.glob("*.npz")) if d.exists() else []
    size = sum(f.stat().st_size for f in files) / 1e6
    total += size
    per = size / max(len(files), 1)
    print(f"{mission:8s} {len(files):5d} products  {size:9.1f} MB  "
          f"({per:6.2f} MB per target)")

for name in ("matched", "processed", "physics", "dataset"):
    d = ROOT / "data" / name
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6 \
        if d.exists() else 0.0
    total += size
    print(f"{name:8s} {'-':>5s}            {size:9.1f} MB")

print(f"\ntotal pipeline footprint: {total:.1f} MB "
      f"({total / 1000:.2f} GB)")