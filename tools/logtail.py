"""Read the tail of a growing log without `type` (which trips the 30s limit)."""

import sys
from pathlib import Path
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/fetch1000.log")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lines = p.read_text(errors="ignore").splitlines()
print(f"FILE={p} BYTES={p.stat().st_size} LINES={len(lines)}")
print("\n".join(lines[-n:]))
