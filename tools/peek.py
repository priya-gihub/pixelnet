"""Print a line range of a file: python tools/peek.py <file> <start> <end>"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo + 60
lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
print(f"FILE={p} TOTAL={len(lines)}")
for i in range(max(1, lo), min(hi, len(lines)) + 1):
    print(f"{i:4d}: {lines[i - 1]}")