import re, pathlib
raw = pathlib.Path("logs/demo_fast.log").read_bytes().decode("utf-8", errors="replace")
t = re.sub(r"\x1b\[[0-9;]*m", "", raw)
ascii_t = "".join(c if ord(c) < 128 else " " for c in t)
out = []
pat = (r"\[(1/5|2/5|3/5|4/5|5/5)\][^\n]*\n.*?parameters\s+([\d,]+)"
       r".*?Accuracy\s+([\d.]+).*?Precision\s+([\d.]+).*?Recall\s+([\d.]+)"
       r".*?F1\s+([\d.]+).*?ROC-AUC\s+([\d.]+).*?PR-AUC\s+([\d.]+)"
       r".*?FPR\s+([\d.]+).*?confusion\s+TN\s+(\d+)\s+FP\s+(\d+)\s+FN\s+(\d+)\s+TP\s+(\d+)")
for m in re.findall(pat, ascii_t, re.S):
    out.append("MODEL " + " | ".join(m))
out.append("---TABLE---")
for line in ascii_t.splitlines():
    s = line.strip()
    if s.startswith("PIXEL") or s.startswith("Baseline"):
        out.append("ROW " + s)
out.append("---ABLATION---")
for line in ascii_t.splitlines():
    s = line.strip()
    if s.startswith("physics_only") or s.startswith("pixel_only") or s.startswith("pixel_mask") or s.startswith("pixel_physics") or s.startswith("full "):
        out.append("ABL " + s)
out.append("---CALIB---")
for line in ascii_t.splitlines():
    s = line.strip()
    if "calibration error" in s or "predictive variance" in s or "MC samples" in s:
        out.append("CAL " + s)
pathlib.Path("logs/nums.txt").write_text("\n".join(out), encoding="ascii")
