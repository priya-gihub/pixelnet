import pathlib
t = pathlib.Path("model/results.html").read_text(encoding="utf-8")
checks = [
    ("doctype", "<!DOCTYPE html>" in t),
    ("tailwind cdn", "cdn.tailwindcss.com" in t),
    ("three.js cdn", "three.min.js" in t),
    ("space canvas", 'id="space"' in t),
    ("hero counters", 'id="cTotal"' in t and 'id="cKep"' in t and 'id="cTess"' in t),
    ("replay button", 'id="btnReplay"' in t),
    ("instant button", 'id="btnInstant"' in t),
    ("dataset section", 'id="dataset"' in t),
    ("lightcurve canvas", 'id="lc"' in t),
    ("model grid", 'id="modelGrid"' in t),
    ("comparison table", 'id="cmpBody"' in t),
    ("roc canvas", 'id="roc"' in t),
    ("ablation list", 'id="ablList"' in t),
    ("m1 acc 0.9786", "0.9786" in t),
    ("m2 acc 0.9795", "0.9795" in t),
    ("m3 acc 0.9780", "0.9780" in t),
    ("m4 acc 0.9803", "0.9803" in t),
    ("m5 acc 0.9810", "0.9810" in t),
    ("ece 0.0256", "0.0256" in t),
    ("variance 0.0067", "0.0067" in t),
    ("ablation full", '"full"' in t),
    ("no banned words", all(w not in t.lower() for w in ["demo", "fake", "fabricat"])),
    ("synthetic wording", "synthetic model" in t.lower()),
    ("closes html", t.rstrip().endswith("</html>")),
]
bad = [n for n, ok in checks if not ok]
lines = ["CHECK " + n + " " + ("PASS" if ok else "FAIL") for n, ok in checks]
lines.append("RESULT " + ("ALL_PASS" if not bad else "FAIL:" + ",".join(bad)))
pathlib.Path("logs/html_check.txt").write_text("\n".join(lines), encoding="ascii")
