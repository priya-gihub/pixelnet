import pathlib, re
t = pathlib.Path("model/results.html").read_text(encoding="utf-8")
out = []
ids_html = set(re.findall(r'id="([A-Za-z]+)"', t))
need = ["space","cTotal","cKep","cTess","btnReplay","btnInstant","runStatus","lc","lcR","lcRv","lcDepth","lcCad","lcDur","lcKepler","lcTess","modelGrid","cmpBody","cmpTable","roc","ablList"]
for i in need:
    out.append(("ID", i, i in ids_html))
funcs = ["renderCards","renderTable","renderAbl","drawLC","drawROC","replay","instant","countUp","fillBars","barRow","lockAll","unlock","setStatus"]
for f in funcs:
    out.append(("FUNC", f, ("function " + f) in t))
calls = ["renderCards()", "renderTable()", "renderAbl()", "drawLC()", "drawROC()", "lockAll()"]
for c in calls:
    out.append(("CALL", c, c in t))
for v in ["var MODELS", "var ABL", "var sortK", "var activeB"]:
    out.append(("VAR", v, t.count(v) == 1))
listeners = t.count("addEventListener")
out.append(("LISTENERS", str(listeners), listeners >= 8))
lines = ["%s %s %s" % (k, n, "PASS" if ok else "FAIL") for k, n, ok in out]
bad = [n for k, n, ok in out if not ok]
lines.append("RESULT " + ("ALL_PASS" if not bad else "FAIL:" + ",".join(bad)))
pathlib.Path("logs/html_wire.txt").write_text("\n".join(lines), encoding="ascii")
