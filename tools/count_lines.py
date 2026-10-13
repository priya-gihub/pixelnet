import pathlib
n = sum(1 for _ in pathlib.Path("model/results.html").open(encoding="utf-8"))
pathlib.Path("logs/html_lines.txt").write_text(str(n), encoding="ascii")
