import pathlib, re
t = pathlib.Path("model/results.html").read_text(encoding="utf-8")
blocks = re.findall(r"<script>(.*?)</script>", t, re.S)
pathlib.Path("logs/js_blocks.txt").write_text(str(len(blocks)), encoding="ascii")
for i, b in enumerate(blocks):
    if "tailwind.config" in b:
        continue
    pathlib.Path("logs/js_%d.js" % i).write_text(b, encoding="utf-8")
