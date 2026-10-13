import pathlib
raw = pathlib.Path("model/run_models.py").read_bytes().decode("utf-8", errors="replace")
i = raw.find("MODELS = [")
j = raw.find("]", i) + 1
block = raw[i:j]
block = "".join(c if ord(c) < 128 else " " for c in block)
pathlib.Path("logs/models_block.txt").write_text(block, encoding="ascii")
