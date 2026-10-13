"""Verify the box-drawing glyphs survive as real UTF-8 in the emitted bytes.

Reading the captured output through a legacy code page mangles box-drawing
characters into '?', so this check decodes the raw bytes as UTF-8 and asserts
the characters are intact - i.e. what the terminal receives is correct.
"""
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent / "run_models.py"


def main() -> int:
    r = subprocess.run([sys.executable, str(DEMO), "--fast"],
                       capture_output=True, cwd=DEMO.parent)
    raw = r.stdout
    try:
        text = raw.decode("utf-8")
        ok = True
    except UnicodeDecodeError as e:
        print(f"[FAIL] stdout is not valid UTF-8: {e}")
        return 1

    for name, ch in (("double horizontal", "\u2550"), ("vertical", "\u2551"),
                     ("light horizontal", "\u2500"), ("full block", "\u2588")):
        present = ch in text
        # print the codepoint, not the glyph: this check may itself run under a
        # legacy code page that cannot encode the character
        print(f"[{'PASS' if present else 'FAIL'}] glyph present: "
              f"{name} (U+{ord(ch):04X})")
        ok = ok and present

    # no replacement / question-mark runs where a glyph should be
    bad = "\ufffd" in text
    print(f"[{'FAIL' if bad else 'PASS'}] no replacement characters in output")
    ok = ok and not bad

    print()
    print("utf-8 glyph check passed" if ok else "utf-8 glyph check FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())