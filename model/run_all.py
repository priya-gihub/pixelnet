"""PIXEL-VerifyNet model showcase launcher (Option B).

Runs the untouched terminal presentation (model/run_models.py) to
completion, then opens model/results.html in the default browser.

Usage:
    python run_all.py            # paced presentation (~35 s) + page at end
    python run_all.py --fast     # no pauses + page at end
    python run_all.py --delay 10 # slower + page at end
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
import webbrowser
from pathlib import Path


def main() -> int:
    sys.argv = ["run_models.py", *sys.argv[1:]]
    try:
        runpy.run_path(str(Path(__file__).with_name("run_models.py")), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
    else:
        code = 0
    tried = []
    try:
        page = Path(__file__).with_name("results.html").resolve()
        if not page.exists():
            print("Results page not found: " + str(page))
            return code
        print("Opening results page: " + str(page))
        launched = False
        # 1) Native association (uses the user's default browser).
        if os.name == "nt":
            try:
                os.startfile(str(page))  # noqa: S606
                print("Opened via default .html association (os.startfile).")
                launched = True
            except Exception as exc:
                tried.append("startfile: " + str(exc))
        # 2) Explicit installed browsers (works with no .html association).
        if not launched:
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            ]
            for exe in candidates:
                if Path(exe).exists():
                    try:
                        subprocess.Popen([exe, str(page)])
                        print("Opened via " + exe)
                        launched = True
                        break
                    except Exception as exc:
                        tried.append(exe + ": " + str(exc))
        # 3) webbrowser module fallback.
        if not launched:
            try:
                ok = webbrowser.open(page.as_uri())
                print("webbrowser.open returned " + str(ok))
                launched = bool(ok)
            except Exception as exc:
                tried.append("webbrowser: " + str(exc))
        if not launched:
            print("Could not auto-open the page. Please open it manually:")
            print("  " + str(page))
            if tried:
                print("Diagnostics: " + " | ".join(tried))
    except Exception as exc:
        print("Could not auto-open the page (" + str(exc) + "). Please open it manually:")
        print("  " + str(Path(__file__).with_name("results.html").resolve()))

    return code


if __name__ == "__main__":
    raise SystemExit(main())
