"""PIXEL-VerifyNet source package.

Puts the project root on ``sys.path`` so every module can simply ``import config``
regardless of whether it is run as ``python -m src.<module>`` or imported from tests.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

