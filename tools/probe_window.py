"""Probe the window-coordinate mapping of cut_window / star detection.

Usage: python tools/probe_window.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import config as C
from src.preprocessing import cut_window, target_pixel

h = w = 7
flux = np.full((3, h, w), 1.0e5, np.float32)
flux[:, 3, 3] *= 0.5                       # brightest / dimmest marker
pmask = np.zeros((h, w), bool)
pmask[2:5, 2:5] = True

cy, cx = target_pixel(flux, pmask)
print(f"aperture centroid (native) = ({cy}, {cx})")
patch = cut_window(flux, cy, cx, C.HEIGHT, C.WIDTH)
print(f"patch shape = {patch.shape}")
med = np.nanmedian(patch, axis=0)
flat = int(np.nanargmax(med))
print(f"brightest WINDOW pixel = ({flat // patch.shape[2]}, "
      f"{flat % patch.shape[2]})   <-- expected (5, 5)")
print(f"native (3,3) value in window at (5,5) = {patch[0, 5, 5]:.1f}")
print(f"value at (3,3) = {patch[0, 3, 3]:.1f}")
print("column medians:", np.array2string(np.nanmedian(patch, axis=(0, 1)),
                                         precision=0, max_line_width=200))
print("row medians   :", np.array2string(np.nanmedian(patch, axis=(0, 2)),
                                         precision=0, max_line_width=200))