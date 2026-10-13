"""Estimate CPU training time per epoch for every model (no checkpoint writes).

Uses synthetic tensors with the real dataset shape so the estimate matches the
final run. Usage: python tools/bench_train.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import config as C
from src.models import MODELS, count_parameters

N_TRAIN = int(C.N_TARGETS_TOTAL * C.TRAIN_FRACTION)   # 700
BATCH = C.BATCH_SIZE


def main():
    torch.manual_seed(0)
    x_all = torch.randn(N_TRAIN, C.TIME_BINS, C.HEIGHT, C.WIDTH)
    m_all = torch.ones_like(x_all)
    p_all = torch.randn(N_TRAIN, C.N_PHYSICS_FEATURES)
    y_all = (torch.rand(N_TRAIN) > 0.5).float()
    print(f"train rows={N_TRAIN} batch={BATCH} threads={torch.get_num_threads()}")
    for name in MODELS:
        model = MODELS[name]()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lossf = torch.nn.BCEWithLogitsLoss()
        model.train()
        t0 = time.time()
        for i in range(0, N_TRAIN, BATCH):
            xb, mb = x_all[i:i + BATCH], m_all[i:i + BATCH]
            pb, yb = p_all[i:i + BATCH], y_all[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(xb, mb, pb), yb)
            loss.backward()
            opt.step()
        dt = time.time() - t0
        print(f"{name:18s} params={count_parameters(model):7d} "
              f"train_epoch={dt:6.1f}s  x{C.EPOCHS}ep = {dt * C.EPOCHS / 60:5.1f} min")


if __name__ == "__main__":
    main()