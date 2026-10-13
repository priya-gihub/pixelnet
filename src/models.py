"""Baseline architectures + PIXEL-VerifyNet.

Every model shares one uniform interface so the training and evaluation loops can
treat them identically:

    forward(x, m=None, p=None) -> logits, shape (B,)

x: (B, T, H, W) flux cube      m: (B, T, H, W) validity mask      p: (B, F) physics
"""
from __future__ import annotations

import torch
from torch import nn

import config as C


# ------------------------------------------------------------- baselines ------
class CNN1D(nn.Module):
    """Baseline 1D CNN over the spatially averaged light curve."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x, m=None, p=None):
        s = x.mean(dim=(2, 3)).unsqueeze(1)          # (B, 1, T)
        return self.fc(self.net(s).flatten(1)).squeeze(1)


class CNN2D(nn.Module):
    """Baseline 2D CNN over the time-averaged pixel image."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x, m=None, p=None):
        img = x.mean(dim=1, keepdim=True)            # (B, 1, H, W)
        return self.fc(self.net(img).flatten(1)).squeeze(1)


class LSTMNet(nn.Module):
    """Baseline LSTM over the spatially averaged light curve."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, m=None, p=None):
        seq = x.mean(dim=(2, 3)).unsqueeze(-1)       # (B, T, 1)
        z, _ = self.rnn(seq)
        return self.fc(z[:, -1]).squeeze(1)


class CNN3D(nn.Module):
    """Baseline 3D CNN over the full T x H x W flux cube."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Linear(32, 1)

    def forward(self, x, m=None, p=None):
        return self.fc(self.net(x.unsqueeze(1)).flatten(1)).squeeze(1)


# --------------------------------------------------- PIXEL-VerifyNet ----------
class PixelBranch(nn.Module):
    """3D CNN over the T x H x W flux cube."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1))
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x.unsqueeze(1)).flatten(1)


class MaskBranch(nn.Module):
    """3D CNN over the observation-validity mask."""

    def __init__(self, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(8, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1))
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x.unsqueeze(1)).flatten(1)


class PhysicsBranch(nn.Module):
    """MLP over the physics / PGSEV evidence features."""

    def __init__(self, n_in: int = C.N_PHYSICS_FEATURES, hidden: int = 32,
                 out_dim: int = 16):
        super().__init__()
        self.n_in = n_in
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Dropout(0.15), nn.Linear(hidden, out_dim),
                                 nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class PIXELVerifyNet(nn.Module):
    """Pixel + Mask + Physics fusion network."""

    def __init__(self, use_pixel=True, use_mask=True, use_physics=True,
                 n_physics: int = C.N_PHYSICS_FEATURES):
        super().__init__()
        self.use_pixel = use_pixel
        self.use_mask = use_mask
        self.use_physics = use_physics
        self.n_physics = n_physics
        self.pixel = PixelBranch() if use_pixel else None
        self.mask = MaskBranch() if use_mask else None
        self.phys = PhysicsBranch(n_physics) if use_physics else None
        n = ((self.pixel.out_dim if self.pixel else 0)
             + (self.mask.out_dim if self.mask else 0)
             + (self.phys.out_dim if self.phys else 0))
        self.head = nn.Sequential(nn.Linear(n, 32), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(32, 1))

    def forward(self, x, m=None, p=None):
        z = []
        if self.pixel is not None:
            z.append(self.pixel(x))
        if self.mask is not None:
            z.append(self.mask(m if m is not None else torch.ones_like(x)))
        if self.phys is not None:
            if p is None:
                p = torch.zeros((x.shape[0], self.n_physics), device=x.device)
            z.append(self.phys(p))
        return self.head(torch.cat(z, 1)).squeeze(1)


MODELS = {
    "baseline_1dcnn": CNN1D,
    "baseline_2dcnn": CNN2D,
    "baseline_lstm": LSTMNet,
    "baseline_3dcnn": CNN3D,
    "pixel_verifynet": PIXELVerifyNet,
}

MODEL_LABELS = {
    "baseline_1dcnn": "Baseline 1D CNN",
    "baseline_2dcnn": "Baseline 2D CNN",
    "baseline_lstm": "Baseline LSTM",
    "baseline_3dcnn": "Baseline 3D CNN",
    "pixel_verifynet": "PIXEL-VerifyNet",
}


def build(name: str, **kw):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kw)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
