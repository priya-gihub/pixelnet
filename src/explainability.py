"""Explainability - Grad-CAM-style pixel saliency + SHAP physics attribution.

Usage:
    python -m src.explainability --checkpoint checkpoints/pixel_verifynet.pt --sample-idx 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as C
from src.models import build
from src.train import apply_physics_stats, device


def saliency(model, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> dict:
    """Gradient of the logit w.r.t. the input flux cube."""
    x = x.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logit = model(x, m, p).sum()
    logit.backward()
    g = x.grad.detach().abs()[0]                     # (T, H, W)
    return {"temporal": g.mean(dim=(1, 2)).cpu().numpy(),
            "spatial": g.mean(dim=0).cpu().numpy(),
            "cube": g.cpu().numpy()}


def physics_shap(model, te, tensors, masks, physics_scaled, dev, n_bg: int = 32):
    """Permutation SHAP over the ten physics features (best-effort)."""
    try:
        import shap
    except Exception as e:                            # pragma: no cover
        return {"available": False, "reason": repr(e)}

    i = int(te.iloc[0]["row_index"])
    x = torch.tensor(np.asarray(tensors[i], np.float32), device=dev)[None]
    m = torch.tensor(np.asarray(masks[i], np.float32), device=dev)[None]

    def f(Parr: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.asarray(Parr, np.float32), device=dev)
            out = torch.sigmoid(model(x.repeat(len(Parr), 1, 1, 1),
                                      m.repeat(len(Parr), 1, 1, 1), t))
        return out.cpu().numpy()

    bg = physics_scaled[:min(n_bg, len(physics_scaled))]
    expl = shap.PermutationExplainer(f, bg)
    sv = expl(physics_scaled[:min(n_bg, len(physics_scaled))])
    return {"available": True, "values": np.asarray(sv.values),
            "base_values": np.asarray(sv.base_values),
            "feature_names": C.PHYSICS_NAMES}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grad-CAM saliency + SHAP physics.")
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINTS / "pixel_verifynet.pt"))
    ap.add_argument("--dataset-dir", default=str(C.DATASET))
    ap.add_argument("--sample-idx", type=int, default=0)
    a = ap.parse_args(argv)

    C.ensure_dirs()
    dev = device()
    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    model = build(ck.get("model", "pixel_verifynet")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    d = Path(a.dataset_dir)
    splits = pd.read_csv(d / "splits.csv")
    te = splits[splits.split == "test"].reset_index(drop=True)
    tensors = np.load(d / "tensors.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    physics = np.load(d / "physics.npy")
    scaled = apply_physics_stats(physics[te.row_index.values], ck["physics_stats"])

    row = te.iloc[a.sample_idx]
    i = int(row["row_index"])
    x = torch.tensor(np.asarray(tensors[i], np.float32), device=dev)[None]
    m = torch.tensor(np.asarray(masks[i], np.float32), device=dev)[None]
    p = torch.tensor(np.asarray(scaled[a.sample_idx], np.float32),
                     dtype=torch.float32, device=dev)[None]

    sal = saliency(model, x, m, p)
    np.savez(C.RESULTS / "saliency.npz", **sal)

    meta = {"target_id": str(row["target_id"]), "mission": str(row["mission"]),
            "label": int(row["label"]), "sample_idx": int(a.sample_idx),
            "peak_time_bin": int(np.argmax(sal["temporal"])),
            "peak_pixel": [int(v) for v in np.unravel_index(np.argmax(sal["spatial"]),
                                                            sal["spatial"].shape)]}

    shap_out = physics_shap(model, te, tensors, masks, scaled, dev)
    if shap_out.get("available"):
        np.savez(C.RESULTS / "physics_shap.npz", values=shap_out["values"],
                 base_values=shap_out["base_values"])
        meta["shap_mean_abs"] = {n: float(np.abs(shap_out["values"][:, j]).mean())
                                 for j, n in enumerate(C.PHYSICS_NAMES)}
    else:
        meta["shap_error"] = shap_out.get("reason")

    (C.RESULTS / "explainability.json").write_text(json.dumps(meta, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(sal["temporal"]); ax[0].set_title("temporal saliency")
        ax[0].set_xlabel("time bin")
        im = ax[1].imshow(sal["spatial"]); ax[1].set_title("spatial saliency")
        fig.colorbar(im, ax=ax[1], fraction=0.046)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "saliency.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        meta["figure_error"] = repr(e)
        (C.RESULTS / "explainability.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
