"""Streamlit dashboard for PIXEL-VerifyNet.

Run from the project root:
    python -m streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402

st.set_page_config(page_title="PIXEL-VerifyNet", layout="wide")
st.title("PIXEL-VerifyNet")
st.caption("Physics-Guided Scientific Evidence Verification for Kepler & TESS "
           "target pixel sequences")

idx_path = C.PROCESSED / "index.csv"
if not idx_path.exists():
    st.warning("No processed data yet. Run, in order: src.data_download -> "
               "src.data_matching -> src.validation -> src.preprocessing -> src.pgsev "
               "-> src.dataset_builder.")
    st.stop()

idx = pd.read_csv(idx_path)
tensors = np.load(C.PROCESSED / "tensors.npy", mmap_mode="r")
masks = np.load(C.PROCESSED / "masks.npy", mmap_mode="r")
physics_path = C.PHYSICS / "physics.npy"
physics = np.load(physics_path) if physics_path.exists() else None

sel = st.sidebar.selectbox("Target", range(len(idx)),
                           format_func=lambda i: str(idx.iloc[i]["target_id"]))
r = idx.iloc[sel]
i = int(r["row_index"])

a, b, c, d = st.columns(4)
a.metric("Mission", str(r["mission"]))
b.metric("Label", "PLANET" if int(r["label"]) else "FALSE POSITIVE")
c.metric("Cadences (native)", int(r["cadences"]))
d.metric("Pixel coverage", f"{int(r['pixel_coverage'])} px")

x = np.asarray(tensors[i])
m = np.asarray(masks[i])
lc = (x * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1)

fig = go.Figure(go.Scatter(y=lc, mode="lines+markers", name="aperture"))
fig.update_layout(title="Normalised aperture light curve",
                  xaxis_title="time bin", yaxis_title="relative flux")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Pixel frame")
k = st.slider("Time bin", 0, x.shape[0] - 1, int(np.argmin(lc)))
left, right = st.columns(2)
f1 = go.Figure(go.Heatmap(z=x[k], colorscale="RdBu", zmid=0))
f1.update_layout(title="flux cube")
left.plotly_chart(f1, use_container_width=True)
f2 = go.Figure(go.Heatmap(z=m[k], colorscale="Greys", zmin=0, zmax=1))
f2.update_layout(title="validity mask")
right.plotly_chart(f2, use_container_width=True)

if physics is not None:
    st.subheader("Physics evidence (PGSEV inputs)")
    st.dataframe(pd.DataFrame({"feature": C.PHYSICS_NAMES,
                               "value": np.asarray(physics[i], dtype=float).round(4)}),
                 use_container_width=True)
    pg = C.PHYSICS / "pgsev.npy"
    if pg.exists():
        st.metric("PGSEV score", f"{float(np.load(pg)[i]):.3f}")

metrics_path = C.RESULTS / "final_metrics.csv"
if metrics_path.exists():
    st.subheader("Model comparison (held-out test targets)")
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True)

sal = C.FIGURES / "saliency.png"
if sal.exists():
    st.subheader("Pixel saliency (Grad-CAM-style)")
    st.image(str(sal), use_container_width=True)
