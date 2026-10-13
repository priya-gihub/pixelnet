"""Print installed versions of the libraries the project depends on."""
import importlib

for m in ["numpy", "pandas", "sklearn", "scipy", "torch", "matplotlib",
          "seaborn", "plotly", "shap", "lightkurve", "astropy", "requests",
          "streamlit", "pytest"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({type(e).__name__})")