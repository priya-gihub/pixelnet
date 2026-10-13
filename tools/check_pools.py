"""Check label pools: Kepler/TESS sizes + ambiguous-target counts. Run: python tools/check_pools.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_download import tap
import config as C

print("== KEPLER ==", flush=True)
kd = C.KEPLER_PLANET_DISP + C.KEPLER_FP_DISP
disp = "','".join(kd)
df = tap(f"select kepid,koi_disposition from cumulative where koi_disposition in ('{disp}')")
print(f"koi rows={len(df)}", flush=True)
df["is_planet"] = df["koi_disposition"].isin(C.KEPLER_PLANET_DISP).astype(int)
g = df.groupby("kepid")["is_planet"]
n_planet = int((g.max() == 1).sum())
n_fp = int((g.min() == 0).sum())
amb = int(((g.max() == 1) & (g.min() == 0)).sum())
pure_planet = int(((g.max() == 1) & (g.min() == 1)).sum())
pure_fp = int(((g.max() == 0) & (g.min() == 0)).sum())
print(f"unique kepid={g.ngroups} pure_planet={pure_planet} pure_fp={pure_fp} ambiguous={amb}", flush=True)

print("== TESS ==", flush=True)
td = C.TESS_PLANET_DISP + C.TESS_FP_DISP
disp2 = "','".join(td)
df2 = tap(f"select tid,tfopwg_disp from toi where tfopwg_disp in ('{disp2}')")
print(f"toi rows={len(df2)}", flush=True)
df2["is_planet"] = df2["tfopwg_disp"].isin(C.TESS_PLANET_DISP).astype(int)
g2 = df2.groupby("tid")["is_planet"]
amb2 = int(((g2.max() == 1) & (g2.min() == 0)).sum())
pure_p2 = int(((g2.max() == 1) & (g2.min() == 1)).sum())
pure_f2 = int(((g2.max() == 0) & (g2.min() == 0)).sum())
print(f"unique tid={g2.ngroups} pure_planet={pure_p2} pure_fp={pure_f2} ambiguous={amb2}", flush=True)
print("OK", flush=True)
