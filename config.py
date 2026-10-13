"""PIXEL-VerifyNet - central configuration (paths, dataset protocol, hyperparameters)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- directories
DATA = ROOT / "data"
RAW = DATA / "raw"
MATCHED = DATA / "matched"
PROCESSED = DATA / "processed"
PHYSICS = DATA / "physics"
DATASET = DATA / "dataset"
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
CHECKPOINTS = ROOT / "checkpoints"
LOGS = ROOT / "logs"

DIRS = [DATA, RAW, MATCHED, PROCESSED, PHYSICS, DATASET,
        RESULTS, TABLES, FIGURES, CHECKPOINTS, LOGS]

EXO_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# ------------------------------------------------------------ reproducibility
SEED = 42

# ----------------------------------------------------------- dataset protocol
# Exactly N_TARGETS_TOTAL unique Kepler + TESS targets, balanced across
# (mission, label) strata: 250 Kepler planets, 250 Kepler FPs,
# 250 TESS planets, 250 TESS FPs.
N_TARGETS_TOTAL = 1000
MISSIONS = ("kepler", "tess")
TARGETS_PER_STRATUM = N_TARGETS_TOTAL // 4      # 250

# real archive dispositions -> label
KEPLER_PLANET_DISP = ("CONFIRMED",)
KEPLER_FP_DISP = ("FALSE POSITIVE",)
TESS_PLANET_DISP = ("CP", "KP")
TESS_FP_DISP = ("FP", "FA")

# ------------------------------------------------------------------ tensors
TIME_BINS = 256         # T - gap-aware temporal bins
HEIGHT = 11             # H - pixels (TESS native frame size)
WIDTH = 11              # W - pixels
WINDOWS_PER_TARGET = 1  # gap-aware windows per target (target-level split)

# TIME_BINS is chosen so that one bin is SHORTER than a transit. A Kepler
# long-cadence light curve spans ~33 d with ~1626 valid cadences, so 64 bins
# would put ~0.52 d in each bin and average a ~0.1 d transit with ~4x more
# out-of-transit flux, diluting the depth ~5x. 256 bins give ~0.13 d per bin,
# which resolves a transit inside a single bin.
MAX_CADENCES = 6000
N_PHYSICS_FEATURES = 10  # folded depth, duration, folding gain, odd/even,
                         # centroid offset, folded SNR, neighbour contrast,
                         # secondary eclipse, target dip share,
                         # absolute raw depth in ppm
MAX_NEIGHBOUR_PIXELS = 10  # best-observed channels compared against the target

# ------------------------------------------------- target-level split (70/15/15)
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# ------------------------------------------------------------------ training
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8
MC_SAMPLES = 30
THRESHOLD = 0.5         # fixed for every model - never tuned per model
NUM_WORKERS = 0

# ------------------------------------------------------- physics / PGSEV
# The eight evidence features, all derived from the pixel cube folded on the
# public catalog ephemeris (see src/physics_features.py).
PHYSICS_NAMES = ["transit_depth", "duration_consistency", "periodicity",
                 "symmetry", "centroid_stability", "snr",
                 "neighbor_contrast", "secondary_eclipse", "target_dip_share",
                 "raw_depth_ppm"]

# Fixed, documented, label-free domain priors. Each entry is
# (transform, lo, hi): the evidence of a feature is 0 at lo and 1 at hi.
# "clip"   -> larger raw value  = stronger planet evidence
# "invert" -> larger raw value  = stronger false-positive evidence
PGSEV_PRIORS = {
    "transit_depth":        ("clip",   0.0,   6.0),    # folded depth in sigma
    "duration_consistency": ("clip",   0.0,   1.0),
    "periodicity":          ("clip",   1.0,   4.0),    # folding coherence gain
    "symmetry":             ("invert", 0.0,   1.0),    # odd/even mismatch is bad
    "centroid_stability":   ("invert", 0.0,   1.5),    # off-centre dip is bad
    "snr":                  ("clip",   0.0,   8.0),
    "neighbor_contrast":    ("clip",   -4.0,  4.0),    # dip vs strongest neighbour
    "secondary_eclipse":    ("invert", 0.0,   3.0),    # secondary dip is bad
    "target_dip_share":     ("clip",   0.0,   1.0),    # dip centred on the target
    # Absolute folded depth of the RAW star channel, log-scaled, and INVERTED.
    # The inversion is basic eclipse physics, fixed a priori: a planet dims its
    # star by 1e-4..1e-3 of its flux, whereas an eclipsing or blended binary -
    # the dominant false-positive population in both KOI and TOI - typically
    # dims it by 1e-2..1e-1. A very deep dip is therefore evidence AGAINST a
    # planet. This is a domain prior, not a value fitted to the labels.
    "raw_depth_ppm":        ("invert", 100.0, 1e4),
}

PGSEV_WEIGHTS = {"transit_depth": 0.12, "duration_consistency": 0.05,
                 "periodicity": 0.12, "symmetry": 0.06,
                 "centroid_stability": 0.10, "snr": 0.06,
                 "neighbor_contrast": 0.10, "secondary_eclipse": 0.05,
                 "target_dip_share": 0.14, "raw_depth_ppm": 0.20}


def ensure_dirs():
    """Create every output directory used by the pipeline."""
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
