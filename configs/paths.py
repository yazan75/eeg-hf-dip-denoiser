# configs/paths.py
# Centralized paths & experiment toggles for the ANPHY + DIP pipeline.

import os

# --- Root folders ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure output dirs exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Data files (adjust to your layout if needed) ---
# Example subject EPCTL01
EDF_FILE = os.path.join(DATA_DIR, "EPCTL01", "EPCTL01 - fixed.edf")
ANNOT_FILE = os.path.join(DATA_DIR, "EPCTL01", "test1.txt")  # hypnogram / labels

# --- Artifact matrix usage ---
# Turn ON to skip windows that are heavily artifacted in the target EEG channel
USE_ARTIFACT_MATRIX = True
ARTIFACT_MAT_FILE = os.path.join(ROOT, "Artifact matrix", "EPCTL01_artndxn.mat")
# Skip a window if >= this fraction of samples are artifact (0.0–1.0)
ARTIFACT_MAX_FRAC = 0.10

# --- Experiment settings ---
LABEL_ORDER = ["W", "N1", "N2", "N3", "R", "L"]   # order to scan for windows
WINDOW_SEC = 30.0                                 # seconds per training window
PSD_FMAX = 500.0                                  # Hz (for PSD plots)

# Channel to visualize by default (must exist in raw.ch_names)
PLOT_CHANNEL_NAME = "C3"

# --- Lightweight reproducibility cache (per-run bundle) ---
RUN_CACHE_DIR = os.path.join(RESULTS_DIR, "runs")
os.makedirs(RUN_CACHE_DIR, exist_ok=True)