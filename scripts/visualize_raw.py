# scripts/visualize_raw.py

import sys
sys.path.append("..")

from configs import paths
from utils.eeg_loader import load_edf
from utils.plot_utils import plot_raw_segment

if __name__ == "__main__":
    raw = load_edf(paths.EDF_FILE)

    print(raw.info)  # Check sampling rate, channels, etc.

    # Plot first 10s from first channel
    plot_raw_segment(raw, channel_idx=0, start=0, duration=10,
                     save_path=f"{paths.RESULTS_DIR}/figures/raw_example.png")