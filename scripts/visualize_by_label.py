# scripts/visualize_by_label.py

import os, sys
sys.path.append("..")

from configs import paths
from utils.eeg_loader import load_edf, load_stage_table, find_channel_index, get_data_window, get_mean_across_channels
from utils.plot_utils import plot_signal, plot_psd, plot_spectrogram

def ensure_dirs():
    os.makedirs(paths.FIG_DIR, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()

    # Load data
    raw = load_edf(paths.EDF_FILE)
    stages = load_stage_table(paths.ANNOT_FILE)
    sf = raw.info["sfreq"]

    # Pick channel index
    ch_idx = find_channel_index(raw, paths.PLOT_CHANNEL_NAME)
    ch_name = raw.ch_names[ch_idx]

    # We’ll try one short window per label
    labels_to_try = paths.LABEL_ORDER
    win = paths.WINDOW_SEC

    found_any = False
    for label in labels_to_try:
        # find first epoch with this label and at least WINDOW_SEC duration
        seg = next((s for s in stages if s["stage"] == label and s["dur"] >= win), None)
        if seg is None:
            continue

        start = seg["start"]
        # 1) single channel
        sig, times = get_data_window(raw, ch_idx, start, win)
        plot_signal(times, sig, f"{label} | {ch_name} | {win}s",
                    save_path=f"{paths.FIG_DIR}/{label}_{ch_name}_{win}s_time.png")

        # 2) mean across channels
        mean_sig, t_mean = get_mean_across_channels(raw, start, win)
        plot_signal(t_mean, mean_sig, f"{label} | Mean across {raw.info['nchan']} ch | {win}s",
                    save_path=f"{paths.FIG_DIR}/{label}_MEAN_{win}s_time.png")

        # 3) PSD (single channel)
        plot_psd(sig, sf, fmax=paths.PSD_FMAX,
                 title=f"{label} | {ch_name} | PSD 0-{paths.PSD_FMAX} Hz",
                 save_path=f"{paths.FIG_DIR}/{label}_{ch_name}_PSD.png")

        # 4) Spectrogram (single channel)
        plot_spectrogram(sig, sf, fmax=paths.PSD_FMAX,
                         title=f"{label} | {ch_name} | Spectrogram 0-{paths.PSD_FMAX} Hz",
                         save_path=f"{paths.FIG_DIR}/{label}_{ch_name}_SPEC.png")

        print(f"Saved plots for label: {label} (start={start}s)")
        found_any = True

    if not found_any:
        print("No labels with sufficient duration were found. Check ANNOT_FILE or WINDOW_SEC.")
    else:
        print(f"Done. Figures in: {paths.FIG_DIR}")