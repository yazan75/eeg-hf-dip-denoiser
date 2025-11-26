# scripts/run_dip.py
"""
DIP-on-EEG experiment runner.

Key behaviors
-------------
• Train per 30 s window (DIP is per-sample), skipping artifacted windows for the "clean" pass.
• Then run a second "artifact-only" pass that intentionally uses windows flagged as artifacted.
• Early-stopping selection:
    - SELECTION="snr" : maximize SNR (computed in SNR_SELECTION_MODE: "raw" or "std")
    - SELECTION="mse" : minimize validation MSE on the (standardized) target
• SNR aggregation over non-EEG groups: AGG_METHOD in {"median","mean"}.
• Model choices via get_dip(): "mini_relu" (your original), "leakydeep" (deeper LeakyReLU).
• Metrics reported in raw volts; DIP training runs on z-scored target.

Notes
-----
DIP does not generalize across windows; each window is optimized separately from noise.
So the "artifact-only" phase also trains per artifact window, but we tag results as is_artifact=1.
"""

import os, sys, json
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from configs import paths
from models.dip_1d import get_dip
from utils.eeg_loader import (
    load_edf, load_stage_table, find_channel_index,
    get_data_window, get_block_window, classify_channels,
    is_artifact_window, amplitude_sanity_report
)
from utils.metrics import (
    compute_band_table,
    compute_snr_eeg_vs_noneeg_raw,
    compute_snr_eeg_vs_noneeg_std,
    append_readable_metrics_csv,
    write_json,
)
from utils.plot_utils import (
    plot_signal,
    plot_psd,
    plot_signal_comparison,
    plot_psd_comparison
)
from utils.metrics_eeg import export_presentation_table
from models.dip_1d import MODEL_ALIASES
# ---------------- Experiment toggles (local to this script) ----------------
MODEL_NAME = "hf_eegnet"     # {"eegnet","mini_relu","leakydeep", "leakymini","hfeegnet}
NORMALIZE_FOR_TRAIN = True   # z-score target EEG window for DIP training
SELECTION = "mse"            # {"snr","mse"} early-stopping
SNR_SELECTION_MODE = "raw"   # {"raw","std"} how SNR is computed for selection (when SELECTION=="snr")
AGG_METHOD = "mean"        # {"median","mean"} aggregation over non-EEG channels in SNR

SNAPSHOT_EVERY = 200
MAX_STEPS = 3000
PATIENCE = 600
NOISE_CH = 1
LR = 1e-3

# ---------------- Utils ----------------
def ensure_dirs():
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    os.makedirs(paths.FIG_DIR, exist_ok=True)
    out_root = os.path.join(paths.RESULTS_DIR, "denoised")
    os.makedirs(out_root, exist_ok=True)
    return out_root

def zscore(x: np.ndarray):
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-12)
    return (x - mu) / sd, mu, sd

def denorm(xn: np.ndarray, mu: float, sd: float):
    return xn * sd + mu

def pick_first_window(stages, label, want_artifact: bool, raw, sfreq, ch_idx):
    """
    Find first 30s window for a given label that matches artifact criterion.
    want_artifact=False -> use clean window (skip artifacts)
    want_artifact=True  -> use artifacted window (>= threshold)
    """
    for s in stages:
        if s["stage"] != label or s["dur"] < paths.WINDOW_SEC:
            continue
        start = s["start"]
        is_art = is_artifact_window(start, paths.WINDOW_SEC, sfreq, ch_idx)
        if want_artifact and is_art:
            return start
        if (not want_artifact) and (not is_art):
            return start
    return None

# ---------------- DIP training loop ----------------
def dip_optimize_single_window(
    sig_raw: np.ndarray,
    sfreq: float,
    eog_block: np.ndarray,
    emg_block: np.ndarray,
    ecg_block: np.ndarray,
    model_name: str,
    selection: str,
    snr_mode: str,
    agg_method: str,
):
    """
    Optimize DIP on a single window.
    Returns:
      y_best (raw volts), best_step, snapshots (list of dicts)
    """
    # Standardize target for training
    if NORMALIZE_FOR_TRAIN:
        sig_std, mu, sd = zscore(sig_raw)
    else:
        sig_std, mu, sd = sig_raw.copy(), 0.0, 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(sig_std.astype(np.float32))[None, None, :].to(device)
    T = x.shape[-1]

    if model_name in ["eegnet", "hf_eegnet", "hfeegnet"]:
        # structured priors expect single-channel noise
        z = torch.randn(1, NOISE_CH, T, device=device)
        net = get_dip(model_name, samples=T).to(device)
    else:
        # classic DIP models take multi-channel noise input
        z = torch.randn(1, NOISE_CH, T, device=device)
        net = get_dip(model_name, noise_ch=NOISE_CH).to(device)
    #opt = optim.Adam(net.parameters(), lr=LR)
    opt = optim.SGD(net.parameters(), lr=LR, momentum=0.8)
    mse_loss = nn.MSELoss()

    # Track best
    if selection == "snr":
        best_score = -1e15
    else:  # "mse"
        best_score = 1e15
    best_step = 0
    best_y_std = None

    no_improve = 0
    last_snap = 0

    snapshots = []

    for step in range(1, MAX_STEPS + 1):
        opt.zero_grad()
        # forward
        if model_name == "eegnet":
            y = net(z, T_target=x.shape[-1])  # EEGNetPrior handles resizing
        else:
            y = net(z)
            # safety resize if shape mismatch
            if y.shape[-1] != x.shape[-1]:
                y = F.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)

        # loss
        loss = mse_loss(y, x)  # train on std scale
        loss.backward()
        opt.step()

        if step % SNAPSHOT_EVERY == 0 or step in (1, MAX_STEPS):
            y_std = y.detach().cpu().numpy().squeeze()
            y_raw = denorm(y_std, mu, sd)

            # SNR for selection
            if selection == "snr":
                if snr_mode == "raw":
                    _, snr_sel_db, _ = compute_snr_eeg_vs_noneeg_raw(
                        eeg_sig_raw=y_raw, sfreq=sfreq,
                        eog_block_raw=eog_block, emg_block_raw=emg_block, ecg_block_raw=ecg_block,
                        agg=agg_method
                    )
                else:  # "std"
                    _, snr_sel_db, _ = compute_snr_eeg_vs_noneeg_std(
                        eeg_sig_raw=y_raw, sfreq=sfreq,
                        eog_block_raw=eog_block, emg_block_raw=emg_block, ecg_block_raw=ecg_block,
                        agg=agg_method
                    )
                score = snr_sel_db
                mse_std = float(np.mean((y_std - sig_std) ** 2))
                if step in (1, 50, 100, 200, 400, 800, 1600):
                    print(f"[dbg] step={step:4d}  SNR_dB({snr_mode})={snr_sel_db: .3f}  MSE(std)={mse_std:.3e}")
            else:
                mse_std = float(np.mean((y_std - sig_std) ** 2))
                score = -mse_std  # maximize -MSE for consistency
                if step in (1, 50, 100, 200, 400, 800, 1600):
                    print(f"[dbg] step={step:4d}  MSE(std)={mse_std:.3e}")

            # record snapshot
            snapshots.append({"step": step, "score": float(score), "mse_std": float(mse_std)})

            # improve?
            improved = score > best_score + (1e-12 if selection == "snr" else 0.0)
            if improved:
                best_score = score
                best_step = step
                best_y_std = y_std.copy()
                no_improve = 0
            else:
                no_improve += (step - last_snap)
            last_snap = step

        if no_improve >= PATIENCE:
            print(f"[early-stop] no improvement ~{PATIENCE} steps, stop @ {step}")
            break

    if best_y_std is None:
        best_y_std = y.detach().cpu().numpy().squeeze()
        best_step = step

    y_best = denorm(best_y_std, mu, sd)  # back to raw volts
    return y_best, int(best_step), snapshots


# ---------------- Main ----------------
if __name__ == "__main__":
    out_root = ensure_dirs()
    raw = load_edf(paths.EDF_FILE)

    # quick amplitude sanity (first 30s)
    amplitude_sanity_report(
        raw, seconds=30.0,
        save_csv_path=os.path.join(paths.RESULTS_DIR, "sanity_amplitudes_first30s.csv")
    )

    stages = load_stage_table(paths.ANNOT_FILE)
    sfreq = raw.info["sfreq"]

    # Channel classification (once)
    groups = classify_channels(raw)
    channel_lists_path = os.path.join(paths.RESULTS_DIR, "channel_lists.json")
    write_json({k: [raw.ch_names[i] for i in v] for k, v in groups.items()}, channel_lists_path)
    print("[Info] Channel groups written to:", channel_lists_path)

    # Choose a single channel to iterate (your current workflow)
    ch_idx = find_channel_index(raw, paths.PLOT_CHANNEL_NAME)
    ch_name = raw.ch_names[ch_idx]

    # CSV (compact, human-readable)
    csv_path = os.path.join(paths.RESULTS_DIR, "summary_metrics_readable.csv")
    header = [
        "label", "channel", "start_s", "best_step",
        "SNR_raw_dB", "SNR_dip_dB", "Delta_SNR_dB",
        "EEG_total_0p5_500_raw", "EEG_total_0p5_500_dip", "EEG_total_delta_pct",
        "NonEEG_total_0p5_500_raw",
        "EEG_0p5_40_raw", "EEG_0p5_40_dip", "EEG_0p5_40_delta_pct",
        # ---- HF splits ----
        "EEG_80_250_raw", "EEG_80_250_dip", "HF1_suppress_pct",
        "EEG_250_500_raw", "EEG_250_500_dip", "HF2_suppress_pct",
        "HF_total_suppress_pct",
        # ---- experiment info ----
        "selection", "agg", "snr_mode", "model", "is_artifact"
    ]

    def run_one(label: str, start: float, is_artifact_flag: int):
        # window data (raw volts)
        sig_raw, times = get_data_window(raw, ch_idx, start, paths.WINDOW_SEC)
        eog_block, _ = get_block_window(raw, groups["EOG"], start, paths.WINDOW_SEC)
        emg_block, _ = get_block_window(raw, groups["EMG"], start, paths.WINDOW_SEC)
        ecg_block, _ = get_block_window(raw, groups["ECG"], start, paths.WINDOW_SEC)

        # SNR BEFORE (raw)
        meta_raw, snr_raw_db, parts_raw = compute_snr_eeg_vs_noneeg_raw(
            eeg_sig_raw=sig_raw, sfreq=sfreq,
            eog_block_raw=eog_block, emg_block_raw=emg_block, ecg_block_raw=ecg_block,
            agg=AGG_METHOD
        )
        print(f"[parts_raw] EEG_total={parts_raw['EEG_total_0p5_500']:.3e}  "
              f"NonEEG_total={parts_raw['NonEEG_total_0p5_500']:.3e}  "
              f"(EOG={parts_raw['NonEEG_EOG_total_0p5_500']:.3e}, "
              f"EMG={parts_raw['NonEEG_EMG_total_0p5_500']:.3e}, "
              f"ECG={parts_raw['NonEEG_ECG_total_0p5_500']:.3e})")

        # ---- train DIP on this window ----
        y_best, best_step, _ = dip_optimize_single_window(
            sig_raw=sig_raw, sfreq=sfreq,
            eog_block=eog_block, emg_block=emg_block, ecg_block=ecg_block,
            model_name=MODEL_NAME, selection=SELECTION,
            snr_mode=SNR_SELECTION_MODE, agg_method=AGG_METHOD
        )

        # SNR AFTER (raw)
        meta_dip, snr_dip_db, parts_dip = compute_snr_eeg_vs_noneeg_raw(
            eeg_sig_raw=y_best, sfreq=sfreq,
            eog_block_raw=eog_block, emg_block_raw=emg_block, ecg_block_raw=ecg_block,
            agg=AGG_METHOD
        )

        # band powers (raw)
        eeg_bands_raw = compute_band_table(sig_raw, sfreq)
        eeg_bands_dip = compute_band_table(y_best, sfreq)
        eeg_low_pre  = sum(eeg_bands_raw[b] for b in ("delta","theta","alpha","beta"))
        eeg_low_post = sum(eeg_bands_dip[b] for b in ("delta","theta","alpha","beta"))
        eeg_low_delta_pct = 100.0 * ((eeg_low_post - eeg_low_pre) / (eeg_low_pre + 1e-15))

        hf1_pre = eeg_bands_raw["HF1"]
        hf1_post = eeg_bands_dip["HF1"]
        hf2_pre = eeg_bands_raw["HF2"]
        hf2_post = eeg_bands_dip["HF2"]

        hf1_suppress_pct = 100.0 * (1.0 - (hf1_post / (hf1_pre + 1e-15)))
        hf2_suppress_pct = 100.0 * (1.0 - (hf2_post / (hf2_pre + 1e-15)))
        hf_total_suppress_pct = 100.0 * (1.0 - ((hf1_post + hf2_post) / (hf1_pre + hf2_pre + 1e-15)))
        # save signals + meta
        tag = f"{label}_{ch_name}_{int(paths.WINDOW_SEC)}s_{'ART' if is_artifact_flag else 'CLEAN'}"
        out_root = ensure_dirs()
        np.save(os.path.join(out_root, f"{tag}_raw.npy"), sig_raw)
        np.save(os.path.join(out_root, f"{tag}_dip.npy"), y_best)
        meta = dict(
            label=label, channel=ch_name, start=float(start), dur=float(paths.WINDOW_SEC),
            sf=float(sfreq), best_step=int(best_step),
            selection=SELECTION, agg=AGG_METHOD,
            snr_mode=SNR_SELECTION_MODE, model=MODEL_NAME, is_artifact=int(is_artifact_flag)
        )
        # ---- Save metadata ----
        with open(os.path.join(out_root, f"{tag}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # ---- Visualization setup ----
        title_suffix = "ARTIFACT" if is_artifact_flag else "CLEAN"
        tag = f"{label}_{ch_name}_{int(paths.WINDOW_SEC)}s_{title_suffix}"

        # Remove DC (baseline) component
        sig_raw_vis = sig_raw - np.mean(sig_raw)
        y_best_vis = y_best - np.mean(y_best)

        # Normalize for better visual comparison (same amplitude scale)
        amp_scale = max(np.abs(sig_raw_vis).max(), np.abs(y_best_vis).max())
        sig_raw_vis /= (amp_scale + 1e-12)
        y_best_vis /= (amp_scale + 1e-12)

        # Build informative title tag
        title_tag = (
            f"Label={label} | CH={ch_name} | Prior={MODEL_ALIASES.get(MODEL_NAME, MODEL_NAME).upper()} | "
            f"Criterion={SELECTION.upper()} | Agg={AGG_METHOD.upper()} | "
            f"Step={best_step} | {title_suffix}"
        )

        # ---- Combined signal plot (Raw vs Denoised) ----
        plot_signal_comparison(
            times, sig_raw_vis, y_best_vis, title_tag,
            save_path=os.path.join(paths.FIG_DIR, f"{tag}_comparison.png")
        )

        # ---- Combined PSD plot ----
        plot_psd_comparison(
            sig_raw_vis, y_best_vis, sfreq, title_tag,
            save_path=os.path.join(paths.FIG_DIR, f"{tag}_PSD_comparison.png")
        )
        # CSV row
        row = {
            "label": label, "channel": ch_name, "start_s": float(start), "best_step": int(best_step),
            "SNR_raw_dB": snr_raw_db, "SNR_dip_dB": snr_dip_db,
            "Delta_SNR_dB": snr_dip_db - snr_raw_db,
            "EEG_total_0p5_500_raw": parts_raw["EEG_total_0p5_500"],
            "EEG_total_0p5_500_dip": parts_dip["EEG_total_0p5_500"],
            "EEG_total_delta_pct": 100.0 * ((parts_dip["EEG_total_0p5_500"] - parts_raw["EEG_total_0p5_500"])
                                            / (parts_raw["EEG_total_0p5_500"] + 1e-15)),
            "NonEEG_total_0p5_500_raw": parts_raw["NonEEG_total_0p5_500"],
            "EEG_0p5_40_raw": eeg_low_pre, "EEG_0p5_40_dip": eeg_low_post,
            "EEG_0p5_40_delta_pct": eeg_low_delta_pct,
            "selection": SELECTION, "agg": AGG_METHOD,
            "snr_mode": SNR_SELECTION_MODE, "model": MODEL_ALIASES.get(MODEL_NAME, MODEL_NAME),
            "is_artifact": int(is_artifact_flag),
            "EEG_80_250_raw": hf1_pre,
            "EEG_80_250_dip": hf1_post,
            "HF1_suppress_pct": hf1_suppress_pct,
            "EEG_250_500_raw": hf2_pre,
            "EEG_250_500_dip": hf2_post,
            "HF2_suppress_pct": hf2_suppress_pct,
            "HF_total_suppress_pct": hf_total_suppress_pct,
        }
        append_readable_metrics_csv(csv_path, header, row)

        return best_step

    # ---------------- Pass 1: CLEAN windows (skip artifacts) ----------------
    for label in paths.LABEL_ORDER:
        start = pick_first_window(stages, label, want_artifact=False, raw=raw, sfreq=sfreq, ch_idx=ch_idx)
        if start is None:
            continue
        print(f"[CLEAN] {label} @ {start:.1f}s")
        best_step = run_one(label, start, is_artifact_flag=0)
        print(f"[{label} CLEAN] best_step={best_step}")

    # ---------------- Pass 2: ARTIFACT windows (evaluate behavior) ----------------
    for label in paths.LABEL_ORDER:
        start = pick_first_window(stages, label, want_artifact=True, raw=raw, sfreq=sfreq, ch_idx=ch_idx)
        if start is None:
            continue
        print(f"[ARTIFACT] {label} @ {start:.1f}s")
        best_step = run_one(label, start, is_artifact_flag=1)
        print(f"[{label} ARTIFACT] best_step={best_step}")
if __name__ == "__main__":
            # ... your main loop code that runs run_one(...)
            # After all CSVs are saved:
            csv_path = os.path.join(paths.RESULTS_DIR, "summary_metrics_readable.csv")
            export_presentation_table(csv_path)