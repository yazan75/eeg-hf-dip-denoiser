# utils/metrics.py
"""
Metrics & reporting helpers for the EEG–DIP pipeline.

Provided API (used by run_dip.py):
- compute_snr_eeg_vs_noneeg_raw(...)
- compute_snr_eeg_vs_noneeg_std(...)
- compute_band_table(x, sfreq)
- append_readable_metrics_csv(csv_path, header, row)
- write_json(obj, path)

Conventions
-----------
• All *reporting* metrics use raw Volts.
• "std" SNR mode z-scores EACH signal independently per window BEFORE power
  (for selection/debug only; do not report those numbers).
• Band powers computed via single-segment Welch (Hann) using numpy FFT.
"""

from __future__ import annotations
import json
import os
from typing import Dict, Tuple, Optional

import numpy as np


# ------------------------ generic helpers ------------------------

def write_json(obj, path: str):
    """Small JSON writer with mkdir."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _band_power_welch(x: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
    """
    Single-segment Welch-like band power using numpy FFT with Hann window.
    Returns integrated power (V^2) in [fmin, fmax].
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8 or sfreq <= 0:
        return 0.0

    win = np.hanning(n)
    xw = x * win

    # rFFT & frequency grid
    spec = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sfreq))

    # one-segment Welch PSD estimate
    psd = (np.abs(spec) ** 2) / (win**2).sum()

    # integrate over band
    sel = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(sel):
        return 0.0
    df = freqs[1] - freqs[0]
    return float(psd[sel].sum() * df)


def _total_band_power(x: np.ndarray, sfreq: float, f_lo: float, f_hi: float) -> float:
    return _band_power_welch(x, sfreq, f_lo, f_hi)


def _aggregate_channel_block(block: Optional[np.ndarray],
                             sfreq: float, f_lo: float, f_hi: float,
                             agg: str = "median") -> float:
    """
    Aggregate per-channel band power for a block (EOG/EMG/ECG).
    block: array shape [n_ch, T] or empty (None or size 0).
    Returns scalar power (V^2).
    """
    if block is None:
        return 0.0
    block = np.asarray(block)
    if block.size == 0:
        return 0.0
    # ensure 2D [n_ch, T]
    if block.ndim == 1:
        block = block[None, :]
    per_ch = [_total_band_power(block[i], sfreq, f_lo, f_hi) for i in range(block.shape[0])]
    if len(per_ch) == 0:
        return 0.0
    return float(np.median(per_ch) if agg == "median" else np.mean(per_ch))


# ------------------------ SNR (RAW) for reporting ------------------------

def compute_snr_eeg_vs_noneeg_raw(
    eeg_sig_raw: np.ndarray,
    sfreq: float,
    eog_block_raw: Optional[np.ndarray],
    emg_block_raw: Optional[np.ndarray],
    ecg_block_raw: Optional[np.ndarray],
    f_lo: float = 0.5,
    f_hi: float = 500.0,
    agg: str = "median",
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    SNR in dB using RAW signals (Volts):
      SNR = 10*log10( EEG_total / (EOG_total + EMG_total + ECG_total + eps) )
    where 'total' is band-integrated power in [f_lo, f_hi].

    Returns (meta, snr_db, parts) where:
      parts = {
        "EEG_total_0p5_500", "NonEEG_EOG_total_0p5_500",
        "NonEEG_EMG_total_0p5_500", "NonEEG_ECG_total_0p5_500",
        "NonEEG_total_0p5_500"
      }
    """
    eps = 1e-15

    eeg_total = _total_band_power(eeg_sig_raw, sfreq, f_lo, f_hi)
    eog_total = _aggregate_channel_block(eog_block_raw, sfreq, f_lo, f_hi, agg=agg)
    emg_total = _aggregate_channel_block(emg_block_raw, sfreq, f_lo, f_hi, agg=agg)
    ecg_total = _aggregate_channel_block(ecg_block_raw, sfreq, f_lo, f_hi, agg=agg)
    noneeg_total = eog_total + emg_total + ecg_total

    snr_db = float(10.0 * np.log10((eeg_total + eps) / (noneeg_total + eps)))

    parts = {
        "EEG_total_0p5_500": float(eeg_total),
        "NonEEG_EOG_total_0p5_500": float(eog_total),
        "NonEEG_EMG_total_0p5_500": float(emg_total),
        "NonEEG_ECG_total_0p5_500": float(ecg_total),
        "NonEEG_total_0p5_500": float(noneeg_total),
    }
    meta = {"f_lo": f_lo, "f_hi": f_hi, "agg": agg}
    return meta, snr_db, parts


# ------------------------ SNR (STD) for selection only ------------------------

def compute_snr_eeg_vs_noneeg_std(
    eeg_sig_raw: np.ndarray,
    sfreq: float,
    eog_block_raw: Optional[np.ndarray],
    emg_block_raw: Optional[np.ndarray],
    ecg_block_raw: Optional[np.ndarray],
    f_lo: float = 0.5,
    f_hi: float = 500.0,
    agg: str = "median",
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Same SNR definition, but EACH signal (EEG and every non-EEG channel) is
    z-scored independently per window BEFORE power. Use ONLY for internal
    selection comparisons. Not for reporting.
    """
    def z(x):
        x = np.asarray(x, dtype=np.float64)
        mu, sd = x.mean(), x.std() + 1e-12
        return (x - mu) / sd

    eeg = z(eeg_sig_raw)

    def z_block(block):
        if block is None or np.size(block) == 0:
            return None
        block = np.asarray(block)
        if block.ndim == 1:
            block = block[None, :]
        return np.stack([z(block[i]) for i in range(block.shape[0])], axis=0)

    eog = z_block(eog_block_raw)
    emg = z_block(emg_block_raw)
    ecg = z_block(ecg_block_raw)

    return compute_snr_eeg_vs_noneeg_raw(
        eeg_sig_raw=eeg,
        sfreq=sfreq,
        eog_block_raw=eog,
        emg_block_raw=emg,
        ecg_block_raw=ecg,
        f_lo=f_lo, f_hi=f_hi, agg=agg
    )


# ------------------------ Band power table (raw) ------------------------

def compute_band_table(x: np.ndarray, sfreq: float) -> Dict[str, float]:
    """
    Compute a dictionary of band powers (V^2) from raw volts signal x.
    Bands aligned to our reporting:
      - delta: 0.5–4
      - theta: 4–8
      - alpha: 8–13
      - beta : 13–30
      - gamma: 30–80
      - HF1  : 80–200
      - HF2  : 200–500
      - total_lo: 0.5–40   (delta+theta+alpha+beta approximately)
      - total_hf: 80–500   (HF1+HF2)
      - total: 0.5–500
    """
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 80.0),
        "HF1":   (80.0, 200.0),
        "HF2":   (200.0, 500.0),
    }
    out = {}
    for k, (f1, f2) in bands.items():
        out[k] = _band_power_welch(x, sfreq, f1, f2)

    out["total_lo"] = _band_power_welch(x, sfreq, 0.5, 40.0)
    out["total_hf"] = _band_power_welch(x, sfreq, 80.0, 500.0)
    out["total"]    = _band_power_welch(x, sfreq, 0.5, 500.0)
    return out


# ------------------------ CSV writer (compact) ------------------------

def append_readable_metrics_csv(csv_path: str, header: list, row: dict):
    """
    Append a compact, human-readable CSV (creates file with header if missing).
    Floats are formatted smartly (fixed or scientific).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a") as f:
        if write_header:
            f.write(",".join(header) + "\n")

        vals = []
        for k in header:
            v = row.get(k, "")
            if isinstance(v, float):
                av = abs(v)
                if av >= 1e3 or (av > 0 and av < 1e-3):
                    vals.append(f"{v:.6e}")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        f.write(",".join(vals) + "\n")