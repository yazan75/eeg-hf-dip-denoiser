# utils/eeg_loader.py
"""
EEG I/O helpers for the DIP pipeline.

Provided functions:
- load_edf(edf_path)
- load_stage_table(txt_or_csv_path)
- find_channel_index(raw, name)
- classify_channels(raw) -> {"EEG":[...], "EOG":[...], "EMG":[...], "ECG":[...]}
- get_data_window(raw, ch_idx, start_sec, dur_sec) -> (sig_raw[T], times[T])
- get_block_window(raw, ch_indices, start_sec, dur_sec) -> (block[n_ch,T], times[T])
- is_artifact_window(start_sec, dur_sec, sfreq, ch_idx) -> bool  (uses artifact matrix if enabled)
- amplitude_sanity_report(raw, seconds=30.0, save_csv_path=None)

Notes
-----
• We assume MNE Raw is in Volts. All computations (SNR/band power) use Volts.
• amplitude_sanity_report() prints/saves microvolt (µV) summaries for human sanity checks.
• Artifact matrix is loaded lazily; supports MATLAB v7.3 (HDF5) or plain .npy. See configs.paths.
"""

from __future__ import annotations
import os
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import mne

from configs import paths

# Optional: HDF5 artifact reader for MATLAB v7.3 .mat
try:
    import h5py
    _H5PY_OK = True
except Exception:
    _H5PY_OK = False


# ---------------- I/O ----------------

def load_edf(edf_path: str) -> mne.io.BaseRaw:
    """Load EDF via MNE (preload=True)."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    return raw


def _parse_line_tokens(toks: List[str]) -> Optional[Dict[str, float]]:
    """
    Robust line parser for stage text/CSV.
    Accepts one of:
      - start,stage,dur
      - stage,start,dur
      - start,stage   (dur defaults to paths.WINDOW_SEC)
      - stage,start   (dur defaults to paths.WINDOW_SEC)
    """
    if len(toks) < 2:
        return None

    def _is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    # Normalize tokens (strip spaces)
    toks = [t.strip() for t in toks]

    if len(toks) >= 3:
        a, b, c = toks[:3]
        # Case 1: start(float), stage(str), dur(float)
        if _is_float(a) and not _is_float(b) and _is_float(c):
            return {"start": float(a), "stage": b, "dur": float(c)}
        # Case 2: stage(str), start(float), dur(float)
        if (not _is_float(a)) and _is_float(b) and _is_float(c):
            return {"start": float(b), "stage": a, "dur": float(c)}

    # Fallback 2-token rows
    a, b = toks[:2]
    if _is_float(a) and (not _is_float(b)):
        return {"start": float(a), "stage": b, "dur": float(paths.WINDOW_SEC)}
    if (not _is_float(a)) and _is_float(b):
        return {"start": float(b), "stage": a, "dur": float(paths.WINDOW_SEC)}

    return None


def load_stage_table(txt_or_csv_path: str) -> List[Dict[str, float]]:
    """
    Load a simple stage table (hypnogram-like) from CSV/TXT.
    Returns a list of dicts: {"start": float, "stage": "W/N1/N2/N3/R/L", "dur": float}.
    """
    stages: List[Dict[str, float]] = []

    if not os.path.exists(txt_or_csv_path):
        raise FileNotFoundError(f"Stage file not found: {txt_or_csv_path}")

    with open(txt_or_csv_path, "r") as f:
        sniffer = csv.Sniffer()
        sample = f.read(2048)
        f.seek(0)
        dialect = None
        has_header = False
        try:
            dialect = sniffer.sniff(sample)
            has_header = sniffer.has_header(sample)
        except Exception:
            pass

        reader = csv.reader(f, dialect=dialect) if dialect else csv.reader(f)
        if has_header:
            # skip header row
            try:
                next(reader)
            except StopIteration:
                pass

        for row in reader:
            if not row:
                continue
            parsed = _parse_line_tokens(row)
            if parsed is not None:
                stages.append(parsed)

    # Sanity sort by start
    stages.sort(key=lambda d: d["start"])
    return stages


def find_channel_index(raw: mne.io.BaseRaw, name: str) -> int:
    """Return exact match index for a channel name (raises if not found)."""
    try:
        idx = raw.ch_names.index(name)
        return idx
    except ValueError:
        raise ValueError(f"Channel '{name}' not found. Available: {raw.ch_names[:10]}...")


# ---------------- Channel groups ----------------

def classify_channels(raw: mne.io.BaseRaw) -> Dict[str, List[int]]:
    """
    Heuristic grouping by name substrings:
      EEG: default bucket (everything that's not EOG/EMG/ECG goes here)
      EOG: contains 'EOG'
      EMG: contains 'EMG' or 'LEG' (tibialis)
      ECG: contains 'ECG' or 'EKG'
    """
    groups = {"EEG": [], "EOG": [], "EMG": [], "ECG": []}
    for i, nm in enumerate(raw.ch_names):
        nm_up = nm.upper()
        if "EOG" in nm_up:
            groups["EOG"].append(i)
        elif ("EMG" in nm_up) or ("LEG" in nm_up):
            groups["EMG"].append(i)
        elif ("ECG" in nm_up) or ("EKG" in nm_up):
            groups["ECG"].append(i)
        else:
            groups["EEG"].append(i)
    return groups


# ---------------- Window extractors ----------------

def _sec_to_samp(raw: mne.io.BaseRaw, t: float) -> int:
    return int(round(t * float(raw.info["sfreq"])))


def get_data_window(raw: mne.io.BaseRaw, ch_idx: int, start_sec: float, dur_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sig[T], times[T]) for one channel in Volts."""
    sf = float(raw.info["sfreq"])
    beg = _sec_to_samp(raw, start_sec)
    end = _sec_to_samp(raw, start_sec + dur_sec)
    data, times = raw.get_data(picks=[ch_idx], start=beg, stop=end, return_times=True)
    sig = data[0].astype(np.float64, copy=True)
    return sig, times


def get_block_window(raw: mne.io.BaseRaw, ch_indices: List[int], start_sec: float, dur_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (block[n_ch,T], times[T]) for a list of channels.
    If ch_indices is empty, returns (empty array with shape (0,T), times).
    """
    sf = float(raw.info["sfreq"])
    beg = _sec_to_samp(raw, start_sec)
    end = _sec_to_samp(raw, start_sec + dur_sec)

    if not ch_indices:
        # build empty with correct T
        T = max(0, end - beg)
        empty = np.zeros((0, T), dtype=np.float64)
        times = np.arange(T) / sf + start_sec
        return empty, times

    data, times = raw.get_data(picks=ch_indices, start=beg, stop=end, return_times=True)
    block = data.astype(np.float64, copy=True)
    return block, times


# ---------------- Artifacts ----------------

_artifact_cache = None  # lazy-loaded (mask[T, n_ch]) or None


def _load_artifact_matrix_lazy() -> Optional[np.ndarray]:
    """
    Load artifact matrix from paths.ARTIFACT_MAT_FILE.
    Supported:
      - MATLAB v7.3 HDF5 (.mat) via h5py: dataset named 'mask' or first dataset found
      - .npy: np.load with shape [T, n_ch] or [n_epochs, n_ch] expanded to sample grid
    Returns mask in {0,1} with shape [T, n_ch] in sample-time, or None if not available.
    """
    global _artifact_cache
    if _artifact_cache is not None:
        return _artifact_cache

    if not paths.USE_ARTIFACT_MATRIX:
        _artifact_cache = None
        return None

    fpath = paths.ARTIFACT_MAT_FILE
    if not os.path.exists(fpath):
        _artifact_cache = None
        return None

    mask = None
    if fpath.lower().endswith(".mat"):
        if not _H5PY_OK:
            raise RuntimeError("Artifact .mat is v7.3/HDF5 but h5py is not available.")
        with h5py.File(fpath, "r") as h5:
            # try common keys
            for key in ("mask", "artifact_mask", "art_mask"):
                if key in h5:
                    d = np.array(h5[key])
                    mask = np.array(d)
                    break
            if mask is None:
                # fallback to the first dataset
                for key in h5.keys():
                    d = np.array(h5[key])
                    mask = np.array(d)
                    break
        # Ensure (T, n_ch)
        if mask is not None and mask.ndim == 2:
            # h5py often loads as (n_ch, T). Heuristic: put time dimension first.
            if mask.shape[0] < mask.shape[1]:
                mask = mask.T
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)

    elif fpath.lower().endswith(".npy"):
        mask = np.load(fpath)
        # Try to coerce to (T, n_ch)
        if mask.ndim == 2:
            if mask.shape[0] < mask.shape[1]:
                mask = mask.T
        mask = (mask > 0).astype(np.uint8)

    _artifact_cache = mask
    return _artifact_cache


def is_artifact_window(start_sec: float, dur_sec: float, sfreq: float, ch_idx: int) -> bool:
    """
    Decide if a window is artifacted based on fraction of artifact samples in the *target channel*.
    Uses paths.ARTIFACT_MAX_FRAC as threshold (>= frac => artifact).
    Returns False if artifact matrix is disabled/unavailable.
    """
    if not paths.USE_ARTIFACT_MATRIX:
        return False
    mask = _load_artifact_matrix_lazy()
    if mask is None:
        return False

    beg = int(round(start_sec * sfreq))
    end = int(round((start_sec + dur_sec) * sfreq))
    end = min(end, mask.shape[0])
    if beg >= end:
        return False

    # Handle possible channel mismatch gracefully
    ch = min(ch_idx, mask.shape[1] - 1)
    frac = float(mask[beg:end, ch].mean())
    return frac >= float(paths.ARTIFACT_MAX_FRAC)


# ---------------- Sanity reporting ----------------

def amplitude_sanity_report(raw: mne.io.BaseRaw, seconds: float = 30.0, save_csv_path: Optional[str] = None):
    """
    Compute simple amplitude stats in microvolts for the first `seconds`.
    Columns: ch, type(eeg/eog/emg/ecg/other), rms_uV, ptp_uV, maxabs_uV
    """
    sf = float(raw.info["sfreq"])
    beg = 0
    end = int(round(seconds * sf))
    end = min(end, raw.n_times)

    # Build type map (heuristic)
    types = {}
    groups = classify_channels(raw)
    for i in groups["EEG"]:
        types[i] = "eeg"
    for i in groups["EOG"]:
        types[i] = "eog"
    for i in groups["EMG"]:
        types[i] = "emg"
    for i in groups["ECG"]:
        types[i] = "ecg"
    for i in range(len(raw.ch_names)):
        if i not in types:
            types[i] = "other"

    rows = []
    for i, nm in enumerate(raw.ch_names):
        sig = raw.get_data(picks=[i], start=beg, stop=end)[0]
        sig_uV = sig * 1e6  # Volts -> microvolts
        rms_uV = float(np.sqrt(np.mean(sig_uV ** 2)))
        ptp_uV = float(np.ptp(sig_uV))
        maxabs_uV = float(np.max(np.abs(sig_uV)))
        rows.append((nm, types[i], rms_uV, ptp_uV, maxabs_uV))

    # Print a quick preview
    print("ch,type,rms_uV,ptp_uV,maxabs_uV")
    for nm, tp, rms, ptp, mx in rows[:10]:
        print(f"{nm},{tp},{rms:.6f},{ptp:.6f},{mx:.6f}")

    # Save CSV if requested
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        with open(save_csv_path, "w") as f:
            f.write("ch,type,rms_uV,ptp_uV,maxabs_uV\n")
            for nm, tp, rms, ptp, mx in rows:
                f.write(f"{nm},{tp},{rms:.6f},{ptp:.6f},{mx:.6f}\n")