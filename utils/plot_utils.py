import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

# -----------------------------------------------------------
# 🧠 Basic individual plots (kept for compatibility)
# -----------------------------------------------------------
def plot_signal(times, signal, title, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(times, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_psd(signal, sfreq, fmax=500, title="PSD", save_path=None):
    f, pxx = welch(signal, fs=sfreq, nperseg=min(4096, len(signal)))
    plt.figure(figsize=(10, 4))
    plt.semilogy(f, pxx)
    plt.xlim(0, fmax)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_spectrogram(signal, sfreq, fmax=500, title="Spectrogram", save_path=None):
    f, t, Sxx = spectrogram(signal, fs=sfreq, nperseg=min(1024, len(signal)//4), noverlap=256)
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading='gouraud')
    plt.ylim(0, fmax)
    plt.colorbar(label="Power (dB)")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------
# ✨ NEW comparison plots for DIP experiments
# -----------------------------------------------------------

def plot_signal_comparison(times, raw_sig, den_sig, title, save_path=None):
    """
    Compare raw vs denoised signals on the same axes.
    Both should already be DC-removed and amplitude-normalized.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(times, raw_sig, color='gray', alpha=0.6, label='Raw EEG')
    plt.plot(times, den_sig, color='tab:blue', linewidth=1.2, label='DIP-Denoised')
    plt.title(title, fontsize=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized µV)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_psd_comparison(raw_sig, den_sig, sfreq, title, fmax=500, save_path=None):
    """
    Compare raw vs denoised PSD on the same axes.
    """
    f_raw, Pxx_raw = welch(raw_sig, fs=sfreq, nperseg=min(4096, len(raw_sig)))
    f_dip, Pxx_dip = welch(den_sig, fs=sfreq, nperseg=min(4096, len(den_sig)))

    plt.figure(figsize=(10, 4))
    plt.semilogy(f_raw, Pxx_raw, color='gray', alpha=0.6, label='Raw EEG')
    plt.semilogy(f_dip, Pxx_dip, color='tab:blue', linewidth=1.2, label='DIP-Denoised')
    plt.xlim(0, fmax)
    plt.title(title, fontsize=10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V²/Hz)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()