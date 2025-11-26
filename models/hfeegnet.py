# models/hfeegnet.py
# ---------------------------------------------------------------------
# HFEEGNetPrior
# ---------------------------------------------------------------------
# A small 1D U-Net–style prior for DIP-on-EEG:
#   - LF branch: 3-level encoder/decoder with residual blocks
#                → smooth 0.5–40 Hz backbone
#   - HF branch: shallow dilated conv stack operating at full rate
#                → controlled HF residual (80–500 Hz-ish)
#   - Output: y = y_lf + gate * hf_scale * tanh(y_hf)
#
# IMPORTANT:
#   - noise_ch is now FLEXIBLE:
#       * If constructed with noise_ch=None (default), it will
#         infer C from the first input z of shape [B, C, T],
#         build the 1x1 projection on-the-fly, and remember C.
#       * Later calls must use the same C (per DIP run).
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- helpers ------------------------------ #

class ConvBlock1D(nn.Module):
    """
    Conv -> Act -> Conv -> Act   (no norm; padding keeps length)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 7,
                 activation: str = "lrelu", lrelu_slope: float = 0.2):
        super().__init__()
        p = k // 2
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            act = nn.LeakyReLU(negative_slope=lrelu_slope, inplace=True)
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=p),
            act,
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=p),
            act,
        )

    def forward(self, x):
        # [B, C_in, T] -> [B, C_out, T]
        return self.net(x)


class ResBlock1D(nn.Module):
    """
    Residual ConvBlock1D with optional 1x1 projection if channels change.
    y = ConvBlock(x) + Proj(x)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 7,
                 activation: str = "lrelu", lrelu_slope: float = 0.2):
        super().__init__()
        self.block = ConvBlock1D(
            c_in, c_out, k=k,
            activation=activation, lrelu_slope=lrelu_slope
        )
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(
            c_in, c_out, kernel_size=1
        )

    def forward(self, x):
        return self.block(x) + self.proj(x)


class Down(nn.Module):
    """AveragePool downsample by 2 (keeps channels)."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)


class Up(nn.Module):
    """Linear upsample by 2 (keeps channels)."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x):
        return self.up(x)


# --------------------------- HFEEGNetPrior --------------------------- #

class HFEEGNetPrior(nn.Module):
    """
    HFEEGNetPrior

    Input:
        z : [B, C, T]  (DIP noise; C can be 1, 16, 32, ...)
    Output:
        y : [B, 1, T]  (denoised EEG-like waveform)

    Structure:
      - Stem: 1x1 conv -> base feature channels.
      - LF branch: 3-level encoder/decoder U-Net with residual blocks.
      - HF branch: dilated conv stack on high-resolution features.
      - Combine: y = y_lf + gate * hf_scale * tanh(y_hf)

    noise_ch:
      - If None (default), adapt to the first input's channel count.
      - If not None, enforce that all inputs match that C.
    """

    def __init__(
        self,
        noise_ch: int | None = None,
        base_ch: int = 32,
        hp_kernel: int = 0,        # set >0 to subtract slow baseline (moving average)
    ):
        super().__init__()
        self.noise_ch = noise_ch         # can start as None
        self.base_ch = base_ch
        self.hp_kernel = hp_kernel

        # --------- stem --------- #
        # We *may* delay creating this until we see the first input
        if noise_ch is None:
            self.in_proj = None          # lazy init in forward
        else:
            self.in_proj = nn.Conv1d(noise_ch, base_ch, kernel_size=1)

        # --------- LF encoder --------- #
        # Level 1 (T)
        self.enc1 = ResBlock1D(base_ch, base_ch, k=7, activation="lrelu")

        # Level 2 (T/2)
        self.down1 = Down()
        self.enc2 = ResBlock1D(base_ch, base_ch + 16, k=7, activation="lrelu")
        ch2 = base_ch + 16  # e.g. 48 when base_ch=32

        # Level 3 (T/4)
        self.down2 = Down()
        self.enc3 = ResBlock1D(ch2, ch2 + 16, k=7, activation="lrelu")
        ch3 = ch2 + 16      # e.g. 64

        # Bottleneck (T/8)
        self.down3 = Down()
        self.b1 = ResBlock1D(ch3, ch3, k=7, activation="lrelu")
        self.b2 = ResBlock1D(ch3, ch3, k=7, activation="lrelu")

        # --------- LF decoder (concat skips) --------- #
        self.up3 = Up()
        self.dec3 = ResBlock1D(ch3 + ch3, ch3, k=7, activation="lrelu")   # [64+64 -> 64]

        self.up2 = Up()
        self.dec2 = ResBlock1D(ch3 + ch2, ch2, k=7, activation="lrelu")   # [64+48 -> 48]

        self.up1 = Up()
        self.dec1 = ResBlock1D(ch2 + base_ch, base_ch, k=7, activation="lrelu")  # [48+32 -> 32]

        # LF head: 32 -> 1
        self.lf_head = nn.Conv1d(base_ch, 1, kernel_size=7, padding=3)

        # Optional slow-baseline remover (very low freq)
        if hp_kernel and hp_kernel > 1:
            self.hp_pool = nn.AvgPool1d(kernel_size=hp_kernel,
                                        stride=1,
                                        padding=hp_kernel // 2)
        else:
            self.hp_pool = None

        # --------- HF residual branch --------- #
        # Operates on high-res feature x0 (after in_proj)
        self.hf_conv1 = nn.Conv1d(base_ch, base_ch, kernel_size=5, padding=2)
        self.hf_conv2 = nn.Conv1d(base_ch, base_ch, kernel_size=5,
                                  padding=4, dilation=2)
        self.hf_conv3 = nn.Conv1d(base_ch, base_ch // 2, kernel_size=3,
                                  padding=4, dilation=4)
        self.hf_smooth = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.hf_out = nn.Conv1d(base_ch // 2, 1, kernel_size=3, padding=1)

        self.hf_act = nn.LeakyReLU(0.2, inplace=True)

        # Learnable global HF gain (kept in [0,1] via sigmoid in forward)
        self.hf_gain = nn.Parameter(torch.tensor(0.0))

        # Fixed gating hyperparameters (smooth, conservative)
        self._gate_offset = 1.0
        self._gate_scale = 1.0

        # Cosmetic name
        self.pretty_name = "HF_EEGNet"

    # ------------------------------------------------------------------ #
    def _lf_branch(self, z: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Low-frequency U-Net backbone.
        Input:
            z : [B, C_noise, T]
        Returns:
            y_lf : [B, 1, T]
            x0   : stem feature [B, base_ch, T] (for HF branch)
        """
        # Stem (in_proj is guaranteed initialized before this is called)
        x0 = self.in_proj(z)              # [B, base_ch, T]

        # Encoder
        e1 = self.enc1(x0)                # [B, base_ch, T]

        x = self.down1(e1)                # [B, base_ch, T/2]
        e2 = self.enc2(x)                 # [B, ch2, T/2]

        x = self.down2(e2)                # [B, ch2, T/4]
        e3 = self.enc3(x)                 # [B, ch3, T/4]

        x = self.down3(e3)                # [B, ch3, T/8]

        # Bottleneck
        x = self.b1(x)
        x = self.b2(x)                    # [B, ch3, T/8]

        # Decoder with concat skips
        x = self.up3(x)                   # [B, ch3, T/4]
        x = torch.cat([x, e3], dim=1)     # [B, ch3+ch3, T/4]
        x = self.dec3(x)                  # [B, ch3, T/4]

        x = self.up2(x)                   # [B, ch3, T/2]
        x = torch.cat([x, e2], dim=1)     # [B, ch3+ch2, T/2]
        x = self.dec2(x)                  # [B, ch2, T/2]

        x = self.up1(x)                   # [B, ch2, T]
        x = torch.cat([x, e1], dim=1)     # [B, ch2+base_ch, T]
        x = self.dec1(x)                  # [B, base_ch, T]

        y_lf_raw = self.lf_head(x)        # [B, 1, T]

        # Optional very-slow drift removal
        if self.hp_pool is not None:
            baseline = self.hp_pool(y_lf_raw)
            y_lf = y_lf_raw - baseline
        else:
            y_lf = y_lf_raw

        return y_lf, x0

    def _hf_branch(self, x0: torch.Tensor) -> torch.Tensor:
        """
        High-frequency residual path.
        Input:
            x0 : [B, base_ch, T]
        Returns:
            y_hf : [B, 1, T]
        """
        x = self.hf_act(self.hf_conv1(x0))
        x = self.hf_act(self.hf_conv2(x))
        x = self.hf_act(self.hf_conv3(x))
        x = self.hf_smooth(x)
        y_hf = self.hf_out(x)             # [B, 1, T]
        return y_hf

    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, C, T]
        y: [B, 1, T]

        Adapts to the first C if self.noise_ch is None.
        """
        B, C, T = z.shape

        # --- lazy initialization of stem (so we accept any C cleanly) ---
        if self.in_proj is None:
            # First time we see an input: fix noise_ch and build conv
            self.noise_ch = C
            self.in_proj = nn.Conv1d(C, self.base_ch, kernel_size=1).to(z.device)
        else:
            # Enforce consistency across calls within the same run
            if self.noise_ch is not None and C != self.noise_ch:
                raise AssertionError(
                    f"HFEEGNetPrior expects noise_ch={self.noise_ch}, got {C}"
                )

        # LF backbone
        y_lf, x0 = self._lf_branch(z)     # [B,1,T], [B,base_ch,T]

        # HF residual
        y_hf_raw = self._hf_branch(x0)    # [B,1,T]

        # Global RMS-based gate (keeps behaviour gentle on huge-amplitude windows)
        rms = torch.sqrt(
            (y_lf ** 2).mean(dim=2, keepdim=True) + 1e-8
        )  # [B,1,1]
        gate = torch.sigmoid(self._gate_offset - self._gate_scale * rms)

        # Learnable HF gain in (0,1)
        hf_scale = torch.sigmoid(self.hf_gain)

        # Bound HF residual to avoid explosions, then apply gate & gain
        y_hf = torch.tanh(y_hf_raw)       # clip extreme spikes
        y = y_lf + gate * hf_scale * y_hf

        return y