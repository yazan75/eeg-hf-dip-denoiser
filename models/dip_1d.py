# models/dip_1d.py
# ---------------------------------------------------------------------
# Minimal & Deep 1D DIP models for EEG denoising
#
# Presets you can request from get_dip():
#   - "mini_relu"  : your original small DIP (ReLU)
#   - "leakymini"  : same depth/width as mini but LeakyReLU activations
#   - "leakydeep"  : deeper U-Net–style with residual blocks (LeakyReLU)
#
# All models accept `noise_ch` (default 32). Lightweight & dependency-free.
# ---------------------------------------------------------------------

import torch
import torch.nn as nn

# ----------------------------- helpers ------------------------------

def _act_relu(inplace: bool = True):
    return nn.ReLU(inplace=inplace)

def _act_lrelu(slope: float = 0.2, inplace: bool = True):
    return nn.LeakyReLU(negative_slope=slope, inplace=inplace)

class ConvBlock1D(nn.Module):
    """
    Conv -> Act -> Conv -> Act   (no norm; padding keeps length)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 9,
                 activation: str = "relu", lrelu_slope: float = 0.2):
        super().__init__()
        p = k // 2
        if activation == "relu":
            act = _act_relu()
        elif activation == "lrelu":
            act = _act_lrelu(lrelu_slope)
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=p),
            act,
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=p),
            act,
        )

    def forward(self, x):  # [B, C, T] -> [B, C_out, T]
        return self.net(x)

class ResBlock1D(nn.Module):
    """
    Residual ConvBlock1D with optional 1x1 projection if channels change.
    y = ConvBlock(x) + Proj(x)
    """
    def __init__(self, c_in: int, c_out: int, k: int = 9,
                 activation: str = "lrelu", lrelu_slope: float = 0.2):
        super().__init__()
        self.block = ConvBlock1D(c_in, c_out, k=k,
                                 activation=activation, lrelu_slope=lrelu_slope)
        self.proj = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        return self.block(x) + self.proj(x)

class Down(nn.Module):
    """AveragePool downsample by 2."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class Up(nn.Module):
    """Linear upsample by 2."""
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def forward(self, x):
        return self.up(x)

# ---------------------- Model A: original small DIP (ReLU) ----------------------

class DIP1D_MiniReLU(nn.Module):
    """
    Small DIP (unchanged architecture, ReLU):
      noise_ch -> enc1(64) -> down -> enc2(128) -> up -> dec1(64) -> out(1)
      simple long skip: out(dec1 + e1)
    """
    def __init__(self, noise_ch: int = 32, k: int = 9):
        super().__init__()
        self.enc1 = ConvBlock1D(noise_ch, 64, activation="relu", k=k)
        self.down1 = Down()
        self.enc2 = ConvBlock1D(64, 128, activation="relu", k=k)

        self.up1  = Up()
        self.dec1 = ConvBlock1D(128, 64, activation="relu", k=k)

        self.out  = nn.Conv1d(64, 1, kernel_size=k, padding=k//2)

    def forward(self, z):
        e1 = self.enc1(z)                 # [B,64,T]
        e2 = self.enc2(self.down1(e1))    # [B,128,T/2]
        d1 = self.dec1(self.up1(e2))      # [B,64,T]
        y  = self.out(d1 + e1)            # simple skip
        return y

# ---------------------- Model A' : small DIP with LeakyReLU ----------------------

class DIP1D_LeakyMini(nn.Module):
    """
    Same depth/width as DIP1D_MiniReLU but with LeakyReLU activations.
    Useful when negative excursions matter or you want a softer nonlinearity.
    """
    def __init__(self, noise_ch: int = 32, k: int = 9, lrelu_slope: float = 0.2):
        super().__init__()
        self.enc1 = ConvBlock1D(noise_ch, 64, activation="lrelu", lrelu_slope=lrelu_slope, k=k)
        self.down1 = Down()
        self.enc2 = ConvBlock1D(64, 128, activation="lrelu", lrelu_slope=lrelu_slope, k=k)

        self.up1  = Up()
        self.dec1 = ConvBlock1D(128, 64, activation="lrelu", lrelu_slope=lrelu_slope, k=k)

        self.out  = nn.Conv1d(64, 1, kernel_size=k, padding=k//2)

    def forward(self, z):
        e1 = self.enc1(z)                 # [B,64,T]
        e2 = self.enc2(self.down1(e1))    # [B,128,T/2]
        d1 = self.dec1(self.up1(e2))      # [B,64,T]
        y  = self.out(d1 + e1)            # simple skip
        return y

# ---------------------- Model B: deeper LeakyReLU U-Net style ----------------------

class DIP1D_LeakyDeep(nn.Module):
    """
    Deeper U-Net–style DIP:
    - LeakyReLU
    - Residual blocks at each depth
    - 5 encoder/decoder levels with additive same-scale skips

    Channels per level: [64, 96, 144, 216, 324]
    """
    def __init__(self, noise_ch: int = 32, k: int = 9, lrelu_slope: float = 0.2):
        super().__init__()

        # Encoder
        self.e1 = ResBlock1D(noise_ch,  64,  k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.d1 = Down()
        self.e2 = ResBlock1D(64,        96,  k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.d2 = Down()
        self.e3 = ResBlock1D(96,        144, k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.d3 = Down()
        self.e4 = ResBlock1D(144,       216, k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.d4 = Down()
        self.e5 = ResBlock1D(216,       324, k=k, activation="lrelu", lrelu_slope=lrelu_slope)

        # Bottleneck
        self.b1 = ResBlock1D(324, 324, k=k, activation="lrelu", lrelu_slope=lrelu_slope)

        # Decoder + additive skips
        self.u1 = Up()
        self.d5 = ResBlock1D(324,       216, k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.u2 = Up()
        self.d6 = ResBlock1D(216,       144, k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.u3 = Up()
        self.d7 = ResBlock1D(144,       96,  k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.u4 = Up()
        self.d8 = ResBlock1D(96,        64,  k=k, activation="lrelu", lrelu_slope=lrelu_slope)

        # Tail + output
        self.tail = ConvBlock1D(64, 64, k=k, activation="lrelu", lrelu_slope=lrelu_slope)
        self.out  = nn.Conv1d(64, 1, kernel_size=k, padding=k//2)

    def forward(self, z):
        e1 = self.e1(z)                         # [B,64,T]
        e2 = self.e2(self.d1(e1))               # [B,96,T/2]
        e3 = self.e3(self.d2(e2))               # [B,144,T/4]
        e4 = self.e4(self.d3(e3))               # [B,216,T/8]
        e5 = self.e5(self.d4(e4))               # [B,324,T/16]
        b  = self.b1(e5)                        # [B,324,T/16]

        x  = self.d5(self.u1(b)) + e4           # [B,216,T/8]
        x  = self.d6(self.u2(x)) + e3           # [B,144,T/4]
        x  = self.d7(self.u3(x)) + e2           # [B,96,T/2]
        x  = self.d8(self.u4(x)) + e1           # [B,64,T]
        x  = self.tail(x)                       # [B,64,T]
        y  = self.out(x)                        # [B,1,T]
        return y

# ------------------------------ factory ------------------------------
from models.eegnet import EEGNetPrior   # <--- import new prior

def get_dip(model_name, noise_ch=32, samples=None):
    if model_name == "mini_relu":
        model = DIP1D_MiniReLU(noise_ch=noise_ch)
    elif model_name == "leakydeep":
        model = DIP1D_LeakyDeep(noise_ch=noise_ch)
    elif model_name == "leakymini":
        model = DIP1D_LeakyMini(noise_ch=noise_ch)
    elif model_name == "eegnet":   # <--- NEW
        if samples is None:
            raise ValueError("Must provide samples=T for EEGNetPrior")
        model = EEGNetPrior(n_channels=1, samples=samples)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # attach the pretty name
    model.pretty_name = MODEL_ALIASES.get(model_name, model_name)
    return model

    # ------------------------------ model name aliases ------------------------------
MODEL_ALIASES = {
        "mini_relu": "Shallow ReLU Prior",
        "leakymini": "Shallow LeakyRelu",
        "leakydeep": "Deep Residual LeakyRelu",
        "eegnet": "EEGNet"
    }