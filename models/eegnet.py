import torch
import torch.nn as nn
import torch.nn.functional as F
class EEGNetPrior(nn.Module):
    """
    EEGNet adapted as a DIP-style prior.
    Takes noise input [B, C, T], reconstructs [B, C, T].
    """
    def __init__(self, n_channels, samples, F1=8, D=2, F2=None, dropout=0.25):
        super().__init__()
        if F2 is None:
            F2 = F1 * D

        self.conv_time = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn_time = nn.BatchNorm2d(F1)
        self.conv_depth = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn_depth = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D,
                                        padding=(0, 8), bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Final conv to reconstruct [C,T]
        self.out_conv = nn.Conv2d(F2, 1, (1, 1))

    def forward(self, x, T_target=None):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)

        # encoder
        x = self.conv_time(x)
        x = self.bn_time(x)
        x = self.conv_depth(x)
        x = self.bn_depth(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv_sep_depth(x)
        x = self.conv_sep_point(x)
        x = self.bn_sep(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # reconstruction head
        y = self.out_conv(x)  # shape [B,1,1,T_out] or [B,1,T_out]

        if y.ndim == 4 and y.shape[2] == 1:
            y = y.squeeze(2)  # -> [B,1,T_out]

        if T_target is not None and y.shape[-1] != T_target:
            y = F.interpolate(y, size=T_target, mode="linear", align_corners=False)

        return y