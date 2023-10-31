import torch
from torch import nn


class AnalysisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.pool = nn.Conv3d(out_channels, out_channels, stride, stride=stride)

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding="same"),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.GELU(),
            nn.GroupNorm(8, out_channels, eps=1e-3),
        )

    def forward(self, x):
        x = self.layers(x)
        y = self.pool(x)
        return y, x


class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, stride, stride=stride
        )
        self.gelu = nn.GELU()

        self.layers = nn.Sequential(
            nn.Conv3d(
                out_channels + skip_channels,
                out_channels,
                3,
                dilation=1,
                padding="same",
            ),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding="same"),
            nn.GELU(),
            nn.GroupNorm(8, out_channels, eps=1e-3),
        )

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.gelu(self.upconv(x))
        x = torch.cat([x, skip_x], 1)  # channel concat
        x = self.layers(x)
        return x
