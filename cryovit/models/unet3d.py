import torch
from torch import nn

from cryovit.models.base_model import BaseModel


class UNet3D(BaseModel):
    def __init__(self, output_layer, **kwargs):
        super(UNet3D, self).__init__(**kwargs)
        self.output_layer = output_layer

        self.bottom_layer = nn.Sequential(
            nn.Conv3d(32, 32, (3, 3, 3), padding="same"),
            nn.Conv3d(32, 64, (3, 3, 3), padding="same"),
        )

        self.analysis_layers = nn.ModuleList(
            [
                AnalysisBlock(1, 8, stride=4),
                AnalysisBlock(8, 16, stride=4),
                AnalysisBlock(16, 32, stride=4),
            ]
        )

        self.synthesis_layers = nn.ModuleList(
            [
                SynthesisBlock(64, 32, 32, stride=4),
                SynthesisBlock(32, 16, 16, stride=4),
                SynthesisBlock(16, 8, 8, stride=4),
            ]
        )

    @torch.compile()
    def forward(self, x):
        analysis_outputs = []

        for layer in self.analysis_layers:
            x, y = layer(x)
            analysis_outputs.append(y)

        x = self.bottom_layer(x)

        for layer, skip_x in zip(self.synthesis_layers, analysis_outputs[::-1]):
            x = layer([x, skip_x])

        return self.output_layer(x)


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
