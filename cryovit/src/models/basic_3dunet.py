import torch
from torch import nn

from .base_module import BaseModule
from .blocks import AnalysisBlock, SynthesisBlock


class Basic3DUNet(BaseModule):
    def __init__(self, output_layer, **kwargs):
        super(Basic3DUNet, self).__init__(**kwargs)
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


if __name__ == "__main__":
    pass
