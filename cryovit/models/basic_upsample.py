import torch
from torch import nn

from .dino_module import DinoModule
from .losses import DiceLoss
from .metrics import DiceMetric


class BasicUpsample(DinoModule):
    def __init__(self, **kwargs):
        super(BasicUpsample, self).__init__(**kwargs)

        self.layers = nn.Sequential(
            nn.Conv3d(1536, 1024, 1, padding="same"),
            nn.GELU(),
            SynthesisBlock(1024, 192, 128, d1=32, d2=24),
            SynthesisBlock(128, 64, 32, d1=16, d2=12),
            SynthesisBlock(32, 32, 32, d1=8, d2=4),
            SynthesisBlock(32, 16, 8, d1=2, d2=1),
        )

        self.output_layer = nn.Sequential(
            nn.Conv3d(8, 8, 3, padding="same"),
            nn.GELU(),
            nn.Conv3d(8, 1, 3, padding="same"),
        )

    # @torch.compile()
    def forward(self, x):
        x = x["dino_features"].unsqueeze(0)
        x = self.layers(x)
        x = self.output_layer(x)
        x = torch.clip(x, -5.0, 5.0)
        return x.squeeze()


class SynthesisBlock(nn.Module):
    def __init__(self, c1, c2, c3, d1, d2) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.GroupNorm(max(8, c1 // 8), c1, eps=1e-3),
            nn.Conv3d(c1, c2, 3, padding="same", dilation=(d1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(c2, c2, 3, padding="same", dilation=(d2, 1, 1)),
            nn.GELU(),
            nn.ConvTranspose3d(c2, c3, (1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model_params = {
        "losses": [(DiceLoss, {})],
        "metrics": [(DiceMetric, {"threshold": 0.5})],
        "lr": 1e-3,
    }

    x = {
        f"dino_F{i}": torch.rand(1, 1536, 128, 32, 32, device="cuda")
        for i in [9, 19, 29, 39]
    }
    model = BasicUpsample(**model_params).cuda()
    model(x)
