"""CryoVIT model architecture for 3D tomogram segmentation."""

import torch
from torch import Tensor
from torch import nn

from cryovit.models.base_model import BaseModel


class CryoVIT(BaseModel):
    """CryoVIT model implementation."""

    def __init__(self, **kwargs) -> None:
        """Initializes the CryoVIT model with specific convolutional and synthesis blocks."""
        super(CryoVIT, self).__init__(**kwargs)

        self.layers = nn.Sequential(
            nn.Conv3d(1536, 1024, 1, padding="same"),  # projection to a lower dimension
            nn.GELU(),
            SynthesisBlock(1024, 192, 128, d1=32, d2=24),
            SynthesisBlock(128, 64, 32, d1=16, d2=12),
            SynthesisBlock(32, 32, 32, d1=8, d2=4),
            SynthesisBlock(32, 16, 8, d1=2, d2=1),
        )

        # output layer for generating the final segmentation
        self.output_layer = nn.Sequential(
            nn.Conv3d(8, 8, 3, padding="same"),
            nn.GELU(),
            nn.Conv3d(8, 1, 3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the CryoVIT model."""
        x = x.unsqueeze(0)
        x = self.layers(x)
        x = self.output_layer(x)
        x = torch.clip(x, -5.0, 5.0)
        return x.squeeze()


class SynthesisBlock(nn.Module):
    """Synthesis block for anisotropic upscaling with dilated convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, d1: int, d2: int) -> None:
        """Initializes the Synthesis block for anisotropic upscaling.

        Args:
            c1 (int): Number of channels in the input volume.
            c2 (int): Number of channels in the intermediate tensor.
            c3 (int): Number of channels in the output upscaled volume.
            d1 (int): Depthwise dilation rate for the first 3D Conv layer.
            d2 (int): Depthwise dilation rate for the second 3D Conv layer.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.GroupNorm(max(8, c1 // 8), c1, eps=1e-3),
            nn.Conv3d(c1, c2, 3, padding="same", dilation=(d1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(c2, c2, 3, padding="same", dilation=(d2, 1, 1)),
            nn.GELU(),
            nn.ConvTranspose3d(c2, c3, (1, 2, 2), stride=(1, 2, 2)),  # upscale by 2
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the synthesis block."""
        return self.layers(x)
