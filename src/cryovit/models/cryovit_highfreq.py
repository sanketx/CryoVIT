"""CryoVIT-HF model architecture for 3D tomogram segmentation."""

import torch
from torch import Tensor, nn

from cryovit.models.base_model import BaseModel
from cryovit.types import BatchedTomogramData


class CryoVIT_HF(BaseModel):
    """CryoVIT_HF model implementation."""

    def __init__(self, **kwargs) -> None:
        """Initializes the CryoVIT_HF model with specific convolutional and synthesis blocks."""

        super().__init__(**kwargs)

        self.bottom_layers = nn.Sequential(
            nn.Conv3d(
                1536, 1024, 1, padding="same"
            ),  # projection to a lower dimension
            nn.GELU(),
            SynthesisBlock(1024, 192, 128, d1=32, d2=24),
        )

        self.synthesis_layers = nn.ModuleList(
            [
                SynthesisBlock(128, 64, 32, d1=16, d2=12),
                SynthesisBlock(64, 32, 32, d1=8, d2=4),
                SynthesisBlock(64, 16, 8, d1=2, d2=1),
            ]
        )

        # output layer for generating the final segmentation
        self.output_layer = nn.Sequential(
            # nn.Conv3d(8, 8, 3, padding="same"),
            nn.Conv3d(24, 8, 3, padding="same"),
            nn.GELU(),
            nn.Conv3d(8, 1, 3, padding="same"),
        )

        self.analysis_layers = nn.ModuleList(
            [
                AnalysisBlock(1, 16, 16),
                AnalysisBlock(16, 32, 32),
            ]
        )

    def forward_volume(self, x: Tensor, data: Tensor) -> Tensor:
        x = self.bottom_layers(x)

        d = data.unsqueeze(0).unsqueeze(0)
        o1 = self.analysis_layers[0].layers(d)
        p0 = self.analysis_layers[0].pool(o1)

        o2 = self.analysis_layers[1].layers(p0)
        p1 = self.analysis_layers[1].pool(o2)

        x = self.synthesis_layers[0](x)

        x = torch.cat([x, p1], dim=1)
        x = self.synthesis_layers[1](x)

        x = torch.cat([x, o2], dim=1)
        x = self.synthesis_layers[2](x)

        x = torch.cat([x, o1], dim=1)
        x = self.output_layer(x)

        x = torch.clip(x, -5.0, 5.0)
        return x

    def forward(self, batch: BatchedTomogramData) -> Tensor:  # type: ignore
        """Forward pass for the CryoVIT_HF model."""

        x = batch.tomo_batch  # (B, D, C, H, W)

        data = torch.from_numpy(batch.aux_data["data"][0])
        data = data.to(x.device, x.dtype, non_blocking=True)
        data = (data / 255.0) - 0.5

        x = x.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
        x = self.forward_volume(x, data)

        x = x.squeeze(1)  # (B, D, H, W)
        return torch.sigmoid(x)


class AnalysisBlock(nn.Module):
    """Block that performs down-sampling and feature extraction in UNet3D's analysis path."""

    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int
    ) -> None:
        """Initializes the AnalysisBlock with convolutional layers and pooling.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, (1, 2, 2), stride=(1, 2, 2)),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, (1, 3, 3), padding="same"),
            nn.InstanceNorm3d(mid_channels, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, (1, 3, 3), padding="same"),
            nn.InstanceNorm3d(out_channels, eps=1e-3, affine=True),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the analysis block."""

        x = self.layers(x)
        y = self.pool(x)
        return y, x


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
            nn.ConvTranspose3d(
                c2, c3, (1, 2, 2), stride=(1, 2, 2)
            ),  # upscale by 2
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the synthesis block."""

        return self.layers(x)
