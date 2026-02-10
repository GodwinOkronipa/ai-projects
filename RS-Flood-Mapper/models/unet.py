"""
PyTorch U-Net Neural Network Architecture for Remote Sensing Semantic Segmentation.

Author: godmode-dev
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Remote Sensing Binary Flood Extent Segmentation.

    Parameters
    ----------
    in_channels : int, default=4
        Number of input spectral bands (e.g. VV, VH, NDWI, NIR).
    out_channels : int, default=1
        Number of output probability mask channels (1 for binary classification).
    features : list of int
        Channel dimensions at encoder levels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: list = [32, 64, 128, 256]
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path (Contracting)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path (Expanding)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 convolution output block
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return torch.sigmoid(self.final_conv(x))


if __name__ == "__main__":
    model = UNet(in_channels=4, out_channels=1)
    x = torch.randn(2, 4, 128, 128)
    output = model(x)
    print("UNet Input Tensor Shape:", x.shape)
    print("UNet Output Mask Tensor Shape:", output.shape)