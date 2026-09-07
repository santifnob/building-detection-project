"""
Arquitectura U-Net usada para entrenar harshinde/spacenet-models.

Fuente: https://huggingface.co/spaces/harshinde/spacenet/blob/main/models/unet.py
(el repo del modelo harshinde/spacenet-models solo trae los pesos, no el código
de la arquitectura; el autor la publicó en el repo del Space de demo).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Doble convolución con conexión residual opcional."""

    def __init__(self, in_channels: int, out_channels: int, residual: bool = True):
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if residual and in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            out = out + self.skip(x)
        return self.relu(out)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        pooled = self.dropout(self.pool(features))
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        base_features: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth

        self.encoders = nn.ModuleList()
        ch_in = in_channels
        encoder_channels: list[int] = []
        for i in range(depth):
            ch_out = base_features * (2 ** i)
            encoder_channels.append(ch_out)
            self.encoders.append(EncoderBlock(ch_in, ch_out, dropout=dropout))
            ch_in = ch_out

        bottleneck_ch = base_features * (2 ** depth)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)

        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = encoder_channels[i]
            ch_out = skip_ch
            self.decoders.append(DecoderBlock(ch_in, skip_ch, ch_out))
            ch_in = ch_out

        self.head = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skips.append(features)

        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.head(x)
