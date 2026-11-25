"""EILNet-style binary adaptation of Li et al. (Natural Gas Industry B, 2025).

The paper's encoder uses LSK selection and its decoder uses PagFM fusion.
The original three-class head is replaced by a two-class head for this dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import OutputHeads


class ConvBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.Dropout2d(0.05),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing),
            nn.Dropout2d(0.05),
            nn.LeakyReLU(0.1, inplace=True),
        )


class LSKBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.small = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.large = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.small_project = nn.Conv2d(channels, channels // 2, 1)
        self.large_project = nn.Conv2d(channels, channels // 2, 1)
        self.select = nn.Conv2d(2, 2, 7, padding=3)
        self.restore = nn.Conv2d(channels // 2, channels, 1)
        self.ffn = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x):
        features = self.norm(x)
        small = self.small_project(self.small(features))
        large = self.large_project(self.large(self.small(features)))
        combined = torch.cat((small, large), dim=1)
        weights = torch.sigmoid(self.select(torch.cat((combined.mean(1, keepdim=True), combined.amax(1, keepdim=True)), dim=1)))
        selected = self.restore(small * weights[:, :1] + large * weights[:, 1:])
        x = x + features * selected
        return x + self.ffn(x)


class PagFM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.skip_project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))
        self.decoder_project = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, skip, decoder):
        decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid((self.skip_project(skip) * self.decoder_project(decoder)).sum(dim=1, keepdim=True))
        return gate * skip + (1.0 - gate) * decoder


class EILNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 16, boundary_supervision: bool = False):
        super().__init__()
        widths = [base * 2**index for index in range(5)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoders.append(nn.Sequential(ConvBlock(previous, width), LSKBlock(width)))
            previous = width
        self.transition = ConvBlock(widths[-1], widths[-1] * 2)
        self.decoder_projections = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = widths[-1] * 2
        for width in reversed(widths):
            self.decoder_projections.append(nn.Conv2d(previous, width, 1))
            self.fusions.append(PagFM(width))
            self.decoders.append(ConvBlock(width, width))
            previous = width
        self.heads = OutputHeads(base, boundary_supervision)

    def forward(self, x):
        skips = []
        features = x
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = F.max_pool2d(features, 2)
        features = self.transition(features)
        for projection, fusion, decoder, skip in zip(
            self.decoder_projections, self.fusions, self.decoders, reversed(skips)
        ):
            features = decoder(fusion(skip, projection(features)))
        return self.heads(features, x.shape[-2:])
