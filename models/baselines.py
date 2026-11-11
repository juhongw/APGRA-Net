from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    lraspp_mobilenet_v3_large,
)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class OutputHeads(nn.Module):
    def __init__(self, channels: int, boundary_supervision: bool = False) -> None:
        super().__init__()
        self.segmentation = nn.Conv2d(channels, 2, 1)
        self.boundary = nn.Conv2d(channels, 1, 1) if boundary_supervision else None

    def forward(self, features: torch.Tensor, output_size) -> dict:
        logits = F.interpolate(
            self.segmentation(features), size=output_size, mode="bilinear", align_corners=False
        )
        boundary_logits = (
            F.interpolate(
                self.boundary(features), size=output_size, mode="bilinear", align_corners=False
            )
            if self.boundary is not None
            else None
        )
        probability = logits.softmax(dim=1)
        return {
            "logits": logits,
            "prob": probability[:, 1:2],
            "background_prob": probability[:, 0:1],
            "boundary_logits": boundary_logits,
            "decoder_features": features,
            "aux_logits": [],
            "prototypes": [],
        }


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32, boundary_supervision: bool = False) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8, base * 16]
        self.encoders = nn.ModuleList(
            [
                DoubleConv(in_channels, widths[0]),
                DoubleConv(widths[0], widths[1]),
                DoubleConv(widths[1], widths[2]),
                DoubleConv(widths[2], widths[3]),
            ]
        )
        self.center = DoubleConv(widths[3], widths[4])
        self.decoders = nn.ModuleList(
            [
                DoubleConv(widths[4] + widths[3], widths[3]),
                DoubleConv(widths[3] + widths[2], widths[2]),
                DoubleConv(widths[2] + widths[1], widths[1]),
                DoubleConv(widths[1] + widths[0], widths[0]),
            ]
        )
        self.heads = OutputHeads(widths[0], boundary_supervision)

    def forward(self, x: torch.Tensor) -> dict:
        output_size = x.shape[-2:]
        skips = []
        features = x
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = F.max_pool2d(features, 2)
        features = self.center(features)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            features = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat([features, skip], dim=1))
        return self.heads(features, output_size)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 24, boundary_supervision: bool = False) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.x00 = DoubleConv(in_channels, widths[0])
        self.x10 = DoubleConv(widths[0], widths[1])
        self.x20 = DoubleConv(widths[1], widths[2])
        self.x30 = DoubleConv(widths[2], widths[3])
        self.x01 = DoubleConv(widths[0] + widths[1], widths[0])
        self.x11 = DoubleConv(widths[1] + widths[2], widths[1])
        self.x21 = DoubleConv(widths[2] + widths[3], widths[2])
        self.x02 = DoubleConv(widths[0] * 2 + widths[1], widths[0])
        self.x12 = DoubleConv(widths[1] * 2 + widths[2], widths[1])
        self.x03 = DoubleConv(widths[0] * 3 + widths[1], widths[0])
        self.heads = OutputHeads(widths[0], boundary_supervision)

    @staticmethod
    def up(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict:
        x00 = self.x00(x)
        x10 = self.x10(F.max_pool2d(x00, 2))
        x20 = self.x20(F.max_pool2d(x10, 2))
        x30 = self.x30(F.max_pool2d(x20, 2))
        x01 = self.x01(torch.cat([x00, self.up(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up(x30, x20)], dim=1))
        x02 = self.x02(torch.cat([x00, x01, self.up(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21, x10)], dim=1))
        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12, x00)], dim=1))
        return self.heads(x03, x.shape[-2:])


class TransformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.norm1(x)
        value, _ = self.attention(value, value, value, need_weights=False)
        x = x + value
        return x + self.mlp(self.norm2(x))


class UNETRLike(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base: int = 24,
        embed: int = 192,
        depth: int = 4,
        boundary_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.e1 = DoubleConv(in_channels, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.patch = nn.Conv2d(base * 4, embed, kernel_size=4, stride=4)
        self.position = nn.Conv2d(embed, embed, 3, padding=1, groups=embed)
        self.blocks = nn.ModuleList([TransformerBlock(embed, 6) for _ in range(depth)])
        self.d3 = DoubleConv(embed + base * 4, base * 4)
        self.d2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.d1 = DoubleConv(base * 2 + base, base)
        self.heads = OutputHeads(base, boundary_supervision)

    def forward(self, x: torch.Tensor) -> dict:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        tokens = self.patch(e3)
        tokens = tokens + self.position(tokens)
        batch, channels, height, width = tokens.shape
        sequence = tokens.flatten(2).transpose(1, 2)
        for block in self.blocks:
            sequence = block(sequence)
        features = sequence.transpose(1, 2).reshape(batch, channels, height, width)
        features = F.interpolate(features, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        features = self.d3(torch.cat([features, e3], dim=1))
        features = F.interpolate(features, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        features = self.d2(torch.cat([features, e2], dim=1))
        features = F.interpolate(features, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.d1(torch.cat([features, e1], dim=1))
        return self.heads(features, x.shape[-2:])


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dimension: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, dimension, kernel_size=kernel, stride=stride, padding=kernel // 2
        )
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.projection(x)
        height, width = x.shape[-2:]
        x = self.normalization(x.flatten(2).transpose(1, 2))
        return x, height, width


class SpatialReductionAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        reduction: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError(f"dimension={dimension} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.scale = self.head_dimension**-0.5
        self.query = nn.Linear(dimension, dimension)
        self.key_value = nn.Linear(dimension, 2 * dimension)
        self.reduction = int(reduction)
        if self.reduction > 1:
            self.spatial_reduction = nn.Conv2d(
                dimension, dimension, kernel_size=self.reduction, stride=self.reduction
            )
            self.reduction_normalization = nn.LayerNorm(dimension)
        else:
            self.spatial_reduction = None
            self.reduction_normalization = None
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dimension, dimension)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        query = self.query(x).reshape(batch, tokens, self.heads, self.head_dimension)
        query = query.permute(0, 2, 1, 3)
        if self.spatial_reduction is not None:
            reduced = x.transpose(1, 2).reshape(batch, channels, height, width)
            reduced = self.spatial_reduction(reduced).flatten(2).transpose(1, 2)
            reduced = self.reduction_normalization(reduced)
        else:
            reduced = x
        key_value = self.key_value(reduced)
        key_value = key_value.reshape(
            batch, reduced.shape[1], 2, self.heads, self.head_dimension
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = self.attention_dropout(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.projection_dropout(self.projection(output))


class MixFeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.expand = nn.Linear(dimension, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(hidden, dimension)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.expand(x)
        batch, _, channels = x.shape
        spatial = x.transpose(1, 2).reshape(batch, channels, height, width)
        x = x + self.depthwise(spatial).flatten(2).transpose(1, 2)
        x = self.dropout(self.activation(x))
        return self.dropout(self.project(x))


class MixTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        reduction: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = SpatialReductionAttention(dimension, heads, reduction)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = MixFeedForward(dimension)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x), height, width))
        return x + self.drop_path(self.feed_forward(self.norm2(x), height, width))


class SegFormerB0(nn.Module):
    """SegFormer-B0 with the canonical MiT-B0 stage dimensions and MLP decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        decoder_dimension: int = 256,
        drop_path_rate: float = 0.1,
        boundary_supervision: bool = False,
    ) -> None:
        super().__init__()
        dimensions = (32, 64, 160, 256)
        depths = (2, 2, 2, 2)
        heads = (1, 2, 5, 8)
        reductions = (8, 4, 2, 1)
        self.patch_embeddings = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.stage_normalizations = nn.ModuleList()
        total_blocks = sum(depths)
        rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        block_index = 0
        previous = in_channels
        for stage, (dimension, depth, stage_heads, reduction) in enumerate(
            zip(dimensions, depths, heads, reductions)
        ):
            self.patch_embeddings.append(
                OverlapPatchEmbedding(
                    previous,
                    dimension,
                    kernel=7 if stage == 0 else 3,
                    stride=4 if stage == 0 else 2,
                )
            )
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    MixTransformerBlock(
                        dimension, stage_heads, reduction, rates[block_index]
                    )
                )
                block_index += 1
            self.blocks.append(nn.ModuleList(stage_blocks))
            self.stage_normalizations.append(nn.LayerNorm(dimension))
            previous = dimension
        self.decoder_projections = nn.ModuleList(
            [nn.Conv2d(dimension, decoder_dimension, 1) for dimension in dimensions]
        )
        self.decoder_fusion = nn.Sequential(
            nn.Conv2d(decoder_dimension * 4, decoder_dimension, 1, bias=False),
            nn.BatchNorm2d(decoder_dimension),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.heads = OutputHeads(decoder_dimension, boundary_supervision)

    def forward(self, x: torch.Tensor) -> dict:
        output_size = x.shape[-2:]
        features = []
        value = x
        for patch, blocks, normalization in zip(
            self.patch_embeddings, self.blocks, self.stage_normalizations
        ):
            sequence, height, width = patch(value)
            for block in blocks:
                sequence = block(sequence, height, width)
            sequence = normalization(sequence)
            value = sequence.transpose(1, 2).reshape(
                sequence.shape[0], sequence.shape[2], height, width
            )
            features.append(value)
        target_size = features[0].shape[-2:]
        decoded = []
        for feature, projection in zip(features, self.decoder_projections):
            projected = projection(feature)
            if projected.shape[-2:] != target_size:
                projected = F.interpolate(
                    projected, size=target_size, mode="bilinear", align_corners=False
                )
            decoded.append(projected)
        decoder_features = self.decoder_fusion(torch.cat(decoded, dim=1))
        return self.heads(decoder_features, output_size)


class TorchvisionSegmentationWrapper(nn.Module):
    """Adapter around an unmodified torchvision semantic segmentation model."""

    def __init__(self, architecture: str, boundary_supervision: bool = False) -> None:
        super().__init__()
        if architecture == "deeplabv3_mobilenet_v3_large":
            self.network = deeplabv3_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=2,
                aux_loss=False,
            )
        elif architecture == "lraspp_mobilenet_v3_large":
            self.network = lraspp_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=2,
            )
        elif architecture == "deeplabv3_resnet50":
            self.network = deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=2,
                aux_loss=False,
            )
        else:
            raise ValueError(architecture)
        self.boundary = nn.Conv2d(2, 1, 3, padding=1) if boundary_supervision else None

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.network(x)["out"]
        probability = logits.softmax(dim=1)
        return {
            "logits": logits,
            "prob": probability[:, 1:2],
            "background_prob": probability[:, 0:1],
            "boundary_logits": self.boundary(logits) if self.boundary is not None else None,
            "decoder_features": logits,
            "aux_logits": [],
            "prototypes": [],
        }


def build_baseline(cfg: dict) -> nn.Module:
    model_cfg = cfg["model"]
    name = str(model_cfg["name"]).lower()
    in_channels = int(model_cfg.get("in_channels", 3))
    base = int(model_cfg.get("base_channels", 32))
    boundary_supervision = float(cfg.get("loss", {}).get("boundary_weight", 0.0)) > 0.0
    if name == "unet":
        return UNet(in_channels, base, boundary_supervision)
    if name == "eilnet":
        from .eilnet import EILNet

        return EILNet(in_channels, base, boundary_supervision)
    if name in {"unetpp", "unet++"}:
        return UNetPlusPlus(in_channels, base, boundary_supervision)
    if name in {"unetr", "unetr_like"}:
        return UNETRLike(
            in_channels,
            base,
            embed=int(model_cfg.get("embed_dim", 192)),
            depth=int(model_cfg.get("transformer_depth", 4)),
            boundary_supervision=boundary_supervision,
        )
    if name in {"torchvision_deeplabv3_mobilenet", "deeplabv3_mobilenet_v3_large"}:
        return TorchvisionSegmentationWrapper(
            "deeplabv3_mobilenet_v3_large", boundary_supervision
        )
    if name in {"torchvision_lraspp_mobilenet", "lraspp_mobilenet_v3_large"}:
        return TorchvisionSegmentationWrapper(
            "lraspp_mobilenet_v3_large", boundary_supervision
        )
    if name in {"torchvision_deeplabv3_resnet50", "deeplabv3_resnet50"}:
        return TorchvisionSegmentationWrapper("deeplabv3_resnet50", boundary_supervision)
    if name in {"segformer", "segformer_b0"}:
        return SegFormerB0(
            in_channels,
            decoder_dimension=int(model_cfg.get("decoder_dim", 256)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.1)),
            boundary_supervision=boundary_supervision,
        )
    raise ValueError(f"Unknown model: {name}")
