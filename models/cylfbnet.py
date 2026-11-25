from __future__ import annotations

"""Deployable segmentation backbone for APGRA-Net.

This module maps a three-channel FMI input ``[I_n, C, R]`` to a binary segmentation:

1. Reflective padding along depth and circular padding along azimuth;
2. A lightweight four-stage encoder and three-stage decoder for multiscale features;
3. ASC aggregates strip context according to acquisition support;
4. GHPI interacts geological-response and background prototypes at each scale;
5. Full-resolution detail fusion produces the geological-response probability.

Proxy pretraining, the EMA teacher, global-local prototype screening, and pseudo-label losses
are implemented in ``pretrain_proxy.py``, ``train_adaptation.py``, and ``src/losses.py``;
they are outside the deployable forward graph. ``CylFBNet`` is the historical class name.
The final ``models.apgranet.APGRANet`` class inherits this implementation.
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    """Select the largest GroupNorm group count that divides the channel count."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class PeriodicConv2d(nn.Module):
    """Two-dimensional convolution with depth reflection and azimuthal wraparound.

    The input layout is ``[B, C, H_z, W_theta]``. Width follows borehole azimuth,
    so the left and right edges are neighbors. Height follows measured depth and does not wrap.
    ``periodic=False`` is used for planar proxy data and nonperiodic ablations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        periodic: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.pad_h = kernel_size[0] // 2
        self.pad_w = kernel_size[1] // 2
        self.periodic = periodic
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Wrap azimuthal indices for FMI; use zero padding for planar proxy images.
        if self.pad_w:
            if self.periodic:
                width = x.shape[-1]
                indices = torch.arange(-self.pad_w, width + self.pad_w, device=x.device) % width
                x = x.index_select(-1, indices)
            else:
                x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="constant")
        if self.pad_h:
            # Reflect padding is invalid for tiny maps; fall back to replicate padding.
            vertical_mode = "reflect" if x.shape[-2] > self.pad_h else "replicate"
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode=vertical_mode)
        return self.conv(x)


def periodic_interpolate(
    x: torch.Tensor,
    size: tuple[int, int] | torch.Size,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Preserve adjacency between the first and last azimuthal columns during interpolation.

    Pad one column on each side before bilinear interpolation, then remove the added
    region to avoid a seam at 0/360 degrees. Nearest-neighbor mode needs no such step.
    """
    target_h, target_w = int(size[0]), int(size[1])
    if mode == "nearest":
        return F.interpolate(x, size=(target_h, target_w), mode=mode)
    scale_w = max(1, int(round(target_w / max(x.shape[-1], 1))))
    padded = torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)
    resized = F.interpolate(
        padded,
        size=(target_h, target_w + 2 * scale_w),
        mode=mode,
        align_corners=False,
    )
    return resized[..., scale_w:-scale_w]


class ConvNormAct(nn.Sequential):
    """Combine periodic convolution, GroupNorm, and optional SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        groups: int = 1,
        periodic: bool = True,
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            PeriodicConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                groups=groups,
                periodic=periodic,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class PeriodicBlurPool2d(nn.Module):
    """Fixed low-pass downsampling by two for the controlled anti-aliasing experiment.

    The final configuration disables ``anti_alias_downsample``. This module remains
    available for reproducing the anti-aliasing ablation.
    """

    def __init__(self, channels: int, periodic: bool = True) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel, persistent=False)
        self.channels = channels
        self.periodic = periodic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.periodic:
            width = x.shape[-1]
            indices = torch.arange(-1, width + 1, device=x.device) % width
            x = x.index_select(-1, indices)
        else:
            x = F.pad(x, (1, 1, 0, 0), mode="constant")
        vertical_mode = "reflect" if x.shape[-2] > 1 else "replicate"
        x = F.pad(x, (0, 0, 1, 1), mode=vertical_mode)
        return F.conv2d(x, self.kernel.to(dtype=x.dtype), stride=2, groups=self.channels)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation channel recalibration."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class InvertedResidual(nn.Module):
    """Lightweight inverted residual block: expansion, depthwise convolution, SE, projection."""

    def __init__(self, in_channels: int, out_channels: int, periodic: bool, expansion: int = 2) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = in_channels == out_channels
        self.block = nn.Sequential(
            ConvNormAct(in_channels, hidden, kernel_size=1, periodic=periodic),
            ConvNormAct(hidden, hidden, kernel_size=3, groups=hidden, periodic=periodic),
            SqueezeExcite(hidden),
            ConvNormAct(hidden, out_channels, kernel_size=1, periodic=periodic, activation=False),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.block(x)
        if self.use_residual:
            value = value + x
        return self.activation(value)


class EncoderStage(nn.Module):
    """Downsample and apply inverted residual blocks at one scale."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        periodic: bool,
        downsample: bool,
        anti_alias: bool = False,
    ) -> None:
        super().__init__()
        if downsample and anti_alias:
            self.down = nn.Sequential(
                ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1, periodic=periodic),
                PeriodicBlurPool2d(out_channels, periodic=periodic),
            )
        elif downsample:
            self.down = ConvNormAct(
                in_channels, out_channels, kernel_size=3, stride=2, periodic=periodic
            )
        else:
            self.down = ConvNormAct(
                in_channels, out_channels, kernel_size=1, periodic=periodic
            )
        self.blocks = nn.Sequential(
            *[InvertedResidual(out_channels, out_channels, periodic=periodic) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class CylindricalStripContext(nn.Module):
    """ASC: low-resolution strip context controlled by image-derived acquisition support."""

    def __init__(self, channels: int, periodic: bool, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        # Two azimuthal strip branches capture lateral traces; one depth branch adds vertical context.
        self.horizontal_7 = ConvNormAct(
            channels, channels, kernel_size=(1, 7), groups=channels, periodic=periodic
        )
        self.horizontal_15 = ConvNormAct(
            channels, channels, kernel_size=(1, 15), groups=channels, periodic=periodic
        )
        self.vertical_7 = ConvNormAct(
            channels, channels, kernel_size=(7, 1), groups=channels, periodic=periodic
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.fill_gate = nn.Sequential(nn.Conv2d(channels + 1, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        # Max-pool the invalid flag: any low-support pixel in a source window
        # marks the corresponding low-resolution location as unreliable.
        invalid = 1.0 - reliability
        invalid = F.adaptive_max_pool2d(invalid, output_size=x.shape[-2:])
        reliability = 1.0 - invalid
        reliable_features = x * reliability
        mixed = self.fuse(
            torch.cat(
                [
                    self.horizontal_7(reliable_features),
                    self.horizontal_15(reliable_features),
                    self.vertical_7(reliable_features),
                ],
                dim=1,
            )
        )
        gate = self.fill_gate(torch.cat([mixed, reliability], dim=1))
        # Reliable locations retain measured features with a small context addition;
        # low-support locations receive gated context rather than measured-response features.
        reliable_output = x + 0.25 * mixed
        reconstructed_output = gate * mixed
        return reliability * reliable_output + (1.0 - reliability) * reconstructed_output


class ForegroundBackgroundInteraction(nn.Module):
    """GHPI: image-derived geological-response/background prototype interaction.

    The final model uses ``dual`` prototype competition. ``foreground`` and ``none``
    select single-foreground and no-prototype ablations. These are image-level
    statistics inside the deployed model, not the depth-local pseudo-label screening prototypes.
    """

    def __init__(
        self,
        channels: int,
        mode: str = "dual",
        periodic: bool = True,
        has_prior: bool = True,
        foreground_prior: float = 0.05,
        signed_gate: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"dual", "foreground", "none"}:
            raise ValueError(f"Unsupported prototype mode: {mode}")
        self.mode = mode
        self.enabled = mode != "none"
        self.periodic = periodic
        self.signed_gate = signed_gate
        self.score = nn.Conv2d(channels, 2, 1)
        if not 0.0 < foreground_prior < 1.0:
            raise ValueError("foreground_prior must be in (0, 1)")
        # Initialize class biases with a 5% sparse-foreground prior; training sets the learned balance.
        nn.init.constant_(self.score.bias[0], 0.0)
        nn.init.constant_(self.score.bias[1], math.log(foreground_prior / (1.0 - foreground_prior)))
        if self.enabled:
            self.gate = nn.Sequential(
                nn.Conv2d(channels + 2, channels, 1, bias=False),
                nn.GroupNorm(_groups(channels), channels),
                nn.Sigmoid(),
            )
            self.refine = InvertedResidual(channels, channels, periodic=periodic)
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
            self.prototype_alpha = nn.Parameter(torch.tensor(-2.0))
            if signed_gate:
                # Enabled only in the signed-gate refinement experiment.
                self.gate_amplitude = nn.Parameter(torch.tensor(math.atanh(0.25)))
        self.prior_alpha = nn.Parameter(torch.tensor(0.0)) if has_prior else None

    def forward(
        self,
        features: torch.Tensor,
        prior_logits: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return refined features, class scores, and two image-level prototypes."""
        raw_logits = self.score(features)
        if prior_logits is not None:
            if self.prior_alpha is None:
                raise RuntimeError("prior_logits were passed to an interaction without prior fusion")
            resized_prior = (
                periodic_interpolate(prior_logits, size=features.shape[-2:], mode="bilinear")
                if self.periodic
                else F.interpolate(
                    prior_logits, size=features.shape[-2:], mode="bilinear", align_corners=False
                )
            )
            raw_logits = raw_logits + torch.tanh(self.prior_alpha) * resized_prior
        probabilities = raw_logits.softmax(dim=1)
        if reliability is None:
            reliability = torch.ones_like(probabilities[:, :1])
        else:
            invalid = F.adaptive_max_pool2d(1.0 - reliability, output_size=features.shape[-2:])
            reliability = 1.0 - invalid
        # Compute two per-image prototypes from soft classes and valid acquisition support.
        weights = probabilities * reliability
        denominator = weights.flatten(2).sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        prototypes = torch.einsum("bkhw,bchw->bkc", weights, features) / denominator
        if not self.enabled:
            return features, raw_logits, prototypes

        normal_features = F.normalize(features, dim=1, eps=1.0e-6)
        normal_prototypes = F.normalize(prototypes, dim=-1, eps=1.0e-6)
        similarities = torch.einsum("bchw,bkc->bkhw", normal_features, normal_prototypes)
        # Dual mode uses response similarity minus background similarity as competing evidence.
        evidence = (
            similarities[:, 1:2] - similarities[:, 0:1]
            if self.mode == "dual"
            else similarities[:, 1:2]
        )
        # Binary uncertainty peaks at 0.5 and approaches zero near 0 or 1.
        uncertainty = 4.0 * probabilities[:, 0:1] * probabilities[:, 1:2]
        gate = self.gate(torch.cat([features, evidence, uncertainty], dim=1))
        if self.signed_gate:
            amplitude = torch.tanh(self.gate_amplitude)
            modulated = features * (1.0 + amplitude * (2.0 * gate - 1.0))
        else:
            modulated = features * (1.0 + gate)
        refined = self.refine(modulated)
        scale = self.logit_scale.exp().clamp(max=20.0) * torch.sigmoid(self.prototype_alpha)
        if self.mode == "dual":
            prototype_logits = similarities
        else:
            prototype_logits = torch.cat(
                [torch.zeros_like(similarities[:, :1]), similarities[:, 1:2]], dim=1
            )
        logits = raw_logits + scale * prototype_logits
        return refined, logits, prototypes


class DecoderStage(nn.Module):
    """Periodic upsampling, same-scale skip connection, and lightweight feature fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, periodic: bool) -> None:
        super().__init__()
        self.periodic = periodic
        self.fuse = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels, kernel_size=1, periodic=periodic),
            InvertedResidual(out_channels, out_channels, periodic=periodic),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = (
            periodic_interpolate(x, size=skip.shape[-2:], mode="bilinear")
            if self.periodic
            else F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        )
        return self.fuse(torch.cat([x, skip], dim=1))


class CylFBNet(nn.Module):
    """Four-scale periodic encoder-decoder segmentation network for APGRA-Net.

    The final configuration uses three input channels, two classes, encoder widths
    ``24/48/96/160``, and block counts ``1/2/2/2``. Input shape is ``[B, 3, H, W]``;
    logit shape is ``[B, 2, H, W]``. Class 1 denotes the target geological response.
    """

    def __init__(self, cfg: dict, boundary_supervision: bool = False) -> None:
        super().__init__()
        widths: Sequence[int] = tuple(int(value) for value in cfg.get("widths", [24, 48, 96, 160]))
        depths: Sequence[int] = tuple(int(value) for value in cfg.get("blocks", [1, 2, 2, 2]))
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("CylFBNet requires exactly four encoder widths and depths")
        periodic = bool(cfg.get("periodic", True))
        self.periodic = periodic
        # These two switches support refinement experiments; both are disabled in the final model.
        anti_alias = bool(cfg.get("anti_alias_downsample", False))
        signed_gate = bool(cfg.get("signed_interaction_gate", False))
        prototype_mode = str(cfg.get("prototype_mode", "dual"))
        if not bool(cfg.get("fg_bg_interaction", True)):
            prototype_mode = "none"
        foreground_prior = float(cfg.get("foreground_prior", 0.05))

        in_channels = int(cfg.get("in_channels", 3))
        num_classes = int(cfg.get("num_classes", 2))
        if in_channels != 3 or num_classes != 2:
            raise ValueError("CylFBNet currently requires in_channels=3 and num_classes=2")
        detail_channels = max(12, widths[0] // 2)
        # The full-resolution detail branch preserves thin trace boundaries for final fusion.
        self.detail_stem = ConvNormAct(in_channels, detail_channels, 3, stride=1, periodic=periodic)
        if anti_alias:
            self.stem = nn.Sequential(
                ConvNormAct(detail_channels, widths[0], 3, stride=1, periodic=periodic),
                PeriodicBlurPool2d(widths[0], periodic=periodic),
            )
        else:
            self.stem = ConvNormAct(detail_channels, widths[0], 3, stride=2, periodic=periodic)
        # The stem halves resolution; encoder stages output 1/2, 1/4, 1/8, and 1/16 features.
        self.stages = nn.ModuleList(
            [
                EncoderStage(
                    widths[0], widths[0], depths[0], periodic, downsample=False,
                    anti_alias=anti_alias,
                ),
                EncoderStage(
                    widths[0], widths[1], depths[1], periodic, downsample=True,
                    anti_alias=anti_alias,
                ),
                EncoderStage(
                    widths[1], widths[2], depths[2], periodic, downsample=True,
                    anti_alias=anti_alias,
                ),
                EncoderStage(
                    widths[2], widths[3], depths[3], periodic, downsample=True,
                    anti_alias=anti_alias,
                ),
            ]
        )
        self.context = CylindricalStripContext(
            widths[3],
            periodic=periodic,
            enabled=bool(cfg.get("strip_context", True)),
        )
        self.coarse_interaction = ForegroundBackgroundInteraction(
            widths[3],
            mode=prototype_mode,
            periodic=periodic,
            has_prior=False,
            foreground_prior=foreground_prior,
            signed_gate=signed_gate,
        )
        # The decoder restores resolution from 1/16 through 1/8 and 1/4 to 1/2.
        self.decoder_stages = nn.ModuleList(
            [
                DecoderStage(widths[3], widths[2], widths[2], periodic),
                DecoderStage(widths[2], widths[1], widths[1], periodic),
                DecoderStage(widths[1], widths[0], widths[0], periodic),
            ]
        )
        # A GHPI block is already at the bottleneck; these three serve the decoder scales.
        self.interactions = nn.ModuleList(
            [
                ForegroundBackgroundInteraction(
                    widths[2], mode=prototype_mode, periodic=periodic,
                    foreground_prior=foreground_prior,
                    signed_gate=signed_gate,
                ),
                ForegroundBackgroundInteraction(
                    widths[1], mode=prototype_mode, periodic=periodic,
                    foreground_prior=foreground_prior,
                    signed_gate=signed_gate,
                ),
                ForegroundBackgroundInteraction(
                    widths[0], mode=prototype_mode, periodic=periodic,
                    foreground_prior=foreground_prior,
                    signed_gate=signed_gate,
                ),
            ]
        )
        self.full_resolution_refine = nn.Sequential(
            ConvNormAct(widths[0] + detail_channels, widths[0], 1, periodic=periodic),
            InvertedResidual(widths[0], widths[0], periodic=periodic),
        )
        # The boundary head serves a controlled ablation. The final model has boundary_weight=0,
        # so it does not instantiate this auxiliary branch or add deployment parameters.
        self.boundary_head = (
            nn.Sequential(
                ConvNormAct(widths[0], widths[0], 3, periodic=periodic),
                nn.Conv2d(widths[0], 1, 1),
            )
            if boundary_supervision
            else None
        )
        if bool(cfg.get("adaptive_reliability", False)):
            # Adaptive reliability is experimental; the final configuration uses input support R.
            reliability_floor = float(cfg.get("reliability_floor_init", 0.50))
            if not 0.0 < reliability_floor < 1.0:
                raise ValueError("reliability_floor_init must be in (0, 1)")
            self.reliability_floor_logit = nn.Parameter(
                torch.tensor(math.log(reliability_floor / (1.0 - reliability_floor)))
            )
        else:
            self.reliability_floor_logit = None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the full deployable forward pass."""
        output_size = x.shape[-2:]
        # The third input channel is image-derived acquisition support R, clamped to [0, 1].
        reliability = x[:, 2:3].clamp(0.0, 1.0)
        if self.reliability_floor_logit is not None:
            floor = torch.sigmoid(self.reliability_floor_logit)
            reliability = floor + (1.0 - floor) * reliability
        # 1) Full-resolution detail; 2) four-scale encoding; 3) bottleneck ASC and GHPI.
        detail_features = self.detail_stem(x)
        features = self.stem(detail_features)
        skips = []
        for stage in self.stages:
            features = stage(features)
            skips.append(features)
        features = self.context(features, reliability)
        features, logits, prototypes = self.coarse_interaction(features, reliability=reliability)
        auxiliary_logits = [logits]
        all_prototypes = [prototypes]
        # Fuse encoder skips after each upsampling step, then refine with scale-specific GHPI.
        for stage, interaction, skip in zip(
            self.decoder_stages,
            self.interactions,
            reversed(skips[:-1]),
        ):
            features = stage(features, skip)
            features, logits, prototypes = interaction(
                features,
                prior_logits=logits,
                reliability=reliability,
            )
            auxiliary_logits.append(logits)
            all_prototypes.append(prototypes)
        features = (
            periodic_interpolate(features, size=output_size, mode="bilinear")
            if self.periodic
            else F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)
        )
        # Restore original resolution and fuse details to retain narrow responses.
        features = self.full_resolution_refine(torch.cat([features, detail_features], dim=1))
        local_logits = self.interactions[-1].score(features)
        resized_logits = (
            periodic_interpolate(logits, size=output_size, mode="bilinear")
            if self.periodic
            else F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        )
        # Final classification combines full-resolution local evidence and coarse periodic scores.
        full_logits = local_logits + torch.tanh(self.interactions[-1].prior_alpha) * resized_logits
        boundary_logits = self.boundary_head(features) if self.boundary_head is not None else None
        probabilities = full_logits.softmax(dim=1)
        # Shared output interface for supervised loss, EMA adaptation, evaluation, and visualization.
        return {
            "logits": full_logits,
            "prob": probabilities[:, 1:2],
            "background_prob": probabilities[:, 0:1],
            "boundary_logits": boundary_logits,
            "decoder_features": features,
            "aux_logits": auxiliary_logits,
            "prototypes": all_prototypes,
        }
