from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    periodic = torch.cat([image[..., -1:], image, image[..., :1]], dim=-1)
    horizontal = -F.max_pool2d(-periodic, kernel_size=(1, 3), stride=1, padding=0)
    return torch.minimum(vertical, horizontal)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    periodic = torch.cat([image[..., -1:], image, image[..., :1]], dim=-1)
    return F.max_pool2d(periodic, kernel_size=3, stride=1, padding=(1, 0))


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int) -> torch.Tensor:
    """Periodic adaptation of soft morphology used by clDice.

    Method credit: Shit et al., CVPR 2021, doi:10.1109/CVPR46437.2021.01629.
    This implementation changes horizontal morphology to wrap azimuthally.
    """
    image = image.float()
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = _soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 5,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    probability = probability.float()
    target = target.float()
    pred_skeleton = soft_skeleton(probability, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    topology_precision = (pred_skeleton * target).sum(dim=(1, 2, 3)) / (
        pred_skeleton.sum(dim=(1, 2, 3)) + eps
    )
    topology_sensitivity = (target_skeleton * probability).sum(dim=(1, 2, 3)) / (
        target_skeleton.sum(dim=(1, 2, 3)) + eps
    )
    cldice = 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity + eps
    )
    return (1.0 - cldice).mean()


def boundary_target(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    periodic_mask = torch.cat([mask[..., -1:], mask, mask[..., :1]], dim=-1)
    dilated = F.max_pool2d(periodic_mask, kernel_size=3, stride=1, padding=(1, 0))
    inverse = 1.0 - mask
    periodic_inverse = torch.cat([inverse[..., -1:], inverse, inverse[..., :1]], dim=-1)
    eroded = 1.0 - F.max_pool2d(periodic_inverse, kernel_size=3, stride=1, padding=(1, 0))
    return (dilated - eroded).clamp(0.0, 1.0)


def prototype_contrast_loss(
    features: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    mode: str = "dual",
) -> torch.Tensor:
    """Project-specific supervised two-prototype separation objective.

    Prototype learning and foreground/background contrast are established
    concepts; see the provenance document for the academic claim boundary.
    """
    labels = F.interpolate(target.float(), size=features.shape[-2:], mode="nearest").squeeze(1).long()
    normal_features = F.normalize(features.float(), dim=1, eps=1.0e-6)
    one_hot = F.one_hot(labels, num_classes=2).permute(0, 3, 1, 2).float()
    denominator = one_hot.flatten(2).sum(dim=-1, keepdim=True).clamp_min(1.0)
    prototypes = torch.einsum("bkhw,bchw->bkc", one_hot, features.float()) / denominator
    prototypes = F.normalize(prototypes, dim=-1, eps=1.0e-6)
    similarities = torch.einsum("bchw,bkc->bkhw", normal_features, prototypes)
    if mode == "foreground":
        foreground_similarity = similarities[:, 1]
        foreground = labels.float()
        positive = ((1.0 - foreground_similarity) * foreground).sum() / foreground.sum().clamp_min(1.0)
        hard_background = (F.relu(foreground_similarity - 0.2) * (1.0 - foreground)).sum()
        hard_background = hard_background / (1.0 - foreground).sum().clamp_min(1.0)
        return positive + 0.25 * hard_background
    if mode == "none":
        return features.new_zeros(())
    class_weights = features.new_tensor([1.0, 1.5])
    return F.cross_entropy(similarities / temperature, labels, weight=class_weights)


class SegmentationLoss(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg["loss"]
        foreground_weight = float(self.cfg.get("foreground_ce_weight", 1.0))
        self.register_buffer("ce_weights", torch.tensor([1.0, foreground_weight], dtype=torch.float32))
        model_cfg = cfg.get("model", {})
        self.contrast_mode = str(model_cfg.get("prototype_mode", "dual"))

    def forward(self, outputs: dict, target: torch.Tensor) -> dict[str, torch.Tensor]:
        target = target.long()
        target_1c = target[:, None].float()
        logits = outputs["logits"].float()
        probability = logits.softmax(dim=1)[:, 1:2]
        ce = F.cross_entropy(logits, target, weight=self.ce_weights)
        dice = soft_dice_loss(probability, target_1c)

        boundary_weight = float(self.cfg.get("boundary_weight", 0.0))
        if boundary_weight > 0:
            if outputs.get("boundary_logits") is None:
                raise RuntimeError("boundary_weight > 0 requires a boundary prediction head")
            boundary = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"].float(), boundary_target(target_1c)
            )
        else:
            boundary = logits.new_zeros(())

        cldice = logits.new_zeros(())
        if float(self.cfg.get("cldice_weight", 0.0)) > 0:
            cldice = soft_cldice_loss(
                probability,
                target_1c,
                iterations=int(self.cfg.get("cldice_iterations", 5)),
            )

        contrast = logits.new_zeros(())
        if float(self.cfg.get("contrast_weight", 0.0)) > 0:
            contrast = prototype_contrast_loss(
                outputs["decoder_features"],
                target_1c,
                temperature=float(self.cfg.get("contrast_temperature", 0.15)),
                mode=self.contrast_mode,
            )

        auxiliary = logits.new_zeros(())
        auxiliary_logits = outputs.get("aux_logits", [])
        if auxiliary_logits and float(self.cfg.get("aux_weight", 0.0)) > 0:
            for aux_logits in auxiliary_logits:
                aux_target = F.interpolate(target_1c, size=aux_logits.shape[-2:], mode="nearest").squeeze(1).long()
                aux_probability = aux_logits.float().softmax(dim=1)[:, 1:2]
                auxiliary = auxiliary + F.cross_entropy(aux_logits.float(), aux_target, weight=self.ce_weights)
                auxiliary = auxiliary + soft_dice_loss(aux_probability, aux_target[:, None].float())
            auxiliary = auxiliary / len(auxiliary_logits)

        total = (
            float(self.cfg.get("ce_weight", 1.0)) * ce
            + float(self.cfg.get("dice_weight", 1.0)) * dice
            + boundary_weight * boundary
            + float(self.cfg.get("cldice_weight", 0.0)) * cldice
            + float(self.cfg.get("contrast_weight", 0.0)) * contrast
            + float(self.cfg.get("aux_weight", 0.0)) * auxiliary
        )
        return {
            "loss": total,
            "ce": ce.detach(),
            "dice_loss": dice.detach(),
            "boundary_loss": boundary.detach(),
            "cldice_loss": cldice.detach(),
            "contrast_loss": contrast.detach(),
            "aux_loss": auxiliary.detach(),
        }


def build_loss(cfg: dict) -> SegmentationLoss:
    return SegmentationLoss(cfg)
