from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize


EPS = 1.0e-7


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


def _periodic_morphology(mask: np.ndarray, operation, iterations: int = 1) -> np.ndarray:
    pad = max(1, iterations)
    padded = np.pad(mask.astype(bool), ((0, 0), (pad, pad)), mode="wrap")
    value = operation(padded, structure=np.ones((3, 3), dtype=bool), iterations=iterations)
    return value[:, pad:-pad]


def _periodic_skeleton(mask: np.ndarray) -> np.ndarray:
    width = mask.shape[1]
    tiled = np.tile(mask.astype(bool), (1, 3))
    return skeletonize(tiled)[:, width : 2 * width]


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return _periodic_morphology(mask, binary_dilation) ^ _periodic_morphology(mask, binary_erosion)


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    pred_boundary = _boundary(pred)
    target_boundary = _boundary(target)
    if not pred_boundary.any() and not target_boundary.any():
        return 1.0
    if not pred_boundary.any() or not target_boundary.any():
        return 0.0
    precision = _divide(
        (pred_boundary & _periodic_morphology(target_boundary, binary_dilation, tolerance)).sum(),
        pred_boundary.sum(),
    )
    recall = _divide(
        (target_boundary & _periodic_morphology(pred_boundary, binary_dilation, tolerance)).sum(),
        target_boundary.sum(),
    )
    return _divide(2.0 * precision * recall, precision + recall)


def cldice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 1.0
    pred_skeleton = _periodic_skeleton(pred)
    target_skeleton = _periodic_skeleton(target)
    topology_precision = _divide((pred_skeleton & target).sum(), pred_skeleton.sum())
    topology_sensitivity = _divide((target_skeleton & pred).sum(), target_skeleton.sum())
    return _divide(
        2.0 * topology_precision * topology_sensitivity,
        topology_precision + topology_sensitivity,
    )


def surface_distances(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0, 0.0
    if not pred.any() or not target.any():
        diagonal = float(math.hypot(*pred.shape))
        return diagonal, diagonal
    pred_surface = _boundary(pred)
    target_surface = _boundary(target)
    width = pred.shape[1]
    target_distance = distance_transform_edt(~np.tile(target_surface, (1, 3)))[:, width : 2 * width]
    pred_distance = distance_transform_edt(~np.tile(pred_surface, (1, 3)))[:, width : 2 * width]
    pred_to_target = target_distance[pred_surface]
    target_to_pred = pred_distance[target_surface]
    if pred_to_target.size == 0 or target_to_pred.size == 0:
        return 0.0, 0.0
    hd95 = max(float(np.percentile(pred_to_target, 95)), float(np.percentile(target_to_pred, 95)))
    assd = 0.5 * (float(pred_to_target.mean()) + float(target_to_pred.mean()))
    return hd95, assd


def endpoint_count(mask: np.ndarray) -> int:
    skeleton = _periodic_skeleton(mask.astype(bool)).astype(np.uint8)
    if not skeleton.any():
        return 0
    vertical = np.pad(skeleton, ((1, 1), (0, 0)), mode="constant")
    top = vertical[:-2]
    middle = vertical[1:-1]
    bottom = vertical[2:]
    neighbor_count = (
        np.roll(top, 1, axis=1)
        + top
        + np.roll(top, -1, axis=1)
        + np.roll(middle, 1, axis=1)
        + np.roll(middle, -1, axis=1)
        + np.roll(bottom, 1, axis=1)
        + bottom
        + np.roll(bottom, -1, axis=1)
    )
    return int((skeleton & (neighbor_count == 1)).sum())


def component_count(mask: np.ndarray) -> int:
    labelled, count = label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    parent = list(range(count + 1))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        if left == 0 or right == 0:
            return
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    height = labelled.shape[0]
    for row in range(height):
        for offset in (-1, 0, 1):
            other_row = row + offset
            if 0 <= other_row < height:
                union(int(labelled[row, 0]), int(labelled[other_row, -1]))
    return len({find(value) for value in range(1, count + 1)})


def compute_metrics(pred: np.ndarray, target: np.ndarray, boundary_tolerance: int = 2) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = float((pred & target).sum())
    tn = float((~pred & ~target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    if not pred.any() and not target.any():
        iou = dice = precision = recall = specificity = 1.0
    else:
        iou = _divide(tp, tp + fp + fn)
        dice = _divide(2.0 * tp, 2.0 * tp + fp + fn)
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        specificity = _divide(tn, tn + fp)
    hd95, assd = surface_distances(pred, target)
    pred_components = component_count(pred)
    target_components = component_count(target)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "cldice": cldice_score(pred, target),
        "boundary_f1": boundary_f1(pred, target, boundary_tolerance),
        "hd95": hd95,
        "assd": assd,
        "component_error": float(abs(pred_components - target_components)),
        "endpoint_error": float(abs(endpoint_count(pred) - endpoint_count(target))),
    }


def aggregate_case_metrics(cases: list[dict]) -> dict[str, float]:
    metric_names = [key for key in cases[0] if key not in {"sample_id", "session_id", "group_id"}]
    result = {}
    for name in metric_names:
        values = np.asarray([float(case[name]) for case in cases], dtype=np.float64)
        result[name] = float(values.mean())
        result[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result
