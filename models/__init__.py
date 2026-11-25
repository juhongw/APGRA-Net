from .cylfbnet import CylFBNet
from .apgranet import APGRANet


def build_model(cfg):
    name = str(cfg["model"]["name"]).lower()
    if name in {"cylfbnet", "apgranet"}:
        boundary_supervision = float(cfg.get("loss", {}).get("boundary_weight", 0.0)) > 0.0
        model_class = APGRANet if name == "apgranet" else CylFBNet
        return model_class(cfg["model"], boundary_supervision=boundary_supervision)
    from .baselines import build_baseline

    return build_baseline(cfg)


def load_model_state(model, state_dict, strict: bool = True):
    """Load historical checkpoints after removing inactive boundary heads.

    Final segmentation logits never depended on these branches.  Only
    unexpected keys belonging to a disabled boundary projection are accepted;
    every task-path parameter remains strictly checked.
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_prefixes = ("boundary_head.", "heads.boundary.", "boundary.")
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith(allowed_prefixes)
    ]
    missing = list(incompatible.missing_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return incompatible


__all__ = ["APGRANet", "CylFBNet", "build_model", "load_model_state"]
