"""Model registry for module_net_train."""

from net_train.models.unet_multitask import UNetMultiTask, build_unet_multitask_from_cfg



def build_model(model_cfg: dict):
    name = str(model_cfg.get("name", "unet_multitask")).lower()
    if name == "unet_multitask":
        return build_unet_multitask_from_cfg(model_cfg)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "UNetMultiTask",
    "build_model",
    "build_unet_multitask_from_cfg",
]
