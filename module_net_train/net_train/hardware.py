from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import multiprocessing

from net_train.config import HardwareConfig, TrainConfig, get_nested

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class RuntimePlan:
    device: str
    precision: str
    amp_enabled: bool
    amp_dtype: str

    crop_size: int
    batch_size: int
    grad_accum_steps: int

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int

    channels_last: bool
    cudnn_benchmark: bool
    allow_tf32: bool
    torch_compile: bool

    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]

    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for module_net_train runtime planning")


def _pick_device(hw_cfg: HardwareConfig) -> str:
    _require_torch()

    mode = str(get_nested(hw_cfg.raw, ["device", "mode"], "auto")).lower()
    prefer_cuda = bool(get_nested(hw_cfg.raw, ["device", "prefer_cuda"], True))

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("hardware_config requests CUDA, but CUDA is not available")
        return "cuda"

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_precision(hw_cfg: HardwareConfig, device: str) -> tuple[str, bool, str]:
    _require_torch()

    mode = str(get_nested(hw_cfg.raw, ["precision", "mode"], "auto")).lower()
    prefer_bf16 = bool(get_nested(hw_cfg.raw, ["precision", "prefer_bf16"], True))

    if device == "cpu":
        return "fp32", False, "float32"

    # CUDA modes
    if mode == "fp32":
        return "fp32", False, "float32"
    if mode == "fp16":
        return "fp16", True, "float16"
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return "bf16", True, "bfloat16"
        return "fp16", True, "float16"

    # auto mode
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return "bf16", True, "bfloat16"
    return "fp16", True, "float16"


def _gpu_vram_gb(device: str) -> Optional[float]:
    _require_torch()
    if device != "cuda":
        return None
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / (1024.0 ** 3)


def _gpu_name(device: str) -> Optional[str]:
    _require_torch()
    if device != "cuda":
        return None
    return str(torch.cuda.get_device_name(0))


def _choose_batch_and_crop(train_cfg: TrainConfig, hw_cfg: HardwareConfig, vram_gb: Optional[float]) -> tuple[int, int, int]:
    sampling = train_cfg.raw.get("sampling", {}) or {}
    train = train_cfg.raw.get("train", {}) or {}
    batch = train.get("batch", {}) or {}

    autotune_enabled = bool(batch.get("autotune", True)) and bool(get_nested(hw_cfg.raw, ["autotune", "enabled"], True))
    if not autotune_enabled:
        crop_size = int(sampling.get("crop_size", 256))
        batch_size = int(batch.get("batch_size", 4))
        grad_accum = int(batch.get("grad_accum_steps", 1))
        return crop_size, batch_size, grad_accum

    crop_candidates = sorted(
        [int(v) for v in get_nested(hw_cfg.raw, ["autotune", "crop_sizes"], [512, 384, 256])],
        reverse=True,
    )
    batch_candidates = sorted(
        [int(v) for v in get_nested(hw_cfg.raw, ["autotune", "batch_sizes"], [8, 4, 2, 1])],
        reverse=True,
    )
    grad_accum_candidates = sorted(
        [int(v) for v in get_nested(hw_cfg.raw, ["autotune", "grad_accum_steps"], [1, 2, 4, 8])]
    )

    # Heuristic table without runtime probing.
    # This keeps deterministic behavior and matches requirement "autotune" baseline.
    if vram_gb is None:
        return min(crop_candidates), 1, 1

    if vram_gb >= 18:
        wanted = (512, 8, 1)
    elif vram_gb >= 10:
        wanted = (512, 4, 1)
    elif vram_gb >= 6:
        wanted = (384, 4, 1)
    elif vram_gb >= 4:
        wanted = (256, 2, 2)
    else:
        wanted = (256, 1, 4)

    crop_size = next((c for c in crop_candidates if c <= wanted[0]), crop_candidates[-1])
    batch_size = next((b for b in batch_candidates if b <= wanted[1]), batch_candidates[-1])
    grad_accum = next((g for g in grad_accum_candidates if g >= wanted[2]), grad_accum_candidates[-1])

    return int(crop_size), int(batch_size), int(grad_accum)


def _choose_num_workers(hw_cfg: HardwareConfig) -> int:
    dl_cfg = hw_cfg.raw.get("dataloader", {}) or {}
    n = dl_cfg.get("num_workers", "auto")
    max_workers = int(dl_cfg.get("max_num_workers", 8))

    if n != "auto":
        return max(0, int(n))

    cpu = multiprocessing.cpu_count()
    # leave some room for OS and rasterio IO threads
    auto = max(0, min(max_workers, cpu - 2))
    return int(auto)


def apply_torch_runtime_flags(plan: RuntimePlan) -> None:
    _require_torch()

    if torch.backends is not None:
        try:
            torch.backends.cudnn.benchmark = bool(plan.cudnn_benchmark)
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(plan.allow_tf32)
        except Exception:
            pass


def build_runtime_plan(train_cfg: TrainConfig, hw_cfg: HardwareConfig) -> RuntimePlan:
    _require_torch()

    warnings: List[str] = []

    device = _pick_device(hw_cfg)
    precision, amp_enabled, amp_dtype = _pick_precision(hw_cfg, device)
    gpu_vram = _gpu_vram_gb(device)
    crop_size, batch_size, grad_accum = _choose_batch_and_crop(train_cfg, hw_cfg, gpu_vram)

    if device == "cpu" and bool(get_nested(hw_cfg.raw, ["runtime", "warn_if_cpu"], True)):
        warnings.append("Training will run on CPU. This is expected only for debug runs.")

    dl_cfg = hw_cfg.raw.get("dataloader", {}) or {}

    plan = RuntimePlan(
        device=device,
        precision=precision,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        crop_size=crop_size,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        num_workers=_choose_num_workers(hw_cfg),
        pin_memory=bool(dl_cfg.get("pin_memory", True)) and device == "cuda",
        persistent_workers=bool(dl_cfg.get("persistent_workers", True)),
        prefetch_factor=int(dl_cfg.get("prefetch_factor", 2)),
        channels_last=bool(get_nested(hw_cfg.raw, ["gpu", "channels_last"], True)) and device == "cuda",
        cudnn_benchmark=bool(get_nested(hw_cfg.raw, ["gpu", "cudnn_benchmark"], True)),
        allow_tf32=bool(get_nested(hw_cfg.raw, ["precision", "allow_tf32"], True)),
        torch_compile=bool(get_nested(hw_cfg.raw, ["gpu", "torch_compile"], False)),
        gpu_name=_gpu_name(device),
        gpu_vram_gb=gpu_vram,
        warnings=warnings,
    )

    return plan


def amp_dtype_from_plan(plan: RuntimePlan):
    _require_torch()
    if plan.amp_dtype == "bfloat16":
        return torch.bfloat16
    if plan.amp_dtype == "float16":
        return torch.float16
    return torch.float32
