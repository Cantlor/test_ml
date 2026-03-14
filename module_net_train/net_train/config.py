from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be mapping: {path}")
    return obj


def _resolve_paths(module_root: Path, paths_raw: Dict[str, Any]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for key, value in (paths_raw or {}).items():
        paths[key] = _resolve(module_root, str(value))
    return paths


def _module_and_project_root() -> tuple[Path, Path]:
    # Always resolve relative paths against module_net_train root,
    # independent from where the config file itself is located.
    module_root = Path(__file__).resolve().parents[1]
    project_root = module_root.parent
    return module_root, project_root


@dataclass
class TrainConfig:
    config_path: Path
    module_root: Path
    project_root: Path
    raw: Dict[str, Any]
    paths: Dict[str, Path]


@dataclass
class HardwareConfig:
    config_path: Path
    module_root: Path
    project_root: Path
    raw: Dict[str, Any]


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path).resolve()
    module_root, project_root = _module_and_project_root()

    raw = _load_yaml(config_path)
    paths = _resolve_paths(module_root, raw.get("paths", {}) or {})

    return TrainConfig(
        config_path=config_path,
        module_root=module_root,
        project_root=project_root,
        raw=raw,
        paths=paths,
    )


def load_hardware_config(path: str | Path) -> HardwareConfig:
    config_path = Path(path).resolve()
    module_root, project_root = _module_and_project_root()
    raw = _load_yaml(config_path)

    return HardwareConfig(
        config_path=config_path,
        module_root=module_root,
        project_root=project_root,
        raw=raw,
    )


def get_nested(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path_from_train_cfg(train_cfg: TrainConfig, path_value: str | Path) -> Path:
    return _resolve(train_cfg.module_root, path_value)


def resolve_inference_manifest_path(train_cfg: TrainConfig) -> Path:
    infer = (train_cfg.raw.get("inference", {}) or {})
    aoi = (infer.get("aoi", {}) or {})

    manifest_cfg = aoi.get("aoi_manifest")
    if manifest_cfg is not None:
        return resolve_path_from_train_cfg(train_cfg, str(manifest_cfg))

    if "aoi_manifest" in train_cfg.paths:
        return train_cfg.paths["aoi_manifest"]

    # Preferred manifest in refactored module_prep_data.
    strict_path = resolve_path_from_train_cfg(train_cfg, "../output_data/module_prep_data_work/aoi_manifest.json")
    if strict_path.exists():
        return strict_path

    # Backward-compatible fallback.
    return resolve_path_from_train_cfg(train_cfg, "../output_data/module_prep_data_work/aoi_rasters_manifest.json")


def resolve_run_train_config_path(config_path: str | Path, run_dir: str | Path) -> tuple[Path, bool]:
    """
    Prefer run_dir/config_resolved.yaml for eval/infer parity.
    Returns (path, from_run_dir_flag).
    """
    rd = Path(run_dir).resolve()
    resolved = rd / "config_resolved.yaml"
    if resolved.exists():
        return resolved, True
    return Path(config_path).resolve(), False


def dump_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=False, sort_keys=False)
