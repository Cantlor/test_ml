from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io import read_json, read_yaml


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    pred_dir: Path
    extent_prob_path: Path
    boundary_prob_path: Path
    valid_mask_path: Optional[Path]
    footprint_path: Optional[Path]
    predict_manifest_path: Optional[Path]
    config_used_path: Optional[Path]
    valid_nodata_value: Optional[float]
    valid_nodata_rule: str
    valid_control_band_1based: int

    def to_input_context(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "pred_dir": str(self.pred_dir),
            "predict_manifest_path": str(self.predict_manifest_path) if self.predict_manifest_path is not None else None,
            "config_used_path": str(self.config_used_path) if self.config_used_path is not None else None,
            "extent_prob_path": str(self.extent_prob_path),
            "boundary_prob_path": str(self.boundary_prob_path),
            "valid_mask_path": str(self.valid_mask_path) if self.valid_mask_path is not None else None,
            "footprint_path": str(self.footprint_path) if self.footprint_path is not None else None,
            "valid_nodata_value": self.valid_nodata_value,
            "valid_nodata_rule": self.valid_nodata_rule,
            "valid_control_band_1based": self.valid_control_band_1based,
        }


def _resolve_existing_path(value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).resolve()
    if candidate.exists():
        return candidate
    return None


def _load_predict_manifest(path: Path) -> Dict[str, Any]:
    obj = read_json(path.resolve())
    if not isinstance(obj, dict):
        raise TypeError(f"predict manifest must be a JSON object: {path}")
    return obj


def _read_valid_config_from_train_config(path: Optional[Path]) -> Tuple[Optional[float], str, int]:
    if path is None or not path.exists():
        return None, "control-band", 1

    try:
        cfg = read_yaml(path)
    except Exception:
        return None, "control-band", 1

    dataset_cfg = cfg.get("dataset", {}) or {}
    inputs_cfg = dataset_cfg.get("inputs", {}) or {}

    nodata_raw = inputs_cfg.get("nodata_value")
    nodata_value = None if nodata_raw is None else float(nodata_raw)
    nodata_rule = str(inputs_cfg.get("nodata_rule", "control-band"))
    control_band_1based = int(inputs_cfg.get("control_band_1based", 1))
    return nodata_value, nodata_rule, control_band_1based


def resolve_prediction_sample_from_pred_dir(
    pred_dir: Path,
    *,
    sample_id: str,
    extent_name: str = "extent_prob.tif",
    boundary_name: str = "boundary_prob.tif",
    valid_name: str = "valid_mask.tif",
    manifest_name: str = "predict_manifest.json",
) -> PredictionSample:
    pred_dir = pred_dir.resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    manifest_path = pred_dir / manifest_name
    manifest_obj: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest_obj = _load_predict_manifest(manifest_path)

    extent_prob_path = _resolve_existing_path(manifest_obj.get("extent_prob")) or (pred_dir / extent_name)
    boundary_prob_path = _resolve_existing_path(manifest_obj.get("boundary_prob")) or (pred_dir / boundary_name)
    valid_mask_path = _resolve_existing_path(manifest_obj.get("valid_mask"))
    if valid_mask_path is None:
        candidate_valid = pred_dir / valid_name
        if candidate_valid.exists():
            valid_mask_path = candidate_valid.resolve()

    footprint_path = None
    for key in ("aoi_raster", "source_raster", "input_raster"):
        footprint_path = _resolve_existing_path(manifest_obj.get(key))
        if footprint_path is not None:
            break

    config_used_path = _resolve_existing_path(manifest_obj.get("config_used"))
    valid_nodata_value, valid_nodata_rule, valid_control_band_1based = _read_valid_config_from_train_config(config_used_path)

    if not extent_prob_path.exists():
        raise FileNotFoundError(f"extent probability raster not found: {extent_prob_path}")
    if not boundary_prob_path.exists():
        raise FileNotFoundError(f"boundary probability raster not found: {boundary_prob_path}")

    return PredictionSample(
        sample_id=sample_id,
        pred_dir=pred_dir,
        extent_prob_path=extent_prob_path.resolve(),
        boundary_prob_path=boundary_prob_path.resolve(),
        valid_mask_path=valid_mask_path,
        footprint_path=footprint_path,
        predict_manifest_path=(manifest_path.resolve() if manifest_path.exists() else None),
        config_used_path=config_used_path,
        valid_nodata_value=valid_nodata_value,
        valid_nodata_rule=valid_nodata_rule,
        valid_control_band_1based=valid_control_band_1based,
    )


def resolve_prediction_sample_from_manifest(
    manifest_path: Path,
    *,
    sample_id: Optional[str] = None,
    extent_name: str = "extent_prob.tif",
    boundary_name: str = "boundary_prob.tif",
    valid_name: str = "valid_mask.tif",
) -> PredictionSample:
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Predict manifest does not exist: {manifest_path}")
    pred_dir = manifest_path.parent
    resolved_sample_id = sample_id or pred_dir.name
    return resolve_prediction_sample_from_pred_dir(
        pred_dir,
        sample_id=resolved_sample_id,
        extent_name=extent_name,
        boundary_name=boundary_name,
        valid_name=valid_name,
        manifest_name=manifest_path.name,
    )


def discover_prediction_samples(
    pred_root: Path,
    extent_name: str = "extent_prob.tif",
    boundary_name: str = "boundary_prob.tif",
    valid_name: str = "valid_mask.tif",
    manifest_name: str = "predict_manifest.json",
) -> List[PredictionSample]:
    root = pred_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {root}")

    direct_extent = root / extent_name
    extent_paths = [direct_extent] if direct_extent.exists() else sorted(root.rglob(extent_name))

    samples: List[PredictionSample] = []
    for extent_path in extent_paths:
        pred_dir = extent_path.parent
        boundary_path = pred_dir / boundary_name
        if not boundary_path.exists():
            continue

        sample_id = pred_dir.name if pred_dir == root else str(pred_dir.relative_to(root))
        samples.append(
            resolve_prediction_sample_from_pred_dir(
                pred_dir,
                sample_id=sample_id,
                extent_name=extent_name,
                boundary_name=boundary_name,
                valid_name=valid_name,
                manifest_name=manifest_name,
            )
        )

    samples = sorted(samples, key=lambda item: item.sample_id)
    return samples


def resolve_prediction_sample_from_run(
    run_dir: Path,
    *,
    dataset_key: Optional[str] = None,
    extent_name: str = "extent_prob.tif",
    boundary_name: str = "boundary_prob.tif",
    valid_name: str = "valid_mask.tif",
    manifest_name: str = "predict_manifest.json",
) -> PredictionSample:
    pred_root = run_dir.resolve() / "pred"
    if dataset_key:
        pred_dir = pred_root / str(dataset_key)
        return resolve_prediction_sample_from_pred_dir(
            pred_dir,
            sample_id=str(dataset_key),
            extent_name=extent_name,
            boundary_name=boundary_name,
            valid_name=valid_name,
            manifest_name=manifest_name,
        )

    samples = discover_prediction_samples(
        pred_root=pred_root,
        extent_name=extent_name,
        boundary_name=boundary_name,
        valid_name=valid_name,
        manifest_name=manifest_name,
    )
    if not samples:
        raise FileNotFoundError(f"No prediction samples found under {pred_root}")
    if len(samples) > 1:
        raise ValueError(
            f"Run contains multiple prediction samples under {pred_root}; pass --dataset_key explicitly"
        )
    return samples[0]
