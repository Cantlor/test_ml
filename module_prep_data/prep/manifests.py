from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import json


SCHEMA_VERSION = "1.0"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {path}")
    return obj


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise RuntimeError(f"{ctx}: missing required key '{key}'")
    return d[key]


def _opt_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


@dataclass
class CheckInputsDatasetResult:
    dataset: str
    root: str
    raw_raster_path: str
    raw_vector_path: str
    vector_layer: Optional[str]
    vector_id_field: Optional[str]
    qa_raster_path: str
    qa_raster_source: str
    prepared_vector_path: Optional[str]
    prepared_vector_layer: Optional[str]
    qa_ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "CheckInputsDatasetResult":
        ctx = "CheckInputsDatasetResult"
        return cls(
            dataset=str(_require(obj, "dataset", ctx)),
            root=str(_require(obj, "root", ctx)),
            raw_raster_path=str(_require(obj, "raw_raster_path", ctx)),
            raw_vector_path=str(_require(obj, "raw_vector_path", ctx)),
            vector_layer=_opt_str(obj.get("vector_layer")),
            vector_id_field=_opt_str(obj.get("vector_id_field")),
            qa_raster_path=str(_require(obj, "qa_raster_path", ctx)),
            qa_raster_source=str(_require(obj, "qa_raster_source", ctx)),
            prepared_vector_path=_opt_str(obj.get("prepared_vector_path")),
            prepared_vector_layer=_opt_str(obj.get("prepared_vector_layer")),
            qa_ok=bool(_require(obj, "qa_ok", ctx)),
            errors=[str(x) for x in (obj.get("errors", []) or [])],
            warnings=[str(x) for x in (obj.get("warnings", []) or [])],
        )


@dataclass
class CheckInputsManifest:
    schema_version: str
    created_at: str
    config_path: str
    work_dir: str
    data_check_json_path: str
    datasets: List[CheckInputsDatasetResult]
    deferred_config_keys: List[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        config_path: str | Path,
        work_dir: str | Path,
        data_check_json_path: str | Path,
        datasets: List[CheckInputsDatasetResult],
        deferred_config_keys: Optional[List[str]] = None,
    ) -> "CheckInputsManifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=now_utc_iso(),
            config_path=str(Path(config_path).resolve()),
            work_dir=str(Path(work_dir).resolve()),
            data_check_json_path=str(Path(data_check_json_path).resolve()),
            datasets=list(datasets),
            deferred_config_keys=list(deferred_config_keys or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "CheckInputsManifest":
        ctx = "CheckInputsManifest"
        datasets_raw = _require(obj, "datasets", ctx)
        if not isinstance(datasets_raw, list):
            raise RuntimeError(f"{ctx}: 'datasets' must be list")
        return cls(
            schema_version=str(_require(obj, "schema_version", ctx)),
            created_at=str(_require(obj, "created_at", ctx)),
            config_path=str(_require(obj, "config_path", ctx)),
            work_dir=str(_require(obj, "work_dir", ctx)),
            data_check_json_path=str(_require(obj, "data_check_json_path", ctx)),
            datasets=[CheckInputsDatasetResult.from_dict(x) for x in datasets_raw],
            deferred_config_keys=[str(x) for x in (obj.get("deferred_config_keys", []) or [])],
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        _write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CheckInputsManifest":
        p = Path(path).resolve()
        return cls.from_dict(_read_json(p))


@dataclass
class AoiDatasetResult:
    dataset: str
    source_raster_path: str
    vector_path: str
    vector_layer: Optional[str]
    aoi_raster_path: Optional[str]
    mode: str
    buffer_m: float
    mask_outside: bool
    wrote_mask_outside: bool
    status: str
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "AoiDatasetResult":
        ctx = "AoiDatasetResult"
        return cls(
            dataset=str(_require(obj, "dataset", ctx)),
            source_raster_path=str(_require(obj, "source_raster_path", ctx)),
            vector_path=str(_require(obj, "vector_path", ctx)),
            vector_layer=_opt_str(obj.get("vector_layer")),
            aoi_raster_path=_opt_str(obj.get("aoi_raster_path")),
            mode=str(_require(obj, "mode", ctx)),
            buffer_m=float(_require(obj, "buffer_m", ctx)),
            mask_outside=bool(_require(obj, "mask_outside", ctx)),
            wrote_mask_outside=bool(_require(obj, "wrote_mask_outside", ctx)),
            status=str(_require(obj, "status", ctx)),
            message=_opt_str(obj.get("message")),
        )


@dataclass
class AoiManifest:
    schema_version: str
    created_at: str
    config_path: str
    work_dir: str
    out_dir: str
    enabled: bool
    datasets: List[AoiDatasetResult]
    deferred_config_keys: List[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        config_path: str | Path,
        work_dir: str | Path,
        out_dir: str | Path,
        enabled: bool,
        datasets: List[AoiDatasetResult],
        deferred_config_keys: Optional[List[str]] = None,
    ) -> "AoiManifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=now_utc_iso(),
            config_path=str(Path(config_path).resolve()),
            work_dir=str(Path(work_dir).resolve()),
            out_dir=str(Path(out_dir).resolve()),
            enabled=bool(enabled),
            datasets=list(datasets),
            deferred_config_keys=list(deferred_config_keys or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "AoiManifest":
        ctx = "AoiManifest"
        datasets_raw = _require(obj, "datasets", ctx)
        if not isinstance(datasets_raw, list):
            raise RuntimeError(f"{ctx}: 'datasets' must be list")
        return cls(
            schema_version=str(_require(obj, "schema_version", ctx)),
            created_at=str(_require(obj, "created_at", ctx)),
            config_path=str(_require(obj, "config_path", ctx)),
            work_dir=str(_require(obj, "work_dir", ctx)),
            out_dir=str(_require(obj, "out_dir", ctx)),
            enabled=bool(_require(obj, "enabled", ctx)),
            datasets=[AoiDatasetResult.from_dict(x) for x in datasets_raw],
            deferred_config_keys=[str(x) for x in (obj.get("deferred_config_keys", []) or [])],
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        _write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AoiManifest":
        p = Path(path).resolve()
        return cls.from_dict(_read_json(p))


@dataclass
class PatchesDatasetResult:
    dataset: str
    raster_path: str
    raster_source: str
    vector_path: str
    vector_layer: Optional[str]
    vector_id_field: Optional[str]
    dataset_manifest_path: str
    cleaned_vector_raster_crs_path: Optional[str]
    output_dataset_dir: str
    status: str
    message: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PatchesDatasetResult":
        ctx = "PatchesDatasetResult"
        return cls(
            dataset=str(_require(obj, "dataset", ctx)),
            raster_path=str(_require(obj, "raster_path", ctx)),
            raster_source=str(_require(obj, "raster_source", ctx)),
            vector_path=str(_require(obj, "vector_path", ctx)),
            vector_layer=_opt_str(obj.get("vector_layer")),
            vector_id_field=_opt_str(obj.get("vector_id_field")),
            dataset_manifest_path=str(_require(obj, "dataset_manifest_path", ctx)),
            cleaned_vector_raster_crs_path=_opt_str(obj.get("cleaned_vector_raster_crs_path")),
            output_dataset_dir=str(_require(obj, "output_dataset_dir", ctx)),
            status=str(_require(obj, "status", ctx)),
            message=_opt_str(obj.get("message")),
        )


@dataclass
class PatchesManifest:
    schema_version: str
    created_at: str
    config_path: str
    work_dir: str
    patches_all_root: str
    datasets: List[PatchesDatasetResult]
    deferred_config_keys: List[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        config_path: str | Path,
        work_dir: str | Path,
        patches_all_root: str | Path,
        datasets: List[PatchesDatasetResult],
        deferred_config_keys: Optional[List[str]] = None,
    ) -> "PatchesManifest":
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=now_utc_iso(),
            config_path=str(Path(config_path).resolve()),
            work_dir=str(Path(work_dir).resolve()),
            patches_all_root=str(Path(patches_all_root).resolve()),
            datasets=list(datasets),
            deferred_config_keys=list(deferred_config_keys or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PatchesManifest":
        ctx = "PatchesManifest"
        datasets_raw = _require(obj, "datasets", ctx)
        if not isinstance(datasets_raw, list):
            raise RuntimeError(f"{ctx}: 'datasets' must be list")
        return cls(
            schema_version=str(_require(obj, "schema_version", ctx)),
            created_at=str(_require(obj, "created_at", ctx)),
            config_path=str(_require(obj, "config_path", ctx)),
            work_dir=str(_require(obj, "work_dir", ctx)),
            patches_all_root=str(_require(obj, "patches_all_root", ctx)),
            datasets=[PatchesDatasetResult.from_dict(x) for x in datasets_raw],
            deferred_config_keys=[str(x) for x in (obj.get("deferred_config_keys", []) or [])],
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        _write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PatchesManifest":
        p = Path(path).resolve()
        return cls.from_dict(_read_json(p))


@dataclass
class SplitManifest:
    schema_version: str
    created_at: str
    config_path: str
    patches_manifest_path: str
    mode: str
    seed: int
    ratios: Dict[str, float]
    split_roots: Dict[str, str]
    export_folders: Dict[str, str]
    assign_info: Dict[str, Any]
    copied_new_counts: Dict[str, int]
    final_meta_counts: Dict[str, int]
    notes: Dict[str, Any]
    deferred_config_keys: List[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        config_path: str | Path,
        patches_manifest_path: str | Path,
        mode: str,
        seed: int,
        ratios: Dict[str, float],
        split_roots: Dict[str, Path],
        export_folders: Dict[str, str],
        assign_info: Dict[str, Any],
        copied_new_counts: Dict[str, int],
        final_meta_counts: Dict[str, int],
        notes: Dict[str, Any],
        deferred_config_keys: Optional[List[str]] = None,
    ) -> "SplitManifest":
        roots = {k: str(Path(v).resolve()) for k, v in split_roots.items()}
        return cls(
            schema_version=SCHEMA_VERSION,
            created_at=now_utc_iso(),
            config_path=str(Path(config_path).resolve()),
            patches_manifest_path=str(Path(patches_manifest_path).resolve()),
            mode=str(mode),
            seed=int(seed),
            ratios={k: float(v) for k, v in ratios.items()},
            split_roots=roots,
            export_folders={k: str(v) for k, v in export_folders.items()},
            assign_info=dict(assign_info),
            copied_new_counts={k: int(v) for k, v in copied_new_counts.items()},
            final_meta_counts={k: int(v) for k, v in final_meta_counts.items()},
            notes=dict(notes),
            deferred_config_keys=list(deferred_config_keys or []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "SplitManifest":
        ctx = "SplitManifest"
        return cls(
            schema_version=str(_require(obj, "schema_version", ctx)),
            created_at=str(_require(obj, "created_at", ctx)),
            config_path=str(_require(obj, "config_path", ctx)),
            patches_manifest_path=str(_require(obj, "patches_manifest_path", ctx)),
            mode=str(_require(obj, "mode", ctx)),
            seed=int(_require(obj, "seed", ctx)),
            ratios={str(k): float(v) for k, v in (_require(obj, "ratios", ctx) or {}).items()},
            split_roots={str(k): str(v) for k, v in (_require(obj, "split_roots", ctx) or {}).items()},
            export_folders={str(k): str(v) for k, v in (_require(obj, "export_folders", ctx) or {}).items()},
            assign_info=dict(_require(obj, "assign_info", ctx) or {}),
            copied_new_counts={str(k): int(v) for k, v in (_require(obj, "copied_new_counts", ctx) or {}).items()},
            final_meta_counts={str(k): int(v) for k, v in (_require(obj, "final_meta_counts", ctx) or {}).items()},
            notes=dict(_require(obj, "notes", ctx) or {}),
            deferred_config_keys=[str(x) for x in (obj.get("deferred_config_keys", []) or [])],
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path).resolve()
        _write_json(p, self.to_dict())
        return p

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        p = Path(path).resolve()
        return cls.from_dict(_read_json(p))


M = TypeVar("M", CheckInputsManifest, AoiManifest, PatchesManifest, SplitManifest)


def load_manifest(path: str | Path, cls: Type[M]) -> M:
    p = Path(path).resolve()
    if not p.exists():
        raise RuntimeError(f"Manifest not found: {p}")
    try:
        return cls.load(p)
    except Exception as e:
        raise RuntimeError(f"Failed to read manifest {p}: {e}") from e
