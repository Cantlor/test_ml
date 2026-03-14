from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import rasterio
from rich.console import Console
from rich.logging import RichHandler

from ..artifacts import PREPARED_VECTOR_LAYER, check_inputs_manifest_path, get_work_dir
from ..config import load_config
from ..manifests import CheckInputsDatasetResult, CheckInputsManifest
from ..qa_raster import estimate_valid_ratio, read_raster_info
from ..qa_vector import check_and_prepare_vector
from ..utils import ensure_dir, find_single_by_globs, write_json


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("prep_check")


def _report_path(cfg) -> Path:
    out_dir = cfg.paths.get("output_data_dir", (cfg.project_root / "../output_data").resolve())
    ensure_dir(out_dir)
    report_path = Path(cfg.raw.get("reporting", {}).get("data_check_json_path", str(out_dir / "data_check.json")))
    return report_path if report_path.is_absolute() else (cfg.project_root / report_path).resolve()


def _deferred_config_keys(cfg) -> List[str]:
    keys: List[str] = []
    rp = cfg.raw.get("raster_preprocess", {}) or {}
    if rp:
        keys.append("raster_preprocess.*")
    if cfg.patching.samples_per_feature != 1:
        keys.append("patching.sampling.samples_per_feature")
    if cfg.patching.near_nodata.enabled:
        keys.append("patching.sampling.near_nodata.*")
    if cfg.split.spatial_blocking_enabled:
        keys.append("split.spatial_blocking.enabled")
    if str(cfg.qa.vector.holes_policy).strip().lower() != "background":
        keys.append("qa.vector.holes_policy")
    reporting = cfg.raw.get("reporting", {}) or {}
    if reporting.get("save_summary_csv", False):
        keys.append("reporting.save_summary_csv")
    if reporting.get("save_previews", False):
        keys.append("reporting.save_previews")
    if cfg.logging.log_file:
        keys.append("logging.log_file")
    return keys


def run(config_path: str | Path) -> int:
    cfg = load_config(config_path)
    logger = setup_logger(cfg.logging.level)
    console = Console()

    report_path = _report_path(cfg)
    ensure_dir(report_path.parent)
    work_dir = get_work_dir(cfg)
    ensure_dir(work_dir)

    min_valid_ratio_global = float(cfg.qa.raster.min_valid_ratio_global)
    allow_rot = bool(cfg.qa.raster.allow_rotated_geotransform)

    min_area_m2 = float(cfg.qa.vector.min_area_m2)
    fix_invalid_mode = str(cfg.qa.vector.fix_invalid_geometries).strip()
    explode_mp = bool(cfg.qa.vector.explode_multipolygons)
    clip_to_bounds = bool(cfg.qa.vector.clip_to_raster_bounds)
    clip_geometry = bool(cfg.qa.vector.clip_geometry)
    keep_holes = bool(cfg.qa.vector.keep_holes)
    drop_empty = bool(cfg.qa.vector.drop_empty)

    expectations = cfg.raw.get("expectations", {}) or {}
    exp_band = expectations.get("band_count", "auto_from_first")
    exp_dtype = expectations.get("dtype", "auto_from_first")
    exp_nodata = expectations.get("nodata", "auto_from_first")

    global_expect: Dict[str, Any] = {
        "expected_band_count": None,
        "expected_dtype": None,
        "expected_nodata_metadata": None,
        "expected_nodata_policy_value": float(cfg.nodata_policy.nodata_value),
        "expected_nodata_policy_rule": cfg.nodata_policy.rule,
        "expected_nodata_policy_control_band_1based": int(cfg.nodata_policy.control_band_1based),
        "source": "auto_from_first"
        if (exp_band, exp_dtype, exp_nodata) == ("auto_from_first", "auto_from_first", "auto_from_first")
        else "config",
    }

    datasets_report: List[Dict[str, Any]] = []
    datasets_manifest: List[CheckInputsDatasetResult] = []
    errors: List[str] = []
    warnings: List[str] = []

    console.print("[bold]Checking inputs (strict manifest mode)...[/bold]")
    logger.info(f"Config: {cfg.config_path}")
    logger.info(f"Output report: {report_path}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(
        f"NoData policy: value={cfg.nodata_policy.nodata_value} rule={cfg.nodata_policy.rule} "
        f"control_band_1based={cfg.nodata_policy.control_band_1based}"
    )

    for i, ds in enumerate(cfg.datasets):
        logger.info(f"\n[dataset] {ds.name}  root={ds.root}")
        ds_item: Dict[str, Any] = {"name": ds.name, "root": str(ds.root), "ok": True, "errors": [], "warnings": []}

        if not ds.root.exists():
            msg = f"{ds.name}: root does not exist: {ds.root}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            datasets_manifest.append(
                CheckInputsDatasetResult(
                    dataset=ds.name,
                    root=str(ds.root),
                    raw_raster_path="",
                    raw_vector_path="",
                    vector_layer=ds.vector_layer,
                    vector_id_field=ds.vector_id_field,
                    qa_raster_path="",
                    qa_raster_source="raw",
                    prepared_vector_path=None,
                    prepared_vector_layer=None,
                    qa_ok=False,
                    errors=[msg],
                    warnings=[],
                )
            )
            continue

        raster_path, raster_matches = find_single_by_globs(ds.root, ds.raster_glob, ds.raster_require_single)
        vector_path, vector_matches = find_single_by_globs(ds.root, ds.vector_glob, ds.vector_require_single)
        ds_item["raster_matches"] = [str(p) for p in raster_matches]
        ds_item["vector_matches"] = [str(p) for p in vector_matches]

        if raster_path is None:
            msg = f"{ds.name}: expected single raster by {ds.raster_glob}, got {len(raster_matches)}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        if vector_path is None:
            msg = f"{ds.name}: expected single vector by {ds.vector_glob}, got {len(vector_matches)}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        if raster_path is None or vector_path is None:
            datasets_report.append(ds_item)
            datasets_manifest.append(
                CheckInputsDatasetResult(
                    dataset=ds.name,
                    root=str(ds.root),
                    raw_raster_path=str(raster_path) if raster_path else "",
                    raw_vector_path=str(vector_path) if vector_path else "",
                    vector_layer=ds.vector_layer,
                    vector_id_field=ds.vector_id_field,
                    qa_raster_path=str(raster_path) if raster_path else "",
                    qa_raster_source="raw",
                    prepared_vector_path=None,
                    prepared_vector_layer=None,
                    qa_ok=False,
                    errors=[str(x) for x in ds_item["errors"]],
                    warnings=[str(x) for x in ds_item["warnings"]],
                )
            )
            continue

        ds_item["raster_path"] = str(raster_path)
        ds_item["raster_source"] = "raw"
        ds_item["vector_path"] = str(vector_path)
        ds_item["vector_layer"] = ds.vector_layer

        try:
            rinfo = read_raster_info(str(raster_path))
            ds_item["raster_info"] = asdict(rinfo)
        except Exception as e:
            msg = f"{ds.name}: raster open failed: {e}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            datasets_manifest.append(
                CheckInputsDatasetResult(
                    dataset=ds.name,
                    root=str(ds.root),
                    raw_raster_path=str(raster_path),
                    raw_vector_path=str(vector_path),
                    vector_layer=ds.vector_layer,
                    vector_id_field=ds.vector_id_field,
                    qa_raster_path=str(raster_path),
                    qa_raster_source="raw",
                    prepared_vector_path=None,
                    prepared_vector_layer=None,
                    qa_ok=False,
                    errors=[str(x) for x in ds_item["errors"]],
                    warnings=[str(x) for x in ds_item["warnings"]],
                )
            )
            continue

        if rinfo.crs is None:
            msg = f"{ds.name}: raster CRS is missing"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        if not allow_rot:
            try:
                with rasterio.open(str(raster_path)) as dsrio:
                    tr = dsrio.transform
                    if abs(tr.b) > 1e-12 or abs(tr.d) > 1e-12:
                        msg = f"{ds.name}: rotated geotransform detected (tr.b={tr.b}, tr.d={tr.d})"
                        ds_item["ok"] = False
                        ds_item["errors"].append(msg)
                        errors.append(msg)
            except Exception:
                pass

        if i == 0:
            global_expect["expected_band_count"] = rinfo.count if exp_band == "auto_from_first" else int(exp_band)
            global_expect["expected_dtype"] = (
                (rinfo.dtypes[0] if rinfo.dtypes else None) if exp_dtype == "auto_from_first" else str(exp_dtype)
            )
            global_expect["expected_nodata_metadata"] = (
                rinfo.nodata
                if exp_nodata == "auto_from_first"
                else (float(exp_nodata) if exp_nodata is not None else None)
            )

        if global_expect["expected_band_count"] is not None and rinfo.count != int(global_expect["expected_band_count"]):
            msg = f"{ds.name}: band_count mismatch: got {rinfo.count}, expected {global_expect['expected_band_count']}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        if global_expect["expected_dtype"] is not None and (rinfo.dtypes and rinfo.dtypes[0] != global_expect["expected_dtype"]):
            msg = f"{ds.name}: dtype mismatch: got {rinfo.dtypes[0]}, expected {global_expect['expected_dtype']}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        exp_nd_meta = global_expect.get("expected_nodata_metadata", None)
        if exp_nd_meta is not None and rinfo.nodata != exp_nd_meta:
            msg = (
                f"{ds.name}: nodata-metadata mismatch: got {rinfo.nodata}, expected {exp_nd_meta} "
                f"(policy value={cfg.nodata_policy.nodata_value})"
            )
            ds_item["warnings"].append(msg)
            warnings.append(msg)
        if rinfo.nodata is None:
            msg = f"{ds.name}: raster metadata nodata is None (policy still used: value={cfg.nodata_policy.nodata_value})"
            ds_item["warnings"].append(msg)
            warnings.append(msg)

        try:
            vr, vr_meta = estimate_valid_ratio(
                str(raster_path),
                nodata_value=float(cfg.nodata_policy.nodata_value),
                nodata_rule=str(cfg.nodata_policy.rule),
                control_band_1based=int(cfg.nodata_policy.control_band_1based),
                sample_target_pixels=2_000_000,
                window_size=512,
                seed=123,
            )
            ds_item["raster_valid_ratio_estimate"] = vr
            ds_item["raster_valid_ratio_meta"] = vr_meta
            if vr is not None and vr < min_valid_ratio_global:
                msg = f"{ds.name}: low valid_ratio_estimate={vr:.4f} < {min_valid_ratio_global}"
                ds_item["warnings"].append(msg)
                warnings.append(msg)
        except Exception as e:
            msg = f"{ds.name}: valid ratio sampling failed: {e}"
            ds_item["warnings"].append(msg)
            warnings.append(msg)

        prepared_vector_path: Path | None = None
        if rinfo.crs is None:
            msg = f"{ds.name}: skip vector checks because raster CRS missing"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
        else:
            try:
                gdf_prep, vinfo, vextra = check_and_prepare_vector(
                    vector_path=str(vector_path),
                    raster_bounds=rinfo.bounds,
                    raster_crs_str=rinfo.crs,
                    min_area_m2=min_area_m2,
                    vector_layer=ds.vector_layer,
                    fix_invalid_mode=fix_invalid_mode,
                    drop_empty=drop_empty,
                    explode_multipolygons=explode_mp,
                    clip_to_bounds=clip_to_bounds,
                    clip_geometry=clip_geometry,
                    keep_holes=keep_holes,
                )
                ds_item["vector_info"] = asdict(vinfo)
                ds_item["vector_extra"] = vextra

                prepared_vector_path = work_dir / f"{ds.name}_vector_prepared.gpkg"
                try:
                    gdf_prep.to_file(prepared_vector_path, driver="GPKG", layer=PREPARED_VECTOR_LAYER)
                    ds_item["vector_prepared_saved"] = str(prepared_vector_path)
                    ds_item["vector_prepared_layer"] = PREPARED_VECTOR_LAYER
                except Exception as e:
                    msg = f"{ds.name}: failed to save prepared vector gpkg: {e}"
                    ds_item["warnings"].append(msg)
                    warnings.append(msg)
                    prepared_vector_path = None
            except Exception as e:
                msg = f"{ds.name}: vector check failed: {e}"
                ds_item["ok"] = False
                ds_item["errors"].append(msg)
                errors.append(msg)

        datasets_report.append(ds_item)
        datasets_manifest.append(
            CheckInputsDatasetResult(
                dataset=ds.name,
                root=str(ds.root),
                raw_raster_path=str(raster_path),
                raw_vector_path=str(vector_path),
                vector_layer=ds.vector_layer,
                vector_id_field=ds.vector_id_field,
                qa_raster_path=str(raster_path),
                qa_raster_source="raw",
                prepared_vector_path=str(prepared_vector_path) if prepared_vector_path else None,
                prepared_vector_layer=PREPARED_VECTOR_LAYER if prepared_vector_path else None,
                qa_ok=bool(ds_item["ok"]),
                errors=[str(x) for x in ds_item["errors"]],
                warnings=[str(x) for x in ds_item["warnings"]],
            )
        )

    report: Dict[str, Any] = {
        "config": str(cfg.config_path),
        "nodata_policy": {
            "nodata_value": float(cfg.nodata_policy.nodata_value),
            "rule": str(cfg.nodata_policy.rule),
            "control_band_1based": int(cfg.nodata_policy.control_band_1based),
        },
        "global_expectations": global_expect,
        "datasets": datasets_report,
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0,
    }
    write_json(report_path, report)

    m = CheckInputsManifest.new(
        config_path=cfg.config_path,
        work_dir=work_dir,
        data_check_json_path=report_path,
        datasets=datasets_manifest,
        deferred_config_keys=_deferred_config_keys(cfg),
    )
    mpath = m.save(check_inputs_manifest_path(cfg))
    logger.info(f"check_inputs manifest: {mpath}")

    if errors:
        console.print(f"[bold red]FAILED[/bold red] errors={len(errors)}  report={report_path}")
        return 2
    console.print(f"[bold green]OK[/bold green] warnings={len(warnings)}  report={report_path}")
    return 0
