from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import rasterio
from rich.console import Console
from rich.logging import RichHandler
import logging

# allow "prep.*" imports when run as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prep.config import load_config
from prep.utils import find_single_by_globs, ensure_dir, write_json
from prep.qa_raster import read_raster_info, estimate_valid_ratio
from prep.qa_vector import check_and_prepare_vector


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("prep_check")


def resolve_raster_path(cfg, ds) -> Tuple[Path, str]:
    """
    Prefer AOI raster if aoi_clip.enabled and file exists:
      <aoi_clip.out_dir>/<ds.name>_aoi.tif
    else use the original raster matched by glob.
    Returns: (path, source) where source in {"aoi", "raw"}.
    """
    raw_raster_path, raw_matches = find_single_by_globs(ds.root, ds.raster_glob, ds.raster_require_single)
    if raw_raster_path is None:
        raise RuntimeError(f"{ds.name}: expected single raster by {ds.raster_glob}, got {len(raw_matches)}")

    if not cfg.aoi_clip.enabled:
        return raw_raster_path, "raw"

    if not cfg.aoi_clip.out_dir:
        return raw_raster_path, "raw"

    out_dir = Path(cfg.aoi_clip.out_dir)
    if not out_dir.is_absolute():
        out_dir = (cfg.project_root / out_dir).resolve()

    aoi_path = out_dir / f"{ds.name}_aoi.tif"
    if aoi_path.exists() and aoi_path.is_file():
        return aoi_path, "aoi"

    return raw_raster_path, "raw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)

    logger = setup_logger(cfg.logging.level)
    console = Console()

    out_dir = cfg.paths.get("output_data_dir", (ROOT / "../output_data").resolve())
    ensure_dir(out_dir)

    # report path
    report_path = Path(cfg.raw.get("reporting", {}).get("data_check_json_path", str(out_dir / "data_check.json")))
    report_path = report_path if report_path.is_absolute() else (cfg.project_root / report_path).resolve()
    ensure_dir(report_path.parent)

    # QA settings (typed)
    min_valid_ratio_global = float(cfg.qa.raster.min_valid_ratio_global)
    allow_rot = bool(cfg.qa.raster.allow_rotated_geotransform)

    min_area_m2 = float(cfg.qa.vector.min_area_m2)
    fix_invalid_mode = str(cfg.qa.vector.fix_invalid_geometries).strip()
    explode_mp = bool(cfg.qa.vector.explode_multipolygons)
    clip_to_bounds = bool(cfg.qa.vector.clip_to_raster_bounds)
    clip_geometry = bool(cfg.qa.vector.clip_geometry)
    keep_holes = bool(cfg.qa.vector.keep_holes)
    drop_empty = bool(cfg.qa.vector.drop_empty)

    # expectations (keep behavior, but nodata is special)
    expectations = cfg.raw.get("expectations", {}) or {}
    exp_band = expectations.get("band_count", "auto_from_first")
    exp_dtype = expectations.get("dtype", "auto_from_first")
    exp_nodata = expectations.get("nodata", "auto_from_first")

    global_expect: Dict[str, Any] = {
        "expected_band_count": None,
        "expected_dtype": None,
        # НОДАТА: по spec фиксируем через cfg.nodata_policy, но ещё сравним с метаданными и зарепортим
        "expected_nodata_metadata": None,
        "expected_nodata_policy_value": float(cfg.nodata_policy.nodata_value),
        "expected_nodata_policy_rule": cfg.nodata_policy.rule,
        "expected_nodata_policy_control_band_1based": int(cfg.nodata_policy.control_band_1based),
        "source": "auto_from_first" if (exp_band, exp_dtype, exp_nodata) == ("auto_from_first", "auto_from_first", "auto_from_first") else "config",
    }

    datasets_report: List[Dict[str, Any]] = []
    errors: List[str] = []
    warnings: List[str] = []

    console.print("[bold]Checking inputs (AOI-aware)...[/bold]")
    logger.info(f"Config: {cfg.config_path}")
    logger.info(f"Output report: {report_path}")
    logger.info(f"NoData policy: value={cfg.nodata_policy.nodata_value} rule={cfg.nodata_policy.rule} "
                f"control_band_1based={cfg.nodata_policy.control_band_1based}")

    for i, ds in enumerate(cfg.datasets):
        logger.info(f"\n[dataset] {ds.name}  root={ds.root}")
        ds_item: Dict[str, Any] = {"name": ds.name, "root": str(ds.root), "ok": True, "errors": [], "warnings": []}

        if not ds.root.exists():
            msg = f"{ds.name}: root does not exist: {ds.root}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            continue

        # resolve vector path
        vector_path, vector_matches = find_single_by_globs(ds.root, ds.vector_glob, ds.vector_require_single)
        ds_item["vector_matches"] = [str(p) for p in vector_matches]
        if vector_path is None:
            msg = f"{ds.name}: expected single vector by {ds.vector_glob}, got {len(vector_matches)}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            continue

        # resolve raster path (raw or aoi)
        try:
            raster_path, raster_source = resolve_raster_path(cfg, ds)
        except Exception as e:
            msg = str(e)
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            continue

        ds_item["raster_path"] = str(raster_path)
        ds_item["raster_source"] = raster_source
        ds_item["vector_path"] = str(vector_path)
        ds_item["vector_layer"] = ds.vector_layer

        # ---- Raster QA ----
        try:
            rinfo = read_raster_info(str(raster_path))
            ds_item["raster_info"] = asdict(rinfo)
        except Exception as e:
            msg = f"{ds.name}: raster open failed: {e}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            continue

        if rinfo.crs is None:
            msg = f"{ds.name}: raster CRS is missing"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        # rotated geotransform check (north-up)
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

        # Determine expected band/dtype/nodata(meta) from first raster if auto
        if i == 0:
            global_expect["expected_band_count"] = rinfo.count if exp_band == "auto_from_first" else int(exp_band)
            global_expect["expected_dtype"] = (rinfo.dtypes[0] if rinfo.dtypes else None) if exp_dtype == "auto_from_first" else str(exp_dtype)

            # метаданные nodata (не “политика”, а именно ds.nodata). часто бывает None — это ок, просто репортим.
            global_expect["expected_nodata_metadata"] = rinfo.nodata if exp_nodata == "auto_from_first" else (
                float(exp_nodata) if exp_nodata is not None else None
            )

        # Compare with expected band/dtype
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

        # nodata metadata comparison (warning-level, because policy is explicit)
        exp_nd_meta = global_expect.get("expected_nodata_metadata", None)
        if exp_nd_meta is not None and rinfo.nodata != exp_nd_meta:
            msg = f"{ds.name}: nodata-metadata mismatch: got {rinfo.nodata}, expected {exp_nd_meta} (policy value={cfg.nodata_policy.nodata_value})"
            ds_item["warnings"].append(msg)
            warnings.append(msg)
        if rinfo.nodata is None:
            msg = f"{ds.name}: raster metadata nodata is None (policy still used: value={cfg.nodata_policy.nodata_value})"
            ds_item["warnings"].append(msg)
            warnings.append(msg)

        # Sample valid ratio on chosen raster (AOI if exists) using POLICY (not ds.nodata)
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
                msg = f"{ds.name}: low valid_ratio_estimate={vr:.4f} < {min_valid_ratio_global} (source={raster_source})"
                ds_item["warnings"].append(msg)
                warnings.append(msg)
        except Exception as e:
            msg = f"{ds.name}: valid ratio sampling failed: {e}"
            ds_item["warnings"].append(msg)
            warnings.append(msg)

        # ---- Vector QA (+ prepare) using bounds of chosen raster ----
        if rinfo.crs is None:
            msg = f"{ds.name}: skip vector checks because raster CRS missing"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)
            datasets_report.append(ds_item)
            continue

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

            # Save prepared vector (DERIVED, not overwriting GT)
            work_dir = cfg.paths.get("work_dir", (out_dir / "module_prep_data_work"))
            ensure_dir(Path(work_dir))
            cleaned_path = Path(work_dir) / f"{ds.name}_vector_prepared.gpkg"
            try:
                gdf_prep.to_file(cleaned_path, driver="GPKG", layer="fields_prepared")
                ds_item["vector_prepared_saved"] = str(cleaned_path)
            except Exception as e:
                msg = f"{ds.name}: failed to save prepared vector gpkg: {e}"
                ds_item["warnings"].append(msg)
                warnings.append(msg)

        except Exception as e:
            msg = f"{ds.name}: vector check failed: {e}"
            ds_item["ok"] = False
            ds_item["errors"].append(msg)
            errors.append(msg)

        datasets_report.append(ds_item)

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
        "ok": (len(errors) == 0),
    }

    write_json(report_path, report)

    if errors:
        console.print(f"[bold red]FAILED[/bold red] errors={len(errors)}  report={report_path}")
        return 2
    console.print(f"[bold green]OK[/bold green] warnings={len(warnings)}  report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())