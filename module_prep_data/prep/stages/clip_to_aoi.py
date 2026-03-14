from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from rich.console import Console

from ..artifacts import (
    aoi_manifest_path,
    check_dataset_entry,
    get_work_dir,
    load_check_inputs_manifest_required,
)
from ..clip_raster import clip_raster_by_vectors
from ..config import load_config
from ..manifests import AoiDatasetResult, AoiManifest
from ..utils import ensure_dir, write_json


def _aoi_out_dir(cfg) -> Path:
    if cfg.aoi_clip.out_dir:
        p = Path(cfg.aoi_clip.out_dir)
        return p if p.is_absolute() else (cfg.project_root / p).resolve()
    return (get_work_dir(cfg) / "aoi_rasters").resolve()


def _deferred_config_keys(cfg) -> List[str]:
    keys: List[str] = []
    if cfg.performance.num_workers:
        keys.append("performance.num_workers")
    if cfg.performance.gdal_cache_mb:
        keys.append("performance.gdal_cache_mb")
    return keys


def run(config_path: str | Path) -> int:
    cfg = load_config(config_path)
    console = Console()

    check_manifest = load_check_inputs_manifest_required(cfg)
    out_dir = _aoi_out_dir(cfg)
    ensure_dir(out_dir)
    work_dir = get_work_dir(cfg)
    ensure_dir(work_dir)

    mode = str(cfg.aoi_clip.mode)
    buffer_m = float(cfg.aoi_clip.buffer_m)
    mask_outside = bool(cfg.aoi_clip.mask_outside)

    results: List[AoiDatasetResult] = []

    if not cfg.aoi_clip.enabled:
        for ds in cfg.datasets:
            c = check_dataset_entry(check_manifest, ds.name)
            results.append(
                AoiDatasetResult(
                    dataset=ds.name,
                    source_raster_path=str(c.raw_raster_path),
                    vector_path=str(c.raw_vector_path),
                    vector_layer=c.vector_layer,
                    aoi_raster_path=None,
                    mode=mode,
                    buffer_m=buffer_m,
                    mask_outside=mask_outside,
                    wrote_mask_outside=False,
                    status="disabled",
                    message="aoi_clip.enabled=false",
                )
            )
        m = AoiManifest.new(
            config_path=cfg.config_path,
            work_dir=work_dir,
            out_dir=out_dir,
            enabled=False,
            datasets=results,
            deferred_config_keys=_deferred_config_keys(cfg),
        )
        mpath = m.save(aoi_manifest_path(cfg))
        console.print("[yellow]aoi_clip.enabled=false -> manifest written[/yellow]")
        console.print(f"aoi manifest: {mpath}")
        return 0

    for ds in cfg.datasets:
        c = check_dataset_entry(check_manifest, ds.name)
        raster_path = Path(c.raw_raster_path).resolve()
        vector_path = Path(c.raw_vector_path).resolve()
        if not raster_path.exists():
            raise RuntimeError(f"{ds.name}: raw raster missing from check_inputs manifest: {raster_path}")
        if not vector_path.exists():
            raise RuntimeError(f"{ds.name}: raw vector missing from check_inputs manifest: {vector_path}")

        out_path = out_dir / f"{ds.name}_aoi.tif"
        console.print(f"[bold]{ds.name}[/bold]")
        console.print(f"  raster: {raster_path}")
        console.print(f"  vector: {vector_path}" + (f" (layer={c.vector_layer})" if c.vector_layer else ""))
        console.print(f"  -> out: {out_path}  (mode={mode}, buffer_m={buffer_m}, mask_outside={mask_outside})")

        res = clip_raster_by_vectors(
            raster_path=str(raster_path),
            vector_path=str(vector_path),
            out_path=str(out_path),
            mode=mode,
            buffer_m=buffer_m,
            mask_outside=mask_outside,
            compress=str(cfg.aoi_clip.compress),
            tiled=bool(cfg.aoi_clip.tiled),
            bigtiff=str(cfg.aoi_clip.bigtiff),
            vector_layer=c.vector_layer,
            nodata_value=float(cfg.nodata_policy.nodata_value),
        )
        results.append(
            AoiDatasetResult(
                dataset=ds.name,
                source_raster_path=str(raster_path),
                vector_path=str(vector_path),
                vector_layer=c.vector_layer,
                aoi_raster_path=str(res.out_path),
                mode=str(res.mode),
                buffer_m=buffer_m,
                mask_outside=mask_outside,
                wrote_mask_outside=bool(res.wrote_mask_outside),
                status="clipped",
                message=None,
            )
        )

    manifest = AoiManifest.new(
        config_path=cfg.config_path,
        work_dir=work_dir,
        out_dir=out_dir,
        enabled=True,
        datasets=results,
        deferred_config_keys=_deferred_config_keys(cfg),
    )
    mpath = manifest.save(aoi_manifest_path(cfg))

    # Backward-compatible manifests for existing consumers.
    detailed_path = out_dir / "aoi_rasters_manifest.json"
    detailed_results = [
        {
            "dataset": item.dataset,
            "out_raster": item.aoi_raster_path,
            "mode": item.mode,
            "wrote_mask_outside": item.wrote_mask_outside,
            "status": item.status,
        }
        for item in results
        if item.status == "clipped" and item.aoi_raster_path
    ]
    write_json(detailed_path, {"results": detailed_results})

    canonical_map: Dict[str, str] = {item.dataset: str(item.aoi_raster_path) for item in results if item.aoi_raster_path}
    canonical_manifest = work_dir / "aoi_rasters_manifest.json"
    write_json(canonical_manifest, canonical_map)

    console.print(f"\n[green]DONE[/green] aoi manifest: {mpath}")
    console.print(f"compat manifest (map): {canonical_manifest}")
    return 0
