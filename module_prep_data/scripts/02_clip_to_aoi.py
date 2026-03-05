from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prep.config import load_config
from prep.utils import find_single_by_globs, ensure_dir, write_json
from prep.clip_raster import clip_raster_by_vectors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    console = Console()

    if not cfg.aoi_clip.enabled:
        console.print("[yellow]aoi_clip.enabled=false -> nothing to do[/yellow]")
        return 0

    mode = str(cfg.aoi_clip.mode)
    buffer_m = float(cfg.aoi_clip.buffer_m)
    mask_outside = bool(cfg.aoi_clip.mask_outside)

    # out_dir: config -> default
    if cfg.aoi_clip.out_dir:
        out_dir = Path(cfg.aoi_clip.out_dir)
        out_dir = out_dir if out_dir.is_absolute() else (cfg.project_root / out_dir).resolve()
    else:
        work_dir = cfg.paths.get("work_dir", (cfg.project_root / "../output_data/module_prep_data_work").resolve())
        out_dir = Path(work_dir) / "aoi_rasters"

    ensure_dir(out_dir)

    results = []
    for ds in cfg.datasets:
        raster_path, _ = find_single_by_globs(ds.root, ds.raster_glob, ds.raster_require_single)
        vector_path, _ = find_single_by_globs(ds.root, ds.vector_glob, ds.vector_require_single)
        if raster_path is None or vector_path is None:
            raise RuntimeError(f"{ds.name}: cannot resolve raster/vector (check config globs)")

        out_path = out_dir / f"{ds.name}_aoi.tif"

        console.print(f"[bold]{ds.name}[/bold]")
        console.print(f"  raster: {raster_path}")
        console.print(f"  vector: {vector_path}" + (f" (layer={ds.vector_layer})" if ds.vector_layer else ""))
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
            vector_layer=ds.vector_layer,
            nodata_value=float(cfg.nodata_policy.nodata_value),  # если ds.nodata None — используем policy
        )

        results.append(
            {
                "dataset": ds.name,
                "out_raster": res.out_path,
                "mode": res.mode,
                "bounds_used": res.bounds_used,
                "window": res.window,
                "wrote_mask_outside": res.wrote_mask_outside,
            }
        )

    # Detailed manifest (backward compatible)
    manifest = out_dir / "aoi_rasters_manifest.json"
    write_json(manifest, {"results": results})

    # Canonical manifest for downstream consumers:
    # { "<dataset_key>": "<absolute path to AOI tif>" }
    work_dir2 = cfg.paths.get("work_dir", (cfg.project_root / "../output_data/module_prep_data_work").resolve())
    ensure_dir(Path(work_dir2))
    canonical_manifest = Path(work_dir2) / "aoi_rasters_manifest.json"
    canonical_map = {item["dataset"]: item["out_raster"] for item in results}
    write_json(canonical_manifest, canonical_map)

    console.print(f"\n[green]DONE[/green] manifest: {manifest}")
    console.print(f"canonical manifest: {canonical_manifest}")
    console.print("Теперь в следующих шагах используем AOI-расты вместо исходных (ускорит пайплайн).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())