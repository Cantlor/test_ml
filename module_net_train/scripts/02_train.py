from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net_train.config import dump_yaml, load_hardware_config, load_train_config, resolve_inference_manifest_path
from net_train.data import (
    AugmentConfig,
    DatasetOptions,
    PatchDataset,
    build_index,
    compute_normalization_stats,
    save_stats_npz,
)
from net_train.hardware import apply_torch_runtime_flags, build_runtime_plan
from net_train.infer import predict_aoi_raster, resolve_aoi_path
from net_train.models import build_model
from net_train.train import CheckpointManager, create_optimizer, create_scheduler, load_checkpoint, run_training
from net_train.utils.io import ensure_dir, utc_now_compact, write_json
from net_train.utils.logging import setup_logger
from net_train.utils.seed import seed_everything



def _make_loader(dataset, plan, shuffle: bool):
    kwargs = {
        "batch_size": int(plan.batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(plan.num_workers),
        "pin_memory": bool(plan.pin_memory),
        "persistent_workers": bool(plan.persistent_workers and plan.num_workers > 0),
        "drop_last": False,
    }
    if plan.num_workers > 0:
        kwargs["prefetch_factor"] = int(plan.prefetch_factor)
    return DataLoader(dataset, **kwargs)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train_config.yaml"))
    ap.add_argument("--hardware", default=str(ROOT / "configs/hardware_config.yaml"))
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--no_infer", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    train_cfg = load_train_config(args.config)
    hw_cfg = load_hardware_config(args.hardware)

    runs_root = train_cfg.paths.get("runs_root", (train_cfg.project_root / "output_data/module_net_train/runs").resolve())
    run_id = args.run_id or utc_now_compact()
    run_dir = runs_root / run_id

    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "logs" / "tensorboard")
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    ensure_dir(run_dir / "pred")

    logger = setup_logger("train", level=args.log_level, log_file=run_dir / "logs" / "train.log")
    console = Console()

    plan = build_runtime_plan(train_cfg, hw_cfg)
    apply_torch_runtime_flags(plan)

    for w in plan.warnings:
        logger.warning(w)

    # Apply runtime plan into effective training config (resolved config contract).
    train_cfg.raw.setdefault("sampling", {})["crop_size"] = int(plan.crop_size)
    train_cfg.raw.setdefault("train", {}).setdefault("batch", {})["batch_size"] = int(plan.batch_size)
    train_cfg.raw.setdefault("train", {}).setdefault("batch", {})["grad_accum_steps"] = int(plan.grad_accum_steps)

    write_json(run_dir / "hardware.json", {
        "hardware_config": str(hw_cfg.config_path),
        "runtime_plan": plan.to_dict(),
    })
    dump_yaml(run_dir / "config_resolved.yaml", train_cfg.raw)

    seed = int((train_cfg.raw.get("train", {}) or {}).get("seed", 123))
    deterministic = bool((train_cfg.raw.get("train", {}) or {}).get("deterministic", False))
    seed_everything(seed=seed, deterministic=deterministic)

    splits_cfg = ((train_cfg.raw.get("dataset", {}) or {}).get("splits", {}) or {})
    splits = {
        "train": str(splits_cfg.get("train", "train")),
        "val": str(splits_cfg.get("val", "validation")),
        "test": str(splits_cfg.get("test", "test")),
    }

    prep_data_root = train_cfg.paths["prep_data_root"]
    index = build_index(prep_data_root=prep_data_root, splits=splits)

    for key in ["train", "val"]:
        if index[key].missing_files:
            raise RuntimeError(f"{key}: dataset index has missing files, run 01_check_prep_data.py first")

    train_records = index["train"].records
    val_records = index["val"].records

    if not train_records or not val_records:
        raise RuntimeError("train/val split is empty")

    ds_cfg = train_cfg.raw.get("dataset", {}) or {}
    inputs_cfg = (ds_cfg.get("inputs", {}) or {})
    targets_cfg = (ds_cfg.get("targets", {}) or {})
    extent_ignore = int(((targets_cfg.get("extent", {}) or {}).get("ignore_value", 255)))
    boundary_ignore = int(((targets_cfg.get("boundary_bwbl", {}) or {}).get("ignore_value", 2)))
    num_bands = int(inputs_cfg.get("num_bands", 8))
    add_valid_channel = bool(inputs_cfg.get("add_valid_channel", True))
    nodata_rule = str(inputs_cfg.get("nodata_rule", "control-band"))
    control_band_1based = int(inputs_cfg.get("control_band_1based", 1))
    nodata = inputs_cfg.get("nodata_value", 65536)
    nodata = None if nodata is None else float(nodata)

    expected_in_channels = int(num_bands + (1 if add_valid_channel else 0))
    model_cfg = train_cfg.raw.setdefault("model", {})
    model_in_channels = int(model_cfg.get("in_channels", expected_in_channels))
    if model_in_channels != expected_in_channels:
        logger.warning(
            "model.in_channels=%s does not match dataset contract (%s bands + valid=%s => %s). Overriding.",
            model_in_channels,
            num_bands,
            add_valid_channel,
            expected_in_channels,
        )
        model_cfg["in_channels"] = expected_in_channels
    else:
        model_cfg.setdefault("in_channels", expected_in_channels)
    dump_yaml(run_dir / "config_resolved.yaml", train_cfg.raw)

    norm_cfg = train_cfg.raw.get("normalization", {}) or {}
    stats_split = str(norm_cfg.get("stats_split", "train")).lower()
    stats_records_map = {
        "train": index["train"].records,
        "val": index["val"].records,
        "validation": index["val"].records,
        "test": index["test"].records,
    }
    stats_records = stats_records_map.get(stats_split)
    if stats_records is None:
        raise RuntimeError(f"Unsupported normalization.stats_split: {stats_split}")
    if not stats_records:
        raise RuntimeError(f"normalization.stats_split={stats_split} is empty")

    norm_stats = compute_normalization_stats(
        records=stats_records,
        mode=str(norm_cfg.get("type", "robust_percentile")),
        per_band=bool(norm_cfg.get("per_band", True)),
        nodata_value=nodata,
        ignore_nodata=bool(norm_cfg.get("ignore_nodata", True)),
        p_low=float(norm_cfg.get("p_low", 2.0)),
        p_high=float(norm_cfg.get("p_high", 98.0)),
        seed=seed,
        image_bands=num_bands,
    )
    save_stats_npz(run_dir / "band_stats.npz", norm_stats)

    aug_cfg_raw = train_cfg.raw.get("augmentations", {}) or {}
    aug_cfg = AugmentConfig(
        enabled=bool(aug_cfg_raw.get("enabled", True)),
        hflip=bool(aug_cfg_raw.get("hflip", True)),
        vflip=bool(aug_cfg_raw.get("vflip", True)),
        rotate90=bool(aug_cfg_raw.get("rotate90", True)),
    )

    train_ds = PatchDataset(
        records=train_records,
        norm_stats=norm_stats,
        options=DatasetOptions(
            crop_size=int(plan.crop_size),
            num_bands=num_bands,
            nodata_value=nodata,
            extent_ignore_value=extent_ignore,
            boundary_ignore_value=boundary_ignore,
            is_train=True,
            add_valid_channel=add_valid_channel,
            nodata_rule=nodata_rule,
            control_band_1based=control_band_1based,
        ),
        augment_cfg=aug_cfg,
        seed=seed,
    )

    val_ds = PatchDataset(
        records=val_records,
        norm_stats=norm_stats,
        options=DatasetOptions(
            crop_size=0,
            num_bands=num_bands,
            nodata_value=nodata,
            extent_ignore_value=extent_ignore,
            boundary_ignore_value=boundary_ignore,
            is_train=False,
            add_valid_channel=add_valid_channel,
            nodata_rule=nodata_rule,
            control_band_1based=control_band_1based,
        ),
        augment_cfg=AugmentConfig(enabled=False),
        seed=seed,
    )

    train_loader = _make_loader(train_ds, plan=plan, shuffle=True)
    val_loader = _make_loader(val_ds, plan=plan, shuffle=False)

    model = build_model(train_cfg.raw.get("model", {}) or {})
    model = model.to(torch.device(plan.device))
    if plan.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if plan.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = create_optimizer(train_cfg.raw, model)
    scheduler = create_scheduler(train_cfg.raw, optimizer)

    ck_cfg = ((train_cfg.raw.get("train", {}) or {}).get("checkpoint", {}) or {})
    ckpt_manager = CheckpointManager(
        ckpt_dir=run_dir / "checkpoints",
        monitor=str(ck_cfg.get("monitor", "val/extent_iou")),
        mode=str(ck_cfg.get("mode", "max")),
        save_last=bool(ck_cfg.get("save_last", True)),
        save_best=bool(ck_cfg.get("save_best", True)),
    )

    console.print(f"[bold]Run:[/bold] {run_id}")
    console.print(f"device={plan.device} precision={plan.precision} crop={plan.crop_size} batch={plan.batch_size}")

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        plan=plan,
        train_cfg=train_cfg.raw,
        ckpt_manager=ckpt_manager,
        history_csv_path=run_dir / "metrics" / "history.csv",
        logger=logger,
    )

    do_infer = bool((train_cfg.raw.get("inference", {}) or {}).get("enabled", True)) and not args.no_infer
    if do_infer:
        infer_cfg = train_cfg.raw.get("inference", {}) or {}
        aoi_cfg = infer_cfg.get("aoi", {}) or {}
        tiling_cfg = infer_cfg.get("tiling", {}) or {}
        out_cfg = infer_cfg.get("outputs", {}) or {}

        best_ckpt = run_dir / "checkpoints" / "best.pt"
        last_ckpt = run_dir / "checkpoints" / "last.pt"
        ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt

        ckpt = load_checkpoint(ckpt_path, map_location=plan.device)
        model.load_state_dict(ckpt["model_state"])

        manifest_path = resolve_inference_manifest_path(train_cfg)
        dataset_key = str(aoi_cfg.get("dataset_key", "first_raster"))
        aoi_path = resolve_aoi_path(manifest_path, dataset_key)

        pred_dir = run_dir / "pred" / dataset_key
        extent_name = str(out_cfg.get("extent_prob_name", "extent_prob.tif"))
        boundary_name = str(out_cfg.get("boundary_prob_name", "boundary_prob.tif"))

        infer_batch_cfg = tiling_cfg.get("batch_size", "auto")
        infer_batch = int(plan.batch_size if infer_batch_cfg == "auto" else infer_batch_cfg)

        pred_info = predict_aoi_raster(
            model=model,
            plan=plan,
            aoi_raster_path=aoi_path,
            norm_stats=norm_stats,
            out_extent_path=pred_dir / extent_name,
            out_boundary_path=pred_dir / boundary_name,
            window_size=int(tiling_cfg.get("window_size", 512)),
            stride=int(tiling_cfg.get("stride", 384)),
            batch_size=int(max(1, infer_batch)),
            blend=str(tiling_cfg.get("blend", "mean")),
            out_dtype=str(out_cfg.get("dtype", "float32")),
            compress=str(out_cfg.get("compress", "DEFLATE")).upper(),
            tiled=bool(out_cfg.get("tiled", True)),
            bigtiff=str(out_cfg.get("bigtiff", "if_needed")),
            num_bands=num_bands,
            add_valid_channel=add_valid_channel,
            nodata_value=nodata,
            nodata_rule=nodata_rule,
            control_band_1based=control_band_1based,
        )
        write_json(pred_dir / "predict_manifest.json", pred_info)
        console.print(f"[green]Inference done[/green] -> {pred_dir}")

    console.print("[bold green]Training pipeline finished[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
