from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net_train.config import load_hardware_config, load_train_config
from net_train.data import AugmentConfig, DatasetOptions, PatchDataset, build_index, load_stats_npz
from net_train.hardware import apply_torch_runtime_flags, build_runtime_plan
from net_train.models import build_model
from net_train.train import load_checkpoint, validate_one_epoch
from net_train.utils.io import write_json
from net_train.utils.logging import setup_logger



def _make_loader(dataset, plan):
    kwargs = {
        "batch_size": int(plan.batch_size),
        "shuffle": False,
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
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    console = Console()
    logger = setup_logger("eval", level=args.log_level)

    train_cfg = load_train_config(args.config)
    hw_cfg = load_hardware_config(args.hardware)
    plan = build_runtime_plan(train_cfg, hw_cfg)
    apply_torch_runtime_flags(plan)

    run_dir = Path(args.run_dir).resolve()
    stats_path = run_dir / "band_stats.npz"
    if not stats_path.exists():
        raise RuntimeError(f"Missing normalization stats: {stats_path}")
    norm_stats = load_stats_npz(stats_path)

    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else (run_dir / "checkpoints" / "best.pt")
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found in {run_dir / 'checkpoints'}")

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
    if int(model_cfg.get("in_channels", expected_in_channels)) != expected_in_channels:
        logger.warning(
            "model.in_channels mismatch, overriding to dataset contract value: %s",
            expected_in_channels,
        )
    model_cfg["in_channels"] = expected_in_channels

    splits_cfg = (ds_cfg.get("splits", {}) or {})
    splits = {
        "train": str(splits_cfg.get("train", "train")),
        "val": str(splits_cfg.get("val", "validation")),
        "test": str(splits_cfg.get("test", "test")),
    }

    index = build_index(prep_data_root=train_cfg.paths["prep_data_root"], splits=splits)
    if index["test"].missing_files:
        raise RuntimeError("test split has missing files, run 01_check_prep_data.py first")
    if not index["test"].records:
        raise RuntimeError("test split is empty")

    test_ds = PatchDataset(
        records=index["test"].records,
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
        seed=123,
    )
    test_loader = _make_loader(test_ds, plan)

    model = build_model(train_cfg.raw.get("model", {}) or {})
    model = model.to(torch.device(plan.device))

    ckpt = load_checkpoint(ckpt_path, map_location=plan.device)
    model.load_state_dict(ckpt["model_state"])

    metrics = validate_one_epoch(
        model=model,
        loader=test_loader,
        plan=plan,
        train_cfg=train_cfg.raw,
    )

    out = {
        "checkpoint": str(ckpt_path),
        "run_dir": str(run_dir),
        "split": "test",
        "metrics": metrics,
    }
    out_path = run_dir / "metrics" / "eval_test.json"
    write_json(out_path, out)

    logger.info(f"Eval metrics: {metrics}")
    console.print(f"[bold green]DONE[/bold green] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
