from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

from net_train.data import NormalizationStats, save_stats_npz
from net_train.models import build_model


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, allow_unicode=False, sort_keys=False), encoding="utf-8")


def _make_test_raster(path: Path, base_value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = 32
    w = 32
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 8,
        "dtype": "uint32",
        "crs": "EPSG:32642",
        "transform": from_origin(500000.0, 4200000.0, 10.0, 10.0),
        "nodata": 65536.0,
    }
    with rasterio.open(path, "w", **profile) as ds:
        for band in range(1, 9):
            arr = np.full((h, w), base_value + band, dtype=np.uint32)
            arr[:2, :2] = np.uint32(65536)
            ds.write(arr, band)


def _make_run_dir(run_dir: Path, config_obj: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_resolved.yaml", config_obj)

    model = build_model(config_obj.get("model", {}) or {})
    torch.save({"model_state": model.state_dict()}, run_dir / "checkpoints" / "best.pt")

    stats = NormalizationStats(
        mode="robust_percentile",
        per_band=True,
        p_low=2.0,
        p_high=98.0,
        q_low=[0.0] * 8,
        q_high=[4000.0] * 8,
        mean=None,
        std=None,
    )
    save_stats_npz(run_dir / "band_stats.npz", stats)


def _minimal_train_cfg() -> dict:
    return {
        "paths": {
            "prep_data_root": "../prep_data",
            "runs_root": "../output_data/module_net_train/runs",
        },
        "dataset": {
            "inputs": {
                "num_bands": 8,
                "nodata_value": 65536,
                "add_valid_channel": True,
                "nodata_rule": "control-band",
                "control_band_1based": 1,
            }
        },
        "model": {
            "name": "unet_multitask",
            "in_channels": 9,
            "base_channels": 4,
            "depth": 2,
            "norm": "batchnorm",
            "dropout": 0.0,
        },
        "inference": {
            "tiling": {
                "window_size": 32,
                "stride": 32,
                "batch_size": 1,
                "blend": "mean",
                "gaussian_sigma": 0.30,
                "gaussian_min_weight": 0.05,
                "invalid_edge_guard": {
                    "enabled": False,
                    "radius_px": 2,
                    "extent_scale": 0.9,
                    "boundary_scale": 0.75,
                },
            },
            "outputs": {
                "extent_prob_name": "extent_prob.tif",
                "boundary_prob_name": "boundary_prob.tif",
                "dtype": "float32",
                "compress": "deflate",
                "tiled": True,
                "bigtiff": "if_needed",
            },
        },
    }


def _minimal_hw_cfg() -> dict:
    return {
        "device": {"mode": "cpu", "prefer_cuda": False},
        "precision": {"mode": "fp32", "prefer_bf16": False, "allow_tf32": False},
        "autotune": {"enabled": False},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2},
        "runtime": {"warn_if_cpu": False},
    }


def test_predict_input_data_batch_smoke(tmp_path: Path) -> None:
    module_root = Path(__file__).resolve().parents[1]
    script = module_root / "scripts" / "06_predict_input_data_batch.py"

    config_path = tmp_path / "train_config.yaml"
    hardware_path = tmp_path / "hardware_config.yaml"
    train_cfg = _minimal_train_cfg()
    _write_yaml(config_path, train_cfg)
    _write_yaml(hardware_path, _minimal_hw_cfg())

    run_dir = tmp_path / "run_demo"
    _make_run_dir(run_dir, train_cfg)

    input_dir = tmp_path / "input_data"
    _make_test_raster(input_dir / "My Scene.tif", base_value=1000)
    _make_test_raster(input_dir / "My@Scene.tiff", base_value=2000)

    output_root = tmp_path / "direct_predict"
    cmd = [
        sys.executable,
        str(script),
        "--input_dir",
        str(input_dir),
        "--run_dir",
        str(run_dir),
        "--output_root",
        str(output_root),
        "--python",
        str(sys.executable),
        "--config",
        str(config_path),
        "--hardware",
        str(hardware_path),
        "--log_level",
        "ERROR",
    ]
    env = dict(os.environ)
    env["DISABLE_PROGRESS"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    batch_manifest_path = output_root / "batch_predict_manifest.json"
    assert batch_manifest_path.exists()
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))

    assert batch_manifest["summary"]["found"] == 2
    assert batch_manifest["summary"]["ok"] == 2
    assert batch_manifest["summary"]["failed"] == 0
    assert batch_manifest["summary"]["skipped"] == 0

    items = batch_manifest["items"]
    assert len(items) == 2
    sample_ids = {it["sample_id"] for it in items}
    assert len(sample_ids) == 2
    assert "my_scene" in sample_ids
    assert "my_scene_2" in sample_ids

    for item in items:
        assert item["status"] == "ok"
        assert Path(item["output_dir"]).exists()
        assert Path(item["predict_manifest"]).exists()
        assert Path(item["extent_prob"]).exists()
        assert Path(item["boundary_prob"]).exists()


def test_predict_input_data_batch_empty_dir(tmp_path: Path) -> None:
    module_root = Path(__file__).resolve().parents[1]
    script = module_root / "scripts" / "06_predict_input_data_batch.py"

    input_dir = tmp_path / "input_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_root = tmp_path / "direct_predict"

    cmd = [
        sys.executable,
        str(script),
        "--input_dir",
        str(input_dir),
        "--output_root",
        str(output_root),
        "--runs_root",
        str(tmp_path / "runs"),
        "--python",
        str(sys.executable),
        "--log_level",
        "ERROR",
    ]
    env = dict(os.environ)
    env["DISABLE_PROGRESS"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    batch_manifest_path = output_root / "batch_predict_manifest.json"
    assert batch_manifest_path.exists()
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    assert batch_manifest["summary"]["found"] == 0
    assert batch_manifest["summary"]["ok"] == 0
    assert batch_manifest["summary"]["failed"] == 0
    assert batch_manifest["summary"]["skipped"] == 0
