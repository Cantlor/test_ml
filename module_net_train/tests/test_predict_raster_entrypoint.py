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


def _make_test_raster(path: Path) -> None:
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
            arr = np.full((h, w), 1000 + band, dtype=np.uint32)
            arr[:2, :2] = np.uint32(65536)
            ds.write(arr, band)


def _make_run_dir(run_dir: Path, config_obj: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    config_resolved = run_dir / "config_resolved.yaml"
    _write_yaml(config_resolved, config_obj)

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


def test_predict_raster_entrypoint_writes_outputs(tmp_path: Path) -> None:
    module_root = Path(__file__).resolve().parents[1]
    script = module_root / "scripts" / "05_predict_raster.py"

    train_cfg_obj = {
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
    hw_cfg_obj = {
        "device": {"mode": "cpu", "prefer_cuda": False},
        "precision": {"mode": "fp32", "prefer_bf16": False, "allow_tf32": False},
        "autotune": {"enabled": False},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2},
        "runtime": {"warn_if_cpu": False},
    }

    config_path = tmp_path / "train_config.yaml"
    hardware_path = tmp_path / "hardware_config.yaml"
    _write_yaml(config_path, train_cfg_obj)
    _write_yaml(hardware_path, hw_cfg_obj)

    run_dir = tmp_path / "run_demo"
    _make_run_dir(run_dir, train_cfg_obj)

    raster_path = tmp_path / "input_raster.tif"
    _make_test_raster(raster_path)

    output_dir = tmp_path / "pred_out"
    cmd = [
        sys.executable,
        str(script),
        "--raster",
        str(raster_path),
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--hardware",
        str(hardware_path),
        "--sample_id",
        "demo_sample",
        "--log_level",
        "ERROR",
    ]
    env = dict(os.environ)
    env["DISABLE_PROGRESS"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    extent_path = output_dir / "extent_prob.tif"
    boundary_path = output_dir / "boundary_prob.tif"
    manifest_path = output_dir / "predict_manifest.json"
    assert extent_path.exists()
    assert boundary_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["aoi_raster"] == str(raster_path.resolve())
    assert manifest["sample_id"] == "demo_sample"
    assert manifest["config_used"] == str((run_dir / "config_resolved.yaml").resolve())
    assert Path(manifest["extent_prob"]).resolve() == extent_path.resolve()
    assert Path(manifest["boundary_prob"]).resolve() == boundary_path.resolve()


def test_predict_aoi_entrypoint_still_available() -> None:
    module_root = Path(__file__).resolve().parents[1]
    script = module_root / "scripts" / "03_predict_aoi.py"
    res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--run_dir" in res.stdout


def test_predict_raster_entrypoint_with_postprocess(tmp_path: Path) -> None:
    module_root = Path(__file__).resolve().parents[1]
    script = module_root / "scripts" / "05_predict_raster.py"
    post_cfg = module_root.parent / "module_postprocess_vectorize" / "configs" / "postprocess_config.yaml"

    train_cfg_obj = {
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
    hw_cfg_obj = {
        "device": {"mode": "cpu", "prefer_cuda": False},
        "precision": {"mode": "fp32", "prefer_bf16": False, "allow_tf32": False},
        "autotune": {"enabled": False},
        "dataloader": {"num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2},
        "runtime": {"warn_if_cpu": False},
    }

    config_path = tmp_path / "train_config.yaml"
    hardware_path = tmp_path / "hardware_config.yaml"
    _write_yaml(config_path, train_cfg_obj)
    _write_yaml(hardware_path, hw_cfg_obj)

    run_dir = tmp_path / "run_demo"
    _make_run_dir(run_dir, train_cfg_obj)

    raster_path = tmp_path / "input_raster.tif"
    _make_test_raster(raster_path)

    output_dir = tmp_path / "pred_out"
    post_out = tmp_path / "post_out"
    cmd = [
        sys.executable,
        str(script),
        "--raster",
        str(raster_path),
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--config",
        str(config_path),
        "--hardware",
        str(hardware_path),
        "--with-postprocess",
        "--postprocess-config",
        str(post_cfg),
        "--postprocess-output-dir",
        str(post_out),
        "--log_level",
        "ERROR",
    ]
    env = dict(os.environ)
    env["DISABLE_PROGRESS"] = "1"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    manifest = json.loads((output_dir / "predict_manifest.json").read_text(encoding="utf-8"))
    assert "postprocess" in manifest
    assert "fields_pred_gpkg" in manifest["postprocess"]
    assert Path(manifest["postprocess"]["fields_pred_gpkg"]).exists()
    assert (post_out / "postprocess_manifest.json").exists()
