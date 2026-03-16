from __future__ import annotations

from pathlib import Path

from prep.config import load_config
from prep.patching.budget import derive_auto_patch_budget_from_inputs
from prep.patching.manifest import build_dataset_summary


def test_config_parses_patch_budget_auto(tmp_path: Path):
    cfg_path = tmp_path / "prep_config.yaml"
    cfg_path.write_text(
        """
version: 1
paths:
  work_dir: ./work
datasets: []
patching:
  patch_size_px: 256
  patch_budget:
    mode: auto
    min_valid_patches_to_keep: 25
    min_patches_per_dataset: 10
    max_patches_per_dataset: 500
    capacity_overlap_factor: 0.4
    boundary_pixels_per_patch: 640
    valid_ratio_sample_windows: 11
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    pb = cfg.patching.patch_budget
    assert pb.mode == "auto"
    assert pb.min_valid_patches_to_keep == 25
    assert pb.min_patches_per_dataset == 10
    assert pb.max_patches_per_dataset == 500
    assert pb.capacity_overlap_factor == 0.4
    assert pb.boundary_pixels_per_patch == 640
    assert pb.valid_ratio_sample_windows == 11


def test_auto_budget_heuristic_produces_reasonable_targets():
    out = derive_auto_patch_budget_from_inputs(
        patch_size_px=100,
        train_crop_px=100,
        negatives_ratio=0.2,
        center_weight=0.6,
        boundary_weight=0.4,
        capacity_overlap_factor=0.5,
        boundary_pixels_per_patch=100.0,
        min_valid_patches_to_keep=10,
        min_patches_per_dataset=0,
        max_patches_per_dataset=0,
        raster_area_pixels=1_000_000.0,
        valid_area_ratio=0.8,
        field_area_pixels=300_000.0,
        boundary_length_pixels=5_000.0,
    )

    assert out["estimated_total_capacity"] == 62
    assert out["estimated_total_target"] == 62
    assert out["estimated_center_capacity"] == 30
    assert out["estimated_boundary_capacity"] == 50
    assert out["estimated_negative_capacity"] == 50
    assert out["effective_stride_px"] == 100.0
    assert out["skipped_due_to_low_capacity"] is False
    assert out["estimated_total_target"] == (
        out["estimated_center_target"] + out["estimated_boundary_target"] + out["estimated_negative_target"]
    )


def test_auto_budget_can_skip_low_capacity_dataset():
    out = derive_auto_patch_budget_from_inputs(
        patch_size_px=100,
        train_crop_px=100,
        negatives_ratio=0.2,
        center_weight=0.6,
        boundary_weight=0.4,
        capacity_overlap_factor=0.5,
        boundary_pixels_per_patch=100.0,
        min_valid_patches_to_keep=70,
        min_patches_per_dataset=0,
        max_patches_per_dataset=0,
        raster_area_pixels=1_000_000.0,
        valid_area_ratio=0.8,
        field_area_pixels=300_000.0,
        boundary_length_pixels=5_000.0,
    )

    assert out["estimated_total_capacity"] == 62
    assert out["estimated_total_target"] == 0
    assert out["skipped_due_to_low_capacity"] is True
    assert "min_valid_patches_to_keep" in str(out["skip_reason"])


def test_dataset_summary_contains_budget_fields():
    summary = build_dataset_summary(
        ds_name="ds",
        raster_path="/tmp/r.tif",
        vector_path="/tmp/v.gpkg",
        vector_layer=None,
        vector_id_field=None,
        field_id_source="orig_fid",
        written_total=10,
        written_center=4,
        written_boundary=3,
        written_negative=3,
        target_total=12,
        pos_target=9,
        center_target=5,
        boundary_target=4,
        neg_target=3,
        center_attempts=20,
        boundary_attempts=20,
        neg_attempts=20,
        rejects={"oob": 1},
        nodata_value=65536.0,
        nodata_rule="control-band",
        control_band_1based=1,
        patch_size_px=512,
        patch_budget_mode="auto",
        budget_info={
            "estimated_total_capacity": 14,
            "estimated_center_capacity": 6,
            "estimated_boundary_capacity": 4,
            "estimated_negative_capacity": 4,
            "estimated_total_target": 12,
            "estimated_center_target": 5,
            "estimated_boundary_target": 4,
            "estimated_negative_target": 3,
            "valid_area_pixels": 100000.0,
            "valid_area_ratio": 0.9,
            "field_area_pixels": 20000.0,
            "boundary_length_pixels": 1500.0,
            "capacity_limiters": ["boundary"],
            "capacity_bounds": {"boundary": 14.0},
        },
        skipped_due_to_low_capacity=False,
        skip_reason=None,
        shortfall_reasons=[{"type": "center", "reasons": ["mask_ratio_constraints"]}],
    )

    assert summary["patch_budget_mode"] == "auto"
    assert summary["estimated_total_capacity"] == 14
    assert summary["capacity_limiters"] == ["boundary"]
    assert summary["shortfall_reasons"]
