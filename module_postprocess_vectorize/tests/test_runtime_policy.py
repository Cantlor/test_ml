from __future__ import annotations

from module_postprocess_vectorize.postprocess import runtime


def _base_config() -> dict:
    return {
        "gaussian_sigma_px": 1.2,
        "use_watershed": True,
        "sobel_weight": 0.2,
    }


def _base_memory_cfg() -> dict:
    return {
        "prob_dtype": "float16",
        "smooth_dtype": "auto",
        "smooth_dtype_large": "float16",
        "ram_budget_fraction": 0.30,
        "ram_guard_mb": 256,
        "min_ram_budget_mb": 256,
        "degrade_gaussian_pressure": 0.60,
        "disable_gaussian_pressure": 1.50,
        "disable_watershed_pressure": 0.80,
        "switch_fast_clean_pressure": 0.75,
        "auto_disable_watershed": True,
        "auto_disable_gaussian_large": True,
        "max_pixels_for_watershed": 1_000_000_000,
        "max_pixels_for_gaussian": 1_000_000_000,
        "clean_labels_mode": "auto",
        "clean_labels_fast_pixels_threshold": 100_000_000,
    }


def test_runtime_policy_switches_to_safer_mode_under_low_ram(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "detect_available_ram_bytes", lambda: 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr(runtime, "detect_cpu_count", lambda: 4)

    plan = runtime.build_runtime_policy(
        config=_base_config(),
        memory_cfg=_base_memory_cfg(),
        n_pixels=80_000_000,
        valid_pixels=70_000_000,
    )

    assert plan.use_watershed is False
    assert plan.clean_labels_mode == "fast"
    assert plan.gaussian_sigma_px_effective < 1.2
    assert plan.estimated_pressure > 1.0


def test_runtime_policy_keeps_quality_path_under_high_ram(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "detect_available_ram_bytes", lambda: 64 * 1024 * 1024 * 1024)
    monkeypatch.setattr(runtime, "detect_cpu_count", lambda: 16)

    plan = runtime.build_runtime_policy(
        config=_base_config(),
        memory_cfg=_base_memory_cfg(),
        n_pixels=8_000_000,
        valid_pixels=8_000_000,
    )

    assert plan.use_watershed is True
    assert plan.clean_labels_mode == "exact"
    assert abs(plan.gaussian_sigma_px_effective - 1.2) < 1e-9
    assert plan.estimated_pressure < 0.2
