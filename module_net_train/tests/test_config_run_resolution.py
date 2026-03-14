from __future__ import annotations

from pathlib import Path

from net_train.config import resolve_run_train_config_path


def test_resolve_run_train_config_path_prefers_run_config(tmp_path: Path) -> None:
    cli_cfg = tmp_path / "train_config.yaml"
    cli_cfg.write_text("paths: {}\n", encoding="utf-8")

    run_dir = tmp_path / "run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = run_dir / "config_resolved.yaml"
    run_cfg.write_text("paths: {}\n", encoding="utf-8")

    path, from_run = resolve_run_train_config_path(cli_cfg, run_dir)
    assert path == run_cfg.resolve()
    assert from_run is True


def test_resolve_run_train_config_path_falls_back_to_cli(tmp_path: Path) -> None:
    cli_cfg = tmp_path / "train_config.yaml"
    cli_cfg.write_text("paths: {}\n", encoding="utf-8")
    run_dir = tmp_path / "run_2"
    run_dir.mkdir(parents=True, exist_ok=True)

    path, from_run = resolve_run_train_config_path(cli_cfg, run_dir)
    assert path == cli_cfg.resolve()
    assert from_run is False
