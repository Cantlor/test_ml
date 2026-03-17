from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from net_train.utils.io import read_json, write_json
from net_train.utils.logging import setup_logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_python_bin() -> Path:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return venv_py
    return Path(sys.executable)


def _abs_no_resolve(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).absolute()


def _discover_rasters(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    rasters = [
        p.resolve()
        for p in sorted(input_dir.iterdir(), key=lambda x: x.name.lower())
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    return rasters


def _safe_sample_id(stem: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", stem.strip().lower())
    safe = safe.strip("_")
    return safe or "sample"


def _assign_unique_sample_ids(rasters: List[Path]) -> Dict[Path, str]:
    assigned: Dict[Path, str] = {}
    used: set[str] = set()
    for raster in rasters:
        base = _safe_sample_id(raster.stem)
        candidate = base
        n = 2
        while candidate in used:
            candidate = f"{base}_{n}"
            n += 1
        used.add(candidate)
        assigned[raster] = candidate
    return assigned


def _resolve_run_dir(run_dir_arg: str | None, runs_root: Path) -> Path:
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"--run_dir does not exist: {run_dir}")
        return run_dir

    if not runs_root.exists():
        raise FileNotFoundError(
            f"--runs_root does not exist and --run_dir was not provided: {runs_root}"
        )
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise RuntimeError(
            f"No run directories found under {runs_root}. Provide --run_dir explicitly."
        )
    return candidates[-1].resolve()


def _predict_one_raster(
    *,
    python_bin: Path,
    script_path: Path,
    raster_path: Path,
    run_dir: Path,
    output_dir: Path,
    sample_id: str,
    config: str | None,
    hardware: str | None,
    checkpoint: str | None,
    overwrite: bool,
    with_postprocess: bool,
    postprocess_config: str | None,
    postprocess_params_override: str | None,
    log_level: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(python_bin),
        str(script_path),
        "--raster",
        str(raster_path),
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--sample_id",
        str(sample_id),
        "--log_level",
        str(log_level),
    ]
    if config is not None:
        cmd += ["--config", str(Path(config).resolve())]
    if hardware is not None:
        cmd += ["--hardware", str(Path(hardware).resolve())]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(Path(checkpoint).resolve())]
    if overwrite:
        cmd.append("--overwrite")
    if with_postprocess:
        cmd.append("--with-postprocess")
    if postprocess_config is not None:
        cmd += ["--postprocess-config", str(Path(postprocess_config).resolve())]
    if postprocess_params_override is not None:
        cmd += ["--postprocess-params-override", str(Path(postprocess_params_override).resolve())]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch direct-raster prediction: read *.tif from input_data and run 05_predict_raster.py for each",
    )
    ap.add_argument("--input_dir", default=str(REPO_ROOT / "input_data"))
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--runs_root", default=str(REPO_ROOT / "output_data/module_net_train/runs"))
    ap.add_argument("--output_root", default=str(REPO_ROOT / "output_data/module_net_train/direct_predict"))
    ap.add_argument("--python", default=None, help="Python executable used to call 05_predict_raster.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hardware", default=str(ROOT / "configs/hardware_config.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--with-postprocess", action="store_true")
    ap.add_argument(
        "--postprocess-config",
        default=str(REPO_ROOT / "module_postprocess_vectorize/configs/postprocess_config.yaml"),
    )
    ap.add_argument("--postprocess-params-override", default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger("predict_input_data_batch", level=args.log_level)

    input_dir = Path(args.input_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_manifest_path = output_root / "batch_predict_manifest.json"

    rasters = _discover_rasters(input_dir)
    sample_ids = _assign_unique_sample_ids(rasters)

    manifest_obj: Dict[str, object] = {
        "schema_version": "1.0",
        "created_at": _utc_now_iso(),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "run_dir": None,
        "settings": {
            "overwrite": bool(args.overwrite),
            "with_postprocess": bool(args.with_postprocess),
            "log_level": str(args.log_level),
        },
        "summary": {
            "found": int(len(rasters)),
            "ok": 0,
            "failed": 0,
            "skipped": 0,
        },
        "items": [],
    }

    if not rasters:
        logger.warning(f"No .tif/.tiff files found in {input_dir}")
        write_json(batch_manifest_path, manifest_obj)
        logger.info(f"Batch manifest written: {batch_manifest_path}")
        return 0

    run_dir = _resolve_run_dir(args.run_dir, Path(args.runs_root).resolve())
    manifest_obj["run_dir"] = str(run_dir)
    logger.info(f"Using run_dir: {run_dir}")

    python_bin = _abs_no_resolve(args.python) if args.python is not None else _default_python_bin()
    if not python_bin.exists():
        raise FileNotFoundError(f"Python executable not found: {python_bin}")
    predict_script = (ROOT / "scripts" / "05_predict_raster.py").resolve()

    ok = failed = skipped = 0
    items: List[Dict[str, object]] = []

    for idx, raster in enumerate(rasters):
        sample_id = sample_ids[raster]
        sample_out_dir = (output_root / sample_id).resolve()
        sample_manifest_path = sample_out_dir / "predict_manifest.json"

        item: Dict[str, object] = {
            "input_raster": str(raster),
            "sample_id": sample_id,
            "output_dir": str(sample_out_dir),
            "status": "pending",
            "extent_prob": None,
            "boundary_prob": None,
            "predict_manifest": str(sample_manifest_path),
            "postprocess": None,
            "error": None,
        }

        if sample_manifest_path.exists() and not args.overwrite:
            item["status"] = "skipped"
            item["error"] = "outputs already exist (use --overwrite to regenerate)"
            try:
                sm = read_json(sample_manifest_path)
                item["extent_prob"] = sm.get("extent_prob")
                item["boundary_prob"] = sm.get("boundary_prob")
                item["postprocess"] = sm.get("postprocess")
            except Exception:
                pass
            skipped += 1
            items.append(item)
            continue

        result = _predict_one_raster(
            python_bin=python_bin,
            script_path=predict_script,
            raster_path=raster,
            run_dir=run_dir,
            output_dir=sample_out_dir,
            sample_id=sample_id,
            config=args.config,
            hardware=args.hardware,
            checkpoint=args.checkpoint,
            overwrite=bool(args.overwrite),
            with_postprocess=bool(args.with_postprocess),
            postprocess_config=args.postprocess_config,
            postprocess_params_override=args.postprocess_params_override,
            log_level=args.log_level,
        )

        if result.returncode == 0 and sample_manifest_path.exists():
            item["status"] = "ok"
            sm = read_json(sample_manifest_path)
            item["extent_prob"] = sm.get("extent_prob")
            item["boundary_prob"] = sm.get("boundary_prob")
            item["postprocess"] = sm.get("postprocess")
            ok += 1
        else:
            item["status"] = "failed"
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            item["error"] = stderr if stderr else stdout if stdout else f"process return code {result.returncode}"
            failed += 1
            if args.fail_fast:
                items.append(item)
                for remaining in rasters[idx + 1:]:
                    rem_sample_id = sample_ids[remaining]
                    rem_out_dir = (output_root / rem_sample_id).resolve()
                    items.append(
                        {
                            "input_raster": str(remaining),
                            "sample_id": rem_sample_id,
                            "output_dir": str(rem_out_dir),
                            "status": "skipped",
                            "extent_prob": None,
                            "boundary_prob": None,
                            "predict_manifest": str(rem_out_dir / "predict_manifest.json"),
                            "postprocess": None,
                            "error": "not processed due to --fail-fast after previous failure",
                        }
                    )
                    skipped += 1
                break

        items.append(item)

    manifest_obj["items"] = items
    manifest_obj["summary"] = {
        "found": int(len(rasters)),
        "ok": int(ok),
        "failed": int(failed),
        "skipped": int(skipped),
    }

    write_json(batch_manifest_path, manifest_obj)
    logger.info(
        "Batch completed: found=%s ok=%s failed=%s skipped=%s",
        len(rasters),
        ok,
        failed,
        skipped,
    )
    logger.info(f"Batch manifest written: {batch_manifest_path}")

    if failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
