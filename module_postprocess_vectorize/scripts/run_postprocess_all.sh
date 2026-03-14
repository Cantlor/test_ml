#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

RUN_DIR=""
RUNS_ROOT="${PROJECT_ROOT}/output_data/module_net_train/runs"
PRED_ROOT=""
OUTPUT_ROOT=""
CONFIG_PATH="${MODULE_ROOT}/configs/postprocess_config.yaml"
PARAMS_OVERRIDE=""
GT_ROOT=""
GT_MODE="auto"
SEARCH_OUT=""
MAX_TRIALS=""
LOG_LEVEL="INFO"
MANIFEST_NAME=""
CONTINUE_ON_ERROR=0

AUTO_PREDICT_IF_MISSING=1
NET_CONFIG_PATH="${PROJECT_ROOT}/module_net_train/configs/train_config.yaml"
NET_HARDWARE_PATH="${PROJECT_ROOT}/module_net_train/configs/hardware_config.yaml"
PREDICT_DATASET_KEY=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Main options:
  --run_dir PATH                Specific run directory. If omitted, latest run is used automatically.
  --runs_root PATH              Runs root for auto run discovery (default: output_data/module_net_train/runs)
  --config PATH                 Postprocess config YAML
  --params_override PATH        Tuned params YAML (best_params.yaml)
  --gt_root PATH                GT root/file. If set (and no --params_override), search runs first.

Batch/search options:
  --pred_root PATH              Prediction root (default: <run_dir>/pred)
  --output_root PATH            Postprocess output root (default: <run_dir>/postprocess)
  --gt_mode MODE                auto|vector|raster (default: auto)
  --search_out PATH             Search output dir (default: <output_root>/search)
  --max_trials N                Limit number of grid-search trials
  --manifest_name NAME          Predict manifest filename (default from config)

Auto-predict options:
  --no-auto-predict             Do not auto-run module_net_train prediction when pred files are missing
  --net-config PATH             train_config for auto-predict
  --net-hardware PATH           hardware_config for auto-predict
  --dataset_key KEY             Optional dataset key for 03_predict_aoi.py

Other:
  --log_level LEVEL             INFO|DEBUG|WARNING|ERROR
  --continue_on_error           Continue batch run if one sample fails
  -h, --help                    Show this help

Examples:
  # Minimal: auto-pick latest run, auto-pick defaults, run postprocess
  $(basename "$0")

  # Full flow: search on GT + apply to latest run
  $(basename "$0") --gt_root /path/to/gt_vectors_or_gt_rasters --gt_mode vector

  # Explicit run directory + tuned params
  $(basename "$0") --run_dir output_data/module_net_train/runs/20260312_131232 --params_override /path/to/best_params.yaml
EOF
}

resolve_latest_run_dir() {
  local runs_root="$1"
  if [[ ! -d "${runs_root}" ]]; then
    echo ""; return 0
  fi

  local latest
  latest="$(find "${runs_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
  echo "${latest}"
}

count_prediction_pairs() {
  local pred_root="$1"
  local cfg_path="$2"

  PRED_ROOT_ENV="${pred_root}" CONFIG_PATH_ENV="${cfg_path}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

pred_root = Path(os.environ["PRED_ROOT_ENV"]).resolve()
cfg_path = Path(os.environ["CONFIG_PATH_ENV"]).resolve()

if not pred_root.exists():
    print(0)
    raise SystemExit(0)

cfg = {}
try:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    cfg = {}

extent_name = str(cfg.get("extent_prob_name", "extent_prob.tif"))
boundary_name = str(cfg.get("boundary_prob_name", "boundary_prob.tif"))

direct = pred_root / extent_name
extent_paths = [direct] if direct.exists() else list(pred_root.rglob(extent_name))
count = 0
for p in extent_paths:
    if (p.parent / boundary_name).exists():
        count += 1
print(count)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir)
      RUN_DIR="$2"; shift 2 ;;
    --runs_root)
      RUNS_ROOT="$2"; shift 2 ;;
    --pred_root)
      PRED_ROOT="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --params_override)
      PARAMS_OVERRIDE="$2"; shift 2 ;;
    --gt_root)
      GT_ROOT="$2"; shift 2 ;;
    --gt_mode)
      GT_MODE="$2"; shift 2 ;;
    --search_out)
      SEARCH_OUT="$2"; shift 2 ;;
    --max_trials)
      MAX_TRIALS="$2"; shift 2 ;;
    --log_level)
      LOG_LEVEL="$2"; shift 2 ;;
    --manifest_name)
      MANIFEST_NAME="$2"; shift 2 ;;
    --continue_on_error)
      CONTINUE_ON_ERROR=1; shift ;;
    --no-auto-predict)
      AUTO_PREDICT_IF_MISSING=0; shift ;;
    --net-config)
      NET_CONFIG_PATH="$2"; shift 2 ;;
    --net-hardware)
      NET_HARDWARE_PATH="$2"; shift 2 ;;
    --dataset_key)
      PREDICT_DATASET_KEY="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

CONFIG_PATH="$(realpath "${CONFIG_PATH}")"
[[ -f "${CONFIG_PATH}" ]] || { echo "Error: config not found: ${CONFIG_PATH}" >&2; exit 1; }

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(resolve_latest_run_dir "${RUNS_ROOT}")"
  if [[ -z "${RUN_DIR}" ]]; then
    echo "Error: no run directories found under ${RUNS_ROOT}" >&2
    exit 1
  fi
  echo "[postprocess] auto-selected latest run_dir: ${RUN_DIR}"
fi

RUN_DIR="$(realpath "${RUN_DIR}")"
[[ -d "${RUN_DIR}" ]] || { echo "Error: run_dir does not exist: ${RUN_DIR}" >&2; exit 1; }

if [[ -z "${PRED_ROOT}" ]]; then
  PRED_ROOT="${RUN_DIR}/pred"
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${RUN_DIR}/postprocess"
fi

mkdir -p "${OUTPUT_ROOT}"

# Auto-use tuned params from local search output when available.
if [[ -z "${PARAMS_OVERRIDE}" && -z "${GT_ROOT}" && -f "${OUTPUT_ROOT}/search/best_params.yaml" ]]; then
  PARAMS_OVERRIDE="${OUTPUT_ROOT}/search/best_params.yaml"
  echo "[postprocess] auto-selected params_override: ${PARAMS_OVERRIDE}"
fi

# Auto-run predict_aoi if prediction files are missing.
PRED_PAIR_COUNT="$(count_prediction_pairs "${PRED_ROOT}" "${CONFIG_PATH}")"
if [[ "${PRED_PAIR_COUNT}" == "0" ]]; then
  if [[ "${AUTO_PREDICT_IF_MISSING}" == "1" ]]; then
    echo "[postprocess] prediction rasters not found, running module_net_train/scripts/03_predict_aoi.py ..."

    NET_CONFIG_PATH="$(realpath "${NET_CONFIG_PATH}")"
    NET_HARDWARE_PATH="$(realpath "${NET_HARDWARE_PATH}")"

    [[ -f "${NET_CONFIG_PATH}" ]] || { echo "Error: net config not found: ${NET_CONFIG_PATH}" >&2; exit 1; }
    [[ -f "${NET_HARDWARE_PATH}" ]] || { echo "Error: net hardware config not found: ${NET_HARDWARE_PATH}" >&2; exit 1; }

    PREDICT_CMD=(
      "${PYTHON_BIN}" "${PROJECT_ROOT}/module_net_train/scripts/03_predict_aoi.py"
      --config "${NET_CONFIG_PATH}"
      --hardware "${NET_HARDWARE_PATH}"
      --run_dir "${RUN_DIR}"
      --log_level "${LOG_LEVEL}"
    )
    if [[ -n "${PREDICT_DATASET_KEY}" ]]; then
      PREDICT_CMD+=(--dataset_key "${PREDICT_DATASET_KEY}")
    fi
    "${PREDICT_CMD[@]}"

    PRED_PAIR_COUNT="$(count_prediction_pairs "${PRED_ROOT}" "${CONFIG_PATH}")"
    if [[ "${PRED_PAIR_COUNT}" == "0" ]]; then
      echo "Error: prediction files are still missing after auto-predict (pred_root=${PRED_ROOT})" >&2
      exit 1
    fi
  else
    echo "Error: prediction files not found in ${PRED_ROOT} and auto-predict is disabled" >&2
    exit 1
  fi
fi

# 1) Optional search stage
if [[ -n "${GT_ROOT}" && -z "${PARAMS_OVERRIDE}" ]]; then
  if [[ -z "${SEARCH_OUT}" ]]; then
    SEARCH_OUT="${OUTPUT_ROOT}/search"
  fi
  mkdir -p "${SEARCH_OUT}"

  echo "[postprocess] running parameter search..."
  SEARCH_CMD=(
    "${PYTHON_BIN}" "${MODULE_ROOT}/scripts/01_search_postprocess_params.py"
    --pred_root "${PRED_ROOT}"
    --gt_root "${GT_ROOT}"
    --gt_mode "${GT_MODE}"
    --config "${CONFIG_PATH}"
    --output_dir "${SEARCH_OUT}"
    --log_level "${LOG_LEVEL}"
  )

  if [[ -n "${MAX_TRIALS}" ]]; then
    SEARCH_CMD+=(--max_trials "${MAX_TRIALS}")
  fi
  if [[ -n "${MANIFEST_NAME}" ]]; then
    SEARCH_CMD+=(--manifest_name "${MANIFEST_NAME}")
  fi

  "${SEARCH_CMD[@]}"
  PARAMS_OVERRIDE="${SEARCH_OUT}/best_params.yaml"
fi

# 2) Batch postprocess stage
POST_CMD=(
  "${PYTHON_BIN}" "${MODULE_ROOT}/scripts/03_postprocess_run.py"
  --run_dir "${RUN_DIR}"
  --pred_root "${PRED_ROOT}"
  --output_root "${OUTPUT_ROOT}"
  --config "${CONFIG_PATH}"
  --log_level "${LOG_LEVEL}"
)

if [[ -n "${PARAMS_OVERRIDE}" ]]; then
  POST_CMD+=(--params_override "${PARAMS_OVERRIDE}")
fi
if [[ -n "${MANIFEST_NAME}" ]]; then
  POST_CMD+=(--manifest_name "${MANIFEST_NAME}")
fi
if [[ ${CONTINUE_ON_ERROR} -eq 1 ]]; then
  POST_CMD+=(--continue_on_error)
fi

echo "[postprocess] running batch postprocess..."
"${POST_CMD[@]}"

echo "[postprocess] done: ${OUTPUT_ROOT}"
