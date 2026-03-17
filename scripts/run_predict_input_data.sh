#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Error: python executable not found" >&2
  exit 1
fi

INPUT_DIR="${PROJECT_ROOT}/input_data"
OUTPUT_ROOT="${PROJECT_ROOT}/output_data/module_net_train/direct_predict"
RUN_DIR=""
RUNS_ROOT="${PROJECT_ROOT}/output_data/module_net_train/runs"
CONFIG_PATH=""
HARDWARE_PATH="${PROJECT_ROOT}/module_net_train/configs/hardware_config.yaml"
CHECKPOINT_PATH=""
WITH_POSTPROCESS=0
POSTPROCESS_CONFIG="${PROJECT_ROOT}/module_postprocess_vectorize/configs/postprocess_config.yaml"
POSTPROCESS_PARAMS_OVERRIDE=""
OVERWRITE=0
FAIL_FAST=0
LOG_LEVEL="INFO"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [options]

Options:
  --input_dir PATH
  --output_root PATH
  --run_dir PATH
  --runs_root PATH
  --config PATH
  --hardware PATH
  --checkpoint PATH
  --with-postprocess
  --postprocess-config PATH
  --postprocess-params-override PATH
  --overwrite
  --fail-fast
  --log-level LEVEL
  -h, --help

Defaults:
  input_dir:    ${INPUT_DIR}
  output_root:  ${OUTPUT_ROOT}
  runs_root:    ${RUNS_ROOT}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_dir) INPUT_DIR="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run_dir) RUN_DIR="$2"; shift 2 ;;
    --runs_root) RUNS_ROOT="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --hardware) HARDWARE_PATH="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
    --with-postprocess) WITH_POSTPROCESS=1; shift ;;
    --postprocess-config) POSTPROCESS_CONFIG="$2"; shift 2 ;;
    --postprocess-params-override) POSTPROCESS_PARAMS_OVERRIDE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/module_net_train/scripts/06_predict_input_data_batch.py"
  --input_dir "${INPUT_DIR}"
  --output_root "${OUTPUT_ROOT}"
  --runs_root "${RUNS_ROOT}"
  --hardware "${HARDWARE_PATH}"
  --log_level "${LOG_LEVEL}"
)

if [[ -n "${RUN_DIR}" ]]; then
  CMD+=(--run_dir "${RUN_DIR}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  CMD+=(--config "${CONFIG_PATH}")
fi
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi
if [[ ${WITH_POSTPROCESS} -eq 1 ]]; then
  CMD+=(--with-postprocess --postprocess-config "${POSTPROCESS_CONFIG}")
  if [[ -n "${POSTPROCESS_PARAMS_OVERRIDE}" ]]; then
    CMD+=(--postprocess-params-override "${POSTPROCESS_PARAMS_OVERRIDE}")
  fi
fi
if [[ ${OVERWRITE} -eq 1 ]]; then
  CMD+=(--overwrite)
fi
if [[ ${FAIL_FAST} -eq 1 ]]; then
  CMD+=(--fail-fast)
fi

"${CMD[@]}"
