#!/usr/bin/env bash
set -euo pipefail

# ============================
# run_train_all.sh
# Runs: 01_check_prep_data -> 02_train -> 03_predict_aoi -> 04_eval
# ============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"       # module_net_train/
PROJ_ROOT="$(cd "${MOD_ROOT}/.." && pwd)"        # my_project/

CONFIG="${MOD_ROOT}/configs/train_config.yaml"
HARDWARE="${MOD_ROOT}/configs/hardware_config.yaml"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
LOG_LEVEL="INFO"

SKIP_CHECK="0"
SKIP_PREDICT="0"
SKIP_EVAL="0"
TRAIN_NO_INFER="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --hardware) HARDWARE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --log-level) LOG_LEVEL="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK="1"; shift 1 ;;
    --skip-predict) SKIP_PREDICT="1"; shift 1 ;;
    --skip-eval) SKIP_EVAL="1"; shift 1 ;;
    --train-infer) TRAIN_NO_INFER="0"; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--config <path>] [--hardware <path>] [--run-id <id>] [--log-level <level>] [--skip-check] [--skip-predict] [--skip-eval] [--train-infer]"
      echo
      echo "Defaults:"
      echo "  --config   module_net_train/configs/train_config.yaml"
      echo "  --hardware module_net_train/configs/hardware_config.yaml"
      echo "  --run-id   UTC timestamp"
      echo
      echo "Behavior:"
      echo "  by default 02_train is run with --no_infer, then 03_predict_aoi runs explicitly"
      echo "  use --train-infer if you want inference inside 02_train"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(realpath "${CONFIG}")"
fi
if [[ "${HARDWARE}" != /* ]]; then
  HARDWARE="$(realpath "${HARDWARE}")"
fi

BOLD="\033[1m"
GRN="\033[32m"
YEL="\033[33m"
RED="\033[31m"
CYN="\033[36m"
RST="\033[0m"

step () {
  echo -e "\n${BOLD}${CYN}==> $1${RST}"
}

ok () {
  echo -e "${GRN}[OK]${RST} $1"
}

warn () {
  echo -e "${YEL}[WARN]${RST} $1"
}

die () {
  echo -e "${RED}[FAIL]${RST} $1"
  exit 2
}

cd "${MOD_ROOT}"

PY="${PROJ_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY=python3
  if ! command -v ${PY} >/dev/null 2>&1; then
    die "python3 not found"
  fi
fi

[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"
[[ -f "${HARDWARE}" ]] || die "Hardware config not found: ${HARDWARE}"

RUN_DIR="$(CONFIG_PATH="${CONFIG}" RUN_ID_VALUE="${RUN_ID}" ${PY} - <<'PY'
import os
from pathlib import Path
from net_train.config import load_train_config

cfg = load_train_config(os.environ["CONFIG_PATH"])
runs_root = cfg.paths.get("runs_root", (cfg.project_root / "output_data/module_net_train/runs").resolve())
run_id = os.environ["RUN_ID_VALUE"]
print((Path(runs_root) / run_id).resolve())
PY
)"

if [[ -e "${RUN_DIR}" ]]; then
  die "Run directory already exists: ${RUN_DIR} (use another --run-id)"
fi

step "Environment"
echo "module_net_train: ${MOD_ROOT}"
echo "project_root:      ${PROJ_ROOT}"
echo "python:            ${PY}"
echo "config:            ${CONFIG}"
echo "hardware:          ${HARDWARE}"
echo "run_id:            ${RUN_ID}"
echo "run_dir:           ${RUN_DIR}"

if [[ "${SKIP_CHECK}" == "0" ]]; then
  step "01_check_prep_data.py"
  ${PY} scripts/01_check_prep_data.py \
    --config "${CONFIG}" \
    --out_json "${RUN_DIR}/prep_data_summary.json" \
    --log_level "${LOG_LEVEL}"
  ok "01_check_prep_data finished"
else
  warn "Skipping 01_check_prep_data"
fi

step "02_train.py"
TRAIN_ARGS=(
  --config "${CONFIG}"
  --hardware "${HARDWARE}"
  --run_id "${RUN_ID}"
  --log_level "${LOG_LEVEL}"
)
if [[ "${TRAIN_NO_INFER}" == "1" ]]; then
  TRAIN_ARGS+=( --no_infer )
fi
${PY} scripts/02_train.py "${TRAIN_ARGS[@]}"
ok "02_train finished"

if [[ "${SKIP_PREDICT}" == "0" ]]; then
  step "03_predict_aoi.py"
  ${PY} scripts/03_predict_aoi.py \
    --config "${CONFIG}" \
    --hardware "${HARDWARE}" \
    --run_dir "${RUN_DIR}" \
    --log_level "${LOG_LEVEL}"
  ok "03_predict_aoi finished"
else
  warn "Skipping 03_predict_aoi"
fi

if [[ "${SKIP_EVAL}" == "0" ]]; then
  step "04_eval.py"
  ${PY} scripts/04_eval.py \
    --config "${CONFIG}" \
    --hardware "${HARDWARE}" \
    --run_dir "${RUN_DIR}" \
    --log_level "${LOG_LEVEL}"
  ok "04_eval finished"
else
  warn "Skipping 04_eval"
fi

if [[ "${TRAIN_NO_INFER}" == "1" && "${SKIP_PREDICT}" == "1" ]]; then
  warn "Inference was not run (02_train with --no_infer and 03_predict skipped)."
fi

step "DONE"
ok "module_net_train pipeline completed: ${RUN_DIR}"
