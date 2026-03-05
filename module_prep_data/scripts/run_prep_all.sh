#!/usr/bin/env bash
set -euo pipefail

# ============================
# run_prep_all.sh
# Runs: 01_check_inputs -> 02_clip_to_aoi -> 03_make_patches -> 04_split_dataset -> smoke checks -> (optional) pytest
# ============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"      # module_prep_data/
PROJ_ROOT="$(cd "${MOD_ROOT}/.." && pwd)"       # my_project/

CONFIG="${MOD_ROOT}/prep_config.yaml"
N_PATCHES="80"
SEED="123"
OVERWRITE="1"
RUN_PYTEST="0"

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --n) N_PATCHES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no-overwrite) OVERWRITE="0"; shift 1 ;;
    --pytest) RUN_PYTEST="1"; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--config <path>] [--n <patches>] [--seed <seed>] [--no-overwrite] [--pytest]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# colors
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

step "Environment"
echo "module_prep_data: ${MOD_ROOT}"
echo "project_root:     ${PROJ_ROOT}"
echo "config:           ${CONFIG}"
echo "n_patches:        ${N_PATCHES}"
echo "seed:             ${SEED}"

# pick python
PY=python3
if ! command -v ${PY} >/dev/null 2>&1; then
  die "python3 not found"
fi

# sanity: config exists
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"

# ---------- 01_check_inputs ----------
step "01_check_inputs.py"
${PY} scripts/01_check_inputs.py --config "${CONFIG}"
ok "01_check_inputs finished"

# ---------- 02_clip_to_aoi ----------
# handle typo file name
CLIP_SCRIPT=""
if [[ -f "scripts/02_clip_to_aoi.py" ]]; then
  CLIP_SCRIPT="scripts/02_clip_to_aoi.py"
elif [[ -f "scripts/02_clop_to_aoi.py" ]]; then
  CLIP_SCRIPT="scripts/02_clop_to_aoi.py"
else
  warn "02_clip_to_aoi script not found (skipping AOI clip)"
fi

if [[ -n "${CLIP_SCRIPT}" ]]; then
  step "$(basename "${CLIP_SCRIPT}")"
  ${PY} "${CLIP_SCRIPT}" --config "${CONFIG}"
  ok "02_clip_to_aoi finished"
fi

# ---------- 03_make_patches ----------
step "03_make_patches.py (generate patches_all/*)"
${PY} scripts/03_make_patches.py --config "${CONFIG}" --n "${N_PATCHES}" --seed "${SEED}"
ok "03_make_patches finished"

# ---------- 04_split_dataset ----------
step "04_split_dataset.py (patches_all -> prep_data/*)"
PATCHES_ALL="${PROJ_ROOT}/output_data/module_prep_data_work/patches_all"
OUT_PREP="${PROJ_ROOT}/prep_data"

SPLIT_ARGS=( --patches_all "${PATCHES_ALL}" --out_prep_data "${OUT_PREP}" --seed "${SEED}" --train 0.80 --val 0.10 --test 0.10 )
if [[ "${OVERWRITE}" == "1" ]]; then
  SPLIT_ARGS+=( --overwrite )
else
  warn "Running split without --overwrite (may fail if prep_data not empty)"
fi

${PY} scripts/04_split_dataset.py "${SPLIT_ARGS[@]}"
ok "04_split_dataset finished"

# ---------- Smoke checks ----------
step "Smoke checks (valid_mask + NoData ignore policy)"
${PY} scripts/smoke_check_patches.py --patches_all "${PATCHES_ALL}" --k 12
ok "Smoke checks passed"

# ---------- Optional: pytest ----------
if [[ "${RUN_PYTEST}" == "1" ]]; then
  step "pytest"
  if command -v pytest >/dev/null 2>&1; then
    pytest -q
    ok "pytest passed"
  else
    warn "pytest not found; install it (pip install pytest) or add to requirements"
  fi
fi

step "DONE"
ok "module_prep_data pipeline completed successfully"