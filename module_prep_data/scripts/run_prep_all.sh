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
N_PATCHES=""
SEED=""
OVERWRITE="0"
RUN_PYTEST="0"

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --n) N_PATCHES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --overwrite) OVERWRITE="1"; shift 1 ;;
    --no-overwrite) OVERWRITE="0"; shift 1 ;;
    --pytest) RUN_PYTEST="1"; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--config <path>] [--n <patches>] [--seed <seed>] [--overwrite] [--pytest]"
      echo "Defaults:"
      echo "  --n    from patching.target_patches_per_dataset in config"
      echo "  --seed from split.seed in config"
      echo "Default behavior: append to existing prep_data (safe incremental update)."
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# Resolve config path before changing directory.
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(realpath "${CONFIG}")"
fi

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

# pick python
PY="${PROJ_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY=python3
  if ! command -v ${PY} >/dev/null 2>&1; then
    die "python3 not found"
  fi
fi

# sanity: config exists
[[ -f "${CONFIG}" ]] || die "Config not found: ${CONFIG}"

mapfile -t CFG_VALUES < <("${PY}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1]).resolve()
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

patching = cfg.get("patching", {}) or {}
split = cfg.get("split", {}) or {}
ratios = split.get("ratios", {}) or {}

print(int(patching.get("target_patches_per_dataset", 800)))
print(int(split.get("seed", 123)))
print(float(ratios.get("train", 0.80)))
print(float(ratios.get("validation", 0.10)))
print(float(ratios.get("test", 0.10)))
PY
)

CFG_N_PATCHES="${CFG_VALUES[0]}"
CFG_SEED="${CFG_VALUES[1]}"
CFG_TRAIN_RATIO="${CFG_VALUES[2]}"
CFG_VAL_RATIO="${CFG_VALUES[3]}"
CFG_TEST_RATIO="${CFG_VALUES[4]}"

EFFECTIVE_N_PATCHES="${N_PATCHES:-${CFG_N_PATCHES}}"
EFFECTIVE_SEED="${SEED:-${CFG_SEED}}"
NP_SRC=$([[ -n "${N_PATCHES}" ]] && echo "cli" || echo "config")
SEED_SRC=$([[ -n "${SEED}" ]] && echo "cli" || echo "config")

cd "${MOD_ROOT}"

step "Environment"
echo "module_prep_data: ${MOD_ROOT}"
echo "project_root:     ${PROJ_ROOT}"
echo "config:           ${CONFIG}"
echo "n_patches:        ${EFFECTIVE_N_PATCHES} (${NP_SRC})"
echo "seed:             ${EFFECTIVE_SEED} (${SEED_SRC})"
echo "split_ratios:     train=${CFG_TRAIN_RATIO} val=${CFG_VAL_RATIO} test=${CFG_TEST_RATIO} (config)"
echo "split_mode:       $([[ \"${OVERWRITE}\" == \"1\" ]] && echo \"overwrite\" || echo \"append\")"

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
${PY} scripts/03_make_patches.py --config "${CONFIG}" --n "${EFFECTIVE_N_PATCHES}" --seed "${EFFECTIVE_SEED}"
ok "03_make_patches finished"

# ---------- 04_split_dataset ----------
step "04_split_dataset.py (patches_all -> prep_data/*)"
PATCHES_ALL="${PROJ_ROOT}/output_data/module_prep_data_work/patches_all"
OUT_PREP="${PROJ_ROOT}/prep_data"

SPLIT_ARGS=( --patches_all "${PATCHES_ALL}" --out_prep_data "${OUT_PREP}" --seed "${EFFECTIVE_SEED}" --train "${CFG_TRAIN_RATIO}" --val "${CFG_VAL_RATIO}" --test "${CFG_TEST_RATIO}" )
if [[ "${OVERWRITE}" == "1" ]]; then
  SPLIT_ARGS+=( --overwrite )
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
  ${PY} -m pytest -q
  ok "pytest passed"
fi

step "DONE"
ok "module_prep_data pipeline completed successfully"
