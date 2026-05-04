#!/usr/bin/env bash
# Usage:
#   cd /localhome/local-pengfeig/pengfeig/IsaacLab
#   ./download_assemble_trocar_artifacts.sh
#   ./download_assemble_trocar_artifacts.sh --list
#   ./download_assemble_trocar_artifacts.sh --only trocar-assets,orca-dev-test-state
#   HF_TOKEN=<your-token> ./download_assemble_trocar_artifacts.sh --only orca-g1-visual
#
# What this downloads:
#   trocar-assets:
#     Omniverse USD assets used by assemble_trocar/g129_dex3_env_cfg.py.
#   orca-dev-test-state:
#     nvidia/orca-dev-test/rlinf/actor/model_state_dict -> models/gr00t/orca-dev-test.
#   orca-g1-visual:
#     nvidia/ORCA-Assemble-Trocar-GR00T-RL-Dev/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual.
#   sim6-gr00t-n15-50ksteps:
#     nvidia/ORCA-Assemble-Trocar-GR00T-RL-Dev/sim6_gr00t_n15_50ksteps.
#   sim2real-rlinf-20260501:
#     nvidia/sim2real_rlinf_dev_pg RLinf run, preserving global_step_16 and tensorboard only.
#
# Auth:
#   - Hugging Face private repos require HF_TOKEN in the environment or a cached
#     huggingface-cli login on the remote machine.
#   - Omniverse/Nucleus may prompt for device-flow auth the first time.
#
# Useful environment overrides:
#   CONDA_ENV=isaaclab_develop_6.0_pg
#   CONDA_SH=/localhome/local-pengfeig/miniconda3/etc/profile.d/conda.sh
#   MODEL_ROOT=/localhome/local-pengfeig/pengfeig/models/gr00t
#   HF_MAX_WORKERS=8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ISAACLAB_ROOT:-$SCRIPT_DIR}"
CONDA_SH="${CONDA_SH:-/localhome/local-pengfeig/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-isaaclab_develop_6.0_pg}"
MODEL_ROOT="${MODEL_ROOT:-/localhome/local-pengfeig/pengfeig/models/gr00t}"
RLINF_LOG_ROOT="${RLINF_LOG_ROOT:-$REPO_ROOT/scripts/reinforcement_learning/rlinf/logs/rlinf}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-8}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/download_logs}"

ALL_TARGETS=(
  trocar-assets
  orca-dev-test-state
  orca-g1-visual
  sim6-gr00t-n15-50ksteps
  sim2real-rlinf-20260501
)

TARGETS=()
VERIFY_ONLY=0

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; seen = 1; next } seen { exit }' "$0"
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

join_by_newline() {
  printf '%s\n' "$@"
}

add_targets_csv() {
  local csv="$1"
  local item
  IFS=',' read -r -a _items <<< "$csv"
  for item in "${_items[@]}"; do
    [[ -n "$item" ]] && TARGETS+=("$item")
  done
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --all)
        TARGETS=("${ALL_TARGETS[@]}")
        shift
        ;;
      --only|--target)
        [[ $# -ge 2 ]] || die "$1 requires a comma-separated target list"
        add_targets_csv "$2"
        shift 2
        ;;
      --verify-only)
        VERIFY_ONLY=1
        shift
        ;;
      --list)
        join_by_newline "${ALL_TARGETS[@]}"
        exit 0
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done

  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${ALL_TARGETS[@]}")
  fi
}

validate_target() {
  local target="$1"
  local known
  for known in "${ALL_TARGETS[@]}"; do
    [[ "$target" == "$known" ]] && return 0
  done
  die "unknown target '$target'. Use --list to see valid targets."
}

activate_conda() {
  [[ -f "$CONDA_SH" ]] || die "conda profile not found: $CONDA_SH"
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
}

ensure_python_deps() {
  python - <<'PY'
import importlib.util
import sys

missing = []
for module in ("huggingface_hub",):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))
print("python_env_ok")
PY
}

verify_nonempty_file() {
  local path="$1"
  [[ -s "$path" ]] || die "missing or empty file: $path"
}

verify_dir_has_files() {
  local path="$1"
  [[ -d "$path" ]] || die "missing directory: $path"
  find "$path" -type f -print -quit | grep -q . || die "directory contains no files: $path"
}

verify_trocar_assets() {
  verify_nonempty_file "$REPO_ROOT/assets/assemble_trocar/Assets/Trocar002/Trocar004_test.usd"
  verify_nonempty_file "$REPO_ROOT/assets/assemble_trocar/Assets/DisposableLaparoscopicPunctureDevice001/DisposableLaparoscopicPunctureDevice006_test.usd"
  du -sh "$REPO_ROOT/assets/assemble_trocar"
}

verify_orca_dev_test_state() {
  local dir="$MODEL_ROOT/orca-dev-test/rlinf/actor/model_state_dict"
  verify_nonempty_file "$dir/config.json"
  verify_nonempty_file "$dir/full_weights.pt"
  verify_nonempty_file "$dir/pytorch_model.bin"
  du -sh "$dir"
}

verify_orca_g1_visual() {
  local dir="$MODEL_ROOT/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual"
  verify_nonempty_file "$dir/config.json"
  verify_nonempty_file "$dir/model.safetensors.index.json"
  verify_nonempty_file "$dir/model-00001-of-00002.safetensors"
  verify_nonempty_file "$dir/model-00002-of-00002.safetensors"
  du -sh "$dir"
}

verify_sim6_gr00t() {
  local dir="$MODEL_ROOT/sim6_gr00t_n15_50ksteps"
  verify_dir_has_files "$dir"
  du -sh "$dir"
}

verify_sim2real_rlinf() {
  local dir="$RLINF_LOG_ROOT/20260501-15:10:53-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0"
  verify_nonempty_file "$dir/tensorboard/config.yaml"
  verify_nonempty_file "$dir/test_gr00t/checkpoints/global_step_16/actor/model_state_dict/full_weights.pt"
  du -sh "$dir"
}

hf_snapshot_download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local patterns=("$@")

  mkdir -p "$local_dir"
  python - "$repo_id" "$local_dir" "$HF_MAX_WORKERS" "${patterns[@]}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2])
max_workers = int(sys.argv[3])
patterns = sys.argv[4:]
token = os.environ.get("HF_TOKEN") or None

print(f"repo={repo_id}")
print(f"local_dir={local_dir}")
print("allow_patterns=" + ",".join(patterns))
try:
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=patterns,
        local_dir=str(local_dir),
        token=token,
        max_workers=max_workers,
    )
except Exception as exc:
    print(f"download_failed={type(exc).__name__}: {exc}", file=sys.stderr)
    print(
        "If this is a private repo, export HF_TOKEN or run huggingface-cli login "
        "on the remote machine.",
        file=sys.stderr,
    )
    raise
print(f"snapshot_dir={path}")
PY
}

copy_tree_contents() {
  local src="$1"
  local dst="$2"
  python - "$src" "$dst" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if not src.is_dir():
    raise SystemExit(f"missing source directory: {src}")
dst.mkdir(parents=True, exist_ok=True)
for child in src.iterdir():
    target = dst / child.name
    if child.is_dir():
        shutil.copytree(child, target, dirs_exist_ok=True)
    else:
        shutil.copy2(child, target)
print(f"copied {src} -> {dst}")
PY
}

download_trocar_assets() {
  local log_file="$LOG_DIR/trocar-assets.log"
  {
    log "target=trocar-assets"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
      python "$REPO_ROOT/scripts/download_trocar_assets.py" \
        --dest-root "$REPO_ROOT/assets/assemble_trocar"
    fi
    verify_trocar_assets
  } 2>&1 | tee "$log_file"
}

download_orca_dev_test_state() {
  local log_file="$LOG_DIR/orca-dev-test-state.log"
  {
    log "target=orca-dev-test-state"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
      hf_snapshot_download \
        "nvidia/orca-dev-test" \
        "$MODEL_ROOT/orca-dev-test" \
        "rlinf/actor/model_state_dict/**"
    fi
    verify_orca_dev_test_state
  } 2>&1 | tee "$log_file"
}

download_orca_g1_visual() {
  local name="g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual"
  local log_file="$LOG_DIR/orca-g1-visual.log"
  {
    log "target=orca-g1-visual"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
      hf_snapshot_download \
        "nvidia/ORCA-Assemble-Trocar-GR00T-RL-Dev" \
        "$MODEL_ROOT" \
        "$name/**"
    fi
    verify_orca_g1_visual
  } 2>&1 | tee "$log_file"
}

download_sim6_gr00t() {
  local name="sim6_gr00t_n15_50ksteps"
  local log_file="$LOG_DIR/sim6-gr00t-n15-50ksteps.log"
  {
    log "target=sim6-gr00t-n15-50ksteps"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
      hf_snapshot_download \
        "nvidia/ORCA-Assemble-Trocar-GR00T-RL-Dev" \
        "$MODEL_ROOT" \
        "$name/**"
    fi
    verify_sim6_gr00t
  } 2>&1 | tee "$log_file"
}

download_sim2real_rlinf() {
  local run_name="20260501-15:10:53-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0"
  local repo_prefix="rlinf_logs/$run_name"
  local staging="$RLINF_LOG_ROOT/.hf_download_staging_sim2real_20260501_151053"
  local dest="$RLINF_LOG_ROOT/$run_name"
  local log_file="$LOG_DIR/sim2real-rlinf-20260501.log"
  {
    log "target=sim2real-rlinf-20260501"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
      hf_snapshot_download \
        "nvidia/sim2real_rlinf_dev_pg" \
        "$staging" \
        "$repo_prefix/tensorboard/**" \
        "$repo_prefix/test_gr00t/checkpoints/global_step_16/**"
      copy_tree_contents "$staging/$repo_prefix" "$dest"
    fi
    verify_sim2real_rlinf
  } 2>&1 | tee "$log_file"
}

run_target() {
  local target="$1"
  validate_target "$target"
  case "$target" in
    trocar-assets) download_trocar_assets ;;
    orca-dev-test-state) download_orca_dev_test_state ;;
    orca-g1-visual) download_orca_g1_visual ;;
    sim6-gr00t-n15-50ksteps) download_sim6_gr00t ;;
    sim2real-rlinf-20260501) download_sim2real_rlinf ;;
    *) die "unhandled target: $target" ;;
  esac
}

main() {
  parse_args "$@"
  mkdir -p "$LOG_DIR" "$MODEL_ROOT" "$RLINF_LOG_ROOT"
  activate_conda
  ensure_python_deps
  log "repo_root=$REPO_ROOT"
  log "model_root=$MODEL_ROOT"
  log "rlinf_log_root=$RLINF_LOG_ROOT"
  log "targets=${TARGETS[*]}"
  [[ $VERIFY_ONLY -eq 1 ]] && log "mode=verify-only"

  local target
  for target in "${TARGETS[@]}"; do
    run_target "$target"
  done
  log "all_done"
}

main "$@"
