#!/usr/bin/env bash
# Launch managed training; CPU encoding preflight is an explicit separate mode.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_root="${SFT_WORK_ROOT:-/mnt/data/home/liqi/manipulation_sft}"
python_bin="${SFT_PYTHON:-}"
data="${SFT_DATA:-/mnt/nas/home--liqi/latest94_scene_split_real_tokens17408_20261104}"
model="${SFT_MODEL_DIR:-}"
recipe="$repo_root/configs/training/qwen35_4b_lora.toml"
launch="$repo_root/configs/training/launch/single_gpu.toml"
gpu="${CUDA_VISIBLE_DEVICES-0}"
max_length=17408
encoding_workers=1
preflight=""
output=""
wandb_mode=""
smoke=false
dry_run=false
preflight_only=false

usage() {
    cat <<'USAGE'
Usage: train_qwen35_4b.sh [options]

Run full Qwen3.5-4B LoRA training without a startup CPU encoding preflight.
Existing Python dependencies and local model weights are required.

  --gpu DEVICES          CUDA_VISIBLE_DEVICES (default: current value, else 0)
  --smoke                Use the fixed 16/8 sample smoke run; stop at step 2
  --dry-run              Validate and print the requested training/preflight plan;
                         do not encode, train, or create output directories
  --preflight-only       Run only the real encoding preflight (CPU)
  --preflight DIR        Optionally check a completed report; does not re-encode data
  --output DIR           New training output (default: WORK_ROOT/runs/unique-name)
  --work-root DIR        External working root (default: SFT_WORK_ROOT or local root)
  --python PATH          Python executable (default: SFT_PYTHON or WORK_ROOT/env/bin/python)
  --data DIR             Dataset package (default: SFT_DATA or current real-token package)
  --model DIR            Complete base model (default: SFT_MODEL_DIR or pinned local snapshot)
  --recipe FILE          Training recipe (default: configs/training/qwen35_4b_lora.toml)
  --launch FILE          Resource settings (default: configs/training/launch/single_gpu.toml)
  --max-length N         Context budget (default: 17408)
  --encoding-workers N   Workers for --preflight-only, 1..8 (default: 1)
  --wandb-mode MODE      online, offline, disabled (default: recipe/environment)
  -h, --help             Show this help

Examples:
  scripts/train_qwen35_4b.sh --gpu 0 --dry-run
  scripts/train_qwen35_4b.sh --gpu 0 --smoke --wandb-mode offline
  scripts/train_qwen35_4b.sh --gpu 0 --preflight /data/checks/encode_001
  scripts/train_qwen35_4b.sh --gpu 0,1 --launch configs/training/launch/ddp_2gpu.toml

Only --preflight-only creates a new directory under WORK_ROOT/preflights/.
Smoke resume uses manip-train train --resume-from CHECKPOINT --output NEW_DIR.
USAGE
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --smoke) smoke=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --preflight-only) preflight_only=true; shift ;;
        --gpu|--preflight|--output|--work-root|--python|--data|--model|--recipe|--launch|--max-length|--encoding-workers|--wandb-mode)
            require_value "$@"
            case "$1" in
                --gpu) gpu="$2" ;;
                --preflight) preflight="$2" ;;
                --output) output="$2" ;;
                --work-root) work_root="$2" ;;
                --python) python_bin="$2" ;;
                --data) data="$2" ;;
                --model) model="$2" ;;
                --recipe) recipe="$2" ;;
                --launch) launch="$2" ;;
                --max-length) max_length="$2" ;;
                --encoding-workers) encoding_workers="$2" ;;
                --wandb-mode) wandb_mode="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

[[ "$max_length" =~ ^[1-9][0-9]*$ ]] || die "--max-length must be positive"
[[ "$encoding_workers" =~ ^[1-8]$ ]] || die "--encoding-workers must be 1..8"
case "$wandb_mode" in ""|online|offline|disabled) ;; *) die "Invalid --wandb-mode" ;; esac
if $preflight_only && [[ -n "$preflight" ]]; then
    die "--preflight-only cannot be combined with --preflight"
fi
if ! $dry_run && ! $preflight_only; then
    [[ -n "$gpu" && "$gpu" != -1 ]] || die "Select a CUDA GPU with --gpu"
fi

python_bin="${python_bin:-$work_root/env/bin/python}"
python_bin="$(command -v -- "$python_bin")" || die "Python executable not found; use --python"
model="${model:-$work_root/models/Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
[[ -d "$data" && -d "$model" ]] || die "Dataset/model directory missing; use --data and --model"
[[ -f "$recipe" && -f "$launch" ]] || die "Recipe/launch file missing"

mode=full
if $smoke; then mode=smoke; fi
run_tag="qwen35_4b_${mode}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
output="${output:-$work_root/runs/$run_tag}"
if $preflight_only; then
    preflight="$work_root/preflights/$run_tag"
elif [[ -n "$preflight" ]]; then
    [[ -f "$preflight/summary.json" ]] || die "--preflight must contain summary.json"
fi

check_output() {
    local destination
    destination="$(realpath -m -- "$1")"
    case "$destination" in
        "$repo_root"|"$repo_root"/*) die "Generated outputs must be outside the repository: $destination" ;;
    esac
    [[ ! -e "$destination" && ! -L "$destination" ]] || die "Output already exists: $destination"
}
if $preflight_only; then check_output "$preflight"; fi
if ! $preflight_only; then check_output "$output"; fi

# Use this checkout even when launched from another directory or environment.
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset MAX_PIXELS MIN_PIXELS
cli=("$python_bin" -u -m manipulation_agent.training.cli)
common=(--data "$data" --model "$model" --recipe "$recipe" --max-length "$max_length")
print_command() { printf ' %q' "$@"; printf '\n'; }

if $preflight_only; then
    check_cmd=("${cli[@]}" preflight "${common[@]}" --encoding-workers "$encoding_workers" --output "$preflight")
    printf 'CPU preflight (CUDA_VISIBLE_DEVICES=""):\n'
    print_command "${check_cmd[@]}"
    if ! $dry_run; then CUDA_VISIBLE_DEVICES="" "${check_cmd[@]}"; fi
    exit 0
elif [[ -n "$preflight" ]]; then
    printf 'Using preflight: %s\n' "$preflight"
fi

train_cmd=("${cli[@]}" train "${common[@]}" --launch "$launch" --output "$output")
if [[ -n "$preflight" ]]; then train_cmd+=(--preflight "$preflight"); fi
if $smoke; then train_cmd+=(--smoke); fi
if [[ -n "$wandb_mode" ]]; then train_cmd+=(--wandb-mode "$wandb_mode"); fi
if $dry_run; then train_cmd+=(--dry-run); fi
export CUDA_VISIBLE_DEVICES="$gpu"
printf 'Training mode: %s; CUDA_VISIBLE_DEVICES=%s\n' "$mode" "$gpu"
print_command "${train_cmd[@]}"
exec "${train_cmd[@]}"
