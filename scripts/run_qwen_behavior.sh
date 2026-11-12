#!/usr/bin/env bash
# Local Qwen weights, original BEHAVIOR supervisor and native MCP task loop.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# ==================== LOCAL CONFIGURATION: EDIT HERE ====================
# Keep one assignment uncommented per group. To switch, comment out the active
# assignment and uncomment the alternative. The default site runs both model and
# simulator on this machine; --prepare-only checks configuration without launch.
# CLI options take precedence. Existing QWEN_* / BEHAVIOR_CONTROLLER_PYTHON
# environment variables also override their corresponding settings below.

# 1. Controller Python: original project core environment for MCP and supervision.
controller_python="/home/liqi/agentic_mobile_manipulation/.venv/bin/python"
# controller_python="/tmp/mas-qwen-native-20261107-7_fq5k15/env/bin/python"  # Isolated environment with core and inference dependencies

# 2. Qwen Python: existing isolated online environment referencing the training
# environment's ML packages and the core environment's MCP packages.
# This /tmp environment may be cleaned up; recreate it and update this path if needed.
# /mnt/data/home/liqi/manipulation_sft/env/bin/python lacks MCP and cannot replace it directly.
qwen_python="/tmp/mas-qwen-native-20261107-7_fq5k15/env/bin/python"

# 3. Base model: complete local Qwen3.5-4B snapshot; keep this same base for LoRA.
model="/mnt/data/home/liqi/manipulation_sft/models/Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# 4. Weights: empty means base only. All three alternatives are existing PEFT checkpoints.
# Enable exactly one explicit LoRA path. No automatic latest-checkpoint selection
# or training recipe inheritance is performed.
lora=""
# lora="/mnt/data/home/liqi/manipulation_sft/runs/qwen35_4b_5090_resume_20261106_005600/swift/checkpoint-1122"
# lora="/mnt/data/home/liqi/manipulation_sft/runs/qwen35_4b_5090_resume_20261106_005600/swift/checkpoint-1120"
# lora="/mnt/data/home/liqi/manipulation_sft/runs/qwen35_4b_5090_resume_20261106_005600/swift/checkpoint-720"

# 5. Inference GPU: physical UUIDs of the two local 4090s, taken from existing
# records; they do not indicate current availability. The model runs on this
# controller host; the default local simulator worker uses the other physical GPU.
gpu_uuid="GPU-2b8f760b-8426-6048-d6a7-ea84765c5802"  # Local GPU 0
# gpu_uuid="GPU-eb6d2cfb-d887-6805-74e8-fa2e37a7e02f"  # Local GPU 1

# 6. Local runner: ZJU3DV-V02, local simulator Python/assets/cache and GPU 1.
# The original 100-task manifest is copied under operations/qwen_local_runtime/;
# the template uses one shared local lock directory for all new local runs.
runner_template="$repo_root/configs/inference/behavior_local.json"
# Remote alternatives require matching --worker IDs and the original SSH access.
# runner_template="/mnt/nas/home--xujingyi/agentic_mobile_manip/operations/executor_full100_restart_20261025/batch/config.json"
# runner_template="/mnt/nas/home--xujingyi/agentic_mobile_manip/operations/executor_full100_20261025/batch/config.json"
# runner_template="/mnt/nas/home--xujingyi/agentic_mobile_manip/operations/executor_full100_20261024b/batch/config.json"

# Empty follows the selected runner's batch/manifest.json, keeping both selections aligned.
manifest=""
# manifest="/mnt/nas/home--xujingyi/agentic_mobile_manip/operations/executor_full100_restart_20261025/batch/manifest.json"

# 7. Simulator workers: the default template contains only local-gpu1.
# Select a matching remote runner_template before enabling a remote alternative.
worker_ids=("local-gpu1")  # This machine, physical GPU 1
# worker_ids=("s134-gpu1")  # 134: xujingyi@10.130.136.134, GPU 1
# worker_ids=("s134-gpu4")  # 134, GPU 4
# worker_ids=("s115-gpu0")  # 115: xujingyi@10.130.140.115, GPU 0
# worker_ids=("s112-gpu0")  # 112: xujingyi@10.76.2.112, GPU 0
# worker_ids=("s118-gpu5")  # 118: xujingyi@10.130.140.118, GPU 5
# worker_ids=("s134-gpu1" "s134-gpu4")  # Multiple candidates; one inference GPU still admits episodes serially through existing leases

# 8. Tasks: an empty array defaults to radio. The default manifest uses instance
# 301 and seed 0; original task instructions are preserved.
tasks=()
# tasks=("turning_on_radio" "clean_a_keyboard")  # Select only these explicit tasks

all_tasks=false
# all_tasks=true  # Select the full manifest; keep tasks=() above when enabled

limit=""
# limit="10"  # Select the first 10 tasks; empty keeps every selected task
# Episodes remain unbounded. This selects tasks, without action, turn, token or time caps.

# 9. Output: use the local NVMe filesystem, outside Git. /mnt/data currently
# lacks the original runner's required 40 GiB disk headroom.
output_root="/home/liqi/agentic_mobile_manipulation_runs/qwen_behavior"
output=""
# output="/home/liqi/agentic_mobile_manipulation_runs/qwen_behavior/radio-base-001"  # Explicit directory; must not already exist
# output="/home/liqi/agentic_mobile_manipulation_runs/qwen_behavior/radio-lora-1122-001"

# 10. Execute / prepare only: execution is the default; uncommitted source
# changes are still rejected by the existing frozen-source requirement.
prepare_only=false
# prepare_only=true  # Write configs only; no model or simulator launch. CLI: --prepare-only

# Use --resume DIR for an existing run, retaining weights and completed/failed attempts.
resume=""
# ==================== END LOCAL CONFIGURATION ====================

controller_python="${BEHAVIOR_CONTROLLER_PYTHON:-$controller_python}"
qwen_python="${QWEN_PYTHON:-$qwen_python}"
model="${QWEN_MODEL:-$model}"
gpu_uuid="${QWEN_GPU_UUID:-$gpu_uuid}"
new_options=false
selection_cli=""
worker_cli_selected=false

usage() {
    cat <<'USAGE'
Usage: run_qwen_behavior.sh [options]

Local paths and alternatives are filled in the commented configuration block
at the top of this file. Running without arguments uses that configuration.
Create an independent Qwen BEHAVIOR run from an existing site configuration
and task manifest, then execute it through the original batch supervisor.
The default task is turning_on_radio; instance, seed and instruction come
unchanged from the source manifest. No model, simulator or tests are run by
--prepare-only.

New run:
  --runner-template FILE  Override configured BEHAVIOR runner JSON (SSH, assets, simulator,
                          resource admission and recording settings are reused)
  --manifest FILE         Source task manifest (default: TEMPLATE.batch/manifest.json)
  --qwen-python PATH      Separate inference Python (or QWEN_PYTHON)
  --model DIR             Local Qwen3.5-4B base weights (or QWEN_MODEL)
  --gpu-uuid UUID         Physical inference GPU UUID (or QWEN_GPU_UUID)
  --lora DIR              Override configured PEFT checkpoint; empty lora= means base
  --output DIR            New run directory outside Git; default: configured output_root/
                          base-or-lora timestamp directory; must not already exist
  --task NAME             Select a task; repeat for multiple tasks
  --worker ID             Override configured simulator lanes; repeat to select workers
  --all                   Select every task in the source manifest
  --limit N               Keep only the first N selected tasks, in manifest order
  --prepare-only          Write configs and print the launch command; do not launch

Existing run:
  --resume DIR            Resume DIR/runner.json without rewriting its configs,
                          task identities, weights selection or attempt records

Common:
  --controller-python PATH  Core/MCP Python (or BEHAVIOR_CONTROLLER_PYTHON;
                            default: repository .venv/bin/python)
  -h, --help                Show this help

Examples using the filled local configuration:
  scripts/run_qwen_behavior.sh --prepare-only
  scripts/run_qwen_behavior.sh
  scripts/run_qwen_behavior.sh --all --limit 10
  # For LoRA, uncomment one lora= line at the top, or pass --lora DIR.
  # Resume using --resume /absolute/path/to/an/already-created/run.

The controller host loads Qwen locally. CUDA inference and simulation must use
different physical GPUs; simulator workers need expected_gpu_uuid in the template.
Execution requires a clean frozen checkout. Preparation alone permits local edits.
Episodes remain unbounded; --limit selects tasks, never actions or model tokens.
Execution and --resume print live task stages, model tool calls/results and
independent evaluation, with a waiting heartbeat every 30 seconds.
USAGE
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
require_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
}
print_command() { printf ' %q' "$@"; printf '\n'; }
require_frozen_source() {
    local changes
    changes="$(git -C "$repo_root" status --porcelain)" || die "Cannot inspect source version"
    [[ -z "$changes" ]] || die "Execution requires a clean frozen checkout. Commit the intended changes first; --prepare-only writes configs without launching."
}

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --all)
            [[ "$selection_cli" != tasks ]] || die "--all and --task are mutually exclusive"
            selection_cli=all; tasks=(); all_tasks=true; new_options=true; shift ;;
        --prepare-only) prepare_only=true; new_options=true; shift ;;
        --runner-template|--manifest|--qwen-python|--model|--gpu-uuid|--lora|--output|--task|--worker|--limit|--resume|--controller-python)
            require_value "$@"
            case "$1" in
                --runner-template) runner_template="$2"; new_options=true ;;
                --manifest) manifest="$2"; new_options=true ;;
                --qwen-python) qwen_python="$2"; new_options=true ;;
                --model) model="$2"; new_options=true ;;
                --gpu-uuid) gpu_uuid="$2"; new_options=true ;;
                --lora) lora="$2"; new_options=true ;;
                --output) output="$2"; new_options=true ;;
                --task)
                    [[ "$selection_cli" != all ]] || die "--all and --task are mutually exclusive"
                    if [[ "$selection_cli" != tasks ]]; then tasks=(); all_tasks=false; fi
                    selection_cli=tasks; tasks+=("$2"); new_options=true ;;
                --worker)
                    if ! $worker_cli_selected; then worker_ids=(); worker_cli_selected=true; fi
                    worker_ids+=("$2"); new_options=true ;;
                --limit) limit="$2"; new_options=true ;;
                --resume) resume="$2" ;;
                --controller-python) controller_python="$2" ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done

controller_python="$(command -v -- "$controller_python")" || die "Controller Python not found; use --controller-python"
if [[ "$controller_python" != /* ]]; then controller_python="$PWD/$controller_python"; fi
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -n "$resume" ]]; then
    if $new_options; then die "--resume only accepts --controller-python; existing run settings are retained"; fi
    [[ -f "$resume/runner.json" ]] || die "--resume must contain runner.json"
    runner_config="$(cd -- "$resume" && pwd -P)/runner.json"
    require_frozen_source
    printf 'Resume original supervisor:\n'
    print_command "$controller_python" "$repo_root/scripts/run_behavior100.py" --config "$runner_config" --console-progress
    cd -- "$repo_root"
    exec "$controller_python" -u "$repo_root/scripts/run_behavior100.py" --config "$runner_config" --console-progress
fi

if [[ -z "$output" ]]; then
    weights_arm=base
    if [[ -n "$lora" ]]; then weights_arm=lora; fi
    output="$output_root/qwen_${weights_arm}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
fi
[[ -n "$runner_template" ]] || die "A new run requires a runner template in the configuration block or --runner-template"
[[ -n "$qwen_python" && -n "$model" && -n "$gpu_uuid" ]] || die "Specify --qwen-python, --model and --gpu-uuid (or the QWEN_* environment variables)"
[[ -z "$limit" || "$limit" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive task count"
if $all_tasks && ((${#tasks[@]})); then die "--all and --task are mutually exclusive"; fi
if ! $prepare_only; then require_frozen_source; fi
qwen_python="$(command -v -- "$qwen_python")" || die "Inference Python not found; use --qwen-python"
output="$("$controller_python" -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$output")"

# Paths and strings travel as argv, never as executable shell or Python source.
# This step only reads input definitions and writes fresh run configuration.
"$controller_python" - "$repo_root" "$runner_template" "$manifest" "$output" \
    "$qwen_python" "$model" "$gpu_uuid" "$lora" "$all_tasks" "$limit" \
    "${tasks[@]}" -- "${worker_ids[@]}" <<'PY'
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo


def main():
    (repo_arg, template_arg, manifest_arg, output_arg, python_arg, model_arg,
     gpu_uuid, lora_arg, all_arg, limit_arg, *selection) = sys.argv[1:]
    separator = selection.index('--')
    names, worker_ids = selection[:separator], set(selection[separator + 1:])
    repo = Path(repo_arg).resolve()
    template_path = Path(template_arg).expanduser().resolve(strict=True)
    config = json.loads(template_path.read_text())
    source_batch = Path(config['batch']).expanduser()
    if not source_batch.is_absolute():
        raise ValueError('Template batch must be an absolute path, as used by the original supervisor')
    manifest_path = (Path(manifest_arg).expanduser() if manifest_arg else
                     source_batch / 'manifest.json').resolve(strict=True)
    source_manifest = json.loads(manifest_path.read_text())
    requested_output = Path(output_arg).expanduser()
    if requested_output.exists() or requested_output.is_symlink():
        raise ValueError('Output already exists; use --resume to retain its records')
    output = requested_output.resolve()
    model = Path(model_arg).expanduser().resolve(strict=True)
    lora = Path(lora_arg).expanduser().resolve(strict=True) if lora_arg else None
    # Keep venv Python symlinks intact so its own site-packages are used.
    python = Path(os.path.abspath(python_arg))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Inference Python must be executable')
    if not model.is_dir() or (lora is not None and not lora.is_dir()):
        raise ValueError('Model and optional LoRA paths must be directories')
    if lora is not None and not (lora / 'adapter_config.json').is_file():
        raise ValueError('--lora must name the exact PEFT checkpoint containing adapter_config.json')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+', gpu_uuid):
        raise ValueError('--gpu-uuid must be a physical NVIDIA GPU UUID')
    for protected in (repo, model, lora, source_batch.resolve()):
        if protected is not None and (output == protected or protected in output.parents):
            raise ValueError('Output must be outside Git, weights and the source batch')
    rows = source_manifest['tasks']
    selected_names = set(names or ['turning_on_radio'])
    if all_arg != 'true':
        missing = selected_names - {row['task'] for row in rows}
        if missing:
            raise ValueError('Tasks absent from source manifest: ' + ', '.join(sorted(missing)))
        rows = [row for row in rows if row['task'] in selected_names]
    if limit_arg:
        rows = rows[:int(limit_arg)]
    if not rows:
        raise ValueError('No tasks selected')
    if len({row['index'] for row in rows}) != len(rows):
        raise ValueError('Source manifest has duplicate task indices')

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    arm = 'lora' if lora is not None else 'base'
    tag = f'qwen_{arm}_{stamp}_{uuid.uuid4().hex[:8]}'
    # Recreate definitions, never historical outcomes, retries or process ownership.
    definition_fields = (
        'index', 'task', 'name', 'scene', 'instance', 'seed', 'split', 'instruction',
        'instruction_sha256', 'instruction_source', 'bddl_sha256',
        'goal_predicates_offline_only', 'candidate_capabilities_offline_only',
    )
    required_fields = ('index', 'task', 'name', 'scene', 'instance', 'seed', 'instruction',
                       'instruction_sha256', 'instruction_source', 'bddl_sha256')
    selected = []
    for original in rows:
        absent = set(required_fields) - original.keys()
        if absent:
            raise ValueError('Incomplete source task definition: ' + ', '.join(sorted(absent)))
        row = {key: copy.deepcopy(original[key]) for key in definition_fields if key in original}
        if not re.fullmatch(r'[A-Za-z0-9_]+', row['task']):
            raise ValueError('Invalid task identifier')
        if hashlib.sha256(row['instruction'].encode()).hexdigest() != row['instruction_sha256']:
            raise ValueError('Source instruction hash mismatch for ' + row['task'])
        row.update(status='planned', run_id=(f'mas_{tag}_{row["index"]:03d}_{row["task"]}'
                                            f'_i{row["instance"]}_s{row["seed"]}_r1'))
        selected.append(row)

    qwen_path = output / 'qwen.toml'
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        tasks=selected, controller_client='qwen_cli', model='Qwen/Qwen3.5-4B',
        model_reasoning_effort=None, qwen_config=str(qwen_path),
        episode_budget_policy='unbounded', max_agent_steps=None,
        model_timeout_seconds=None, simulator_runtime_max_seconds=None,
        official_submission_eligible=False,
    )
    # These legacy numeric fields remain inactive under the explicit policy,
    # matching prepare_behavior100.py and the original remote launcher.
    manifest.setdefault('max_actions', 50)
    manifest.setdefault('max_sim_steps', 20000)
    if 'instruction_counts' in manifest:
        from collections import Counter
        manifest['instruction_counts'] = dict(Counter(row['instruction_source'] for row in selected))

    # Retain the original shared lease namespace, including its implicit default.
    # A new output directory must not create an isolated competing GPU lock pool.
    config.setdefault('shared_lease_dir', str(source_batch.parent / 'resource_leases'))
    # The original runner belonged to another controller host. Local Qwen has
    # no provider API key requirement and uses this host's inherited PATH/env.
    # Leave spool unset to use the original supervisor's per-UID, per-batch
    # local default; never write a foreign controller's checkpoint directory.
    omitted_controller_fields = [key for key in
                                ('controller_path', 'required_controller_env', 'supervisor_spool_dir')
                                if key in config]
    for key in ('adopt_inflight', 'adopt_startup', 'comparison', 'source_commit',
                'controller_path', 'required_controller_env', 'supervisor_spool_dir'):
        config.pop(key, None)
    config.update(batch=str(output), batch_tag=tag, reports=str(output / 'reports'), source_path=str(repo),
                  expected_task_count=len(selected), qwen_config=str(qwen_path))
    for worker in config.get('workers') or []:
        worker.pop('qwen_config', None)
    workers = config.get('workers')
    if workers is None:
        workers = [{'id': f'gpu{gpu}', 'gpu': gpu} for gpu in config['gpus']]
    if worker_ids:
        missing = worker_ids - {worker['id'] for worker in workers}
        if missing:
            raise ValueError('Simulator workers absent from template: ' + ', '.join(sorted(missing)))
        workers = [worker for worker in workers if worker['id'] in worker_ids]
        config['workers'] = workers
    if not workers:
        raise ValueError('Runner template has no simulator workers')
    for worker in workers:
        simulator_uuid = worker.get('expected_gpu_uuid', config.get('expected_gpu_uuid'))
        if not simulator_uuid:
            raise ValueError('Each simulator worker needs its original expected_gpu_uuid')
        if simulator_uuid == gpu_uuid:
            raise ValueError('Simulator and inference GPUs must be distinct; select other lanes with --worker')
    lease_dir = Path(config['shared_lease_dir'])
    if not lease_dir.is_absolute():
        raise ValueError('Template shared_lease_dir must be absolute')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

    source = {'path': str(repo), 'commit': git('rev-parse', 'HEAD'),
              'dirty': bool(git('status', '--porcelain'))}
    if not source['dirty']:
        config['source_commit'] = source['commit']
    provenance = {
        'schema_version': 1, 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'launcher': 'scripts/run_qwen_behavior.sh', 'source_at_preparation': source,
        'runner_template': str(template_path),
        'runner_template_sha256': hashlib.sha256(template_path.read_bytes()).hexdigest(),
        'template_controller_fields_omitted': omitted_controller_fields,
        'source_manifest': str(manifest_path),
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'weights_arm': arm, 'model': str(model), 'adapter': str(lora) if lora else None,
        'inference_python': str(python), 'inference_gpu_uuid': gpu_uuid,
        'validation_level': 'configuration_only_no_execution',
        'task_origins': [{'index': row['index'], 'task': row['task'],
                          'source_run_id': row.get('run_id')} for row in rows],
    }
    qwen = {'python': str(python), 'model': str(model), 'device': 'cuda:0', 'gpu_uuid': gpu_uuid}
    if lora is not None:
        qwen['adapter'] = str(lora)

    # Everything above completes before reserving a fresh immutable output.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('manifest.json', manifest), ('runner.json', config),
                        ('launch.json', provenance)):
        with (output / name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    with qwen_path.open('x', encoding='utf-8') as stream:
        stream.write(''.join(f'{key} = {json.dumps(value, ensure_ascii=False)}\n'
                             for key, value in qwen.items()))
    print(f'Prepared {len(selected)} task(s), weights={arm}: {output}')
    for row in selected:
        print(f'  {row["task"]} / instance {row["instance"]} / seed {row["seed"]}')


try:
    main()
except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
    sys.exit('Error preparing Qwen run: ' + str(exc))
PY

runner_config="$(cd -- "$output" && pwd -P)/runner.json"
printf 'Original supervisor command:\n'
print_command "$controller_python" "$repo_root/scripts/run_behavior100.py" --config "$runner_config" --console-progress
if $prepare_only; then
    printf 'Configuration only; no model, simulator or tests started. Resume with:\n'
    print_command "$repo_root/scripts/run_qwen_behavior.sh" --controller-python "$controller_python" --resume "$(dirname -- "$runner_config")"
    exit 0
fi
cd -- "$repo_root"
exec "$controller_python" -u "$repo_root/scripts/run_behavior100.py" --config "$runner_config" --console-progress
