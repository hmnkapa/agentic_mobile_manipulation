"""Opt-in, real offline SDK/Transformers logging. No model or GPU training."""
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("MANIP_TEST_WANDB") != "1", reason="Set MANIP_TEST_WANDB=1 for real offline SDK checks")
@pytest.mark.parametrize("exit_code", [0, 1, 130])
def test_real_offline_transformers_metrics(tmp_path, exit_code):
    # Use a separate interpreter so SDK state and callback imports cannot affect
    # the CPU-only unit suite or subsequent real training processes.
    program = r'''
import json, os, platform, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wandb
from manipulation_agent.training.artifacts import write_json
from manipulation_agent.training.config import load_config, config_fingerprint
from manipulation_agent.training.tracking import (tracking_plan, tracking_environment,
    start_tracking, log_checkpoint, finish_tracking)

root = Path(sys.argv[1])
exit_code = int(sys.argv[2])
config = load_config()
config['wandb']['mode'] = 'offline'
config['wandb']['name'] = 'synthetic-cpu-tracking-check'
config['wandb']['tags'] = ['synthetic', 'cpu', 'tracking-check']
plan = tracking_plan(config, root)
os.environ.update(tracking_environment(config, root, os.environ))
write_json(root / 'training_contract.json', {
    'config': config, 'config_fingerprint': config_fingerprint(config), 'wandb': plan,
    'package': {'fingerprint': 'synthetic-no-dataset'},
    'model': {'fingerprint': 'synthetic-no-model', 'revision': None},
    'sample_ids': [], 'expected_splits': {'train': 0, 'val': 0}, 'packages': {},
    'encoder_source': {}, 'trainer_source': {}, 'expected_start_step': 0})
write_json(root / 'environment.json', {'source': {'synthetic': True}, 'host': platform.node(),
    'python': sys.executable, 'gpu_inventory': []})
write_json(root / 'run.json', {'parent_run': None, 'validation_scope': 'synthetic_cpu_tracking_check'})
run = start_tracking(root)
try:
    from transformers import TrainerState, TrainerControl
    from transformers.integrations import WandbCallback
    callback = WandbCallback()
    args = SimpleNamespace(run_name=config['wandb']['name'], output_dir=str(root),
        logging_steps=1, to_dict=lambda: {'learning_rate': 0.0001, 'synthetic': True})
    state, control = TrainerState(), TrainerControl()
    with patch('wandb.log', wraps=wandb.log) as logged:
        callback.on_train_begin(args, state, control, model=SimpleNamespace())
        state.global_step = 1
        callback.on_log(args, state, control, logs={'loss': 0.5, 'learning_rate': 0.0001, 'grad_norm': 0.25})
        callback.on_log(args, state, control, logs={'eval_loss': 0.4})
        history = {key: value for call in logged.call_args_list for key, value in call.args[0].items()}
    log_checkpoint({'path': 'synthetic-no-checkpoint', 'global_step': 1, 'fingerprint': 'synthetic'})
    assert run.summary['checkpoint/fingerprint'] == 'synthetic'
    assert run.summary['validation/task_evaluation_performed'] is False
finally:
    finish_tracking(root, run, exit_code=exit_code)

record = json.loads((root / 'wandb.json').read_text())
assert record['mode'] == 'offline' and record['url'] is None
paths = list(Path(record['sync_dir']).glob('*.wandb'))
assert len(paths) == 1
payload = paths[0].read_bytes()
# The native callback sends actual metrics through the SDK (wraps, not a mock
# transport). Verify those keys and final state reached the offline event file.
for key in (*history, 'checkpoint/fingerprint', 'validation/task_evaluation_performed', 'lifecycle/status'):
    assert key.encode() in payload, key
assert history['train/loss'] == 0.5, history
assert history['eval/loss'] == 0.4, history
assert history['train/global_step'] == 1
assert history['train/learning_rate'] == 0.0001
assert history['train/grad_norm'] == 0.25
expected_status = 'trainer_completed' if exit_code == 0 else 'interrupted' if exit_code == 130 else 'failed'
assert record['status'] == expected_status and record['child_exit_code'] == exit_code
assert expected_status.encode() in payload
assert not list(root.rglob('*.safetensors'))
print(json.dumps({'validation_level': 'real_offline_sdk_synthetic_metrics',
    'wandb': record, 'history': history, 'event_bytes': len(payload)}))
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "offline", "RANK": "0",
           "XDG_CACHE_HOME": str(tmp_path / "cache")}
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path), str(exit_code)], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence["validation_level"] == "real_offline_sdk_synthetic_metrics"
    (tmp_path / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
