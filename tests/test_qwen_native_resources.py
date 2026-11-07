"""Inference ownership shares the simulator's physical-device lock namespace."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from manipulation_agent.batch_lifecycle import FileLease
from manipulation_agent.clients.qwen.resources import (
    InferenceLease, acquire_inference_lease, owned_process_alive, process_start,
)


def test_inference_and_simulator_physical_locks_conflict(tmp_path):
    uuid='GPU-1234-abcd'
    path=tmp_path/('gpu-'+hashlib.sha256(uuid.encode()).hexdigest()[:24]+'.lock')
    simulator=FileLease(path,{'unit':'owned-simulator.service'})
    assert simulator.acquire()
    simulator.record(released=False)
    runtime={'device':'cuda:0','gpu_uuid':uuid}
    assert acquire_inference_lease(tmp_path,runtime,{}) is None
    simulator.release()  # flock gone, remote simulator ownership still active
    assert acquire_inference_lease(tmp_path,runtime,{}) is None
    simulator=FileLease(path,{})
    assert simulator.acquire()
    simulator.release(cleared=True)
    model=acquire_inference_lease(tmp_path,runtime,{})
    assert model is not None
    model.release(cleared=True)


def test_living_model_survives_supervisor_restart_and_requires_exact_adoption(tmp_path):
    uuid='GPU-abcd-1234'
    runtime={'device':'cuda:0','gpu_uuid':uuid}
    lease=acquire_inference_lease(tmp_path,runtime,{})
    lease.record(run_id='original',controller_pid=os.getpid(),pid_start=process_start(os.getpid()))
    lease.release(cleared=True)
    assert not json.loads(lease.metadata_path.read_text())['released']
    assert acquire_inference_lease(tmp_path,runtime,{}) is None
    adopted=acquire_inference_lease(tmp_path,runtime,{},adopting='original')
    assert adopted is not None
    assert adopted.owner['controller_pid']==os.getpid()
    adopted.release()


def test_native_child_ownership_and_pid_reuse_are_checked(tmp_path):
    state={'pid':os.getpid(),'pid_start':process_start(os.getpid())}
    (tmp_path/'qwen_runtime.json').write_text(json.dumps(state))
    lease=InferenceLease(tmp_path/'gpu.lock',{})
    # Test ownership of a known current process with an injected argv check.
    with patch('manipulation_agent.clients.qwen.resources.owned_process_alive',return_value=True):
        assert lease.active({'controller_output':str(tmp_path)})
    assert not owned_process_alive(os.getpid(),'reused-pid')
    assert not owned_process_alive(os.getpid(),process_start(os.getpid()),'/another/episode')


def test_unrecorded_launch_retains_lease_instead_of_claiming_cleanup(tmp_path):
    lease=InferenceLease(tmp_path/'gpu.lock',{'launch_pending':True})
    assert lease.acquire()
    lease.release(cleared=True)
    assert not json.loads(lease.metadata_path.read_text())['released']


def test_missing_child_record_cannot_silently_release_gpu(tmp_path):
    lease=InferenceLease(tmp_path/'gpu.lock',{'controller_output':str(tmp_path/'unreadable-output')})
    assert lease.acquire()
    lease.release(cleared=True)
    assert not json.loads(lease.metadata_path.read_text())['released']
