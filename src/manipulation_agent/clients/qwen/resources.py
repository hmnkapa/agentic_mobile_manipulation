"""Use the existing physical-GPU lease namespace for the local model process."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ...batch_lifecycle import FileLease


def process_start(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
    except FileNotFoundError:
        return None


def owned_process_alive(pid, start, output=None):
    if not pid or not start or process_start(pid)!=start:
        return False
    proc=Path('/proc')/str(pid)
    try:
        if proc.stat().st_uid!=os.getuid():return False
        if (proc/'stat').read_text().rsplit(') ',1)[1].split()[0]=='Z':return False
        if output is not None and str(output) not in (proc/'cmdline').read_bytes().decode():return False
        return True
    except FileNotFoundError:
        return False


def gpu_admission(runtime):
    if runtime['device']=='cpu':return
    command=['nvidia-smi','--id='+runtime['gpu_uuid'],
             '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']
    values=subprocess.check_output(command,text=True,stderr=subprocess.PIPE,timeout=15).strip().split(',')
    if len(values)!=2 or values[0].strip()!=runtime['gpu_uuid']:
        raise ValueError('Inference GPU identity mismatch')
    weights=sum(p.stat().st_size for p in Path(runtime['model']).glob('*.safetensors'))
    if not weights:raise ValueError('Inference model has no local safetensors weights')
    if int(values[1].strip())*1024**2 < weights:
        raise RuntimeError('Inference GPU has less free memory than the base weights')


class InferenceLease(FileLease):
    def active(self, previous):
        if previous.get('released'):return False
        # Simulator leases can outlive the supervisor's flock. Only its own
        # unit cleanup/adoption may clear that existing remote ownership.
        if previous.get('unit') or previous.get('launch_pending'):return True
        output=previous.get('controller_output')
        if output:
            try:
                state=json.loads((Path(output)/'qwen_runtime.json').read_text())
            except (FileNotFoundError,ValueError):
                # A dead wrapper alone cannot prove that an unrecorded child
                # exited. Keep unknown ownership for explicit reconciliation.
                return True
            if owned_process_alive(state.get('pid'),state.get('pid_start'),output):return True
        return owned_process_alive(previous.get('controller_pid'),previous.get('pid_start'),output)

    def acquire(self, *, adopting=None):
        if not super().acquire():return False
        if self.active(self.previous_owner) and (not adopting or
                self.previous_owner.get('run_id')!=adopting or self.previous_owner.get('unit')):
            super().release()
            return False
        if adopting and self.active(self.previous_owner):
            self.owner.update(self.previous_owner)
        self.record(released=False)
        return True

    def release(self, *, cleared=False):
        super().release(cleared=cleared and not self.active(self.owner))


def acquire_inference_lease(directory,runtime,owner,*,adopting=None):
    if runtime['device']=='cpu':return None
    key=hashlib.sha256(runtime['gpu_uuid'].encode()).hexdigest()[:24]
    lease=InferenceLease(Path(directory)/('gpu-'+key+'.lock'),
                         {**owner,'host':os.uname().nodename,'gpu_uuid':runtime['gpu_uuid']})
    return lease if lease.acquire(adopting=adopting) else None
