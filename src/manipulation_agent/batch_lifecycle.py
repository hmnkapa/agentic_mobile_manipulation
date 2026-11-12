"""Small shared lifecycle helpers for task outcomes and cooperative GPU leases."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


class ExecutionCapacity:
    """Reserve inherited executions before admitting work under a lower limit.

    Every inherited execution must remain supervised, including when their
    count already exceeds the new limit. Failed adoption admission retains its
    reservation until the same execution can be reconciled.
    """
    def __init__(self, limit, inherited=()):
        if type(limit) is not int or limit < 1:
            raise ValueError('Execution capacity must be a positive integer')
        self.limit = limit
        self.reserved = set(inherited)
        self.active = set()
        self.lock = threading.Lock()

    def acquire(self, key, *, adopting=False):
        with self.lock:
            if key in self.active:
                raise RuntimeError('Execution capacity already acquired: ' + key)
            if adopting:
                if key not in self.reserved:
                    raise RuntimeError('Inherited execution was not reserved: ' + key)
                self.reserved.remove(key)
            elif len(self.reserved) + len(self.active) >= self.limit:
                return False
            self.active.add(key)
            return True

    def release(self, key, *, preserve=False):
        with self.lock:
            self.active.remove(key)
            if preserve:
                self.reserved.add(key)


def preferred_gpu_worker(workers, states, host, gpu_csv, *, gpu_available=None):
    """Choose the most free eligible GPU from the existing admission snapshot.

    Shared lab jobs can grow after admission. Prefer available headroom rather
    than whichever worker thread wins the host lease; never claim another GPU
    here or touch external processes. Busy lanes and known occupied ports are
    excluded so preference cannot block independent lanes on that host. The
    optional lock probe also excludes GPUs leased by another coordinator.
    """
    memory = {}
    for line in gpu_csv.splitlines():
        fields = [v.strip() for v in line.split(',')]
        try:
            memory[int(fields[0])] = (int(fields[3]) - int(fields[2]), int(fields[2]))
        except (ValueError, IndexError):
            continue
    busy = {'starting', 'controller_starting', 'running', 'cleanup_pending', 'archive_pending', 'disabled', 'drained'}
    choices = []
    for worker in workers:
        if worker['ssh'][-1] != host or worker['gpu'] not in memory:
            continue
        state = states.get(worker['id'], {})
        if state.get('stage') in busy or state.get('checks', {}).get('bridge_port_free') is False:
            continue
        free, used = memory[worker['gpu']]
        minimum = worker.get('minimum_free_gpu_mib', 0)
        if worker.get('simulator_gpu_memory_admission',True) and (
                (minimum and free < minimum) or (not minimum and used >= 1024)):
            continue
        if gpu_available is not None and not gpu_available(worker):
            continue
        choices.append((free, -worker['gpu'], worker['id']))
    return max(choices)[2] if choices else None


def _gpu_lease_key(worker):
    device = worker.get('expected_gpu_uuid') or worker['ssh'][-1] + ':' + str(worker['gpu'])
    return hashlib.sha256(device.encode()).hexdigest()[:24]


def gpu_lease_available(directory, worker):
    """Probe only the GPU lock; never reserve a host slot or edit owner metadata.

    The caller already owns its GPU/host lease and must exempt that GPU. A peer
    blocked on the caller's host slot can still be preferred once we release it.
    This is a preference hint; real admission still acquires both leases.
    """
    path = Path(directory) / ('gpu-' + _gpu_lease_key(worker) + '.lock')
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream, fcntl.LOCK_UN)
        return True


def classify_episode_outcome(row):
    """Task outcome is independent of optional replay/media validation.

    Historical callers can supply ``controller_timeout`` from the original
    controller metadata. A missing score alone never proves a timeout.
    """
    if row.get('status') not in {'passed', 'failed', 'blocked'} and not row.get('execution_finished_at'):
        return None
    # Official task success is independent of controller shutdown and media.
    if row.get('task_success') is True:
        return 'success'
    if row.get('startup_timed_out') or row.get('termination_reason')=='startup_timeout':
        return 'failure'
    if row.get('budget_basis')=='model_execution_excludes_initialization' and not isinstance(row.get('execution_started_at_unix'),(int,float)):
        return 'failure'
    finished_in_time=(row.get('deadline_expired_at_finish') is False and
                      isinstance(row.get('execution_finished_at_unix'),(int,float)))
    if row.get('deadline_expired_at_finish') is True:
        return 'timeout'
    if not finished_in_time and (row.get('episode_outcome') == 'timeout' or row.get('episode_timed_out')
            or row.get('controller_timeout') or row.get('timeout')
            or row.get('termination_reason') == 'episode_deadline_exceeded'):
        return 'timeout'
    if row.get('episode_outcome') in {'success', 'failure'}:
        return row['episode_outcome']
    if (row.get('task_success') is True and (finished_in_time or
            (row.get('controller_status', 'passed') == 'passed' and not row.get('failure')))):
        return 'success'
    return 'failure'


def validate_manifest_coverage(rows, expected_count=None, expected_tasks=None):
    """Check identities only; never create outcomes for tasks that never ran."""
    if expected_count is not None and len(rows)!=expected_count:
        raise ValueError(f'Manifest coverage mismatch: expected {expected_count}, found {len(rows)}')
    indices=[row['index'] for row in rows]
    run_ids=[row['run_id'] for row in rows]
    if len(set(indices))!=len(indices) or len(set(run_ids))!=len(run_ids):
        raise ValueError('Manifest contains duplicate task indices or run IDs')
    if expected_tasks is not None:
        actual={row['index']:row['task'] for row in rows}
        if actual!=expected_tasks:raise ValueError('Saved records changed frozen manifest task identities')


def require_completed_scope(rows, *, draining=False):
    if draining:return
    pending=[row.get('run_id',str(row.get('index'))) for row in rows
             if row.get('status') not in {'passed','failed','blocked'}]
    if pending:
        raise RuntimeError('Incomplete batch coverage: '+str(len(pending))+
                           ' tasks remain nonterminal; resume existing records: '+', '.join(pending[:10]))


class FileLease:
    """An advisory cross-process flock; never unlink the shared lock inode."""
    def __init__(self, path, owner):
        self.path, self.owner = Path(path), owner
        self.metadata_path=self.path.with_suffix('.json')
        self.stream = None
        self.previous_owner = {}

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open('a+')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return False
        self.stream = stream
        stream.seek(0)
        try:
            raw=self.metadata_path.read_text() if self.metadata_path.exists() else stream.read()
            self.previous_owner = json.loads(raw) if raw.strip() else {}
        except (ValueError, OSError):
            self.release()
            raise RuntimeError('Lease owner metadata unreadable; refusing unknown slot ownership')
        return True

    def record(self, **extra):
        self.owner.update(extra)
        temporary=self.metadata_path.with_suffix('.json.tmp.'+str(os.getpid()))
        temporary.write_text(json.dumps({**self.owner, 'lease_pid': os.getpid(), 'updated_unix': time.time()}))
        temporary.replace(self.metadata_path)

    def release(self, *, cleared=False):
        if self.stream is None:
            return
        try:
            if cleared:self.record(released=True)
        finally:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class WorkerLease:
    """Reserve one account/host capacity slot and one physical GPU.

    All methods must share the directory and host slot sizing. Host capacity is
    independent of the number of candidate GPU workers configured by an arm.
    """
    def __init__(self, directory, worker, *, host_budget_gib, slot_memory_gib=28, owner=None,
                 adaptive_slot_count=None, owner_active=None):
        self.directory = Path(directory)
        self.worker = worker
        self.owner = dict(owner or {}, worker_id=worker['id'], host=worker['ssh'][-1],
                          gpu=worker['gpu'], gpu_uuid=worker.get('expected_gpu_uuid'),
                          hard_limit_gib=worker.get('memory_budget_gib', 28),
                          memory_admission_mode='observed_peak' if adaptive_slot_count is not None else 'full_reservation')
        if worker.get('memory_budget_gib', 28) > slot_memory_gib:
            raise ValueError('Worker memory exceeds shared host slot size')
        self.slot_count = math.floor(host_budget_gib / slot_memory_gib)
        if adaptive_slot_count is not None:
            if type(adaptive_slot_count) is not int or not 1 <= adaptive_slot_count <= 16:
                raise ValueError('Invalid adaptive host slot count')
            self.slot_count = adaptive_slot_count
        if self.slot_count < 1:
            raise ValueError('Host budget has no usable simulator slot')
        self.host_key = hashlib.sha256(worker['ssh'][-1].encode()).hexdigest()[:20]
        self.gpu_key = _gpu_lease_key(worker)
        self.host = self.gpu = None
        self.owner_active = owner_active
        self.blocked_owner = None

    def _owner_is_active(self, previous):
        if previous.get('released') or not previous.get('unit'):
            return False
        # An available flock does not prove that a remote simulator exited.
        # Without a checker, or with an inconclusive response, retain ownership.
        return self.owner_active is None or self.owner_active(previous) is not False

    def acquire(self, *, adopting=None):
        def occupied(previous):
            if (adopting and previous.get('run_id') == adopting.get('run_id')
                    and previous.get('unit') == adopting.get('simulator_unit')
                    and previous.get('unit')):
                return False
            active = self._owner_is_active(previous)
            if active:
                self.blocked_owner = previous
            return active

        self.blocked_owner = None
        gpu = FileLease(self.directory / ('gpu-' + self.gpu_key + '.lock'), self.owner.copy())
        if not gpu.acquire():
            return False
        try:
            if occupied(gpu.previous_owner):
                return False
            for slot in range(self.slot_count):
                host = FileLease(self.directory / f'host-{self.host_key}-{slot}.lock', self.owner.copy())
                if host.acquire():
                    try:
                        if occupied(host.previous_owner):
                            continue
                        self.gpu, self.host = gpu, host
                        return True
                    finally:
                        if self.host is not host:
                            host.release()
        finally:
            if self.gpu is not gpu:
                gpu.release()
        return False

    @property
    def previous_owners(self):
        return [lease.previous_owner for lease in (self.gpu, self.host) if lease]

    def record(self, **details):
        for lease in (self.gpu, self.host):
            if lease:
                lease.record(**details)

    def other_reserved_gib(self, slot_memory_gib=28):
        return slot_memory_gib * len(self.active_peers())

    def active_peers(self):
        """Count live remote owners even if their coordinator lost its flock.

        The startup gate protects new reservations. Probe locks are released
        without changing metadata, including when owner inspection fails.
        """
        peers = []
        for path in self.directory.glob(f'host-{self.host_key}-*.lock'):
            if self.host and path == self.host.path:
                continue
            probe = FileLease(path, {})
            if not probe.acquire():
                try:
                    peer = json.loads(path.with_suffix('.json').read_text())
                except (OSError, ValueError):
                    peer = {}  # Unknown held slot retains a full reservation.
                peers.append(peer)
            else:
                try:
                    if self._owner_is_active(probe.previous_owner):
                        peers.append(probe.previous_owner)
                finally:
                    probe.release()
        return peers

    def release(self, *, cleared=False):
        try:
            if self.gpu:self.gpu.release(cleared=cleared)
        finally:
            try:
                if self.host:self.host.release(cleared=cleared)
            finally:self.gpu=self.host=None
