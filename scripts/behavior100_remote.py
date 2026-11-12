"""Host-side batch checks and launch, without importing the simulator for checks."""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def local_quota():
    """Avoid unrelated NFS quota RPCs; unavailable diagnostics do not stop a run."""
    try:
        result = command(['quota', '-w', '-v', '-l'])
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
        # Keep partial numeric output: an already reported exhausted quota must
        # still block admission even if another local filesystem timed out.
        return {'status': 'unavailable', 'scope': 'local_filesystems',
                'exit_code': None, 'error_type': type(exc).__name__,
                'stdout': text(getattr(exc, 'stdout', None)),
                'stderr': text(getattr(exc, 'stderr', None))}
    return {**result, 'status': 'completed', 'scope': 'local_filesystems'}


def manifest_task(manifest, index):
    """Resolve the benchmark identity, including in a selected-task manifest."""
    for row in manifest['tasks']:
        if row['index'] == index:
            return row
    raise ValueError(f'Task index {index} is absent from this manifest')


def quota_headroom(output, *, devices=None):
    """Read numeric `quota -w -v` block limits without guessing human units."""
    remaining=[]
    for line in output.splitlines():
        fields=line.split()
        if len(fields)<4 or not fields[0].startswith('/'):
            continue
        if devices is not None and fields[0] not in devices:
            continue
        try:
            used,soft,hard=(int(v.rstrip('*')) for v in fields[1:4])
        except ValueError:
            continue
        limit=min(v for v in (soft,hard) if v>0) if soft>0 or hard>0 else None
        if limit is not None:remaining.append(max(0,limit-used)*1024)
    return min(remaining) if remaining else None


def source_import_readiness(data_root):
    """Detect relocated editable installs without importing the GPU simulator."""
    source = Path(data_root) / 'src/BEHAVIOR-1K'
    packages = {'bddl': source / 'bddl3/bddl/__init__.py',
                'omnigibson': source / 'OmniGibson/omnigibson/__init__.py',
                'gello': source / 'joylo/gello/__init__.py'}
    checks = {}
    for name, expected in packages.items():
        try:
            spec = importlib.util.find_spec(name)
            origin = Path(spec.origin) if spec is not None and spec.origin else None
            matches = origin is not None and origin.is_file() and origin.resolve() == expected.resolve()
            checks[name] = {'status': 'passed' if matches else 'failed',
                            'origin': str(origin) if origin else None, 'expected': str(expected)}
        except (ImportError, ValueError) as exc:
            checks[name] = {'status': 'failed', 'error': str(exc), 'expected': str(expected)}
    return {'status': 'passed' if all(x['status'] == 'passed' for x in checks.values()) else 'failed',
            'packages': checks, 'simulator_imported': False}


def runtime_storage_readiness(paths):
    """Resolve symlinks so a roomy output mount cannot hide a full cache disk."""
    checks = []
    for value in dict.fromkeys(map(str, paths)):
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            free = shutil.disk_usage(resolved).free
            writable = resolved.is_dir() and os.access(resolved, os.W_OK | os.X_OK)
            checks.append({'path': value, 'resolved_path': str(resolved),
                           'free_bytes': free, 'minimum_free_bytes': 16 * 1024**3,
                           'writable': writable,
                           'status': 'passed' if writable and free > 16 * 1024**3 else 'blocked'})
        except OSError as exc:
            checks.append({'path': value, 'status': 'blocked', 'error': str(exc)})
    return {'status': 'passed' if all(c['status'] == 'passed' for c in checks) else 'blocked',
            'paths': checks}


def preflight(gpu, port, data_root, min_free_gpu_mib=0, memory_budget_gib=28, *, light=False,
              reserved_host_memory_gib=0, runtime_storage_paths=(), memory_policy=None,
              validated_driver_version=None, gpu_memory_admission=True):
    # A worker performs the full version/encoder check once. Resource polling
    # only rechecks quantities that can change while it waits for a GPU lease.
    encoding_check = None
    if not light:
        from manipulation_agent.video import ffmpeg_executable
        try:
            encoder = ffmpeg_executable()
            encoding_check = command([encoder, '-hide_banner', '-loglevel', 'error',
                '-f', 'lavfi', '-i', 'color=size=64x64:rate=1', '-frames:v', '1',
                '-c:v', 'libx264', '-threads', '1', '-f', 'null', '-'])
            encoding_check['executable'] = encoder
            encoding_check['sha256'] = hashlib.sha256(Path(encoder).read_bytes()).hexdigest()
        except Exception as exc:
            encoding_check = {'exit_code': -1, 'error': str(exc)}
    queries = {
        'gpus': command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,driver_version', '--format=csv,noheader,nounits']),
        'disk': command(['df', '-B1', str(data_root), os.getcwd()]),
        'quota': local_quota(),
    }
    if not light:
        queries['gpu_processes'] = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'])
        queries['video_encoder'] = encoding_check
    selected = next((s.split(',') for s in queries['gpus']['stdout'].splitlines() if s.split(',')[0].strip() == str(gpu)), None)
    if selected is None:
        raise RuntimeError('Selected physical GPU is unavailable')
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    cgroup = Path('/sys/fs/cgroup/user.slice') / f'user-{os.getuid()}.slice'
    cg = {k: (cgroup/k).read_text().strip() for k in ('memory.current', 'memory.max')}
    cg['memory_stat']={k:int(v) for k,v in (line.split() for line in (cgroup/'memory.stat').read_text().splitlines())}
    stats=cg['memory_stat']
    # Clean file cache and reclaimable slab are not live process working memory.
    # This is an estimate; the simulator still has its own hard cgroup limit.
    cg['reclaimable_estimate']=max(0,stats['file']-stats.get('shmem',0)-stats.get('file_dirty',0)-stats.get('file_writeback',0))+stats.get('slab_reclaimable',0)
    cg_free = None if cg['memory.max'] == 'max' else int(cg['memory.max'])-int(cg['memory.current'])+cg['reclaimable_estimate']
    with socket.socket() as sock:
        port_free = sock.connect_ex(('127.0.0.1', port)) != 0
    if gpu_memory_admission and min_free_gpu_mib and min_free_gpu_mib<12288:
        raise ValueError('Shared GPU mode requires at least 12 GiB free before each episode')
    if not 12<=memory_budget_gib<=28:
        raise ValueError('Simulator working-memory budget must be between 12 and 28 GiB')
    if not 0<=reserved_host_memory_gib<=1024:
        raise ValueError('Invalid shared host memory reservation')
    imports = source_import_readiness(data_root) if not light else None
    checks = {'validated_driver_floor':(tuple(map(int,selected[4].strip().split('.'))) >= (580,65,6)
                                        or selected[4].strip()==validated_driver_version),
              'host_memory_available_40GiB': int(mem['MemAvailable'].split()[0])*1024 > 40*1024**3,
              'cgroup_reclaimable_headroom': cg_free is None or cg_free > memory_budget_gib*1024**3,
              'data_disk_free_40GiB': shutil.disk_usage(data_root).free > 40*1024**3,
              'bridge_port_free': port_free}
    if gpu_memory_admission:
        checks['gpu_capacity'] = ((int(selected[3])-int(selected[2])>=min_free_gpu_mib)
                                  if min_free_gpu_mib else int(selected[2])<1024)
    host_available=int(mem['MemAvailable'].split()[0])*1024
    checks['host_memory_including_peer_reservations']=host_available>(40+reserved_host_memory_gib)*1024**3
    adaptive = None
    if memory_policy is not None:
        from manipulation_agent.memory_admission import host_memory_admission
        adaptive = host_memory_admission(memory_policy,
            account_current=int(cg['memory.current']), account_stats=stats,
            account_limit=None if cg['memory.max']=='max' else int(cg['memory.max']))
        checks['adaptive_account_memory'] = adaptive['status']=='passed'
        required = adaptive['candidate_reserve_bytes']
        checks['cgroup_reclaimable_headroom'] = cg_free is None or cg_free > required
        checks.pop('host_memory_available_40GiB')
        checks['host_working_memory_headroom'] = host_available > required + 2*1024**3
        checks['host_memory_including_peer_reservations'] = (
            host_available > required + 2*1024**3 + adaptive['peer_growth_reserve_bytes'])
        # An abandoned startup must be adopted/cleaned before another starts.
        if memory_policy.get('profile') != 'throughput':
            checks['peer_initialization_complete'] = not any(
                peer.get('runtime_stage')=='starting' for peer in memory_policy['peers'])
    runtime_storage = runtime_storage_readiness(runtime_storage_paths)
    checks['runtime_storage_headroom_16GiB'] = runtime_storage['status'] == 'passed'
    if not light:
        checks.update(video_encoder=encoding_check['exit_code'] == 0,
                      pinned_source_imports=imports['status'] == 'passed')
    # A quota on /home must not veto an episode whose output and runtime writes
    # are on separate mounts. Keep the complete diagnostic and validate every
    # configured write filesystem. Fall back to all quotas if df is unavailable.
    quota_devices = None
    if runtime_storage_paths:
        queries['quota_write_filesystems'] = command(['df', '--output=source', str(data_root),
            os.getcwd(), *map(str, runtime_storage_paths)])
        if queries['quota_write_filesystems']['exit_code'] == 0:
            quota_devices = queries['quota_write_filesystems']['stdout'].splitlines()[1:]
            quota_devices = [device.strip() for device in quota_devices if device.strip()]
    quota_free=quota_headroom(queries['quota']['stdout'],devices=quota_devices)
    if quota_free is not None:
        checks['quota_headroom_2GiB']=quota_free>2*1024**3
    return {'at': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
            'interpreter': sys.executable, 'pid': os.getpid(), 'gpu_index': gpu,
            'gpu_uuid': selected[1].strip(), 'cgroup': cg, 'checks': checks, 'source_imports': imports,
            'driver_version':selected[4].strip(),
            'driver_validation_basis':'exact_renderer_validated_version' if selected[4].strip()==validated_driver_version else 'default_version_floor',
            'required_working_memory_gib':memory_budget_gib,
            'quota_free_bytes':quota_free,
            'quota_write_filesystems':quota_devices,
            'runtime_storage':runtime_storage,
            'adaptive_memory':adaptive,
            'host_available_bytes':host_available,'reserved_host_memory_gib':reserved_host_memory_gib,
            'driver_version':selected[4].strip(),'gpu_policy':{
                'memory_admission_enabled':gpu_memory_admission,
                'shared':bool(min_free_gpu_mib),'minimum_free_mib':min_free_gpu_mib},
            'validation_scope': 'dynamic_resources' if light else 'resources_and_static_runtime',
            'status': 'passed' if all(checks.values()) else 'blocked', 'queries': queries}


def assets(manifest, data_root):
    import yaml
    root = data_root/'data/omnigibson/2026-challenge-task-instances'
    config = root/'metadata/available_tasks.yaml'
    catalog = yaml.safe_load(config.read_text())
    rows = []
    for task in manifest['tasks']:
        custom = None
        if task.get('custom_task_spec'):
            custom_path = Path(task['custom_task_spec'])
            if hashlib.sha256(custom_path.read_bytes()).hexdigest() != task['custom_task_spec_sha256']:
                raise ValueError('Frozen custom task specification changed')
            custom = json.loads(custom_path.read_text())
            base = custom['base']
            name, instance = base['task'], base['instance']
        else:
            name, instance = task['task'], task['instance']
        scene = catalog[name][0]['scene_model']
        folder = root/'scene_test/public'/scene/'json'
        paths = {'scene_template': folder/f'{scene}_task_{name}_0_0_template-partial_rooms.json',
                 'instance_state': folder/f'{scene}_task_{name}_instances'/f'{scene}_task_{name}_0_{instance}_template-tro_state.json',
                 'bddl': data_root/'src/BEHAVIOR-1K/bddl3/bddl/activity_definitions'/name/'problem0.bddl'}
        row = {'task': task['task'], 'base_task': name, 'scene': scene,
               'scene_matches_catalog': scene == task['scene'], 'files': {}}
        for key, path in paths.items():
            row['files'][key] = {'path': str(path), 'exists': path.is_file(),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
        row['bddl_matches_manifest'] = row['files']['bddl']['sha256'] == task['bddl_sha256']
        from manipulation_agent.asset_preflight import inspect_scene_assets
        try:
            row['runtime_assets'] = inspect_scene_assets(paths['scene_template'], paths['instance_state'],
                data_root/'data/omnigibson/behavior-1k-assets', scene, task_name=name)
            if custom is not None:
                frozen = (custom_path.parent/custom['frozen_instance']['path']).resolve()
                digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
                if digest != custom['frozen_instance']['sha256'] or digest != task['custom_instance_sha256']:
                    raise ValueError('Custom joint initial scene changed')
                row['custom_scene_assets'] = inspect_scene_assets(frozen, paths['instance_state'],
                    data_root/'data/omnigibson/behavior-1k-assets', scene, task_name=name)
                row['custom_scene_sha256'] = digest
        except (OSError, ValueError, KeyError) as exc:
            row['runtime_assets'] = {'status':'failed','error':str(exc)}
        row['status'] = 'passed' if all(v['exists'] for v in row['files'].values()) and row['scene_matches_catalog'] and row['bddl_matches_manifest'] and row['runtime_assets']['status']=='passed' and row.get('custom_scene_assets',{'status':'passed'})['status']=='passed' else 'failed'
        rows.append(row)
    return {'status': 'passed' if all(r['status']=='passed' for r in rows) else 'failed',
            'available_tasks_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'tasks': rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode', choices=['preflight', 'assets', 'simulate'])
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--index',type=int)
    p.add_argument('--gpu',type=int); p.add_argument('--port',type=int); p.add_argument('--unit'); p.add_argument('--run-id')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--min-free-gpu-mib',type=int,default=0)
    p.add_argument('--skip-gpu-memory-admission',action='store_true',
                   help='Skip simulator VRAM capacity admission; keep GPU identity and other resource checks')
    p.add_argument('--memory-budget-gib',type=int,default=28)
    p.add_argument('--validated-driver-version', help='Exact driver version with recorded renderer validation')
    p.add_argument('--light', action='store_true', help='Only recheck dynamic worker resources')
    p.add_argument('--deadline-unix', type=float, help='Legacy absolute task deadline; new runs arm the execution clock after initialization')
    p.add_argument('--reserved-host-memory-gib',type=int,default=0)
    p.add_argument('--runtime-storage-path',type=Path,action='append',default=[])
    policy=p.add_mutually_exclusive_group()
    policy.add_argument('--memory-policy-json', help='Shared measured-peak admission snapshot')
    policy.add_argument('--memory-policy-base64', help='Base64 admission snapshot safe through GDB argv')
    a=p.parse_args(); manifest=json.loads(a.manifest.read_text())
    memory_policy=(json.loads(base64.b64decode(a.memory_policy_base64,validate=True))
                   if a.memory_policy_base64 else json.loads(a.memory_policy_json)
                   if a.memory_policy_json else None)
    if a.mode == 'assets':
        selected=manifest if a.index is None else {'tasks':[manifest_task(manifest,a.index)]}
        print(json.dumps(assets(selected,a.data_root))); return
    if a.mode == 'preflight':
        print(json.dumps(preflight(a.gpu,a.port,a.data_root,a.min_free_gpu_mib,a.memory_budget_gib,light=a.light,
                                  reserved_host_memory_gib=a.reserved_host_memory_gib,
                                  runtime_storage_paths=a.runtime_storage_path,
                                  memory_policy=memory_policy,
                                  validated_driver_version=a.validated_driver_version,
                                  gpu_memory_admission=not a.skip_gpu_memory_admission))); return
    row=manifest_task(manifest,a.index)
    if a.run_id:
        import re
        if not re.fullmatch(re.escape(row['run_id'].rsplit('_r',1)[0])+r'_r[1-9][0-9]*',a.run_id):
            raise ValueError('Attempt ID must preserve the frozen task identity')
        row['run_id']=a.run_id
    os.environ['MAS_UNIT']=a.unit
    if a.deadline_unix is not None:
        if not __import__('math').isfinite(a.deadline_unix) or a.deadline_unix <= 0:
            raise ValueError('Task deadline must be a finite epoch timestamp')
        os.environ['MAS_EPISODE_DEADLINE_UNIX'] = str(a.deadline_unix)
    result=preflight(a.gpu,a.port,a.data_root,a.min_free_gpu_mib,a.memory_budget_gib,light=a.light,
                     reserved_host_memory_gib=a.reserved_host_memory_gib,
                     runtime_storage_paths=a.runtime_storage_path,
                     memory_policy=memory_policy, validated_driver_version=a.validated_driver_version,
                     gpu_memory_admission=not a.skip_gpu_memory_admission)
    # Persist a second check inside the allocated unit, immediately before startup.
    check_path=a.manifest.parent/'preflight'/f"{row['run_id']}_in_unit.json"
    check_path.write_text(json.dumps(result,indent=2)+'\n')
    if result['status'] != 'passed': raise RuntimeError('Resource preflight blocked simulator startup')
    asset_check=assets({'tasks':[row]},a.data_root)
    asset_path=a.manifest.parent/'preflight'/f"{row['run_id']}_assets_in_unit.json"
    asset_path.write_text(json.dumps(asset_check,indent=2)+'\n')
    if asset_check['status'] != 'passed':
        raise RuntimeError('Task asset preflight blocked simulator startup; see '+str(asset_path))
    os.environ['MAS_GPU_UUID']=result['gpu_uuid']
    args=[sys.executable,'-m','manipulation_agent.vision_cli','--backend','omnigibson','--policy','serve',
          '--task',row['task'],'--instance',str(row['instance']),'--seed',str(row['seed']),
          '--instruction',row['instruction'],'--agent-profile',manifest['agent_profile'],'--record-video',
          '--port',str(a.port),'--controller',manifest.get('controller_client','codex_cli').removesuffix('_cli')+':'+manifest['model'],
          '--max-actions',str(manifest['max_actions']),'--max-sim-steps',str(manifest['max_sim_steps']),
          '--output',str(a.data_root/'runs'/row['run_id'])]
    args+=['--goal-skill',manifest.get('goal_skill','hierarchical')]
    if row.get('custom_task_spec'):
        args+=['--custom-task',row['custom_task_spec']]
    if manifest.get('require_task_notes'): args.append('--require-task-notes')
    if manifest.get('episode_budget_policy')=='agent_steps':
        args+=['--max-agent-steps',str(manifest['max_agent_steps'])]
    elif manifest.get('episode_budget_policy')=='unbounded':
        args+=['--unbounded-episode']
    os.execv(sys.executable,args)


if __name__=='__main__': main()
