"""Durable 100-task runner. One immutable attempt per row; never filter failures.

Run from a clean, frozen checkout on the controller host. JSON config supplies
site paths; authentication stays in the existing local SSH/model clients.
"""
import argparse
import base64
import errno
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from manipulation_agent.records import now, write_json, read_run, source_version
from manipulation_agent.replay import render_replay
from manipulation_agent.tools import tool_specs
from manipulation_agent.batch_lifecycle import ExecutionCapacity, FileLease, WorkerLease, classify_episode_outcome, validate_manifest_coverage, require_completed_scope, preferred_gpu_worker, gpu_lease_available
from manipulation_agent.startup_progress import STARTUP_POLICY, startup_progress_update
from manipulation_agent.closure import closure_watchdog
from manipulation_agent.clients.events import parse_events
from manipulation_agent.clients.controller import controller_client, controller_prefix, controller_options, qwen_config_path
from manipulation_agent.batch_storage import BatchStorage

FINAL = {'passed','failed','blocked'}
TRANSIENT_STORAGE_ERRORS = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT,
                            errno.ENOTCONN, errno.ECONNRESET, errno.EHOSTUNREACH,
                            errno.EAGAIN}


def inline_systemd_launcher(script):
    """Deliver launcher bytes without waiting for cross-host NAS visibility."""
    # systemd expands ExecStart dollars before bash runs. Preserve them for the
    # sourced simulator environment, including ${PYTHONPATH:-} and shell quotes.
    return ['/bin/bash','-c',script.replace('$','$$')]


def evaluation_summary(evaluation):
    """Keep combinatorial goal arrays in the immutable episode, not live progress."""
    if not isinstance(evaluation, dict):
        return evaluation
    return {key: value for key, value in evaluation.items()
            if key not in {'goal_options', 'initial_goal_options', 'goal_events'}}


def episode_artifact_hashes(runid, episode, controller):
    """Stable archive keys also cover an adopted controller in another batch."""
    hashes={}
    for label,folder in [('runs',episode),('controllers',controller)]:
        if not folder.exists():continue
        for path in sorted(folder.iterdir()):
            if path.is_file():
                with path.open('rb') as stream:
                    hashes[f'{label}/{runid}/{path.name}']=hashlib.file_digest(stream,'sha256').hexdigest()
    return hashes


def infrastructure_failure(row):
    """Keep normal scored policy/action failures separate from broken runs."""
    if not isinstance(row.get('task_success'),bool):
        return 'missing_final_score'
    if row.get('archive_failure'):
        return 'archive_failure'
    for field in ('video_validation','observation_validation','evidence_alignment'):
        if row.get(field)!='passed':
            return field
    return None


def worker_configs(config):
    """Resolve explicit host/GPU lanes; legacy single-host configs still work."""
    lanes = config.get('workers')
    if lanes is None:
        lanes = [{'id':f'gpu{gpu}', 'gpu':gpu} for gpu in config['gpus']]
    if not lanes:
        raise ValueError('At least one worker is required')
    allowed = {'id', 'gpu', 'ssh', 'data_root', 'sim_python', 'sim_env',
               'base_port', 'minimum_free_gpu_mib', 'simulator_gpu_memory_admission', 'simulator_memory_max',
               'simulator_env', 'expected_gpu_uuid', 'memory_budget_gib', 'admission_delay_seconds',
               'simulator_appdata_path', 'simulator_tmpdir', 'startup_memory_gib',
               'validated_driver_version', 'driver_validation_evidence', 'qwen_config'}
    result=[]; ids=set(); devices=set(); ports=set()
    for lane in lanes:
        if set(lane)-allowed: raise ValueError('Unsupported worker fields')
        merged={**config, **lane}
        worker_id=lane.get('id')
        if not worker_id or worker_id in ids: raise ValueError('Duplicate or missing worker id')
        host=tuple(merged['ssh'][-1:]);gpu=merged['gpu'];port=merged['base_port']+gpu
        if type(gpu) is not int or gpu<0 or not 1024<=port<=65535: raise ValueError('Invalid GPU or port')
        if (host,gpu) in devices or (host,port) in ports: raise ValueError('Worker GPU or port collision')
        env=merged.get('simulator_env',{})
        if type(merged.get('simulator_gpu_memory_admission',True)) is not bool:
            raise ValueError('Simulator GPU memory admission must be boolean')
        budget=merged.get('memory_budget_gib',28)
        if type(budget) is not int or not 12<=budget<=28: raise ValueError('Invalid memory budget')
        if merged.get('simulator_memory_max',str(budget)+'G')!=str(budget)+'G':
            raise ValueError('Resource admission and hard simulator memory limit must agree')
        if not 12<=merged.get('startup_memory_gib',budget)<=budget:
            raise ValueError('Startup estimate must fit the simulator hard limit')
        if merged.get('validated_driver_version'):
            evidence=json.loads(Path(merged['driver_validation_evidence']).read_text())
            if evidence.get('status')!='passed' or evidence.get('driver')!=merged['validated_driver_version']:
                raise ValueError('Driver exception requires passed renderer evidence for that exact version')
        if set(env)-{'MAS_VIDEO_RENDER_STRIDE','MAS_VIDEO_RENDER_FLUSHES',
                     'MAS_LOCAL_MANIPULATION','MAS_RENDERER_PIXEL_OWNER'}:
            raise ValueError('Only reviewed recording, local manipulation and pixel ownership options may be worker environment overrides')
        for flag in ('MAS_LOCAL_MANIPULATION', 'MAS_RENDERER_PIXEL_OWNER'):
            if flag in env and env[flag] not in ('0','1'):
                raise ValueError(flag+' must be an explicit string 0 or 1')
        if merged.get('simulator_appdata_path') and not Path(merged['simulator_appdata_path']).is_absolute():
            raise ValueError('Simulator appdata path must be absolute')
        if merged.get('simulator_tmpdir') and not Path(merged['simulator_tmpdir']).is_absolute():
            raise ValueError('Simulator temporary directory must be absolute')
        ids.add(worker_id);devices.add((host,gpu));ports.add((host,port));result.append(merged)
    # Workers are candidates, not simultaneous reservations. Shared host/GPU
    # leases below constrain actual admission across both comparison arms.
    slot_size=config.get('host_slot_memory_gib',28)
    adaptive=config.get('adaptive_memory_admission',False)
    if type(adaptive) is not bool:
        raise ValueError('adaptive_memory_admission must be boolean')
    if adaptive and not 1<=config.get('adaptive_host_slot_count',4)<=16:
        raise ValueError('Invalid adaptive host slot count')
    profile=config.get('memory_admission_profile','observed_peak')
    if profile not in ('observed_peak','throughput') or (profile=='throughput' and not adaptive):
        raise ValueError('Invalid memory admission profile')
    for worker in result:
        limit=config.get('host_worker_memory_budget_gib',{}).get(worker['ssh'][-1],slot_size)
        if limit is None and profile!='throughput':raise ValueError('Dynamic host budget requires throughput admission')
        if worker.get('memory_budget_gib',28)>slot_size or (limit is not None and limit<slot_size):
            raise ValueError('Worker memory exceeds shared host slot capacity')
    return result


def runtime_storage_args(config):
    """Check scratch/cache disks separately from potentially remote run output."""
    paths = [config.get(key) for key in ('simulator_appdata_path', 'simulator_tmpdir')]
    return [arg for path in dict.fromkeys(p for p in paths if p)
            for arg in ('--runtime-storage-path', str(path))]


def driver_admission_args(config):
    version=config.get('validated_driver_version')
    return ['--validated-driver-version',version] if version else []


def gpu_memory_admission_args(config):
    return [] if config.get('simulator_gpu_memory_admission',True) else ['--skip-gpu-memory-admission']


def controller_process_alive(pid, output):
    """Check an owned controller wrapper without signaling or restarting it."""
    proc=Path('/proc')/str(pid)
    try:
        if proc.stat().st_uid!=os.getuid(): raise RuntimeError('Controller UID mismatch')
        state=(proc/'stat').read_text().rsplit(') ',1)[1].split()[0]
        if state=='Z': return False
        args=(proc/'cmdline').read_bytes().decode().split('\0')
    except FileNotFoundError:
        return False
    if not any(x.endswith(('/scripts/run_codex_controller.py','/scripts/run_model_controller.py')) for x in args):
        raise RuntimeError('Controller PID reused by another process')
    if '--output' not in args or args[args.index('--output')+1]!=str(output):
        raise RuntimeError('Controller output ownership mismatch')
    return True


def prepolicy_failure(record, controller):
    if record.get('status')!='failed': return False
    stream=controller/'model_events.jsonl'
    if record.get('actions') is None:
        # A pre-recorder startup failure has no action counter; archival may
        # preserve that unknown value as JSON null. Admit only the
        # explicitly pre-policy stages, without inventing fields in its record.
        if record.get('failure_stage') not in {'asset_preflight','simulator_starting'}:
            return False
        started=('controller_pid','controller_wrapper_pid','controller_status',
                 'execution_started_at_unix','episode_deadline_unix','bridge_ready_at')
        if any(record.get(key) is not None for key in started): return False
        if (controller/'controller.json').exists(): return False
        return not stream.exists() or stream.stat().st_size==0
    if record.get('actions')!=0: return False
    if not stream.exists(): return record.get('controller_pid') is None
    events=[json.loads(s) for s in stream.read_text().splitlines()]
    # Client error items are not model decisions. Any generated policy item,
    # including reasoning or a started tool call, prevents this retry route.
    return all(e.get('item',{}).get('type') in (None,'error') for e in events)


def startup_resource_rejection(record, controller, preflight):
    """Only a documented dynamic resource race may requeue without a policy run."""
    if not prepolicy_failure({**record,'status':'failed'},controller):return None
    if preflight.get('status')!='blocked':return None
    failed={key for key,value in preflight.get('checks',{}).items() if value is False}
    dynamic={'gpu_capacity','cgroup_reclaimable_headroom','bridge_port_free',
             'host_memory_including_peer_reservations','adaptive_account_memory',
             'host_working_memory_headroom','data_disk_free_40GiB','runtime_storage_headroom_16GiB'}
    return sorted(failed) if failed and failed<=dynamic else None


def summarize(rows):
    counts = dict(Counter(r['status'] for r in rows))
    completed = sum(r['status'] in FINAL for r in rows)
    successes = sum(r.get('task_success') is True for r in rows)
    first=[r.get('previous_attempts',[r])[0] for r in rows]
    extra=sum(len(r.get('previous_attempts',[])) for r in rows)
    infrastructure=sum(len(r.get('previous_attempts',[])) for r in rows
                       if r.get('retry_kind','infrastructure')=='infrastructure')
    outcomes=Counter(classify_episode_outcome(r) for r in rows)
    return {'total':len(rows), 'completed':completed, 'counts':counts, 'task_successes':outcomes['success'],
            'official_goal_successes':successes,
            'outcome_counts':{name:outcomes[name] for name in ('success','failure','timeout')},
            'active_executions':sum(r.get('status')=='running' and r.get('stage') in
                {'simulator_starting','controller_starting','policy_running','finishing'} for r in rows),
            'first_attempt_task_successes':sum(r.get('task_success') is True for r in first),
            'extra_infrastructure_attempts':infrastructure,
            'extra_policy_attempts':extra-infrastructure,
            'success_fraction_all_tasks': outcomes['success']/len(rows) if rows else 0,
            'final_evaluations':sum(r.get('task_success') is not None for r in rows),
            'complete_videos':sum(r.get('video_validation')=='passed' for r in rows),
            'execution_status':'completed' if completed == len(rows) else 'running'}


def render_dashboard(progress):
    if progress.get('protocol',{}).get('comparison_arm'):
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from batch_comparison import render_arm_dashboard
        return render_arm_dashboard(progress)
    esc=lambda x: html.escape(str(x))
    s=progress['summary']; rows=[]
    for r in progress['tasks']:
        links=[]
        for key,label in [('replay_url','Replay'),('video_url','视频'),('record_url','记录')]:
            if r.get(key): links.append(f'<a href="{esc(r[key])}">{label}</a>')
        for previous in r.get('previous_attempts',[]):
            if previous.get('replay_url'):
                links.append(f'<a href="{esc(previous["replay_url"])}">历史尝试</a>')
        outcome='成功' if r.get('task_success') is True else ('未成功' if r.get('task_success') is False else '未取得最终评分')
        rows.append(f'<tr data-status="{esc(r["status"])}"><td>{r["index"]+1}</td><td>{esc(r["name"])}<small>{esc(r["task"])}</small><details><summary>任务输入与来源</summary><p>{esc(r["instruction"])}</p><p>{esc(r["instruction_source"])}</p></details></td><td class="{r["status"]}">{esc(r["status"])}<small>{esc(r.get("stage","queued"))}</small></td><td>{outcome}<small>Q={esc(r.get("q_score","—"))}</small></td><td>{esc(r.get("agent_steps",r.get("actions","—")))}<small>{esc(r.get("actions","—"))} executor actions</small><small>{esc(r.get("tool_calls","—"))} tool calls</small></td><td>{esc(r.get("evidence_alignment","—"))}<small>video: {esc(r.get("video_validation","—"))}</small></td><td>{" · ".join(links)}<small>{esc(r.get("failure",""))}</small></td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BEHAVIOR 100 · 批量测试</title>
<style>body{font:16px system-ui;background:#eef3f8;color:#142c43;margin:25px auto;max-width:1500px;padding:0 18px}a{color:#07599b}article,.cards>div{background:white;border:1px solid #d3dfe9;border-radius:10px;padding:18px;margin:14px 0}.cards{display:flex;gap:14px;flex-wrap:wrap}.cards>div{flex:1;min-width:150px}.cards strong{font-size:30px;display:block}table{border-collapse:collapse;width:100%;min-width:1050px}td,th{text-align:left;border-bottom:1px solid #ddd;padding:12px;vertical-align:top}small{display:block;color:#586d81;margin-top:7px;overflow-wrap:anywhere}td:nth-child(2){max-width:360px}td:last-child{max-width:270px}summary{cursor:pointer}input,select{font:inherit;padding:8px;margin:8px}.table{overflow-x:auto}.passed{color:#087044}.failed{color:#ad3030}.blocked{color:#9c6700}.running{color:#065fa7}p{line-height:1.65}code{overflow-wrap:anywhere}</style>
<h1>BEHAVIOR 2026 · 100 项任务测试</h1><p>RGB agent + ideal motor executor · gpt-6-astra · public instance 301 · seed 0</p>''' + f'''
<div class="cards"><div><strong>{s['completed']} / {s['total']}</strong>已结束（包括失败和受阻）</div><div><strong>{s['task_successes']} / {s['total']}</strong>独立评估成功 / 固定总数</div><div><strong>{s['final_evaluations']}</strong>取得最终评分</div><div><strong>{s['complete_videos']}</strong>录像完整性验证通过</div></div>
<article><p><b>失败与重试：</b>首次尝试成功 {s['first_attempt_task_successes']} / {s['total']}；另有 {s['extra_infrastructure_attempts']} 次模型启动前的基础设施重试、{s['extra_policy_attempts']} 次执行器修复后重测。上方成功数含修复后的最新尝试，原始失败记录保留在该任务行，任务总数始终为 100。</p><p><b>协议：</b>每种任务运行一个公开测试实例，共 100 个 episode，不等于所有公开实例或官方排行榜提交。机器人输入是同时采集的前、后、左、右 RGB，无腕部相机；模型通过 9 个 MCP tools 和 4 个可读取 skills 操作当前理想执行器。每项上限 80 次动作、20,000 控制步、模型 30 分钟。切割、擦洗等能力未扩展，相关失败保留在总数中。</p><p><b>输入来源：</b>50 项使用官方原文，50 项按静态 BDDL 目标补写，逐项标注。独立评估、几何和 spectator 录像只供执行器或离线审阅，不作为模型观测。</p><p><b>回放：</b>连续录像记录实际控制步；模型等待时间不铺成静止画面。Replay 同步四路 RGB、模型公开 assistant 原文、工具参数和返回结果。不补写思考，不展示隐藏推理；原始私有运行记录保存在实验目录。</p><p><b>状态：</b>passed 要求任务成功、控制器正常结束、证据对齐、四相机观测和录像验证通过；failed/blocked 分别保留具体阶段。失败任务有最终评分时显示 Q，仿真在初始化前失败时无法生成观测或视频。部分录像不标成完整。</p><p>批次状态：{esc(s['execution_status'])} · 更新：{esc(progress['updated_at'])} · <a href="behavior100/progress.json">实时 JSON</a> · <a href="behavior100/manifest.json">冻结清单</a> · <a href="behavior100/validation.json">验证记录</a> · <a href="behavior100/journal.md">迭代日志</a></p></article>
<input id="search" placeholder="搜索任务 / 阶段"><select id="status"><option value="">所有状态</option><option>planned</option><option>running</option><option>passed</option><option>failed</option><option>blocked</option></select><label><input id="refresh" type="checkbox" checked>每 60 秒刷新</label><div class="table"><table><thead><tr><th>#</th><th>任务</th><th>运行状态</th><th>独立评估</th><th>动作数</th><th>证据与录像</th><th>回放 / 原始记录</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>''' + '''<script>const search=document.querySelector('#search'), status=document.querySelector('#status');function filter(){document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(search.value.toLowerCase())&&(!status.value||r.dataset.status===status.value)))}search.oninput=status.onchange=filter;setInterval(()=>{if(document.querySelector('#refresh').checked&&!search.value&&!status.value)location.reload()},60000)</script></html>'''
    protocol=progress.get('protocol',{})
    model=protocol.get('model','gpt-6-astra')
    effort=protocol.get('model_reasoning_effort')
    page=page.replace(' · gpt-6-astra · ', ' · '+esc(model)+((' / '+esc(effort)) if effort else '')+' · ')
    if protocol.get('episode_budget_policy')=='agent_steps':
        page=page.replace('模型通过 9 个 MCP tools 和 4 个可读取 skills 操作当前理想执行器。每项上限 80 次动作、20,000 控制步、模型 30 分钟。',
            '模型通过 MCP 工具和任务分解 skill 操作当前理想执行器。每项上限 '+str(protocol['max_agent_steps'])+' 个 agent step（act/look 请求，失败计步，重复传输不计）；无任务执行时间上限。初始化与结束归档单独监控。')
        page=page.replace('动作数</th>','agent step / 执行动作</th>')
    elif protocol.get('episode_budget_policy')=='unbounded':
        page=page.replace('模型通过 9 个 MCP tools 和 4 个可读取 skills 操作当前理想执行器。每项上限 80 次动作、20,000 控制步、模型 30 分钟。',
            '后续新启动任务不设动作数、回合控制步或执行时间上限；继续记录每次act/look。此前已启动任务保留原配置，逐项记录。初始化与结束归档单独监控。')
    total = s['total']
    official=sum(r['instruction_source']=='official_gallery' for r in progress['tasks'])
    page=page.replace('50 项使用官方原文，50 项按静态 BDDL 目标补写',f'{official} 项使用官方原文，{total-official} 项按静态 BDDL 目标补写')
    page = page.replace('100 项任务测试',f'{total} 项任务测试').replace('任务总数始终为 100',f'任务总数始终为 {total}').replace('共 100 个 episode',f'共 {total} 个 episode')
    if progress.get('comparison'):
        c = progress['comparison']
        banner = '<article><b>'+esc(c.get('title','GT 导航修复后重测'))+'</b><p>'+esc(c['scope'])+'</p><p><a href="'+esc(c['baseline_url'])+'">上次测试账本</a> · <a href="'+esc(c['report_url'])+'">修复与验证报告</a> · <a href="behavior100_realtime.html">真实耗时视频</a></p><p>'+esc(c.get('display_note','本页仅统计新批次，不覆盖或合并旧成绩。其他任务未排队。'))+'</p></article>'
        page = page.replace('<div class="cards">',banner+'<div class="cards">',1)
    if progress.get('workers'):
        worker_rows=[]
        for w in progress['workers']:
            active=next((r for r in progress['tasks'] if r.get('worker_id')==w['id'] and r['status']=='running'),None)
            worker_rows.append('<li>'+esc(w['id'])+' · '+esc(w['host'])+' · GPU '+str(w['gpu'])+' · '+(esc(active['task'])+' / '+esc(active.get('stage')) if active else '等待下一项 / 已结束')+'</li>')
        panel='<article><b>独立仿真 workers</b><ul>'+''.join(worker_rows)+'</ul><p>源版本逐任务记录：'+esc(progress.get('run_source_commits',{}))+'。此前已完成和正在执行的尝试保留原始源码版本；不把不同渲染配置当成单变量实验。</p></article>'
        page=page.replace('<div class="cards">',panel+'<div class="cards">',1)
    return page


class Batch:
    @property
    def c(self):
        return getattr(self._worker_local, 'config', self._config)

    def __init__(self, config):
        self._config=config; self._worker_local=threading.local(); self.workers=worker_configs(config)
        self.root=Path(config['batch']); self.scheduler_source=Path(__file__).resolve().parents[1]
        self.source=Path(config.get('source_path',self.scheduler_source)).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text()); self.reports=Path(config['reports'])
        self.lock=threading.RLock(); self.queue=queue.Queue(); self.rows=self.manifest['tasks']
        self.worker_states={};self._archive_pool=None
        self._draining=False
        self.storage=BatchStorage(self.root,self.reports,self.manifest,
            spool=config.get('supervisor_spool_dir'),retry_seconds=config.get('publication_retry_seconds',5),
            renderer=render_dashboard)
        recovered=self.storage.recovered_rows()
        validate_manifest_coverage(self.rows,config.get('expected_task_count'))
        expected_tasks={r['index']:r['task'] for r in self.rows}
        for directory in ('records','preflight','controllers','runs','launchers','logs'):
            (self.root/directory).mkdir(exist_ok=True)
        (self.root/'journal.md').touch(exist_ok=True)
        selection_path=self.root/'infrastructure_retries.json'
        selections=json.loads(selection_path.read_text()) if selection_path.exists() else {}
        for r in self.rows:
            saved=self.root/'records'/f"{r['run_id']}.json"
            selection=selections.get(str(r['index']))
            if selection:
                previous=[]
                for runid in selection.get('previous_run_ids',[r['run_id']]):
                    record=json.loads((self.root/'records'/f'{runid}.json').read_text())
                    if not prepolicy_failure(record,Path(record.get('controller_output',self.root/'controllers'/runid))):
                        raise RuntimeError('Only failed attempts before any policy output may be retried here')
                    previous.append({k:v for k,v in record.items() if k!='previous_attempts'})
                r.update(run_id=selection['run_id'],attempt=len(previous)+1,previous_attempts=previous,retry_reason=selection['reason'])
                saved=self.root/'records'/f"{r['run_id']}.json"
            local=recovered.get(r['run_id'])
            try:r.update(json.loads(saved.read_text()))
            except FileNotFoundError:pass
            except json.JSONDecodeError as exc:
                # A complete identity-checked local checkpoint is authoritative
                # when publication on shared storage produced invalid JSON.
                # Never guess a score or retry a task from a partial record.
                if not local:raise
                self.storage.journal(f'Recover invalid shared record {saved.name} from durable local checkpoint: {exc.msg}')
            except OSError:
                if not local:raise
            if local and local.get('updated_at','')>r.get('updated_at',''):
                if (local['index'],local['task'])!=(r['index'],r['task']):
                    raise RuntimeError('Local checkpoint task identity mismatch')
                r.update(local)
            if (r['status']=='running' or (r.get('simulator_cleanup') or {}).get('status')=='blocked') and r['run_id'] not in config.get('adopt_inflight',{}):
                if r.get('execution_finished_at') and (r.get('simulator_cleanup') or {}).get('status')=='passed':
                    continue  # CPU archival is recovered independently below.
                match=next((w for w in self.workers if w['gpu']==r.get('gpu_index') and
                    w['ssh'][-1]==r.get('worker_host') and w['sim_python']==r.get('simulator_interpreter')),None)
                if not match:raise RuntimeError('Unfinished attempt has no matching original execution lane: '+r['run_id'])
                config.setdefault('adopt_inflight',{})[r['run_id']]=match['id']
        validate_manifest_coverage(self.rows,config.get('expected_task_count'),expected_tasks)
        adoptions=config.get('adopt_inflight',{})
        for runid,worker_id in adoptions.items():
            row=next((r for r in self.rows if r['run_id']==runid),None)
            worker=next((w for w in self.workers if w['id']==worker_id),None)
            if row is None or worker is None: raise RuntimeError('Unknown adoption target')
            if row['status']=='running' and (row.get('gpu_index')!=worker['gpu'] or row.get('simulator_interpreter')!=worker['sim_python']):
                raise RuntimeError('Inflight task must retain its original execution lane')
        self.scheduler_info=source_version()
        self.source_info=self.scheduler_info
        if self.source!=self.scheduler_source:
            probe=subprocess.run([sys.executable,'-c',
                'import json; from manipulation_agent.records import source_version; print(json.dumps(source_version()))'],
                cwd=self.source,env={**os.environ,'PYTHONPATH':str(self.source/'src')},
                check=True,capture_output=True,text=True,timeout=30)
            self.source_info=json.loads(probe.stdout)
        if self.scheduler_info['dirty']:raise RuntimeError('Scheduler source must be a clean frozen checkout')
        if self.source_info['dirty']: raise RuntimeError('Batch source must be a clean frozen checkout')
        if config.get('source_commit') and self.source_info['commit']!=config['source_commit']:
            raise RuntimeError('Frozen executor commit mismatch')
        self.environment={**os.environ,'PYTHONPATH':str(self.source/'src')}
        missing=[k for k in config.get('required_controller_env',[]) if not self.environment.get(k)]
        if missing: raise RuntimeError('Required controller environment variables absent: '+','.join(missing))
        if config.get('controller_path'): self.environment['PATH']=config['controller_path']
        name=controller_client(self.manifest)
        if name=='qwen':
            from manipulation_agent.clients.qwen.config import load_config
            for worker in self.workers:
                controller_prefix(self.source,self.manifest,worker)
                runtime=load_config(qwen_config_path(self.manifest,worker))
                if runtime['device']!='cpu' and not worker.get('expected_gpu_uuid'):
                    raise ValueError('Qwen workers require expected_gpu_uuid for simulator/inference lease isolation')
                subprocess.run([runtime['python'],'-c',
                    'import importlib.metadata as m; [m.version(p) for p in ("torch","transformers","peft","mcp")]'],
                    check=True,capture_output=True,text=True,timeout=30)
        else:
            client=shutil.which(name,path=self.environment.get('PATH'))
            if not client: raise RuntimeError('Model executable unavailable before any simulator launch')
            subprocess.run([client,'--version'],check=True,capture_output=True,text=True,timeout=20)

    def ssh(self, args, timeout=40):
        cmd=shlex.join(args) if isinstance(args,list) else args
        return subprocess.run(self.c['ssh']+[cmd],capture_output=True,text=True,timeout=timeout)

    def journal(self, message):
        with self.lock:
            self.storage.journal(message)

    def drain_requested(self):
        # Only admission reads the shared control file. Publication during an
        # active episode must not depend on even a stat() against shared storage.
        try:self._draining=(self.root/'drain_requested.json').exists()
        except OSError as exc:
            self.journal('Admission held: cannot read drain control: '+str(exc))
            self._draining=True
        return self._draining

    def publish(self):
        with self.lock:
            progress={'updated_at':now(),'source':self.source_info,'supervisor':{'host':os.uname().nodename,'pid':os.getpid(),'unit':os.environ.get('MAS_UNIT'),'interpreter':sys.executable},
                      'scheduler_source':getattr(self,'scheduler_info',self.source_info),
                      'protocol':{k:v for k,v in self.manifest.items() if k!='tasks'},'summary':summarize(self.rows),'tasks':self.rows,'comparison':self.manifest.get('comparison'),
                      'workers':[{'id':w['id'],'host':w['ssh'][-1], 'gpu':w['gpu'],
                          'port':w['base_port']+w['gpu'], 'interpreter':w['sim_python'],
                          'recording_options':w.get('simulator_env',{})} for w in self.workers],
                      'draining':getattr(self,'_draining',False),
                      'storage_sync':self.storage.status(),
                      'worker_states':self.worker_states,
                      'run_source_commits':dict(Counter((r.get('run_source') or r.get('source') or {}).get('commit','pending') for r in self.rows))}
            progress['summary']['waiting_resource_workers']=sum(v.get('stage') in
                {'waiting_for_lease','waiting_for_resources','waiting_for_initialization'} for v in self.worker_states.values())
            validation={'status':'running' if progress['summary']['execution_status']=='running' else ('passed' if all(r['status']=='passed' for r in self.rows) else 'failed'),
                        'level':'real_rgb_simulator_batch','task_count':len(self.rows),'summary':progress['summary'],
                        'source':progress['source'],'official_submission_eligible':False,
                        'note':'Task success, controller termination, transport alignment and video completeness are separate fields.'}
            self.storage.submit(progress,validation)

    def update(self, row, **values):
        with self.lock:
            row.update(values,updated_at=now())
            self.publish()

    def unit_state(self, unit):
        try:
            p=self.ssh(['systemctl','--user','show',unit,'-p','ActiveState','-p','MainPID','-p','ExecMainStatus','-p','Result','-p','Description','-p','MemoryPeak'])
        except (subprocess.TimeoutExpired,OSError) as exc:
            return {'query_status':'unavailable','reason':type(exc).__name__}
        if p.returncode:
            return {'query_status':'unavailable','returncode':p.returncode}
        return dict(line.split('=',1) for line in p.stdout.splitlines() if '=' in line)

    def initializing_unit_state(self, unit):
        """An unavailable SSH query is not evidence that the simulator exited."""
        state=self.unit_state(unit)
        if state.get('ActiveState') in {'inactive','failed'} and state.get('MainPID')=='0':
            raise RuntimeError('Simulator exited before bridge ready: '+str(state))
        if state.get('ActiveState') not in {'active','activating'} or not state.get('MainPID'):
            self.journal({'event':'startup_unit_query_unavailable','unit':unit,'query':state})
            return None
        return state

    def confirm_simulator_launch(self, row, launch):
        """Reconcile a lost launch reply without launching the episode twice."""
        try:
            response=self.ssh(launch)
        except (subprocess.TimeoutExpired,OSError) as exc:
            response=None
            self.update(row,launch_acknowledgment={'status':'unavailable','reason':type(exc).__name__})
        else:
            (self.root/'logs'/f"{row['run_id']}_launch.log").write_text(response.stdout+response.stderr)
            if response.returncode not in (0,255):
                raise RuntimeError('Simulator unit launch failed')
        while True:
            state=self.unit_state(row['simulator_unit'])
            if state.get('ActiveState') in {'active','activating','deactivating'}:
                if state.get('Description')!=f"BEHAVIOR100 owned {row['run_id']}":
                    raise RuntimeError('Simulator launch ownership mismatch')
                if int(state.get('MainPID') or 0)>0:
                    self.update(row,launch_acknowledgment={'status':'passed',
                        'recovered_missing_reply':response is None or response.returncode!=0})
                    return state
            if state.get('Description')==f"BEHAVIOR100 owned {row['run_id']}" and state.get('ActiveState') in {'inactive','failed'}:
                raise RuntimeError('Owned simulator exited during launch: '+str(state))
            # A lost SSH reply does not prove the remote systemd request failed.
            # Keep the declared unit leased until confirmed or its existing
            # initialization watchdog expires; never resend the launch blindly.
            if time.time()>=row['startup_deadline_unix']:
                raise TimeoutError('Simulator launch could not be confirmed before startup watchdog expiry')
            self.journal({'event':'launch_confirmation_pending','unit':row['simulator_unit'],'query':state})
            time.sleep(4)

    def adopted_unit_state(self, row):
        """Unavailable transport cannot be interpreted as an absent process."""
        while True:
            state=self.unit_state(row['simulator_unit'])
            if state.get('ActiveState') in {'active','activating','deactivating','inactive','failed'}:
                return state
            self.update(row,adoption_transport={'status':'unavailable','at':now(),'query':state})
            self.journal({'event':'adoption_waiting_for_unit_query','run_id':row['run_id']})
            time.sleep(4)

    def health(self, port):
        # This is an observation, not a simulator operation. Failure to obtain
        # it must not escape into run_one's controller-stop / forced-finish path.
        # Callers retain the last closure state and check unit liveness; the
        # existing startup and post-finish watchdogs still apply.
        try:
            p=self.ssh(['curl','--noproxy','*','-fsS','--max-time','3',f'http://127.0.0.1:{port}/healthz'])
        except (subprocess.TimeoutExpired,OSError) as exc:
            self.journal({'event':'health_query_unavailable','port':port,
                          'reason':type(exc).__name__})
            return None
        if p.returncode:return None
        try:health=json.loads(p.stdout)
        except (ValueError,TypeError):
            self.journal({'event':'health_query_unavailable','port':port,
                          'reason':'invalid_json'})
            return None
        return health if isinstance(health,dict) else None

    def cleanup_owned_unit(self, row):
        """Verify this lane is reusable; episode evidence never decides that."""
        unit=row['simulator_unit'];runid=row['run_id'];state={}
        try:
            state=self.unit_state(unit)
            if state.get('ActiveState') in {'active','activating','deactivating'}:
                if state.get('Description')!=f'BEHAVIOR100 owned {runid}':
                    raise RuntimeError('Simulator ownership changed; refusing stop')
                stopped=self.ssh(['systemctl','--user','stop',unit],timeout=60)
                if stopped.returncode:
                    raise RuntimeError('Owned simulator stop command failed: '+stopped.stderr[-1200:])
                state=self.unit_state(unit)
            # A collected transient unit may no longer retain its Description.
            # Only the live unit being signaled needs the ownership match.
            if state.get('ActiveState') not in {'inactive','failed'} or str(state.get('MainPID'))!='0':
                raise RuntimeError('Simulator termination could not be verified')
            cleanup={'status':'passed','at':now(),'unit':unit,'state':state}
        except Exception as exc:
            cleanup={'status':'blocked','at':now(),'unit':unit,'state':state,
                     'reason':f'{type(exc).__name__}: {exc}'}
            self.journal('WORKER CLEANUP BLOCKED '+runid+': '+cleanup['reason'])
        self.update(row,unit_final_state=state,simulator_cleanup=cleanup)
        return cleanup

    def hold_worker_for_cleanup(self, row):
        """Keep an unresolved simulator on its own lane, without draining peers."""
        cleanup=row.get('simulator_cleanup') or {}
        if cleanup.get('status')!='blocked':
            return False
        hold={'status':'blocked','at':now(),'worker_id':self.c['id'],
              'host':self.c['ssh'][-1],'gpu':self.c['gpu'],
              'run_id':row['run_id'],'unit':row.get('simulator_unit'),
              'reason':cleanup['reason']}
        self.update(row,worker_hold=hold)
        self.journal('WORKER HELD '+self.c['id']+' after '+row['run_id']+
                     '; other lanes continue: '+cleanup['reason'])
        return True

    def worker_state(self, stage, **details):
        with self.lock:
            self.worker_states[self.c['id']]={'stage':stage,'at':now(),
                'host':self.c['ssh'][-1],'gpu':self.c['gpu'],**details}
            self.publish()

    def acquire_worker_lease(self, *, adopting=None):
        directory=self._config.get('shared_lease_dir',str(self.root.parent/'resource_leases'))
        slot_size=self._config.get('host_slot_memory_gib',28)
        host_budget=self._config.get('host_worker_memory_budget_gib',{}).get(self.c['ssh'][-1],slot_size)
        if host_budget is None:host_budget=slot_size*self._config.get('adaptive_host_slot_count',4)
        def owner_active(previous):
            state=self.unit_state(previous['unit'])
            return state.get('ActiveState') not in {'inactive','failed'} or state.get('MainPID')!='0'
        lease=WorkerLease(directory,self.c,host_budget_gib=host_budget,slot_memory_gib=slot_size,
                          owner={'batch':str(self.root),'coordinator_pid':os.getpid()},
                          adaptive_slot_count=self._config.get('adaptive_host_slot_count',4)
                          if self._config.get('adaptive_memory_admission') else None,
                          owner_active=owner_active)
        if not lease.acquire(adopting=adopting):
            self.worker_state('waiting_for_lease',previous_unit=(lease.blocked_owner or {}).get('unit'))
            return None
        # A supervisor crash releases flock, but its remote simulator may live.
        # Do not reuse that slot until its unit is terminal, or explicitly adopt
        # exactly that existing attempt without starting another model.
        try:
            lease.record(released=False,run_id=adopting['run_id'] if adopting else None,
                         unit=adopting.get('simulator_unit') if adopting else None,
                         startup_memory_gib=self.c.get('startup_memory_gib',self.c.get('memory_budget_gib',28)))
        except BaseException:
            lease.release()
            raise
        return lease

    def acquire_inference_resources(self, adopting=None):
        if controller_client(getattr(self,'manifest',{}))!='qwen':return True
        from manipulation_agent.clients.qwen.config import load_config
        from manipulation_agent.clients.qwen.resources import acquire_inference_lease, gpu_admission
        runtime=load_config(qwen_config_path(self.manifest,self.c))
        if runtime['device']=='cpu':return True
        if runtime['gpu_uuid']==self.c.get('expected_gpu_uuid'):
            raise RuntimeError('Permanent admission error: simulator and inference GPUs must be distinct')
        directory=self._config.get('shared_lease_dir',str(self.root.parent/'resource_leases'))
        lease=acquire_inference_lease(directory,runtime,
            {'batch':str(self.root),'worker_id':self.c['id']},adopting=adopting['run_id'] if adopting else None)
        if lease is None:
            self.worker_state('waiting_for_inference_gpu');return False
        self._worker_local.inference_lease=lease
        if not adopting:
            try:gpu_admission(runtime)
            except ValueError as exc:
                raise RuntimeError('Permanent admission error: '+str(exc)) from exc
            except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
                self.worker_state('waiting_for_inference_memory',reason=str(exc));return False
        return True

    def record_inference_controller(self,row,pid,controller):
        lease=getattr(self._worker_local,'inference_lease',None)
        if lease:
            from manipulation_agent.clients.qwen.resources import process_start
            lease.record(run_id=row['run_id'],controller_pid=pid,pid_start=process_start(pid),
                         controller_output=str(controller),launch_pending=False)

    def record_inference_pending(self,row,controller):
        lease=getattr(self._worker_local,'inference_lease',None)
        if lease:lease.record(run_id=row['run_id'],controller_output=str(controller),launch_pending=True)

    def acquire_startup_gate(self):
        """Serialize reservation publication; legacy mode also holds through init."""
        if not self._config.get('adaptive_memory_admission'):
            return True
        directory=self._config['shared_lease_dir']
        host_key=hashlib.sha256(self.c['ssh'][-1].encode()).hexdigest()[:20]
        throughput=self._config.get('memory_admission_profile')=='throughput'
        prefix='admission' if throughput else 'startup'
        gate=FileLease(Path(directory)/f'{prefix}-{host_key}.lock',
                       {'batch':str(self.root),'worker_id':self.c['id']})
        if not gate.acquire():
            self.worker_state('waiting_for_admission' if throughput else 'waiting_for_initialization');return False
        self._worker_local.startup_gate=gate
        return True

    def release_startup_gate(self):
        gate=getattr(self._worker_local,'startup_gate',None)
        if gate:
            gate.release();self._worker_local.startup_gate=None

    def memory_policy_args(self):
        if not self._config.get('adaptive_memory_admission'):
            return []
        lease=self._worker_local.lease
        policy={'peers':lease.active_peers(),
                'candidate_gib':self.c.get('memory_budget_gib',28),
                'startup_gib':self.c.get('startup_memory_gib',self.c.get('memory_budget_gib',28)),
                'profile':self._config.get('memory_admission_profile','observed_peak'),
                'host_budget_gib':self._config['host_worker_memory_budget_gib'][self.c['ssh'][-1]]}
        # GDB with startup-with-shell disabled does not round-trip quoted JSON
        # arguments. A single ASCII token survives both systemd and GDB argv.
        encoded=base64.b64encode(json.dumps(policy).encode()).decode('ascii')
        return ['--memory-policy-base64',encoded]

    def resource_admission(self):
        """No task is claimed here; immutable host checks run once per worker."""
        self.worker_state('admitting')
        args=[self.c['sim_python'],str(getattr(self,'scheduler_source',self.source)/'scripts/behavior100_remote.py'),'preflight',
              '--manifest',str(self.root/'manifest.json'),'--gpu',str(self.c['gpu']),
              '--port',str(self.c['base_port']+self.c['gpu']),'--data-root',self.c['data_root'],
              '--memory-budget-gib',str(self.c.get('memory_budget_gib',28)),
              '--min-free-gpu-mib',str(self.c.get('minimum_free_gpu_mib',0))]
        args += runtime_storage_args(self.c)
        args += driver_admission_args(self.c)
        args += gpu_memory_admission_args(self.c)
        args += self.memory_policy_args()
        lease=getattr(self._worker_local,'lease',None)
        if lease:args+=['--reserved-host-memory-gib',str(lease.other_reserved_gib(self._config.get('host_slot_memory_gib',28)))]
        if getattr(self._worker_local,'static_ready',False):args.append('--light')
        # SSH normally starts in /home, while the simulator unit runs from its
        # frozen source directory. Probe the actual working filesystem: a full
        # home quota must not reject a run whose output/cache/temp are elsewhere.
        response=self.ssh(['/bin/bash','-lc',
                          'cd '+shlex.quote(str(self.source))+'\nexec '+shlex.join(args)])
        if response.returncode:
            self.worker_state('waiting_for_resources',reason='Resource query unavailable')
            return False
        check=json.loads(response.stdout)
        write_json(self.root/'preflight'/('worker_'+self.c['id']+'.json'),check)
        if self.c.get('expected_gpu_uuid') and check.get('gpu_uuid')!=self.c['expected_gpu_uuid']:
            raise RuntimeError('Permanent admission error: physical GPU UUID changed')
        static=('video_encoder','pinned_source_imports','validated_driver_floor')
        if any(check.get('checks',{}).get(name) is False for name in static):
            raise RuntimeError('Permanent admission error: static host environment check failed')
        self._worker_local.static_ready=True
        if check.get('status')!='passed':
            self.worker_state('waiting_for_resources',checks=check.get('checks',{}))
            return False
        if self._config.get('prefer_gpu_headroom', True):
            directory=self._config.get('shared_lease_dir',str(self.root.parent/'resource_leases'))
            with self.lock:
                preferred = preferred_gpu_worker(self.workers, self.worker_states,
                    self.c['ssh'][-1], check.get('queries', {}).get('gpus', {}).get('stdout', ''),
                    gpu_available=lambda worker: worker['id']==self.c['id'] or gpu_lease_available(directory,worker))
            if preferred is not None and preferred != self.c['id']:
                self.worker_state('waiting_for_resources', reason='Prefer available GPU with more free memory',
                                  preferred_worker=preferred)
                return False
        self._worker_local.admission=check
        return True

    def recover_worker_cleanup(self, row):
        """Retain both leases while only this lane retries owned cleanup."""
        while (row.get('simulator_cleanup') or {}).get('status')=='blocked':
            self.hold_worker_for_cleanup(row)
            self.worker_state('cleanup_pending',run_id=row['run_id'],
                              reason=row['simulator_cleanup'].get('reason'))
            time.sleep(self._config.get('cleanup_retry_seconds',10))
            self.cleanup_owned_unit(row)
        row.pop('worker_hold',None)

    def stop_controller(self, process, controller, wrapper_pid=None):
        """Bounded cleanup of only the model/wrapper owned by this episode."""
        if process is None:
            class AdoptedProcess:
                def poll(self):return None if controller_process_alive(wrapper_pid,controller) else 0
                def wait(self,timeout):
                    end=time.monotonic()+timeout
                    while self.poll() is None:
                        if time.monotonic()>=end:raise subprocess.TimeoutExpired('adopted controller',timeout)
                        time.sleep(.1)
                    return 0
                def terminate(self):
                    if self.poll() is None:os.kill(wrapper_pid,signal.SIGTERM)
                def kill(self):
                    if self.poll() is None:os.kill(wrapper_pid,signal.SIGKILL)
            process=AdoptedProcess()
        metadata_path=controller/'controller.json'
        try:metadata=json.loads(metadata_path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):metadata={}
        except OSError as exc:
            if exc.errno not in TRANSIENT_STORAGE_ERRORS:raise
            metadata=getattr(self,'_controller_metadata_cache',{}).get(str(controller),{})
        pid=metadata.get('pid')
        identity=None
        def owned_model():
            nonlocal identity
            if not pid:return False
            proc=Path('/proc')/str(pid)
            try:
                args=(proc/'cmdline').read_bytes().decode().split('\0')
                started=(proc/'stat').read_text().rsplit(') ',1)[1].split()[19]
                if proc.stat().st_uid!=os.getuid() or not any(str(controller) in arg for arg in args) or os.getpgid(pid)!=pid:
                    return False
                if identity is None:identity=started
                return identity==started
            except (ProcessLookupError,FileNotFoundError):return False
        if owned_model():
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if owned_model():
                try:
                    os.killpg(pid,signal.SIGKILL)
                except (ProcessLookupError,FileNotFoundError):pass
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=5)

    def finalize_episode(self, row, controller, remote_run, config):
        """CPU archival never owns the GPU/host lease or changes task outcome."""
        self._worker_local.config=config
        self.update(row,stage='archiving',archive_status='running')
        try:self.capture_result(row,remote_run)
        except Exception as exc:self.update(row,result_capture_error=f'{type(exc).__name__}: {exc}')
        try:
            self.archive(row,controller,remote_run)
            self.update(row,archive_status='passed')
        except Exception as exc:
            self.update(row,archive_failure=f'{type(exc).__name__}: {exc}',archive_status='failed')
        self.complete_row(row)

    def capture_result(self, row, remote_run):
        """Fetch the compact immutable outcome before potentially large media IO."""
        keys=('task_success','actions','agent_steps','max_agent_steps','agent_step_budget_exhausted','tool_calls','sim_steps','agent_outcome','finish_reason','failure','termination_reason',
              'source','backend','execution_finished_at_unix','deadline_expired_at_finish',
              'evaluation_finished_at_unix','evaluation_finished_after_deadline','episode_deadline_unix','execution_started_at_unix','episode_outcome')
        code=("import json; from pathlib import Path; p=Path("+repr(str(remote_run/'run.json'))+
              "); r=json.loads(p.read_text()); d={k:r[k] for k in "+repr(keys)+" if k in r}; "
              "e=r.get('evaluation') or {}; d['evaluation']={k:v for k,v in e.items() if k not in ('goal_options','initial_goal_options')}; print(json.dumps(d))")
        result=self.ssh([self.c['sim_python'],'-c',code],timeout=30)
        if result.returncode:raise RuntimeError('Final outcome unavailable remotely')
        raw=json.loads(result.stdout)
        values={key:raw[key] for key in keys if key in raw and key not in {'failure','source','episode_outcome'}}
        values.update(evaluation=raw.get('evaluation'),q_score=(raw.get('evaluation') or {}).get('official_metrics',{}).get('q_score',{}).get('final'),
                      result_capture_status='passed')
        if (raw.get('evaluation') or {}).get('score_type')=='custom_joint_terminal_and_public_history_v1':
            values['custom_score']=raw['evaluation']['goal_satisfaction_fraction']
        if raw.get('source'):values['run_source']=raw['source']
        if raw.get('failure') and raw.get('task_success') is None:values['simulator_failure_type']=raw['failure']
        if row.get('outcome_provisional') and isinstance(raw.get('task_success'),bool):
            values.update(episode_outcome=None,outcome_provisional=False,termination_reason=None)
        self.update(row,**values)

    def complete_row(self, row):
        # Media/evidence errors remain visible, but cannot turn a valid scored
        # task into a policy failure or turn a deadline into a late success.
        row.setdefault('execution_finished_at',now())
        outcome=classify_episode_outcome(row)
        reason=row.get('termination_reason')
        if not reason:
            reason=('episode_deadline_exceeded' if outcome=='timeout' else
                    'task_completed' if outcome=='success' else
                    'execution_error' if row.get('failure') else
                    'task_goal_not_satisfied' if row.get('task_success') is False and (
                        row.get('controller_finish_acknowledged') or
                        ((row.get('episode_closure') or {}).get('origin')=='controller' and
                         (row.get('episode_closure') or {}).get('phase')=='result_ready')) else
                    'controller_failed' if row.get('controller_status')!='passed' else
                    'task_goal_not_satisfied')
        self.update(row,status='passed' if outcome=='success' else 'failed',episode_outcome=outcome,
                    termination_reason=reason,stage='complete',finished_at=now())
        self.journal(f"END {row['run_id']}: outcome={outcome}; reason={reason}; task_success={row.get('task_success')}; Q={row.get('q_score')}.")

    def execution_clock_command(self, path, seconds, runtime=None, row=None):
        runtime=Path(runtime or self.source)
        manifest=getattr(self,'manifest',{})
        policy=(row or {}).get('episode_budget_policy',manifest.get('episode_budget_policy'))
        if policy in {'agent_steps','unbounded'}:seconds=None
        options=(',max_agent_steps='+repr((row or {}).get('max_agent_steps',manifest.get('max_agent_steps')))
                 if policy=='agent_steps' else ',unbounded=True' if policy=='unbounded' else '')
        code=('import json; from manipulation_agent.deadline import write_execution_clock; '
              'print(json.dumps(write_execution_clock('+repr(str(path))+','+repr(seconds)+options+')))')
        remote=shlex.join(['env','PYTHONPATH='+str(runtime/'src'),self.c['sim_python'],'-c',code])
        return self.c['ssh']+[remote]

    def controller_metadata(self, row, controller):
        path=controller/'controller.json'
        cache=getattr(self,'_controller_metadata_cache',None)
        if cache is None:cache=self._controller_metadata_cache={}
        try:metadata=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):return {}
        except OSError as exc:
            if exc.errno not in TRANSIENT_STORAGE_ERRORS:raise
            if (row.get('controller_storage') or {}).get('status')!='degraded':
                self.update(row,controller_storage={'status':'degraded','errno':exc.errno,
                    'error':str(exc),'at':now(),'path':str(path)})
            # An unreadable status file does not mean the live model stopped.
            # Preserve its already observed execution clock while storage heals.
            metadata=cache.get(str(controller),{})
        else:
            cache[str(controller)]=metadata
            if (row.get('controller_storage') or {}).get('status')=='degraded':
                self.update(row,controller_storage={'status':'passed','recovered_at':now(),
                                                    'path':str(path)})
        deadline=metadata.get('episode_deadline_unix')
        started=metadata.get('execution_started_at_unix')
        policy=(metadata.get('execution_clock') or {}).get('budget_policy')
        no_wall=policy in {'agent_steps','unbounded'}
        if isinstance(started,(int,float)) and (isinstance(deadline,(int,float)) or no_wall):
            if row.get('execution_started_at_unix')!=started or row.get('episode_deadline_unix')!=deadline:
                self.update(row,execution_started_at_unix=started,episode_deadline_unix=deadline,
                            execution_budget_seconds=metadata.get('execution_budget_seconds',row.get('episode_timeout_seconds')),
                            deadline_origin='model_start_after_mcp_handshake',
                            budget_basis=policy if no_wall else 'model_execution_excludes_initialization',stage='policy_running')
                self.worker_state('running',run_id=row['run_id'],deadline_unix=deadline)
        return metadata

    def wait_controller(self, row, process, controller):
        """Simulator/handshake startup and actual model execution have separate clocks."""
        while process.poll() is None:
            self.controller_metadata(row,controller)
            self.refresh_startup_progress(row)
            deadline=row.get('episode_deadline_unix')
            cutoff=deadline or (None if row.get('execution_started_at_unix') else row['startup_deadline_unix'])
            if not row.get('execution_started_at_unix') and (row.get('controller_storage') or {}).get('status')=='degraded':
                # A missing first clock due to EIO is not evidence that startup
                # stalled. Continue checking the process and simulator lifecycle.
                cutoff=None
            closing=self.step_episode_closed(row)
            if closing:
                reason=self.closure_timeout(row)
                if reason:
                    self.cleanup_controller_after_finish(row,process,controller,reason)
                    break
            elif cutoff is not None and time.time()>=cutoff:
                if deadline:self.update(row,episode_timed_out=True)
                else:self.update(row,startup_timed_out=True,termination_reason='startup_timeout')
                if deadline:
                    try:process.wait(timeout=min(20,row.get('cleanup_grace_seconds',120)))
                    except subprocess.TimeoutExpired:self.stop_controller(process,controller)
                else:self.stop_controller(process,controller)
                break
            time.sleep(2 if closing or cutoff is None else min(2,max(0,cutoff-time.time())))
            if time.monotonic()-getattr(process,'last_heartbeat',0)>30:
                process.last_heartbeat=time.monotonic();self.update(row,heartbeat_at=now())
        return self.controller_metadata(row,controller)

    def step_episode_closed(self, row):
        if not row.get('port'):return False
        last=getattr(self._worker_local,'closure_poll',None);current=time.monotonic()
        known=row.get('episode_closure') or {}
        if last and last[0]==row['run_id'] and current-last[1]<2:
            return known.get('phase','running')!='running'
        self._worker_local.closure_poll=(row['run_id'],current)
        health=self.health(row['port'])
        if health is not None:
            if not health.get('closed'):return False
            state=health.get('closure') or {'phase':'legacy_closed'}
            # An early closed flag is not a completed evaluation or reply.
            if state.get('phase')=='running':state={'phase':'legacy_closed'}
            if state!=known:self.update(row,episode_closure=state)
            return True
        state=self.unit_state(row['simulator_unit'])
        exited=state.get('ActiveState') in {'inactive','failed'} and str(state.get('MainPID'))=='0'
        if exited:
            if known.get('phase')!='result_ready':
                self.update(row,episode_closure={**known,'phase':'simulator_exited'})
            return True
        # A transient health failure must not erase observed shutdown progress.
        return known.get('phase','running')!='running'

    def closure_timeout(self, row):
        previous=dict(row.get('closure_watchdog') or {})
        reason=closure_watchdog(row,now=time.time(),
            stage_seconds=self._config.get('finish_stage_timeout_seconds',600),
            client_seconds=self._config.get('controller_exit_grace_seconds',120))
        if previous!=row.get('closure_watchdog'):
            self.update(row,closure_watchdog=row['closure_watchdog'])
        return reason

    def cleanup_controller_after_finish(self,row,process,controller,reason,wrapper_pid=None):
        cleanup={'status':'forced','reason':reason,'at':now(),
                 'episode_closure':row.get('episode_closure')}
        self.update(row,controller_cleanup=cleanup)
        write_json(controller/'controller_cleanup.json',cleanup)
        self.stop_controller(process,controller,wrapper_pid=wrapper_pid)

    def record_controller_end(self,row,controller,metadata):
        """Keep original client metadata; record receipt and cleanup separately."""
        receipt=bool(metadata.get('formal_finish_observed'))
        evidence_error=None
        try:
            client=self.manifest.get('controller_client','codex_cli').removesuffix('_cli')
            receipt=receipt or parse_events(controller/'model_events.jsonl',client).formal_finish_observed
        except (OSError,ValueError,TypeError) as exc:
            evidence_error=type(exc).__name__
        status=metadata.get('status')
        if status in {None,'running'}:
            status=('not_started' if not metadata and not row.get('controller_wrapper_pid') and not row.get('controller_pid')
                    else 'terminated' if row.get('controller_cleanup') else 'exited_without_final_metadata')
        fields={'controller_pid':metadata.get('pid'),'controller_status':status,
                'controller_timeout':bool(metadata.get('timeout')),
                'model_duration_seconds':metadata.get('duration_seconds'),
                'controller_finish_acknowledged':receipt}
        self.update(row,**fields)
        controller.mkdir(parents=True,exist_ok=True)
        write_json(controller/'controller_closure.json',{
            'at':now(),'formal_finish_observed':receipt,'process_status':status,
            'exit_code':metadata.get('exit_code'),'cleanup':row.get('controller_cleanup'),
            'evidence_error':evidence_error})
        return {**metadata,'formal_finish_observed':receipt}

    def refresh_startup_progress(self, row):
        """Read actual remote milestones, never mistake polling for progress."""
        if row.get('startup_watchdog_policy') != STARTUP_POLICY or row.get('execution_started_at_unix') or row.get('episode_deadline_unix'):
            return
        # Avoid launching a Python reader on every two-second health poll.
        last=getattr(self._worker_local,'startup_progress_poll',None)
        current=time.time()
        if (last and last[0]==row['run_id'] and current-last[1]<10
                and current<row['startup_deadline_unix']):
            return
        self._worker_local.startup_progress_poll=(row['run_id'],current)
        path=Path(row['simulator_output'])/'startup_stages.jsonl'
        code=("import json; from pathlib import Path; p=Path("+repr(str(path))+
              "); result=[]\n"
              "for line in (p.read_text().splitlines() if p.exists() else []):\n"
              " try: result.append(json.loads(line))\n"
              " except ValueError: pass\n"
              "print(json.dumps(result))")
        try:
            result=self.ssh(['/usr/bin/python3','-c',code],timeout=15)
            if result.returncode:return
            events=json.loads(result.stdout)
            if not isinstance(events,list):return
            updates=startup_progress_update(row,events,time.time())
        except (subprocess.TimeoutExpired,OSError,ValueError):
            return  # Unreadable progress cannot extend the watchdog.
        if updates:self.update(row,**updates)

    def controller_paths(self, runid):
        controller=self.storage.spool/'controllers'/runid
        controller.parent.mkdir(parents=True,exist_ok=True)
        log=self.storage.spool/'logs'/f'{runid}_controller.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        return controller,log

    def run_one(self, row, gpu):
        runid=row['run_id'];port=self.c['base_port']+gpu
        unit=f"mas-b100-{row['index']:03d}-r{row.get('attempt',1)}-{self.c['batch_tag']}.service"
        remote_run=Path(self.c['data_root'])/'runs'/runid
        # Model events, project files and status updates must not depend on NAS
        # publication during execution. Archive this local directory afterwards.
        controller,controller_log=self.controller_paths(runid)
        own_unit=False;launch_requested=False;process=None;deadline=None;startup_deadline=None
        native_fault_capture=self.c.get('native_fault_capture',False) is True
        self.update(row,status='running',stage='asset_preflight',started_at=now(),gpu_index=gpu,
                    simulator_unit=unit,simulator_output=str(remote_run),controller_output=str(controller),
                    controller_log=str(controller_log),controller_storage_policy='local_during_execution',
                    worker_id=self.c['id'],worker_host=self.c['ssh'][-1],runtime_source=str(self.source),
                    port=port,simulator_interpreter=self.c['sim_python'],source=self.source_info,
                    scheduler_source=getattr(self,'scheduler_info',self.source_info),
                    recording_options=self.c.get('simulator_env',{}),
                    native_fault_capture=native_fault_capture)
        if self.c.get('driver_validation_evidence'):
            self.update(row,driver_validation_evidence=self.c['driver_validation_evidence'],
                        validated_driver_version=self.c['validated_driver_version'])
        self.journal(f'START {runid}; admitted worker {self.c["id"]}; unit {unit}.')
        try:
            asset_cmd=[self.c['sim_python'],str(self.source/'scripts/behavior100_remote.py'),'assets',
                       '--manifest',str(self.root/'manifest.json'),'--index',str(row['index']),
                       '--data-root',self.c['data_root']]
            # Task assets include hashing shared-storage files. A short SSH
            # telemetry timeout can expire before a healthy preflight finishes
            # while other instances initialize; no simulator exists yet.
            checked=self.ssh(asset_cmd,timeout=self._config.get('asset_preflight_timeout_seconds',180))
            if checked.returncode:raise RuntimeError('Task asset preflight command failed')
            asset_check=json.loads(checked.stdout)
            write_json(self.root/'preflight'/f'{runid}_assets.json',asset_check)
            if asset_check['status']!='passed':
                self.update(row,termination_reason='invalid_task_assets')
                raise RuntimeError('Task assets missing or incompatible; simulator was not started')
            state=self.unit_state(unit)
            if state.get('ActiveState') in {'active','activating','deactivating'}:
                raise RuntimeError('Unit collision; refusing to reuse')
            if self.ssh(['test','-e',str(remote_run)]).returncode==0:
                raise RuntimeError('Output exists; immutable attempt cannot be overwritten')
            # Initialization has its own watchdog; the execution clock is
            # armed after MCP handshake, with no duration in step-only mode.
            seconds=(None if self.manifest.get('episode_budget_policy') in {'agent_steps','unbounded'} else
                     self._config.get('episode_timeout_seconds',self.manifest.get('model_timeout_seconds',1800)))
            startup_seconds=self._config.get('startup_timeout_seconds',1800)
            started=time.time();startup_deadline=started+startup_seconds
            clock_path=remote_run/'execution_clock.json'
            grace=self._config.get('cleanup_grace_seconds',120)
            launcher=self.root/'launchers'/f'{runid}.sh';q=shlex.quote
            command=[self.c['sim_python'],str(getattr(self,'scheduler_source',self.source)/'scripts/behavior100_remote.py'),'simulate',
                '--manifest',str(self.root/'manifest.json'),'--index',str(row['index']),'--run-id',runid,
                '--gpu',str(gpu),'--port',str(port),'--unit',unit,'--data-root',self.c['data_root'],
                '--memory-budget-gib',str(self.c.get('memory_budget_gib',28)),
                '--min-free-gpu-mib',str(self.c.get('minimum_free_gpu_mib',0)),
                '--light']
            command += runtime_storage_args(self.c)
            command += driver_admission_args(self.c)
            command += gpu_memory_admission_args(self.c)
            command += self.memory_policy_args()
            # Optional external first-fault capture; no in-process tracing thread.
            # GDB follows the helper's execv into vision_cli and preserves normal
            # exit codes. The capture script creates files only on a native fault.
            fault_env=''
            if native_fault_capture:
                command=['/usr/bin/gdb','--batch','-q','-nx','-x',
                    str(self.source/'scripts/capture_native_fault.gdb'),'--args',*command]
                fault_env='\nexport MAS_NATIVE_FAULT_DIR='+q(str(remote_run))
            overrides=''.join('\nexport '+key+'='+q(str(value)) for key,value in self.c.get('simulator_env',{}).items())
            appdata=self.c.get('simulator_appdata_path',self.c['data_root']+'/cache/behavior100/gpu'+str(gpu))
            scratch=('\nexport TMPDIR='+q(self.c['simulator_tmpdir'])+
                     '\nmkdir -p "$TMPDIR"') if self.c.get('simulator_tmpdir') else ''
            launcher.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport GAP_BEHAVIOR_GPU_ID='+str(gpu)+
                '\nsource '+q(self.c['sim_env'])+overrides+fault_env+'\nunset MAS_EPISODE_DEADLINE_UNIX'+
                '\nexport MAS_EXECUTION_CLOCK_PATH='+q(str(clock_path))+
                '\nexport PYTHONPATH='+q(str(self.source/'src'))+':${PYTHONPATH:-}\nexport OMNIGIBSON_APPDATA_PATH='+q(appdata)+
                '\nmkdir -p "$OMNIGIBSON_APPDATA_PATH"'+scratch+'\nexec '+shlex.join(command)+'\n')
            launch=['systemd-run','--user',f'--unit={unit}',f'--description=BEHAVIOR100 owned {runid}',
                '-p','MemoryMax='+self.c.get('simulator_memory_max',str(self.c.get('memory_budget_gib',28))+'G'),
                '-p','CPUQuota=800%',
                '-p','TasksMax=2048','-p','LimitCORE=0','-p','TimeoutStopSec=30',
                '-p','KillMode=control-group','-p','SuccessExitStatus=2','-p','WorkingDirectory='+str(self.source)]
            launch+=inline_systemd_launcher(launcher.read_text())
            lease=getattr(self._worker_local,'lease',None)
            if lease:lease.record(run_id=runid,unit=unit,released=False,runtime_stage='starting')
            self.update(row,stage='simulator_starting',simulator_started_at_unix=started,
                        startup_deadline_unix=startup_deadline,startup_timeout_seconds=startup_seconds,
                        startup_watchdog_policy=STARTUP_POLICY,
                        execution_clock_path=str(clock_path),deadline_origin='model_start_after_mcp_handshake',
                        budget_basis=self.manifest.get('episode_budget_policy') if self.manifest.get('episode_budget_policy') in {'agent_steps','unbounded'} else 'model_execution_excludes_initialization',
                        episode_budget_policy=self.manifest.get('episode_budget_policy'),
                        max_agent_steps=self.manifest.get('max_agent_steps'),
                        episode_timeout_seconds=seconds,
                        cleanup_grace_seconds=grace,port=port,simulator_interpreter=self.c['sim_python'],
                        source=getattr(self,'source_info',None) or source_version())
            launch_requested=True
            state=self.confirm_simulator_launch(row,launch)
            own_unit=True
            # The shared lease already publishes the starting unit and its
            # estimate. Other coordinators can now admit another initialization.
            if self._config.get('memory_admission_profile')=='throughput':
                self.release_startup_gate()
            check=getattr(self._worker_local,'admission',{})
            self.update(row,stage='simulator_starting',simulator_owned=True,simulator_pid=int(state.get('MainPID',0)),
                gpu_uuid=check.get('gpu_uuid',self.c.get('expected_gpu_uuid')),simulator_host=check.get('host'),
                simulator_interpreter=self.c['sim_python'],source=getattr(self,'source_info',None) or source_version(),
                port=port,simulator_started_at_unix=started,startup_deadline_unix=startup_deadline,
                episode_timeout_seconds=seconds,cleanup_grace_seconds=grace)
            self.worker_state('starting',run_id=runid,startup_deadline_unix=startup_deadline)
            while True:
                self.refresh_startup_progress(row)
                if time.time()>=row['startup_deadline_unix']:
                    self.update(row,startup_timed_out=True,termination_reason='startup_timeout')
                    raise TimeoutError('Simulator initialization made no milestone progress before watchdog expiry')
                state=self.initializing_unit_state(unit)
                health=self.health(port)
                if health and health.get('ready') and not health.get('closed'):
                    if {t['name'] for t in health['tools']}!={t['name'] for t in tool_specs(self.manifest['agent_profile'])}:
                        raise RuntimeError('Bridge tool profile mismatch')
                    write_json(self.root/'logs'/f'{runid}_health.json',health)
                    break
                time.sleep(min(2,max(0,row['startup_deadline_unix']-time.time())))
            self.update(row,stage='controller_starting',bridge_ready_at=now())
            if lease:lease.record(runtime_stage='ready')
            self.release_startup_gate()
            self.worker_state('controller_starting',run_id=runid,startup_deadline_unix=row['startup_deadline_unix'])
            remote=shlex.join(['env','PYTHONPATH='+str(self.source/'src'),self.c['sim_python'],
                '-m','manipulation_agent.mcp_server','--bridge',f'http://127.0.0.1:{port}'])
            cmd=controller_prefix(self.source,self.manifest,self.c)+['--model',self.manifest['model'],
                 '--instruction',row['instruction'],'--mcp-command',self.c['ssh'][0],
                 '--mcp-args-json',json.dumps(self.c['ssh'][1:]+[remote]),'--output',str(controller),
                 '--timeout',str(seconds if seconds is not None else 0),'--execution-clock-command-json',
                 json.dumps(self.execution_clock_command(clock_path,seconds,row=row)),
                 '--agent-profile',self.manifest['agent_profile'],'--goal-skill',self.manifest.get('goal_skill','hierarchical')]
            cmd+=controller_options(self.manifest,self.c)
            self.record_inference_pending(row,controller)
            with controller_log.open('w') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=self.environment)
                self.record_inference_controller(row,process.pid,controller)
                self.update(row,controller_wrapper_pid=process.pid,controller_host=os.uname().nodename,
                            controller_interpreter=sys.executable)
                self.wait_controller(row,process,controller)
                self.update(row,controller_wrapper_exit_code=process.returncode,stage='finishing')
            metadata=self.controller_metadata(row,controller)
            if not metadata:self.update(row,failure='Controller metadata unavailable',failure_stage='controller_launch')
            metadata=self.record_controller_end(row,controller,metadata)
            self.finish_and_wait(row,metadata)
        except Exception as exc:
            deadline=row.get('episode_deadline_unix')
            expired=deadline is not None and time.time()>=deadline
            if deadline is None and not row.get('execution_started_at_unix') and startup_deadline is not None and time.time()>=row['startup_deadline_unix']:
                self.update(row,startup_timed_out=True,termination_reason='startup_timeout')
            self.update(row,failure=f'{type(exc).__name__}: {exc}',failure_stage=row.get('stage'),
                        episode_timed_out=bool(row.get('episode_timed_out') or expired))
            (self.root/'logs'/f'{runid}_supervisor_error.log').write_text(traceback.format_exc())
            if process is not None and process.poll() is None:self.stop_controller(process,controller)
            if own_unit:
                try:self.finish_and_wait(row,{})
                except Exception as finish_error:
                    self.update(row,finish_failure=f'{type(finish_error).__name__}: {finish_error}')
        finally:
            # An SSH timeout may hide a successful remote systemd launch. The
            # declared unit remains reserved until cleanup is positively verified.
            if launch_requested:own_unit=True
            if own_unit:self.cleanup_owned_unit(row)
            self.update(row,execution_finished_at=now(),stage='cleanup_pending' if
                (row.get('simulator_cleanup') or {}).get('status')=='blocked' else 'execution_finished')
            if not own_unit:
                self.complete_row(row)
            elif not getattr(self._worker_local,'defer_archive',False):
                if (row.get('simulator_cleanup') or {}).get('status')=='passed':
                    self.finalize_episode(row,controller,remote_run,self.c)
                else:
                    self.update(row,archive_status='waiting_for_cleanup')
        return (controller,remote_run) if own_unit else None

    def finish_and_wait(self, row, metadata):
        """An accepted finish gets time to score, persist, reply and exit."""
        state=self.unit_state(row['simulator_unit'])
        if state.get('ActiveState') in {'inactive','failed'} and str(state.get('MainPID'))=='0':return
        health=self.health(row['port'])
        known=(row.get('episode_closure') or {}).get('phase','running')!='running'
        if not metadata.get('formal_finish_observed') and not (health or {}).get('closed') and not known:
            self.update(row,supervisor_intervention=True,stage='finishing')
            timeout=self._config.get('finish_stage_timeout_seconds',600)
            code=("from manipulation_agent.bridge import rpc; print(rpc("+repr(f'http://127.0.0.1:{row["port"]}')+
                ",'finish',{'outcome':'aborted','reason':'Controller ended or episode deadline reached'},'batch-supervisor-finish',timeout="+str(timeout)+",origin='supervisor'))")
            result=self.ssh(['env','PYTHONPATH='+str(self.source/'src'),self.c['sim_python'],'-c',code],timeout=timeout+5)
            (self.root/'logs'/f'{row["run_id"]}_forced_finish.log').write_text(result.stdout+result.stderr)
            self.update(row,supervisor_finish_exit_code=result.returncode)
        fallback_end=time.time()+self._config.get('finish_stage_timeout_seconds',600)
        while True:
            state=self.unit_state(row['simulator_unit'])
            if state.get('ActiveState') in {'inactive','failed'} and str(state.get('MainPID'))=='0':return
            if self.step_episode_closed(row):
                reason=self.closure_timeout(row)
                if reason:
                    self.update(row,simulator_closure_timeout=reason)
                    return
            elif time.time()>=fallback_end:
                self.update(row,simulator_closure_timeout='finish_unresponsive')
                return
            time.sleep(2)

    def archive(self,row,controller,remote_run):
        dest=self.root/'runs'/row['run_id'];dest.mkdir(exist_ok=True)
        archived_controller=self.root/'controllers'/row['run_id']
        if controller!=archived_controller and controller.is_dir():
            # Only the archival worker touches shared storage here. Failure
            # retains the local raw evidence and cannot cancel a live episode.
            shutil.copytree(controller,archived_controller,dirs_exist_ok=True,symlinks=True)
            self.update(row,controller_archive_output=str(archived_controller))
        if row.get('controller_log'):
            source_log=Path(row['controller_log'])
            if source_log.is_file():shutil.copyfile(source_log,self.root/'logs'/source_log.name)
        log=self.ssh(['journalctl','--user','-u',row['simulator_unit'],'--no-pager','-o','short-iso'],timeout=60)
        (self.root/'logs'/f"{row['run_id']}_simulator.log").write_text(log.stdout+log.stderr)
        # rsync needs the SSH command without the final host argument.
        # Full episodes can contain many camera frames. Bound stalled I/O, not
        # total copy time: a healthy transfer must not lose its replay at 180s.
        # Partial files make a later archival recovery resume the original data.
        cmd=['rsync','-a','--partial','--timeout=180','-e',shlex.join(self.c['ssh'][:-1]),self.c['ssh'][-1]+':'+str(remote_run)+'/',str(dest)+'/']
        p=subprocess.run(cmd,capture_output=True,text=True)
        (self.root/'logs'/f"{row['run_id']}_archive.log").write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError('Episode rsync failed; remote raw output retained')
        simlog=self.root/'logs'/f"{row['run_id']}_simulator.log"
        if simlog.exists():shutil.copyfile(simlog,dest/'simulator.log')
        if not (dest/'run.json').exists():
            self.update(row,failure=row.get('failure') or 'No simulator recorder created',video_validation='unavailable_before_initialization')
            return
        raw=json.loads((dest/'run.json').read_text())
        if raw['status']=='running':
            write_json(dest/'termination.json',{'status':'failed','at':now(),'reason':row.get('failure') or 'Simulator ended without final recorder',
                       'unit':row['simulator_unit'],'unit_state':row.get('unit_final_state'),'final_evaluation_available':False})
        run=read_run(dest)
        if (dest/'closure.json').exists():
            self.update(row,episode_closure=json.loads((dest/'closure.json').read_text()))
        if row.get('outcome_provisional') and isinstance(run.get('task_success'),bool):
            self.update(row,episode_outcome=None,outcome_provisional=False,termination_reason=None)
        self.update(row,**{key:run[key] for key in ('execution_finished_at_unix','deadline_expired_at_finish',
            'evaluation_finished_at_unix','evaluation_finished_after_deadline','episode_deadline_unix','execution_started_at_unix',
            'agent_steps','max_agent_steps','agent_step_budget_exhausted','termination_reason') if key in run})
        if run.get('deadline_expired_at_finish') is True or run.get('episode_outcome')=='timeout':
            self.update(row,episode_timed_out=True)
        if run.get('failure') and run.get('task_success') is None:
            self.update(row,failure=row.get('failure') or 'Simulator runtime failure: '+str(run['failure']),
                        failure_stage=row.get('failure_stage') or 'simulator_runtime',
                        simulator_failure_type=run['failure'])
        self.update(row,task_success=run.get('task_success'),actions=run.get('actions'),tool_calls=run.get('tool_calls'),
                    sim_steps=run.get('sim_steps'),evaluation=evaluation_summary(run.get('evaluation')),
                    evaluation_record='runs/'+row['run_id']+'/run.json',
                    q_score=(run.get('evaluation') or {}).get('official_metrics',{}).get('q_score',{}).get('final'),
                    custom_score=((run.get('evaluation') or {}).get('goal_satisfaction_fraction')
                                  if (run.get('evaluation') or {}).get('score_type')=='custom_joint_terminal_and_public_history_v1' else None),
                    run_source=run.get('source'),backend=run.get('backend'),agent_outcome=run.get('agent_outcome'),finish_reason=run.get('finish_reason'))
        errors=[]
        for script,filename in [('validate_async_observation.py','observation_validation.json'),('validate_episode_video.py','video_validation.json')]:
            cmd=[sys.executable,str(self.source/'scripts'/script)]
            if script=='validate_async_observation.py': cmd.extend(['--allow-sync-only','--run-dir'])
            cmd.append(str(dest))
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=300,env=self.environment)
            (dest/(script+'.log')).write_text(p.stdout+p.stderr)
            # Failed validation never gets converted to success merely because the file exists.
            if not (dest/filename).exists(): write_json(dest/filename,{'status':'failed','reason':'Validator could not complete','exit_code':p.returncode})
        controller_ready=(controller/'controller.json').is_file() and (controller/'model_events.jsonl').is_file()
        if self.c.get('export_spectator_video'):
            media_validation_field='spectator_video_validation'
            try:
                from manipulation_agent.spectator_video import export_spectator_video
                spectator = export_spectator_video(dest)
                self.update(row,spectator_video_validation=spectator['status'])
                if controller_ready:
                    media_validation_field='spectator_walltime_validation'
                    from manipulation_agent.spectator_video import export_spectator_walltime
                    walltime_spectator = export_spectator_walltime(dest,controller)
                    self.update(row,spectator_walltime_validation=walltime_spectator['status'])
            except Exception as exc:
                errors.append(media_validation_field+' failed: '+str(exc))
                self.update(row,**{media_validation_field:'failed'})
        try:
            replay=render_replay(dest,controller if controller_ready else None)
        except Exception as exc:
            errors.append('Controller/replay join failed: '+str(exc))
            replay=render_replay(dest)
        video=json.loads((dest/'video_validation.json').read_text())
        audit=replay.get('audit') or {}
        public=self.reports/'manipulation_runs'/row['run_id']
        # This directory contains simulator artifacts and public-only model export.
        # Raw controller stream and authentication remain outside the web root.
        shutil.copytree(dest,public,dirs_exist_ok=True)
        prefix='manipulation_runs/'+row['run_id']+'/'
        observation=json.loads((dest/'observation_validation.json').read_text())
        self.update(row,video_validation=video['status'],observation_validation=observation['status'],evidence_alignment=audit.get('evidence_alignment','unavailable'),
                    model_finish=audit.get('model_finish','unavailable'),controller_exit=audit.get('controller_exit','unavailable'),
                    supervisor_finish_calls=audit.get('supervisor_finish_calls'),
                    planning_metrics=replay.get('planning_metrics'),
                    model_usage=audit.get('usage'),replay_url=prefix+'replay.html',record_url=prefix+'run.json',
                    video_url=prefix+'episode.mp4' if (dest/'episode.mp4').exists() else None,
                    spectator_video_url=prefix+'spectator.mp4' if (dest/'spectator.mp4').exists() else None,
                    replay_errors=errors,recording_warnings=sum(1 for _ in (dest/'recording_warnings.jsonl').open()) if (dest/'recording_warnings.jsonl').exists() else 0,video_frames=video.get('frame_count'),video_seconds=video.get('duration_seconds'))
        # Checksums cover original evidence as well as exports; streamed for large videos.
        hashes=episode_artifact_hashes(row['run_id'],dest,controller)
        write_json(dest/'artifact_hashes.json',hashes)
        shutil.copyfile(dest/'artifact_hashes.json',public/'artifact_hashes.json')

    def adopt_one(self, row):
        """Resume supervision of this exact attempt; never restart its model."""
        controller=Path(row['controller_output']);remote_run=Path(row['simulator_output'])
        if (row.get('stage')=='asset_preflight' and not row.get('simulator_started_at_unix')
                and not row.get('controller_wrapper_pid') and not row.get('controller_pid')):
            # Restart interrupted only a read-only asset check. No launch was
            # submitted, so there is no simulator/controller to finish or stop.
            self.update(row,status='planned',stage='queued',
                prelaunch_coordinator_recovery={'at':now(),'previous_stage':'asset_preflight'})
            self.queue.put(row)
            self.journal('REQUEUE PRELAUNCH '+row['run_id']+'; no simulator or model had been launched.')
            return None
        state=self.adopted_unit_state(row);process=None;metadata={}
        if state.get('ActiveState') in {'active','activating','deactivating'}:
            if state.get('Description')!=f"BEHAVIOR100 owned {row['run_id']}":
                raise RuntimeError('Inflight simulator ownership check failed')
        self.update(row,worker_id=self.c['id'],worker_host=self.c['ssh'][-1],coordinator_adopted_at=now())
        self.journal('ADOPT '+row['run_id']+'; original model, simulator, source and deadline retained.')
        if row.get('status') in FINAL and (row.get('simulator_cleanup') or {}).get('status')=='blocked':
            self.cleanup_owned_unit(row)
            return controller,remote_run
        try:
            path=controller/'controller.json'
            metadata=self.controller_metadata(row,controller)
            if not row.get('episode_deadline_unix') and not row.get('execution_clock_path') and self.manifest.get('episode_budget_policy') not in {'agent_steps','unbounded'}:
                started=datetime.fromisoformat(metadata.get('started_at') or row['started_at']).timestamp()
                row['episode_deadline_unix']=started+self.manifest['model_timeout_seconds']
            row.setdefault('cleanup_grace_seconds',self._config.get('cleanup_grace_seconds',120))
            pid=row.get('controller_wrapper_pid')
            if pid is None and not path.exists() and state.get('ActiveState') in {'active','activating'}:
                process=self.start_adopted_policy(row);pid=process.pid
            while pid and controller_process_alive(pid,controller):
                self.controller_metadata(row,controller)
                self.refresh_startup_progress(row)
                deadline=row.get('episode_deadline_unix')
                cutoff=deadline or (None if row.get('execution_started_at_unix') else row['startup_deadline_unix'])
                if not row.get('execution_started_at_unix') and (row.get('controller_storage') or {}).get('status')=='degraded':cutoff=None
                closing=self.step_episode_closed(row)
                if closing:
                    reason=self.closure_timeout(row)
                    if reason:
                        self.cleanup_controller_after_finish(row,process,controller,reason,wrapper_pid=pid)
                        break
                elif cutoff is not None and time.time()>=cutoff:
                    if deadline:self.update(row,episode_timed_out=True)
                    else:self.update(row,startup_timed_out=True,termination_reason='startup_timeout')
                    if process is not None:self.stop_controller(process,controller)
                    else:
                        cleanup_until=time.time()+(20 if deadline else 0)
                        while controller_process_alive(pid,controller) and time.time()<cleanup_until:time.sleep(.2)
                        if controller_process_alive(pid,controller):self.stop_controller(None,controller,wrapper_pid=pid)
                    break
                self.update(row,heartbeat_at=now());time.sleep(2)
            if process is not None:process.wait(timeout=10)
            if path.exists():metadata=self.controller_metadata(row,controller)
            metadata=self.record_controller_end(row,controller,metadata)
            self.finish_and_wait(row,metadata)
        except Exception as exc:
            if not row.get('execution_started_at_unix') and not row.get('episode_deadline_unix') and row.get('startup_deadline_unix',float('inf'))<=time.time():
                self.update(row,startup_timed_out=True,termination_reason='startup_timeout')
            self.update(row,failure=f'{type(exc).__name__}: {exc}',failure_stage='adopted_controller')
        finally:
            self.cleanup_owned_unit(row)
            self.update(row,execution_finished_at=now())
        if not getattr(self._worker_local,'defer_archive',False) and row['simulator_cleanup']['status']=='passed':
            self.finalize_episode(row,controller,remote_run,self.c)
        return controller,remote_run

    def start_adopted_policy(self, row):
        """Wait for an original initializing simulator, then start its first policy."""
        spec=self._config.get('adopt_startup',{}).get(row['run_id']) or {
            'runtime':row['runtime_source'],'deadline_unix':row.get('startup_deadline_unix') or row['episode_deadline_unix'],
            'simulator_pid':row.get('simulator_pid')}
        runtime=Path(spec['runtime']);deadline=float(spec['deadline_unix'])
        original=subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()
        if original!=row['source']['commit']:raise RuntimeError('Original inflight runtime source mismatch')
        controller=Path(row['controller_output']);unit=row['simulator_unit'];port=row['port']
        if (controller/'controller.json').exists():raise RuntimeError('Refusing to launch a duplicate model controller')
        while True:
            self.refresh_startup_progress(row)
            cutoff=row['startup_deadline_unix'] if row.get('startup_watchdog_policy')==STARTUP_POLICY else deadline
            if time.time()>=cutoff:
                raise TimeoutError('Original simulator startup exceeded its progress watchdog' if
                    row.get('startup_watchdog_policy')==STARTUP_POLICY else
                    'Original simulator startup exceeded its unchanged deadline')
            state=self.initializing_unit_state(unit)
            if state is None:
                time.sleep(4)
                continue
            if state.get('Description')!=f"BEHAVIOR100 owned {row['run_id']}" or int(state['MainPID'])!=spec['simulator_pid']:
                raise RuntimeError('Original simulator identity changed during initialization')
            h=self.health(port)
            if h and h.get('ready') and not h.get('closed'):
                if {t['name'] for t in h['tools']}!={t['name'] for t in tool_specs(self.manifest['agent_profile'])}:
                    raise RuntimeError('Original bridge tool catalog mismatch')
                break
            self.update(row,heartbeat_at=now(),simulator_pid=spec['simulator_pid']);time.sleep(4)
        self.update(row,stage='controller_starting' if row.get('execution_clock_path') else 'policy_running',
                    bridge_ready_at=now(),simulator_pid=spec['simulator_pid'])
        remote=shlex.join(['env','PYTHONPATH='+str(runtime/'src'),self.c['sim_python'],'-m','manipulation_agent.mcp_server','--bridge',f'http://127.0.0.1:{port}'])
        cmd=controller_prefix(runtime,self.manifest,self.c)+['--model',self.manifest['model'],
             '--instruction',row['instruction'],'--mcp-command',self.c['ssh'][0],
             '--mcp-args-json',json.dumps(self.c['ssh'][1:]+[remote]),'--output',str(controller),
             '--timeout',str(self.manifest['model_timeout_seconds'] or 0),'--agent-profile',self.manifest['agent_profile'],
             '--goal-skill',self.manifest.get('goal_skill','hierarchical')]
        cmd+=controller_options(self.manifest,self.c)
        if row.get('execution_clock_path'):
            cmd+=['--execution-clock-command-json',json.dumps(self.execution_clock_command(
                row['execution_clock_path'],row.get('episode_timeout_seconds',self.manifest['model_timeout_seconds']),runtime,row=row))]
        elif row.get('episode_deadline_unix'):cmd+=['--deadline-unix',str(row['episode_deadline_unix'])]
        self.record_inference_pending(row,controller)
        with (self.root/'logs'/f"{row['run_id']}_controller.log").open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                env={**self.environment,'PYTHONPATH':str(runtime/'src')})
        self.record_inference_controller(row,process.pid,controller)
        self.update(row,controller_wrapper_pid=process.pid,controller_host=os.uname().nodename,controller_interpreter=sys.executable)
        self.journal('READY adopted initialization '+row['run_id']+'; first model policy starts against the unchanged original simulator and source.')
        return process

    def requeue_resource_rejection(self,row):
        """Preserve the rejected attempt, then wait for admission again.

        No episode exists when the in-unit check rejects startup. Archiving a
        nonexistent directory used to turn this recoverable race into a final
        task failure. Cleanup must be positively verified before another try.
        """
        if (row.get('simulator_cleanup') or {}).get('status')!='passed':return False
        check_path=self.root/'preflight'/f"{row['run_id']}_in_unit.json"
        try:preflight=json.loads(check_path.read_text())
        except (OSError,ValueError):return False
        failed=startup_resource_rejection(row,self.root/'controllers'/row['run_id'],preflight)
        if not failed:return False
        # Preserve startup logs even though there is no simulator episode.
        log=self.ssh(['journalctl','--user','-u',row['simulator_unit'],'--no-pager','-o','short-iso'],timeout=60)
        (self.root/'logs'/f"{row['run_id']}_simulator.log").write_text(log.stdout+log.stderr)
        identity_keys={'index','task','name','scene','instance','seed','split','instruction',
            'instruction_sha256','instruction_source','bddl_sha256','asset_sha256',
            'goal_predicates_offline_only','candidate_capabilities_offline_only',
            'source_commit','source_tree','evaluation_revision','model_provider_profile',
            'original_instruction','original_instruction_source','original_instruction_sha256',
            'instruction_correction','task_instructions_changed','episode_budget_policy',
            'max_agent_steps','custom_task_spec'}
        with self.lock:
            old={**row,'status':'failed','task_success':None,'failure_kind':'infrastructure_resource_admission',
                 'resource_rejected_checks':failed,'archive_status':'not_applicable_before_simulator_start',
                 'stage':'resource_admission_rejected','updated_at':now()}
            write_json(self.root/'records'/f"{old['run_id']}.json",old)
            previous=[*old.get('previous_attempts',[]),{k:v for k,v in old.items() if k!='previous_attempts'}]
            runid=old['run_id'].rsplit('_r',1)[0]+'_r'+str(int(old.get('attempt',1))+1)
            reason='In-unit dynamic resource rejection before simulator/controller startup: '+', '.join(failed)
            path=self.root/'infrastructure_retries.json'
            selections=json.loads(path.read_text()) if path.exists() else {}
            selections[str(row['index'])]={'run_id':runid,'previous_run_ids':[p['run_id'] for p in previous],
                                          'reason':reason}
            write_json(path,selections)
            identity={k:v for k,v in row.items() if k in identity_keys}
            row.clear();row.update(identity,run_id=runid,status='planned',stage='queued',
                attempt=len(previous)+1,previous_attempts=previous,retry_kind='infrastructure',retry_reason=reason)
            write_json(self.root/'records'/f'{runid}.json',row)
            self.update(row)
            self.queue.put(row)
        self.journal('RESOURCE RETRY '+old['run_id']+' -> '+runid+'; '+reason)
        return True

    def run(self,limit=None):
        self.publish()
        self.queue=queue.Queue()
        planned=[r for r in self.rows if r['status']=='planned']
        selected=planned if limit is None else planned[:limit]
        required_rows=list(self.rows) if limit is None else [r for r in self.rows if r['status']!='planned' or r in selected]
        for row in selected:self.queue.put(row)
        archived=[];archive_lock=threading.Lock()
        archive_slots=threading.BoundedSemaphore(self._config.get('max_pending_archives',4))
        inherited=[r for r in self.rows if
            (r['status']=='running' or (r.get('simulator_cleanup') or {}).get('status')=='blocked') and
            r['run_id'] in self._config.get('adopt_inflight',{}) and
            not (r.get('execution_finished_at') and (r.get('simulator_cleanup') or {}).get('status')=='passed')]
        capacity=ExecutionCapacity(self._config.get('max_active_workers',len(self.workers)),
                                   [r['run_id'] for r in inherited])
        poll=self._config.get('resource_poll_seconds',30)
        def submit_archive(row,paths,config):
            if paths is None:return
            archive_slots.acquire()
            future=self._archive_pool.submit(self.finalize_episode,row,*paths,config)
            future.add_done_callback(lambda _:archive_slots.release())
            with archive_lock:archived.append(future)
        def worker(config):
            self._worker_local.config=config;self._worker_local.defer_archive=True
            time.sleep(config.get('admission_delay_seconds',0))
            inflight=[r for r in inherited if self._config['adopt_inflight'][r['run_id']]==config['id']]
            while inflight or not self.queue.empty():
                adopting=inflight[0] if inflight else None
                if not adopting and self.drain_requested():
                    self.worker_state('drained');return
                if not adopting and getattr(self,'storage',None) and self.storage.status()['local_checkpoint_error']:
                    self.worker_state('waiting_for_local_storage');time.sleep(poll);continue
                capacity_key=adopting['run_id'] if adopting else config['id']
                if not capacity.acquire(capacity_key,adopting=adopting is not None):
                    self.worker_state('waiting_for_capacity');time.sleep(poll);continue
                lease=None;row=None;paths=None;cleared=True
                self._worker_local.inference_lease=None
                try:
                    if not adopting and not self.acquire_startup_gate():continue
                    lease=self.acquire_worker_lease(adopting=adopting)
                    if lease is None:continue
                    self._worker_local.lease=lease
                    if not self.acquire_inference_resources(adopting):continue
                    if adopting:
                        row=inflight.pop(0);cleared=False;paths=self.adopt_one(row)
                    else:
                        if not self.resource_admission():continue
                        if self.drain_requested():continue
                        try:row=self.queue.get_nowait()
                        except queue.Empty:return
                        cleared=False;paths=self.run_one(row,config['gpu'])
                    if (row.get('simulator_cleanup') or {}).get('status')=='blocked':
                        try:self.capture_result(row,Path(row['simulator_output']))
                        except Exception:pass
                        if not isinstance(row.get('task_success'),bool):self.update(row,outcome_provisional=True)
                        self.complete_row(row)
                        self.update(row,stage='cleanup_pending')
                        self.recover_worker_cleanup(row)
                    cleared=(not (row.get('simulator_started_at_unix') or row.get('episode_started_at_unix')) or
                             (row.get('simulator_cleanup') or {}).get('status')=='passed')
                except Exception as exc:
                    self.worker_state('worker_error',reason=f'{type(exc).__name__}: {exc}',run_id=row.get('run_id') if row else None)
                    self.journal('WORKER ERROR '+config['id']+': '+str(exc))
                    if row is not None:
                        self.update(row,failure=f'{type(exc).__name__}: {exc}',failure_stage='worker_supervision')
                        if row.get('simulator_started_at_unix') or row.get('episode_started_at_unix') or adopting:
                            self.cleanup_owned_unit(row);self.recover_worker_cleanup(row)
                            paths=(Path(row['controller_output']),Path(row['simulator_output']))
                        else:self.complete_row(row)
                    elif 'Permanent admission error' in str(exc):
                        self.worker_state('disabled',reason=str(exc));return
                finally:
                    try:
                        inference=getattr(self._worker_local,'inference_lease',None)
                        if inference:inference.release(cleared=True)
                        if lease:lease.release(cleared=cleared)
                    except Exception as exc:
                        if row is not None:self.update(row,lease_metadata_cleanup_error=f'{type(exc).__name__}: {exc}')
                        self.journal('LEASE METADATA CLEANUP '+config['id']+': '+str(exc))
                    finally:
                        self._worker_local.lease=None;self._worker_local.inference_lease=None;self.release_startup_gate()
                        capacity.release(capacity_key,preserve=adopting is not None and adopting in inflight)
                    if row is None:time.sleep(poll)
                # GPU and host capacity are free before any bulk copy or decode.
                if row is not None:
                    if cleared and self.requeue_resource_rejection(row):
                        self.worker_state('waiting_for_resources')
                        time.sleep(poll)
                        continue
                    if cleared:self.worker_state('archive_pending')
                    submit_archive(row,paths,config)
                    if cleared:self.worker_state('available')
                    time.sleep(self._config.get('worker_yield_seconds',1))
            self.worker_state('idle')
        with ThreadPoolExecutor(max_workers=self._config.get('archive_workers',2)) as archive_pool:
            self._archive_pool=archive_pool
            for row in self.rows:
                if row['status']=='running' and row.get('execution_finished_at') and (row.get('simulator_cleanup') or {}).get('status')=='passed':
                    config=next(w for w in self.workers if w['id']==row['worker_id'])
                    submit_archive(row,(Path(row['controller_output']),Path(row['simulator_output'])),config)
            with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
                futures=[pool.submit(worker,config) for config in self.workers]
                for future in futures:
                    try:future.result()
                    except Exception as exc:self.journal('WORKER EXIT: '+str(exc))
            if not self.queue.empty() and not self.drain_requested() and all(
                    self.worker_states.get(w['id'],{}).get('stage')=='disabled' for w in self.workers):
                raise RuntimeError('No usable worker: static admission failed; unstarted tasks remain planned')
            for future in archived:
                try:future.result()
                except Exception as exc:self.journal('ARCHIVE EXIT: '+str(exc))
        self._archive_pool=None
        self.publish()
        require_completed_scope(required_rows,draining=self.drain_requested())


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--publish-only',action='store_true');p.add_argument('--bootstrap-first',action='store_true')
    p.add_argument('--console-progress',action='store_true',help='Print live task stages, model calls and independent results')
    a=p.parse_args();config=json.loads(a.config.read_text());root=Path(config['batch'])
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.console_progress:print('Initializing supervisor; checking source version and model environment...',flush=True)
        batch=Batch(config)
        console=None
        try:
            if a.console_progress:
                from manipulation_agent.console_progress import ConsoleProgress
                console=ConsoleProgress(batch)
                console.start()
            if a.publish_only:batch.publish()
            elif a.bootstrap_first:
                first=next(r for r in batch.rows if r['status']=='planned')
                batch.run(1)
                if first.get('evidence_alignment')!='passed' or first.get('video_validation')!='passed':
                    raise RuntimeError('First episode did not validate the real model/replay pipeline; remaining tasks stay planned')
                batch.journal('Real model/replay pipeline passed; continuing all remaining planned tasks on the configured GPU slots.')
                batch.run(a.limit)
            else:batch.run(a.limit)
        finally:
            if console is not None:console.close()
            if not batch.storage.close(config.get('publication_flush_seconds',30)):
                print('Supervisor metadata sync pending; durable local checkpoint: '+str(batch.storage.spool),file=sys.stderr)
                # Restart-on-failure must retry the durable final checkpoint.
                # A clean exit stranded completed tasks at the last published
                # stage (often "archiving") after a final NAS write failed.
                # Batch recovery retains terminal rows; it does not rerun them.
                raise SystemExit(75)


if __name__=='__main__':main()
