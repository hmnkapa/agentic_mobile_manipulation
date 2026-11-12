import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import threading
import queue
import shlex
import shutil
import subprocess
import sys
import uuid
from subprocess import CompletedProcess
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('batch_runner',Path(__file__).resolve().parents[1]/'scripts/run_behavior100.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)

class BatchEvidenceTests(unittest.TestCase):
    def test_local_manipulation_protocol_is_explicit_and_validated(self):
        config={'workers':[{'id':'local','gpu':0,'ssh':['ssh','fixture'],'base_port':36000}]}
        for value in ('0','1'):
            config['simulator_env']={'MAS_LOCAL_MANIPULATION':value}
            self.assertEqual(batch.worker_configs(config)[0]['simulator_env']['MAS_LOCAL_MANIPULATION'],value)
        for env in ({'MAS_LOCAL_MANIPULATION':'true'},{'MAS_LOCAL_MANIPULATION':1},{'UNREVIEWED_OPTION':'1'}):
            config['simulator_env']=env
            with self.assertRaises(ValueError):batch.worker_configs(config)

    def test_fresh_batch_publish_only_creates_journal_and_preserves_existing_history(self):
        for history in (None, '# Existing experiment\n\n- Original failure evidence.\n'):
            with self.subTest(existing_journal=history is not None), tempfile.TemporaryDirectory() as folder:
                root=Path(folder);reports=root/'report'
                manifest={'comparison_arm':'original','standalone_batch':True,'tasks':[
                    {'index':0,'task':'fixture_task','name':'Fixture task','run_id':'fixture',
                     'status':'planned','stage':'queued'}]}
                (root/'manifest.json').write_text(json.dumps(manifest))
                config={'batch':str(root),'reports':str(reports),'expected_task_count':1,
                        'supervisor_spool_dir':str(root/'spool'),
                        'workers':[{'id':'fixture-worker','gpu':0,'ssh':['ssh','fixture-host'],
                                    'base_port':36000,'sim_python':sys.executable}]}
                path=root/'config.json';path.write_text(json.dumps(config))
                if history is not None:(root/'journal.md').write_text(history)
                # Exercise the real CLI initialization and publication. Only
                # checkout identity and the CLI --version probe are fixtures.
                with patch.object(sys,'argv',['run_behavior100.py','--config',str(path),'--publish-only','--console-progress']), \
                        patch.object(batch,'source_version',return_value={'commit':'fixture','dirty':False}), \
                        patch.object(batch.shutil,'which',return_value='/fixture/codex'), \
                        patch.object(batch.subprocess,'run',return_value=CompletedProcess([],0)) as command:
                    batch.main()
                command.assert_called_once_with(['/fixture/codex','--version'],check=True,
                                                capture_output=True,text=True,timeout=20)
                self.assertEqual((reports/'behavior100/journal.md').read_text(),history or '')
                self.assertEqual((root/'journal.md').read_text(),history or '')
                progress=json.loads((reports/'behavior100/progress.json').read_text())
                self.assertEqual(progress['summary']['completed'],0)
                self.assertEqual(progress['summary']['counts'],{'planned':1})
                self.assertIn('Fixture task',(reports/'behavior100.html').read_text())

    def test_remote_assets_selects_benchmark_index_in_single_task_manifest(self):
        spec=importlib.util.spec_from_file_location('remote_subset_check',
            Path(__file__).resolve().parents[1]/'scripts/behavior100_remote.py')
        remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
        row={'index':93,'task':'clean_a_keyboard','instance':301,'seed':0}
        with tempfile.TemporaryDirectory() as folder:
            manifest=Path(folder)/'manifest.json'
            manifest.write_text(json.dumps({'tasks':[row]}))
            argv=['behavior100_remote.py','assets','--manifest',str(manifest),
                  '--index','93','--data-root',folder]
            with patch.object(sys,'argv',argv), patch.object(remote,'assets',return_value={'status':'passed'}) as assets, patch('builtins.print'):
                remote.main()
            assets.assert_called_once_with({'tasks':[row]},Path(folder))

    def make_queue_runner(self, folder, workers=None, count=3):
        runner=batch.Batch.__new__(batch.Batch)
        runner.root=Path(folder);runner._worker_local=threading.local()
        # Existing configs need not be rewritten to remove the excessive drain.
        runner._config={'stop_on_infrastructure_failure':True,'id':'one','gpu':1,'ssh':['ssh','fixture'],
                        'resource_poll_seconds':.005,'cleanup_retry_seconds':.005,'worker_yield_seconds':0}
        runner.source_info={'commit':'fixture','dirty':False};runner.worker_states={}
        runner.resource_admission=Mock(return_value=True)
        runner.acquire_worker_lease=Mock(side_effect=lambda **kw:Mock())
        runner.workers=workers or [{'id':'one','gpu':1,'ssh':['ssh','fixture']}]
        runner.lock=threading.RLock();runner.queue=queue.Queue()
        runner.publish=Mock();runner.journal=Mock()
        runner.update=lambda row,**values:row.update(values)
        runner.rows=[{'run_id':str(i),'status':'planned'} for i in range(count)]
        return runner

    def test_episode_failures_remain_recorded_and_next_tasks_continue(self):
        failures=[{'task_success':False}, {'task_success':None},
                  {'video_validation':'failed'}, {'observation_validation':'failed'},
                  {'evidence_alignment':'failed'}, {'archive_failure':'copy interrupted'},
                  {'controller_status':'failed','failure':'Model timeout',
                   'supervisor_intervention':True,'evidence_alignment':'failed'}]
        for failure in failures:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as folder:
                runner=self.make_queue_runner(folder)
                def run_one(row,gpu):
                    row.update(status='failed',task_success=False,video_validation='passed',
                               observation_validation='passed',evidence_alignment='passed',
                               simulator_cleanup={'status':'passed'})
                    row.update(failure)
                runner.run_one=Mock(side_effect=run_one)
                runner.run()
                self.assertEqual(runner.run_one.call_count,3)
                self.assertFalse((runner.root/'drain_requested.json').exists())
                for row in runner.rows:
                    self.assertEqual(row['status'],'failed')
                    for key,value in failure.items():self.assertEqual(row[key],value)

    def test_lower_capacity_adopts_every_existing_execution_before_new_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            # Put the fresh lane first to exercise reservation before adoption.
            workers=[{'id':name,'gpu':i,'ssh':['ssh','fixture']}
                     for i,name in enumerate(('fresh','a','b','c'))]
            runner=self.make_queue_runner(folder,workers=workers,count=1)
            runner._config.update(max_active_workers=1,
                adopt_inflight={name:name for name in ('a','b','c')})
            runner.rows.extend({'run_id':name,'status':'running'} for name in ('a','b','c'))
            entered=threading.Barrier(4)
            release={name:threading.Event() for name in ('a','b','c')}
            completed={name:threading.Event() for name in ('a','b','c')}
            errors=[]
            runner.acquire_startup_gate=Mock(return_value=True)
            runner.release_startup_gate=Mock()
            runner.requeue_resource_rejection=Mock(return_value=False)
            def adopt(row):
                entered.wait(timeout=3)
                if not release[row['run_id']].wait(3):
                    raise RuntimeError('Test did not release adopted execution')
                row.update(status='passed',simulator_cleanup={'status':'passed'})
            runner.adopt_one=Mock(side_effect=adopt)
            def state(stage,**kwargs):
                if stage=='available' and runner.c['id'] in completed:
                    completed[runner.c['id']].set()
            runner.worker_state=Mock(side_effect=state)
            runner.run_one=Mock(side_effect=lambda row,gpu:row.update(status='passed'))
            def run():
                try:runner.run()
                except Exception as exc:errors.append(exc)
            thread=threading.Thread(target=run)
            thread.start()
            try:
                entered.wait(timeout=3)
                runner.run_one.assert_not_called()
                for name in ('a','b'):
                    release[name].set()
                    self.assertTrue(completed[name].wait(3))
                # One inherited execution still fills the reduced capacity.
                runner.run_one.assert_not_called()
            finally:
                for event in release.values():event.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors,[])
            self.assertEqual(runner.adopt_one.call_count,3)
            runner.run_one.assert_called_once()

    def test_failed_adoption_keeps_capacity_reserved_for_retry(self):
        capacity=batch.ExecutionCapacity(1,['existing'])
        self.assertFalse(capacity.acquire('new'))
        self.assertTrue(capacity.acquire('existing',adopting=True))
        capacity.release('existing',preserve=True)
        self.assertFalse(capacity.acquire('new'))
        self.assertTrue(capacity.acquire('existing',adopting=True))
        capacity.release('existing')
        self.assertTrue(capacity.acquire('new'))
        capacity.release('new')

    def test_explicit_drain_stops_new_admission(self):
        for before_start in (False,True):
            with self.subTest(before_start=before_start),tempfile.TemporaryDirectory() as folder:
                runner=self.make_queue_runner(folder)
                drain=runner.root/'drain_requested.json'
                request={'automatic':False,'reason':'User requested pause'}
                if before_start:drain.write_text(json.dumps(request))
                def run_one(row,gpu):
                    row.update(status='failed',task_success=False)
                    drain.write_text(json.dumps(request))
                runner.run_one=Mock(side_effect=run_one)
                runner.run()
                self.assertEqual(runner.run_one.call_count,0 if before_start else 1)
                self.assertEqual(json.loads(drain.read_text()),request)
                self.assertEqual(runner.rows[-1]['status'],'planned')

    def test_unresolved_cleanup_holds_only_affected_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            workers=[{'id':'blocked','gpu':1,'ssh':['ssh','fixture']},
                     {'id':'healthy','gpu':2,'ssh':['ssh','fixture']}]
            runner=self.make_queue_runner(folder,workers,count=5)
            blocked=threading.Event();healthy_done=threading.Event();healthy_count=[]
            def cleanup(row):
                self.assertTrue(healthy_done.wait(5))
                row['simulator_cleanup']={'status':'passed'}
            runner.cleanup_owned_unit=cleanup
            def run_one(row,gpu):
                row.update(status='failed',task_success=False)
                if gpu==1:
                    row.update(simulator_unit='owned.service',
                               simulator_cleanup={'status':'blocked','reason':'Still running'})
                    blocked.set()
                else:
                    self.assertTrue(blocked.wait(5))
                    row.update(simulator_cleanup={'status':'passed'})
                    healthy_count.append(row)
                    if len(healthy_count)==4:healthy_done.set()
            runner.run_one=Mock(side_effect=run_one)
            runner.run()
            calls=runner.run_one.call_args_list
            self.assertEqual(sum(call.args[1]==1 for call in calls),1)
            self.assertEqual(sum(call.args[1]==2 for call in calls),4)
            self.assertFalse((runner.root/'drain_requested.json').exists())
            self.assertTrue(all(row['status']=='failed' for row in runner.rows))
            self.assertTrue(healthy_done.is_set())
            self.assertTrue(all(row['simulator_cleanup']['status']=='passed' for row in runner.rows))

    def test_owned_cleanup_verifies_exit_before_reusing_lane(self):
        owned={'ActiveState':'active','MainPID':'123','Description':'BEHAVIOR100 owned run1'}
        terminal={'ActiveState':'inactive','MainPID':'0','Description':'owned.service'}
        cases=[('stopped',[owned,terminal],0,'passed',1),
               ('already_terminal',[terminal],0,'passed',0),
               ('failed_terminal',[{**terminal,'ActiveState':'failed'}],0,'passed',0),
               ('still_running',[owned,owned],0,'blocked',1),
               ('stop_failed',[owned],1,'blocked',1),
               ('ownership_changed',[{**owned,'Description':'unrelated'}],0,'blocked',0),
               ('unverified_state',[{}],0,'blocked',0),
               ('live_pid',[{**terminal,'MainPID':'123'}],0,'blocked',0),
               ('ssh_error',[TimeoutError('SSH timed out')],0,'blocked',0)]
        for label,states,returncode,expected,stop_calls in cases:
            with self.subTest(case=label),tempfile.TemporaryDirectory() as folder:
                runner=self.make_queue_runner(folder)
                runner.unit_state=Mock(side_effect=states)
                runner.ssh=Mock(return_value=CompletedProcess([],returncode,'','stop error'))
                row={'run_id':'run1','simulator_unit':'owned.service'}
                result=runner.cleanup_owned_unit(row)
                self.assertEqual(result['status'],expected)
                self.assertEqual(row['simulator_cleanup'],result)
                self.assertEqual(runner.ssh.call_count,stop_calls)
                if expected=='blocked':self.assertTrue(result['reason'])

    def test_progress_preserves_scores_without_copying_combinatorial_goal_arrays(self):
        goals = [[True, False] for _ in range(10000)]
        raw = {'goal_options': goals, 'initial_goal_options': goals,
               'official_metrics': {'q_score': {'final': .5}}, 'task_success': False}
        summary = batch.evaluation_summary(raw)
        self.assertEqual(summary['official_metrics']['q_score']['final'], .5)
        self.assertIs(summary['task_success'], False)
        self.assertLess(len(json.dumps(summary)), 200)
        self.assertIs(raw['goal_options'], goals)
        self.assertEqual(len(raw['initial_goal_options']), 10000)
        self.assertIsNone(batch.evaluation_summary(None))

    def test_missing_assets_block_before_gpu_wait_or_simulator_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for directory in ('logs','preflight'):(root/directory).mkdir()
            runner=batch.Batch.__new__(batch.Batch)
            runner.root=root;runner.source=Path(__file__).resolve().parents[1];runner.source_info={'commit':'fixture'}
            runner.storage=Mock(spool=root/'local-spool')
            runner._worker_local=threading.local()
            runner._config={'id':'fixture','base_port':31000,'batch_tag':'test','data_root':'/fixture',
                            'sim_python':'/fixture/python','ssh':['ssh','fixture']}
            runner.update=lambda row,**values:row.update(values)
            runner.journal=Mock();runner.publish=Mock()
            asset={'status':'failed','tasks':[{'runtime_assets':{'missing_model_usds':['missing.usd']}}]}
            runner.ssh=Mock(return_value=CompletedProcess([],0,json.dumps(asset),''))
            row={'index':0,'run_id':'fixture_r1','task':'fixture'}
            runner.run_one(row,0)
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['failure_stage'],'asset_preflight')
            self.assertIsNone(row.get('task_success'))
            self.assertEqual(runner.ssh.call_count,1)
            self.assertIn('assets',runner.ssh.call_args.args[0])
            self.assertEqual(json.loads((root/'preflight/fixture_r1_assets.json').read_text()),asset)

    def test_simulator_launch_sends_retained_script_instead_of_shared_path(self):
        with tempfile.TemporaryDirectory() as folder:
            runner=self.make_queue_runner(folder)
            runner.storage=Mock(spool=Path(folder)/'local-spool')
            for directory in ('logs','preflight','launchers'):(runner.root/directory).mkdir()
            runner.source=Path('/source snapshot')
            runner.manifest={'simulator_runtime_max_seconds':1800}
            runner._config.update(base_port=31000,batch_tag='test',data_root='/fixture',
                                  sim_python='/fixture/python',sim_env="/fixture/source '$env.sh",
                                  ssh=['ssh','fixture'])
            runner.unit_state=Mock(return_value={'ActiveState':'inactive','MainPID':'0'})
            def ssh(args,**kwargs):
                if 'assets' in args:return CompletedProcess(args,0,'{"status":"passed"}','')
                if 'preflight' in args:return CompletedProcess(args,0,'{"status":"passed","checks":{}}','')
                return CompletedProcess(args,1,'','fixture stops after capturing launch')
            runner.ssh=Mock(side_effect=ssh)
            row={'index':0,'run_id':'fixture_r1','task':'fixture'}
            runner.run_one(row,0)
            launch=next(call.args[0] for call in runner.ssh.call_args_list if call.args[0][0]=='systemd-run')
            retained=(runner.root/'launchers/fixture_r1.sh').read_text()
            self.assertEqual(launch[-3:],batch.inline_systemd_launcher(retained))
            self.assertNotIn(str(runner.root/'launchers/fixture_r1.sh'),launch)
            self.assertIn('${PYTHONPATH:-}',retained)
            self.assertIn('unset MAS_EPISODE_DEADLINE_UNIX',retained)
            self.assertIn('export MAS_EXECUTION_CLOCK_PATH=',retained)
            self.assertNotIn('export MAS_EPISODE_DEADLINE_UNIX=',retained)
            self.assertEqual(shlex.split(shlex.join(launch)),launch)
            self.assertIs(row['native_fault_capture'],False)
            self.assertNotIn('MAS_NATIVE_FAULT_DIR',retained)
            self.assertNotIn('--skip-gpu-memory-admission',retained)

            runner._config['native_fault_capture']=True
            runner._config['simulator_gpu_memory_admission']=False
            captured={'index':0,'run_id':'fixture_r2','task':'fixture'}
            runner.run_one(captured,0)
            retained=(runner.root/'launchers/fixture_r2.sh').read_text()
            command=shlex.split(next(line[5:] for line in retained.splitlines() if line.startswith('exec ')))
            self.assertEqual(command[:5],['/usr/bin/gdb','--batch','-q','-nx','-x'])
            self.assertEqual(command[5],'/source snapshot/scripts/capture_native_fault.gdb')
            self.assertEqual(command[6:9],['--args','/fixture/python','/source snapshot/scripts/behavior100_remote.py'])
            self.assertIn('--skip-gpu-memory-admission',command)
            self.assertIn('export MAS_NATIVE_FAULT_DIR=/fixture/runs/fixture_r2',retained)
            self.assertIs(captured['native_fault_capture'],True)

    def test_inline_launcher_preserves_shell_bytes_through_real_systemd(self):
        if not shutil.which('systemd-run') or not shutil.which('systemctl'):
            self.skipTest('systemd user service manager unavailable')
        manager=subprocess.run(['systemctl','--user','show','--property=Version'],capture_output=True,timeout=10)
        if manager.returncode:self.skipTest('systemd user service manager unavailable')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);env_path=root/"environment '$literal.sh"
            value="spaces 'quotes' \"double\" $VAR $(printf wrong) `printf wrong`\nnext line"
            env_path.write_text('export LAUNCHER_VALUE='+shlex.quote(value)+'\n'
                                'export PYTHONPATH='+shlex.quote('from sourced environment')+'\n')
            probe=root/'probe.py'
            probe.write_text('import json,os,sys\nprint(json.dumps(dict(argument=sys.argv[1],'
                             'value=os.environ["LAUNCHER_VALUE"],pythonpath=os.environ["PYTHONPATH"])))\n')
            prefix="source '$path with spaces"
            script='#!/bin/bash\nset -euo pipefail\nsource '+shlex.quote(str(env_path))+'\n'
            script+='export PYTHONPATH='+shlex.quote(prefix)+':${PYTHONPATH:-}\n'
            script+='exec '+shlex.join([sys.executable,str(probe),value])+'\n'
            archived=root/'launcher.sh';archived.write_text(script)
            command=['systemd-run','--user','--quiet','--wait','--pipe','--collect',
                     '--unit=mas-test-inline-launcher-'+uuid.uuid4().hex]
            command+=batch.inline_systemd_launcher(archived.read_text())
            archived.unlink()  # No launcher file is visible to the service host.
            # Bash parsing here is the same additional quoting layer as SSH.
            result=subprocess.run(['/bin/bash','-c','exec '+shlex.join(command)],
                                  capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),
                             {'argument':value,'value':value,'pythonpath':prefix+':from sourced environment'})

    def test_adopted_controller_outside_batch_keeps_stable_hash_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'original/runs/run1';run.mkdir(parents=True)
            ctrl=root/'other/controllers/run1';ctrl.mkdir(parents=True)
            (run/'run.json').write_text('{}');(ctrl/'model_events.jsonl').write_text('{}\n')
            hashes=batch.episode_artifact_hashes('run1',run,ctrl)
            self.assertEqual(set(hashes),{'runs/run1/run.json','controllers/run1/model_events.jsonl'})
            self.assertTrue(all(len(value)==64 for value in hashes.values()))

    def test_model_decision_prevents_infrastructure_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);record={'status':'failed','actions':0,'controller_pid':123}
            stream=path/'model_events.jsonl'
            stream.write_text(json.dumps({'type':'item.completed','item':{'type':'error'}})+'\n')
            self.assertTrue(batch.prepolicy_failure(record,path))
            for kind in ('agent_message','reasoning','mcp_tool_call'):
                stream.write_text(json.dumps({'type':'item.started','item':{'type':kind}})+'\n')
                self.assertFalse(batch.prepolicy_failure(record,path))

    def test_missing_action_count_before_simulator_start_can_retry_without_record_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            for stage in ('asset_preflight','simulator_starting'):
                record={'status':'failed','stage':'complete','failure_stage':stage,
                        'simulator_pid':1678698,'failure':'quota subprocess timed out',
                        'execution_clock_path':'/remote/not-yet-created/execution_clock.json'}
                original=dict(record)
                self.assertTrue(batch.prepolicy_failure(record,path))
                self.assertEqual(record,original)
                self.assertNotIn('actions',record)
                archived={**record,'actions':None}
                self.assertTrue(batch.prepolicy_failure(archived,path))
                self.assertIsNone(archived['actions'])

    def test_missing_action_count_never_allows_retry_after_controller_or_policy_start(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);record={'status':'failed','failure_stage':'simulator_starting'}
            for changed in ({'failure_stage':None},{'failure_stage':'controller_starting'},
                            {'controller_pid':123},{'controller_wrapper_pid':123},
                            {'controller_status':'failed'},{'execution_started_at_unix':0},
                            {'episode_deadline_unix':123},{'bridge_ready_at':'timestamp'}):
                for counter in ({},{'actions':None}):
                    with self.subTest(changed=changed,counter=counter):
                        self.assertFalse(batch.prepolicy_failure({**record,**counter,**changed},path))
            stream=path/'model_events.jsonl'
            for kind in ('agent_message','reasoning','mcp_tool_call','error'):
                for counter in ({},{'actions':None}):
                    with self.subTest(kind=kind,counter=counter):
                        stream.write_text(json.dumps({'type':'item.started','item':{'type':kind}})+'\n')
                        self.assertFalse(batch.prepolicy_failure({**record,**counter},path))
            stream.unlink()
            (path/'controller.json').write_text('{}')
            self.assertFalse(batch.prepolicy_failure(record,path))

    def test_infrastructure_retry_retains_first_attempt_failure(self):
        row={'status':'passed','task_success':True,'previous_attempts':[{'status':'failed','task_success':False}]}
        result=batch.summarize([row])
        self.assertEqual(result['total'],1)
        self.assertEqual(result['task_successes'],1)
        self.assertEqual(result['first_attempt_task_successes'],0)
        self.assertEqual(result['extra_infrastructure_attempts'],1)

    def test_failures_and_unattempted_tasks_remain_in_denominator(self):
        rows=[{'status':'passed','task_success':True,'video_validation':'passed'},
              {'status':'failed','task_success':False}, {'status':'blocked','task_success':None},
              {'status':'running'}, {'status':'planned'}]
        result=batch.summarize(rows)
        self.assertEqual(result['total'],5)
        self.assertEqual(result['completed'],3)
        self.assertEqual(result['success_fraction_all_tasks'],.2)
        self.assertEqual(result['final_evaluations'],2)
        self.assertEqual(result['complete_videos'],1)
        self.assertEqual(result['execution_status'],'running')

    def test_executor_retest_keeps_task_denominator_and_first_result(self):
        rows=[{'status':'passed','task_success':True,'retry_kind':'executor_fix',
               'previous_attempts':[{'status':'failed','task_success':False}]},
              {'status':'planned'}]
        result=batch.summarize(rows)
        self.assertEqual(result['total'],2)
        self.assertEqual(result['task_successes'],1)
        self.assertEqual(result['first_attempt_task_successes'],0)
        self.assertEqual(result['extra_infrastructure_attempts'],0)
        self.assertEqual(result['extra_policy_attempts'],1)

    def test_missing_video_does_not_get_a_fabricated_link(self):
        row={'index':0,'status':'failed','name':'Task <script>bad()</script>',
             'task':'test','instruction':'<unsafe>','instruction_source':'project_translation_of_static_bddl_goal'}
        text=batch.render_dashboard({'summary':batch.summarize([row]),'tasks':[row],'updated_at':'test'})
        self.assertIn('未取得最终评分',text)
        self.assertIn('&lt;script&gt;',text)
        self.assertNotIn('href="None"',text)
        self.assertNotIn('href="episode.mp4"',text)
        self.assertEqual(batch.summarize([row])['execution_status'],'completed')

    def test_subset_reports_actual_denominator_and_separate_baseline(self):
        row={'index':0,'status':'planned','name':'Radio','task':'radio',
             'instruction':'Turn on radio','instruction_source':'official_gallery'}
        text=batch.render_dashboard({'summary':batch.summarize([row]),'tasks':[row],'updated_at':'test',
             'comparison':{'scope':'29 completed + 3 interrupted','baseline_url':'../old.html','report_url':'../fix.html'}})
        self.assertIn('1 项任务测试',text)
        self.assertIn('1 项使用官方原文，0 项按静态',text)
        self.assertIn('href="../old.html"',text)
        self.assertNotIn('共 100 个 episode',text)

if __name__=='__main__': unittest.main()
