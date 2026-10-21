"""Persistent, resumable baseline/improved campaign. No account model fallback."""
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-harness/bin/python'
SERVICES={}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)+'\n'); temp.replace(path)


def status(state,**kwargs):
    write(ROOT/'status.json',dict(state=state,updated_unix=time.time(),**kwargs))
    print(state,kwargs,flush=True)


def ready(url):
    try:
        with urllib.request.urlopen(url,timeout=3) as res:
            return res.status==200
    except Exception:
        return False


def start_service(name, command, env):
    with (ROOT/'logs'/f'{name}.log').open('a') as log:
        child=subprocess.Popen(list(map(str,command)),env=env,cwd=ROOT,stdout=log,stderr=log,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    SERVICES[name]=child
    pidfiles={'policy':'policy-standalone.pid','sam3':'sam3-standalone.pid','qwen':'qwen-standalone.pid'}
    (ROOT/pidfiles[name]).write_text(str(child.pid)+'\n')
    write(ROOT/'services.json',{n:dict(pid=p.pid) for n,p in SERVICES.items()})
    return child


class AdoptedService:
    def __init__(self, process):
        self.process=process
        self.pid=process.pid
    def poll(self):
        import psutil
        return None if self.process.is_running() and self.process.status()!=psutil.STATUS_ZOMBIE else 1


def adopt_service(name, pidfile, script):
    import psutil
    path=ROOT/pidfile
    if not path.exists(): return False
    try:
        proc=psutil.Process(int(path.read_text()))
        if proc.username()!=__import__('getpass').getuser() or not (str(ROOT/script) in proc.cmdline() or (script=='vllm.entrypoints.openai.api_server' and script in proc.cmdline() and str(ROOT/'checkpoints/qwen3-vl-32b') in proc.cmdline())):
            return False
        if proc.status()==psutil.STATUS_ZOMBIE: return False
        SERVICES[name]=AdoptedService(proc)
        return True
    except (psutil.NoSuchProcess,ValueError):
        return False


def wait_service(name,url,timeout=1800):
    start=time.monotonic()
    while not ready(url):
        if SERVICES[name].poll() is not None:
            raise RuntimeError(name+' exited; inspect logs/'+name+'.log')
        if time.monotonic()-start>timeout:
            raise TimeoutError(name+' readiness timeout')
        time.sleep(10)


def execute(arm,suite,task,seed):
    label=f'{suite}-t{task}-s{seed}'
    target=ROOT/'runs'/arm/label
    path=target/'result.json'
    if path.exists():
        return json.loads(path.read_text())
    target.mkdir(parents=True,exist_ok=True)
    status('running',arm=arm,case=label)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=os.environ.get('ROBOHARNESS_POLICY_GPU','0'),
               MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    for key in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','CODEX_API_KEY']:
        env.pop(key,None)
    with (target/'episode.log').open('a') as log:
        subprocess.run([str(PYTHON),'-u',str(ROOT/'run_episode.py'),'--arm',arm,'--suite',suite,
            '--task',str(task),'--seed',str(seed),'--output',str(target)],
            env=env,cwd=ROOT,stdout=log,stderr=log,check=True,timeout=7200)
    return json.loads(path.read_text())


def summary():
    report={}
    records={arm:{p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs'/arm).glob('*/result.json')}
             for arm in ['baseline','improved']}
    for arm,items in records.items():
        rows=list(items.values()); executed=[r for r in rows if r.get('executed')]
        report[arm]=dict(recorded=len(rows),executed=len(executed),successes=sum(r['success'] for r in rows),
             horizon_violations=sum(r.get('steps',0)>r['horizon'] for r in rows),
             vlm_calls=sum(r.get('vlm_calls',0) for r in rows),
             episode_seconds=sum(r.get('wall_seconds',0) for r in rows),
             success_rate_executed=sum(r['success'] for r in executed)/len(executed) if executed else None,
             unavailable=len(rows)-len(executed))
    common=records['baseline'].keys()&records['improved'].keys()
    pairs=[(records['baseline'][k],records['improved'][k]) for k in sorted(common)]
    for b,i in pairs:
        if b.get('initial_state_sha256') != i.get('initial_state_sha256'):
            raise RuntimeError('Mismatched paired initial states')
    report['paired']=dict(cases=len(pairs),executed=sum(b.get('executed') and i.get('executed') for b,i in pairs),baseline_successes=sum(b['success'] for b,i in pairs),
        improved_successes=sum(i['success'] for b,i in pairs),
        improved_only=sum(i['success'] and not b['success'] for b,i in pairs),
        baseline_only=sum(b['success'] and not i['success'] for b,i in pairs))
    report['planned_cases_per_arm']=400
    cap=ROOT/'evaluation-limit.json'
    if cap.exists(): report['pilot_limit']=json.loads(cap.read_text())
    report['timing_caveat']='Shared busy GPUs; latency is descriptive, not dedicated-hardware speed evidence.'
    write(ROOT/'comparison.json',report)
    return report


def main():
    ROOT.joinpath('logs').mkdir(exist_ok=True)
    lock=(ROOT/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('waiting_for_assets')
    while not (ROOT/'checkpoints/manifest.json').exists() or not (ROOT/'checkpoints/support-manifest.json').exists():
        time.sleep(15)
    # The first manifest is written after Qwen completes, but require all support encoders too.
    while not (ROOT/'checkpoints/minilm/model.safetensors').exists():
        time.sleep(15)
    while not (ROOT/'.venv-policy311/lib/python3.11/site-packages/lerobot').exists():
        time.sleep(15)
    subprocess.run([str(PYTHON),str(ROOT/'repair_release.py')],check=True)
    # Freeze all decisions before any OOD evaluation outcome is seen.
    files=['env_adapter.py','protocol.py','grounding.py','run_episode.py','policy_service.py','repair_release.py','serve_sam3.py']
    protocol=dict(upstream_commit='716defc19c2062e613e81dd19592595ef289b3b6',
        files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        models=json.loads((ROOT/'checkpoints/support-manifest.json').read_text()),
        orchestrator=json.loads((ROOT/'checkpoints/manifest.json').read_text())['qwen3-vl-32b'],
        horizon_multiplier=1.0,horizons={'spatial':220,'object':280},
        cold_start='standard LIBERO spatial/object, all ten tasks, reset indices 40 and 41',
        evaluation='LIBERO-PRO spatial/object x swap/task, ten tasks, reset indices 0..9',
        paper_parity='Paper does not identify exact policy checkpoint or release its initialized memory. Official lerobot/pi05-libero is pinned; 10 released resets/task differ from paper 199 randomized rollouts.',
        candidate='RGB-driven call allocation (10..30 steps), bounded recent rollback (10 reverse + 5 hold), max 2 recoveries with 40-step cooldown; retain nonvisual refined tasks, expire visual rewrites with their action chunk, clear stale policy actions without reseeding; on empty text masks, reuse the existing VLM box for one SAM3 positive-box pass, requiring score >=0.25 and box IoU >0.1',
        test_time_learning=False,
        policy_runtime=json.loads((ROOT/'transformers-replacement.json').read_text()),
        tokenizer=json.loads((ROOT/'tokenizer/artifact.json').read_text()),
        vlm_runtime=json.loads((ROOT/'qwen-serving.json').read_text()))
    old=ROOT/'experiment.json'
    if old.exists() and list((ROOT/'runs/baseline').glob('*/result.json')):
        if json.loads(old.read_text()) != protocol:
            raise RuntimeError('Frozen experiment changed after evaluation started')
    write(old,protocol)
    base_env=os.environ.copy(); base_env.update(OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for k in ['OPENAI_API_KEY','ANTHROPIC_API_KEY','CODEX_API_KEY']:
        base_env.pop(k,None)
    status('starting_models')
    # User authorized sharing busy GPUs; do not wait for zero utilization.
    # Leave unrelated workloads untouched. Exact weights, CPU offload, no quantization.
    if not adopt_service('qwen','qwen-standalone.pid','vllm.entrypoints.openai.api_server'):
        raise RuntimeError('Exact Qwen server must be started with start_qwen.py')
    penv=base_env.copy(); penv['CUDA_VISIBLE_DEVICES']=os.environ.get('ROBOHARNESS_POLICY_GPU','0')
    if not adopt_service('policy','policy-standalone.pid','policy_service.py'):
        start_service('policy',[ROOT/'.venv-policy311/bin/python','-u',ROOT/'policy_service.py'],penv)
    wait_service('policy','http://127.0.0.1:21911/health')
    if not adopt_service('sam3','sam3-standalone.pid','serve_sam3.py'):
        start_service('sam3',[PYTHON,'-u',ROOT/'serve_sam3.py'],dict(penv,CUDA_VISIBLE_DEVICES='0'))
    wait_service('sam3','http://127.0.0.1:21912/health')
    wait_service('qwen','http://127.0.0.1:21910/v1/models')
    # Rebuild the missing cold-start bank on ordinary tasks, never on OOD test outcomes.
    if not (ROOT/'memory-frozen/manifest.json').exists():
        for seed in [40,41]:
            for suite in ['libero_spatial','libero_object']:
                for task in range(10):
                    execute('cold-start',suite,task,seed)
        shutil.copytree(ROOT/'memory-development',ROOT/'memory-frozen',dirs_exist_ok=True)
        hashes={str(p.relative_to(ROOT/'memory-frozen')):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/'memory-frozen').rglob('*') if p.is_file()}
        write(ROOT/'memory-frozen/manifest.json',hashes)
    for seed in range(10):
        for suite in ['libero_spatial_swap','libero_spatial_task','libero_object_swap','libero_object_task']:
            for task in range(10):
                cap=ROOT/'evaluation-limit.json'
                if cap.exists():
                    limit=json.loads(cap.read_text())['max_executed_pairs']
                    if not isinstance(limit,int) or isinstance(limit,bool) or limit<1:
                        raise ValueError('max_executed_pairs must be a positive integer')
                    report=summary()
                    if report['paired']['executed']>=limit:
                        status('complete',reason='user_requested_pilot_limit',comparison=report)
                        return
                order=['baseline','improved'] if (seed+task)%2==0 else ['improved','baseline']
                for arm in order:
                    execute(arm,suite,task,seed)
                    summary()
    status('complete',comparison=summary())


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
    finally:
        # Only stop services created by this campaign.
        import psutil
        for process in SERVICES.values():
            if process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    children=parent.children(recursive=True)
                    for child in children: child.terminate()
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
