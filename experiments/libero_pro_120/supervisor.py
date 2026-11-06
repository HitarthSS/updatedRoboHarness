"""Finite, unattended 120-pair evaluation. Never replace failed cases."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
import requests

import pilot_common as c
c.configure()
from report import summarize

SERVICES={}


def adopt():
    specs={'qwen':('vllm.entrypoints.openai.api_server',str(c.ROOT/'checkpoints/qwen3-vl-32b')),
        'policy':(str(c.ROOT/'policy_service.py'),),'sam3':(str(c.ROOT/'serve_sam3.py'),),
        'meter':(str(c.ROOT/'meter_proxy.py'),)}
    for name,needles in specs.items():
        pid=int((c.ROOT/(name+'.pid')).read_text());p=psutil.Process(pid)
        cmd=' '.join(p.cmdline())
        if p.username()!='hitarth' or any(v not in cmd for v in needles):raise RuntimeError('Wrong service PID: '+name)
        SERVICES[name]=p
    c.atomic(c.ROOT/'services.json',{k:dict(pid=p.pid) for k,p in SERVICES.items()})


def wait_ready():
    endpoints={'qwen':'http://127.0.0.1:22909/v1/models','policy':'http://127.0.0.1:22911/health',
        'sam3':'http://127.0.0.1:22912/health','meter':'http://127.0.0.1:22910/v1/models'}
    for name,url in endpoints.items():
        c.status('waiting_for_service',service=name)
        until=time.monotonic()+600
        while True:
            if not SERVICES[name].is_running() or SERVICES[name].status()==psutil.STATUS_ZOMBIE:
                raise RuntimeError(name+' exited; inspect logs')
            try:
                r=requests.get(url,timeout=5)
                if r.ok:
                    if name in ('qwen','meter') and not any(x['id']=='qwen3-vl-32b-instruct' for x in r.json()['data']):
                        raise RuntimeError('Unexpected Qwen identity')
                    break
            except requests.RequestException:pass
            if time.monotonic()>until:raise TimeoutError(name+' readiness timeout')
            time.sleep(2)


def freeze():
    path=c.ROOT/'FROZEN.json'
    if path.exists():c.verify_freeze();return
    if c.sha(c.ROOT/'protocol.py')!=c.sha('/home/hitarth/roboharness-full-eval/protocol.py'):
        raise RuntimeError('Existing candidate protocol changed')
    manifest=json.loads((c.ROOT/'manifest.json').read_text())
    sources=c.source_hashes()
    assets={}
    for p in (c.ROOT/'memory-frozen').glob('*/metadata.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            image=Path(r['keyframe_path'])
            if not image.is_file():raise RuntimeError('Missing cold-start memory image: '+str(image))
            assets[str(image)]=c.sha(image)
    sources.update(assets)
    for bank,digest in manifest['state_banks'].items():
        if c.sha(bank)!=digest:raise RuntimeError('Changed bank before freeze')
        sources[bank]=digest
    model_metadata={}
    for model in ['qwen3-vl-32b','pi05-libero','clip','minilm']:
        folder=c.ROOT/'checkpoints'/model
        for p in list(folder.glob('*.json'))+list(folder.glob('*.model')):
            model_metadata[str(p)]=c.sha(p)
    sources.update(model_metadata)
    c.atomic(path,dict(config=c.CONFIG,sources=sources,memory=c.tree_hashes(c.ROOT/'memory-frozen'),
        frozen_unix=time.time(),test_outcomes_observed=0,
        model_catalog=json.loads((c.ROOT/'checkpoints/manifest.json').read_text()),
        support_catalog=json.loads((c.ROOT/'checkpoints/support-manifest.json').read_text()),
        sam3=json.loads((c.ROOT/'sam3-provenance.json').read_text()),
        original_protocol_sha256=c.sha('/home/hitarth/roboharness-full-eval/protocol.py'),
        copied_protocol_sha256=c.sha(c.ROOT/'protocol.py'),
        memory_initialization='40 standard-LIBERO episodes, indices40/41; copied unchanged, 38 success and 2 failure memories',
        initial_memory_image_count=len(assets)))


def execute(case,arm,phase='test'):
    label=f'{case["suite"]}-t{case["task"]}-s{case["state_index"]}'
    target=c.ROOT/('runs' if phase=='test' else 'smoke')/arm/label
    result=target/'result.json'
    if result.exists():
        r=json.loads(result.read_text())
        if r['frozen_sha256']!=c.verify_freeze():raise RuntimeError('Completed result belongs to another freeze')
        return r
    if (target/'error.json').exists():raise RuntimeError('Previously failed case requires inspection: '+str(target))
    target.mkdir(parents=True,exist_ok=True)
    frozen=c.verify_freeze()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',PYTHONHASHSEED='0')
    cmd=[str(c.ROOT/'.venv-harness/bin/python'),'-u',str(c.ROOT/'run_episode.py'),
        '--arm',arm,'--suite',case['suite'],'--task',str(case['task']),
        '--seed',str(case['state_index']),'--state-bank',case['state_bank'],'--output',str(target)]
    if phase=='smoke':cmd+=['--smoke-limit','40']
    summary=summarize()
    c.status('running_smoke' if phase=='smoke' else 'running',current_case=label,current_arm=arm,
        completed_pairs=summary['completed_pairs'],completed_episodes=summary['completed_episodes'],planned_episodes=240)
    with (target/'episode.log').open('a') as log:
        child=subprocess.Popen(cmd,env=env,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
        c.atomic(c.ROOT/'current-process.json',dict(pid=child.pid,case=label,arm=arm,phase=phase,path=str(target)))
        try:code=child.wait(timeout=7200)
        except BaseException:
            p=psutil.Process(child.pid)
            for q in p.children(recursive=True):q.terminate()
            p.terminate();raise
    if code:raise RuntimeError(f'Episode failed: {target}; exit={code}; no replacement case')
    r=json.loads(result.read_text())
    if not r['executed'] or r['steps']>r['horizon'] or r['horizon_violations']:raise RuntimeError('Invalid episode')
    if r['initial_state_sha256']!=case['initial_state_sha256']:raise RuntimeError('Unexpected initial state')
    if r['state_bank_sha256']!=case['state_bank_sha256']:raise RuntimeError('Unexpected state bank')
    if r['frozen_sha256']!=frozen:raise RuntimeError('Wrong frozen sources')
    c.verify_freeze()
    r['initial_png_sha256']=c.sha(target/'initial.png')
    c.atomic(result,r)
    c.atomic(c.ROOT/'SUMMARY.json',summarize())
    return r


def verify_pair(results):
    a,b=results['baseline'],results['improved']
    for key in ['initial_state_sha256','initial_png_sha256','instruction','state_bank_sha256',
                'post_settling_physics_sha256','model_xml_sha256','construction_seed',
                'initial_public_state_sha256','initial_wrist_sha256']:
        if a[key]!=b[key]:raise RuntimeError('Paired mismatch: '+key)


def main():
    lock=(c.ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adopt();wait_ready()
    manifest=json.loads((c.ROOT/'manifest.json').read_text())
    if len(manifest['cases'])!=120:raise RuntimeError('Wrong case count')
    freeze()
    import hashlib,numpy as np
    first=manifest['conditions'][0];state=np.load(first['state_bank'],allow_pickle=False)[190]
    smoke=dict(first,state_index=190,initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest())
    results={arm:execute(smoke,arm,'smoke') for arm in c.CONFIG['arms']}
    verify_pair(results)
    c.atomic(c.ROOT/'COMPATIBILITY.json',dict(status='PASS',matched_initial_state_and_image=True,
        episodes=2,smoke_state_index=190,step_cap_per_episode=40,excluded_from_test=True,
        transport_errors=0,horizon_violations=0,matched_model_physics_and_both_camera_inputs=True))
    c.atomic(c.ROOT/'LAUNCH_RECEIPT.json',dict(status='running',supervisor_pid=os.getpid(),
        started_unix=time.time(),config=c.CONFIG,frozen_sha256=c.verify_freeze(),
        services={k:v.pid for k,v in SERVICES.items()},compatibility='PASS'))
    for case in manifest['cases']:
        results={arm:execute(case,arm) for arm in case['arms']}
        verify_pair(results)
        c.atomic(c.ROOT/'SUMMARY.json',summarize())
    final=summarize();c.atomic(c.ROOT/'FINAL_ANALYSIS.json',final)
    c.status('completed',summary=final)


def cleanup():
    evidence={}
    for name,p in reversed(list(SERVICES.items())):
        try:
            descendants=p.children(recursive=True)
            for child in descendants:child.terminate()
            p.terminate()
            _,alive=psutil.wait_procs([p,*descendants],timeout=15)
            for child in alive:child.kill()
            evidence[name]=dict(pid=p.pid,stopped=True)
        except psutil.NoSuchProcess:evidence[name]=dict(pid=p.pid,already_exited=True)
    c.atomic(c.ROOT/'cleanup.json',evidence)


if __name__=='__main__':
    try:main()
    except BaseException:
        c.status('FAILED',error=traceback.format_exc())
        c.atomic(c.ROOT/'LAST_ERROR.json',dict(time=time.time(),error=traceback.format_exc()))
        raise
    finally:cleanup()
