"""Select immutable states without consulting evaluation outcomes."""
import json
from pathlib import Path
import numpy as np
import pilot_common as c


def build():
    conditions=json.loads((c.ROOT/'conditions.json').read_text())
    if len(conditions)!=20:raise RuntimeError('Expected all 20 paper LIBERO-PRO conditions')
    counts={kind:sum(x['suite'].endswith(kind) for x in conditions) for kind in ('swap','task')}
    if counts!={'swap':6,'task':14}:raise RuntimeError('Unexpected paper condition split')
    observed=set()
    for root in [Path('/home/hitarth/roboharness-reproduction/runs'),Path('/home/hitarth/roboharness-full-eval/runs')]:
        for p in root.glob('*/*/result.json'):
            r=json.loads(p.read_text())
            if r.get('executed') and r.get('initial_state_sha256'):observed.add(r['initial_state_sha256'])
    cases=[];digests={};selected_hashes=set()
    for ci,condition in enumerate(conditions):
        path=Path(condition['state_bank'])
        if c.sha(path)!=condition['state_bank_sha256']:raise RuntimeError('Changed original state bank')
        digests[str(path)]=c.sha(path)
        states=np.load(path,allow_pickle=False)
        if len(states)!=199 or states.ndim!=2 or not np.isfinite(states).all():raise RuntimeError('Invalid bank')
        for si,seed in enumerate(c.CONFIG['state_indices']):
            import hashlib
            h=hashlib.sha256(states[seed].tobytes()).hexdigest()
            if h in observed:raise RuntimeError('Selected state was already evaluated')
            key=(condition['label'],h)
            if key in selected_hashes:raise RuntimeError('Repeated selected case')
            selected_hashes.add(key)
            arms=['baseline','improved'] if (ci+si)%2==0 else ['improved','baseline']
            cases.append(dict(condition_index=ci,state_index=seed,arms=arms,
                initial_state_sha256=h,**condition))
    cases.sort(key=lambda r:(r['state_index'],r['condition_index']))
    manifest=dict(config=c.CONFIG,conditions=conditions,cases=cases,state_banks=digests,
        prior_executed_state_hashes_checked=len(observed),overlap_with_prior_executed_states=0,
        condition_counts=counts,selection='Fixed indices 100..105 from pre-existing frozen native reset banks; no outcomes consulted')
    target=c.ROOT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise RuntimeError('Refusing to replace an existing manifest')
    c.atomic(target,manifest)
    print(json.dumps(dict(cases=len(cases),condition_counts=counts,overlap=0)),flush=True)


if __name__=='__main__':build()
