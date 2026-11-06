"""Only matched completed cases enter comparative metrics."""
from pathlib import Path
import json
import numpy as np
import pilot_common as c


def metrics(rows):
    n=len(rows);s=sum(r['success'] for r in rows)
    result=dict(episodes=n,successes=s,success_percent=100*s/n if n else None)
    for key in ['steps','recovery_steps','policy_forward_calls','policy_seconds','vlm_all_calls',
                'vlm_prompt_tokens','vlm_completion_tokens','vlm_missing_usage','vlm_all_seconds',
                'wall_seconds','total_episode_seconds','model_json_failures','horizon_violations','paid_api_calls']:
        result[key]=sum(r.get(key,0) for r in rows)
    return result


def summarize():
    grouped={};completed=0
    for arm in c.CONFIG['arms']:
        for p in (c.ROOT/'runs'/arm).glob('*/result.json'):
            r=json.loads(p.read_text())
            if r.get('phase')!='test' or not r.get('executed'):continue
            completed+=1
            grouped.setdefault(p.parent.name,{})[arm]=r
    pairs=[v for v in grouped.values() if set(v)==set(c.CONFIG['arms'])]
    for pair in pairs:
        a,b=pair['baseline'],pair['improved']
        if a['initial_state_sha256']!=b['initial_state_sha256']:raise RuntimeError('Initial state mismatch')
        if a['instruction']!=b['instruction']:raise RuntimeError('Instruction mismatch')
    arms={a:metrics([p[a] for p in pairs]) for a in c.CONFIG['arms']}
    d=np.array([int(p['improved']['success'])-int(p['baseline']['success']) for p in pairs])
    helped=int((d==1).sum());harmed=int((d==-1).sum())
    ci=cluster=pvalue=None
    if len(d):
        rng=np.random.default_rng(20261026)
        boots=rng.choice(d,size=(10000,len(d))).mean(axis=1)*100
        ci=np.quantile(boots,[.025,.975]).tolist()
        clusters={}
        for p,delta in zip(pairs,d):clusters.setdefault((p['baseline']['suite'],p['baseline']['task']),[]).append(int(delta))
        means=np.array([np.mean(v) for v in clusters.values()])
        cluster=np.quantile(rng.choice(means,size=(10000,len(means))).mean(axis=1)*100,[.025,.975]).tolist()
        from math import comb
        n=helped+harmed
        pvalue=min(1.,2*sum(comb(n,k) for k in range(min(helped,harmed)+1))/2**n) if n else 1.
    shift={}
    for kind in ('swap','task'):
        selected=[p for p in pairs if p['baseline']['suite'].endswith(kind)]
        shift[kind]={a:metrics([p[a] for p in selected]) for a in c.CONFIG['arms']}
    conditions={}
    for p in pairs:
        r=p['baseline'];label=f'{r["suite"]}-t{r["task"]}'
        conditions.setdefault(label,[]).append(p)
    per_condition={key:{a:metrics([p[a] for p in ps]) for a in c.CONFIG['arms']} for key,ps in conditions.items()}
    return dict(completed_episodes=completed,planned_episodes=240,completed_pairs=len(pairs),planned_pairs=120,
        arms=arms,by_shift=shift,per_condition=per_condition,
        paired=dict(helped=helped,harmed=harmed,difference_pp=float(d.mean()*100) if len(d) else None,
            episode_bootstrap_95_difference_pp=ci,condition_bootstrap_95_difference_pp=cluster,
            mcnemar_exact_two_sided_p=pvalue),
        horizon='1x: Spatial220/Object280; every recovery charged; 15 reset-settling actions separately disclosed',
        coverage='Only identical completed pairs; smoke/development/prior pilots excluded',
        timing='Shared GPUs; model response, policy and total episode time recorded separately; no isolated speed claim',
        interpretation='Reduced matched pilot on 20 mapped paper conditions, with explicitly documented reproduction gaps')


if __name__=='__main__':print(json.dumps(summarize(),indent=2))
