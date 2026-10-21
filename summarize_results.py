"""Validate and summarize published metrics; standard library only."""
from collections import defaultdict
import json
from pathlib import Path


def main():
    data=json.loads((Path(__file__).parent/'results/pilot-20.json').read_text())
    arms={arm:{} for arm in ['baseline','improved']}
    for row in data['episodes']:
        key=(row['suite'],row['task'],row['seed'])
        assert key not in arms[row['arm']], 'Duplicate episode'
        arms[row['arm']][key]=row
    assert arms['baseline'].keys()==arms['improved'].keys()
    valid=[]
    for key,b in arms['baseline'].items():
        i=arms['improved'][key]
        assert b['executed']==i['executed']
        if b['executed']:
            assert b['initial_state_sha256']==i['initial_state_sha256']
            assert b['horizon']==i['horizon']
            valid.append(key)
    report={}
    for arm,rows in arms.items():
        rs=[rows[k] for k in valid]
        summary=data['comparison'][arm]
        assert len(rs)==summary['executed']==20
        assert sum(r['success'] for r in rs)==summary['successes']
        assert sum(r['vlm_calls'] for r in rs)==summary['vlm_calls']
        assert abs(sum(r['wall_seconds'] for r in rs)-summary['episode_seconds'])<1e-5
        assert all(r['steps']<=r['horizon'] for r in rs)
        suites=defaultdict(lambda:dict(cases=0,successes=0))
        for r in rs:
            suites[r['suite']]['cases']+=1
            suites[r['suite']]['successes']+=int(r['success'])
        report[arm]=dict(cases=len(rs),successes=sum(r['success'] for r in rs),
            average_seconds=sum(r['wall_seconds'] for r in rs)/len(rs),
            direct_vlm_calls=sum(r['vlm_calls'] for r in rs),
            mean_action_steps=sum(r['steps'] for r in rs)/len(rs),
            hit_horizon=sum(r['steps']==r['horizon'] for r in rs),suites=dict(suites))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
