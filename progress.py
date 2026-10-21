"""Read-only status of the persistent campaign; no model requests or credentials."""
from pathlib import Path
import json
import time
root = Path(__file__).resolve().parent
report = {'status': json.loads((root/'status.json').read_text())}
for arm in ['cold-start','baseline','improved']:
    rows=[json.loads(p.read_text()) for p in (root/'runs'/arm).glob('*/result.json')]
    executed=[r for r in rows if r['executed']]
    report[arm]=dict(recorded=len(rows), executed=len(executed),
        successes=sum(r['success'] for r in executed),
        horizon_violations=sum(r['steps']>r['horizon'] for r in executed),
        policy_actions=sum(r['steps'] for r in executed),
        policy_forwards=sum(r.get('policy_forward_calls',0) for r in executed))
report['planned']={'cold-start':40,'baseline':400,'improved':400}
report['available_cases_per_arm']=10*sum(bool(r['initial_states']) for r in json.loads((root/'assets-audit.json').read_text()))
report['status_age_seconds']=round(time.time()-report['status']['updated_unix'],1)
comparison=root/'comparison.json'
if comparison.exists(): report['comparison']=json.loads(comparison.read_text())
print(json.dumps(report,indent=2))
