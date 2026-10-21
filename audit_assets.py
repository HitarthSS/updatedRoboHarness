import json
import os
from pathlib import Path
os.environ['LIBERO_TYPE']='pro'
import sys
from settings import RPENT_ROOT
sys.path.insert(0,str(RPENT_ROOT))
from rlinf.envs.sim.libero.utils import benchmark
from protocol import HORIZONS
ROOT=Path(__file__).resolve().parent
rows=[]
for name in ['libero_spatial_swap','libero_spatial_task','libero_object_swap','libero_object_task']:
    suite=benchmark.get_benchmark(name)()
    for task in range(suite.n_tasks):
        n=len(suite.get_task_init_states(task))
        rows.append(dict(suite=name,task=task,instruction=suite.get_task(task).language,
                         initial_states=n,horizon=HORIZONS[name.split('_')[1]]))
(ROOT/'assets-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Task settings',len(rows),'missing',[dict(suite=r['suite'],task=r['task']) for r in rows if not r['initial_states']])
