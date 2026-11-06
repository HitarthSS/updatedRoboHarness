"""Install one immutable, task-specific reset bank before constructing RLinf."""
import hashlib,json
from pathlib import Path
import numpy as np

def install_state_bank(suite, task_id, path):
 from strict_bddl import install
 install()
 path=Path(path)
 meta=json.loads(path.with_suffix('.json').read_text())
 payload=path.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=meta['sha256']:raise ValueError('State bank digest mismatch')
 bddl=Path(suite.get_task_bddl_file_path(task_id))
 if hashlib.sha256(bddl.read_bytes()).hexdigest()!=meta['bddl_sha256']:raise ValueError('State bank belongs to a different task')
 states=np.load(path,allow_pickle=False)
 if states.ndim!=2 or len(states)!=199 or not np.isfinite(states).all():raise ValueError('Invalid state bank')
 if len({hashlib.sha256(s.tobytes()).hexdigest() for s in states})!=199:raise ValueError('Repeated states in bank')
 states.setflags(write=False)
 cls=type(suite); original=cls.get_task_init_states
 def load(self,index):return states if index==task_id else original(self,index)
 cls.get_task_init_states=load
 return dict(state_bank_sha256=meta['sha256'],state_bank_sampling='199 fresh reset samples; fixed before both arms',bddl_sha256=meta['bddl_sha256'])
