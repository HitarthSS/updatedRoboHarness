"""Seed inside spawned workers before constructing the simulator model."""
import hashlib
import random
import numpy as np


def seeded_factory(factory,seed):
    seed=int(seed)
    random.seed(seed);np.random.seed(seed)
    from liberopro.liberopro.envs.env_wrapper import ControlEnv
    original=getattr(ControlEnv.__init__,'_paired_original',ControlEnv.__init__)

    def construct(self,*args,**kwargs):
        # Native RLinf otherwise applies seed() only after construction.
        random.seed(seed);np.random.seed(seed)
        kwargs.setdefault('seed',seed)
        original(self,*args,**kwargs)
        self.paired_model_xml_sha256=hashlib.sha256(self.env.model.get_xml().encode()).hexdigest()
        self.paired_construction_seed=seed

    construct._paired_original=original
    ControlEnv.__init__=construct
    return factory()


def install():
    from functools import partial
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    if getattr(LiberoEnv,'_paired_factory_installed',False):return
    original=LiberoEnv.get_env_fns

    def factories(self):
        return [partial(seeded_factory,fn,int(self.seed)+i) for i,fn in enumerate(original(self))]

    LiberoEnv.get_env_fns=factories
    LiberoEnv._paired_factory_installed=True


def fingerprints(env):
    worker=env.env.workers[0]
    def query(cmd,data):
        worker.parent_remote.send([cmd,data]);return worker.parent_remote.recv()
    state=np.asarray(query('get_sim_state',None))
    return dict(post_settling_physics_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        model_xml_sha256=query('getattr','paired_model_xml_sha256'),
        construction_seed=query('getattr','paired_construction_seed'))
