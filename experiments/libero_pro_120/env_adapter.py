"""Official LIBERO environment with separate settling and policy horizons."""

def make_budgeted_env(task_id, seed, suite_name, horizon):
    import sys
    sys.path.insert(0,'/home/hitarth/rpent-reproduction/RPent')
    from robots.libero.env_server import build_env_cfg
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from deterministic_reset import install
    install()
    from rlinf.envs.sim.libero.utils import benchmark
    suite=benchmark.get_benchmark(suite_name)()
    trials=len(suite.get_task_init_states(task_id))
    if not trials:
        raise ValueError('No released initial states')
    reset_id=sum(len(suite.get_task_init_states(i)) for i in range(task_id))+seed%trials
    cfg=build_env_cfg(task_suite_name=suite_name,specific_reset_id=reset_id,
                      seed=seed,max_episode_steps=horizon)
    # Robosuite counts the reset settling actions in its internal clock. Its
    # own safety horizon must allow those actions in addition to the benchmark
    # policy budget. The RLinf truncation counter starts after settling.
    cfg.init_params.horizon=horizon+15
    # This installed RLinf adapter reads perturbation files directly in its
    # reset method. Explicitly route our fresh immutable bank there as well.
    import hashlib
    import numpy as np
    expected_states = suite.get_task_init_states(task_id)
    assert len(expected_states) == 199 and 0 <= seed < 199
    original_reset_states = LiberoEnv._get_reset_states
    def fresh_reset_states(self, env_idx):
        if type(self.task_suite) is not type(suite):
            return original_reset_states(self, env_idx)
        ids = np.arange(self.num_envs) if env_idx is None else env_idx
        if len(ids) != 1:
            raise RuntimeError('Full-eval reset adapter expects one environment')
        env_id = int(ids[0])
        if int(self.task_ids[env_id]) != task_id or int(self.trial_ids[env_id]) != seed:
            raise RuntimeError('Native task/trial routing disagrees with frozen case')
        chosen = np.asarray(expected_states[seed]).copy()
        self.full_eval_loaded_state_sha256 = hashlib.sha256(chosen.tobytes()).hexdigest()
        return [chosen]
    LiberoEnv._get_reset_states = fresh_reset_states
    return LiberoEnv(cfg=cfg,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
