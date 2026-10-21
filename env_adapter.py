"""Official LIBERO environment with separate settling and policy horizons."""

def make_budgeted_env(task_id, seed, suite_name, horizon):
    import sys
    from settings import RPENT_ROOT
    sys.path.insert(0,str(RPENT_ROOT))
    from robots.libero.env_server import build_env_cfg
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
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
    return LiberoEnv(cfg=cfg,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
