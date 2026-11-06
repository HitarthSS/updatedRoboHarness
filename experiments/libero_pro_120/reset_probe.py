"""Independent-process reset pairing check; no policy or VLM inference."""
import argparse
import hashlib
import json
import os
import random
import sys
import numpy as np
from PIL import Image
import pilot_common as c
c.configure()
import full_bootstrap


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True)
    p.add_argument('--trial',type=int,required=True);args=p.parse_args()
    condition=json.loads((c.ROOT/'conditions.json').read_text())[args.condition]
    from rlinf.envs.sim.libero.utils import benchmark
    from state_bank import install_state_bank
    from env_adapter import make_budgeted_env
    from deterministic_reset import fingerprints
    from protocol import libero_policy_image
    seed=190;random.seed(seed);np.random.seed(seed)
    suite=benchmark.get_benchmark(condition['suite'])()
    install_state_bank(suite,condition['task'],condition['state_bank'])
    env=make_budgeted_env(condition['task'],seed,condition['suite'],condition['horizon'])
    try:
        obs,_=env.reset();raw=env.current_raw_obs[0]
        image=libero_policy_image(raw['agentview_image'])
        result=dict(fingerprints(env),state_sha256=hashlib.sha256(np.asarray(obs['states']).tobytes()).hexdigest(),
            image_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest())
        folder=c.ROOT/'reset-probe';folder.mkdir(exist_ok=True)
        name=f'condition{args.condition}-trial{args.trial}'
        c.atomic(folder/(name+'.json'),result)
        Image.fromarray(image).save(folder/(name+'.png'))
        print(json.dumps(result),flush=True)
    finally:env.env.close()


if __name__=='__main__':main()
