"""Fixed 20-condition, six-state RoboHarness comparison; local models only."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
CONFIG=dict(
    conditions=20, state_indices=[100,101,102,103,104,105],
    episodes_per_arm=120, physical_episodes=240, arms=['baseline','improved'],
    horizons={'spatial':220,'object':280}, horizon_multiplier=1.0,
    settling_actions=15, settling_excluded_from_policy_horizon=True,
    qwen_gpus=[0,3], policy_sam_sim_gpu=7,
    qwen_model='Qwen/Qwen3-VL-32B-Instruct',
    qwen_revision='0cfaf48183f594c314753d30a4c4974bc75f3ccb',
    qwen_dtype='bfloat16', qwen_quantization=None, qwen_max_context=8192,
    qwen_tensor_parallel=2, qwen_pipeline_parallel=1,
    policy_model='lerobot/pi05-libero',policy_revision='9ba5431256f4136a01c224ab0307b0d3035cc92c',
    upstream_commit='716defc19c2062e613e81dd19592595ef289b3b6',
    candidate='Existing frozen adaptive scheduling, bounded rollback, instruction persistence and candidate-only SAM3 box fallback; no retuning',
    paid_api_calls=0, test_time_learning=False,
    paper_reference={'position_percent':57.2,'task_percent':55.2,'rollouts_per_condition':199},
    paper_parity_gaps=[
        'Authors do not identify immutable policy checkpoint or release their initialized memory; use pinned official Pi0.5 and our existing standard-LIBERO cold-start memory.',
        'Table III symbolic labels mapped to released BDDL task identities; mappings not author-confirmed.',
        'Six held-out states per condition, not paper 199. State banks are reconstructed from native benchmark resets, not the authors randomized manifest.',
        'Local serving context and sampling infrastructure are documented; exact paper serving configuration is unavailable.',
        'Strict 1x horizon is enforced here; exact paper accounting for every recovery is not established.',
    ],
    common_runtime_changes=[
        'Malformed 200-OK model replies follow upstream parsing fallback and are counted; they are not simulator or transport errors.',
        'Transport/HTTP errors, wrong models and invalid action/state routing stop the batch without replacing cases.',
        'All Qwen calls, including SAM3 bounding-box calls, pass through a local metering proxy.',
        'Per-request seeds are matched by case and call ordinal, without changing upstream temperature/token limits.',
        'Qwen BF16 TP2 context8192 versus the earlier PP4 context4096 setup; same configuration in both arms.',
        'Seed Python, NumPy and the Robosuite constructor inside spawned workers before model construction; verify compiled XML, post-settling physics, public state and both initial camera observations.',
    ],
)


def configure():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1',WANDB_MODE='disabled',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',
        LIBERO_TYPE='pro',VLM_API_KEY='local-only',
        VLM_BASE_URL='http://127.0.0.1:22910/v1')
    for k in ('OPENAI_API_KEY','ANTHROPIC_API_KEY','CODEX_API_KEY','NRP_API_KEY'):
        os.environ.pop(k,None)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def tree_hashes(root):
    root=Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def source_hashes():
    paths=list(ROOT.glob('*.py'))+[ROOT/'README.md',ROOT/'manifest.json',ROOT/'qwen-serving.json',ROOT/'package-versions.json',ROOT/'RESET_VALIDATION.json']
    paths+=sorted((ROOT/'runtime').glob('*.py'))
    return {str(p):sha(p) for p in paths}


def verify_freeze():
    r=json.loads((ROOT/'FROZEN.json').read_text())
    if r['config']!=CONFIG:raise RuntimeError('Frozen configuration changed')
    for path,digest in r['sources'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    if tree_hashes(ROOT/'memory-frozen')!=r['memory']:raise RuntimeError('Frozen memory changed')
    return sha(ROOT/'FROZEN.json')


def status(state,**fields):
    record=dict(state=state,updated_unix=time.time(),supervisor_pid=os.getpid(),**fields)
    atomic(ROOT/'status.json',record)
    print(json.dumps(record),flush=True)
