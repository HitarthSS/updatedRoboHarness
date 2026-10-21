from pathlib import Path
import os
import subprocess
from settings import QWEN_PYTHON
root=Path(__file__).resolve().parent
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=os.environ.get('ROBOHARNESS_QWEN_GPUS','0,1,2,3'),VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_PP_LAYER_PARTITION='28,12,12,12',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='NVL',NCCL_IB_DISABLE='1',NCCL_CUMEM_ENABLE='0',VLLM_USE_FLASHINFER_SAMPLER='0')
cmd=[str(QWEN_PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',str(root/'checkpoints/qwen3-vl-32b'),'--served-model-name','qwen3-vl-32b-instruct','--host','127.0.0.1','--port','21910','--tensor-parallel-size','1','--pipeline-parallel-size','4','--dtype','bfloat16','--max-model-len','4096','--max-num-seqs','1','--gpu-memory-utilization','0.25','--cpu-offload-gb','0','--max-num-batched-tokens','1024','--kv-cache-memory-bytes','805306368','--skip-mm-profiling','--enforce-eager','--limit-mm-per-prompt','{"image":5}','--mm-processor-kwargs','{"min_pixels":65536,"max_pixels":262144}','--disable-custom-all-reduce']
import json
(root/'qwen-serving.json').write_text(json.dumps(dict(backend='vllm',tensor_parallel_size=1,pipeline_parallel_size=4,layer_partition=[28,12,12,12],cpu_offload_gib_per_rank=0,gpus=env['CUDA_VISIBLE_DEVICES'].split(','),max_model_len=4096,dtype='bfloat16',quantization=None,nccl_p2p_disable=env['NCCL_P2P_DISABLE'],nccl_p2p_level=env.get('NCCL_P2P_LEVEL'),command=cmd),indent=2)+'\n')
(root/'logs').mkdir(exist_ok=True)
with (root/'logs/qwen-standalone.log').open('a') as log:
    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'qwen-standalone.pid').write_text(str(child.pid)+'\n')
print(child.pid)
