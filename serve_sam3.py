"""Run the released SAM3 tool server with exact local Qwen and existing SAM3 weights."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS'] = '4'
from repair_release import prepare
sys.path.insert(0,str(prepare()))
import torch
import sam3_service as service
from soma_vlm import Qwen3VLAPIClient
torch.set_num_threads(4)
service.vlm_client = Qwen3VLAPIClient(api_key='local-only', base_url='http://127.0.0.1:21910/v1', model_id='qwen3-vl-32b-instruct')
service.vlm_client.client.timeout = 1200
service.vlm_client.client.max_retries = 0
if not service.init_sam3_model('cuda', str(__import__('settings').SAM3_CHECKPOINT)):
    raise RuntimeError('SAM3 initialization failed')
# SAM3 emits BF16 activations internally while retaining FP32 parameters.
# Autocast must be entered in the request thread, not only at startup.
service._get_mask = torch.inference_mode()(
    torch.autocast(device_type='cuda', dtype=torch.bfloat16)(service._get_mask))
@service.app.after_request
def grounding_trace(response):
    from flask import g
    trace = g.get('grounding_fallback')
    if trace is not None and response.is_json:
        data = response.get_json()
        data['grounding_fallback'] = trace
        response.set_data(__import__('json').dumps(data))
    return response
service.app.run(host='127.0.0.1', port=21912, threaded=False, debug=False)
