"""Isolated LeRobot pi05 inference process, accessed only over localhost."""
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent


def main():
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    checkpoint = ROOT/'checkpoints'/'pi05-libero'
    from lerobot.configs.policies import PreTrainedConfig
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.compile_model = False  # Same eager execution in both arms on shared GPUs.
    policy = PI05Policy.from_pretrained(str(checkpoint), config=config).to('cuda').eval()
    policy.config.device = 'cuda'
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                 'tokenizer_processor': {'tokenizer_name': str(ROOT/'tokenizer')}})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ready":true}')

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/reset':
                    torch.manual_seed(int(payload['seed']))
                    torch.cuda.manual_seed_all(int(payload['seed']))
                    policy.reset(); result = {'ok': True}
                elif self.path == '/replan':
                    # Discard queued actions without reseeding policy sampling.
                    policy.reset(); result = {'ok': True}
                else:
                    obs = {'task': [payload['task']],
                           'observation.state': torch.tensor(payload['state'], dtype=torch.float32)[None]}
                    for key, image in [('image', payload['main']), ('image2', payload['wrist'])]:
                        arr = np.array(Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB'))
                        obs['observation.images.'+key] = torch.from_numpy(arr.copy()).permute(2,0,1)[None].float()/255
                    with torch.inference_mode():
                        forward = not policy._action_queue
                        started = time.monotonic()
                        action = post(policy.select_action(pre(obs))).cpu().numpy()[0]
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError('invalid policy action')
                    result = {'action': action.tolist(), 'policy_forward': forward,
                              'queued_actions_remaining': len(policy._action_queue),
                              'policy_seconds': time.monotonic()-started}
                body = json.dumps(result).encode()
                self.send_response(200)
            except Exception as exc:
                body = json.dumps({'error': str(exc)}).encode(); self.send_response(500)
            self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)

        def log_message(self, *args):
            pass

    print('LeRobot pi05 ready on 127.0.0.1:21911', flush=True)
    HTTPServer(('127.0.0.1', 21911), Handler).serve_forever()


if __name__ == '__main__':
    main()
