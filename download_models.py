"""Download the exact open model revisions used in the pilot, not latest main."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
pilot = json.loads((root/'results/pilot-20.json').read_text())['experiment']
models = dict(pilot['models'], **{'qwen3-vl-32b': pilot['orchestrator']})
for name, item in models.items():
    print('Downloading', item['repo'], item['revision'], flush=True)
    snapshot_download(item['repo'], revision=item['revision'],
        local_dir=root/'checkpoints'/name,
        ignore_patterns=['*.bin', '*.msgpack', '*.h5', '*.onnx', 'onnx/*', 'openvino/*'],
        max_workers=3)
support = {k: models[k] for k in pilot['models']}
(root/'checkpoints/support-manifest.json').write_text(json.dumps(support, indent=2)+'\n')
(root/'checkpoints/manifest.json').write_text(json.dumps(models, indent=2)+'\n')
print('SAM3 and the original tokenizer must be provided separately; see SETUP.md.')
