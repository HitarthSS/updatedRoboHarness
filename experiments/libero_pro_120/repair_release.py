"""Create an auditable runtime copy, keeping the upstream checkout unchanged."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


def prepare():
    source = ROOT/'upstream'/'src'
    if not source.exists():
        source = ROOT/'vendor'/'src'
    target = ROOT/'runtime'
    target.mkdir(exist_ok=True)
    manifest = {}
    for path in sorted(source.glob('*.py')):
        original = path.read_text()
        text = original
        changes = []
        if path.name == 'soma_perception.py':
            assert text.count('MAX_RETRIES = 0') == 1
            text = text.replace('MAX_RETRIES = 0', 'MAX_RETRIES = 1')
            changes.append('Enable one planner/tool attempt; released loop executes zero times.')
        if path.name == 'sam3_service.py':
            text = text.replace('.detach().cpu().numpy()', '.detach().float().cpu().numpy()')
            changes.append('Convert BF16 inference outputs to FP32 before NumPy conversion.')
            text = text.replace('from flask import Flask, jsonify, request',
                'from flask import Flask, jsonify, request, has_request_context, g')
            text = text.replace('    if vlm_client:\n', '    bbox = None\n    if vlm_client:\n')
            empty = '    if len(scores) == 0: return np.zeros(image.shape[:2], dtype=np.uint8), 0.0'
            assert text.count(empty) == 1
            text = text.replace(empty, '''    if len(scores) == 0:
        adaptive = has_request_context() and bool((request.get_json(silent=True) or {}).get("adaptive_grounding"))
        if adaptive and bbox is not None:
            from grounding import grounded_box_mask
            trace = {}
            recovered = grounded_box_mask(sam3_predictor, state, bbox, image.shape[:2], score_th, trace)
            g.grounding_fallback = trace
            return recovered
        return np.zeros(image.shape[:2], dtype=np.uint8), 0.0''')
            changes.append('Optional request-gated geometric fallback; disabled for baseline requests.')
        (target/path.name).write_text(text)
        manifest[path.name] = dict(upstream_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                  runtime_sha256=hashlib.sha256(text.encode()).hexdigest(), changes=changes)
    (ROOT/'release-repairs.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return target


if __name__ == '__main__':
    prepare()
