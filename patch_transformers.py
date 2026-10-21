"""Apply the official OpenPI AdaRMS compatibility files only in our isolated venv."""
import hashlib
import json
from pathlib import Path
import urllib.request
import transformers
root=Path(__file__).resolve().parent
target=Path(transformers.__file__).parent
assert str(target).startswith(str(root/'.venv-policy311')),target
assert transformers.__version__=='4.53.2'
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'RoboHarness-reproduction'})
    return urllib.request.urlopen(req,timeout=45).read()
repo='https://api.github.com/repos/Physical-Intelligence/openpi'
revision='215abfb217dbac7d5f1273282331b9b1866c0479'
tree=json.loads(get(repo+'/git/trees/'+revision+'?recursive=1'))['tree']
prefix='src/openpi/models_pytorch/transformers_replace/'
paths=[x['path'] for x in tree if x['type']=='blob' and x['path'].startswith(prefix) and x['path'].endswith('.py')]
assert len(paths)>=4
manifest={}
for path in paths:
    content=get('https://raw.githubusercontent.com/Physical-Intelligence/openpi/'+revision+'/'+path)
    if path.endswith('/check.py'):
        assert b'4.53.2' in content
    dest=target/path.removeprefix(prefix)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(content)
    manifest[str(dest.relative_to(target))]=hashlib.sha256(content).hexdigest()
(root/'transformers-replacement.json').write_text(json.dumps(dict(openpi_revision=revision,transformers_version=transformers.__version__,files=manifest),indent=2)+'\n')
print('Official OpenPI patches installed',revision,len(paths))
