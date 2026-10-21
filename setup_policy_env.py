from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
from settings import BASE_PYTHON
BASE=BASE_PYTHON.parent.parent
TARGET=ROOT/'.venv-policy311'
subprocess.run([str(BASE/'bin/python'),'-m','venv',str(TARGET)],check=True)
(TARGET/'lib/python3.11/site-packages/existing_runtime.pth').write_text(str(BASE/'lib/python3.11/site-packages')+'\n')
subprocess.run([str(TARGET/'bin/pip'),'install','--no-cache-dir','--no-deps','lerobot==0.4.4'],check=True)

deps=[line.strip() for line in (ROOT/'requirements-policy.txt').read_text().splitlines() if line.strip() and not line.startswith('#') and not line.startswith('lerobot')]
subprocess.run([str(TARGET/'bin/pip'),'install','--no-cache-dir',*deps],check=True)
subprocess.run([str(TARGET/'bin/python'),str(ROOT/'patch_transformers.py')],check=True)
subprocess.run([str(TARGET/'bin/python'),str(ROOT/'prepare_tokenizer.py')],check=True)
