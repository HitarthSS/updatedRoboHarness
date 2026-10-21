from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
from settings import BASE_PYTHON
BASE=BASE_PYTHON.parent.parent
subprocess.run([str(BASE/'bin/python'),'-m','venv',str(ROOT/'.venv-harness')],check=True)
site=ROOT/'.venv-harness/lib/python3.11/site-packages'
(site/'existing_runtime.pth').write_text(str(BASE/'lib/python3.11/site-packages')+'\n')
subprocess.run([str(ROOT/'.venv-harness/bin/pip'),'install','--no-cache-dir','flask','flask-cors'],check=True)
