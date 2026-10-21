from pathlib import Path
import os
import subprocess
root=Path(__file__).resolve().parent
(root/'logs').mkdir(exist_ok=True)
with (root/'logs/campaign.log').open('a') as log:
    child=subprocess.Popen([str(root/'.venv-harness/bin/python'),'-u',str(root/'campaign.py')],
        cwd=root,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
(root/'campaign.pid').write_text(str(child.pid)+'\n')
print(child.pid)
