"""Apply isolated full-eval setup in the parent and spawned simulator workers."""
import os,sys
os.environ.update(LIBERO_TYPE='pro',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
rpent_source=os.environ.get('RPENT_SOURCE')
if not rpent_source:
    raise RuntimeError('Set RPENT_SOURCE to the pinned RPent checkout before running the simulator integration')
sys.path.insert(0,rpent_source)
from robots.libero.env_server import build_env_cfg
from strict_bddl import install
install()
