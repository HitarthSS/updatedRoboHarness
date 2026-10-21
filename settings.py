"""Machine-specific paths; experiment logic and model identity stay fixed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RPENT_ROOT = Path(os.environ.get('ROBOHARNESS_RPENT_ROOT', ROOT/'dependencies/RPent')).expanduser().resolve()
BASE_PYTHON = Path(os.environ.get('ROBOHARNESS_BASE_PYTHON', RPENT_ROOT/'.venv/bin/python')).expanduser().absolute()
QWEN_PYTHON = Path(os.environ.get('ROBOHARNESS_QWEN_PYTHON', ROOT/'.venv-qwen/bin/python')).expanduser().absolute()
SAM3_CHECKPOINT = Path(os.environ.get('ROBOHARNESS_SAM3_CHECKPOINT', ROOT/'checkpoints/sam3/sam3.pt')).expanduser().resolve()
TOKENIZER_MODEL = Path(os.environ.get('ROBOHARNESS_TOKENIZER_MODEL', Path.home()/'.cache/openpi/big_vision/paligemma_tokenizer.model')).expanduser().resolve()
