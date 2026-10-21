# Setup and reproduction

This is a research runner layered on an existing **Linux, Python 3.11, CUDA
RPent/LIBERO installation**. It is not a standalone simulator distribution.
The published portable paths and download helpers have been checked offline;
a complete fresh-machine installation has not been rerun for this release.

## 1. Prepare the simulator and base runtime

Install [RLinf/RPent](https://github.com/RLinf/RPent) at
`945682a55c4e17d6bb686a80b162ff801e768157` following its LIBERO installation
instructions. Install both ordinary LIBERO assets (initialization) and
LIBERO-PRO Spatial/Object swap/task assets (evaluation). Retain the released
initial-state files, including empty ones. RPent supplies only the simulator
integration here: this project does not use its Flash or Codex planners.

The prepared Python 3.11 base environment must expose RPent/RLinf and its
simulator dependencies, SAM3, and the inference libraries needed by the upstream
RoboHarness modules. The tested base used:

| Package | Version |
|---|---|
| torch / torchvision | 2.7.1+cu126 / 0.22.1 |
| transformers | 4.57.6 |
| SAM3 | 0.1.0 |
| robosuite / mujoco | 1.5.2 / 3.3.0 |
| numpy | 1.26.4 |
| openai (localhost client only) | 3.35.1 |
| Flask | 3.1.3 |
| huggingface-hub | 0.36.2 |
| accelerate | 1.15.0 |

These versions describe the tested environment, not a complete dependency lock.
The base runtime is a prerequisite, not something the overlay scripts install
from scratch. Follow the official [SAM3 installation](https://github.com/facebookresearch/sam3)
for its code and checkpoint access. Package version alone does not identify
the exact SAM3 source checkout; its original source revision was not recorded.

## 2. Set machine paths

From this repository's root, set paths for your prepared installation:

```bash
export ROBOHARNESS_RPENT_ROOT=/absolute/path/to/RPent
export ROBOHARNESS_BASE_PYTHON=/absolute/path/to/base-venv/bin/python
export ROBOHARNESS_QWEN_PYTHON=/absolute/path/to/qwen-venv/bin/python
export ROBOHARNESS_SAM3_CHECKPOINT=/absolute/path/to/sam3.pt
export ROBOHARNESS_TOKENIZER_MODEL=/absolute/path/to/paligemma_tokenizer.model
export ROBOHARNESS_POLICY_GPU=0
export ROBOHARNESS_QWEN_GPUS=0,1,2,3
```

Keep these variables set for setup and launch. The two Python variables must
point to virtual-environment interpreters, preserving their venv paths.
`settings.py` also documents defaults under this checkout. Do not copy an
existing machine's PID files or experiment outputs into a new checkout.

SAM3 checkpoint used in the pilot:

```text
bytes: 3450062241
sha256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

The original SAM3 download revision was not recorded. An equivalent filename
alone does not prove matching weights. Obtain the original PaliGemma tokenizer
through your authorized OpenPI setup. The conversion helper checks all token
IDs and example strings before writing the HF tokenizer under `tokenizer/`.

## 3. Prepare the isolated environments and assets

```bash
python3 setup_harness_env.py
.venv-harness/bin/python download_models.py
python3 setup_policy_env.py
.venv-harness/bin/python audit_assets.py
```

The setup scripts create `.venv-harness` and `.venv-policy311` overlays referencing
the prepared base environment. Policy installation pins LeRobot 0.4.4 and
Transformers 4.53.2, then installs the official OpenPI AdaRMS compatibility files
from commit `215abfb217dbac7d5f1273282331b9b1866c0479`. It changes the isolated
policy overlay, not the base environment. `download_models.py` pins the four
downloaded checkpoints to the revisions in `results/pilot-20.json`; it does not
download SAM3 or the original tokenizer.

The Qwen service needs a separate vLLM environment. The pilot used vLLM 0.32.0,
Torch 2.13.0+cu132, and Transformers 5.19.0 in that environment. Keep it separate
from the simulator and policy environments to avoid dependency conflicts.

## 4. Validate without starting model inference

```bash
.venv-harness/bin/python -m unittest \
  test_protocol test_release_loop test_infrastructure test_grounding -q
.venv-policy311/bin/python -m unittest test_protocol -q
python3 summarize_results.py
```

The harness suite contains 13 checks, with the official LeRobot camera-contract
check skipped when LeRobot is absent. The policy environment runs all seven
protocol checks, including that contract. These are infrastructure tests, not
new benchmark episodes.

## 5. Launch the campaign

```bash
.venv-harness/bin/python start_qwen.py
# Wait until the local service is ready before starting the campaign:
curl --fail http://127.0.0.1:21910/v1/models
.venv-harness/bin/python start_campaign.py
.venv-harness/bin/python progress.py
```

`start_qwen.py` uses the pilot's four-stage pipeline partition (28/12/12/12),
BF16, no quantization or CPU offload, 4096 context, and bounded image resolution.
The default four visible GPU IDs are configurable. This topology was chosen
for shared 48 GB GPUs and is not an optimized universal serving recipe. The
policy and SAM3 use the configured policy GPU; allow room for them in addition
to Qwen when sharing a device. CUDA and vLLM must be compatible with your
driver. Adapt hardware serving settings explicitly and use the same settings
for both arms. The runner records them in `qwen-serving.json` / `experiment.json`.

Services bind to localhost: Qwen 21910, policy 21911, SAM3 21912. The OpenAI
Python package is an HTTP client for local Qwen; its `local-only` placeholder is
not an external API credential. No external planner fallback exists.

The campaign initializes memory with 40 standard LIBERO episodes, freezes it,
and then alternates baseline/improved order for each matched case. It resumes
existing result files and stops on infrastructure errors. It checks the frozen
protocol when resuming; change experiment directories rather than mixing
different implementations into one result set.

`evaluation-limit.json` caps the queue at **20 executed matched pairs**. Empty
official reset cases are recorded but do not count toward the cap. Increasing
the cap or removing that file is an explicit new evaluation decision; the full
grid requests 400 cases per arm (380 available in the tested assets).

Inspect `status.json`, `comparison.json`, `logs/`, and `runs/<arm>/<case>/` for
live outputs. These generated files are ignored by Git. On completion the
manager shuts down model processes it started or explicitly adopted using this
checkout's PID files; it does not stop unrelated workloads. The published
results under `results/` are separate from newly generated campaign outputs.
