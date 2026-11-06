# My harness for RoboHarness

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2603.24060). The evaluated behavior is in [`experiments/libero_pro_120/protocol.py`](experiments/libero_pro_120/protocol.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | 57.2% position shifts; 55.2% task shifts |

| Our run of the authors' harness | 35/120 (29.17%) |

| **My harness** | **42/120 (35.00%)** |


Twenty mapped LIBERO-PRO conditions, six fresh states each. Position: 19/36 versus 20/36. Task: 16/84 versus 22/84.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Schedule RGB-driven supervision between 10 and 30 actions, checking more often during persistent stillness.

2. Use bounded reverse/hold recovery while preserving the gripper command and charging every action.

3. Keep accepted instruction refinements, expire image-only rewrites, and discard stale queued actions after interventions.

4. Use a checked Qwen-box/SAM3 fallback when text grounding is empty.


Categories: `observation_grounding`, `execution_monitoring`, `uncertainty_compute`, `language_alignment`.

## Models and inputs

- Frozen lerobot/pi05-libero at 9ba5431256f4136a01c224ab0307b0d3035cc92c.

- Qwen/Qwen3-VL-32B-Instruct BF16 at 0cfaf48183f594c314753d30a4c4974bc75f3ccb; SAM3, CLIP and MiniLM support models.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/LZY-1021/RoboHarness) before installing its model stack.
2. Controller dependency: Use the existing SETUP.md for isolated policy and harness environments.
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
PYTHONPATH=experiments/libero_pro_120 python -m unittest discover -s experiments/libero_pro_120 -p "test_protocol.py" -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Use AdaptiveGovernor, Budget and InterventionState in the native action loop. The evaluated integration is run_episode.py beside these modules; policy and perception services run locally.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

Spatial 220 / Object 280 controlled actions, 1.0x; recovery included. Fifteen reset-settling actions are reported separately.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `experiments/libero_pro_120/protocol.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Paper task mappings are inferred, not author-confirmed.

- The exact paper memory, randomized-state manifest and immutable policy checkpoint are unavailable. Our frozen memory is a separately initialized bank.

- The earlier 20-case result is invalid as a controlled comparison because most initial RGB pairs differed.

- Positive paired pilot difference; not full paper reproduction or a confirmed benchmark win.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

6 CPU tests passed; 1 environment-specific contract test skipped. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
