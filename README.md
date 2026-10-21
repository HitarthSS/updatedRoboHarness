# Updated RoboHarness

An independent extension of [RoboHarness](https://github.com/LZY-1021/RoboHarness)
that reduces unnecessary planning calls, bounds recovery, prevents stale action
chunks, and adds a checked segmentation fallback around a **frozen Pi0.5 policy**.

This repository includes the repaired authors' harness, our improved arm, a
matched evaluation runner, offline checks, and the results of a **20-pair pilot**.
It is separate from the RPent Flash / Updated HarnessVLA project.

## Pilot results

Both arms used the same models, frozen initialization memory, initial simulator
states, and action horizons. Evaluation stopped at the user-requested 20 valid
matched pairs; two unavailable reset cases per arm are excluded from execution
denominators and retained in the published records.

| Metric | Authors' harness + shared repairs | Improved harness |
|---|---:|---:|
| Native simulator successes | 8/20 (40%) | 11/20 (55%) |
| Mean episode time | 23.1 minutes | 15.9 minutes |
| Direct planning/verification VLM calls | 464 | 363 |
| Mean charged action steps | 188.6 | 176.0 |
| Episodes reaching the action limit | 12 | 9 |
| Horizon violations | 0 | 0 |

There were **four improved-only successes and one baseline-only success**:
a net gain of three cases, or 15 percentage points. Recorded VLM calls decreased
22% and episode time decreased 31%.

**These are exploratory pilot results, not a full benchmark or paper-score
replication.** The cap was reduced after viewing interim outcomes. Coverage is
uneven: Spatial swap 10 pairs (5 vs 6 successes), Spatial task shift 8 (3 vs 3),
and Object swap 2 (0 vs 2). Object task shift, Goal, and LIBERO-10 were not tested
in this pilot. Shared busy GPUs confound wall-time comparisons. The call counter
excludes SAM3's internal bounding-box VLM queries; episode time includes their
synchronous execution. Episode timing starts after environment setup and
includes harness/memory setup; service startup and reset setup are excluded.

Machine-readable outcomes, model revisions, and the evaluated source hashes are
in [results/pilot-20.json](results/pilot-20.json). Recompute the tables without
models, GPUs, or network calls:

```bash
python3 summarize_results.py
```

## How the improved harness works

RoboHarness retrieves relevant successes/failures from a frozen experience bank.
Qwen inspects the current scene and can refine an instruction, propose visual
tools, or request a recovery. SAM3 grounds visual interventions. Pi0.5 produces
robot actions from the observation and instruction. Proposed interventions pass
through the released verification loop.

Our arm changes four parts of that loop:

1. **Adaptive monitoring:** check at step 0 and every 30 actions during visible
   change; check every 10 when the last two sampled RGB differences are below
   2/255. This is a visual-change heuristic, not a calibrated success detector.
2. **Bounded recovery:** use the last 20 policy actions to construct 10 clipped
   reverse actions and 5 holds, preserve the gripper command, allow at most two
   recoveries, enforce a 40-action cooldown, and reserve further policy budget.
3. **Fresh instructions and actions:** retain accepted nonvisual rewrites,
   discard queued policy actions after an accepted intervention/recovery, and
   expire highlight-specific language with the corresponding action chunk.
4. **Grounding fallback:** only when text grounding returns no mask, reuse the
   existing Qwen box and cached SAM3 image features for one positive-box pass.
   Require score >= 0.25 and mask-box IoU > 0.1. No extra VLM query is added by
   this fallback; an extra SAM3 pass may be needed.

`protocol.py` implements scheduling, recovery, action charging, and instruction
lifetime. `grounding.py` implements box validation and mask acceptance.
`run_episode.py` connects these to the released agent and simulator. There is no
policy training or evaluation-time memory update.

## Models

| Role | Frozen model |
|---|---|
| Robot policy | `lerobot/pi05-libero` |
| Live planning and verification | `Qwen/Qwen3-VL-32B-Instruct`, BF16, no quantization |
| Segmentation | Official SAM3 checkpoint; SHA-256 recorded |
| Memory encoders | `openai/clip-vit-large-patch14`, `sentence-transformers/all-MiniLM-L6-v2` |

All inference uses localhost services. There are no GPT, Claude, Codex, or NRP
inference calls. **This project does not replay RPent's GPT-generated Flash plans.**
Exact model revisions are in the results provenance; `download_models.py`
downloads those revisions rather than the current repository heads.

## Baseline repairs and evaluation limits

The baseline is the authors' **harness on top of the policy**, not bare Pi0.5.
Both arms receive these shared integration repairs:

- Enable one perception attempt: upstream `MAX_RETRIES = 0` prevents the loop
  from running.
- Supply the checkpoint-compatible EEF position, axis-angle, and both fingers;
  rotate main/wrist images by 180 degrees once, matching LeRobot's processor.
- Apply per-request SAM3 BF16 autocast and FP32 conversion for NumPy outputs.
- Detect swallowed inference/HTTP errors and reject invalid episodes or memory
  writes rather than silently counting them as benchmark results.

Unmodified upstream modules remain in `vendor/src/`. `repair_release.py` creates
a runtime copy with a repair manifest. Repairs are separate from candidate gains.

Action limits are **Spatial 220 / Object 280 at 1.0x**. Every policy, reverse, and
hold action is charged to the same episode budget. Fifteen reset-settling actions
are recorded separately; inclusive totals are also published. Native simulator
termination determines success. No extra episode attempts extend the budget.

Initialization used 40 ordinary LIBERO episodes (38 successes), then froze memory
before held-out LIBERO-PRO evaluation. The authors' initialized memory and exact
immutable policy checkpoint were unavailable. This experiment also differs from
the paper's 199 randomized rollouts per task, so it cannot directly reproduce
the paper's aggregate. Candidate settings were frozen before held-out outcomes;
the later sample-size reduction is explicitly recorded.

## Setup and execution

See [SETUP.md](SETUP.md) for the Linux/CUDA runtime, pinned dependencies,
checkpoint preparation, configuration, commands, and tests. The default queue
cap is **20 executed matched pairs** in `evaluation-limit.json`. The runner
retains the original larger task grid for optional future work; removing the
cap requests up to 400 cases per arm, of which 380 have released reset states in
the tested assets. It does not expand automatically after the pilot.

The release excludes machine logs, screenshots, videos, credentials, checkpoints,
and one-off server repair scripts. See [NOTICE.md](NOTICE.md) for attribution and
the distinction between evaluated source and portability edits in this release.
