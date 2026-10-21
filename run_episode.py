"""Run a frozen RoboHarness arm on official LIBERO resets with a strict horizon.

LeRobot policy runs in an isolated service. The simulator uses installed official
LIBERO/LIBERO-PRO assets, not RoboHarness's custom OOD suite. No policy training.
"""
import argparse
import base64
from collections import deque
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import types
import numpy as np
from PIL import Image
import requests
from protocol import Budget, AdaptiveGovernor, HORIZONS, InterventionState, libero_policy_image
from repair_release import prepare

ROOT = Path(__file__).resolve().parent
POLICY_URL = 'http://127.0.0.1:21911'
MODEL = 'qwen3-vl-32b-instruct'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def image64(frame):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def policy_request(route, payload=None):
    res = requests.post(POLICY_URL+route, json=payload or {}, timeout=300)
    if not res.ok:
        raise RuntimeError(f'Policy service HTTP {res.status_code}: {res.text[:2000]}')
    data = res.json()
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data


class AuditedPost:
    """Keep infrastructure errors visible even when upstream catches them."""
    def __init__(self, original, adaptive_grounding=False):
        self.original = original
        self.adaptive_grounding = adaptive_grounding
        self.calls = self.errors = self.rejections = 0
        self.grounding_fallback_calls = self.grounding_fallback_successes = 0
        self.seconds = 0.0

    def __call__(self, path, payload):
        start = time.monotonic()
        self.calls += 1
        try:
            if self.adaptive_grounding:
                payload = dict(payload, adaptive_grounding=True)
            response = self.original(path, payload)
            self.rejections += int(response.get('success') is False)
            trace = response.get('grounding_fallback') or {}
            self.grounding_fallback_calls += int(bool(trace.get('attempted')))
            self.grounding_fallback_successes += int(bool(trace.get('accepted')))
            return response
        except Exception:
            self.errors += 1
            raise
        finally:
            self.seconds += time.monotonic()-start


class AuditedCompletions:
    def __init__(self, original):
        self.original = original
        self.calls = self.errors = 0
        self.seconds = 0.0
        self.responses = []

    def create(self, **kwargs):
        start = time.monotonic()
        self.calls += 1
        try:
            response = self.original.create(**kwargs)
            if response.model.lower() != MODEL:
                raise RuntimeError('Unexpected model: '+str(response.model))
            text = response.choices[0].message.content
            # All upstream prompts require a JSON object. Detect swallowed model failures.
            from soma_vlm import _extract_json
            if not isinstance(_extract_json(text or ''), dict):
                raise RuntimeError('VLM response was not a JSON object')
            self.responses.append(text)
            return response
        except Exception:
            self.errors += 1
            raise
        finally:
            self.seconds += time.monotonic()-start


def create_agent(memory, adaptive_grounding=False):
    sys.path.insert(0, str(prepare()))
    import soma_agent
    from soma_encoder import AdvancedEmbeddingEncoder
    from openai import OpenAI
    class Encoder(AdvancedEmbeddingEncoder):
        def __init__(self, **kwargs):
            kwargs.update(vision_model_name=str(ROOT/'checkpoints/clip'),
                          text_model_name=str(ROOT/'checkpoints/minilm'), device='cpu')
            super().__init__(**kwargs)
            if self.text_model is None:
                raise RuntimeError('MiniLM loading failed; do not silently change memory embeddings')
    soma_agent.AdvancedEmbeddingEncoder = Encoder
    agent = soma_agent.SOMAAgent(dict(device='cpu', memory_dir=str(memory),
        vlm_base_url='http://127.0.0.1:21910/v1', vlm_api_key='local-only',
        model_id=MODEL, sam3_base_url='http://127.0.0.1:21912', tool_timeout_s=1800))
    # Both generation and verification use the same audited exact local model.
    client = OpenAI(base_url='http://127.0.0.1:21910/v1', api_key='local-only', timeout=1200, max_retries=0)
    audit = AuditedCompletions(client.chat.completions)
    agent.vlm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=audit))
    audit.tools = AuditedPost(agent.perception.tools.client._post, adaptive_grounding)
    agent.perception.tools.client._post = audit.tools
    # The released logger swallows diagnosis errors and otherwise stores an empty
    # "Unknown" diagnosis. Such infrastructure failures must not enter the bank.
    add_experience = agent.memory.add_experience
    def checked_add_experience(*args, **kwargs):
        if audit.errors:
            raise RuntimeError('Refusing memory write after an inference error')
        return add_experience(*args, **kwargs)
    agent.memory.add_experience = checked_add_experience
    agent.perception.debug_output_dir = ROOT/'debug'
    return agent, audit


def episode(args):
    # These must precede importing simulator libraries (which cache LIBERO_TYPE).
    os.environ['LIBERO_TYPE'] = 'standard' if args.arm == 'cold-start' else 'pro'
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.set_num_threads(4)
    from settings import RPENT_ROOT
    sys.path.insert(0, str(RPENT_ROOT))
    from robots.libero.env_server import LiberoEnvFacade
    from env_adapter import make_budgeted_env
    from rlinf.envs.sim.libero.utils import benchmark
    from soma_control_flow import RollbackState, TaskDecomposeState, KeyStepRetryState

    np.random.seed(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    horizon = HORIZONS[args.suite.split('_')[1]]
    result = dict(arm=args.arm, suite=args.suite, task=args.task, seed=args.seed,
                  horizon=horizon, horizon_multiplier=1.0, executed=False,
                  success=False, reset_settling_steps=15)
    suite = benchmark.get_benchmark(args.suite)()
    states = suite.get_task_init_states(args.task)
    if not len(states):
        result['status'] = 'unavailable_initial_states'
        atomic(args.output/'result.json', result)
        return result
    state = np.asarray(states[args.seed % len(states)])
    result['initial_state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
    # RLinf reset settles for 15 steps before resetting its action counter.
    # An external Budget still guards every environment action after settling.
    setup_start = time.monotonic()
    env = make_budgeted_env(args.task, args.seed, args.suite, horizon)
    facade = LiberoEnvFacade(env, meta={})
    obs, _ = facade.reset()
    def native_images(obs):
        raw = env.current_raw_obs[0]
        obs['main_images'] = libero_policy_image(raw['agentview_image'])
        obs['wrist_images'] = libero_policy_image(raw['robot0_eye_in_hand_image'])
        return obs
    obs = native_images(obs)
    task = str(obs['task_descriptions'])
    result['instruction'] = task
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs['main_images']).save(args.output/'initial.png')
    policy_request('/reset', {'seed': args.seed})
    budget = Budget(horizon)
    governor = AdaptiveGovernor()
    intervention = InterventionState(task)
    policy_calls = policy_forwards = policy_replans = 0
    policy_seconds = 0.0
    rollback, task_ctrl, retry = RollbackState(), TaskDecomposeState(), KeyStepRetryState()
    accumulated = np.zeros((1,7), dtype=np.float32)
    recovery_queue = deque()
    start = time.monotonic()
    setup_seconds = start-setup_start
    agent = audit = None
    if args.arm != 'cold-start':
        agent, audit = create_agent(ROOT/'memory-frozen', adaptive_grounding=args.arm == 'improved')
        rag = agent.init_episode(obs['main_images'], task)
        if not rag or not any(rag.values()):
            raise RuntimeError('Frozen memory retrieval returned no experiences')
    else:
        rag = {}
    events, frames = [], [Image.fromarray(obs['main_images'])]
    success = False
    try:
        while budget.remaining:
            step = budget.steps
            current_task = task_ctrl.current_task(
                default_task=intervention.task if args.arm == 'improved' else task)
            frame = obs['main_images']
            recovering = bool(recovery_queue) if args.arm == 'improved' else rollback.active
            perceive = agent is not None and not recovering and (
                governor.should_perceive(step, frame) if args.arm == 'improved' else step%10 == 0)
            flags = {}
            if perceive:
                before_errors = (audit.errors, audit.tools.errors)
                frame, refined, flags = agent.step(frame, current_task, step, rag)
                if (audit.errors, audit.tools.errors) != before_errors:
                    raise RuntimeError('Upstream swallowed an inference error; episode is invalid')
                if step == 0 and audit.calls == 0:
                    raise RuntimeError('Harness did not call its VLM')
                if refined and not task_ctrl.subtasks:
                    current_task = refined
                    if args.arm == 'improved':
                        intervention.update(refined=refined,
                            image_modified=flags.get('image_modified', False))
                task_ctrl.update_from_control_flags(flags)
                retry.update_from_control_flags(flags)
                if flags.get('encore'):
                    if args.arm == 'improved':
                        recovery_queue.extend(governor.recovery(step, budget))
                    else:
                        rollback.start_from_accumulated(accumulated, reverse_steps=40, buffer_steps=20, reason='SOMA_Encore')
                events.append(dict(step=step, kind='perception', flags=flags, refined_task=current_task,
                                   vlm_calls=audit.calls))
            recovering = bool(recovery_queue) if args.arm == 'improved' else rollback.active
            if args.arm == 'improved':
                intervention.update(image_modified=flags.get('image_modified', False),
                                    recovering=recovering)
            if recovering:
                action = recovery_queue.popleft() if args.arm == 'improved' else rollback.step_action(1)[0]
                accumulated.fill(0)
            else:
                # Official pi05-libero statistics describe EEF xyz + axis-angle + both fingers.
                # The release's joint-position flattening violates this checkpoint contract.
                state = np.asarray(obs['states'], dtype=np.float32)
                if args.arm == 'improved' and intervention.needs_replan(current_task):
                    policy_request('/replan')
                    policy_replans += 1
                prediction = policy_request('/action', dict(task=current_task, main=image64(frame),
                    wrist=image64(obs['wrist_images']), state=state.tolist()))
                action = np.array(prediction['action'], dtype=np.float32)
                policy_calls += 1
                policy_forwards += int(prediction['policy_forward'])
                policy_seconds += prediction['policy_seconds']
                if args.arm == 'improved':
                    intervention.commit(current_task, prediction['queued_actions_remaining'])
                accumulated += action[None]
                governor.record(action)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError('Invalid action')
            # Charge once for every native env.step, including reversals and buffers.
            budget.charge(recovery=recovering)
            obs, _, terminated, truncated, info = facade.step(action)
            obs = native_images(obs)
            if int(env.elapsed_steps[0]) != budget.steps:
                raise RuntimeError('Native simulator and horizon counters diverged')
            success = bool(np.asarray(terminated).any())
            task_ctrl.maybe_advance(success_any=success)
            if retry.should_trigger(step=step, success_any=success):
                retry.mark_triggered()
                if args.arm == 'improved':
                    recovery_queue.extend(governor.recovery(budget.steps, budget))
                else:
                    rollback.start_from_accumulated(accumulated, reverse_steps=50, reason='Timeout_Retry')
            if budget.steps%50 == 0:
                frames.append(Image.fromarray(obs['main_images']))
            if success or bool(np.asarray(truncated).any()):
                break
        Image.fromarray(obs['main_images']).save(args.output/'final.png')
        result.update(executed=True, success=success, status='success' if success else 'horizon',
                      steps=budget.steps, total_native_steps_including_settling=budget.steps+15,
                      recovery_steps=budget.recovery_steps,
                      policy_action_calls=policy_calls, policy_forward_calls=policy_forwards,
                      policy_replans=policy_replans, policy_seconds=policy_seconds,
                      wall_seconds=time.monotonic()-start, setup_seconds=setup_seconds,
                      total_episode_seconds=time.monotonic()-setup_start,
                      vlm_calls=audit.calls if audit else 0, vlm_seconds=audit.seconds if audit else 0,
                      tool_calls=audit.tools.calls if audit else 0,
                      tool_seconds=audit.tools.seconds if audit else 0,
                      tool_rejections=audit.tools.rejections if audit else 0,
                      grounding_fallback_calls=audit.tools.grounding_fallback_calls if audit else 0,
                      grounding_fallback_successes=audit.tools.grounding_fallback_successes if audit else 0,
                      perception_calls=len(events), events=events,
                      model_responses=audit.responses if audit else [])
        if args.arm == 'cold-start':
            agent, audit = create_agent(ROOT/'memory-development')
            frames.append(Image.fromarray(obs['main_images']))
            before_count = sum(agent.memory.get_stats()[k] for k in ['success_count','failure_count'])
            agent.logger.log_episode(task_desc=task, success=success, video_path=str(args.output/'episode.mp4'),
                keyframe_path=str(args.output/'initial.png'), frames=frames,
                additional_info={'split':'standard_LIBERO_initialization', 'seed':args.seed})
            if audit.errors:
                raise RuntimeError('Memory diagnosis failed')
            if sum(agent.memory.get_stats()[k] for k in ['success_count','failure_count']) != before_count+1:
                raise RuntimeError('Memory write failed')
            result['initialization_vlm_calls'] = audit.calls
            result['initialization_vlm_seconds'] = audit.seconds
        if budget.steps > horizon:
            raise AssertionError('horizon violated')
        atomic(args.output/'result.json', result)
        return result
    finally:
        facade.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['cold-start','baseline','improved'], required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0,str(prepare()))
    logging.basicConfig(level=logging.INFO)
    try:
        print(json.dumps(episode(args)), flush=True)
    except Exception as exc:
        atomic(args.output/'error.json', {'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
