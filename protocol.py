"""Frozen episode accounting and candidate decisions; no simulator access."""
from dataclasses import dataclass, field
import numpy as np

HORIZONS = {'spatial': 220, 'object': 280, 'goal': 300, '10': 520}


def libero_policy_image(raw):
    """Match LeRobot LiberoProcessorStep's camera rotation exactly once."""
    return np.asarray(raw)[::-1, ::-1].copy()


@dataclass
class InterventionState:
    """Keep accepted instructions and invalidate stale queued policy actions."""
    task: str
    base_task: str = field(init=False)
    last_policy_task: str | None = None
    pending_replan: bool = False
    visual_instruction: bool = False
    queued_actions: int = 0

    def __post_init__(self):
        self.base_task = self.task

    def update(self, refined=None, image_modified=False, recovering=False):
        if refined:
            self.task = refined
            self.visual_instruction = bool(image_modified)
        if recovering and self.visual_instruction:
            self.task = self.base_task
            self.visual_instruction = False
        self.pending_replan |= bool(image_modified or recovering)

    def needs_replan(self, current_task):
        return self.queued_actions > 0 and self.last_policy_task is not None and (
            self.pending_replan or current_task != self.last_policy_task)

    def commit(self, current_task, queued_actions=1):
        self.last_policy_task = current_task
        self.queued_actions = queued_actions
        self.pending_replan = False
        # A visual rewrite describes the modified frame that produced this
        # chunk. Future raw frames must not refer to an expired highlight.
        if self.visual_instruction and queued_actions == 0:
            self.task = self.base_task
            self.visual_instruction = False


@dataclass
class Budget:
    limit: int
    steps: int = 0
    recovery_steps: int = 0

    @property
    def remaining(self):
        return max(0, self.limit - self.steps)

    def charge(self, recovery=False):
        if not self.remaining:
            raise RuntimeError('episode horizon exhausted')
        self.steps += 1
        self.recovery_steps += int(recovery)


@dataclass
class AdaptiveGovernor:
    """Use observed RGB change to allocate calls; bound repeated recovery."""
    last_call: int = -30
    previous_frame: object = None
    changes: list = field(default_factory=list)
    recent_actions: list = field(default_factory=list)
    last_recovery: int = -40
    recoveries: int = 0

    def should_perceive(self, step, frame):
        frame = np.asarray(frame)[::8, ::8].astype(np.float32)
        if self.previous_frame is not None:
            self.changes.append(float(np.abs(frame-self.previous_frame).mean()))
            self.changes = self.changes[-2:]
        self.previous_frame = frame.copy()
        elapsed = step-self.last_call
        stalled = len(self.changes) == 2 and max(self.changes) < 2.0
        decision = step == 0 or elapsed >= 30 or (elapsed >= 10 and stalled)
        if decision:
            self.last_call = step
        return decision

    def record(self, action):
        self.recent_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.recent_actions = self.recent_actions[-20:]

    def recovery(self, step, budget):
        if (self.recoveries >= 2 or step-self.last_recovery < 40
                or len(self.recent_actions) < 5 or budget.remaining < 20):
            return []
        recent = np.stack(self.recent_actions)
        # Reverse a bounded recent displacement, preserving the gripper command.
        inverse = np.clip(-recent[:, :6].sum(0)/10, -1, 1)
        command = np.r_[inverse, recent[-1, -1]].astype(np.float32)
        hold = np.zeros_like(command)
        hold[-1] = command[-1]
        self.last_recovery, self.recoveries = step, self.recoveries+1
        self.recent_actions.clear()
        return [command.copy() for _ in range(10)] + [hold.copy() for _ in range(5)]
