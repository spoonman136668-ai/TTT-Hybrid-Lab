"""P54 ENG: pinned published MountainCarContinuous-v0 physical mechanics bridge.

Original upstream: Farama-Foundation/Gymnasium e6be96abfb0a9df36da5e7884522c4f1f2432e53
  gymnasium/envs/classic_control/continuous_mountain_car.py
  Git blob 96dbb0d31a17d144468562f4bc68fce902e2a69f

This independent small reference transcribes its documented transition equations;
*running this implementation is NOT the same as running the pinned official code*.
The optional official-code check requires the installed exact upstream source.
All actual physical steps precede factual memory commits; preview never calls step.
No trained causal-science performance, Gymnasium runtime acceptance, or deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from native_factual_hybrid_stream import FactualStreamState, NativeFactualHybridStream

UPSTREAM_COMMIT = 'e6be96abfb0a9df36da5e7884522c4f1f2432e53'
UPSTREAM_GIT_BLOB = '96dbb0d31a17d144468562f4bc68fce902e2a69f'
UPSTREAM_PATH = 'gymnasium/envs/classic_control/continuous_mountain_car.py'
ENV_ID = 'MountainCarContinuous-v0'


def source_git_blob(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError('P54_SOURCE_BYTES_REQUIRED')
    return hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()


def verify_pinned_official_source(path: str | Path) -> str:
    """Identity-check *authentic installed upstream bytes*, not merely its version."""
    source = Path(path).read_bytes()
    sha = source_git_blob(source)
    if sha != UPSTREAM_GIT_BLOB:
        raise ValueError('P54_UPSTREAM_GYMNASIUM_BLOB_MISMATCH')
    return sha


@dataclass(frozen=True)
class PhysicalStep:
    next_sensor: np.ndarray  # [position, velocity] float32
    reward: float
    terminated: bool


def published_step(sensor: np.ndarray, action: float) -> PhysicalStep:
    """Independent one-step transcription of the fixed upstream mechanics.

    The environment's official time-limit 999 applies at a wrapper level.
    The P54 *bridge* admits only executed nonzero motor commands in [-1,+1],
    but this published physics reference supports the original full action range.
    """
    state = np.asarray(sensor)
    if state.shape != (2,) or not np.issubdtype(state.dtype, np.number):
        raise ValueError('P54_BAD_PHYSICAL_SENSOR')
    if not np.isfinite(state).all():
        raise ValueError('P54_NONFINITE_PHYSICAL_SENSOR')
    if not isinstance(action, (float, int, np.floating, np.integer)) or not math.isfinite(float(action)):
        raise ValueError('P54_NONFINITE_PHYSICAL_ACTION')
    pos, vel = float(state[0]), float(state[1])
    if not -1.2 - 1e-6 <= pos <= .6 + 1e-6 or not -.07 - 1e-6 <= vel <= .07 + 1e-6:
        raise ValueError('P54_PHYSICAL_SENSOR_OUT_OF_BOUNDS')
    # Published Gymnasium Continuous_MountainCarEnv.step transition.
    force = min(max(float(action), -1.), 1.)
    vel = max(min(vel + force * .0015 - .0025 * math.cos(3 * pos), .07), -.07)
    pos = max(min(pos + vel, .6), -1.2)
    if pos == -1.2 and vel < 0:
        vel = 0.
    terminated = bool(pos >= .45 and vel >= 0)
    reward = (100. if terminated else 0.) - .1 * float(action)**2
    return PhysicalStep(np.array([pos, vel], dtype=np.float32), float(reward), terminated)


def published_reset(seed: int, low: float = -.6, high: float = -.4) -> np.ndarray:
    """PCG64 Generator reset compatible with Gymnasium's seeded default RNG."""
    if type(seed) is not int or seed < 0 or not (-1.2 <= low < high <= .6):
        raise ValueError('P54_INVALID_RESET_SOURCE')
    rng = np.random.default_rng(seed)
    return np.array([rng.uniform(low, high), 0.], dtype=np.float32)


@dataclass(frozen=True)
class FactualPhysicalSession:
    physical: torch.Tensor        # [batch,2] actual state returned by physical source
    memory: FactualStreamState   # native P50 observed-action memory, tensor-only
    steps: torch.Tensor           # [batch] actual physical step counter
    rewards: torch.Tensor         # [batch] accumulated real scalar rewards
    terminated: torch.Tensor      # [batch], cannot step after terminal
    truncated: torch.Tensor       # [batch], max 999 steps


class NativeMountainCarFactualBridge:
    """P50 integration. Simulator physics is a separated *source boundary*.

    There is no API for hypothetical counterfactual source feedback: preview
    is delegated exclusively to the unmodified original P50 neural model.
    """
    def __init__(self, native: NativeFactualHybridStream, max_steps: int = 999):
        if not isinstance(native, NativeFactualHybridStream) or native.sensor_dim != 2:
            raise TypeError('P54_P50_TWO_SENSOR_MODEL_REQUIRED')
        if type(max_steps) is not int or not 1 <= max_steps <= 999:
            raise ValueError('P54_MAX_STEPS_INVALID')
        self.native = native
        self.max_steps = max_steps

    def start(self, seeds: tuple[int, ...]) -> FactualPhysicalSession:
        if not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError('P54_INDEPENDENT_SEEDS_REQUIRED')
        states = np.stack([published_reset(seed) for seed in seeds])
        p = next(self.native.parameters())
        sensor = torch.tensor(states, dtype=p.dtype, device=p.device)
        b = len(seeds)
        i = lambda dtype: torch.zeros((b,), dtype=dtype, device=sensor.device)
        return FactualPhysicalSession(sensor,self.native.initial_state(sensor),i(torch.int64),
                                      i(sensor.dtype),i(torch.bool),i(torch.bool))

    def _check(self, state: FactualPhysicalSession) -> None:
        if not isinstance(state, FactualPhysicalSession):
            raise ValueError('P54_PHYSICAL_SESSION_REQUIRED')
        sensor = state.physical
        self.native._check(sensor, state.memory)
        b = sensor.shape[0]
        for key, v, dtype in [('steps',state.steps,torch.int64),
                              ('rewards',state.rewards,sensor.dtype),
                              ('terminated',state.terminated,torch.bool),
                              ('truncated',state.truncated,torch.bool)]:
            if not isinstance(v,torch.Tensor) or v.shape!=(b,) or v.dtype!=dtype or v.device!=sensor.device:
                raise ValueError('P54_SESSION_'+key.upper()+'_DRIFT')
        if (not bool(torch.isfinite(state.rewards).all())
            or not bool((state.steps>=0).all())
            or not bool((state.steps<=self.max_steps).all())
            or not bool(torch.equal(state.steps,state.memory.observations))
            or not bool(torch.equal(state.truncated,state.steps>=self.max_steps))
            or not bool((sensor[:,0]>=-1.200001).all()) or not bool((sensor[:,0]<=.600001).all())
            or not bool((sensor[:,1]>=-.070001).all()) or not bool((sensor[:,1]<=.070001).all())):
            raise ValueError('P54_SESSION_FACTUAL_CUSTODY_DRIFT')

    def actual_step(self, state: FactualPhysicalSession,
                    executed_action: torch.Tensor) -> FactualPhysicalSession:
        """The only operation that advances the physical world or P50 memory."""
        self._check(state)
        if bool((state.terminated | state.truncated).any()):
            raise ValueError('P54_TERMINAL_PHYSICAL_STEP_FORBIDDEN')
        sensor = state.physical
        self.native._actions(sensor, executed_action)
        if not bool((executed_action.abs()<=1.).all()):
            raise ValueError('P54_ACTION_OUTSIDE_PHYSICAL_BOUNDS')
        # Source executes here. Hypothetical previews do not enter this route.
        results=[published_step(sensor[i].detach().cpu().numpy(),
                                float(executed_action[i,0].detach().cpu()))
                 for i in range(sensor.shape[0])]
        actual_next=torch.tensor(np.stack([x.next_sensor for x in results]),
                                 dtype=sensor.dtype,device=sensor.device)
        physical_reward=torch.tensor([x.reward for x in results],dtype=sensor.dtype,
                                     device=sensor.device)
        done=torch.tensor([x.terminated for x in results],dtype=torch.bool,device=sensor.device)
        dt=torch.ones((sensor.shape[0],1),dtype=sensor.dtype,device=sensor.device)
        factual=self.native.observe(sensor,executed_action,dt,actual_next,state.memory)
        steps=state.steps+1
        out=FactualPhysicalSession(actual_next,factual,steps,
                                   state.rewards+physical_reward,
                                   state.terminated|done,steps>=self.max_steps)
        self._check(out)
        return out

    def preview(self,state: FactualPhysicalSession,
                candidates: torch.Tensor):
        self._check(state)
        # Original P50 returns predictions and a rank certificate; it does not
        # inspect an official source or advance the physical world.
        sensor=state.physical
        self.native._actions(sensor,candidates,candidate=True)
        if not bool((candidates.abs()<=1.).all()):
            raise ValueError('P54_CANDIDATE_OUTSIDE_PHYSICAL_BOUNDS')
        dt=torch.ones((sensor.shape[0],1),dtype=sensor.dtype,device=sensor.device)
        return self.native.preview(sensor,candidates,dt,state.memory)
