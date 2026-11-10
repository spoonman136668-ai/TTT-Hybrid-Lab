"""P57: exact isolated method from pinned upstream Gymnasium source.

Official upstream Git blob 96dbb0d31a17d144468562f4bc68fce902e2a69f
at Farama-Foundation/Gymnasium@e6be96abfb0a9df36da5e7884522c4f1f2432e53.
The step body is byte-exact SHA256 405213fb980f17ab3a8593cd16504bd5d3c6fae5eb7b88a6c375fe5416d03b04
from upstream. Tested via a minimal isolated class with pinned constructor fields;
this is NOT executing the installed official Gymnasium package or wrappers.
Original upstream code attributed to Gymnasium contributors, MIT licensed.
"""
from __future__ import annotations
import math
import inspect
import hashlib
import numpy as np
import pytest
from native_p54_mountaincar_factual_bridge import published_step

class UpstreamStepIsolated:
    def __init__(self,state):
        self.min_action=-1.0; self.max_action=1.0
        self.min_position=-1.2; self.max_position=0.6
        self.max_speed=0.07;self.goal_position=0.45
        self.goal_velocity=0;self.power=0.0015
        self.render_mode=None
        self.state=np.array(state,dtype=np.float32)

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self.state, reward, terminated, False, {}


PINNED_STEP_SHA256 = '405213fb980f17ab3a8593cd16504bd5d3c6fae5eb7b88a6c375fe5416d03b04'
MAX_ALLOWED_FLOAT_ABS = 5e-7  # exploratory engineering tolerance, not preregistered science


def check(state, action):
    original=UpstreamStepIsolated(state).step(np.array([action],dtype=np.float32))
    candidate=published_step(np.asarray(state,dtype=np.float32),float(np.float32(action)))
    distance=float(np.abs(original[0].astype(np.float64)-candidate.next_sensor.astype(np.float64)).max())
    assert original[1] == candidate.reward
    assert original[2] == candidate.terminated
    assert original[3] is False and original[4] == {}
    assert distance <= MAX_ALLOWED_FLOAT_ABS
    return distance, np.array_equal(original[0],candidate.next_sensor)


def test_pinned_upstream_step_method_exact_cryptographic_source_fragment():
    actual=inspect.getsource(UpstreamStepIsolated.step).encode('utf8')
    assert hashlib.sha256(actual).hexdigest() == PINNED_STEP_SHA256
    assert b'force = min(max(action[0], self.min_action), self.max_action)' in actual
    assert b'reward -= math.pow(action[0], 2) * 0.1' in actual


def test_twenty_thousand_source_state_motor_pairs_no_reward_or_termination_disagreement():
    rng=np.random.default_rng(576201)
    states=rng.uniform((-1.2,-.07),(.6,.07),(20000,2)).astype(np.float32)
    actions=rng.uniform(-1,1,20000).astype(np.float32)
    distances=[];identical=0
    for state,action in zip(states,actions):
        d,exact=check(state,action)
        distances.append(d);identical+=int(exact)
    assert max(distances) <= MAX_ALLOWED_FLOAT_ABS
    assert identical >= 15000  # informational, robust to small float32 operation differences


def test_edge_states_neutral_positive_negative_and_clipped_raw_action_reward():
    for state in [[-1.2,-.01],[-1.2,0],[.6,.07],[.44,.02],[-.5,0],[.6,-.07]]:
        for action in [-1.3,-1.,0.,1.,1.3]:
            check(np.array(state,np.float32),np.float32(action))
    # Upstream clips the physical *force*, but the penalty uses the raw action.
    upstream=UpstreamStepIsolated(np.array([-.5,0],dtype=np.float32))
    _,reward,_,_,_=upstream.step(np.array([1.3],dtype=np.float32))
    assert reward == pytest.approx(-.169,abs=1e-7)


def test_256_multistep_64_step_actual_rollouts_without_divergent_reward_or_termination():
    rng=np.random.default_rng(576201)
    max_error=0.0; count=0
    for episode in range(256):
        sensor=np.array([rng.uniform(-.6,-.4),0.],dtype=np.float32)
        upstream=UpstreamStepIsolated(sensor)
        for step in range(64):
            action=float(rng.choice([-1.,0.,1.]))
            original=upstream.step(np.asarray([action],dtype=np.float32))
            reference=published_step(sensor,action)
            sensor=reference.next_sensor
            max_error=max(max_error,float(np.max(np.abs(original[0].astype(np.float64)-sensor.astype(np.float64)))))
            assert original[1] == reference.reward
            assert original[2] == reference.terminated
            count+=1
            if reference.terminated:break
    assert count==16384
    assert max_error<=MAX_ALLOWED_FLOAT_ABS


def test_pinned_source_identity_is_not_conflated_with_full_official_runtime():
    # This method-level test gives no evidence of import/version/wrapper parity.
    assert not ('gymnasium' in UpstreamStepIsolated.__module__)
    assert PINNED_STEP_SHA256 != '96dbb0d31a17d144468562f4bc68fce902e2a69f'
