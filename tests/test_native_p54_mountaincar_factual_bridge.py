"""P54 engineering controls: published physics reference and real-step custody.

Only optional upstream Gymnasium parity executes official code, and ONLY when
installed files match the frozen original source blob; skip is not a parity PASS.
"""
from __future__ import annotations

import io
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from model import Configuration
from native_factual_hybrid_stream import FactualStreamState,NativeFactualHybridStream
from native_p54_mountaincar_factual_bridge import (
    UPSTREAM_COMMIT,UPSTREAM_GIT_BLOB,UPSTREAM_PATH,ENV_ID,
    NativeMountainCarFactualBridge, FactualPhysicalSession,
    published_step, published_reset, source_git_blob, verify_pinned_official_source)


torch.set_num_threads(1)


def fixture(b=2,limit=8):
    torch.manual_seed(5401)
    model=NativeFactualHybridStream(2,capacity=8,cfg=Configuration(width=16,heads=2)).eval()
    bridge=NativeMountainCarFactualBridge(model,max_steps=limit)
    session=bridge.start(tuple(54000+i for i in range(b)))
    return bridge,session


def choices(b=2):
    return torch.tensor([[[-1.],[1.]]],dtype=torch.float32).expand(b,-1,-1)


def physical_snapshot(state):
    return [state.physical.clone(),state.steps.clone(),state.rewards.clone(),
            state.terminated.clone(),state.truncated.clone()] + [
             getattr(state.memory,n).clone() for n in FactualStreamState.__dataclass_fields__]


def test_pinned_external_source_identity_and_wrong_bytes_fail_closed(tmp_path):
    assert UPSTREAM_COMMIT=='e6be96abfb0a9df36da5e7884522c4f1f2432e53'
    assert UPSTREAM_GIT_BLOB=='96dbb0d31a17d144468562f4bc68fce902e2a69f'
    assert UPSTREAM_PATH=='gymnasium/envs/classic_control/continuous_mountain_car.py'
    assert ENV_ID=='MountainCarContinuous-v0'
    path=tmp_path/'continuous_mountain_car.py';path.write_text('fake source')
    assert source_git_blob(path.read_bytes())!=UPSTREAM_GIT_BLOB
    with pytest.raises(ValueError,match='P54_UPSTREAM_GYMNASIUM_BLOB_MISMATCH'):
        verify_pinned_official_source(path)


def test_published_seeded_reset_follows_pcg64_reference():
    np.testing.assert_array_equal(published_reset(123,low=-.7,high=-.5),
                                  np.array([-.5635296,0],dtype=np.float32))
    assert np.array_equal(published_reset(99),published_reset(99))
    assert not np.array_equal(published_reset(99),published_reset(100))
    with pytest.raises(ValueError,match='P54_INVALID_RESET_SOURCE'):
        published_reset(-1)


def test_published_transition_closed_form_on_ordinary_state():
    s=np.array([-.5,.01],dtype=np.float32)
    x=published_step(s,+1.)
    want_velocity=.01+.0015-.0025*math.cos(-1.5)
    expected=np.array([-.5+want_velocity,want_velocity],dtype=np.float32)
    np.testing.assert_allclose(x.next_sensor,expected,rtol=0,atol=1e-8)
    assert x.reward==pytest.approx(-.1)
    assert not x.terminated
    np.testing.assert_array_equal(s,np.array([-.5,.01],dtype=np.float32))


def test_collision_limit_velocity_clamp_goal_and_true_reward():
    wall=published_step(np.array([-1.2,-.01],dtype=np.float32),-1.)
    assert wall.next_sensor.tolist()==pytest.approx([-1.2,0],abs=1e-7)
    vmax=published_step(np.array([-.6,.07],dtype=np.float32),+1.)
    assert vmax.next_sensor[1]<=.07000001
    vmin=published_step(np.array([-.6,-.07],dtype=np.float32),-1.)
    assert vmin.next_sensor[1]>=-.07000001
    won=published_step(np.array([.44,.02],dtype=np.float32),1.)
    assert won.terminated and won.reward==pytest.approx(99.9)
    neutral=published_step(np.array([-.5,0],dtype=np.float32),0.)
    assert neutral.reward==0.
    clipped=published_step(np.array([-.5,0],dtype=np.float32),2.)
    right=published_step(np.array([-.5,0],dtype=np.float32),1.)
    np.testing.assert_array_equal(clipped.next_sensor,right.next_sensor)
    assert clipped.reward==pytest.approx(-.4)  # original reward uses raw action


def test_step_only_after_real_source_outcome_and_read_only_candidate_preview():
    bridge,state=fixture()
    prev=physical_snapshot(state)
    original_weights={k:v.detach().clone() for k,v in bridge.native.state_dict().items()}
    candidate=choices()
    for _ in range(2):bridge.preview(state,candidate)
    assert all(torch.equal(a,b) for a,b in zip(prev,physical_snapshot(state)))
    assert all(torch.equal(v,original_weights[k]) for k,v in bridge.native.state_dict().items())
    result=bridge.actual_step(state,torch.ones((2,1)))
    assert result.steps.tolist()==[1,1] and result.memory.observations.tolist()==[1,1]
    assert torch.equal(state.steps,torch.zeros_like(state.steps))
    np.testing.assert_array_equal(result.physical[0].numpy(),
                                  published_step(state.physical[0].numpy(),1.).next_sensor)


def test_constant_actions_unidentifiable_but_alternating_completed_actions_rank_ready():
    bridge,baseline=fixture(limit=9)
    repeated=baseline
    for _ in range(8):repeated=bridge.actual_step(repeated,torch.ones((2,1)))
    assert not bridge.preview(repeated,choices()).identified.any()
    alternating=baseline
    for action in (1,-1,1,-1,1,-1,1,-1):
        alternating=bridge.actual_step(alternating,torch.full((2,1),float(action)))
    assert bridge.preview(alternating,choices()).identified.tolist()==[True,True]
    assert alternating.memory.observations.tolist()==[8,8]


def test_bounded_8_step_episode_and_terminal_protection():
    bridge,state=fixture(limit=8)
    for a in (1,-1,1,-1,1,-1,1,-1):
        state=bridge.actual_step(state,torch.full((2,1),float(a)))
    assert state.truncated.tolist()==[True,True]
    original=physical_snapshot(state)
    with pytest.raises(ValueError,match='P54_TERMINAL_PHYSICAL_STEP_FORBIDDEN'):
        bridge.actual_step(state,torch.ones((2,1)))
    assert all(torch.equal(x,y) for x,y in zip(original,physical_snapshot(state)))


def test_batch_isolation_from_other_physical_trajectory():
    bridge,state=fixture()
    first=bridge.actual_step(state,torch.tensor([[1.],[-1.]]))
    assert not torch.equal(first.physical[0],first.physical[1])
    assert first.memory.actions[0,0,0].item()==1
    assert first.memory.actions[1,0,0].item()==-1
    assert first.rewards.tolist()==pytest.approx([-.1,-.1],abs=1e-7)


def test_invalid_actions_nonfinite_source_and_state_custody_are_atomic():
    bridge,state=fixture()
    original=physical_snapshot(state)
    for bad,pattern in [(torch.full((2,1),1.01),'P54_ACTION_OUTSIDE_PHYSICAL_BOUNDS'),
                        (torch.tensor([[1.],[float('nan')]]),'P50_INVALID_EXECUTED_ACTION')]:
        with pytest.raises(ValueError,match=pattern):bridge.actual_step(state,bad)
    with pytest.raises(ValueError,match='P54_SESSION_FACTUAL_CUSTODY_DRIFT'):
        bad=FactualPhysicalSession(state.physical,state.memory,state.steps+1,
                                   state.rewards,state.terminated,state.truncated)
        bridge.preview(bad,choices())
    assert all(torch.equal(x,y) for x,y in zip(original,physical_snapshot(state)))


def test_tensor_only_cold_restore_bit_equal_physical_and_model_preview():
    bridge,state=fixture()
    for a in (1,-1,1,-1):state=bridge.actual_step(state,torch.full((2,1),float(a)))
    c=choices()
    pred=bridge.preview(state,c)
    checkpoint={'model':bridge.native.state_dict(),
          'physical':state.physical,'steps':state.steps,'rewards':state.rewards,
          'terminated':state.terminated,'truncated':state.truncated,
          'memory':{k:getattr(state.memory,k) for k in FactualStreamState.__dataclass_fields__}}
    stream=io.BytesIO();torch.save(checkpoint,stream);stream.seek(0)
    restored=torch.load(stream,weights_only=True)
    new=NativeFactualHybridStream(2,capacity=8,cfg=Configuration(width=16,heads=2)).eval()
    new.load_state_dict(restored['model'])
    other=NativeMountainCarFactualBridge(new,max_steps=8)
    cold=FactualPhysicalSession(restored['physical'],FactualStreamState(**restored['memory']),
        restored['steps'],restored['rewards'],restored['terminated'],restored['truncated'])
    after=other.preview(cold,c)
    assert torch.equal(pred.predictions,after.predictions)
    assert torch.equal(pred.identified,after.identified)
    assert all(torch.equal(a,b) for a,b in zip(physical_snapshot(state),physical_snapshot(cold)))


def test_native_p50_neutral_action_is_valid_real_physical_fact():
    bridge,state=fixture()
    assert published_step(state.physical[0].numpy(),0.).reward==0
    observed=bridge.actual_step(state,torch.zeros((2,1)))
    assert observed.steps.tolist()==[1,1]
    assert observed.memory.observations.tolist()==[1,1]
    assert observed.memory.actions[:,0,0].tolist()==[0.,0.]
    assert observed.rewards.tolist()==[0.,0.]
    assert not bridge.preview(observed,torch.zeros((2,1,1))).identified.any()


def test_official_gymnasium_runtime_when_present_and_exact_source_only():
    gym=pytest.importorskip('gymnasium',reason='OFFICIAL_RUNTIME_NOT_INSTALLED_UNVERIFIED')
    from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
    verify_pinned_official_source(inspect.getsourcefile(Continuous_MountainCarEnv))
    for seed in (54001,54002):
        env=Continuous_MountainCarEnv();ob,_=env.reset(seed=seed)
        np.testing.assert_array_equal(ob,published_reset(seed))
        for a in (1.,-1.,.2,-.75,0.,.5):
            expected=published_step(ob,a)
            env.state=ob.copy()
            ob,reward,done,truncated,_=env.step(np.array([a],dtype=np.float32))
            np.testing.assert_allclose(ob,expected.next_sensor,rtol=0,atol=1e-6)
            assert reward==pytest.approx(expected.reward,abs=1e-6)
            assert done==expected.terminated and not truncated
        env.close()
