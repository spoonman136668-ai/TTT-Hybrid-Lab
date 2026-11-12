"""P113 genuine original F4 API *smoke*, using fictional P80 episodes only.

The native original DiagonalContinuousSSM is UNTRAINED here. Nothing is
downloaded, dispatched, or sourced from CartPoleWind. This test only checks
actual readout tensor shapes, target units, prequential before-write behavior,
and parity of F4 native loss with P113's fixture scalar audit.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENTS=Path(__file__).resolve().parents[1]/"experiments"
sys.path.insert(0,str(EXPERIMENTS))

from p85mr_dynamics_baselines import (
    new_baseline, prequential, factual_objective, model_metadata
)
from p80mr_causalgym_trace_adapter import _action_key, _observed_effect
from experiments import p112_f6_holdout_preflight as p112
from experiments import p113_f6_normalized_ssm_preflight as p113

torch.set_num_threads(1)


def fictional_inputs():
    train=[p112.fixture_train_row(block,index,25)
           for block,index in p112.TRAIN_IDS]
    worlds=[]
    for index in range(8,16):
        state=[.02*(index-11),.02,.02 if index%2==0 else -.02,.005]
        events=[]
        for step in range(3):
            selected=int(state[2]+.5*state[3]>0)
            after=[state[0]+(.02 if selected else -.01),
                   state[1]+(.01 if selected else -.015),
                   state[2]+(.002 if selected else -.003),
                   state[3]+(.01 if selected else -.007)]
            events.append(p113.fixture_event(step+1,selected,state,after,
                                              1.0+.1*step,
                                              terminated=(step==2)))
            state=after
        worlds.append(p113.fixture_world(index,events))
    return train,worlds


def original_f4_facts(world):
    keys=[]
    labels=[]
    for event in world["events"]:
        state=tuple(event["state_before"])
        post=tuple(event["state_after"])
        act=event["executed_action"]
        assert act==int(state[2]+.5*state[3]>0)
        keys.append(_action_key(state,act))
        labels.append(_observed_effect(
            state,post,event["reward"],event["terminated"],event["truncated"]))
        assert labels[-1]==pytest.approx(p113.label_from_fact(
            state,post,event["reward"],event["terminated"],event["truncated"]),
            rel=0,abs=1e-15)
    return {
        "action":torch.tensor([keys],dtype=torch.float32),
        "effect":torch.tensor([labels],dtype=torch.float32),
    }


def test_native_original_ssm_f4_prequential_matches_p113_full_fixture_metric():
    training,worlds=fictional_inputs()
    ids=p113.validate_complete_fixture(training,worlds)
    assert ids["event_count"]==24 and ids["scored_count"]==16
    models={}
    native_expected={}
    for seed in p113.SEEDS:
        native=new_baseline("ssm256",seed).eval()
        info=model_metadata(native)
        assert info["kind"]=="ssm256"
        assert info["persistent_floats_per_world"]==256
        assert info["external_science_qualified"] is False
        forecasts=[]
        loss_sum=0.
        with torch.no_grad():
            for world in worlds:
                facts=original_f4_facts(world)
                prediction=prequential(native,facts)
                assert tuple(prediction.shape)==(1,3,16)
                assert bool(torch.isfinite(prediction).all())
                # Original F4 loss excludes event 0. The nine forecast pads
                # are unconstrained and must NOT count toward the loss.
                loss_sum+=2*float(factual_objective(native,facts)["loss"])
                forecasts.append(prediction[0].tolist())
        native_expected[seed]=loss_sum/16.
        ckpt=b"p113-native-untrained-fixture-ONLY:"+str(seed).encode()
        manifest=p113.fixture_manifest(seed,ckpt,training,worlds,forecasts)
        models[seed]={"manifest":manifest,"bytes":ckpt,
                      "predictions":forecasts}
    report=p113.score_p80_f4_fixture(training,worlds,models)
    assert report["prequential_scored_events_after_first_fixture"]==16
    assert report["native_SSM_readout_authenticated"] is False
    assert report["original_F4_trained_weights_loaded"] is False
    assert report["scientific_pass"] is False
    for seed in p113.SEEDS:
        got=report["all_seed_seven_channel_metrics_fixture"][seed][
            "f4_weighted_objective_fixture"]
        assert math.isclose(got,native_expected[seed],rel_tol=2e-4,abs_tol=5e-7)


def test_native_p69_current_outcome_cannot_change_its_own_prequential_prediction():
    _,worlds=fictional_inputs()
    facts=original_f4_facts(worlds[0])
    modified={k:v.clone() for k,v in facts.items()}
    modified["effect"][0,1,2]+=.6
    modified["effect"][0,2,4]-=.4
    native=new_baseline("ssm256",p113.SEEDS[0]).eval()
    with torch.no_grad():
        before=prequential(native,facts)
        after=prequential(native,modified)
    assert torch.equal(before[:,:2,:],after[:,:2,:])
    assert tuple(before.shape)==(1,3,16)
    # This is an initialized native-model readout smoke on fictional effects;
    # it is not a trained-model result, source-firstuse, or scientific pass.


def test_p113_padding_predictions_do_not_change_seven_channel_native_loss():
    training,worlds=fictional_inputs()
    all_models={}
    for seed in p113.SEEDS:
        native=new_baseline("ssm256",seed).eval()
        data=[]
        with torch.no_grad():
            for w in worlds:
                data.append(prequential(native,original_f4_facts(w))[0].tolist())
        ckpt=("native-untrained-fixture:"+str(seed)).encode()
        all_models[seed]={"bytes":ckpt,"predictions":data,
                          "manifest":p113.fixture_manifest(
                              seed,ckpt,training,worlds,data)}
    original=p113.score_p80_f4_fixture(training,worlds,all_models)
    # Modify only the nine forecast padding channels, then re-seal fixture.
    changed={}
    for seed,e in all_models.items():
        predictions=[[row[:7]+[1234.5]*9 for row in world]
                     for world in e["predictions"]]
        changed[seed]={
            "bytes":e["bytes"],
            "predictions":predictions,
            "manifest":p113.fixture_manifest(
                seed,e["bytes"],training,worlds,predictions),
        }
    updated=p113.score_p80_f4_fixture(training,worlds,changed)
    assert original["all_seed_seven_channel_metrics_fixture"]==(
        updated["all_seed_seven_channel_metrics_fixture"])
    assert original["selected_seed_fixture_only"]==updated["selected_seed_fixture_only"]
    assert updated["original_F6_actions_executed"]==0
    assert updated["real_F6_training_executed"] is False
    assert updated["real_CAL_admitted"] is False
