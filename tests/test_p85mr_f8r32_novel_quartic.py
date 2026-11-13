"""R32: new quartic-law neural vs factual-control, strict source-neutral fence."""
import inspect
import math
import torch
import pytest
from experiments import p85mr_f8r32_novel_quartic as x

torch.set_num_threads(1)


def test_frozen_protocol_and_prohibited_external_scientific_firstuse():
    f=x.locked()
    assert f["parent"]["head"]=="784f068d9db28778556101c8471c66d589e65a6e"
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["score"]["development_native_win_both_panels_at_least"]==2
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_new_quartic_identifiability_and_action_consequence_barriers():
    assert x.after(.15,0,0,+1)==x.after(.15,0,0,-1)
    assert x.after(-.15,1,1,+1)==x.after(-.15,1,1,-1)
    assert x.after(.42,0,0,+1)!=x.after(.42,0,0,-1)
    assert x.after(.42,1,0,+1)==x.after(.42,1,0,-1)
    assert x.after(.78,0,0,+1)!=x.after(.78,0,0,-1)
    for panel in x.PANELS:
        worlds=x.worlds(panel)
        assert len(worlds)==48
        for g in range(12):
            for c in (0,1):
                lo=worlds[4*g+2*c]
                hi=worlds[4*g+2*c+1]
                assert lo["b_history"]==hi["b_history"]
                assert lo["q_evaluator_only"]==-hi["q_evaluator_only"]
                assert x.evaluator_regret(lo,c) != x.evaluator_regret(hi,c)
    with pytest.raises(ValueError,match="INVALID_PHYSICS"):
        x.after(.3,2,0,1.)


def test_new_factual_FIT_and_preaction_eraser_no_oracle_labels():
    fit=x.fit_cohort()
    assert len(fit)==3072
    assert set(fit[0])=={"action","effect","physical_target","direct_target"}
    assert fit[0]["action"].shape==(14,16)
    assert fit[0]["effect"].shape==(14,16)
    lo=x._fit_episode(0,0,-1,1.,0,0)
    hi=x._fit_episode(0,0,+1,1.,0,0)
    assert torch.equal(lo["effect"][:4],hi["effect"][:4])
    assert not torch.equal(lo["effect"][4],hi["effect"][4])
    assert torch.equal(lo["effect"][5:13],hi["effect"][5:13])
    ww=x.worlds("dev")[0]
    a,e=x.candidate_inputs(ww,"active_information")
    ae,ee=x.candidate_inputs(ww,"active_information",erase=True)
    assert a.shape==(2,14,16)
    assert torch.equal(a,ae)
    assert torch.count_nonzero(ee[:,4])==0
    assert torch.count_nonzero(ee[:,-1])==0
    assert torch.equal(e[:,:4],ee[:,:4])
    assert torch.equal(e[:,5:],ee[:,5:])


def test_factual_baseline_only_training_and_fit_validation():
    fit=x.fit_cohort()
    model=x.fit_factual_control(fit)
    assert model["fit_examples"]==2304 and model["validation_examples"]==768
    assert model["feature_count"]==7
    assert 0<=model["normal_gradient_max"]<1e-5
    assert math.isfinite(model["validation_mse"])
    assert model["variance"]["uncertain"]>model["variance"]["safe"]
    assert model["variance"]["uncertain"]>model["variance"]["informed"]
    for panel in x.PANELS:
        result=x.score_policy("factual_control",model,panel,"active_information")
        assert len(result["choices"])==48
        assert math.isfinite(result["regret"])


def test_actual_neural_training_all_arms_interference_and_cold_restore():
    data=x.run()
    assert data["optimizer_updates"]==1536
    assert data["neural_model_instances"]==6
    assert data["factual_fit_episodes"]==3072
    assert data["train_factual_episodes"]==2304
    assert data["validation_factual_episodes"]==768
    assert data["all_new_synthetic_score_worlds"]==96
    assert data["original_external_effects"]==0
    assert not data["spent_F7_replay"] and data["deployment"]=="NONE"
    assert not data["capacity_FLOPs_matched"]
    assert not data["conventional_independent_convergence_qualified"]
    assert not data["independent_scientific_pass"]
    assert isinstance(data["judgement"]["source_neutral_native_development_pass"],bool)
    for model in data["models"]:
        assert model["optimizer_updates"]==256
        assert model["initial_hash"]!=model["trained_hash"]
        assert model["cold_restore_actions_equal"]
        assert [k["updates"] for k in model["validation_trace"]]==[128,256]
        for panel in x.PANELS:
            scores=model["panels"][panel]
            assert set(scores["policies"])==set(x.POLICIES)
            assert math.isfinite(scores["erasure_regret_delta"])
            for pol in x.POLICIES:
                sc=scores["policies"][pol]
                assert len(sc["choices"])==48
                assert len(sc["pergroup"])==12
                assert math.isfinite(sc["regret"])
    x.retain_ci_receipt(data)


def test_no_hidden_oracle_training_and_no_external_execution_paths():
    for fn in (x._fit_episode,x._ridge_features,x.fit_factual_control,
               x._choices,x.train_arm):
        src=inspect.getsource(fn)
        for banned in ("q_evaluator_only","evaluator_regret(",
                       "r29.actual_regret(","oracle_label"):
            assert banned not in src
    src=inspect.getsource(x)
    for banned in ("import causal_gym","from causal_gym","workflow_dispatch(",
                   "subprocess.run(","requests.get(","urlopen("):
        assert banned not in src
