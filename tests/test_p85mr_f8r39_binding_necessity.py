"""F8-R39 preregistered source-neutral postfit binder mechanism/transfer."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r39_binding_necessity as x

torch.set_num_threads(1)


def test_frozen_preregistration_source_and_external_science_fence():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_R39_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["parent"]["head"]=="99a2fa617ece5529534b977f198f4e7d7d9c69f4"
    assert f["frozen_judgment"]["mechanism_product_necessary_seeds_min"]==2
    assert f["frozen_judgment"]["transfer_both_new_laws_seed_success_min"]==2
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_two_prospective_laws_have_early_ambiguous_q_and_discriminating_return():
    f=x.locked()
    for panel in x.PANELS:
        for s in x.r33.r32.EARLY:
            state,action=s
            for c in (0,1):
                assert x._generator_law(state,action,c,-1,panel)==x._generator_law(
                    state,action,c,1,panel)
        ww=x.worlds(panel)
        assert len(ww)==64
        assert len(set(w["group"] for w in ww))==16
        assert len(set(w["b_history"] for w in ww))==16
        flips=0
        margin=float("inf")
        for i in range(0,len(ww),2):
            negative,positive=ww[i:i+2]
            assert negative["b_history"]==positive["b_history"]
            assert negative["public_context"]==positive["public_context"]
            assert negative["q_evaluator_only"]==-positive["q_evaluator_only"]
            costs=[]
            for w in (negative,positive):
                actions=[(x._generator_law(w["return_state"],act,
                        w["public_context"],w["q_evaluator_only"],panel)-1.)**2
                        for act in (0,1)]
                margin=min(margin,abs(actions[0]-actions[1]))
                costs.append(actions)
            flips+=int(costs[0].index(min(costs[0]))!=
                       costs[1].index(min(costs[1])))
        assert flips==32 and margin>.4


def test_preaction_fact_only_and_zero_product_channel_does_not_change_parameters():
    w=x.worlds("oscillatory_even")[0]
    a,e=x.candidate_inputs(w)
    assert a.shape==e.shape==(2,14,16)
    assert torch.count_nonzero(e[:,-1,:])==0
    model=x.r38.TrainableBinding(x.SEEDS[0]).eval()
    original=[p.clone() for p in model.parameters()]
    with torch.no_grad():
        pred=x._prediction(model,a,e)
        inherited=x.r38.predict(model,"binder",a,e)
        assert torch.allclose(pred,inherited,atol=1e-6,rtol=1e-6)
        ablated=x._prediction(model,a,e,zero_product=True)
        assert ablated.shape==pred.shape==(2,)
        # Perturb physically unavailable current return while leaving factual probe.
        poison=e.clone()
        poison[:,-1,0]=.8
        assert torch.allclose(pred,x._prediction(model,a,poison),atol=0,rtol=0)
    assert all(torch.equal(before,after) for before,after in
               zip(original,model.parameters()))
    b,eb=x.candidate_inputs(x.worlds("nonlinear_odd")[0])
    assert b.shape==eb.shape==(2,22,16)


def test_exact_R38_replay_and_full_frozen_necessity_transfer_receipt():
    result=x.run()
    assert result["schema"]==x.SCHEMA
    assert result["freeze_blob"]==x.FREEZE_BLOB
    assert result["original_R38_training_replays"]==3
    assert result["replayed_original_optimizer_updates"]==1536
    assert result["new_law_training_updates"]==0
    assert result["evaluated_new_law_worlds_per_panel"]==64
    assert result["original_external_effects"]==0
    assert result["spent_F7_replay"] is False
    assert result["independent_scientific_pass"] is False
    assert result["deployment"]=="NONE"
    j=result["judgment"]
    assert isinstance(j["product_channel_necessity_development_pass"],bool)
    assert isinstance(j["two_new_synthetic_laws_development_pass"],bool)
    assert len(result["models"])==3
    for record in result["models"]:
        assert record["optimizer_updates"]==512
        assert record["initial_weight_hash"]!=record["trained_weight_hash"]
        assert record["replayed_R38_decision_fingerprints_match"]
        assert record["cold_restored_action_identity"]
        for panel in x.ORIGINAL_PANELS:
            row=record["original"][panel]
            for name in ("normal","zero_product","erased"):
                values=row[name]
                assert len(values["choices"])==64
                assert math.isfinite(values["mean_regret"])
                assert 0<=values["sign_pair_flips"]<=32
            assert row["normal"]["action_vector_sha256"]==x.REFERENCE_CHOICES[
                record["seed"]]
        for panel in x.PANELS:
            row=record["new_laws"][panel]
            for name in ("normal","zero_product","erased"):
                values=row[name]
                assert len(values["choices"])==64
                assert len(values["pergroup_regret"])==16
                assert math.isfinite(values["mean_regret"])
                assert 0<=values["sign_pair_flips"]<=32
    x.retain_ci_receipt(result)


def test_factual_fit_and_neural_prediction_never_read_generator_hidden_sign():
    for fn in (x._prediction,x.replay_original_seed):
        source=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","_generator_law(",
                          "regret_evaluator_only(",
                          "r33.r32.after("):
            assert forbidden not in source
    # The original-score helper intentionally calls the R38 evaluator at
    # *scoring* time, so it is not a model/supervision leakage.
    source=inspect.getsource(x)
    for forbidden in ("workflow_dispatch(","subprocess.run(","requests.get(",
                      "urlopen(","from causal_gym","import causal_gym"):
        assert forbidden not in source
