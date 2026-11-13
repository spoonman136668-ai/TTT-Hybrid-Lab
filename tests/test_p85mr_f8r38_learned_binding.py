"""F8-R38 frozen learned binding versus additive/raw/native controls."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r38_learned_binding as x

torch.set_num_threads(1)


def test_original_freeze_and_pretrain_cubic_null_amendment_fences():
    f,a=x.locked()
    assert f["schema"]==x.SCHEMA
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULT"
    assert f["parent"]["head"]=="a5bbb361387bb49c7b2fabeb820a1216c56e6ef7"
    assert a["parent"]["R38_freeze_blob"]==x.FREEZE_BLOB
    assert a["score_only_novel_law"].startswith("baseline(s)+2.7*q*s*")
    assert not any(f["authority"].values())
    assert not any(a["authority"].values())
    assert len(x.ARMS)==15
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_all_laws_keep_early_probe_uninformative_and_score_has_32_sign_pairs():
    for panel in x.PANELS:
        for state in (-.15,.15):
            for c in (0,1):
                for action in (0,1):
                    neg=x.physical(state,action,c,-1,panel)
                    pos=x.physical(state,action,c,1,panel)
                    assert abs(pos-neg)<1e-8
        worlds=x.worlds(panel)
        assert len(worlds)==64
        flips=0
        for j in range(0,64,2):
            negative,positive=worlds[j:j+2]
            assert negative["b_history"]==positive["b_history"]
            assert negative["q_evaluator_only"]==-positive["q_evaluator_only"]
            costs=[[(
                x.physical(w["return_state"],act,w["public_context"],
                           w["q_evaluator_only"],panel)-w["goal"])**2
                    for act in (0,1)] for w in (negative,positive)]
            flips+=int(costs[0].index(min(costs[0])) !=
                       costs[1].index(min(costs[1])))
        assert flips==32
        a,e=x.candidate_inputs(worlds[0],"active_information")
        assert a.shape==e.shape==(2,14 if panel=="dev" else 22,16)
        assert torch.count_nonzero(e[:,-1,:])==0
        for policy in x.POLICIES:
            assert x._probe(worlds[0],policy) in (0,1)


def test_learned_factorization_matches_additive_parameters_but_not_output():
    a,e=x.candidate_inputs(x.worlds("dev")[0],"active_information")
    factor=x.TrainableBinding(x.SEEDS[0],"factorized").eval()
    additive=x.TrainableBinding(x.SEEDS[0],"additive").eval()
    assert sum(v.numel() for v in factor.parameters())==sum(
        v.numel() for v in additive.parameters())
    for v,w in zip(factor.parameters(),additive.parameters()):
        assert torch.equal(v,w)
    with torch.no_grad():
        original=factor(a,e)
        additive_output=additive(a,e)
        assert original.shape==additive_output.shape==(2,)
        assert not torch.equal(original,additive_output)
        changed=e.clone()
        changed[:,-1,0]=.8
        assert torch.allclose(
            x.predict(factor,"binder",a,e),
            x.predict(factor,"binder",a,changed),
            atol=0,rtol=0)
        observed=x.predict(factor,"binder",a,e,erase=True)
        removed=e.clone()
        removed[:,4,:].zero_()
        assert torch.allclose(observed,
                              x.predict(factor,"binder",a,removed),
                              atol=0,rtol=0)
    assert not torch.equal(e[:,4,:],removed[:,4,:])


def test_all_factual_training_pairs_heldout_disjoint_and_donor_fence():
    fit=x.r33.fit_cohort()
    winners=x.r36.fit_pairs(fit)
    pool=x.r37.selection_pools(fit,winners)
    donors,phi=x.r37._donor_map(fit,pool)
    assert len(fit)==3072 and len(winners)==1536
    assert len(pool["informative"])==576
    assert len(pool["heldout_informative"])==192
    assert len(pool["heldout_noninformative"])==192
    assert all(i!=donors[i] for i in donors)
    assert abs(phi)<=.1
    assert not set(pool["informative"]) & set(pool["heldout_informative"])
    picked=pool["informative"][:16]
    a,e,label=x.selected_batch(fit,winners,picked)
    assert a.shape==e.shape==(32,14,16)
    assert label.shape==(16,)
    assert torch.count_nonzero(e[:,-1,:])>0
    donor_a,donor_e,donor_y=x.selected_batch(
        fit,winners,picked,donors=donors)
    assert torch.equal(a,donor_a)
    assert torch.equal(label,donor_y)
    assert all(torch.equal(donor_e[2*i,4,:],donor_e[2*i+1,4,:])
               for i in range(16))
    assert any(not torch.equal(donor_e[2*i,4,:],e[2*i,4,:])
               for i in range(16))
    with pytest.raises(ValueError,match="SHUFFLE_NONINFORMATIVE"):
        x.selected_batch(fit,winners,pool["noninformative"][:16],
                         donors=donors)


def test_native_gru_ssm_source_controls_and_last_return_effect_mask():
    a,e=x.candidate_inputs(x.worlds("dev")[0],"active_information")
    offset=-.3
    for arm in ("native","gru","ssm"):
        model=x.model_for(arm,x.SEEDS[0],offset).eval()
        with torch.no_grad():
            source=x.predict(model,arm,a,e)
            assert source.shape==(2,)
            poison=e.clone()
            poison[:,-1,0]=.8
            assert torch.allclose(source,x.predict(model,arm,a,poison),
                                  atol=0,rtol=0)
            binder_only=x.predict(model,arm,a,e,zero_substrate=True)
            substrate_only=x.predict(model,arm,a,e,zero_binding=True)
            assert torch.allclose(source,binder_only+substrate_only,
                                  atol=1e-6,rtol=1e-6)
            assert x.predict(model,arm,a,e,zero_backbone=True).shape==(2,)


def test_full_frozen_real_optimization_and_negative_controls():
    result=x.run()
    assert result["schema"]==x.SCHEMA
    assert result["model_instances"]==15
    assert result["optimizer_updates"]==7680
    assert result["fit_factual_episodes"]==3072
    assert result["training_informative_factual_pairs"]==576
    assert result["heldout_informative_factual_pairs"]==192
    assert result["heldout_noninformative_factual_pairs"]==192
    assert result["original_external_effects"]==0
    assert not result["spent_F7_replay"]
    assert not result["capacity_FLOPs_matched_across_architectures"]
    assert not result["independent_scientific_pass"]
    assert result["deployment"]=="NONE"
    assert isinstance(result["judgment"]["source_neutral_binder_development_pass"],bool)
    assert isinstance(result["judgment"]["synthetic_cubic_transfer_development_pass"],bool)
    for model in result["models"]:
        assert model["optimizer_updates"]==512
        assert model["initial_weight_sha256"]!=model["trained_weight_sha256"]
        assert model["cold_restore_actions_equal"]
        assert [v["updates"] for v in model["training_checkpoints"]]==[128,256,512]
        assert all(math.isfinite(v[panel]["choice_CE"])
                   for v in model["training_checkpoints"]
                   for panel in ("heldout_informative","heldout_noninformative"))
        for panel in x.PANELS:
            row=model["panels"][panel]
            assert set(row["policies"])==set(x.POLICIES)
            assert math.isfinite(row["erasure_penalty"])
            assert len(row["erased_active"]["choices"])==64
            for pol in x.POLICIES:
                score=row["policies"][pol]
                assert len(score["choices"])==64
                assert len(score["pergroup_regret"])==16
                assert math.isfinite(score["mean_regret"])
                assert 0<=score["sign_pair_flips"]<=32
    x.retain_ci_receipt(result)


def test_factual_supervision_excludes_generator_hidden_sign_and_external_actions():
    for fn in (x.selected_batch,x.predict,x.validation,
               x.TrainableBinding.forward,x.train_arm):
        source=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","regret_evaluator_only(",
                          "r33.r32.after(","oracle_label","hidden_sign"):
            assert forbidden not in source
    source=inspect.getsource(x)
    for forbidden in ("workflow_dispatch(","subprocess.run(","requests.get(",
                      "urlopen(","from causal_gym","import causal_gym"):
        assert forbidden not in source
