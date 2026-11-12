"""R24 real gradient interference on novel B law (source neutral)."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r24_online_interference as x

torch.set_num_threads(1)


def test_frozen_online_learning_protocol_and_source_denials():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["models"]["actual_B_update_steps"]==432
    assert f["world"]["A_score_episodes_before_first_return"]==96
    assert f["authority"]["original_external_source_firstuse"] is False
    assert f["score"]["scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_external_scientific_firstuse()


def test_two_interacting_hidden_B_factors_and_causal_factual_support():
    tr=x.cohort("B_train")
    ho=x.cohort("B_holdout")
    score=x.cohort("A_score")
    a=x.audit(tr,ho,score)
    assert a["B_train_factual_episodes"]==64
    assert a["B_holdout_factual_episodes"]==32
    assert a["A_score_matched_A_pairs"]==48
    assert {p["z_b"] for p in tr}=={-1,1}
    assert {p["z_c"] for p in tr}=={-1,1}
    assert len({p["group"] for p in tr})==16
    assert len({p["group"] for p in ho})==8
    assert len({p["group"] for p in score})==12
    assert set(x.factual(score[0]))=={"action","effect"}
    for idx in range(0,len(score),2):
        a,b=score[idx:idx+2]
        assert torch.equal(a["action"],b["action"])
        assert torch.equal(a["effect"][1:],b["effect"][1:])
        assert not torch.equal(a["effect"][0],b["effect"][0])
        assert a["effect"].shape==(17,16)
    with pytest.raises(ValueError,match="INVALID_TWO_FACTOR_LAW"):
        x.b_law(0.,True,1,1)
    with pytest.raises(ValueError,match="INVALID_WORLD"):
        x.episode("A_score",12,1,1,-1)
    assert x.b_law(0.,1,1,1)!=x.b_law(0.,1,1,-1)
    assert x.b_law(.7,1,1,1)!=x.b_law(.7,1,-1,1)


def test_original_checkpoint_and_no_probe_observation_negative():
    pair=x.cohort("A_score")[:2]
    for arm in x.ARMS:
        model=x.r22.model_for(arm,985201).cpu().eval()
        erased=x._predicted_return(arm,model,pair,erase_early=True)
        assert erased.shape==(2,2)
        assert torch.allclose(torch.tensor(erased[0]),
                              torch.tensor(erased[1]),atol=1e-6,rtol=0)


def test_qualified_factual_A_memory_and_frozen_no_history_bound():
    fit=x.r22.cohort("fit")
    ep=x.cohort("A_score")
    ctrl=x._fixed_A_controls(ep,fit)
    assert min(ctrl["fixed"].values())==pytest.approx(.015,abs=1e-12)
    assert ctrl["FIT_only_early_A_ridge_mean_regret"]<.001


def test_actual_nine_models_pretrain_and_432_online_B_gradient_updates():
    r=x.run()
    assert r["actual_model_count"]==9
    assert r["original_pretrain_update_count"]==3456
    assert r["actual_online_B_gradient_update_count"]==432
    assert r["new_B_train_factual_selected_effects"]==1024
    assert r["new_B_holdout_factual_selected_effects"]==512
    assert r["new_score_factual_A_and_B_effects"]==1632
    assert r["original_external_effects"]==0
    assert not r["F7_replayed"]
    assert not r["independent_scientific_pass"]
    assert not r["independent_conventional_convergence_qualified"]
    assert r["deployment"]=="NONE"
    assert r["decision"]["frozen_DEV_gate_pass"] in (True,False)
    assert len(r["decision"]["native_anchor_seed_gates"])==3
    assert {(z["arm"],z["seed"]) for z in r["models"]}=={
        (a,s) for a in x.ARMS for s in x.SEEDS}
    for row in r["models"]:
        assert row["initial_sha"]!=row["A_pretrained_sha"]
        assert row["parameters"]>0
        assert math.isfinite(row["original_DEV_A_mean_regret"])
        assert set(row["conditions"])==set(x.CONDITIONS)
        assert row["conditions"]["no_update"]["updates"]==0
        for cond in ("plain_B_update","anchored_B_update"):
            d=row["conditions"][cond]
            assert d["updates"]==24
            assert d["adapted_sha"]!=d["pretrain_sha"]
            assert d["pretrain_sha"]==row["A_pretrained_sha"]
        for c in x.CONDITIONS:
            d=row["conditions"][c]
            assert d["cold_restore_exact_A_action_parity"]
            assert math.isfinite(d["B_heldout_mse"])
            assert len(d["A_score"]["pergroup_regret"])==12
            assert d["A_score"]["synthetic_first_return_actions"]==192
            assert d["A_score"]["A_sign_contrast_opportunities"]==96
    x.retain_ci_receipt(r)


def test_no_original_source_or_runner_execution():
    source=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in source
