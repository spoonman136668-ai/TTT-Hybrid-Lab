"""R25 factual action-specific multi-parameter dynamics and unknown-mechanism memory."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r25_multimechanism as x

torch.set_num_threads(1)


def test_frozen_source_custody_and_firstuse_guard():
    f=x.locked()
    assert f["stage"]=="PROSPECTIVE_FROZEN_BEFORE_R25_LINKED_IMPLEMENTATION_OR_OUTCOMES"
    assert f["world"]["FIT_factual_effects"]==3328
    assert f["train"]["total_updates"]==3456
    assert f["authority"]["original_external_firstuse"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_external_firstuse()


def test_4_independent_parameters_actually_change_physics():
    b=(-.3,.3);k=(-.25,.25)
    assert x.A_after(0.,0,b,k)==pytest.approx(-.3)
    assert x.A_after(0.,1,b,k)==pytest.approx(.3)
    assert x.A_after(.2,0,b,k)==pytest.approx(.4*.2-.3-.25*.2)
    assert x.A_after(.2,1,b,k)==pytest.approx(.4*.2+.3+.25*.2)
    with pytest.raises(ValueError,match="INVALID_ACTION_SPECIFIC_PHYSICS"):
        x.A_after(0.,True,b,k)
    with pytest.raises(ValueError,match="INVALID_INDEPENDENT_B"):
        x.B_after(0.,True,1)


def test_factually_complete_training_and_no_mixed_counterfactual_episodes():
    fit=x.cohort("fit")
    assert x.audit(fit,"fit")=={
        "factorial_factual_cells":256,"selected_effects":3328}
    assert len(fit)==256
    assert all(ep["action"].shape==(13,16) for ep in fit)
    assert all(ep["selected_return"] in (0,1) for ep in fit)
    assert {(ep["selected_return"],ep["z_B_evaluator_only"]) for ep in fit}=={
        (0,-1),(0,1),(1,-1),(1,1)}
    with pytest.raises(ValueError,match="UNREGISTERED_FACTUAL_FIT"):
        x.fit_episode((-.3,.3),(-.25,.25),1,2,1)


def test_swapped_action_mechanisms_share_all_B_actions_and_observations():
    for split,n in (("dev",12),("id",12),("ood",20)):
        eps=x.cohort(split)
        v=x.audit(eps,split)
        assert v["matched_twin_groups"]==16
        assert v["identical_B_pair"]
        assert v["measured_four_early_factual_probes"]
        assert v["steps_before_return"]==n
        assert all(ep["selected_return"] is None for ep in eps)
        for i in range(0,len(eps),2):
            a,b=eps[i:i+2]
            assert torch.equal(a["action"],b["action"])
            assert torch.equal(a["effect"][4:],b["effect"][4:])
            assert not torch.equal(a["effect"][:4],b["effect"][:4])


def test_explicit_four_probe_memory_sufficient_and_no_current_effect_in_input():
    for split in ("id","ood"):
        eps=x.cohort(split)
        oracle=x.analytic_controls(eps,split)
        assert oracle["mean"]["full_information_oracle"]==0.
        assert oracle["mean"]["fixed0"]>0.04
        assert oracle["mean"]["fixed1"]>0.04
        learned=x.explicit_memory_score(eps,split)
        assert learned["regret"]<1e-10
        assert learned["inputs"]=="four_original_factual_probes_only"
        for ep in eps[:4]:
            b0,b1,k0,k1=x.four_probe_predictions(ep)
            for a,actual_b,actual_k in ((0,b0,k0),(1,b1,k1)):
                for s in x.RETURNS:
                    expected=actual_b+(.4+actual_k)*s
                    assert expected==pytest.approx(
                        x.A_after(s,a,ep["b_evaluator_only"],ep["k_evaluator_only"]),
                        abs=1e-5)
        for arm in x.ARMS:
            model=x.r22.model_for(arm,985201).eval()
            erase=x.predicted(arm,model,eps[:2],True)
            assert erase.shape==(2,2,2)
            assert torch.allclose(torch.tensor(erase[0]),
                                  torch.tensor(erase[1]),atol=1e-6,rtol=0)


def test_actual_9_trainers_3456_steps_retention_and_source_neutral_receipt():
    out=x.run()
    assert out["schema"]==x.SCHEMA
    assert out["model_count"]==9
    assert out["trained_optimizer_updates"]==3456
    assert out["factual_sessions"]==256
    assert out["factual_selected_effects"]==3328
    assert out["external_original_source_effects"]==0
    assert out["F7_replay"] is False
    assert out["independent_scientific_pass"] is False
    assert out["deployment"]=="NONE"
    assert out["param_compute_matched"] is False
    assert out["conventional_convergence_independently_qualified"] is False
    assert len(out["decision"]["candidate_seeds"])==3
    assert out["decision"]["frozen_development_pass"] in (True,False)
    for arm in x.ARMS:
        assert len([z for z in out["models"] if z["arm"]==arm])==3
    for row in out["models"]:
        assert row["steps"]==384
        assert row["initial_weights"]!=row["final_weights"]
        assert row["cold_checkpoint_actions_identical"]
        assert len(row["dev_curve"])==3
        assert math.isfinite(row["training_seconds"])
        for split in ("id","ood"):
            sc=row["panels"][split]
            assert sc["actual_selected_actions"]==128
            assert sc["hypothetical_model_queries"]==256
            assert sc["contrast_opportunities"]==64
            assert len(sc["paired_group_regret"])==16
            assert math.isfinite(sc["mean_regret"])
    x.retain_ci_receipt(out)


def test_no_original_simulator_or_new_controller():
    text=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in text
