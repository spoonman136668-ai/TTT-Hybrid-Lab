"""R27 source-neutral active intervention vs same-budget fixed and random probes."""
import inspect
import math
import pytest
import torch

from experiments import p85mr_f8r27_active_interventions as x
torch.set_num_threads(1)


def test_precommitted_source_owner_deny_and_no_future_data():
    f=x.locked()
    assert f["stage"]=="FROZEN_BEFORE_LINKED_R27_CODE_OR_RESULTS"
    assert f["training"]["total_optimizer_updates"]==1152
    assert "p85mr_f8r27_active_intervention_dev.json" in inspect.getsource(x.retain_ci_receipt)
    assert f["authority"]["original_external_firstuse"] is False
    assert f["scoring"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_scientific_source()


def test_two_independent_nonlinear_curvatures_are_unobservable_in_four_probes():
    b=(-.3,.1);k=(-.25,.25)
    p=(-.55,.55);other=(.55,-.55)
    for s,act in x.PROBES:
        assert x.after_A(s,act,b,k,p)==pytest.approx(
            x.after_A(s,act,b,k,other),abs=1e-12)
    assert x.after_A(.7,0,b,k,p)!=x.after_A(.7,0,b,k,other)
    assert x.after_A(.7,1,b,k,p)!=x.after_A(.7,1,b,k,other)
    with pytest.raises(ValueError,match="INVALID_ACTION_SPECIFIC_PHYSICS"):
        x.after_A(.2,True,b,k,p)


def test_matched_B_worlds_and_factual_fifth_probe_is_action_specific():
    for split in x.B_STEPS:
        ws=x.worlds(split)
        assert x.audit(ws,split)["matched_B_twin_groups"]==12
        assert all(w["B"]==ws[2*(w["group"]) ]["B"] for w in ws)
        for policy in x.POLICIES:
            plans=x._plans(ws,policy)
            assert len(plans)==96
            assert all(len(p["history"])==5+x.B_STEPS[split] for p in plans)
            assert all(p["fifth_action"] in (0,1) for p in plans)
            for p in plans[:4]:
                base=x.factual_b_and_slope(p["w"])
                assert all(math.isfinite(z) for z in base[0]+base[1])
                assert p["fifth"][1]==p["history"][4][1]


def test_active_prior_uses_no_true_hidden_q_or_future_return_effect():
    for split in x.B_STEPS:
        worlds=x.worlds(split)
        for w in worlds[:4]:
            b,slope=x.factual_b_and_slope(w)
            for state in x.STATES:
                for goal in x.GOALS:
                    action=x.active_probe(b,slope,state,goal)
                    assert action in (0,1)
                    # Policy observes only historical A effects, public target,
                    # symmetric prior, not any evaluator-only q.
                    w2=dict(w);w2["q"]=(10.,-10.)
                    assert x.active_probe(*x.factual_b_and_slope(w2),state,goal)==action
        for policy in x.POLICIES:
            scored=x.scripted_score(x._plans(worlds,policy))
            assert len(scored["pergroup_regret"])==12
            assert len(scored["choices"])==96
            assert math.isfinite(scored["mean_regret"])
            assert scored["oracle_mean_regret"]==0.


def test_actual_three_native_gradient_trainers_and_retained_dev_gate():
    result=x.run()
    assert result["model_count"]==3
    assert result["actual_optimizer_updates"]==1152
    assert result["train_factual_R25_sessions"]==256
    assert result["original_external_source_effects"]==0
    assert result["F7_replay"] is False
    assert result["deployment"]=="NONE"
    assert result["independent_scientific_pass"] is False
    assert result["native_conventional_convergence_qualified"] is False
    assert result["decision"]["native_development_pass"] in (True,False)
    assert result["decision"]["scripted_both_panel_pass"] in (True,False)
    for row in result["model_results"]:
        assert row["steps"]==384
        assert row["initial_hash"]!=row["trained_hash"]
        assert row["cold_checkpoint_action_parity"]
        for panel in x.B_STEPS:
            for name in x.POLICIES:
                for who in ("scripted","native"):
                    score=row["panels"][panel][name][who]
                    assert score["actual_selected_actions"]==96
                    assert score["factual_fifth_probe_effects"]==96
                    assert score["oracle_mean_regret"]==0.
                    assert len(score["pergroup_regret"])==12
    x.retain_ci_receipt(result)


def test_no_external_simulator_workflow_or_promotion():
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","env.reset(",
                      "env.do(","ctf_do(","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in src
