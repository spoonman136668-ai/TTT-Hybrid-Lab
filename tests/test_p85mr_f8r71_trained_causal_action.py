"""R71 frozen trainable stateful action-consequence policy acceptance controls."""
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments import p85mr_f8r71_trained_causal_action as r71


def test_freeze_custody_pending_R70_bilateral_and_protected_denial():
    freeze=r71.locked()
    assert r71.git_blob(r71.FREEZE.read_bytes())==r71.FREEZE_BLOB
    assert freeze["parent"]["R70_SHA"]==r71.PARENT
    assert freeze["parent"]["R70_Windows"]=="PENDING_AT_FREEZE"
    assert freeze["parent"]["R70_bilateral_acceptance_required_before_science_promotion"]
    assert freeze["parent"]["all_R64_to_R70_scored_worlds_spent"]
    assert freeze["metrics"]["no_post_result_tuning"]
    assert all(v is False for v in freeze["authority"].values())
    for panel in ("OOD","R70_DEV","R69_DEV","R68_DEV","PROXY_NEGATIVE"):
        with pytest.raises(PermissionError,match="F8R71_SPENT_OR_PROTECTED_SOURCE_DENIED"):
            r71.worlds(panel)
    with pytest.raises(PermissionError,match="F8R71_PROTECTED_EXTERNAL_FIRSTUSE_DENIED"):
        r71.deny_protected_external_firstuse()


def test_new_source_balances_laws_and_selected_actuator_transition():
    all_worlds={panel:r71.worlds(panel) for panel in r71.COUNTS}
    def digest(w):
        return hashlib.sha256(
            w["potential_innovations_evaluator_only"].tobytes()).hexdigest()
    assert set(map(digest,all_worlds["FIT"])).isdisjoint(
        map(digest,all_worlds["DEV"]))
    for panel,source in all_worlds.items():
        assert len(source)==r71.COUNTS[panel]
        assert sum(w["switched_evaluator_only"] for w in source)==len(source)//2
        assert sum(w["initial_sign_evaluator_only"]==1 for w in source)==len(source)//2
        for w in source:
            assert w["source_panel"]==panel
            assert w["potential_innovations_evaluator_only"].shape==(40,4,4)
            assert w["unobserved_nuisance_proxy"].shape==(40,4)
            assert 9<=w["B_start_evaluator_only"]<=18
            assert 22<=w["A_return_start_evaluator_only"]<=35
            assert np.isfinite(w["potential_innovations_evaluator_only"]).all()
            assert len(w["latent_law_evaluator_only"])==40
        assert all(w["panel"] in ("FIT","DEV") for w in source)
    for world in all_worlds["FIT"][:4]:
        _,corrected=r71.r70.transition(world,0,0,0)
        _,unfixed=r71.r70.transition(world,0,1,0)
        next_sign=int(world["latent_law_evaluator_only"][1])
        act=2 if next_sign>0 else 3
        factual,_=r71.r70.transition(world,1,act,corrected)
        sham,_=r71.r70.transition(world,1,act,unfixed)
        assert float(factual[0]-sham[0])==pytest.approx(.8)
    negative=r71.negative_proxy(all_worlds["DEV"])
    for a,b in zip(all_worlds["DEV"],negative):
        assert np.array_equal(a["latent_law_evaluator_only"],
                              b["latent_law_evaluator_only"])
        assert np.array_equal(a["potential_innovations_evaluator_only"],
                              b["potential_innovations_evaluator_only"])
        assert not np.array_equal(a["unobserved_nuisance_proxy"],
                                  b["unobserved_nuisance_proxy"])
    with pytest.raises(PermissionError,match="F8R71_PROXY_ONLY_FRESH_DEV"):
        r71.negative_proxy(all_worlds["FIT"])


def test_trainable_policies_use_only_selected_public_information_and_gradients():
    fit=r71.worlds("FIT")
    for arm in r71.TRAINED:
        model=r71.TrainablePolicy(arm,r71.SEEDS[0])
        generator=torch.Generator(device="cpu").manual_seed(12345)
        out=r71.rollout(model,fit[0],greedy=False,action_gen=generator)
        assert len(out["actions"])==40
        assert all(action in (0,2,3) for action in out["actions"])
        assert len(out["observations"])==40
        assert len(out["logprobs"])==40
        assert len(out["entropies"])==40
        assert np.isfinite(out["rewards"]).all()
        loss=-torch.stack(out["logprobs"]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        greedy=r71.rollout(model,fit[0],greedy=True)
        assert len(greedy["actions"])==40
        assert not greedy["logprobs"]
        assert np.isfinite(greedy["rewards"]).all()
    with pytest.raises(ValueError,match="F8R71_UNREGISTERED_TRAINED_POLICY"):
        r71.TrainablePolicy("ORACLE",r71.SEEDS[0])
    with pytest.raises(ValueError,match="F8R71_SELECTED_ACTION_NOT_ALLOWED"):
        r71._public_input(1,np.zeros(4),2)
    with pytest.raises(ValueError,match="F8R71_SELECTED_OBSERVATION_INVALID"):
        r71._public_input(0,np.array([0,0,float("nan"),0]),2)


def test_fixed_controls_new_worlds_and_no_future_outcome_shortcut():
    w=r71.worlds("FIT")[0]
    for arm in r71.FIXED:
        scored=r71.fixed_rollout(arm,w)
        assert len(scored["actions"])==40
        assert np.isfinite(scored["rewards"]).all()
        assert len(scored["alignment_state"])==40
    mutated={**w,"potential_innovations_evaluator_only":
             w["potential_innovations_evaluator_only"].copy()}
    mutated["potential_innovations_evaluator_only"][39,:,0]+=70
    for arm in r71.TRAINED:
        model=r71.TrainablePolicy(arm,r71.SEEDS[0])
        a=r71.rollout(model,w,greedy=True)
        b=r71.rollout(model,mutated,greedy=True)
        assert a["actions"]==b["actions"]
    for arm in r71.FIXED:
        a=r71.fixed_rollout(arm,w)
        b=r71.fixed_rollout(arm,mutated)
        assert a["actions"]==b["actions"]


def test_full_frozen_two_seed_action_policy_score_and_no_false_promotion():
    outcome=r71.run()
    assert outcome["schema"]==r71.SCHEMA
    assert outcome["freeze_blob"]==r71.FREEZE_BLOB
    assert outcome["parent"]==r71.PARENT
    assert outcome["source"]=={
        "FIT":128,"CAL":32,"DEV":96,
        "switched_DEV":48,"protected_external_OOD":0}
    assert len(outcome["verdicts"])==2
    for seed in r71.SEEDS:
        receipts=outcome["training"][str(seed)]
        assert set(receipts)==set(r71.TRAINED)
        for arm in r71.TRAINED:
            z=receipts[arm]
            assert z["optimizer_steps"]==32
            assert z["FIT_worlds_per_step"]==8
            assert z["FIT_selected_factual_event_exposures"]==32*8*40
            assert len(z["weights_sha256"])==64
            assert np.isfinite(z["loss_first"])
            assert np.isfinite(z["loss_last"])
        scored=outcome["scores"][str(seed)]
        assert set(scored)==set(r71.TRAINED+r71.FIXED)
        for arm,s in scored.items():
            assert s["selected_action_events"]==96*40
            assert np.isfinite(s["episode_reward"])
            assert np.isfinite(s["switched_late_return_reward"])
            assert np.isfinite(s["stable_episode_reward"])
    for v in outcome["verdicts"]:
        assert v["seed"] in r71.SEEDS
        assert v["strongest_eligible_control"] in r71.ELIGIBLE
        assert v["checks"]["matched_FIT_rollout_steps_and_actions"]
        assert v["checks"]["all_policy_selected_action_budget"]
        assert v["checks"]["exact_cold_replay"]
        assert v["checks"]["negative_proxy_invariance"]
        assert v["checks"]["fresh_source_nonoverlap"]
        assert v["checks"]["physical_transition_preserved"]
        assert v["checks"]["hidden_law_not_passed_to_learner"]
        assert v["all_frozen_checks"]==all(v["checks"].values())
    assert outcome["DEV_admission"] in ("DEV_NO_GO","DEV_TRAINED_POLICY_SIGNAL_ONLY")
    assert (outcome["DEV_admission"]=="DEV_TRAINED_POLICY_SIGNAL_ONLY")==all(
        v["all_frozen_checks"] for v in outcome["verdicts"])
    assert outcome["independent_scientific_pass"] is False
    assert outcome["full_hybrid_substrate_qualification"] is False
    assert outcome["CUDA_used"] is False
    assert outcome["external_scientific_firstuse"] is False
    assert outcome["deployment"]=="NONE"
    json.dumps(outcome,allow_nan=False)
    r71.retain_ci_receipt(outcome)
