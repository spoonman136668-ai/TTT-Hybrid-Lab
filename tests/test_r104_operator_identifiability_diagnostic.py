"""R104 original-source-free physics identifiability, negative controls."""
from copy import deepcopy

import numpy as np
import pytest

from experiments import r85_delayed_composition as r85
from experiments import r104_operator_identifiability_diagnostic as r104


def test_r104_preimplementation_freeze_and_science_denial():
    f=r104.frozen()
    assert r85.git_blob(r104.FREEZE.read_bytes())==r104.FREEZE_BLOB
    assert f["source"]["UNIT_ONLY"] is True
    assert f["authority"]["protected_original_scientific_first_use"] is False
    assert f["authority"]["merge"] is False
    assert f["authority"]["FORGE_integration"] is False
    assert len(r104.SEEDS)==4
    assert r104.RECEIPT.name=="r104_unit_only_identifiability_audit.json"


def test_all_256_truth_maps_32640_distinct_pairs_and_iid_bounds():
    r=r104.all256_pairwise()
    assert r["all_candidate_operator_count"]==256
    assert r["all_unordered_pairs"]==32640
    assert sum(r["pair_counts_by_hamming_distance"].values())==32640
    assert r["pair_counts_by_hamming_distance"]["1"]==1024
    assert r["pair_counts_by_hamming_distance"]["8"]==128
    assert len(r["eight_selected_evaluator_masks_min_pair_distance"])==8
    assert min(r["eight_selected_evaluator_masks_min_pair_distance"].values())>=4
    assert r["chance_no_distinguishing_uniform_pattern"]["1"]["six"]==(.875)**6
    assert r["chance_no_distinguishing_uniform_pattern"]["1"]["sixteen"]==(.875)**16
    assert not r["scientific_pass"]


@pytest.mark.parametrize("seed",r104.SEEDS)
def test_operator_family_balanced_across_actual_public_pivot(seed):
    counts={(mask,pivot):0 for mask in r104.MASKS for pivot in (0,1)}
    for index in range(r104.N_PER_SEED):
        counts[(r104.MASKS[index%8],(index//8)%2)]+=1
    assert set(counts.values())=={8}


def test_factual_causal_operator_changes_reward_not_public_observation():
    found_opposite=found_same=0
    seed=r104.SEEDS[0]
    for index in range(10):
        for event in range(17):
            a=r104._unit_world(seed,index,event,r104.MASKS[0])
            b=r104._unit_world(seed,index,event,r104.MASKS[1])
            for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
                x,ra=r104._public_view(a,action)
                y,rb=r104._public_view(b,action)
                np.testing.assert_array_equal(x,y)
                if a["law_a"]!=b["law_a"]:
                    found_opposite+=1
                    assert abs(ra-rb)>1
                else:
                    found_same+=1
                    assert ra==rb
    assert found_opposite>20 and found_same>20


def test_unit_episode_factual_only_and_hidden_evaluator_fields_separated():
    x=r104.unit_episode(r104.SEEDS[0],0)
    p,e=x["public"],x["evaluator_only"]
    assert set(p)=={"support_slots","support_actions",
                    "actual_factual_rewards","query_slots"}
    assert "hidden_mask" not in p and "query_law" not in p
    assert p["support_slots"].shape==(16,5,13)
    assert p["actual_factual_rewards"].shape==(16,)
    assert p["query_slots"].shape==(5,13)
    assert e["hidden_mask"] in r104.MASKS
    expected=r85.CALIBRATE if e["query_law"]>0 else r85.EXPLOIT_POS
    assert expected==e["optimal_action"]
    with pytest.raises(PermissionError,match="R104_PRIVATE_SOURCE_FIELD_VISIBLE_TO_INFERENCE"):
        r104.posterior_decision({**p,"hidden_mask":e["hidden_mask"]})


def test_factual_posterior_only_uses_all_256_not_private_masks():
    x=r104.unit_episode(r104.SEEDS[1],3)
    p=x["public"]
    for mode in ("factual","no_feedback","shuffled","role_missing"):
        act=r104.posterior_decision(p,mode=mode)
        assert act in (r85.CALIBRATE,r85.EXPLOIT_POS)
    assert r104.posterior_decision(p,supports=6) in (r85.CALIBRATE,r85.EXPLOIT_POS)
    assert len(r104.HYPOTHESES)==256
    with pytest.raises(ValueError,match="R104_NONFROZEN_CONTROL_OR_SUPPORT_COUNT"):
        r104.posterior_decision(p,supports=8)


def test_stratified_shuffling_never_changes_public_actions_or_marginals():
    p=r104.unit_episode(r104.SEEDS[0],8)["public"]
    d=r104._shuffle_action_strata(p["actual_factual_rewards"],p["support_actions"])
    assert d.shape==(16,)
    assert not np.array_equal(d,p["actual_factual_rewards"])
    for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
        mask=p["support_actions"]==action
        np.testing.assert_array_equal(np.sort(d[mask]),
            np.sort(p["actual_factual_rewards"][mask]))


def test_unit_source_rejects_unregistered_old_scored_seed_and_masks():
    with pytest.raises(PermissionError,match="R104_UNIT_ONLY_GENERATOR_OUT_OF_SCOPE"):
        r104._unit_world(1003011,0,0,r104.MASKS[0])
    with pytest.raises(PermissionError,match="R104_UNIT_EPISODE_OR_SEED_NOT_FROZEN"):
        r104.unit_episode(1003011,0)
    with pytest.raises(PermissionError,match="R104_UNIT_ONLY_GENERATOR_OUT_OF_SCOPE"):
        r104._unit_world(r104.SEEDS[0],0,0,0xff)


def test_non_scoring_unit_run_cannot_call_protected_r103_world(monkeypatch):
    from experiments import r103_relational_binding_diagnostic as r103
    def denied(*args,**kw):
        raise AssertionError("R103_SPENT_PROTECTED_COHORT_REOPENED")
    monkeypatch.setattr(r103,"_world",denied)
    outcome=r104.unit_episode(r104.SEEDS[0],1)
    assert outcome["evaluator_only"]["hidden_mask"]==r104.MASKS[1]
    assert r104.posterior_decision(outcome["public"]) in (
        r85.CALIBRATE,r85.EXPLOIT_POS)


def test_frozen_unit_only_timeout_is_hard_reject_before_physics():
    with pytest.raises(TimeoutError,match="R104_FROZEN_UNIT_ONLY_180_SECONDS_PER_SEED_EXCEEDED"):
        r104.unit_episode(r104.SEEDS[0],0,deadline=-1)
    with pytest.raises(PermissionError,match="R104_UNIT_ONLY_SEED_NOT_REGISTERED"):
        r104.run_seed(1003011)
