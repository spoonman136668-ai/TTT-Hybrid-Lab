"""R102 training/scoring prospective-only tests: no original FIT/DEV opens."""
import copy
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r102_contradiction_evidence_diagnostic as r102
from experiments import r102_training_pipeline as pipe


def test_pipeline_exact_frozen_protocol_and_original_entrypoint_disabled():
    f=pipe.protocol()
    assert f["source"]["counts"]==r102.PANEL_COUNTS
    assert pipe.STEPS==3072 and pipe.BATCH==16 and pipe.SUPPORT==16
    assert len(pipe.ARMS)==11
    assert pipe.SEEDS==r102.SEEDS
    assert tuple(pipe.TRAIN)==("FIT","VALID")
    assert len(pipe.SCORED)==5
    assert not f["authority"]["protected_scored_first_use"]
    with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        pipe.original_ci(r102.SEEDS[0])


@pytest.mark.parametrize("panel",tuple(r102.PANEL_COUNTS))
def test_prospective_episode_and_full_dataset_cannot_open_original_science(panel):
    with pytest.raises(PermissionError,match="R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.episode(panel,r102.SEEDS[0],0,"a"*40,
                     paired=(panel in pipe.SCORED))
    with pytest.raises(PermissionError,match="R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.source(panel,r102.SEEDS[0],"a"*40)


def test_unfrozen_counterfactual_train_and_nonregistered_scoring_rejected():
    with pytest.raises(PermissionError,match="R102_FORBIDDEN_TRAIN_COUNTERFACTUAL_QUERY"):
        pipe.episode("FIT",r102.SEEDS[0],0,"a"*40,paired=True)
    with pytest.raises(PermissionError,match="R102_UNREGISTERED_PROSPECTIVE_EPISODE"):
        pipe.episode("PROTECTED",r102.SEEDS[0],0,"a"*40)
    with pytest.raises(PermissionError,match="R102_PROTECTED_SOURCE_SCOPE_NOT_REGISTERED"):
        pipe.source("R101_DEV_UNKNOWN",r102.SEEDS[0],"a"*40)


def test_unit_factual_outcomes_are_stratified_shuffled_without_law_inspection():
    f=r102.unit_fixture(0)
    p=f["public"]
    y,perm=pipe._observed_shuffled_outcomes(
        p["support_slots"],p["support_actions"],
        p["support_rewards"],"FIT",r102.SEEDS[0],0)
    again,back=pipe._observed_shuffled_outcomes(
        p["support_slots"],p["support_actions"],
        p["support_rewards"],"FIT",r102.SEEDS[0],0)
    np.testing.assert_array_equal(y,again)
    np.testing.assert_array_equal(perm,back)
    assert np.any(perm != np.arange(16))
    assert not np.any(perm == np.arange(16))
    np.testing.assert_array_equal(p["support_actions"][perm],
                                  p["support_actions"])
    np.testing.assert_array_equal(p["support_slots"][perm,-1,12],
                                  p["support_slots"][:,-1,12])
    with pytest.raises(PermissionError,match="R102_NONREGISTERED_SHUFFLE_SOURCE"):
        pipe._observed_shuffled_outcomes(
            p["support_slots"],p["support_actions"],
            p["support_rewards"],"R101_DEV",r102.SEEDS[0],0)


def unit_scoring_set():
    # No scored source is ever opened: evaluator values below come exclusively
    # from independent R102 UNIT_ONLY diagnostic worlds and are NOT science.
    f=[r102.unit_fixture(0),r102.unit_fixture(1)]
    ps=[x["public"] for x in f]
    es=[x["evaluator_only"] for x in f]
    shuffled=[pipe._observed_shuffled_outcomes(
        v["support_slots"],v["support_actions"],v["support_rewards"],
        "FIT",r102.SEEDS[0],i)[0] for i,v in enumerate(ps)]
    return {"support_slots":np.stack([v["support_slots"] for v in ps]),
            "support_rewards":np.stack([v["support_rewards"] for v in ps]),
            "support_rewards_shuffled":np.stack(shuffled),
            "support_actions":np.stack([v["support_actions"] for v in ps]),
            "query0":np.stack([v["query0"] for v in ps]),
            "query2":np.stack([v["query2"] for v in ps]),
            "reward0":np.asarray([v["return0"] for v in es],np.float32),
            "reward2":np.asarray([v["return2"] for v in es],np.float32),
            "private_law":np.asarray([v["private_law"] for v in es],np.int8),
            "cold":np.array([False,True])}


def test_all_eleven_model_values_and_evaluator_only_score_wiring():
    torch.set_num_threads(1)
    data=unit_scoring_set()
    models=r102.make_models(r102.SEEDS[0])
    result=pipe.score(models,"DEV_UNKNOWN",data)
    assert result["physical_episodes"]==2
    assert set(result["metrics"])==set(r102.ARMS)
    assert result["raw_reward"][r102.NATIVE].shape==(2,)
    assert 0<=result["role_relabel_action_agreement"]<=1
    assert np.isfinite(result["oracle_mean"])
    m=pipe.measures(r102.SEEDS[0],
                    {p:result for p in pipe.SCORED})
    assert set(m["measured"])==set(r102.frozen()["evaluation"]["checks"])
    assert isinstance(m["all_pass"],bool)
    assert m["same_source_first_six_vs_sixteen"] is True
    assert m["strongest_conventional"] in (
        r102.EXTERNAL,r102.GRU,r102.ROLE_SET,r102.MLP,r102.STATIC)
    assert 0<=sum(m["checks"].values())<=len(m["checks"])
    # That result is unit-only, untrained and deliberately never promoted.


def test_train_source_has_only_factual_target_and_rejects_mini_fixture():
    data=unit_scoring_set()
    fit=dict(data)
    fit["query"]=fit.pop("query0")
    fit["target"]=fit.pop("reward0")
    for key in ("query2","reward2","private_law","cold"):
        fit.pop(key)
    model=r102.ConventionalEvidenceValue("gru")
    take=np.zeros((pipe.STEPS,pipe.BATCH),np.int64)
    with pytest.raises(RuntimeError,match="R102_FULL_FIT_COHORT_SIZE_INVALID"):
        pipe.fit_arm(model,r102.GRU,fit,fit,take)
    with pytest.raises(PermissionError,match="R102_EVALUATOR_SCORED_SOURCE_INVALID"):
        pipe.score({r102.GRU:model},"DEV_UNKNOWN",fit)


def test_new_role_renaming_actually_relabels_cues_operand_tags_and_public_identity():
    import numpy as np
    old_world=r102._unit_world(0,3,r102.UNSEEN[0])
    new_world=copy.deepcopy(old_world)
    mapping=[2,0,3,1]
    facts=np.empty(4,np.int64)
    for src in range(4):
        facts[mapping[src]]=old_world["factors_a"][src]
    new_world["factors_a"]=facts
    new_world["r102_roles"]=tuple(mapping[k] for k in old_world["r102_roles"])
    new_world["probe"]={
        t:(phase,mapping[role] if phase=="A" else role,marker)
        for t,(phase,role,marker) in old_world["probe"].items()}
    new_world["a_probe_permutation"]=np.array(
        [mapping[k] for k in old_world["a_probe_permutation"]],np.int64)
    new_world["law_a"]=r102._operator_truth(
        facts,new_world["r102_roles"],r102.UNSEEN[0])
    assert old_world["law_a"]==new_world["law_a"]
    for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
        old,ret=r102.public_view(old_world,action)
        new,updated=r102.public_view(new_world,action)
        assert ret==updated
        s=np.repeat(old[None,None],pipe.SUPPORT,axis=1)
        renamed,q=pipe.permute_public_roles(s,old[None])
        np.testing.assert_array_equal(renamed[0,0],new)
        np.testing.assert_array_equal(q[0],new)
        np.testing.assert_array_equal(q[0,-1],old[-1])
        np.testing.assert_array_equal(q[0,:4,9],
                                      np.array([.25,.5,.75,1.],np.float32))


def test_short_native_model_never_sees_extra_ten_outcomes_or_public_events():
    torch.set_num_threads(1)
    f=r102.unit_fixture(0)
    p=f["public"]
    from experiments import r95_learned_relation_diagnostic as r95
    core=r95._native(r102.SEEDS[0])
    torch.manual_seed(r102.SEEDS[0]+102)
    model=r102.NativeEvidenceValue(core,"short")
    x=torch.from_numpy(p["support_slots"][None])
    y=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query2"][None])
    model.eval()
    with torch.no_grad():
        baseline=model(x,y,q)
        changed_x=x.clone();changed_y=y.clone()
        changed_x[:,6:]=9999.
        changed_y[:,6:]=-9999.
        assert torch.equal(baseline,model(changed_x,changed_y,q))


def test_paired_2000_bootstrap_reproducible_in_UNIT_ONLY_and_rejects_bad_seed():
    a=np.array([1.,0.,3.,4.])
    b=np.array([0.,0.,0.,1.])
    assert pipe.bootstrap_lower(a,b,r102.SEEDS[0])==pipe.bootstrap_lower(
        a,b,r102.SEEDS[0])
    with pytest.raises(ValueError,match="R102_BOOTSTRAP_NONMATCHED_COHORT"):
        pipe.bootstrap_lower(a,b,42)
