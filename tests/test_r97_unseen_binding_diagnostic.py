"""R97 prospective source/causality/transfer/negative controls (unit source only).

Pytest does NOT create any new original FIT/VALID/DEV or OOD physical source.
Original source first use belongs solely to the branch-scoped hosted seed jobs
after the exact SHA Linux+Windows G4 full regression.
"""
import copy
import itertools
import os
import numpy as np
import pytest
import torch
from experiments import r97_unseen_binding_diagnostic as r97


def test_frozen_preimplementation_custody_and_scope():
    f=r97.frozen()
    assert r97.r85.git_blob(r97.FREEZE.read_bytes())==r97.FREEZE_BLOB
    assert f["schema"]==r97.SCHEMA
    assert f["stage"]=="FROZEN_BEFORE_R97_IMPLEMENTATION_TESTS_OR_SOURCE_FIRST_USE"
    assert f["parent"]["original_r96_final_sha"]==r97.PARENT
    assert f["parent"]["r96_spent_sha"]==r97.r96.ORIGINAL_R96_SPENT_SHA
    assert f["parent"]["r96_spent_run"]==r97.r96.ORIGINAL_R96_SPENT_RUN_ID
    assert f["source"]["salts"]=={**r97.SALT,"TAG_PERMUTE":r97.TAG_PERMUTE_SALT,
        "BATCH":r97.BATCH_SALT,"BOOTSTRAP":r97.BOOTSTRAP_SALT,
        "UNIT_ONLY":r97.UNIT_SALT}
    assert f["source"]["seeds"]==list(r97.SEEDS)
    assert f["source"]["counts"]==r97.COUNT
    assert f["source"]["training_bindings"]==[list(z) for z in r97.TRAIN_BINDINGS]
    assert f["source"]["unseen_test_binding"]==list(r97.HELDOUT_BINDING)
    assert f["arms"]==list(r97.ARMS)
    assert f["training"]["all_arm_steps"]==r97.STEPS
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["authority"]["scoped_CPU_testonly_diagnostic_allowed"] is True
    assert not any(v for k,v in f["authority"].items()
                   if k!="scoped_CPU_testonly_diagnostic_allowed")


def test_original_source_first_use_denied_and_prior_spent_panels_not_reopened():
    for name in ("R96_DEV","R95_DEV","R89_FIT","OLD","PROTECTED"):
        with pytest.raises(PermissionError,match="R97_PREVIOUS_SPENT_OR_PROTECTED_PANEL"):
            r97._world(name,r97.SEEDS[0],0,"not-authorized")
    with pytest.raises(PermissionError,match="R97_NONFROZEN_SEED_OR_WORLD_INDEX"):
        r97._world("FIT",123,0,"not-authorized")
    with pytest.raises(PermissionError,match="R97_NONFROZEN_SEED_OR_WORLD_INDEX"):
        r97._world("DEV_HELDOUT",r97.SEEDS[0],256,"not-authorized")
    for name in r97.SALT:
        with pytest.raises(PermissionError,match="R97_NEW_SCORING_SOURCE_NOT_ADMITTED"):
            r97._world(name,r97.SEEDS[0],0,"not-authorized")
    with pytest.raises(PermissionError,match="R97_TRAINING_CANNOT_OPEN_FRESH_SCORED_SOURCE"):
        r97._factual("DEV_HELDOUT",r97.SEEDS[0],"not-authorized")
    with pytest.raises(PermissionError,match="R97_EVALUATOR_FORBIDS_TRAINING_OR_OLD_SOURCE"):
        r97._paired("FIT",r97.SEEDS[0],"not-authorized")
    if os.getenv("GITHUB_JOB")!="r97-unseen-binding":
        with pytest.raises(PermissionError,match="R97_ORIGINAL_CLEAN_HOSTED_G4_FIRST_ATTEMPT_ONLY"):
            r97._admit(r97.SEEDS[0])


def test_index_only_design_balance_and_family_exclusion_without_new_worlds():
    summary=r97.balance_preflight()
    for seed in r97.SEEDS:
        for panel,n in r97.COUNT.items():
            groups={}
            for i in range(n):
                key=(i%2,r97._chosen(seed,i))
                groups[key]=groups.get(key,0)+1
                family=r97.binding(panel,i)
                if panel in ("FIT","VALID","DEV_SEEN"):
                    assert family in r97.TRAIN_BINDINGS
                    assert family!=r97.HELDOUT_BINDING
                else:
                    assert family==r97.HELDOUT_BINDING
            assert len(groups)==4 and set(groups.values())=={n//4}
            assert summary[str(seed)][panel]=={"four_action_cold_cells":4,
                                                "count_per_cell":n//4}
    with pytest.raises(PermissionError,match="R97_UNFROZEN_FACTUAL_ACTION"):
        r97._chosen(3,0)


def test_noncommutative_physics_and_public_task_instruction_canonicalization():
    a=np.array([-1,1,-1,1],np.int64)
    assert r97.mux(a,(0,1,2))==-1
    assert r97.mux(a,(1,2,3))==-1
    assert r97.mux(a,(3,0,1))==-1
    b=np.array([1,1,-1,-1],np.int64)
    assert r97.mux(b,(0,1,2))==1
    assert r97.mux(b,(3,0,1))==1
    assert set(r97.tags_for((0,1,2)))=={0,1,2,3}
    assert set(r97.tags_for((3,0,1)))=={0,1,2,3}
    with pytest.raises(ValueError,match="R97_PHYSICAL_MUX_BAD_FACTORS_OR_ROLES"):
        r97.mux(np.array([0,1,-1,1]),(0,1,2))
    with pytest.raises(ValueError,match="R97_INVALID_PUBLIC_ROLE_TAG_BINDING"):
        r97.tags_for((0,0,1))
    for i in range(8):
        world,_=r97.unit_world(i)
        assert world["law_a"]==r97.mux(world["factors_a"],world["r97_binding"])
    with pytest.raises(PermissionError,match="R97_ONLY_INDEPENDENT_UNIT_FIXTURE_SOURCE"):
        r97.unit_world(8)


@pytest.mark.parametrize("i",[0,1,2,3,6,7])
def test_public_tags_are_preaction_data_and_no_future_observation_leak(i):
    world,choice=r97.unit_world(i)
    public,return_value=r97._public_view(world,choice)
    slots=r97.r88._public_slots(public)
    raw=r97._features(slots)
    assert public.shape==(48,13) and slots.shape==(5,13)
    assert public.dtype==np.float32 and raw.shape==(12,)
    assert sorted(np.rint(raw[4:8]*3).astype(int).tolist())==[0,1,2,3]
    assert raw[8]==float(choice==r97.r85.CALIBRATE)
    assert raw[9]==float(choice==r97.r85.EXPLOIT_POS)
    assert raw[11]==0.
    np.testing.assert_array_equal(raw[:4],slots[:4,10])
    canonical=r97._canonical(raw[None,:])
    tags=np.rint(raw[4:8]*3).astype(int)
    for tag in range(4):
        assert canonical[0,tag]==raw[int(np.flatnonzero(tags==tag)[0])]
    altered=copy.deepcopy(world)
    altered["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=30
    second,changed_reward=r97._public_view(altered,choice)
    np.testing.assert_array_equal(second,public)
    assert abs(changed_reward-return_value)>1.


def test_same_world_new_task_assignment_changes_the_physical_consequence():
    world,_=r97.unit_world(7)
    world["factors_a"]=np.array([1,-1,1,-1],np.int64)
    world["r97_binding"]=r97.HELDOUT_BINDING
    world["law_a"]=r97.mux(world["factors_a"],world["r97_binding"])
    alternative=copy.deepcopy(world)
    alternative["r97_binding"]=r97.REASSIGN_BINDING
    alternative["law_a"]=r97.mux(alternative["factors_a"],alternative["r97_binding"])
    assert world["law_a"]!=alternative["law_a"]
    p,r=r97._public_view(world,r97.r85.EXPLOIT_POS)
    q,s=r97._public_view(alternative,r97.r85.EXPLOIT_POS)
    np.testing.assert_array_equal(p[:,9:12],q[:,9:12])
    assert not np.array_equal(p[:,8],q[:,8])
    assert abs(r-s)>1.


def test_role_permutation_negative_retains_exact_combinatorial_information():
    # Formal preimplementation enumeration, NO original scored world access:
    # 4 binary factors × all 24 tag permutations yields 0.625 agreement,
    # hence tag scrambling is a role-binding control NOT full causal erasure.
    for family in (*r97.TRAIN_BINDINGS,r97.HELDOUT_BINDING):
        tags=r97.tags_for(family)
        agreement=[]
        for tup in itertools.product((-1,1),repeat=4):
            factors=np.asarray(tup,np.int64)
            truth=r97.mux(factors,family)
            for perm in itertools.permutations(range(4)):
                newtags=tags[np.asarray(perm)]
                implied=r97._private_implied_law(factors,newtags/3.)
                agreement.append(implied==truth)
        assert np.mean(agreement)==.625
    assert r97.frozen()["source"]["negative_role_disruption_not_causal_information_independence"]


def test_tag_permuted_negative_preserves_factual_cues_action_and_time():
    n=r97.COUNT["VALID"]
    raw=np.zeros((n,12),np.float32)
    raw[:,:4]=np.linspace(-1,1,n*4,dtype=np.float32).reshape(n,4)
    raw[:,4:8]=r97.tags_for((1,2,3))/3.
    raw[:,8]=1.;raw[:,10]=.5
    source=r97._permuted(raw,r97.SEEDS[0],"VALID")
    np.testing.assert_array_equal(source[:,:4],raw[:,:4])
    np.testing.assert_array_equal(source[:,8:],raw[:,8:])
    assert np.any(np.abs(source[:,4:8]-raw[:,4:8])>1.e-5)
    np.testing.assert_array_equal(source,
        r97._permuted(raw,r97.SEEDS[0],"VALID"))
    with pytest.raises(PermissionError,match="R97_TAG_NEGATIVE_ONLY_NEW_FROZEN_SOURCE"):
        r97._permuted(raw,r97.SEEDS[0],"R96_DEV")
    factors=np.tile(np.asarray([1,-1,1,-1],np.int8),(n,1))
    law=np.full(n,-1,np.int8)
    # True tag-only unchanged predicts law 100% and must fail the
    # 0.50–0.75 precommitted role-disruption audit.
    with pytest.raises(RuntimeError,match="R97_ROLE_DISRUPTION_NEGATIVE_CONTROL_INVALID"):
        r97._audit(law,raw,raw,factors,None)


def test_equal_native_cores_equal_initialized_controls_and_learned_set_gradients():
    torch.set_num_threads(1)
    models=r97._models(r97.SEEDS[0])
    assert set(models)==set(r97.ARMS)
    cores=(models[r97.NATIVE_RAW],models[r97.NATIVE_FLAT].core,
           models[r97.NATIVE_SET].core)
    states=[dict(c.named_parameters()) for c in cores]
    assert states[0].keys()==states[1].keys()==states[2].keys()
    assert all(torch.equal(states[0][k],states[1][k])
               and torch.equal(states[0][k],states[2][k]) for k in states[0])
    for a,b in ((r97.MLP_RAW,r97.BLIND),(r97.MLP_SET,r97.NEGATIVE)):
        x,y=[dict(models[arm].named_parameters()) for arm in (a,b)]
        assert x.keys()==y.keys()
        assert all(torch.equal(x[k],y[k]) for k in x)
    assert sum(x.numel() for x in models[r97.MLP_RAW].parameters())==5057
    assert sum(x.numel() for x in models[r97.GRU].parameters())==15233
    slots=[]
    for i in range(4):
        world,act=r97.unit_world(i)
        slots.append(r97.r88._public_slots(r97._public_view(world,act)[0]))
    seq=torch.from_numpy(np.stack(slots))
    raw=np.stack([r97._features(z) for z in slots])
    for name,m in models.items():
        f=(r97._canonical(raw) if name==r97.UPPER else
           r97._blind(raw) if name==r97.BLIND else raw)
        y=r97._forward(m,name,seq,torch.from_numpy(f))
        assert y.shape==(4,) and bool(torch.isfinite(y).all())
    t=torch.from_numpy(raw)
    loss=((models[r97.MLP_SET](t)-torch.tensor([2.,-1.,2.,-1.]))**2).mean()
    loss.backward()
    assert float(models[r97.MLP_SET].role[0].weight.grad.abs().sum())>0
    assert float(models[r97.MLP_SET].out[0].weight.grad.abs().sum())>0


def test_frozen_heldout_transfer_thresholds_and_no_official_native_promotion():
    f=r97.frozen()
    m=f["evaluation"]["mechanism_checks"]
    n=f["evaluation"]["native_checks"]
    assert len(m)==11 and len(n)==4
    assert m["heldout_tagged_set_minus_blind_MLP_reward_min"]==.75
    assert m["heldout_tagged_set_minus_permuted_negative_reward_min"]==.75
    assert m["heldout_tagged_set_minus_raw_MLP_reward_min"]==.15
    assert m["heldout_tagged_set_minus_GRU_reward_min"]==.10
    assert m["heldout_canonical_upper_minus_tagged_set_reward_gap_max"]==.30
    assert m["heldout_tagged_set_public_instruction_accuracy_min"]==.85
    assert m["shifted_OOD_cold_tagged_set_minus_permuted_reward_min"]==.50
    assert m["heldout_tagged_set_vs_permuted_paired_bootstrap_lower95_min"]==.10
    assert m["role_reassignment_true_law_flip_fraction_min"]==.20
    assert m["role_reassignment_tagged_set_action_flip_on_law_change_min"]==.70
    assert m["role_reassignment_tagged_set_private_law_accuracy_min"]==.85
    assert n["heldout_native_set_minus_native_flat_reward_min"]==.20
    assert n["heldout_native_set_minus_original_native_reward_min"]==.50
    assert n["shifted_OOD_cold_native_set_minus_flat_reward_min"]==.20
    assert n["heldout_native_set_vs_conventional_set_reward_gap_max"]==.30
    assert f["model"] if "model" in f else True
    assert f["source"]["OOD_scope"].startswith("Unseen binding")
    assert f["execution"]["new_workflow_file"] is False
    assert f["evaluation"]["independent_scientific_pass"] is False
