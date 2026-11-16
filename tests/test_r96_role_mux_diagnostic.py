"""R96 public noncommutative causal-role / source-custody tests.

All unit physics uses independent UNIT_ONLY salt. No original R96 FIT/DEV,
shifted OOD or ROLE_SWAP source is read in a normal regression.
"""
import copy
import os
import numpy as np
import pytest
import torch
from experiments import r96_role_mux_diagnostic as r96


def test_preimplementation_freeze_and_no_implicit_authority():
    f=r96.frozen()
    assert r96.r85.git_blob(r96.FREEZE.read_bytes())==r96.FREEZE_BLOB
    assert f["schema"]==r96.SCHEMA
    assert f["parent"]["R95_final_engineering_sha"]==r96.PARENT
    assert f["parent"]["R95_spent_original_sha"]==r96.r95.ORIGINAL_R95_SPENT_SHA
    assert f["parent"]["R95_status"]=="R95_RAW_RELATION_NO_GO"
    assert f["source"]["seeds"]==list(r96.SEEDS)
    assert f["source"]["world_counts"]==r96.COUNT
    assert f["source"]["salts"]=={**r96.SALT,"DERANGE":r96.DERANGE_SALT,
        "BOOTSTRAP":r96.BOOTSTRAP_SALT,"BATCH":r96.BATCH_SALT,"UNIT":r96.UNIT_SALT}
    assert f["arms"]==list(r96.ARMS)
    assert f["execution"]["new_workflow_file"] is False
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["authority"]["scoped_cpu_testonly_research_allowed"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="scoped_cpu_testonly_research_allowed")


def test_factual_action_and_pivot_index_design_balance_without_new_physics():
    pre=r96.balance_preflight()
    for seed in r96.SEEDS:
        for panel,n in r96.COUNT.items():
            buckets={}
            for index in range(n):
                key=(index%2,((index%r96.r85.N)//2)%2,r96._chosen(seed,index))
                buckets[key]=buckets.get(key,0)+1
            assert len(buckets)==8 and set(buckets.values())=={n//8}
            assert pre[str(seed)][panel]=={"joint_design_cells":8,"count_per_cell":n//8}
    with pytest.raises(PermissionError,match="R96_UNFROZEN_LOGGED_ACTION"):
        r96._chosen(4,0)


def test_spent_previous_and_fresh_new_source_denied_in_pytest():
    for panel in ("R95_DEV","R94_DEV","R89_FIT","OLD","PROTECTED"):
        with pytest.raises(PermissionError,match="R96_OLD_SPENT_OR_PROTECTED_SOURCE_DENIED"):
            r96._world(panel,r96.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R96_UNREGISTERED_SEED_PANEL_OR_WORLD"):
        r96._world("FIT",9,0,"fake")
    with pytest.raises(PermissionError,match="R96_UNREGISTERED_SEED_PANEL_OR_WORLD"):
        r96._world("DEV",r96.SEEDS[0],256,"fake")
    for panel in r96.SALT:
        with pytest.raises(PermissionError,match="R96_NEW_SOURCE_NOT_ADMITTED"):
            r96._world(panel,r96.SEEDS[0],0,"non-exact-source")
    with pytest.raises(PermissionError,match="R96_CANNOT_TRAIN_ON_NEW_SCORED_DEV"):
        r96._factual("DEV",r96.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R96_NEW_PAIRED_SCORING_NOT_FOR_FIT"):
        r96._scored("FIT",r96.SEEDS[0],"fake")
    if os.getenv("GITHUB_JOB")!="r96-independent-role-mux":
        with pytest.raises(PermissionError,match="R96_ONLY_ORIGINAL_HOSTED_G4_FIRST_ATTEMPT_SCIENCE"):
            r96._admit(r96.SEEDS[0])


def test_true_mux_is_order_dependent_and_independent_of_old_index_law():
    positive=np.asarray([1,1,-1,-1])
    negative=np.asarray([1,-1,1,-1])
    assert r96.mux(positive)==1
    assert r96.mux(negative)==-1
    np.testing.assert_array_equal(np.sort(positive),np.sort(negative))
    assert r96.mux(np.asarray([-1,1,-1,1]))==-1
    assert r96.mux(np.asarray([-1,-1,1,1]))==1
    with pytest.raises(ValueError,match="R96_PRIVATE_PHYSICAL_MUX_BINARY_FACTOR_SHAPE"):
        r96.mux(np.asarray([0,1,1,-1]))
    for i in range(8):
        w,a=r96.unit_world(i)
        assert w["law_a"]==r96.mux(w["factors_a"])
        assert a in (r96.r85.CALIBRATE,r96.r85.EXPLOIT_POS)
    with pytest.raises(PermissionError,match="R96_UNIT_ONLY_NONSCORING_SOURCE"):
        r96.unit_world(8)


@pytest.mark.parametrize("i",[0,1,2,3])
def test_physical_public_inputs_and_factual_future_are_causally_separated(i):
    w,act=r96.unit_world(i)
    pre,ret=r96.r86._view(w,act)
    x=r96.r88._public_slots(pre)
    raw=r96.r95._raw_from_slots(x)
    assert pre.shape==(48,13) and pre.dtype==np.float32
    assert x.shape==(5,13) and raw.shape==(8,)
    np.testing.assert_array_equal(raw[:4],x[:4,10])
    assert (raw[4],raw[5])==((1.,0.) if act==r96.r85.CALIBRATE else (0.,1.))
    assert (raw[0],raw[1],raw[2],raw[3])!=()
    blind=copy.deepcopy(w)
    blind["innovations"][w["pivot"]+1:w["pivot"]+9,0]+=40.
    pre2,ret2=r96.r86._view(blind,act)
    np.testing.assert_array_equal(pre,pre2)
    assert abs(ret2-ret)>1.


def test_public_hand_programmed_upper_and_sorted_negative_only_diagnostic():
    f=np.zeros((2,8),np.float32)
    f[0,:4]=[.8,.8,-.8,.8]
    f[1,:4]=[.8,-.8,.8,.8]
    f[:,4]=1
    upper=r96._upper(f)
    assert upper[:,0].tolist()==[1.,-1.]
    assert np.all(upper[:,1:4]==0)
    sorted_=r96._sorted(f)
    np.testing.assert_array_equal(sorted_[0,:4],sorted_[1,:4])
    np.testing.assert_array_equal(sorted_[:,4:],f[:,4:])
    with pytest.raises(ValueError,match="R96_ROLE_BLIND_REQUIRES_RAW_8D"):
        r96._sorted(np.zeros((2,4),np.float32))


def test_public_strata_derangement_never_falls_back_to_inverted_roll():
    n=r96.COUNT["VALID"];i=np.arange(n)
    f=np.zeros((n,8),np.float32)
    f[:,6]=np.where(i%2==0,.3,.9)
    choices=np.array([r96._chosen(r96.SEEDS[0],int(k)) for k in i],np.int8)
    f[:,4]=(choices==r96.r85.CALIBRATE)
    f[:,5]=(choices==r96.r85.EXPLOIT_POS)
    p=r96._derange(f,r96.SEEDS[0],"VALID",choices)
    assert (p!=i).all()
    np.testing.assert_array_equal(f[p,6],f[:,6])
    np.testing.assert_array_equal(choices[p],choices)
    np.testing.assert_array_equal(p,r96._derange(f,r96.SEEDS[0],"VALID",choices))
    assert not np.array_equal(p,np.roll(i,1))
    with pytest.raises(PermissionError,match="R96_DERANGE_ONLY_FROZEN_NEW_PUBLIC_PANEL"):
        r96._derange(f,r96.SEEDS[0],"R95_DEV",choices)


def test_evaluator_only_law_alignment_independence_checks_do_not_train():
    n=256;ix=np.arange(n)
    public=np.zeros((n,8),np.float32)
    public[:,6]=np.where(ix%2==0,.3,.9)
    law=np.where((ix//2)%2==0,1,-1).astype(np.int8)
    # TEST FIXTURE ONLY: deliberately 50% positive per public pivot.
    p=np.arange(n)
    with pytest.raises(RuntimeError,match="R96_NEGATIVE_LAW_ALIGNMENT_NOT_INDEPENDENT"):
        r96._audit(law,public,p,None)
    assert r96._audit(law,public,None,None)["pass"]
    # No scored R96 observations are read.


def test_native_core_equal_initialization_learned_gate_gradient_and_compute_receipts():
    torch.set_num_threads(1)
    models=r96._models(r96.SEEDS[0])
    assert set(models)==set(r96.ARMS)
    first,flat,gate=models[r96.NATIVE_RAW],models[r96.NATIVE_FLAT].core,models[r96.NATIVE_GATE].core
    a,b,c=[dict(m.named_parameters()) for m in (first,flat,gate)]
    assert a.keys()==b.keys()==c.keys()
    assert all(torch.equal(a[k],b[k]) and torch.equal(a[k],c[k]) for k in a)
    # Truly identical capacities and starting weights for meaningful negative.
    x,y=[dict(models[z].named_parameters()) for z in (r96.MLP_GATE,r96.DERANGED)]
    assert x.keys()==y.keys()
    assert all(torch.equal(x[k],y[k]) for k in x)
    assert sum(p.numel() for p in models[r96.MLP_RAW].parameters())==4801
    assert sum(p.numel() for p in models[r96.GRU].parameters())==15233
    obs=[]
    for i in range(4):
        world,act=r96.unit_world(i)
        obs.append(r96.r88._public_slots(r96.r86._view(world,act)[0]))
    slots=torch.from_numpy(np.stack(obs))
    raw=torch.from_numpy(np.stack([r96.r95._raw_from_slots(x) for x in obs]))
    for arm in r96.ARMS:
        data=torch.from_numpy(r96._upper(raw.numpy())) if arm==r96.UPPER else torch.from_numpy(
            r96._sorted(raw.numpy())) if arm==r96.SORTED else raw
        pred=r96._forward(models[arm],arm,slots,data)
        assert pred.shape==(4,) and bool(torch.isfinite(pred).all())
    value=models[r96.MLP_GATE](raw)
    ((value-torch.tensor([2.,-1.,2.,-1.]))**2).mean().backward()
    assert float(models[r96.MLP_GATE].gate[0].weight.grad.abs().sum())>0
    assert float(models[r96.MLP_GATE].candidate[0].weight.grad.abs().sum())>0


def test_original_frozen_mechanism_native_thresholds_and_gates():
    f=r96.frozen()
    c=f["evaluation"]["mechanism_checks"]
    n=f["evaluation"]["native_checks"]
    assert len(c)==10 and len(n)==4
    assert c["role_gate_vs_sorted_DEV_reward_min"]==.75
    assert c["role_gate_vs_deranged_DEV_reward_min"]==.75
    assert c["role_gate_vs_raw_MLP_DEV_reward_min"]==.15
    assert c["role_gate_vs_explicit_mux_upper_DEV_gap_max"]==.35
    assert c["role_gate_legal_mux_DEV_action_agreement_min"]==.85
    assert c["role_gate_vs_deranged_OOD_cold_reward_min"]==.5
    assert c["role_gate_vs_deranged_DEV_bootstrap_lower95_min"]==.1
    assert c["role_swap_true_law_change_fraction_min"]==.2
    assert c["role_gate_role_swap_action_flip_on_true_change_min"]==.65
    assert c["role_gate_role_swap_accuracy_min"]==.8
    assert n["native_role_gate_vs_flat_DEV_reward_min"]==.2
    assert n["native_role_gate_vs_raw_native_DEV_reward_min"]==.5
    assert n["native_role_gate_vs_flat_OOD_cold_reward_min"]==.2
    assert n["native_role_gate_vs_conventional_gate_DEV_reward_gap_max"]==.3
    assert f["training"]["steps_per_arm"]==r96.STEPS
    assert f["training"]["per_seed_CPU_resource_ceiling_seconds"]==r96.MAX_SECONDS


def test_r96_original_DEV_OOD_ROLE_SWAP_are_consumed_and_cannot_replay(monkeypatch):
    assert r96.ORIGINAL_R96_SPENT_SHA=="54f52d6d8e57941aa8ab75283c9b81285793d8a7"
    assert r96.ORIGINAL_R96_SPENT_RUN_ID=="43833655469"
    context={
        "GITHUB_ACTIONS":"true",
        "GITHUB_EVENT_NAME":"pull_request",
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_HEAD_REF":r96.BRANCH,
        "GITHUB_JOB":"r96-independent-role-mux",
        "R96_SEED":str(r96.SEEDS[0]),
        "RUNNER_OS":"Linux",
        "RUNNER_ENVIRONMENT":"github-hosted",
        "GITHUB_RUN_ATTEMPT":"1",
        "GITHUB_REF":"refs/pull/292/merge",
        "EXPECTED_CANDIDATE_SHA":"f"*40,
        "GITHUB_RUN_ID":"43888888888",
    }
    for key,value in context.items():
        monkeypatch.setenv(key,value)
    with pytest.raises(PermissionError,match="R96_ORIGINAL_FIRST_USE_ALREADY_CONSUMED_REPLAY_DENIED"):
        r96._admit(r96.SEEDS[0])
    with pytest.raises(PermissionError,match="R96_NEW_SOURCE_NOT_ADMITTED"):
        r96._world("FIT",r96.SEEDS[0],0,"f"*40)
    with pytest.raises(PermissionError,match="R96_NEW_SOURCE_NOT_ADMITTED"):
        r96._world("DEV",r96.SEEDS[0],0,"f"*40)
    with pytest.raises(PermissionError,match="R96_NEW_SOURCE_NOT_ADMITTED"):
        r96._world("ROLE_SWAP",r96.SEEDS[0],0,"f"*40)
