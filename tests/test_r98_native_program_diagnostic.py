"""R98 frozen-test-only native internal causal-program experiment regressions.

All physics in pytest is independent UNIT_ONLY test fixture. FIT, VALID,
DEV_COMPOSE, shifted OOD and program-change source is never opened here.
"""
import copy
import os
import numpy as np
import pytest
import torch
from experiments import r98_native_program_diagnostic as r98


def test_r98_prospectively_frozen_parent_and_scoped_authority():
    f=r98.frozen()
    assert r98.r85.git_blob(r98.FREEZE.read_bytes())==r98.FREEZE_BLOB
    assert f["schema"]==r98.SCHEMA
    assert f["stage"]=="PROSPECTIVE_FREEZE_BEFORE_IMPLEMENTATION_TESTS_OR_FRESH_SOURCE_FIRST_USE"
    assert f["parent"]["r97_qualified_nonreplaying_SHA"]==r98.PARENT
    assert f["parent"]["r97_original_scored_SHA"]==r98.r97.ORIGINAL_R97_SPENT_SHA
    assert f["parent"]["r97_original_scored_run"]==r98.r97.ORIGINAL_R97_SPENT_RUN_ID
    assert f["source"]["seeds"]==list(r98.SEEDS)
    assert f["source"]["salts"]=={**r98.SALT,"TAG_NEGATIVE":r98.NEGATIVE_SALT,
                                  "BATCH":r98.BATCH_SALT,"BOOTSTRAP":r98.BOOTSTRAP_SALT,
                                  "UNIT_ONLY":r98.UNIT_SALT}
    assert f["source"]["counts"]==r98.COUNT
    assert f["arms"]==list(r98.ARMS)
    assert f["training"]["full_steps_each_arm"]==r98.STEPS
    assert f["training"]["batch"]==r98.BATCH
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["authority"]["scoped_CPU_testonly_diagnostic"] is True
    assert not any(x for k,x in f["authority"].items()
                   if k!="scoped_CPU_testonly_diagnostic")


def test_old_spent_and_fresh_original_physics_denied_outside_admitted_ci():
    for name in ("R97_DEV_HELDOUT","R96_DEV","R95_DEV","PRIOR","PROTECTED"):
        with pytest.raises(PermissionError,match="R98_PREVIOUS_SPENT_OR_PROTECTED_SOURCE"):
            r98._world(name,r98.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R98_UNREGISTERED_SEED_OR_INDEX"):
        r98._world("DEV_COMPOSE",8,0,"fake")
    with pytest.raises(PermissionError,match="R98_UNREGISTERED_SEED_OR_INDEX"):
        r98._world("DEV_COMPOSE",r98.SEEDS[0],256,"fake")
    for panel in r98.SALT:
        with pytest.raises(PermissionError,match="R98_FRESH_PHYSICAL_SOURCE_NOT_ADMITTED"):
            r98._world(panel,r98.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R98_FIT_CANNOT_ACCESS_HELDOUT_OR_OLD_SCORED_PANEL"):
        r98._factual("DEV_COMPOSE",r98.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R98_EVALUATOR_ONLY_FRESH_POSTTRAIN_PANEL"):
        r98._scored("FIT",r98.SEEDS[0],"fake")
    if os.getenv("GITHUB_JOB")!="r98-internal-program":
        with pytest.raises(PermissionError,match="R98_ORIGINAL_HOSTED_G4_FIRST_ATTEMPT_ONLY"):
            r98._admit(r98.SEEDS[0])


def test_train_vs_unseen_program_is_index_only_and_source_action_balanced():
    preflight=r98.balance_preflight()
    for seed in r98.SEEDS:
        for panel,n in r98.COUNT.items():
            counts={}
            for i in range(n):
                pair=(i%2,r98._chosen(seed,i))
                counts[pair]=counts.get(pair,0)+1
                op=r98.program(panel,i)
                if panel in ("FIT","VALID","DEV_SEEN"):
                    assert op in r98.TRAIN_PROGRAMS
                    assert op!=r98.COMPOSE_PROGRAM
                else:
                    assert op==r98.COMPOSE_PROGRAM
            assert len(counts)==4 and set(counts.values())=={n//4}
            assert preflight[str(seed)][panel]=={"joint_cold_and_action_cells":4,"count_per_cell":n//4}
    with pytest.raises(PermissionError,match="R98_UNFROZEN_FACTUAL_INTERVENTION"):
        r98._chosen(-1,0)


def test_truth_tables_and_fresh_composition_of_known_primitives_are_distinct():
    # Four private A signs, three observed public role identities. All truth
    # tables exist as evaluator-only physics and are NOT learner supervision.
    for bits in range(16):
        a=np.array([1 if bits&(1<<i) else -1 for i in range(4)],np.int64)
        for roles in r98.ROLE_BINDINGS+(r98.HELDOUT_BINDING,):
            mux=r98.law_of(a,roles,(0,0))
            neg_mux=r98.law_of(a,roles,(0,1))
            parity=r98.law_of(a,roles,(1,0))
            neg_parity=r98.law_of(a,roles,(1,1))
            assert mux==-neg_mux
            assert parity==-neg_parity
            assert parity==int(np.prod(a))
            assert all(z in (-1,1) for z in (mux,neg_mux,parity,neg_parity))
    with pytest.raises(ValueError,match="R98_INVALID_PRIVATE_CAUSAL_PROGRAM"):
        r98.law_of(np.array([0,1,-1,1]),(0,1,2),(1,1))
    assert set(r98.tag_for((0,1,2)))=={0,1,2,3}
    with pytest.raises(ValueError,match="R98_INVALID_PUBLIC_ROLE_ASSIGNMENT"):
        r98.tag_for((0,0,1))


@pytest.mark.parametrize("i",[0,1,2,8,9,10,11])
def test_public_operator_cues_present_before_physical_action_and_no_future_return_leak(i):
    world,chosen=r98.unit_world(i)
    assert world["law_a"]==r98.law_of(world["factors_a"],
                                     world["r98_roles"],world["r98_op"])
    public,ret=r98._public_view(world,chosen)
    slots=r98.r88._public_slots(public)
    raw=r98._raw(slots)
    assert public.shape==(48,13)
    assert slots.shape==(5,13) and slots.dtype==np.float32
    assert raw.shape==(14,) and raw.dtype==np.float32
    assert tuple(raw[8:10].astype(int))==world["r98_op"]
    assert sorted(np.rint(raw[4:8]*3).astype(int).tolist())==[0,1,2,3]
    assert raw[10]==float(chosen==r98.r85.CALIBRATE)
    assert raw[11]==float(chosen==r98.r85.EXPLOIT_POS)
    assert raw[13]==0.
    np.testing.assert_array_equal(raw[:4],slots[:4,10])
    assert r98._private_public_rule(raw[None,:])[0] in (-1,1)
    different=copy.deepcopy(world)
    different["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=40
    pre2,ret2=r98._public_view(different,chosen)
    np.testing.assert_array_equal(pre2,public)
    assert abs(ret2-ret)>1.
    with pytest.raises(PermissionError,match="R98_UNIT_ONLY_NONSCORING_SOURCE"):
        r98.unit_world(12)


def test_same_world_program_change_preserves_public_noisy_factors_but_changes_truth():
    world,_=r98.unit_world(9)
    world["factors_a"]=np.array([1,1,-1,1],np.int64)
    world["r98_roles"]=r98.HELDOUT_BINDING
    world["r98_op"]=(1,1)
    world["law_a"]=r98.law_of(world["factors_a"],world["r98_roles"],world["r98_op"])
    counter=copy.deepcopy(world)
    counter["r98_op"]=(0,1)
    counter["law_a"]=r98.law_of(counter["factors_a"],counter["r98_roles"],counter["r98_op"])
    assert world["law_a"]!=counter["law_a"]
    p0,r0=r98._public_view(world,r98.r85.EXPLOIT_POS)
    p1,r1=r98._public_view(counter,r98.r85.EXPLOIT_POS)
    np.testing.assert_array_equal(p0[:,9:12],p1[:,9:12])
    assert not np.array_equal(p0[:,8],p1[:,8])
    assert abs(r0-r1)>1.


def test_public_canonical_rule_explicitly_engineered_and_negative_not_claimed_independent():
    raw=np.zeros((2,14),np.float32)
    raw[:,:4]=np.array([[.9,-.7,.9,-.9],[.9,.7,.9,.9]],np.float32)
    raw[:,4:8]=r98.tag_for(r98.HELDOUT_BINDING)/3.
    raw[:,8:10]=[1.,1.]
    raw[:,10]=1.;raw[:,12]=.5
    upper=r98._canonical(raw)
    np.testing.assert_array_equal(upper[:,0],r98._private_public_rule(raw))
    assert np.all(upper[:,1:10]==0)
    np.testing.assert_array_equal(upper[:,10:],raw[:,10:])
    blind=r98._blind(raw)
    assert np.all(blind[:,4:10]==0)
    neg=r98._corrupt(np.tile(raw[0],(r98.COUNT["VALID"],1)),
                    r98.SEEDS[0],"VALID")
    assert neg.shape==(r98.COUNT["VALID"],14)
    assert np.any(neg[:,8:10]!=raw[0,8:10])
    np.testing.assert_array_equal(neg[:,:8],np.tile(raw[0,:8],(r98.COUNT["VALID"],1)))
    assert r98.frozen()["source"]["negative_anticipated_caveat"].find("retains")>=0


def test_initial_native_state_equal_across_four_arms_and_internal_gradient_enters_trunk():
    torch.set_num_threads(1)
    models=r98._models(r98.SEEDS[0])
    assert set(models)==set(r98.ARMS)
    cores=[models[r98.NATIVE_RAW],models[r98.NATIVE_FLAT].core,
           models[r98.NATIVE_INTERNAL].core,models[r98.NATIVE_SIDECAR].core]
    states=[dict(x.named_parameters()) for x in cores]
    assert all(x.keys()==states[0].keys() for x in states)
    assert all(torch.equal(states[0][k],z[k])
               for z in states[1:] for k in states[0])
    a,b=[dict(models[name].named_parameters()) for name in (r98.SET,r98.NEGATIVE)]
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    slots=[];raw=[]
    for i in (0,1,9,10):
        world,chosen=r98.unit_world(i)
        x=r98.r88._public_slots(r98._public_view(world,chosen)[0])
        slots.append(x)
        raw.append(r98._raw(x))
    x=torch.from_numpy(np.stack(slots))
    f=torch.from_numpy(np.stack(raw))
    for name,m in models.items():
        o=r98._canonical(f.numpy()) if name==r98.UPPER else f.numpy()
        v=r98._forward(m,name,x,torch.from_numpy(o))
        assert v.shape==(4,) and bool(torch.isfinite(v).all())
    model=models[r98.NATIVE_INTERNAL]
    y=model(x,f)
    ((y-torch.tensor([1.,-1.,1.,-1.]))**2).mean().backward()
    assert float(model.adapter.event[0].weight.grad.abs().sum())>0
    assert float(model.adapter.program[0].weight.grad.abs().sum())>0
    assert float(model.core.stem.weight.grad.abs().sum())>0
    assert float(model.core.head.weight.grad.abs().sum())>0
    # This is an ACTUAL gradient inside the native trunk, not external head.
    assert not hasattr(model,"head")


def test_frozen_native_internal_and_unseen_operator_thresholds():
    f=r98.frozen()
    t=f["evaluation"]["mechanism_checks"]
    assert len(t)==10
    assert t["composed_internal_minus_original_native_DEV_reward_min"]==.5
    assert t["composed_internal_minus_flat_native_stem_DEV_reward_min"]==.2
    assert t["composed_internal_minus_external_role_sidecar_DEV_reward_min"]==.2
    assert t["composed_internal_vs_strongest_conventional_DEV_reward_margin_min"]==.1
    assert t["composed_internal_true_law_accuracy_min"]==.8
    assert t["composed_internal_vs_original_native_shifted_cold_reward_min"]==.3
    assert t["composed_internal_minus_original_native_DEV_bootstrap_lower95_min"]==.1
    assert t["program_switch_true_law_change_fraction_min"]==.2
    assert t["program_switch_internal_action_change_on_true_change_min"]==.65
    assert t["program_switch_internal_true_law_accuracy_both_min"]==.8
    assert f["execution"]["two_fresh_seed_jobs_only_in_existing_G4"]
    assert f["architecture"]["original_native_weights_not_promoted"]


def test_r98_original_scored_source_was_consumed_and_all_new_runs_are_denied(monkeypatch):
    assert r98.ORIGINAL_R98_SPENT_SHA=="8eda7b186605691a4ee0183440c4d897092cd01e"
    assert r98.ORIGINAL_R98_SPENT_RUN_ID=="43841693572"
    context={
        "GITHUB_ACTIONS":"true",
        "GITHUB_EVENT_NAME":"pull_request",
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_HEAD_REF":r98.BRANCH,
        "GITHUB_JOB":"r98-internal-program",
        "R98_SEED":str(r98.SEEDS[0]),
        "RUNNER_OS":"Linux",
        "RUNNER_ENVIRONMENT":"github-hosted",
        "GITHUB_RUN_ATTEMPT":"1",
        "GITHUB_REF":"refs/pull/294/merge",
        "EXPECTED_CANDIDATE_SHA":"f"*40,
        "GITHUB_RUN_ID":"43888888888",
    }
    for key,value in context.items():
        monkeypatch.setenv(key,value)
    with pytest.raises(PermissionError,match="R98_SPENT_TEST_ONLY_FIRST_USE_REPLAY_DENIED"):
        r98._admit(r98.SEEDS[0])
    for panel in ("FIT","VALID","DEV_SEEN","DEV_COMPOSE","OOD_SHIFT","ROLE_REASSIGN"):
        with pytest.raises(PermissionError,match="R98_FRESH_PHYSICAL_SOURCE_NOT_ADMITTED"):
            r98._world(panel,r98.SEEDS[0],0,"f"*40)
