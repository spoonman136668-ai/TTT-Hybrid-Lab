"""R99 independent unit-only physical and microfeedback source-custody checks.

None of these tests generate any new R99 scored FIT/VALID/DEV/OOD/SWITCH
worlds: only the separately salted tiny non-scoring UNIT fixtures.
"""
import copy
import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments import r99_microfeedback_diagnostic as r99


def test_R99_prospective_protocol_frozen_before_any_source_first_use():
    f=r99.frozen()
    assert r99.r85.git_blob(r99.FREEZE.read_bytes())==r99.FREEZE_BLOB
    assert f["schema"]==r99.SCHEMA
    assert f["stage"]=="FREEZE_BEFORE_IMPLEMENTATION_TESTS_OR_PHYSICAL_SOURCE_FIRST_USE"
    assert f["parent"]["r98_final_nonreplaying_SHA"]==r99.PARENT
    assert f["parent"]["r98_original_scored_SHA"]==r99.r98.ORIGINAL_R98_SPENT_SHA
    assert f["parent"]["r98_original_scored_run"]==r99.r98.ORIGINAL_R98_SPENT_RUN_ID
    assert not f["parent"]["reuse_prior_scored_source_or_weights"]
    assert f["source"]["seeds"]==list(r99.SEEDS)
    assert f["source"]["salts"]=={**r99.SALT,"AUX_SHUFFLE":r99.SHUFFLE_SALT,
        "BATCH":r99.BATCH_SALT,"BOOTSTRAP":r99.BOOTSTRAP_SALT,
        "UNIT_ONLY":r99.UNIT_SALT}
    assert f["source"]["counts"]==r99.COUNT
    assert f["arms"]==list(r99.ARMS)
    assert not f["evaluation"]["independent_scientific_pass"]
    assert f["authority"]["scoped_CPU_TEST_only_original_first_use"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="scoped_CPU_TEST_only_original_first_use")


def test_R99_registered_world_denied_outside_exact_original_first_use():
    for old in ("R98_DEV_COMPOSE","R97_DEV_HELDOUT","R96_DEV","PROTECTED","R85_FIT"):
        with pytest.raises(PermissionError,match="R99_PREVIOUS_SPENT_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r99._world(old,r99.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R99_UNREGISTERED_SOURCE_SEED_INDEX"):
        r99._world("DEV_COMPOSE",42,0,"fake")
    with pytest.raises(PermissionError,match="R99_UNREGISTERED_SOURCE_SEED_INDEX"):
        r99._world("DEV_COMPOSE",r99.SEEDS[0],256,"fake")
    for panel in r99.SALT:
        with pytest.raises(PermissionError,match="R99_FIRST_USE_PHYSICAL_SOURCE_NOT_ADMITTED"):
            r99._world(panel,r99.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R99_FACTUAL_TRAINING_CANNOT_OPEN_UNSEEN_OR_SCORED_PANEL"):
        r99._factual("DEV_COMPOSE",r99.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R99_SCORER_CANNOT_ACCESS_FIT_OR_SPENT_PANEL"):
        r99._scored("FIT",r99.SEEDS[0],"fake")
    if os.getenv("GITHUB_JOB")!="r99-factual-microfeedback":
        with pytest.raises(PermissionError,match="R99_ORIGINAL_G4_HOSTED_FIRST_ATTEMPT_CPU_ONLY"):
            r99._admit(r99.SEEDS[0])


def test_R99_frozen_action_and_program_balance_without_opening_scored_worlds():
    b=r99.design_preflight()
    for seed in r99.SEEDS:
        for panel,n in r99.COUNT.items():
            cells={}
            for i in range(n):
                key=(i%2,r99.chosen(seed,i))
                cells[key]=cells.get(key,0)+1
                op=r99.program(panel,i)
                if panel in ("FIT","VALID","DEV_SEEN"):
                    assert op in r99.TRAIN_PROGRAMS and op!=r99.COMPOSE_PROGRAM
                    assert r99.binding(panel,i) in r99.TRAIN_BINDINGS
                else:
                    assert op==r99.COMPOSE_PROGRAM
                    assert r99.binding(panel,i)==r99.HELDOUT_BINDING
            assert len(cells)==4 and set(cells.values())=={n//4}
            assert b[str(seed)][panel]=={"cold_action_strata":4,"each_count":n//4}
    with pytest.raises(PermissionError,match="R99_FACTUAL_ACTION_NOT_FROZEN"):
        r99.chosen(-3,0)


@pytest.mark.parametrize("i",[0,1,2,8,9,10,11])
def test_R99_public_preaction_native_input_and_counterfactual_leak_negative(i):
    world,action=r99.unit_world(i)
    assert world["law_a"]==r99.r98.law_of(
        world["factors_a"],world["r98_roles"],world["r98_op"])
    slots,raw,reward=r99._observed(world,action)
    assert slots.shape==(5,13) and slots.dtype==np.float32
    assert raw.shape==(14,) and raw.dtype==np.float32
    assert tuple(raw[8:10].astype(int))==world["r98_op"]
    assert sorted(np.rint(raw[4:8]*3).astype(int))==[0,1,2,3]
    assert raw[10]==float(action==r99.r85.CALIBRATE)
    assert raw[11]==float(action==r99.r85.EXPLOIT_POS)
    assert raw[13]==0.
    np.testing.assert_array_equal(raw[:4],slots[:4,10])
    altered=copy.deepcopy(world)
    altered["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=35.
    x,y,changed_reward=r99._observed(altered,action)
    np.testing.assert_array_equal(x,slots)
    np.testing.assert_array_equal(y,raw)
    assert abs(changed_reward-reward)>1.
    with pytest.raises(PermissionError,match="R99_ONLY_INDEPENDENT_NONSCORING_UNIT_WORLD"):
        r99.unit_world(12)


def test_R99_same_world_actual_program_change_not_just_role_swap():
    world,_=r99.unit_world(9)
    world["factors_a"]=np.array([1,1,-1,1],np.int64)
    world["r98_roles"]=r99.HELDOUT_BINDING
    world["r98_op"]=(1,1)
    world["law_a"]=r99.r98.law_of(world["factors_a"],world["r98_roles"],world["r98_op"])
    switched=copy.deepcopy(world)
    switched["r98_op"]=(0,1)
    switched["law_a"]=r99.r98.law_of(switched["factors_a"],switched["r98_roles"],(0,1))
    assert world["law_a"]!=switched["law_a"]
    before,_,r0=r99._observed(world,r99.r85.EXPLOIT_POS)
    after,_,r1=r99._observed(switched,r99.r85.EXPLOIT_POS)
    np.testing.assert_array_equal(before[:4,9:12],after[:4,9:12])
    assert not np.array_equal(before[:,8],after[:,8])
    assert abs(r0-r1)>1.


def test_R99_independent_derangement_same_public_action_time_without_labels():
    n=r99.COUNT["VALID"]
    raw=np.zeros((n,14),np.float32)
    act=np.array([r99.chosen(r99.SEEDS[0],i) for i in range(n)],np.int8)
    raw[:,12]=np.where(np.arange(n)%2==0,.3,.9)
    raw[:,10]=(act==r99.r85.CALIBRATE)
    raw[:,11]=(act==r99.r85.EXPLOIT_POS)
    p=r99.derange(raw,act,r99.SEEDS[0],"VALID")
    assert len(p)==n and np.all(p!=np.arange(n))
    np.testing.assert_array_equal(raw[p,12],raw[:,12])
    np.testing.assert_array_equal(act[p],act)
    np.testing.assert_array_equal(p,r99.derange(raw,act,r99.SEEDS[0],"VALID"))
    with pytest.raises(PermissionError,match="R99_AUX_DERANGEMENT_ONLY_FRESH_FACTUAL_PANEL"):
        r99.derange(raw,act,r99.SEEDS[0],"R98_DEV_COMPOSE")
    law=np.where(np.arange(n)%4<2,1,-1).astype(np.int8)
    with pytest.raises(RuntimeError,match="R99_AUX_SHUFFLE_PRIVATE_LAW_ALIGNMENT_INVALID"):
        r99._audit(law,raw,act,np.arange(n))


def test_R99_all_four_native_microfeedback_arms_identical_before_training():
    torch.set_num_threads(1)
    models=r99._models(r99.SEEDS[0])
    assert set(models)==set(r99.ARMS)
    core=dict(models[r99.NATIVE_RAW].named_parameters())
    baseline=dict(models[r99.MAIN_ONLY].named_parameters())
    for arm in r99.MICRO_ARMS:
        own=dict(models[arm].core.named_parameters())
        assert own.keys()==core.keys()
        assert all(torch.equal(own[k],core[k]) for k in core)
        p=dict(models[arm].named_parameters())
        assert p.keys()==baseline.keys()
        assert all(torch.equal(p[k],baseline[k]) for k in baseline)
    assert sum(p.numel() for p in models[r99.MAIN_ONLY].parameters())==sum(
        p.numel() for p in models[r99.FACTUAL].parameters())
    assert sum(p.numel() for p in models[r99.FACTUAL].parameters())==sum(
        p.numel() for p in models[r99.DERANGED].parameters())


def test_R99_intermediate_factual_heads_backprop_through_NATIVE_stem_ssm_attention_ttt():
    torch.set_num_threads(1)
    model=r99._models(r99.SEEDS[0])[r99.FACTUAL]
    slots=[];raw=[]
    for i in (0,1,9,10):
        w,a=r99.unit_world(i)
        x,f,_=r99._observed(w,a)
        slots.append(x);raw.append(f)
    t=torch.from_numpy(np.stack(slots));f=torch.from_numpy(np.stack(raw))
    target=torch.tensor([1.,-1.,1.,-1.])
    main,aux=model(t,f)
    assert main.shape==(4,) and aux.shape==(4,4)
    loss=F.mse_loss(main,target)+r99.AUX_WEIGHT*F.mse_loss(
        aux,target.unsqueeze(1).expand_as(aux))
    loss.backward()
    assert float(model.adapter.event[0].weight.grad.abs().sum())>0
    assert float(model.adapter.program[0].weight.grad.abs().sum())>0
    assert float(model.core.stem.weight.grad.abs().sum())>0
    assert float(model.core.head.weight.grad.abs().sum())>0
    assert all(float(h.weight.grad.abs().sum())>0 for h in model.micro)
    assert model.core.ssm is not None and model.core.ttt is not None
    assert model.core.attn is not None and model.core.depth is not None


def test_R99_each_arm_lawful_factual_forward_and_gradient_shape():
    torch.set_num_threads(1)
    models=r99._models(r99.SEEDS[0])
    obs=[]
    for i in range(4):
        w,a=r99.unit_world(i)
        obs.append(r99._observed(w,a)[0])
    seq=torch.from_numpy(np.stack(obs))
    raw=np.stack([r99.r98._raw(x) for x in obs])
    for name,model in models.items():
        f=r99.r98._canonical(raw) if name==r99.UPPER else raw
        out=r99._forward(model,name,seq,torch.from_numpy(f))
        assert out.shape==(4,) and bool(torch.isfinite(out).all())
    with pytest.raises(RuntimeError,match="R99_NATIVE_PUBLIC_EVENT_GEOMETRY_DRIFT"):
        models[r99.FACTUAL](seq[:2],torch.from_numpy(raw))


def test_R99_precommitted_mechanism_thresholds_and_no_scientific_promotion():
    f=r99.frozen()
    t=f["evaluation"]["hypothesis_checks"]
    assert len(t)==12
    assert t["factual_micro_minus_main_only_DEV_reward_min"]==.30
    assert t["factual_micro_minus_deranged_DEV_reward_min"]==.30
    assert t["factual_micro_minus_stem_only_DEV_reward_min"]==.15
    assert t["factual_micro_minus_original_native_DEV_reward_min"]==.5
    assert t["factual_micro_minus_best_fully_trained_conventional_DEV_reward_min"]==.1
    assert t["programmed_upper_minus_factual_micro_DEV_reward_max"]==.35
    assert t["factual_micro_true_law_accuracy_min"]==.8
    assert t["shifted_cold_factual_micro_minus_main_only_reward_min"]==.25
    assert t["paired_bootstrap_lower95_factual_micro_minus_main_only_min"]==.1
    assert t["program_change_true_law_flip_fraction_min"]==.2
    assert t["program_change_factual_micro_flip_when_law_changes_min"]==.65
    assert t["program_change_true_law_accuracy_both_min"]==.8
    assert f["execution"]["no_new_workflow_or_scheduler_or_runner_or_controller"]
    assert f["evaluation"]["independent_scientific_pass"] is False
