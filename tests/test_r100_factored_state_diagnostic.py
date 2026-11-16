"""R100 independent unit-only source, factorization, feedback and owner tests.

NO FIT/VALID/DEV_COMPOSE/OOD/SWITCH original physical science panel generated
in pytest; new original source first use exclusively gated by existing G4.
"""
import copy
import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from experiments import r100_factored_state_diagnostic as r100


def test_frozen_owner_scope_source_and_original_spent_parent():
    f=r100.frozen()
    assert r100.r85.git_blob(r100.FREEZE.read_bytes())==r100.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVE_FREEZE_BEFORE_R100_CODE_TESTS_OR_SOURCE_FIRST_USE"
    assert f["schema"]==r100.SCHEMA
    assert f["parent"]["qualified_r99_sealed_SHA"]==r100.PARENT
    assert f["parent"]["r99_spent_SHA"]==r100.r99.ORIGINAL_R99_SPENT_SHA
    assert f["parent"]["r99_spent_run"]==r100.r99.ORIGINAL_R99_SPENT_RUN_ID
    assert f["parent"]["prior_scored_sources_or_weights_reused"] is False
    assert f["source"]["seeds"]==list(r100.SEEDS)
    assert f["source"]["counts"]==r100.COUNT
    assert f["source"]["salts"]=={**r100.SALT,"RETURN_SHUFFLE":r100.RETURN_SHUFFLE_SALT,
        "BATCH":r100.BATCH_SALT,"BOOTSTRAP":r100.BOOTSTRAP_SALT,
        "UNIT_ONLY":r100.UNIT_SALT}
    assert f["arms"]==list(r100.ARMS)
    assert len(r100.ARMS)==11
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["authority"]["CPU_TEST_ONLY_scoped_original_first_use"] is True
    assert not any(v for k,v in f["authority"].items()
                   if k!="CPU_TEST_ONLY_scoped_original_first_use")


def test_frozen_physical_panels_deny_prior_scored_and_unadmitted_first_use():
    for old in ("R99_DEV_COMPOSE","R98_DEV_COMPOSE","R97_DEV_HELDOUT",
                "PROTECTED","R85_FIT"):
        with pytest.raises(PermissionError,match="R100_OLD_SPENT_OR_PROTECTED_SOURCE_FORBIDDEN"):
            r100._world(old,r100.SEEDS[0],0,"fake")
    for panel in r100.SALT:
        with pytest.raises(PermissionError,match="R100_NEW_SCIENCE_SOURCE_NOT_ADMITTED"):
            r100._world(panel,r100.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R100_UNFROZEN_SEED_OR_WORLD_INDEX"):
        r100._world("DEV_COMPOSE",42,0,"fake")
    with pytest.raises(PermissionError,match="R100_UNFROZEN_SEED_OR_WORLD_INDEX"):
        r100._world("DEV_COMPOSE",r100.SEEDS[0],256,"fake")
    with pytest.raises(PermissionError,match="R100_TRAIN_ONLY_NEW_FIT_VALID_NOT_DEV"):
        r100._factual("DEV_COMPOSE",r100.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R100_EVALUATOR_ONLY_NEW_HELDOUT_PANELS"):
        r100._scored("FIT",r100.SEEDS[0],"fake")
    if os.getenv("GITHUB_JOB")!="r100-factored-native":
        with pytest.raises(PermissionError,match="R100_GITHUB_HOSTED_CPU_ORIGINAL_FIRST_USE_ONLY"):
            r100._admit(r100.SEEDS[0])


def test_index_only_training_programs_selected_actions_and_heldout_role_are_frozen():
    report=r100.design_preflight()
    for seed in r100.SEEDS:
        for panel,n in r100.COUNT.items():
            seen={}
            for i in range(n):
                b=(i%2,r100.chosen(seed,i))
                seen[b]=seen.get(b,0)+1
                if panel in ("FIT","VALID","DEV_SEEN"):
                    assert r100.program(panel,i) in r100.TRAIN_PROGRAMS
                    assert r100.program(panel,i)!=r100.COMPOSE_PROGRAM
                    assert r100.roles(panel,i) in r100.TRAIN_BINDINGS
                else:
                    assert r100.program(panel,i)==r100.COMPOSE_PROGRAM
                    assert r100.roles(panel,i)==r100.HELDOUT_BINDING
            assert len(seen)==4 and set(seen.values())=={n//4}
            assert report[str(seed)][panel]=={"four_action_time_cells":4,"per_cell":n//4}
    with pytest.raises(PermissionError,match="R100_UNREGISTERED_FACTUAL_ACTION"):
        r100.chosen(-11,0)


@pytest.mark.parametrize("index",[0,1,2,8,9,10,11])
def test_only_public_preaction_A_factor_cues_task_tags_and_selected_action(index):
    world,choice=r100.unit_world(index)
    assert world["law_a"]==r100.r98.law_of(
        world["factors_a"],world["r98_roles"],world["r98_op"])
    slots,raw,return_value=r100._public(world,choice)
    assert slots.shape==(5,13) and slots.dtype==np.float32
    assert raw.shape==(14,) and raw.dtype==np.float32
    assert tuple(raw[8:10].astype(int))==world["r98_op"]
    assert sorted(np.rint(raw[4:8]*3).astype(int).tolist())==[0,1,2,3]
    assert raw[10]==float(choice==r100.r85.CALIBRATE)
    assert raw[11]==float(choice==r100.r85.EXPLOIT_POS)
    assert raw[13]==0.
    np.testing.assert_array_equal(raw[:4],slots[:4,10])
    changed=copy.deepcopy(world)
    changed["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=50.
    x,y,new_reward=r100._public(changed,choice)
    np.testing.assert_array_equal(x,slots)
    np.testing.assert_array_equal(y,raw)
    assert abs(new_reward-return_value)>1.
    with pytest.raises(PermissionError,match="R100_INDEPENDENT_UNIT_FIXTURE_ONLY"):
        r100.unit_world(12)


def test_explicit_engineered_factor_algebra_is_not_autonomous_law_learning():
    torch.set_num_threads(1)
    engine=r100.FactorAlgebra()
    with torch.no_grad():
        engine.gain.fill_(9.)
        engine.bias.zero_()
        engine.selector_sharpness.fill_(9.)
    # Four public noisy factor responses with unseen role positions.
    raw=np.zeros((4,14),np.float32)
    raw[:,:4]=np.array([1.,1.,1.,-1.],np.float32)
    raw[:,4:8]=r100.r98.tag_for(r100.HELDOUT_BINDING)/3.
    raw[:,8:10]=np.asarray([[0,0],[0,1],[1,0],[1,1]],np.float32)
    z=engine(torch.from_numpy(raw)).detach().numpy()
    expected=np.asarray([
        r100.r98.law_of(np.asarray([1,1,1,-1],np.int64),
                       r100.HELDOUT_BINDING,(0,0)),
        r100.r98.law_of(np.asarray([1,1,1,-1],np.int64),
                       r100.HELDOUT_BINDING,(0,1)),
        r100.r98.law_of(np.asarray([1,1,1,-1],np.int64),
                       r100.HELDOUT_BINDING,(1,0)),
        r100.r98.law_of(np.asarray([1,1,1,-1],np.int64),
                       r100.HELDOUT_BINDING,(1,1))],np.float32)
    np.testing.assert_allclose(z,expected,atol=.005)
    # The operator algebra is intentionally programmed in source.
    assert "EXPLICIT ENGINEERED OPERATOR PRIOR" in r100.frozen()["architecture"]["factorized_algebra"]


def test_true_opcode_mask_blocks_native_instruction_channel_and_raw_flags():
    world,a=r100.unit_world(9)
    slots,raw,_=r100._public(world,a)
    masked_slots,masked_raw=r100._mask_program(slots[None,:],raw[None,:])
    np.testing.assert_array_equal(masked_slots[:,:,:8],slots[None,:,:8])
    np.testing.assert_array_equal(masked_slots[:,:,9:],slots[None,:,9:])
    np.testing.assert_array_equal(masked_raw[:,:8],raw[None,:8])
    np.testing.assert_array_equal(masked_raw[:,10:],raw[None,10:])
    assert np.all(masked_raw[:,8:10]==0.)
    np.testing.assert_array_equal(masked_slots[0,:4,8],
        np.rint(raw[4:8]*3)/15.)


def test_factual_outcome_sign_shuffle_uses_only_public_time_and_chosen_action():
    n=r100.COUNT["VALID"]
    raw=np.zeros((n,14),np.float32)
    actions=np.array([r100.chosen(r100.SEEDS[0],i) for i in range(n)],np.int8)
    raw[:,12]=np.where(np.arange(n)%2==0,.25,.9)
    raw[:,10]=(actions==r100.r85.CALIBRATE)
    raw[:,11]=(actions==r100.r85.EXPLOIT_POS)
    perm=r100._derange(raw,actions,r100.SEEDS[0],"VALID")
    assert perm.shape==(n,) and np.all(perm!=np.arange(n))
    np.testing.assert_array_equal(actions[perm],actions)
    np.testing.assert_array_equal(raw[perm,12],raw[:,12])
    np.testing.assert_array_equal(perm,r100._derange(raw,actions,r100.SEEDS[0],"VALID"))
    with pytest.raises(PermissionError,match="R100_AUX_RETURN_SHUFFLE_UNFROZEN_PANEL"):
        r100._derange(raw,actions,r100.SEEDS[0],"R99_DEV_COMPOSE")
    law=np.where(np.arange(n)%4<2,1,-1).astype(np.int8)
    with pytest.raises(RuntimeError,match="R100_RETURN_SHUFFLE_PRIVATE_LAW_CONTROL_INVALID"):
        r100._audit(law,raw,actions,np.arange(n))


def test_identical_native_cores_and_equivalent_initial_factorized_variants():
    torch.set_num_threads(1)
    models=r100._models(r100.SEEDS[0])
    assert set(models)==set(r100.ARMS)
    cores=[models[r100.NATIVE_RAW],models[r100.UNSTRUCTURED].core]
    cores.extend(models[arm].core for arm in r100.FACTORED_INTERNAL_ARMS)
    cores.append(models[r100.EXTERNAL].core)
    states=[dict(c.named_parameters()) for c in cores]
    assert all(x.keys()==states[0].keys() for x in states)
    assert all(torch.equal(states[0][k],z[k])
               for z in states[1:] for k in states[0])
    baseline=dict(models[r100.NOAUX].named_parameters())
    for arm in (r100.AUX,r100.SHUFFLED,r100.OPMASK):
        a=dict(models[arm].named_parameters())
        assert a.keys()==baseline.keys()
        assert all(torch.equal(a[k],baseline[k]) for k in baseline)
    assert sum(p.numel() for p in models[r100.NOAUX].parameters())==sum(
        p.numel() for p in models[r100.AUX].parameters())


def test_native_factorization_gradients_pass_inside_stem_ssm_attn_ttt():
    torch.set_num_threads(1)
    model=r100._models(r100.SEEDS[0])[r100.AUX]
    slots=[];raw=[]
    for i in (0,1,9,10):
        w,a=r100.unit_world(i)
        x,y,_=r100._public(w,a)
        slots.append(x);raw.append(y)
    x=torch.from_numpy(np.stack(slots))
    y=torch.from_numpy(np.stack(raw))
    main,aux=model(x,y)
    assert main.shape==(4,) and aux.shape==(4,)
    actual=torch.tensor([5.,-5.,5.,-5.])
    loss=F.mse_loss(main,actual)+r100.LAMBDA*F.binary_cross_entropy_with_logits(
        aux,(actual>0).float())
    loss.backward()
    assert float(model.algebra.gain.grad.abs().sum())>0
    assert float(model.projection[0].weight.grad.abs().sum())>0
    assert float(model.sign_head[0].weight.grad.abs().sum())>0
    assert float(model.core.stem.weight.grad.abs().sum())>0
    assert float(model.core.head.weight.grad.abs().sum())>0
    assert model.core.ssm is not None and model.core.ttt is not None


def test_all_eleven_arms_valid_original_public_input_no_private_teacher():
    torch.set_num_threads(1)
    models=r100._models(r100.SEEDS[0])
    seq=[];features=[]
    for i in range(4):
        w,a=r100.unit_world(i)
        x,y,_=r100._public(w,a)
        seq.append(x);features.append(y)
    slots=np.stack(seq);raw=np.stack(features)
    for name,model in models.items():
        src={"slots":slots,"raw":raw,"canonical":r100.r98._canonical(raw)}
        x,y=r100._arrays(src,name)
        out=r100._forward(model,name,torch.from_numpy(x),torch.from_numpy(y))
        assert out.shape==(4,) and bool(torch.isfinite(out).all())
    assert r100.frozen()["architecture"]["equal_factual_information_actions_updates"]


def test_frozen_separate_structural_feedback_and_native_specific_verdicts():
    e=r100.frozen()["evaluation"]
    assert len(e["structural_checks"])==10
    assert len(e["feedback_checks"])==4
    assert len(e["native_specific_checks"])==2
    assert e["structural_checks"]["noaux_true_law_accuracy_min"]==.8
    assert e["feedback_checks"]["factual_aux_minus_noaux_DEV_reward_min"]==.2
    assert e["native_specific_checks"]["noaux_minus_external_same_algebra_DEV_reward_min"]==.15
    assert e["independent_scientific_pass"] is False
    assert r100.frozen()["architecture"]["precise_same_structure_noaux_feedback_shuffled_opcode"]


def test_original_R100_scored_source_spent_and_all_new_SHA_run_replays_denied(monkeypatch):
    assert r100.ORIGINAL_R100_SPENT_SHA=="e231895bddfaf6d41ac6d3ffcce8858eaaf4b784"
    assert r100.ORIGINAL_R100_SPENT_RUN_ID=="43851236967"
    env={
        "GITHUB_ACTIONS":"true",
        "GITHUB_EVENT_NAME":"pull_request",
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_HEAD_REF":r100.BRANCH,
        "GITHUB_JOB":"r100-factored-native",
        "R100_SEED":str(r100.SEEDS[0]),
        "RUNNER_OS":"Linux",
        "RUNNER_ENVIRONMENT":"github-hosted",
        "GITHUB_RUN_ATTEMPT":"1",
        "GITHUB_REF":"refs/pull/296/merge",
        "EXPECTED_CANDIDATE_SHA":"f"*40,
        "GITHUB_RUN_ID":"43899999999",
    }
    for k,v in env.items():
        monkeypatch.setenv(k,v)
    with pytest.raises(PermissionError,match="R100_SPENT_ORIGINAL_SOURCE_REPLAY_DENIED"):
        r100._admit(r100.SEEDS[0])
    for panel in ("FIT","VALID","DEV_SEEN","DEV_COMPOSE","OOD_SHIFT","PROGRAM_SWITCH"):
        with pytest.raises(PermissionError,match="R100_NEW_SCIENCE_SOURCE_NOT_ADMITTED"):
            r100._world(panel,r100.SEEDS[0],0,"f"*40)
