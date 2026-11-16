"""R101 original scientific pipeline unit-only audit; no scored source use."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101
from experiments import r101_training_pipeline as pipe


def test_protocol_exact_frozen_panels_training_and_no_science_authority():
    f = pipe.protocol()
    assert tuple(f["source"]["counts"]) == pipe.ALL_PANELS
    assert pipe.TRAIN_PANELS == ("FIT", "VALID")
    assert len(pipe.SCORING_PANELS) == 5
    assert pipe.STEPS == 3072 and pipe.BATCH == 16
    assert f["training"]["all_nine_arms_fully_trained_before_any_DEV_first_use"]
    assert not f["authority"]["protected_scored_first_use"]


@pytest.mark.parametrize("panel", pipe.ALL_PANELS)
def test_every_original_scored_source_denied_before_first_use(panel):
    with pytest.raises(PermissionError,match="R101_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.episode(panel,r101.SEEDS[0],0,"b"*40,
                     paired=(panel in pipe.SCORING_PANELS))
    with pytest.raises(PermissionError,match="R101_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.source(panel,r101.SEEDS[0],"b"*40)


@pytest.mark.parametrize("panel", ["DEV_UNKNOWN","DELAYED_COLD","OOD_SHIFT","REVERSAL"])
def test_unseen_operator_family_sealed_never_known_during_scoring(panel):
    assert pipe.operator_family(panel,0,r101.SUPPORT) in r101.UNSEEN
    assert pipe.operator_family(panel,1,r101.SUPPORT) in r101.UNSEEN


def test_reversal_changes_operator_after_three_factual_interventions():
    old = pipe.operator_family("REVERSAL",0,0)
    final = pipe.operator_family("REVERSAL",0,r101.SUPPORT)
    assert old in r101.UNSEEN and final in r101.UNSEEN
    assert old != final
    assert all(pipe.operator_family("REVERSAL",0,j) == old for j in range(3))
    assert all(pipe.operator_family("REVERSAL",0,j) == final for j in range(3,7))


@pytest.mark.parametrize("panel", ["FIT","VALID","DEV_FAMILIAR"])
def test_train_and_familiar_holdout_never_leak_unknown_family(panel):
    for k in range(10):
        assert pipe.operator_family(panel,k,0) in r101.KNOWN


def test_precommitted_action_policy_exposes_three_of_each_and_one_query():
    for seed in r101.SEEDS:
        for i in (0,1,7):
            v = [pipe.chosen_action(seed,i,j) for j in range(6)]
            assert v.count(r85.CALIBRATE) == 3
            assert v.count(r85.EXPLOIT_POS) == 3
            assert pipe.chosen_action(seed,i,6) in (r85.CALIBRATE,r85.EXPLOIT_POS)
    with pytest.raises(PermissionError):
        pipe.chosen_action(0,0,0)


def test_shuffle_is_fixed_public_action_time_only_and_no_fixed_points():
    public = r101.unit_fixture(0)["public"]
    slots, actions = public["support_slots"], public["support_actions"]
    a = pipe.derange_public_support(actions,slots,r101.SEEDS[0],"FIT",0)
    b = pipe.derange_public_support(actions,slots,r101.SEEDS[0],"FIT",0)
    assert np.array_equal(a,b)
    assert not np.any(a == np.arange(6))
    np.testing.assert_array_equal(actions[a],actions)
    np.testing.assert_array_equal(slots[a,-1,12],slots[:,-1,12])


def _non_scoring_unit_records():
    fixtures = [r101.unit_fixture(i) for i in (0,1,2,3)]
    public = [x["public"] for x in fixtures]
    evaluation = [x["evaluator_only"] for x in fixtures]
    rewards = np.stack([x["support_rewards"] for x in public])
    shuffled = np.stack([x["support_rewards"][pipe.derange_public_support(
        x["support_actions"],x["support_slots"],r101.SEEDS[0],"FIT",i)]
                         for i,x in enumerate(public)])
    return {
        "support_slots":np.stack([x["support_slots"] for x in public]),
        "support_rewards":rewards,
        "support_rewards_shuffled":shuffled,
        "query0":np.stack([x["query0"] for x in public]),
        "query2":np.stack([x["query2"] for x in public]),
        "reward0":np.asarray([x["return0"] for x in evaluation],np.float32),
        "reward2":np.asarray([x["return2"] for x in evaluation],np.float32),
        "private_law":np.asarray([x["private_law"] for x in evaluation],np.int8),
        "cold":np.asarray([bool(i%2) for i in range(len(public))]),
    }


def test_unit_only_source_interfaces_all_nine_actions_and_evaluator_score():
    torch.set_num_threads(1)
    data = _non_scoring_unit_records()
    models = r101.make_models(r101.SEEDS[0])
    result = pipe.score(models,"DEV_UNKNOWN",data)
    assert set(result["metrics"]) == set(r101.ARMS)
    assert result["actual_physical_worlds"] == 4
    assert np.isfinite(result["oracle_mean"])
    for arm in r101.ARMS:
        assert result["choices"][arm].shape == (4,)
        assert np.isfinite(result["rewards"][arm]).all()
        assert 0 <= result["metrics"][arm]["private_true_law_action_agreement"] <= 1
    assert 0 <= result["role_order_permutation_action_agreement"] <= 1


def test_public_support_is_training_factual_and_not_evaluator_privileged():
    data = _non_scoring_unit_records()
    assert set(data) >= {"reward0","reward2","private_law"}
    fitting = dict(data)
    fitting["query"] = fitting.pop("query0")
    fitting["target"] = fitting.pop("reward0")
    fitting.pop("query2")
    fitting.pop("reward2")
    fitting.pop("private_law")
    model = r101.make_models(r101.SEEDS[0])[r101.NATIVE]
    indices = np.zeros((pipe.STEPS,pipe.BATCH),np.int64)
    # This is a four-row unit fixture, not the frozen 1024-row FIT cohort.
    with pytest.raises(RuntimeError,match="R101_FIT_COUNT_DRIFT"):
        pipe.fit_arm(model,r101.NATIVE,fitting,fitting,indices)
    with pytest.raises(PermissionError,match="R101_PROTECTED_EVALUATOR_ONLY_PANEL"):
        pipe.score({r101.NATIVE:model},"DEV_UNKNOWN",fitting)


def test_unit_only_paired_bootstrap_is_deterministic_and_not_science():
    a = np.array([1.,2.,3.,4.])
    b = np.array([0.,0.,0.,0.])
    assert pipe.lower_bootstrap(a,b,r101.SEEDS[0]) == pipe.lower_bootstrap(
        a,b,r101.SEEDS[0])
    with pytest.raises(ValueError):
        pipe.lower_bootstrap(a,b,777)


def test_original_scored_runner_remains_disabled_and_no_receipts_created():
    with pytest.raises(PermissionError,match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        pipe.original_ci(r101.SEEDS[0])



def test_pre_first_use_design_audit_quantifies_operator_nonidentifiability():
    out = pipe.non_scoring_design_audit()
    assert out["original_scored_data_accessed"] is False
    assert out["unit_and_combinatorial_only"] is True
    assert out["operator_signatures_evaluator_only"] is True
    assert out["frozen_first_use_granted"] is False
    assert out["six_supports_cannot_guarantee_family_identification"] is True
    nearest = out["unseen_vs_known"]["EXACTLY_ONE3"][0]
    assert nearest["known"] == "PARITY3"
    assert nearest["different_patterns_of_eight"] == 1
    assert nearest["probability_six_iid_supports_avoid_difference"] == pytest.approx(
        (7 / 8)**6, rel=1e-12)
    assert nearest["probability_query_is_different_after_undistinguishing_support"] == pytest.approx(
        (7 / 8)**6 / 8, rel=1e-12)
    majority = out["unseen_vs_known"]["MAJORITY3"][0]
    assert majority["known"] == "MUX3"
    assert majority["different_patterns_of_eight"] == 2
    assert majority["probability_six_iid_supports_avoid_difference"] == pytest.approx(
        (6 / 8)**6, rel=1e-12)
    assert out["science_thresholds_and_frozen_source_unchanged"]
    assert out["independent_scientific_pass"] is False


def test_design_audit_uses_no_original_physics_and_matches_frozen_panel_counts(monkeypatch):
    def deny_all_original_worlds(*_args, **_kwargs):
        raise AssertionError("R101_FIRST_USE_SOURCE_ILLEGALLY_CREATED_IN_AUDIT")
    monkeypatch.setattr(r101, "_world", deny_all_original_worlds)
    result = pipe.non_scoring_design_audit()
    expected = pipe.protocol()["source"]["counts"]
    assert {p:d["count"] for p,d in
            result["precommitted_panel_source_design"].items()} == expected
    for panel,values in result["precommitted_panel_source_design"].items():
        assert sum(values["family_counts"].values()) == expected[panel]
        assert sum(values["query_action_counts_all_three_seeds"].values()) == (
            expected[panel] * 3)
        assert values["six_support_actions_each_episode"] == (
            "3 CALIBRATE + 3 EXPLOIT_POS")



@pytest.mark.parametrize("family", ["MUX3", "MAJORITY3", "EXACTLY_ONE3"])
@pytest.mark.parametrize("action", [r85.CALIBRATE, r85.EXPLOIT_POS])
def test_public_role_relabel_matches_actual_physics(family, action):
    """Counterfactual semantics validated on independent UNIT_ONLY worlds."""
    import copy
    perm = pipe.ROLE_RELABEL
    old_slots, new_slots, old_returns, new_returns = [], [], [], []
    for event in range(r101.SUPPORT + 1):
        old_world = r101._unit_world(1, event, family)
        transformed = copy.deepcopy(old_world)
        factors = np.empty(4,dtype=np.int64)
        for original_role in range(4):
            factors[perm[original_role]] = old_world["factors_a"][original_role]
        transformed["factors_a"] = factors
        transformed["r101_roles"] = tuple(
            perm[original_role] for original_role in old_world["r101_roles"])
        transformed["probe"] = {
            t:(phase,perm[role] if phase=="A" else role,marker)
            for t,(phase,role,marker) in old_world["probe"].items()}
        transformed["a_probe_permutation"] = np.asarray(
            [perm[role] for role in old_world["a_probe_permutation"]],dtype=np.int64)
        transformed["law_a"] = r101._operator_truth(
            transformed["factors_a"], transformed["r101_roles"],family)
        assert old_world["law_a"] == transformed["law_a"]
        before,br = r101._public_view(old_world,action)
        after,ar = r101._public_view(transformed,action)
        old_slots.append(before)
        new_slots.append(after)
        old_returns.append(br)
        new_returns.append(ar)
    support = np.stack(old_slots[:r101.SUPPORT])[None]
    query = old_slots[-1][None]
    new_support,new_query = pipe.relabel_public_roles(support,query)
    np.testing.assert_array_equal(new_support[0],
                                  np.stack(new_slots[:r101.SUPPORT]))
    np.testing.assert_array_equal(new_query[0],new_slots[-1])
    np.testing.assert_array_equal(np.asarray(old_returns),
                                  np.asarray(new_returns))
    # Neither the reward nor the public action/timing rows change.
    np.testing.assert_array_equal(new_support[:,:,-1,:],support[:,:,-1,:])
    np.testing.assert_array_equal(new_query[:,-1,:],query[:,-1,:])


def test_scoring_role_relabel_keeps_feedback_actions_and_public_role_integrity():
    data = _non_scoring_unit_records()
    ix = np.array([0,1,2,3])
    original = pipe._inputs(data,ix,r101.NATIVE,action=r85.CALIBRATE,
                            permute=False)
    transformed = pipe._inputs(data,ix,r101.NATIVE,action=r85.CALIBRATE,
                               permute=True)
    assert torch.equal(original[1],transformed[1])
    assert not torch.equal(original[0],transformed[0])
    assert not torch.equal(original[2],transformed[2])
    np.testing.assert_array_equal(
        transformed[0].numpy()[:,:,-1,:],original[0].numpy()[:,:,-1,:])
    np.testing.assert_array_equal(
        transformed[2].numpy()[:,-1,:],original[2].numpy()[:,-1,:])
    for sample in transformed[0].numpy().reshape(-1,5,13):
        assert sorted(sample[:4,8].tolist()) == [0.,.25,.5,.75]
        np.testing.assert_array_equal(sample[:4,9],
                                      np.array([.25,.5,.75,1.],dtype=np.float32))
    with pytest.raises(ValueError,match="R101_PHYSICAL_ROLE_RELABEL_INPUT_INVALID"):
        pipe.relabel_public_roles(data["support_slots"][:1],data["query0"])



@pytest.mark.parametrize("arm,mode", [
    (r101.ROLE_SET, "set"),
    (r101.GRU, "gru"),
    (r101.MLP, "flat"),
    (r101.STATIC, "static"),
])
def test_conventional_comparators_have_no_dormant_trainable_parameters(arm, mode):
    """One REAL unit-only optimizer update, never training on frozen FIT/DEV."""
    torch.set_num_threads(1)
    model = r101.ConventionalValue(mode)
    fixture = r101.unit_fixture(2)
    public = fixture["public"]
    # One factual chosen query action; other query return stays evaluator-only.
    chosen = r85.CALIBRATE
    x = torch.from_numpy(public["support_slots"])[None]
    y = torch.from_numpy(public["support_rewards"])[None]
    query = torch.from_numpy(public["query0"])[None]
    selected_return = torch.tensor(
        [fixture["evaluator_only"]["return0"]], dtype=torch.float32)
    initial = {name: p.detach().clone() for name,p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    pred = model(x,y,query)
    assert pred.shape == (1,), arm
    loss = torch.nn.functional.mse_loss(pred, selected_return)
    assert bool(torch.isfinite(loss)), arm
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    named = dict(model.named_parameters())
    assert named, arm
    assert all(p.requires_grad and p.grad is not None
               and torch.isfinite(p.grad).all() for p in named.values()), arm
    # None of these baseline variants is allowed to hide substantial
    # dormant parameters behind a misleading overall capacity count.
    assert sum(p.numel() for p in named.values()) == sum(
        p.numel() for p in named.values() if p.grad is not None), arm
    optimizer.step()
    assert any(not torch.equal(initial[name],p) for name,p in named.items()),arm
    assert all(torch.isfinite(p).all() for p in named.values()), arm


def test_unknown_conventional_mode_rejected():
    with pytest.raises(ValueError,match="R101_CONVENTIONAL_MODE_NOT_REGISTERED"):
        r101.ConventionalValue("unsupported")
