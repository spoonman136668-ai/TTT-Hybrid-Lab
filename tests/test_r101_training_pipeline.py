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
