"""F8-R5: proposed real task boundaries tested only on fabricated summaries."""
import inspect
import pytest
from experiments import p85mr_f8r5_owner_science_trial_proposal as r5


def fabricated_panel(*, qualified_profile=1):
    rows = []
    for p in range(3):
        for policy in r5.CALIBRATION_POLICIES:
            if p == qualified_profile:
                values = ([16, 22, 80, 88, 96, 96, 96, 96]
                          if policy == r5.CALIBRATION_POLICIES[0]
                          else [5, 7, 8, 13, 15, 25, 40, 96])
            else:
                values = ([96]*8 if policy == r5.CALIBRATION_POLICIES[0]
                          else [15]*8)
            for i, v in enumerate(values):
                rows.append({
                    "classification": "FABRICATED_DEV_ONLY",
                    "profile_index": p, "policy": policy, "episode_index": i,
                    "prospective_seed": r5.session_seed(
                        "difficulty_calibration", i, profile_index=p,
                        policy=policy),
                    "steps": v, "terminated": v < 96, "truncated": v == 96})
    return rows


def test_pinned_source_proposal_and_zero_authority():
    f = r5.locked()
    assert f["stage"] == "UNAPPROVED_SOURCE_NEUTRAL_PROSPECTIVE_CONTRACT_ONLY"
    assert f["authority"]["source_FirstUse"] is False
    assert f["authority"]["calibration_FirstUse"] is False
    assert f["authority"]["heldout_FirstUse"] is False
    assert f["authority"]["new_scheduled_task"] is False
    assert f["historical_exclusion"]["spent_F7_actions"] == 12288
    assert f["historical_exclusion"]["prior_linear7_RETROSPECTIVE_only"] is True
    receipt = r5.proposed_seed_receipt()
    assert receipt["calibration_sessions_proposed"] == 48
    assert receipt["fit_and_selection_and_test_worlds_proposed"] == 96
    assert receipt["new_effects_executed"] == 0
    assert receipt["scientific_pass"] is False


def test_fabricated_calibration_prefers_first_predeclared_admissible_profile():
    result = r5.diagnostic_on_fabricated_calibration(fabricated_panel())
    assert result["first_qualifying_profile_index"] == 1
    assert result["profile_diagnostics"][1]["fabricated_admission_result"]
    assert not result["profile_diagnostics"][0]["fabricated_admission_result"]
    assert not result["profile_diagnostics"][2]["fabricated_admission_result"]
    assert not result["original_task_qualified"]
    assert result["original_source_actions"] == 0
    assert not result["scientific_pass"]


def test_no_profile_qualifies_stops_without_retuning():
    result = r5.diagnostic_on_fabricated_calibration(fabricated_panel(
        qualified_profile=None))
    assert result["first_qualifying_profile_index"] is None
    assert result["dev_stop_code"] == "NO_FABRICATED_PROFILE_PASSED"
    assert not result["firstuse_authorized"]


def test_unauthenticated_nonfactual_and_replayed_fixtures_denied():
    rows = fabricated_panel()
    for change in ({"classification": "REAL_SOURCE"},
                   {"prospective_seed": rows[0]["prospective_seed"]+1},
                   {"steps": 0}, {"steps": 97},
                   {"steps": float("nan")}, {"steps": True},
                   {"terminated": False, "truncated": False},
                   {"episode_index": True}):
        corrupted = [dict(row) for row in rows]
        corrupted[0].update(change)
        with pytest.raises(ValueError, match="F8R5_INVALID_FABRICATED_ROW"):
            r5.diagnostic_on_fabricated_calibration(corrupted)
    with pytest.raises(ValueError, match="F8R5_INCOMPLETE_FABRICATED_PANEL"):
        r5.diagnostic_on_fabricated_calibration(rows[:-1])
    duplicate = [dict(row) for row in rows]
    duplicate[1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="F8R5_INVALID_FABRICATED_ROW"):
        r5.diagnostic_on_fabricated_calibration(duplicate)


def test_original_source_dual_termination_and_truncation_at_exact_horizon():
    # Pinned CartPoleWindSCM.step sets truncated=True at step 96 even if
    # underlying CartPole physics also sets terminated=True.
    rows = fabricated_panel()
    exact = next(i for i, row in enumerate(rows) if row["steps"] == 96)
    rows[exact] = dict(rows[exact], terminated=True, truncated=True)
    admitted = r5.diagnostic_on_fabricated_calibration(rows)
    assert admitted["first_qualifying_profile_index"] == 1
    assert admitted["original_source_actions"] == 0
    assert admitted["firstuse_authorized"] is False
    # The same dual flags before the horizon are not valid.
    early = next(i for i, row in enumerate(rows) if row["steps"] < 96)
    rows[early] = dict(rows[early], terminated=True, truncated=True)
    with pytest.raises(ValueError, match="F8R5_INVALID_FABRICATED_ROW"):
        r5.diagnostic_on_fabricated_calibration(rows)


def test_reserved_calibration_fit_selection_test_seeds_strictly_disjoint():
    seeds = set()
    for p in range(3):
        for policy in r5.CALIBRATION_POLICIES:
            for idx in range(8):
                seed = r5.session_seed("difficulty_calibration", idx,
                                       profile_index=p, policy=policy)
                assert seed not in seeds
                seeds.add(seed)
    for name, count in (("fit", 64), ("fit_selector", 16),
                        ("never_tune_on_test", 16)):
        for idx in range(count):
            seed = r5.session_seed(name, idx)
            assert seed not in seeds
            seeds.add(seed)
    assert len(seeds) == 144
    with pytest.raises(ValueError, match="UNREGISTERED_OR_UNAPPROVED_COHORT"):
        r5.session_seed("unfamiliar_family_reserved_not_defined", 0)
    with pytest.raises(ValueError, match="UNREGISTERED_OR_UNAPPROVED_COHORT"):
        r5.session_seed("never_tune_on_test", 16)
    with pytest.raises(ValueError, match="UNREGISTERED_SESSION"):
        r5.session_seed("fit", True)
    with pytest.raises(ValueError, match="UNREGISTERED_CALIBRATION_SCOPE"):
        r5.session_seed("difficulty_calibration", 0, profile_index=3,
                         policy=r5.CALIBRATION_POLICIES[0])


def test_module_has_no_original_source_or_owner_execution_path():
    with pytest.raises(PermissionError, match="OWNER_SCOPED_FIRSTUSE_NOT_AUTHORIZED"):
        r5.require_real_source_firstuse()
    source = inspect.getsource(r5)
    for forbidden in ("from causal_gym", "import causal_gym", "env.reset(",
                      "env.do(", "workflow_dispatch(", "subprocess.run(",
                      "torch.load(", "requests.get(", "urlopen(",
                      "start_task(", "optimizer.step("):
        assert forbidden not in source



def test_f8r14_random_action_contract_is_frozen_and_no_firstuse():
    contract = r5.fixed_seed_random_contract()
    assert contract["classification"] == "SOURCE_NEUTRAL_DEV_ONLY"
    assert contract["scope"]["environment_rng_independent"] is True
    assert contract["scope"]["unpaired_calibration_world_seeds"] is True
    assert contract["authority"]["owner_firstuse"] is False
    assert contract["authority"]["scientific_pass"] is False
    assert r5.proposed_seed_receipt()["new_effects_executed"] == 0


def test_f8r14_stateless_random_action_vectors_and_scope():
    for case in r5.fixed_seed_random_contract()["golden_test_vectors"]:
        actual = r5.fixed_seed_random_action(
            case["profile_index"], case["episode_index"], case["step_index"])
        assert type(actual) is int and actual == case["expected_action"]
    sequences = {}
    for profile in range(3):
        for episode in range(8):
            sequence = tuple(r5.fixed_seed_random_action(profile, episode, step)
                             for step in range(96))
            assert set(sequence) == {0, 1}
            sequences[(profile, episode)] = sequence
    assert len(sequences) == 24
    assert len(set(sequences.values())) == 24
    for (profile, episode), sequence in sequences.items():
        assert sequence == tuple(
            r5.fixed_seed_random_action(profile, episode, step)
            for step in reversed(range(96)))[::-1]
    for profile, episode, step in (
        (0, 0, -1), (0, 0, 96), (0, 0, True),
        (3, 0, 0), (0, 8, 0), (True, 0, 0),
        (0, False, 0)):
        with pytest.raises(ValueError, match="F8R"):
            r5.fixed_seed_random_action(profile, episode, step)


def test_f8r14_action_helper_does_not_touch_environment_or_firstuse():
    helper = inspect.getsource(r5.fixed_seed_random_action)
    for forbidden in ("np_random", "env.action(", "env.do(",
                      "env.reset(", "causal_gym", "random.seed(",
                      "ctf_do(", "GITHUB_TOKEN", "workflow_dispatch("):
        assert forbidden not in helper
    assert r5.fixed_seed_random_contract()["authority"]["external_source_effects"] is False



def test_f8r15_scripted_feedback_frozen_and_owner_denied():
    f = r5.frozen_scripted_feedback_contract()
    assert f["stage"] == "FROZEN_BEFORE_F8R15_IMPLEMENTATION_AND_TEST"
    assert f["authority"]["owner_firstuse"] is False
    assert f["authority"]["scientific_pass"] is False
    assert f["frozen_policy"]["random_policy_unchanged"] is True
    assert r5.proposed_seed_receipt()["new_effects_executed"] == 0


@pytest.mark.parametrize("state,expected", [
    ([0., 0., 0.1, 0.], 1),
    ([0., 0., -0.1, 0.], 0),
    ([0., 0., 0.05, -0.1], 0),
    ([0., 0., 0.05, 0.01], 1),
    ([1.5, -99., 0., 0.], 0),
    ([0., 99., 0., 0.001], 1),
])
def test_f8r15_scripted_balancing_sign_uses_only_public_pole(state, expected):
    assert r5.frozen_scripted_feedback_action(0, 0, 0, state) == expected
    assert r5.frozen_scripted_feedback_action(2, 7, 95, state) == expected
    assert r5.frozen_scripted_feedback_action(1, 3, 42, tuple(state)) == expected


@pytest.mark.parametrize("profile,episode,step,state", [
    (3, 0, 0, [0., 0., 0., 0.]),
    (0, 8, 0, [0., 0., 0., 0.]),
    (True, 0, 0, [0., 0., 0., 0.]),
    (0, False, 0, [0., 0., 0., 0.]),
    (0, 0, -1, [0., 0., 0., 0.]),
    (0, 0, 96, [0., 0., 0., 0.]),
    (0, 0, True, [0., 0., 0., 0.]),
    (0, 0, 0, [0., 0., 0.]),
    (0, 0, 0, [0., 0., 0., 0., 0.]),
    (0, 0, 0, [0., 0., True, 0.]),
    (0, 0, 0, [0., 0., float("nan"), 0.]),
    (0, 0, 0, [0., 0., float("inf"), 0.]),
    (0, 0, 0, "0,0,0,0"),
])
def test_f8r15_invalid_scope_or_state_denied(profile,episode,step,state):
    with pytest.raises(ValueError,match="F8R"):
        r5.frozen_scripted_feedback_action(profile,episode,step,state)


def test_f8r15_scripted_policy_has_no_wind_or_owner_source_effect():
    code = inspect.getsource(r5.frozen_scripted_feedback_action)
    for forbidden in ("np_random", "env.action(", "env.reset(", "env.do(",
                      "ctf_do(", "causal_gym", "current_wind",
                      "subprocess.run(", "workflow_dispatch("):
        assert forbidden not in code
    assert r5.frozen_scripted_feedback_action(
        0, 0, 3, [10., 20., 0.1, -0.01]) == 1
