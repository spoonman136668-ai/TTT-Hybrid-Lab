"""R101 development-only negatives. No FIT/DEV physical first use here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101


def test_freeze_sha_north_star_and_science_fences():
    f = r101.frozen()
    assert r85.git_blob(r101.FREEZE.read_bytes()) == r101.FREEZE_BLOB
    assert f["parent"]["r100_sealed_parent_SHA"] == r101.PARENT
    assert not f["parent"]["prior_scored_source_or_weights_reused"]
    assert f["source"]["support_factual_interventions"] == 6
    assert f["source"]["no_prior_cohort_replay"]
    assert set(f["source"]["known_function_families"]).isdisjoint(
        f["source"]["unseen_function_families"])
    assert len(f["arms"]) == 9
    assert f["evaluation"]["independent_seeds"] == 3
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert not f["authority"]["protected_scored_first_use"]
    assert not f["authority"]["official_GPU_first_use"]
    assert not f["authority"]["merge"]
    assert not f["authority"]["FORGE_integration"]


@pytest.mark.parametrize("panel", ["FIT", "VALID", "DEV_FAMILIAR", "DEV_UNKNOWN",
                                     "OOD_SHIFT", "DELAYED_COLD", "REVERSAL"])
def test_all_new_scored_panels_denied_even_with_plausible_sha(panel):
    with pytest.raises(PermissionError, match="R101_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r101._world(panel, r101.SEEDS[0], 0, "a" * 40)


@pytest.mark.parametrize("panel", ["R100_DEV_COMPOSE", "R99_FIT", "R85_CALIBRATION", "PROTECTED"])
def test_old_spent_and_protected_source_denied(panel):
    with pytest.raises(PermissionError, match="R101_OLD_SCORED_OR_PROTECTED_SOURCE_DENIED"):
        r101._world(panel, r101.SEEDS[0], 0, "a" * 40)


def test_unknown_seed_index_and_original_first_use_always_denied():
    with pytest.raises(PermissionError, match="R101_UNFROZEN_SOURCE_INDEX_OR_SEED"):
        r101._world("FIT", 42, 0)
    with pytest.raises(PermissionError, match="R101_UNFROZEN_SOURCE_INDEX_OR_SEED"):
        r101._world("DEV_UNKNOWN", r101.SEEDS[0], 128)
    with pytest.raises(PermissionError, match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r101._admit("original")
    with pytest.raises(PermissionError, match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r101.original_ci()


@pytest.mark.parametrize("family", r101.KNOWN + r101.UNSEEN)
def test_private_operator_is_evaluator_only_and_deterministic(family):
    factors = np.array([1, -1, 1, -1], dtype=np.int64)
    got = r101._operator_truth(factors, (0, 1, 2), family)
    assert got in (-1, 1)
    assert got == r101._operator_truth(factors.copy(), (0, 1, 2), family)
    with pytest.raises(ValueError, match="R101_UNKNOWN_PHYSICAL_OPERATOR"):
        r101._operator_truth(factors, (0, 1, 2), "FORBIDDEN")


@pytest.mark.parametrize("index", [0, 1, 5, 6, 7, 13])
def test_unit_only_physical_observations_have_seven_events_no_hidden_law(index):
    obj = r101.unit_fixture(index)
    public, evaluator = obj["public"], obj["evaluator_only"]
    assert set(public) == {"support_slots", "support_rewards", "support_actions",
                           "query0", "query2"}
    assert public["support_slots"].shape == (6, 5, 13)
    assert public["support_rewards"].shape == (6,)
    assert public["query0"].shape == public["query2"].shape == (5, 13)
    assert public["support_slots"].dtype == np.float32
    assert np.all(np.isfinite(public["support_rewards"]))
    assert np.all(np.isfinite(public["support_slots"]))
    assert set(public["support_actions"]) == {r85.CALIBRATE, r85.EXPLOIT_POS}
    assert np.sum(public["support_actions"] == r85.CALIBRATE) == 3
    assert evaluator["family"] == (r101.KNOWN + r101.UNSEEN)[index % 7]
    assert evaluator["private_law"] in (-1, 1)
    assert np.array_equal(public["query0"][:4], public["query2"][:4])
    assert np.array_equal(public["query0"][-1, 4:], public["query2"][-1, 4:])
    assert public["query0"][-1, r85.CALIBRATE] == 1.
    assert public["query2"][-1, r85.EXPLOIT_POS] == 1.
    # The public role tags carry a permutation of three roles plus distractor,
    # but NO hidden family opcode, precomputed law sign or private truth label.
    for slots in list(public["support_slots"]) + [public["query0"]]:
        assert sorted(slots[:4, 8].tolist()) == [0., .25, .5, .75]
    with pytest.raises(PermissionError, match="R101_UNIT_ONLY_FIXTURE_RANGE"):
        r101.unit_fixture(14)


def test_unit_support_shuffle_preserves_action_and_time_and_has_no_fixed_points():
    public = r101.unit_fixture(3)["public"]
    perm = r101.support_shuffle_indices(public["support_actions"],
                                        public["support_slots"], 3)
    assert len(set(perm.tolist())) == 6
    assert not np.any(perm == np.arange(6))
    np.testing.assert_array_equal(public["support_actions"][perm],
                                  public["support_actions"])
    np.testing.assert_array_equal(public["support_slots"][perm, -1, 12],
                                  public["support_slots"][:, -1, 12])
    with pytest.raises(PermissionError, match="R101_UNIT_ONLY_SHUFFLE"):
        r101.support_shuffle_indices(public["support_actions"],
                                     public["support_slots"], 999)


def test_native_negative_controls_identically_initialized_and_all_nine_forward():
    torch.set_num_threads(1)
    models = r101.make_models(r101.SEEDS[0])
    assert set(models) == set(r101.ARMS)
    fixture = r101.unit_fixture(4)
    for name, model in models.items():
        a = r101.unit_value(model, fixture)
        assert a.shape == (1,), name
        assert torch.isfinite(a).all(), name
    for name in (r101.SHUFFLED, r101.NO_OUTCOME, r101.RESET):
        a = dict(models[r101.NATIVE].named_parameters())
        b = dict(models[name].named_parameters())
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_native_support_reward_gradient_can_reach_memory_and_native_stem():
    torch.set_num_threads(1)
    model = r101.make_models(r101.SEEDS[0])[r101.NATIVE]
    fixture = r101.unit_fixture(0)
    public = fixture["public"]
    x = torch.from_numpy(public["support_slots"])[None]
    y = torch.from_numpy(public["support_rewards"])[None].clone().requires_grad_(True)
    q = torch.from_numpy(public["query0"])[None]
    value = model(x, y, q)
    assert value.shape == (1,)
    value.square().mean().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert sum(float(p.grad.abs().sum()) for p in model.support.parameters()
               if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in model.core.stem.parameters()
               if p.grad is not None) > 0


def test_no_outcome_and_reset_negative_control_cannot_see_reward():
    torch.set_num_threads(1)
    models = r101.make_models(r101.SEEDS[0])
    fixture = r101.unit_fixture(1)["public"]
    x = torch.from_numpy(fixture["support_slots"])[None]
    y = torch.from_numpy(fixture["support_rewards"])[None]
    q = torch.from_numpy(fixture["query0"])[None]
    with torch.no_grad():
        for name in (r101.NO_OUTCOME, r101.RESET):
            m = models[name]
            assert torch.allclose(m(x, y, q), m(x, y + 100., q))
        # The real arm receives true factual feedback into the state.
        m = models[r101.NATIVE]
        assert not torch.allclose(m(x, y, q), m(x, y + 100., q), atol=1e-7)


def test_no_original_science_or_production_write_from_dev_module():
    assert r101.STEPS == 3072 and r101.BATCH == 16
    assert not hasattr(r101, "run_scored")
    assert not hasattr(r101, "promote")



def test_learned_state_comparator_matches_actual_adapter_not_shared_native_core():
    torch.set_num_threads(1)
    models = r101.make_models(r101.SEEDS[0])
    native = models[r101.NATIVE]
    external = models[r101.EXTERNAL]
    count = lambda m: sum(p.numel() for name,p in m.named_parameters()
                          if not name.startswith("core.") and p.requires_grad)
    native_adapter = count(native)
    external_adapter = count(external)
    assert native_adapter > 0 and external_adapter > 0
    assert abs(native_adapter-external_adapter) <= .10*max(native_adapter,external_adapter)
    # Core size is measured, not presumed. Comparing adapter capacity to
    # total core+adapter capacity can mask a mismatch at either core scale.
    native_full = sum(p.numel() for p in native.parameters())
    external_full = sum(p.numel() for p in external.parameters())
    assert native_full > native_adapter
    assert external_full > external_adapter
    assert native_full - native_adapter == external_full - external_adapter
    a = dict(native.core.named_parameters())
    b = dict(external.core.named_parameters())
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_supplementary_approval_does_not_mutate_original_freeze(monkeypatch):
    from experiments import r101_cpu_firstuse_admission as scope
    assert not r101.frozen()["authority"]["protected_scored_first_use"]
    assert scope.FROZEN_BLOB == r101.FREEZE_BLOB
    assert scope.SEEDS == r101.SEEDS
    assert scope.APPROVAL_PATH == "experiments/r101_cpu_firstuse_authorization.json"
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_REPOSITORY",
                "GITHUB_JOB","RUNNER_ENVIRONMENT"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError):
        scope.admit(r101.SEEDS[0],"e"*40)


def test_original_science_denied_even_with_malicious_spoofed_job_env(monkeypatch):
    from experiments import r101_cpu_firstuse_admission as scope
    for key,val in (
        ("GITHUB_ACTIONS","true"),("GITHUB_EVENT_NAME","pull_request"),
        ("GITHUB_REPOSITORY",scope.REPO),("GITHUB_HEAD_REF",scope.BRANCH),
        ("GITHUB_JOB",scope.JOB),("R101_SEED",str(scope.SEEDS[0])),
        ("RUNNER_OS","Linux"),("RUNNER_ENVIRONMENT","github-hosted"),
        ("GITHUB_RUN_ATTEMPT","2"),("EXPECTED_CANDIDATE_SHA","e"*40),
        ("GITHUB_RUN_ID","12345"),("GITHUB_REF","refs/pull/297/merge")):
        monkeypatch.setenv(key,val)
    with pytest.raises(PermissionError):
        scope.admit(scope.SEEDS[0],"e"*40)
    # Frozen original first use is separately scoped and cannot run in G4
    # qualification even if the seed is valid.
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT","1")
    monkeypatch.setenv("GITHUB_JOB","cpu-qualification")
    with pytest.raises(PermissionError):
        scope.admit(scope.SEEDS[0],"e"*40)


def test_first_use_custody_is_verified_once_per_seed_and_never_persisted(monkeypatch):
    """Unit-only mock: tests caching, NEVER opens physical source."""
    from experiments import r101_cpu_firstuse_admission as scope
    called=[]
    def pretend_admitted(seed,sha):
        called.append((seed,sha))
        return {"seed":seed,"candidate_sha":sha,"unit_mock_only":True}
    monkeypatch.setattr(scope,"admit",pretend_admitted)
    r101._admit_protected.cache_clear()
    try:
        sha="f"*40
        seed=r101.SEEDS[0]
        left=r101._admit(seed=seed,candidate_sha=sha)
        right=r101._admit(seed=seed,candidate_sha=sha)
        assert left == right and len(called)==1
        assert r101._admit_protected.cache_info().maxsize==3
    finally:
        r101._admit_protected.cache_clear()


def test_first_run_api_custody_and_same_sha_g5_g7_with_offline_mock(monkeypatch):
    """Simulated API response only, no scored worlds and no network."""
    from experiments import r101_cpu_firstuse_admission as scope
    candidate="a"*40
    calls=[]
    def fake_get(url,token):
        calls.append(url)
        assert token=="unit-only-placeholder"
        if "/actions/workflows/" in url:
            return {"workflow_runs":[
                {"id":100,"head_sha":candidate,"run_attempt":1,
                 "name":"G4 CPU qualification","event":"pull_request"},
                {"id":101,"head_sha":candidate,"run_attempt":1,
                 "name":"G4 CPU qualification","event":"pull_request"}]}
        return {"workflow_runs":[
            {"head_sha":candidate,"name":"G5 authentic TTT CPU",
             "conclusion":"success"},
            {"head_sha":candidate,"name":"G7 pinned Mamba2 CPU import",
             "conclusion":"success"}]}
    monkeypatch.setenv("GITHUB_TOKEN","unit-only-placeholder")
    monkeypatch.setattr(scope,"_read_api",fake_get)
    good=scope._first_attempt_ci_custody(candidate,100)
    assert good["original_attempt"]==1 and good["g4_run_id"]==100
    assert good["same_sha_G5_G7_verified"]
    assert len(calls)==2
    with pytest.raises(PermissionError,match="R101_NOT_CANONICAL_FIRST_RUN"):
        scope._first_attempt_ci_custody(candidate,101)


def test_final_approval_is_additive_exact_parent_only_not_frozen_protocol():
    """Staging has no approval file; final clean single-file commit will."""
    from experiments import r101_cpu_firstuse_admission as scope
    import subprocess
    assert not r101.frozen()["authority"]["protected_scored_first_use"]
    if scope.AUTH_PATH.is_file():
        approval=scope.approval()
        parent=subprocess.run(
            ["git","rev-parse","HEAD^"],capture_output=True,
            text=True,cwd=scope.ROOT,check=True).stdout.strip()
        diff=subprocess.run(
            ["git","diff-tree","--no-commit-id","--name-only","-r","HEAD"],
            capture_output=True,text=True,cwd=scope.ROOT,check=True).stdout.strip()
        assert parent==approval["exact_parent_sha"]
        assert diff==scope.APPROVAL_PATH
        assert approval["no_gpu_merge_promotion_forge"] is True
        assert approval["no_prior_source_replay"] is True
    else:
        with pytest.raises(PermissionError,match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
            scope.approval()
