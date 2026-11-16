"""R102 UNIT_ONLY negative controls. Absolutely no original FIT/DEV first use."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101
from experiments import r102_contradiction_evidence_diagnostic as r102


def test_r102_freeze_first_and_distinct_from_R101_spent_cohort():
    f=r102.frozen()
    assert r85.git_blob(r102.FREEZE.read_bytes())==r102.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVE_FREEZE_ONLY_BEFORE_IMPLEMENTATION_AND_FIRST_USE"
    assert f["lineage"]["parent_R101_sealed_SHA"]==r102.PARENT
    assert f["lineage"]["prior_R101_original_scored_SHA"]==r102.R101_SPENT
    assert not f["lineage"]["prior_scored_sources_or_weights_reused"]
    assert set(f["source"]["seeds"]).isdisjoint(r101.SEEDS)
    assert set(f["source"]["salts"].values()).isdisjoint(
        r101.frozen()["source"]["salts"].values())
    assert set(r102.UNSEEN).isdisjoint(set(r101.KNOWN+r101.UNSEEN))
    assert len(r102.ARMS)==11
    assert f["source"]["support_interventions_total"]==16
    assert f["training"]["steps_per_arm"]==3072
    assert f["training"]["batch_size"]==16
    assert not f["authority"]["protected_scored_first_use"]
    assert not f["authority"]["GPU_first_use"]
    assert not f["authority"]["merge"]
    assert not f["authority"]["FORGE_integration"]
    assert f["evaluation"]["independent_scientific_pass"] is False


@pytest.mark.parametrize("panel",tuple(r102.PANEL_COUNTS))
def test_all_original_fresh_source_panels_fenced_even_plausible_SHA(panel):
    with pytest.raises(PermissionError,match="R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r102._world(panel,r102.SEEDS[0],0,"a"*40)


@pytest.mark.parametrize("panel",("R101_DEV_UNKNOWN","R100_SCORING","PROTECTED","UNIT_ONLY"))
def test_prior_or_unregistered_scored_physics_never_reused(panel):
    with pytest.raises(PermissionError,match="R102_OLD_SPENT_OR_NONREGISTERED_SOURCE_DENIED"):
        r102._world(panel,r102.SEEDS[0],0)
    with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r102.original_ci(seed=r102.SEEDS[0])


def test_registered_source_seeds_ranges_and_first_use_refused():
    with pytest.raises(PermissionError,match="R102_UNREGISTERED_SEED_OR_INDEX"):
        r102._world("FIT",42,0)
    with pytest.raises(PermissionError,match="R102_UNREGISTERED_SEED_OR_INDEX"):
        r102._world("DEV_UNKNOWN",r102.SEEDS[0],100000)
    with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r102._admit("original",seed=r102.SEEDS[0])


@pytest.mark.parametrize("family",r102.KNOWN+r102.UNSEEN)
def test_evaluator_only_operators_are_deterministic_and_bounded(family):
    factor=np.array([1,-1,1,-1],np.int64)
    assert r102._operator_truth(factor,(0,1,2),family) in (-1,1)
    assert r102._operator_truth(factor,(0,1,2),family)==(
        r102._operator_truth(factor.copy(),(0,1,2),family))
    with pytest.raises(ValueError,match="R102_INVALID_PRIVATE_PHYSICAL_LAW"):
        r102._operator_truth(np.array([0,1,1,-1]),(0,1,2),family)


def test_new_private_boolean_families_are_not_old_unseen_families():
    for fam in r102.UNSEEN:
        outputs=[r102._operator_truth(
            np.array([2*((j>>2)&1)-1,2*((j>>1)&1)-1,2*(j&1)-1,1]),
            (0,1,2),fam) for j in range(8)]
        assert len(set(outputs))==2
        assert any(outputs!=[
            r102._operator_truth(np.array([2*((j>>2)&1)-1,2*((j>>1)&1)-1,
                                          2*(j&1)-1,1]),(0,1,2),old)
            for j in range(8)] for old in r102.KNOWN)


@pytest.mark.parametrize("panel",("DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL"))
def test_heldout_families_balanced_in_joint_public_pivot_and_action_cells(panel):
    report=r102.non_scoring_design_audit()
    assert report["original_scored_source_opened"] is False
    assert report["independent_scientific_pass"] is False
    assert report["R101_scored_cohort_reuse"] is False
    for seed in r102.SEEDS:
        cells=report["public_pivot_action_family_cell_balance"][panel][str(seed)]
        assert len(cells)==(2 if panel=="DELAYED_COLD" else 4)
        for counts in cells.values():
            assert set(counts)==set(r102.UNSEEN)
            assert len(set(counts.values()))==1
            assert min(counts.values())>0
    if panel=="REVERSAL":
        for j in (0,1,2,3,127):
            assert r102.family_for_panel(panel,j,0)!=r102.family_for_panel(panel,j,16)


def test_original_public_family_shortcut_is_rejected_in_negative_audit(monkeypatch):
    original=r102.family_for_panel
    def malicious_parity_family(panel,index,event=16):
        if panel=="DEV_UNKNOWN":
            return r102.UNSEEN[index%2] # public EARLY/COLD
        return original(panel,index,event)
    monkeypatch.setattr(r102,"family_for_panel",malicious_parity_family)
    with pytest.raises(RuntimeError,match="R102_PRIVATE_FAMILY_PUBLIC_TIME_ACTION_LEAK"):
        r102.non_scoring_design_audit()


def test_non_scoring_ideal_feedback_identifiability_increases_6_to_16(monkeypatch):
    monkeypatch.setattr(r102,"_world",lambda *_a,**_k:
                        (_ for _ in ()).throw(AssertionError("REAL_FIT_DEV_OPENED")))
    report=r102.non_scoring_design_audit()
    assert report["original_scored_source_opened"] is False
    for fam,diagnostic in report["combinatorial_family_distinguishability"].items():
        assert fam in r102.UNSEEN
        assert 0 < diagnostic["nearest_known_hamming_distance_of_8"] <= 8
        assert diagnostic["sixteen_iid_uniform_ideal_feedback_ambiguity"] <= (
            diagnostic["six_iid_uniform_ideal_feedback_ambiguity"])
    assert not report["independent_scientific_pass"]


def test_unit_only_factual_physics_and_true_six_subset():
    f=r102.unit_fixture(0)
    p=f["public"]
    e=f["evaluator_only"]
    assert p["support_slots"].shape==(16,5,13)
    assert p["support_rewards"].shape==(16,)
    assert p["support_actions"].tolist()==[
        r85.CALIBRATE if j%2==0 else r85.EXPLOIT_POS for j in range(16)]
    assert (p["support_actions"][:6]==np.array([
        r85.CALIBRATE,r85.EXPLOIT_POS]*3)).all()
    assert p["query0"].shape==(5,13) and p["query2"].shape==(5,13)
    assert np.isfinite(p["support_rewards"]).all()
    np.testing.assert_array_equal(p["query0"][:4],p["query2"][:4])
    assert isinstance(e["private_law"],int)
    assert "private_law" not in p and "private_family" not in p
    assert set(e)>= {"return0","return2","private_law"}
    again=r102.unit_fixture(0)
    np.testing.assert_array_equal(p["support_slots"],again["public"]["support_slots"])
    np.testing.assert_array_equal(p["support_rewards"],again["public"]["support_rewards"])


def test_unit_only_outcome_shuffle_changes_pairings_but_not_public_action_time():
    p=r102.unit_fixture(1)["public"]
    a=r102.public_support_shuffle(p["support_actions"],p["support_slots"],1)
    b=r102.public_support_shuffle(p["support_actions"],p["support_slots"],1)
    np.testing.assert_array_equal(a,b)
    assert not np.any(a==np.arange(16))
    np.testing.assert_array_equal(p["support_actions"][a],p["support_actions"])
    np.testing.assert_array_equal(p["support_slots"][a,-1,12],
                                  p["support_slots"][:,-1,12])
    assert np.any(p["support_rewards"][a]!=p["support_rewards"])


def test_native_contradiction_memory_value_and_differentiable_factual_update():
    torch.set_num_threads(1)
    f=r102.unit_fixture(2)
    p=f["public"]
    core=r95_native_for_test(r102.SEEDS[0])
    torch.manual_seed(r102.SEEDS[0]+102)
    m=r102.NativeEvidenceValue(core,"gated")
    slots=torch.from_numpy(p["support_slots"][None])
    rewards=torch.from_numpy(p["support_rewards"][None])
    query=torch.from_numpy(p["query0"][None])
    target=torch.tensor([f["evaluator_only"]["return0"]],dtype=torch.float32)
    value=m(slots,rewards,query)
    assert value.shape==(1,)
    assert bool(torch.isfinite(value).all())
    loss=torch.nn.functional.mse_loss(value,target)
    loss.backward()
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.gate.parameters())
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.prediction.parameters())
    assert any(param.grad is not None and
               param.grad.abs().sum()>0
               for param in m.support.public.parameters())


def r95_native_for_test(seed):
    from experiments import r95_learned_relation_diagnostic as r95
    return r95._native(seed)


def test_16_and_6_native_modes_match_first_six_factual_events_not_synthetic_future():
    torch.set_num_threads(1)
    f=r102.unit_fixture(3)
    p=f["public"]
    x=torch.from_numpy(p["support_slots"][None])
    y=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query2"][None])
    core=r95_native_for_test(r102.SEEDS[0])
    torch.manual_seed(r102.SEEDS[0]+102)
    six=r102.NativeEvidenceValue(core,"short")
    with torch.no_grad():
        baseline=six(x,y,q)
        changed=y.clone()
        changed[:,6:]=10000.
        assert torch.allclose(six(x,changed,q),baseline)
        assert not torch.allclose(six(x[:,:6],y[:,:6],q),torch.zeros_like(baseline))
    assert baseline.shape==(1,)


def test_all_11_fresh_source_models_equal_core_and_adapter_capacity():
    torch.set_num_threads(1)
    models=r102.make_models(r102.SEEDS[0])
    assert set(models)==set(r102.ARMS)
    adap=lambda m:sum(p.numel() for name,p in m.named_parameters()
                     if p.requires_grad and not name.startswith("core."))
    na=adap(models[r102.NATIVE]);ea=adap(models[r102.EXTERNAL])
    assert abs(na-ea) <=.1*max(na,ea)
    for name in (r102.NATIVE,r102.GATE_BLIND,r102.SHUFFLED,r102.NO_OUTCOME,
                 r102.RESET,r102.SHORT):
        assert adap(models[name])==na
    assert models[r102.NATIVE].mode=="gated"
    assert models[r102.GATE_BLIND].mode=="blind"
    assert models[r102.SHORT].mode=="short"
    assert models[r102.EXTERNAL].core is not models[r102.NATIVE].core


@pytest.mark.parametrize("mode",("gru","set","flat","static"))
def test_new_16_factual_history_conventional_models_trainable_unit_only(mode):
    torch.set_num_threads(1)
    f=r102.unit_fixture(0)
    p=f["public"]
    m=r102.ConventionalEvidenceValue(mode)
    x=torch.from_numpy(p["support_slots"][None])
    y=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query0"][None])
    target=torch.tensor([f["evaluator_only"]["return0"]],dtype=torch.float32)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    init={name:param.detach().clone() for name,param in m.named_parameters()}
    pred=m(x,y,q)
    assert pred.shape==(1,)
    loss=torch.nn.functional.mse_loss(pred,target)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all()
               for param in m.parameters())
    opt.step()
    assert any(not torch.equal(init[name],param)
               for name,param in m.named_parameters())



def test_supplementary_owner_cpu_firstuse_does_not_modify_frozen_authority(monkeypatch):
    """The original immutable freeze stays FALSE before and after admission."""
    from experiments import r102_cpu_firstuse_admission as admission
    assert r102.FREEZE_BLOB==admission.FROZEN_BLOB
    assert r102.SEEDS==admission.SEEDS
    assert r102.frozen()["authority"]["protected_scored_first_use"] is False
    for key in ("GITHUB_ACTIONS","GITHUB_REPOSITORY","GITHUB_JOB",
                "GITHUB_EVENT_NAME","RUNNER_ENVIRONMENT"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError):
        admission.admit(r102.SEEDS[0],"e"*40)


def test_r102_original_first_attempt_denies_spoofed_owner_environment(monkeypatch):
    from experiments import r102_cpu_firstuse_admission as admission
    for key,value in (
        ("GITHUB_ACTIONS","true"),("GITHUB_EVENT_NAME","pull_request"),
        ("GITHUB_REPOSITORY",admission.REPO),
        ("GITHUB_HEAD_REF",admission.BRANCH),
        ("GITHUB_JOB",admission.JOB),
        ("RUNNER_OS","Linux"),("RUNNER_ENVIRONMENT","github-hosted"),
        ("GITHUB_RUN_ATTEMPT","2"),
        ("GITHUB_REF","refs/pull/298/merge"),
        ("R102_SEED",str(r102.SEEDS[0])),
        ("GITHUB_RUN_ID","12345"),
        ("EXPECTED_CANDIDATE_SHA","e"*40)):
        monkeypatch.setenv(key,value)
    with pytest.raises(PermissionError):
        admission.admit(r102.SEEDS[0],"e"*40)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT","1")
    monkeypatch.setenv("GITHUB_JOB","cpu-qualification")
    with pytest.raises(PermissionError):
        admission.admit(r102.SEEDS[0],"e"*40)


def test_r102_firstuse_custody_is_inprocess_only_and_errors_not_cached(monkeypatch):
    from experiments import r102_cpu_firstuse_admission as admission
    calls=[]
    def mocked(seed,sha):
        calls.append((seed,sha))
        return {"mock_unit_only":True,"seed":seed,"sha":sha}
    monkeypatch.setattr(admission,"admit",mocked)
    r102._admit_protected.cache_clear()
    try:
        sha="f"*40
        one=r102._admit(seed=r102.SEEDS[0],candidate_sha=sha)
        two=r102._admit(seed=r102.SEEDS[0],candidate_sha=sha)
        assert one==two and len(calls)==1
        assert r102._admit_protected.cache_info().maxsize==3
    finally:
        r102._admit_protected.cache_clear()


def test_r102_original_cpu_run_verifies_first_sha_dual_os_and_native_controls(monkeypatch):
    from experiments import r102_cpu_firstuse_admission as admission
    sha="c"*40
    url_seen=[]
    def mocked_api(url,token):
        url_seen.append(url)
        assert token=="unit-only-token"
        if "/actions/workflows/" in url:
            return {"workflow_runs":[
                {"id":100,"head_sha":sha,"run_attempt":1,
                 "name":"G4 CPU qualification","event":"pull_request"},
                {"id":101,"head_sha":sha,"run_attempt":1,
                 "name":"G4 CPU qualification","event":"pull_request"}]}
        if "/actions/runs/100/jobs" in url:
            return {"jobs":[
                {"name":"CPU qualification / ubuntu-latest","conclusion":"success"},
                {"name":"CPU qualification / windows-latest","conclusion":"success"}]}
        if url.endswith("/git/ref/heads/"+admission.BRANCH):
            return {"object":{"sha":sha}}
        return {"workflow_runs":[
            {"head_sha":sha,"name":"G5 authentic TTT CPU","conclusion":"success"},
            {"head_sha":sha,"name":"G7 pinned Mamba2 CPU import","conclusion":"success"}]}
    monkeypatch.setenv("GITHUB_TOKEN","unit-only-token")
    monkeypatch.setattr(admission,"_read_api",mocked_api)
    receipt=admission._first_attempt_ci_custody(sha,100)
    assert receipt["same_source_G4_linux_windows"] is True
    assert receipt["same_source_G5_G7"] is True
    assert receipt["original_G4_run_id"]==100
    assert len(url_seen)==4
    with pytest.raises(PermissionError,match="R102_NOT_FIRST_CANONICAL_G4"):
        admission._first_attempt_ci_custody(sha,101)


def test_r102_final_manifest_strict_parent_and_single_file_commit_or_absent():
    from experiments import r102_cpu_firstuse_admission as admission
    import subprocess
    if hasattr(admission,"SEAL_PATH") and admission.SEAL_PATH.is_file():
        receipt=__import__("json").loads(
            admission.SEAL_PATH.read_text(encoding="utf-8"))
        assert receipt["original_scored_SHA"] == admission.SPENT_ORIGINAL_SHA
        assert receipt["original_run_id"] == admission.SPENT_RUN_ID
        assert tuple(receipt["original_seeds"]) == admission.SPENT_SEEDS
        assert receipt["all_original_seeds_consumed"] is True
        assert receipt["independent_scientific_pass"] is False
        assert receipt["final_science_status"] == (
            "R102_FRESH_UNSEEN_CAUSAL_INFERENCE_NO_GO")
        assert set(map(int,receipt["by_seed"]))==set(admission.SPENT_SEEDS)
        assert not any(s["all_frozen_checks_pass"] for s in
                       receipt["by_seed"].values())
        with pytest.raises(PermissionError,
                           match="R102_ORIGINAL_FIRST_USE_PERMANENTLY_CONSUMED"):
            admission.admit(admission.SEEDS[0],admission.SPENT_ORIGINAL_SHA)
    elif admission.AUTH_PATH.exists():
        manifest=admission.approval()
        parent=subprocess.run(["git","rev-parse","HEAD^"],cwd=admission.ROOT,
                              capture_output=True,text=True,check=True).stdout.strip()
        changed=subprocess.run(["git","diff-tree","--no-commit-id","--name-only",
                                "-r","HEAD"],cwd=admission.ROOT,
                               capture_output=True,text=True,check=True).stdout.strip()
        assert manifest["exact_parent_sha"]==parent
        assert changed==admission.APPROVAL_PATH
        assert manifest["no_r101_original_cohort_reuse"] is True
    else:
        with pytest.raises(PermissionError,match="R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
            admission.approval()


def test_r102_permanent_seal_blocks_replay_even_with_spoofed_original_environment(monkeypatch):
    from experiments import r102_cpu_firstuse_admission as admission
    if not hasattr(admission,"SEAL_PATH") or not admission.SEAL_PATH.is_file():
        return
    monkeypatch.setenv("GITHUB_ACTIONS","true")
    monkeypatch.setenv("GITHUB_RUN_ID",str(admission.SPENT_RUN_ID))
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT","1")
    monkeypatch.setenv("EXPECTED_CANDIDATE_SHA",admission.SPENT_ORIGINAL_SHA)
    with pytest.raises(PermissionError,match="R102_ORIGINAL_FIRST_USE_PERMANENTLY_CONSUMED"):
        admission.admit(admission.SEEDS[0],admission.SPENT_ORIGINAL_SHA)
    with pytest.raises(PermissionError,match="R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r102._world("FIT",admission.SEEDS[0],0,admission.SPENT_ORIGINAL_SHA)

