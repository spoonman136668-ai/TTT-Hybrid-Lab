"""P85-MR real first-use collector qualification: NO actual upstream effects."""
import json
import math
import sys
import time
from pathlib import Path
import pytest

EXPERIMENTS=Path(__file__).resolve().parents[1]/"experiments"
sys.path.insert(0,str(EXPERIMENTS))

from p85mr_fitcal_owner_firstuse import (
    SCOPE_BLOB, COHORTS, MAX_STEPS, MAX_ACTIONS,
    locked_scope, owner_context, observed_episode, collect_one_shot,
    _state, blob, _put_new, _append_progress, DEPS, SOURCE_BLOBS,
)
from p84mr_partition_preflight import preflight_partition, proposed_seed, planned_discrete_action
from p80mr_causalgym_trace_adapter import adapt_capture


class FauxEnv:
    """In-process fake only; not a CartPoleWind import or physical source."""
    def __init__(self, *, task, max_episode_steps, wind_mean, wind_std,
                 init_theta_mean, init_theta_std, limit=3, failure=None):
        self.state=[0.01,0.0,0.001,0.0]
        self.reset_count=0
        self.do_count=0
        self.close_count=0
        self.steps=[]
        self.limit=limit
        self.failure=failure

    def reset(self, *, seed):
        self.reset_count+=1
        self.seed=seed
        return list(self.state),{}

    def do(self, policy):
        self.do_count+=1
        if self.failure=="do_exception":
            raise RuntimeError("unknown simulated outcome")
        seen=list(self.state)
        if self.failure=="visible_drift":
            seen[0]+=0.1
        action=policy(seen)
        self.steps.append(action)
        if self.failure=="action_mismatch":
            action=1-action
        self.state=[self.state[0]+0.01,self.state[1]+0.02,
                    self.state[2]+0.001,self.state[3]+0.005]
        return list(self.state),1.0,self.do_count>=self.limit,False,{"action":action,"hidden_wind":55}

    def close(self):
        self.close_count+=1


def regime():
    return locked_scope()["cohorts"]["fit"]


def fake_factory(behaviour=None,limit=3):
    seen=[]
    def new(**kw):
        obj=FauxEnv(**kw,limit=limit,failure=behaviour)
        seen.append(obj)
        return obj
    return new,seen


def test_scope_and_parent_freezes_have_exact_git_blob_and_authorization():
    raw=(EXPERIMENTS/"p85mr_fitcal_firstuse_authorization_freeze.json").read_bytes()
    assert blob(raw)==SCOPE_BLOB
    scope=locked_scope()
    assert scope["operator_approval"]["approved_in_chat"] is True
    assert scope["operator_approval"]["owner_execution_route_verified"] is False
    assert scope["operator_approval"]["effects_started"] is False
    assert scope["lineage"]["parent_pr"]==207
    assert scope["cohorts"]["fit"]["n"]==128
    assert scope["cohorts"]["cal"]["n"]==64


def test_sealed_ood_no_development_collected_and_budgets():
    assert COHORTS==(("fit",128),("cal",64))
    assert MAX_STEPS==96
    assert MAX_ACTIONS==18432
    assert sum(n for _,n in COHORTS)==192
    assert preflight_partition()["cross_split_collisions"]==0
    assert proposed_seed("fit",0)!=proposed_seed("cal",0)


def authorized_env():
    return {
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_ACTOR":"spoonman136668-ai",
        "GITHUB_EVENT_NAME":"workflow_dispatch",
        "GITHUB_REF":"refs/heads/main",
        "GITHUB_REF_PROTECTED":"true",
        "GITHUB_RUN_ATTEMPT":"1",
        "RUNNER_NAME":"TTT-HYBRID-GPU-WIN",
        "GITHUB_RUN_ID":"42999999111",
        "GITHUB_SHA":"a"*40,
    }


@pytest.mark.parametrize("field,value",[
    ("GITHUB_REPOSITORY","spoonman136668-ai/FORGE"),
    ("GITHUB_ACTOR","someone-else"),
    ("GITHUB_EVENT_NAME","pull_request"),
    ("GITHUB_REF","refs/heads/research/candidate"),
    ("GITHUB_REF_PROTECTED","false"),
    ("GITHUB_RUN_ATTEMPT","2"),
    ("RUNNER_NAME","FORGE-CODEX-REMOTE"),
    ("GITHUB_RUN_ID",""),
    ("GITHUB_SHA","not-a-sha"),
])
def test_source_not_authorized_without_exact_owner_context(field,value):
    env=authorized_env()
    env[field]=value
    with pytest.raises(PermissionError,match="P85F1_"):
        owner_context(env)


def test_owner_context_is_only_string_precheck_not_owner_witness():
    r=owner_context(authorized_env())
    assert r["project_sha"]=="a"*40
    assert r["attempt"]==1
    assert r["runner"]=="TTT-HYBRID-GPU-WIN"
    # An ordinary env dictionary is NOT a cryptographic attestation.


def test_no_owner_context_does_not_access_external_source():
    class Poison:
        def __fspath__(self):
            raise AssertionError("SOURCE_ACCESSED_UNAUTHORIZED")
    with pytest.raises(PermissionError,match="P85F1_TRUSTED_OWNER_ROUTE_NOT_VERIFIED"):
        collect_one_shot(source_root=Poison(),project_root=Poison(),environ={})


def test_three_factual_mock_do_events_have_exact_p84_actions():
    factory,objects=fake_factory()
    data=observed_episode(factory,None,"fit",0,regime(),"fake-receipt",
                          time.monotonic()+20)
    assert len(objects)==1
    env=objects[0]
    assert env.reset_count==1 and env.do_count==3 and env.close_count==1
    assert env.seed==proposed_seed("fit",0)
    assert [r["event_id"] for r in data["transitions"]]==[1,2,3]
    assert data["probe_event_ids"]==[2,3]
    assert env.steps==[planned_discrete_action("fit",0,i) for i in (1,2,3)]
    for row in data["transitions"]:
        assert set(row)=={"event_id","state_before","executed_action",
                          "state_after","reward","terminated","truncated",
                          "mode","receipt_ref"}
        assert "hidden_wind" not in row
    assert data["transitions"][1]["state_before"]==data["transitions"][0]["state_after"]


def test_actual_factual_episode_satisfies_p80_immutable_conversion_schema():
    factory,_=fake_factory()
    world=observed_episode(factory,None,"cal",0,locked_scope()["cohorts"]["cal"],
                           "toy-evidence",time.monotonic()+20)
    record={
        "schema":"ttt-hybrid.p80mr.cartpole-do-capture.v1",
        "upstream_repo":"CausalAILab/Causal-Gymnasium",
        "upstream_commit":"c94e9a4cd3641588b6b8507e71ccee5c27032bdf",
        "upstream_path":"causal_gym/envs/cartpole_wind.py",
        "upstream_blob":"d65a6a31941709dc9201259d80dd4ebbcf48b4fa",
        "source_id":"toytest","producer_id":"unit-test","custody_ref":"toy:fake",
        "worlds":[world]}
    conv=adapt_capture(record)
    assert conv["custody"]["scientific_pass"] is False
    assert conv["worlds"][0]["probes"][0]["after_events"]==1


@pytest.mark.parametrize("failure,code",[
    ("do_exception","unknown simulated outcome"),
    ("action_mismatch","P85F1_ACTION_MISMATCH"),
    ("visible_drift","P85F1_WRONG_POLICY_OBSERVATION"),
])
def test_effectful_error_not_retried_and_cleanup_once(failure,code):
    factory,envs=fake_factory(behaviour=failure)
    with pytest.raises((ValueError,RuntimeError),match=code):
        observed_episode(factory,None,"fit",0,regime(),"test",time.monotonic()+20)
    assert envs[0].reset_count==1
    assert envs[0].do_count==1
    assert envs[0].close_count==1


def test_short_true_terminal_kept_without_padding_or_retry():
    factory,envs=fake_factory(limit=1)
    data=observed_episode(factory,None,"fit",0,regime(),"test",time.monotonic()+20)
    assert len(data["transitions"])==1
    assert data["probe_event_ids"]==[]
    assert envs[0].do_count==1 and envs[0].close_count==1


def test_expired_budget_refuses_reset():
    factory,objects=fake_factory()
    with pytest.raises(TimeoutError,match="P85F1_EXPIRED"):
        observed_episode(factory,None,"fit",0,regime(),"test",time.monotonic()-1)
    assert not objects


@pytest.mark.parametrize("bad", [
    [0.0,1.0,2.0],
    [0.0,1.0,float("nan"),2.0],
    [0.0,1.0,True,2.0],
    "not-a-state",
])
def test_bad_observations_fail_closed(bad):
    with pytest.raises(ValueError,match="P85F1_BAD_STATE"):
        _state(bad)


def test_repository_local_evidence_write_is_create_only(tmp_path):
    path=tmp_path/"progress.ndjson"
    _append_progress(path,{"event_id":1,"effect":"toy"})
    _append_progress(path,{"event_id":2,"effect":"toy"})
    assert len(path.read_text().strip().splitlines())==2
    o=tmp_path/"RESULT.json"
    digest=_put_new(o,{"science_pass":False})
    assert len(digest)==64
    with pytest.raises(FileExistsError):
        _put_new(o,{"science_pass":True})


def test_source_dependencies_predeclared_and_not_installed_by_script():
    assert SOURCE_BLOBS["causal_gym/envs/cartpole_wind.py"]=="d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
    assert DEPS["numpy"]=="1.26.4"
    assert DEPS["gymnasium"]=="0.29.1"
    code=(EXPERIMENTS/"p85mr_fitcal_owner_firstuse.py").read_text()
    assert "pip install" not in code
    assert "env.do(policy)" in code  # The real function exists but was NOT called in these tests.
    assert "science_ood" not in code
    assert "subprocess.run(" in code  # Read-only git identity verification only.


def test_durable_journal_orders_reset_intent_and_each_real_do_outcome():
    events=[]
    factory,objects=fake_factory(limit=3)
    result=observed_episode(
        factory,None,"fit",0,regime(),"journal-test",time.monotonic()+20,
        journal=events.append)
    assert len(result["transitions"])==3
    assert [x["phase"] for x in events]==[
        "episode_attempt","reset_attempt","reset_observed",
        "action_attempt","action_observed",
        "action_attempt","action_observed",
        "action_attempt","action_observed"]
    assert objects[0].reset_count==1
    assert objects[0].do_count==3
    assert objects[0].close_count==1
    for index in (3,5,7):
        assert events[index]["step"]==events[index+1]["step"]
        assert events[index]["action"]==events[index+1]["action"]
        assert events[index+1]["state_after"]==result["transitions"][(index-3)//2]["state_after"]


def test_uncertain_effect_has_durable_attempt_without_false_observation_or_replay():
    events=[]
    factory,objects=fake_factory(behaviour="do_exception")
    with pytest.raises(RuntimeError,match="unknown simulated outcome"):
        observed_episode(factory,None,"fit",0,regime(),"journal-test",
                         time.monotonic()+20,journal=events.append)
    assert [entry["phase"] for entry in events]==[
        "episode_attempt","reset_attempt","reset_observed","action_attempt"]
    assert objects[0].do_count==1 and objects[0].close_count==1
    assert not any(e["phase"]=="action_observed" for e in events)


def test_failed_intent_fsync_aborts_before_any_do():
    events=[]
    factory,objects=fake_factory(limit=4)
    def journal(event):
        if event["phase"]=="action_attempt":
            raise OSError("INTENT_DURABILITY_FAILED")
        events.append(event)
    with pytest.raises(OSError,match="INTENT_DURABILITY_FAILED"):
        observed_episode(factory,None,"fit",0,regime(),"journal-test",
                         time.monotonic()+20,journal=journal)
    assert objects[0].reset_count==1
    assert objects[0].do_count==0 and objects[0].close_count==1


def test_failed_observed_fsync_aborts_without_replay():
    events=[]
    factory,objects=fake_factory(limit=4)
    def journal(event):
        if event["phase"]=="action_observed":
            raise OSError("OUTCOME_DURABILITY_FAILED")
        events.append(event)
    with pytest.raises(OSError,match="OUTCOME_DURABILITY_FAILED"):
        observed_episode(factory,None,"fit",0,regime(),"journal-test",
                         time.monotonic()+20,journal=journal)
    assert objects[0].reset_count==1
    assert objects[0].do_count==1 and objects[0].close_count==1
    assert events[-1]["phase"]=="action_attempt"


def test_owner_execution_source_does_not_write_upstream_bytecode():
    code=(EXPERIMENTS/"p85mr_fitcal_owner_firstuse.py").read_text()
    assert "sys.dont_write_bytecode=True" in code
    assert 'attempt/(cohort+".steps.ndjson")' in code
    assert '"phase":"action_attempt"' in code
    assert '"phase":"action_observed"' in code
