"""P109: full FIT fixture-only incumbent and trained-SSM checkpoint admission.

No F3/F6 source acquisition, source replay, model training, owner runner,
external simulator, network or scientific-acceptance action. The fixture
sha256s are only integrity checks, never authenticity/authority signatures.
A real new F6 FIT SSM checkpoint remains unverified until separately qualified.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p107_f6_owner_readiness as p107

FREEZE_COMMIT="7e882b348a55dfed644455de2422873cd1b047f9"
FREEZE_BLOB="293b022bc0d8a430a0aba44908d9d8eba4fbb36b"
PARENT="60fab256da20ae4808722d145682d0718d3c17b6"
CONTROLS=("a_opt","greedy_goal","physical_balance","ssm256_model_based")
CHECKPOINT_ARM="ssm256_model_based"
EPISODE_FIELDS=frozenset((
    "block","index","survival_steps","reward_total","physical_loss",
    "terminated","truncated","action_count","policy","receipt_ref",
    "source_session_sha256","wall_ms"))
CHECKPOINT_FIELDS=frozenset((
    "schema","model_kind","model_source_blob","checkpoint_format",
    "checkpoint_sha256","training_seed","fit_only","trained_worlds",
    "confirmed_factual_effects","train_updates","optimizer",
    "optimizer_steps","model_static_parameters","persistent_floats",
    "cal_or_ood_used","fit_dataset_sha256"))
MIN_SURVIVAL=24
MAX_STEPS=96
BLOCKS=range(3)
WORLDS=range(16)
UPDATES_MIN=400
UPDATES_MAX=800
SSM_BLOB="aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd"
BLOBS={
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p85mr_dynamics_baselines.py":SSM_BLOB,
    "experiments/p85mr_f4_controls_freeze.json":
        "d8136313dd90cc406f00c49770ed3beeff4a1ee4",
    "experiments/p107_f6_owner_contract_freeze.json":
        "95c5d4fe840287c1c86aacd54b6eadd60f2e0d0f",
    "experiments/p107_f6_owner_readiness.py":
        "0ea2cc00b5a7d86bfaa2a10e3a1585560748e504",
    "experiments/p108_f6_confirmed_rls_freeze.json":
        "07f786a898cfe0c6d9841f2b76e1bdf1590f2fd5",
    "experiments/p108_f6_confirmed_rls_fixture.py":
        "30353758463a80a9ab0568aab4cebadfe8926609",
}


def freeze():
    """Fail before in-memory fixture work on source/fork/authority drift."""
    root=Path(__file__).resolve().parents[1]
    raw=(root/"experiments/p109_f6_strong_control_freeze.json").read_bytes()
    if p106._git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("P109_FROZEN_PROTOCOL_BLOB_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p109-strong-control-custody-preflight.v1"
        or f.get("stage")!="FROZEN_SOURCE_NEUTRAL_FIT_CONTROL_REVIEW_BEFORE_IMPLEMENTATION"
        or f["source"]["parent_sha"]!=PARENT
        or f["source"]["F6_protocol_fork"]!="UNRESOLVED_NO_F6_ACTIVATION"
        or f["source"]["new_cartpolewind_firstuse"] is not False
        or f["fixture_contract"]["policy_arms"]!=list(CONTROLS)
        or f["fixture_contract"]["checkpoint_expected_source_blob"]!=SSM_BLOB
        or f["fixture_contract"]["worlds_per_arm"]!=48
        or f["fixture_contract"]["max_steps"]!=MAX_STEPS
        or f["fixture_contract"]["minimum_frozen_SSM_updates"]!=UPDATES_MIN
        or f["fixture_contract"]["maximum_frozen_SSM_updates"]!=UPDATES_MAX):
        raise ValueError("P109_FROZEN_CONTRACT_DRIFT")
    if (f["science"]["authentic_F6_effects"]!=0
        or f["science"]["scientific_pass"] is not False
        or f["science"]["real_CAL_admitted"] is not False
        or f["science"]["protocol_owner_selection_authorized"] is not False
        or f["engineering"]["no_external_source_firstuse"] is not True):
        raise ValueError("P109_SCIENCE_AUTHORITY_DRIFT")
    for path,digest in BLOBS.items():
        if p106._git_blob((root/path).read_bytes())!=digest:
            raise ValueError("P109_PINNED_ORIGINAL_SOURCE_DRIFT")
    p107.frozen()
    return f


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _real(value,reason):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(reason)
    return float(value)


def _session_sha(arm,block,index):
    """Synthetic deterministic ID, NOT an owner authority or session proof."""
    if (arm not in CONTROLS
        or type(block) is not int or block not in BLOCKS
        or type(index) is not int or index not in WORLDS):
        raise ValueError("P109_UNQUALIFIED_FIT_SESSION")
    seed=p106.FUTURE_SEEDS["FIT"]+1000*block+index
    return _sha(["p109-fixture-only","FIT",seed,arm])


def make_fixture_row(arm,block,index,steps,reward,loss,wall,
                     *,terminated=None,truncated=False):
    """Helper exists exclusively for tests; no simulated physics is run."""
    if terminated is None:
        terminated=steps<MAX_STEPS
    row={
        "block":block,"index":index,"policy":arm,
        "survival_steps":steps,"action_count":steps,
        "reward_total":reward,"physical_loss":loss,
        "wall_ms":wall,"terminated":terminated,"truncated":truncated,
        "source_session_sha256":_session_sha(arm,block,index),
    }
    row["receipt_ref"]="fixture:"+_sha(row)
    return row


def _validate_row(row,arm,seen):
    if not isinstance(row,dict) or set(row)!=EPISODE_FIELDS:
        raise ValueError("P109_NOT_EXACT_FIT_EPISODE_FIELDS")
    block,index=row["block"],row["index"]
    if (row["policy"]!=arm or type(block) is not int or block not in BLOCKS
        or type(index) is not int or index not in WORLDS
        or (block,index) in seen):
        raise ValueError("P109_MISSING_DUPLICATE_OR_WRONG_FIT_WORLD")
    seen.add((block,index))
    steps=row["survival_steps"]
    if (type(steps) is not int or not 1<=steps<=MAX_STEPS
        or type(row["action_count"]) is not int
        or row["action_count"]!=steps):
        raise ValueError("P109_UNMATCHED_FACTUAL_ACTION_BUDGET")
    if (type(row["terminated"]) is not bool
        or type(row["truncated"]) is not bool
        or (row["terminated"] and row["truncated"])
        or (steps<MAX_STEPS and not(row["terminated"] or row["truncated"]))):
        raise ValueError("P109_UNTRUTHFUL_TERMINATION")
    _real(row["reward_total"],"P109_BAD_REALIZED_REWARD")
    if _real(row["physical_loss"],"P109_BAD_PHYSICAL_COST")<0:
        raise ValueError("P109_BAD_PHYSICAL_COST")
    if _real(row["wall_ms"],"P109_BAD_WALL_COST")<=0:
        raise ValueError("P109_BAD_WALL_COST")
    if row["source_session_sha256"]!=_session_sha(arm,block,index):
        raise ValueError("P109_WRONG_FIT_SESSION_ID")
    no_receipt={k:v for k,v in row.items() if k!="receipt_ref"}
    if row["receipt_ref"]!="fixture:"+_sha(no_receipt):
        raise ValueError("P109_TAMPERED_FIXTURE_RECEIPT")


def _fit_receipt_sha(rows):
    """Pinned input receipt hash; only a fixture integrity check."""
    return _sha([r["receipt_ref"] for r in sorted(
        rows,key=lambda x:(x["block"],x["index"]))])


def validate_checkpoint(manifest,checkpoint_bytes,fit_rows):
    """Validates a test-only checkpoint. NEVER admits external F6 data."""
    freeze()
    if (not isinstance(manifest,dict) or
        set(manifest)!=CHECKPOINT_FIELDS or
        not isinstance(checkpoint_bytes,bytes) or
        not checkpoint_bytes or len(checkpoint_bytes)>8*1024*1024):
        raise ValueError("P109_INVALID_FIXTURE_CHECKPOINT_FORMAT")
    if (manifest["schema"]!="p109.synthetic-SSM-checkpoint-receipt.v1"
        or manifest["model_kind"]!="ssm256"
        or manifest["model_source_blob"]!=SSM_BLOB
        or manifest["checkpoint_format"]!="fixture-bytes-only"
        or manifest["checkpoint_sha256"]!=hashlib.sha256(checkpoint_bytes).hexdigest()
        or manifest["fit_dataset_sha256"]!=_fit_receipt_sha(fit_rows)):
        raise ValueError("P109_INVALID_OR_WRONG_SOURCE_CHECKPOINT")
    seed=manifest["training_seed"]
    if (type(seed) is not int or not 0<=seed<2**31
        or type(manifest["fit_only"]) is not bool
        or manifest["fit_only"] is not True
        or manifest["cal_or_ood_used"] is not False
        or manifest["optimizer"]!="AdamW"
        or type(manifest["train_updates"]) is not int
        or not UPDATES_MIN<=manifest["train_updates"]<=UPDATES_MAX
        or type(manifest["optimizer_steps"]) is not int
        or manifest["optimizer_steps"]!=manifest["train_updates"]
        or type(manifest["trained_worlds"]) is not int
        or manifest["trained_worlds"]!=48
        or type(manifest["confirmed_factual_effects"]) is not int
        or manifest["confirmed_factual_effects"]!=sum(r["action_count"]
                                                       for r in fit_rows)
        or type(manifest["model_static_parameters"]) is not int
        or manifest["model_static_parameters"]<=0
        or type(manifest["persistent_floats"]) is not int
        or manifest["persistent_floats"]!=256):
        raise ValueError("P109_UNTRAINED_OR_CAL_CONTAMINATED_SSM")
    return {
        "checkpoint_fixture_digest_matched":True,
        "train_steps_fixture":manifest["train_updates"],
        "complete_FIT_fixture_effects":manifest["confirmed_factual_effects"],
        "model_kind":"ssm256",
        "authentic_model_weights_inspected":False,
        "model_trained_on_actual_F6":False,
        "real_owner_checkpoint_authenticated":False,
        "real_CAL_admission":False,
    }


def fit_incumbent_fixture(episodes,manifest,checkpoint_bytes):
    """Validate 4 complete FIT arms; no true source action or CAL exposure."""
    freeze()
    if not isinstance(episodes,dict) or set(episodes)!=set(CONTROLS):
        raise ValueError("P109_MISSING_STRONG_TRAINED_CONVENTIONAL_ARM")
    checked={}
    for arm in CONTROLS:
        rows=episodes[arm]
        if not isinstance(rows,list) or len(rows)!=48:
            raise ValueError("P109_INCOMPLETE_FIT_EPISODES")
        seen=set()
        for row in rows:
            _validate_row(row,arm,seen)
        if seen!={(b,i) for b in BLOCKS for i in WORLDS}:
            raise ValueError("P109_INCOMPLETE_FIT_WORLD_GRID")
        checked[arm]={
            "mean_survival":sum(r["survival_steps"] for r in rows)/48.,
            "mean_realized_reward":sum(r["reward_total"] for r in rows)/48.,
            "mean_physical_loss":sum(r["physical_loss"] for r in rows)/48.,
            "complete_factual_effect_count":sum(r["action_count"] for r in rows),
            "total_wall_ms":sum(r["wall_ms"] for r in rows),
            "FIT_fixture_receipt_sha256":_fit_receipt_sha(rows),
        }
    audit=validate_checkpoint(manifest,checkpoint_bytes,episodes[CHECKPOINT_ARM])
    winner=min(CONTROLS,key=lambda arm:(
        -checked[arm]["mean_survival"],
        checked[arm]["mean_physical_loss"],
        CONTROLS.index(arm)))
    qualified=checked[winner]["mean_survival"]>=MIN_SURVIVAL
    return {
        "schema":"ttt-hybrid.p109-fit-conventional-fixture-report.v1",
        "status":"SOURCE_NEUTRAL_ENG_ONLY_NO_F6_FIRSTUSE",
        "incumbent_fixture_only":winner,
        "incumbent_mean_survival_fixture":checked[winner]["mean_survival"],
        "competence_fixture_only":qualified,
        "source_neutral_four_arm_fairness":checked,
        "trained_ssm_fixture_custody":audit,
        "all_arms_equal_world_opportunities":True,
        "all_arms_have_equal_factual_effect_counts":len({
            checked[arm]["complete_factual_effect_count"] for arm in CONTROLS})==1,
        "trained_real_SSM":False,"real_FIT_baseline_qualified":False,
        "real_CAL_admitted":False,"external_F6_effects":0,
        "external_source_firstuse_authorized":False,
        "scientific_pass":False,"no_oracle_effects_used":True,
    }


def refuse_real_source_or_CAL(*_args,**_kwargs):
    raise PermissionError("P109_F6_REAL_TRAINING_FIRSTUSE_AND_CAL_NOT_AUTHORIZED")
