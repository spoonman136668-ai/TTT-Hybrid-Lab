"""P113: exact seven-channel P80/F4 normalized prequential fixture audit.

Never imports original legacy P80/F4 modules (they use old top-level import
paths), upstream simulator or native owner runtime. Checks their pinned blobs
and duplicates ONLY pure scalar target/metric algebra. Synthetic predictions
are supplied by test fixtures, NEVER an authenticated trained SSM readout.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from experiments import p110_f6_five_arm_readiness as p110
from experiments import p112_f6_holdout_preflight as p112

FREEZE_COMMIT="a24c1534efe238928b09781a0eafe64a649b3ad6"
FREEZE_BLOB="72e3dc57ee078e884150b501850660fec5679b37"
PARENT_SHA="d7c13bd08ea37d067443d366702916a2e493e1c2"
SCALES=(2.4,3.0,0.2095,3.5)
SEEDS=(985201,985203,985207)
UNITS="P80_TANH_NORMALIZED_16D_F4_SEVEN_ACTIVE"
ROOT=Path(__file__).resolve().parents[1]
PINNED={
    "experiments/p112_f6_holdout_freeze.json":
      "76f46135a8f134e1d638615aafad2d1d8f5805c3",
    "experiments/p112_f6_holdout_preflight.py":
      "b54952c626a8a4227bd6b2e7ecb7c3d4165b41f7",
    "experiments/p80mr_causalgym_trace_adapter.py":
      "24eb0e050e90466bd42e3626e419fbac08c94c72",
    "experiments/p85mr_dynamics_baselines.py":
      "aa7f6c34a68ee3118f0fe9e44659aa3dec5524bd",
    "experiments/p69_trainable_competitors.py":
      "ad0d3712e4eff289ee1ab782820e4c3fc80a95aa",
    "model.py":"5d9ea57b827aa1796548447868fa5ee14c251b49",
}
STEP_FIELDS=frozenset(("event_id","mode","executed_action","state_before",
 "state_after","reward","terminated","truncated","receipt_ref"))
WORLD_FIELDS=frozenset(("split","block","index","policy","events","receipt_ref"))
MANIFEST_FIELDS=frozenset(("seed","kind","original_f4_blob","original_p69_blob",
 "checkpoint_format","checkpoint_sha256","train_receipts_sha256",
 "validation_worlds_sha256","predictions_sha256","prediction_units",
 "read_before_current_effect","optimizer_steps_declared","fit_only",
 "cal_or_ood_used","receipt_ref"))


def frozen():
    raw=(ROOT/"experiments/p113_f6_normalized_ssm_freeze.json").read_bytes()
    if p110._blob(raw)!=FREEZE_BLOB:
        raise ValueError("P113_FROZEN_GIT_BLOB_DRIFT")
    f=json.loads(raw)
    if (f.get("schema")!="ttt-hybrid.p113-f6-p80-f4-normalized-complete-trace-fixture.v1"
        or f["source"]["parent_sha"]!=PARENT_SHA
        or f["source"]["p80_source_blob"]!=PINNED["experiments/p80mr_causalgym_trace_adapter.py"]
        or f["source"]["p85_f4_dynamics_blob"]!=PINNED["experiments/p85mr_dynamics_baselines.py"]
        or f["physical"]["scales"]!=list(SCALES)
        or f["physical"]["P80_observed_channels"]!=7
        or f["physical"]["padding_channels"]!=9
        or f["physical"]["component_weights"]!={"deltas":1,"reward":.2,"terminated":.1,"truncated":.1}
        or f["fixture_protocol"]["validation_source"].find("indices8..15")<0
        or f["controls"]["real_F6_firstuse_authorized"] is not False
        or f["controls"]["real_F6_effects"]!=0
        or f["controls"]["trained_SSM_checkpoint_qualified"] is not False):
        raise ValueError("P113_PROTOCOL_OR_AUTHORITY_DRIFT")
    for path,sha in PINNED.items():
        if p110._blob((ROOT/path).read_bytes())!=sha:
            raise ValueError("P113_PINNED_MODEL_OR_LABEL_DRIFT")
    p112.frozen()
    return f


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _real(value,why):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError(why)
    return float(value)


def state4(value):
    if not isinstance(value,(tuple,list)) or len(value)!=4:
        raise ValueError("P113_BAD_PHYSICAL_STATE")
    return tuple(_real(x,"P113_BAD_PHYSICAL_STATE") for x in value)


def label_from_fact(before,after,reward,terminated,truncated):
    """Exact original P80 _observed_effect; forecast never accesses this."""
    a,b=state4(before),state4(after)
    if type(terminated) is not bool or type(truncated) is not bool:
        raise ValueError("P113_BAD_TERMINATION_FLAGS")
    deltas=[y-x for x,y in zip(a,b)]
    if not all(math.isfinite(x) for x in deltas):
        raise ValueError("P113_UNREPRESENTABLE_DELTA")
    effect=[math.tanh(d/scale) for d,scale in zip(deltas,SCALES)]
    effect += [math.tanh(_real(reward,"P113_BAD_REALIZED_REWARD")),
               float(terminated),float(truncated)]
    effect += [0.]*9
    return effect


def fixture_event(number,action,before,after,reward,terminated=False,truncated=False):
    """Fixture helper, never a simulator do(), only a fabricated chosen action."""
    v={"event_id":number,"mode":"do","executed_action":action,
       "state_before":list(before),"state_after":list(after),
       "reward":reward,"terminated":terminated,"truncated":truncated}
    v["receipt_ref"]="fixture:"+_sha(v)
    return v


def fixture_world(index,events):
    if type(index) is not int or not 8<=index<=15:
        raise ValueError("P113_ONLY_EIGHT_HELDOUT_FIT_WORLDS")
    item={"split":"FIT","block":1,"index":index,
          "policy":"physical_balance","events":events}
    item["receipt_ref"]="fixture:"+_sha(item)
    return item


def _validate_world(world):
    if not isinstance(world,dict) or set(world)!=WORLD_FIELDS:
        raise ValueError("P113_WORLD_FIELDS_OR_ORACLE_DENIED")
    if (world["split"]!="FIT" or type(world["block"]) is not int
        or world["block"]!=1 or world["policy"]!="physical_balance"
        or type(world["index"]) is not int or world["index"] not in range(8,16)):
        raise ValueError("P113_HELDOUT_FIT_SOURCE_SCOPE")
    es=world["events"]
    if not isinstance(es,list) or not 2<=len(es)<=96:
        raise ValueError("P113_INCOMPLETE_PREQUENTIAL_EPISODE")
    prev=None
    for i,e in enumerate(es,1):
        if not isinstance(e,dict) or set(e)!=STEP_FIELDS:
            raise ValueError("P113_UNCHOSEN_EFFECT_OR_STEP_SCHEMA")
        if (type(e["event_id"]) is not int or e["event_id"]!=i
            or e["mode"]!="do" or type(e["executed_action"]) is not int
            or e["executed_action"] not in (0,1)):
            raise ValueError("P113_NOT_CONFIRMED_ACTUAL_BINARY_ACTION")
        b,a=state4(e["state_before"]),state4(e["state_after"])
        if prev is not None and b!=prev:
            raise ValueError("P113_BROKEN_PHYSICAL_TRAJECTORY")
        prev=a
        _real(e["reward"],"P113_BAD_REALIZED_REWARD")
        if type(e["terminated"]) is not bool or type(e["truncated"]) is not bool:
            raise ValueError("P113_BAD_TERMINATION_FLAGS")
        if e["terminated"] and e["truncated"]:
            raise ValueError("P113_CONFLICTING_TERMINATION")
        if (e["terminated"] or e["truncated"]) != (i==len(es)):
            raise ValueError("P113_EARLY_TERMINAL_DROP_OR_MISSING_END")
        if e["receipt_ref"]!="fixture:"+_sha({
            k:v for k,v in e.items() if k!="receipt_ref"}):
            raise ValueError("P113_TAMPERED_SELECTED_EFFECT_RECEIPT")
        label_from_fact(b,a,e["reward"],e["terminated"],e["truncated"])
    if world["receipt_ref"]!="fixture:"+_sha({
        k:v for k,v in world.items() if k!="receipt_ref"}):
        raise ValueError("P113_TAMPERED_WORLD_RECEIPT")
    return world


def validate_complete_fixture(training,worlds):
    frozen()
    if not isinstance(worlds,list) or len(worlds)!=8:
        raise ValueError("P113_REQUIRE_EIGHT_COMPLETE_WORLDS")
    seen=set()
    for w in worlds:
        _validate_world(w)
        if w["index"] in seen:
            raise ValueError("P113_DUPLICATE_HELDOUT_WORLD")
        seen.add(w["index"])
    if seen!=set(range(8,16)):
        raise ValueError("P113_INCOMPLETE_WORLD_ID_GRID")
    # Reuse the qualified P112 train24/val8 world-identity check on one
    # representative actually witnessed outcome/world. This derivative is
    # NOT an additional source receipt, just a typed disjointness check.
    stub=[]
    for w in sorted(worlds,key=lambda x:x["index"]):
        e=w["events"][0]
        stub.append(p112.fixture_validation_row(w["index"],e["executed_action"],
                      e["state_before"],e["state_after"]))
    prior=p112.validate_worlds(training,stub)
    return {"train_receipts_sha256":prior["train_receipts_sha256"],
            "worlds_sha256":_sha([w["receipt_ref"] for w in
                     sorted(worlds,key=lambda x:x["index"])]),
            "worlds":sorted(worlds,key=lambda x:x["index"]),
            "event_count":sum(len(w["events"]) for w in worlds),
            "scored_count":sum(len(w["events"])-1 for w in worlds)}


def fixture_manifest(seed,checkpoint,training,worlds,predictions):
    if type(seed) is not int or seed not in SEEDS or not isinstance(checkpoint,bytes):
        raise ValueError("P113_UNKNOWN_TEST_CHECKPOINT")
    ids=validate_complete_fixture(training,worlds)
    m={"seed":seed,"kind":"ssm256",
       "original_f4_blob":PINNED["experiments/p85mr_dynamics_baselines.py"],
       "original_p69_blob":PINNED["experiments/p69_trainable_competitors.py"],
       "checkpoint_format":"fixture-bytes-only",
       "checkpoint_sha256":hashlib.sha256(checkpoint).hexdigest(),
       "train_receipts_sha256":ids["train_receipts_sha256"],
       "validation_worlds_sha256":ids["worlds_sha256"],
       "predictions_sha256":_sha(predictions),
       "prediction_units":UNITS,"read_before_current_effect":True,
       "optimizer_steps_declared":800,"fit_only":True,"cal_or_ood_used":False}
    m["receipt_ref"]="fixture:"+_sha(m)
    return m


def score_p80_f4_fixture(training,worlds,models):
    """Seven-channel original F4 component weights; no source/model calls."""
    info=validate_complete_fixture(training,worlds)
    if not isinstance(models,dict) or set(models)!=set(SEEDS):
        raise ValueError("P113_THREE_SSM_SEEDS_REQUIRED")
    labels=[[label_from_fact(e["state_before"],e["state_after"],e["reward"],
                             e["terminated"],e["truncated"])
             for e in w["events"]] for w in info["worlds"]]
    # Before-write F4 loss excludes first event for each independently
    # restored world. All complete episodes stay in the audit.
    used=[y for world in labels for y in world[1:]]
    zero_delta=sum(sum(v*v for v in y[:4])/4 for y in used)/len(used)
    if zero_delta<=0 or not math.isfinite(zero_delta):
        raise ValueError("P113_UNINFORMATIVE_ZERO_BASELINE")
    scores={}
    for seed in SEEDS:
        d=models[seed]
        if not isinstance(d,dict) or set(d)!={"manifest","bytes","predictions"}:
            raise ValueError("P113_UNQUALIFIED_MODEL_FIXTURE_FIELDS")
        m,raw,pred=d["manifest"],d["bytes"],d["predictions"]
        if not isinstance(m,dict) or set(m)!=MANIFEST_FIELDS or not isinstance(raw,bytes) or not 1<=len(raw)<=8*1024*1024:
            raise ValueError("P113_BAD_CHECKPOINT_MANIFEST")
        if (type(m["seed"]) is not int or m["seed"]!=seed
            or m["kind"]!="ssm256"
            or m["original_f4_blob"]!=PINNED["experiments/p85mr_dynamics_baselines.py"]
            or m["original_p69_blob"]!=PINNED["experiments/p69_trainable_competitors.py"]
            or m["checkpoint_format"]!="fixture-bytes-only"
            or m["checkpoint_sha256"]!=hashlib.sha256(raw).hexdigest()
            or m["train_receipts_sha256"]!=info["train_receipts_sha256"]
            or m["validation_worlds_sha256"]!=info["worlds_sha256"]
            or m["prediction_units"]!=UNITS
            or m["read_before_current_effect"] is not True
            or type(m["optimizer_steps_declared"]) is not int
            or m["optimizer_steps_declared"]!=800
            or m["fit_only"] is not True or m["cal_or_ood_used"] is not False):
            raise ValueError("P113_WRONG_MODEL_OR_OUTPUT_UNITS")
        if (not isinstance(pred,list) or len(pred)!=8
            or any(not isinstance(item,list) or len(item)!=len(world["events"])
                   for item,world in zip(pred,info["worlds"]))):
            raise ValueError("P113_INCOMPLETE_PREQUENTIAL_PREDICTIONS")
        forecasts=[]
        for seq in pred:
            row=[]
            for p in seq:
                if not isinstance(p,(tuple,list)) or len(p)!=16:
                    raise ValueError("P113_REQUIRE_FULL_16D_SSM_FORECAST")
                row.append([_real(x,"P113_NONFINITE_SSM_FORECAST") for x in p])
            forecasts.append(row)
        if (m["predictions_sha256"]!=_sha(pred)
            or m["receipt_ref"]!="fixture:"+_sha({
               k:v for k,v in m.items() if k!="receipt_ref"})):
            raise ValueError("P113_TAMPERED_FORECAST_OR_CHECKPOINT_RECEIPT")
        components=[0.,0.,0.,0.]
        n=0
        for got,true in zip(forecasts,labels):
            for p,y in zip(got[1:],true[1:]):
                components[0]+=sum((a-b)**2 for a,b in zip(p[:4],y[:4]))/4
                for j in range(1,4):
                    components[j]+=(p[j+3]-y[j+3])**2
                n+=1
        if n!=info["scored_count"]:
            raise ValueError("P113_DROPPED_SHORT_WORLD")
        c=[v/n for v in components]
        value=c[0]+.2*c[1]+.1*c[2]+.1*c[3]
        scores[seed]={"f4_weighted_objective_fixture":value,
                     "normalized_delta_mse_fixture":c[0],
                     "reward_mse_fixture":c[1],
                     "terminated_mse_fixture":c[2],
                     "truncated_mse_fixture":c[3],
                     "delta_vs_zero_ratio_fixture":c[0]/zero_delta,
                     "scored_actual_events_fixture":n}
    winner=min(SEEDS,key=lambda seed:(scores[seed]["f4_weighted_objective_fixture"],seed))
    return {
      "schema":"ttt-hybrid.p113-p80-f4-unit-audit-fixture.v1",
      "status":"ENGINEERING_FIXTURE_ONLY_REAL_SSM_UNQUALIFIED",
      "selected_seed_fixture_only":winner,
      "all_seed_seven_channel_metrics_fixture":scores,
      "zero_normalized_delta_mse_fixture":zero_delta,
      "validation_complete_worlds_fixture":8,
      "validation_actual_factual_events_fixture":info["event_count"],
      "prequential_scored_events_after_first_fixture":info["scored_count"],
      "first_event_excluded_from_F4_loss":True,
      "forecast_padding_may_be_nonzero_and_is_excluded_from_F4_loss":True,
      "validation_used_for_seed_selection_not_independent_science":True,
      "original_F4_trained_weights_loaded":False,
      "original_F6_actions_executed":0,
      "real_F6_training_executed":False,
      "native_SSM_readout_authenticated":False,
      "crossworld_RLS_custody_qualified":False,
      "F6_owner_firstuse_authorized":False,
      "real_CAL_admitted":False,"scientific_pass":False,"deployment":False,
    }


def deny_original_source(*_args,**_kwargs):
    raise PermissionError("P113_F6_SCIENCE_FIRSTUSE_NOT_AUTHORIZED")
