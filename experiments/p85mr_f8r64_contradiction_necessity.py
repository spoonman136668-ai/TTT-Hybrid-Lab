"""F8-R64 prospective author-synthetic FIT/DEV causal-directional write falsifier.

No protected OOD, GPU, external simulator, model promotion or independent
scientific claim. Source protocol is frozen in a predecessor-only Git commit.
Each prediction is evaluated BEFORE the actually selected terminal effect
is observed. Every arm uses the same 4x4 fast memory and exactly 48 writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r64_contradiction_necessity_freeze.json"
FREEZE_BLOB = "b3f8943b6e6f4f56bc13c0052fe4e67365c3142d"
SCHEMA = "ttt-hybrid.f8r64-contradiction-necessity-dev-freeze.v1"
PARENT = "fed5b9b1bcd58ae151b62d96b44dfdf0665695cc"
SALT = {"FIT":23610000,"DEV":23620000,"PROXY_NEGATIVE":23630000}
COUNTS = {"FIT":128,"DEV":64,"PROXY_NEGATIVE":64}
MODES = ("all_write_eta_0.5","contradiction","surprise","random","direction_ablated")
D, STEPS, FIRST_RETURN = 4, 48, 32


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R64_PROSPECTIVE_FREEZE_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA
            or f["stage"] != "PROSPECTIVE_FROZEN_BEFORE_R64_DEV_SOURCE_OR_SCORING"
            or f["parent"]["r63_head_sha"] != PARENT
            or f["source"]["salts"] != SALT
            or f["source"]["worlds"] != COUNTS
            or f["source"]["phase_steps"] != [16,16,16]
            or f["memory"]["floats_per_world"] != 16
            or f["memory"]["exact_updates_per_world"] != STEPS
            or f["memory"]["no_optimizer_training"] is not True
            or f["scoring"]["independent_scientific_pass"] is not False
            or any(f["authority"].values())):
        raise ValueError("F8R64_FROZEN_PROTOCOL_DRIFT")
    return f


def worlds(panel):
    if panel not in ("FIT","DEV"):
        raise ValueError("F8R64_NONDEV_SOURCE_DENIED")
    rng = np.random.default_rng(SALT[panel])
    n = COUNTS[panel]
    switches = np.array([False]*(n//2)+[True]*(n//2))
    switches = switches[rng.permutation(n)]
    out = []
    roles = np.arange(STEPS, dtype=np.int64) % D
    for i in range(n):
        law_a = rng.normal(0.0, 0.85, (D,D)).astype(np.float64)
        # Avoid degenerate zero-effect roles; no role is made predictive via ID.
        law_a += np.eye(D, dtype=np.float64)*0.30
        law_b = law_a.copy()
        if switches[i]:
            law_b[:,0:2] *= -1.0
        effects = np.empty((STEPS,D), dtype=np.float64)
        for t,role in enumerate(roles):
            law = law_b if 16 <= t < 32 else law_a
            effects[t] = law[:,role] + rng.normal(0.0, 0.05, D)
        proxy = rng.normal(0.0, 0.7, (STEPS,D))
        out.append({"panel":panel,"index":i,"roles":roles.copy(),
                    "effects":effects,"proxy":proxy,
                    "switch_evaluator_only":bool(switches[i])})
    return out


def proxy_negative(dev_worlds):
    if len(dev_worlds) != COUNTS["DEV"] or any(w["panel"]!="DEV" for w in dev_worlds):
        raise ValueError("F8R64_PROXY_REQUIRES_ORIGINAL_DEV")
    rng = np.random.default_rng(SALT["PROXY_NEGATIVE"])
    return [{**w, "proxy":rng.normal(0.0, 3.0, (STEPS,D))}
            for w in dev_worlds]


def cosine(p,y):
    a=float(np.linalg.norm(p))
    b=float(np.linalg.norm(y))
    return 0.0 if a<1e-12 or b<1e-12 else float(np.dot(p,y)/(a*b))


def factual_trace(world):
    """Public pre-update all-write FIT calibration, never evaluator phase input."""
    w=np.zeros((D,D),dtype=np.float64)
    values=[]
    for role,y in zip(world["roles"],world["effects"]):
        pred=w[:,role].copy()
        values.append((float(np.linalg.norm(y-pred)),
                       cosine(pred,y),float(np.linalg.norm(pred))))
        w[:,role]+=0.5*(y-pred)
    return values


def fit_calibration(fit_worlds):
    if len(fit_worlds)!=COUNTS["FIT"] or any(w["panel"]!="FIT" for w in fit_worlds):
        raise ValueError("F8R64_CALIBRATION_MUST_USE_FIT_ONLY")
    a=np.array([s for w in fit_worlds for s in factual_trace(w)],dtype=np.float64)
    if a.shape!=(COUNTS["FIT"]*STEPS,3) or not np.isfinite(a).all():
        raise ValueError("F8R64_BAD_FIT_TRACE")
    rate=float(np.mean((a[:,1]<-0.25)&(a[:,2]>=0.25)))
    # Quantile is computed ONLY from FIT. Ties are handled by strict >.
    surprise_threshold=float(np.quantile(a[:,0],1-rate))
    direction_threshold=float(np.quantile(np.abs(a[:,1]),1-rate))
    return {"fit_contradiction_boost_fraction":rate,
            "surprise_threshold":surprise_threshold,
            "direction_ablation_threshold":direction_threshold,
            "FIT_events":len(a)}


def random_fraction(panel,index,step):
    data=f"F8R64_RANDOM_V1/{panel}/{index}/{step}".encode()
    value=int.from_bytes(hashlib.sha256(data).digest()[:8],"little")
    return value / (1<<64)


def rollout(world,mode,cal):
    if mode not in MODES:
        raise ValueError("F8R64_UNREGISTERED_MODE")
    roles=world["roles"]
    effects=world["effects"]
    if (roles.shape!=(STEPS,) or effects.shape!=(STEPS,D)
            or np.any(roles<0) or np.any(roles>=D)
            or not np.isfinite(effects).all()):
        raise ValueError("F8R64_INVALID_FACTUAL_EPISODE")
    w=np.zeros((D,D),dtype=np.float64)
    errors=[]
    boosts=0
    before=time.perf_counter()
    for t in range(STEPS):
        role=int(roles[t])
        prediction=w[:,role].copy()
        observed=effects[t]
        errors.append(float(np.mean(np.square(observed-prediction))))
        c=cosine(prediction,observed)
        confidence=float(np.linalg.norm(prediction))
        if mode=="all_write_eta_0.5":
            gain=0.5
        else:
            if mode=="contradiction":
                boost=(c < -0.25 and confidence>=0.25)
            elif mode=="surprise":
                boost=(float(np.linalg.norm(observed-prediction)) >
                       cal["surprise_threshold"])
            elif mode=="direction_ablated":
                boost=(abs(c)>cal["direction_ablation_threshold"])
            else:
                boost=(random_fraction(world["panel"],world["index"],t) <
                       cal["fit_contradiction_boost_fraction"])
            boosts+=int(boost)
            gain=0.25 + 0.5*int(boost)
        # All selectors see exactly this same *confirmed* terminal observation.
        # No private regime/phase, future return, proxy or evaluator label used.
        w[:,role]+=gain*(observed-prediction)
    if not np.isfinite(w).all():
        raise ValueError("F8R64_NONFINITE_FAST_WEIGHTS")
    return {"errors":errors,"boosts":boosts,"writes":STEPS,
            "fast_floats":int(w.size),
            "wall_seconds":time.perf_counter()-before,
            "final_weight_hash":hashlib.sha256(w.astype("<f8").tobytes()).hexdigest()}


def score(worlds_dev,cal):
    result={}
    for mode in MODES:
        rows=[]
        for w in worlds_dev:
            x=rollout(w,mode,cal)
            e=np.array(x["errors"])
            # B changed-role metric covers roles 0/1 after the action is issued,
            # before its result is observed, with no privileged phase gate.
            rows.append({"world":w["index"],"switch":w["switch_evaluator_only"],
                         "B_changed":float(np.mean(e[16:32][np.arange(16)%4<2])),
                         "A_return_first":float(np.mean(e[FIRST_RETURN:FIRST_RETURN+4])),
                         "stable_all":float(np.mean(e)),
                         "boosts":x["boosts"],"writes":x["writes"],
                         "fast_floats":x["fast_floats"],
                         "wall_seconds":x["wall_seconds"],
                         "final_weight_hash":x["final_weight_hash"]})
        sw=[a for a in rows if a["switch"]]
        st=[a for a in rows if not a["switch"]]
        if len(sw)!=len(st) or len(sw)!=COUNTS["DEV"]//2:
            raise ValueError("F8R64_UNBALANCED_WORLD_TYPES")
        result[mode]={"B_changed_MSE":float(np.mean([a["B_changed"] for a in sw])),
                      "A_return_first_MSE":float(np.mean([a["A_return_first"] for a in sw])),
                      "stable_world_MSE":float(np.mean([a["stable_all"] for a in st])),
                      "boosts_total":sum(a["boosts"] for a in rows),
                      "writes_total":sum(a["writes"] for a in rows),
                      "fast_floats":D*D,
                      "total_wall_seconds":sum(a["wall_seconds"] for a in rows),
                      "per_world":rows}
    return result


def judgment(dev,cal):
    base=("all_write_eta_0.5","surprise","random")
    best=min(base,key=lambda a:dev[a]["B_changed_MSE"])
    candidate=dev["contradiction"]
    strongest=dev[best]
    change=1-candidate["B_changed_MSE"]/max(strongest["B_changed_MSE"],1e-12)
    stable=candidate["stable_world_MSE"]/max(strongest["stable_world_MSE"],1e-12)-1
    retained=candidate["A_return_first_MSE"]/max(strongest["A_return_first_MSE"],1e-12)-1
    necessity=dev["direction_ablated"]["B_changed_MSE"]/max(candidate["B_changed_MSE"],1e-12)-1
    paired=np.array([a["B_changed"]-b["B_changed"]
                     for a,b in zip(candidate["per_world"],strongest["per_world"])
                     if a["switch"] and b["switch"]])
    rng=np.random.default_rng(23649999)
    samples=rng.integers(0,len(paired),(2000,len(paired)))
    upper=float(np.quantile(paired[samples].mean(axis=1),.975))
    checks={"changed_B_reduction_10pct":change>=.10,
            "stable_not_worse_5pct":stable<=.05,
            "A_return_not_worse_5pct":retained<=.05,
            "direction_ablation_worse_10pct":necessity>=.10,
            "bootstrap_upper_delta_negative":upper<0}
    return {"strongest_eligible_fixed_control":best,
            "B_relative_reduction":float(change),
            "stable_relative_increase":float(stable),
            "A_return_relative_increase":float(retained),
            "direction_ablated_relative_increase":float(necessity),
            "paired_bootstrap_delta_upper_95pct":upper,
            "checks":checks,
            "DEV_mechanism_signal":all(checks.values()),
            "independent_scientific_pass":False}


def run():
    freeze=locked()
    fit=worlds("FIT")
    dev=worlds("DEV")
    cal=fit_calibration(fit)
    observed=score(dev,cal)
    # Proxy-only intervention must be entirely irrelevant: no prediction
    # or routing path reads proxy. Wall-clock naturally need not match.
    corrupt=score(proxy_negative(dev),cal)
    proxy_invariant=all(
        [x["final_weight_hash"] for x in observed[mode]["per_world"]]
        ==[x["final_weight_hash"] for x in corrupt[mode]["per_world"]]
        for mode in MODES)
    repeat=score(dev,cal)
    cold_identical=all(
        [x["final_weight_hash"] for x in observed[mode]["per_world"]]
        ==[x["final_weight_hash"] for x in repeat[mode]["per_world"]]
        and [x["B_changed"] for x in observed[mode]["per_world"]]
        ==[x["B_changed"] for x in repeat[mode]["per_world"]]
        for mode in MODES)
    j=judgment(observed,cal)
    j["checks"]["proxy_goal_invariance"]=proxy_invariant
    j["checks"]["cold_replay_exact"]=cold_identical
    j["DEV_mechanism_signal"]=all(j["checks"].values())
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "classification":freeze["classification"],
            "FIT_calibration":cal,
            "source":{k:{"worlds":len(v),"switched":sum(int(w["switch_evaluator_only"]) for w in v)}
                      for k,v in (("FIT",fit),("DEV",dev))},
            "method_scores":observed,"judgment":j,
            "total_r64_protected_OOD_calls":0,
            "external_original_first_use":False,"novel_model_training":False,
            "independent_scientific_pass":False,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r64_contradiction_necessity_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R64_REFUSE_OVERWRITE")
    dest.write_text(json.dumps(result,allow_nan=False,sort_keys=True,indent=2)+"\n")


def deny_external_original_first_use(*_args,**_kwargs):
    raise PermissionError("F8R64_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
