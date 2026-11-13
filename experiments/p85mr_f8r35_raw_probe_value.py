"""F8-R35: raw factual probe-to-return neural learnability.

Three paired seeds compare unchanged raw-P80 48D compact neural architecture,
identical FIT factual-return pairs and AdamW budget with weak versus scaled
and strengthened ranking supervision. A seventh arm erases fifth measurements
during both train and score. Evaluator-only unchosen outcomes never train.
This is a minimal neural positive control, not the Native Hybrid substrate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r31_factual_conditional_risk as r31

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r35_raw_probe_value_freeze.json"
FREEZE_BLOB="49b13065e78bf11f2130ffc57f169e272bfb09a0"
SCHEMA="ttt-hybrid.f8r35-raw-probe-value-positive.v1"
SEEDS=(985201,985203,985207)
ARMS=tuple([(a,s) for a in ("compact_weak","compact_strong") for s in SEEDS]+
           [("compact_erased_strong",SEEDS[0])])
PANELS=("dev","ood")
POLICIES=("active_information","fixed_probe0","fixed_probe1","seeded_random")
STEPS=512
PAIR_BATCH=16
FIT_PAIRS=1152
DRAW_COUNT=10000
BOOT_SEED=20979999


def _git_blob(content):
    return hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()


def locked():
    data=FREEZE.read_bytes()
    if _git_blob(data)!=FREEZE_BLOB:
        raise ValueError("F8R35_FREEZE_BLOB_DRIFT")
    f=json.loads(data)
    d,a,o,t,auth=(f["data"],f["arms"],f["optimization"],
                  f["thresholds"],f["authority"])
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_NEURAL_PROBE_VALUE_LEARNABILITY_DEV_ONLY" or
        f["parent"]["head"]!="cdec26f67397d8bd96f419b649321fd0da68f8b6" or
        f["parent"]["r33_freeze_blob"]!=r33.FREEZE_BLOB or
        f["parent"]["r34_freeze_blob"]!="02a7528b5e1a0d9f926d6cf8bb202248490a2e70" or
        d["fit_episodes"]!=3072 or
        d["train_factual_return_pairs"]!=FIT_PAIRS or
        d["validation_factual_pairs"]!=384 or
        d["train_episodes"]!=2304 or d["validation_episodes"]!=768 or
        d["score_panels"]!=list(PANELS) or
        d["policies"]!=list(POLICIES) or
        a["compact_weak_seeds"]!=list(SEEDS) or
        a["compact_strong_seeds"]!=list(SEEDS) or
        a["compact_erased_strong_seeds"]!=[SEEDS[0]] or
        a["model_count"]!=len(ARMS) or
        o["updates_each"]!=STEPS or
        o["total_optimizer_steps"]!=STEPS*len(ARMS) or
        o["training_batch_factual_pairs"]!=PAIR_BATCH or
        o["sampler_seed"]!=20949999 or
        o["checkpoints"]!=[128,256,512] or
        o["weak_objective"].startswith("MSE pred vs unchanged 0.1") is not True or
        o["strong_objective"].startswith("MSE pred vs 10*original") is not True or
        t["seed_success_min_both_panels"]!=2 or
        t["bootstrap_draws"]!=DRAW_COUNT or
        t["bootstrap_seed"]!=BOOT_SEED or
        t["independent_scientific_pass"] is not False or
        any(v is not False for v in auth.values())):
        raise ValueError("F8R35_PROTOCOL_OR_AUTHORITY_DRIFT")
    r33.locked()
    return f


class ProbeValueMLP(nn.Module):
    """All 48 inputs are raw P80 factual records, not hand-built causal keys."""

    def __init__(self,seed):
        super().__init__()
        if seed not in SEEDS:
            raise ValueError("F8R35_UNREGISTERED_MODEL_SEED")
        torch.manual_seed(seed+20930000)
        self.model=nn.Sequential(
            nn.Linear(48,64),nn.GELU(),
            nn.Linear(64,64),nn.GELU(),
            nn.Linear(64,1))

    def forward(self,actions,effects,*,erase_fifth=False):
        if (actions.ndim!=3 or effects.shape!=actions.shape or
            actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
            raise ValueError("F8R35_INVALID_FACTUAL_INPUT_SHAPE")
        fifth=effects[:,4,:]
        if erase_fifth:
            fifth=torch.zeros_like(fifth)
        features=torch.cat([actions[:,4,:],fifth,actions[:,-1,:]],dim=-1)
        value=self.model(features).squeeze(-1)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("F8R35_NONFINITE_VALUE")
        return value


def _factual_pair_integrity(fit):
    if len(fit)!=3072:
        raise ValueError("F8R35_FIT_COUNT_MISMATCH")
    for i in range(0,len(fit),2):
        a,b=fit[i:i+2]
        if (a["action"].shape!=(14,16) or
            not torch.equal(a["action"][:-1],b["action"][:-1]) or
            not torch.equal(a["effect"][:-1],b["effect"][:-1]) or
            torch.equal(a["action"][-1],b["action"][-1])):
            raise ValueError("F8R35_UNPAIRED_FACTUAL_TRAJECTORIES")
    return len(fit)//2


def _batch(fit,pair_ids):
    if (len(fit)!=3072 or len(pair_ids)!=PAIR_BATCH or
        any(type(i) is not int or not 0<=i<FIT_PAIRS for i in pair_ids)):
        raise ValueError("F8R35_UNREGISTERED_TRAIN_PAIR")
    rows=[fit[2*i+j] for i in pair_ids for j in (0,1)]
    return (torch.stack([r["action"] for r in rows]),
            torch.stack([r["effect"] for r in rows]),
            torch.stack([r["direct_target"] for r in rows]))


def objective(prediction,selected_cost,arm):
    if arm not in ("compact_weak","compact_strong","compact_erased_strong"):
        raise ValueError("F8R35_UNREGISTERED_OBJECTIVE")
    if (prediction.ndim!=1 or prediction.shape!=selected_cost.shape
        or prediction.numel()!=2*PAIR_BATCH):
        raise ValueError("F8R35_INVALID_PAIRED_FACTUAL_LABELS")
    strong=arm!="compact_weak"
    # These are two actually selected FIT-return outcomes, NOT evaluator
    # counterfactual labels. No q or hidden status enters either objective.
    target=selected_cost*(10. if strong else 1.)
    factual=F.mse_loss(prediction,target)
    signed=torch.sign((target[1::2]-target[::2]).detach())
    pair=F.softplus(-signed*(prediction[1::2]-prediction[::2])).mean()
    return factual+(2. if strong else .1)*pair


def _validation(model,fit,arm):
    model.eval()
    rows=[]
    with torch.no_grad():
        for i in range(2304,len(fit),64):
            part=fit[i:i+64]
            actions=torch.stack([x["action"] for x in part])
            effects=torch.stack([x["effect"] for x in part])
            selected=torch.stack([x["direct_target"] for x in part])
            out=model(actions,effects,erase_fifth=arm=="compact_erased_strong")
            desired=selected*(1. if arm=="compact_weak" else 10.)
            rows.extend((out-desired).square().tolist())
    if len(rows)!=768:
        raise ValueError("F8R35_HELDOUT_FIT_LEAKAGE")
    return statistics.fmean(rows)


def _candidate_choices(model,panel,policy,*,erase=False):
    ww=r33.worlds(panel)
    observations=[r33.candidate_inputs(w,policy,erase) for w in ww]
    actions=torch.cat([x[0] for x in observations])
    effects=torch.cat([x[1] for x in observations])
    model.eval()
    with torch.no_grad():
        raw=model(actions,effects,erase_fifth=erase)
    if raw.shape!=(2*len(ww),):
        raise ValueError("F8R35_UNQUALIFIED_MODEL_PREDICTIONS")
    values=raw.detach().tolist()
    return [min((0,1),key=lambda a:(values[2*i+a],a))
            for i in range(len(ww))]


def _policy_score(model,panel,policy,*,erase=False):
    actions=_candidate_choices(model,panel,policy,erase=erase)
    regrets=r33.evaluate_regret_only(r33.worlds(panel),actions)
    return {"choices":actions,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[g*4:(g+1)*4])
                               for g in range(12)],
            "sign_pair_flips":sum(actions[i]!=actions[i+1]
                                  for i in range(0,len(actions),2)),
            "action1_count":sum(actions)}


def _evaluate(model,arm):
    results={}
    for panel in PANELS:
        erased_training=arm=="compact_erased_strong"
        policy={p:_policy_score(model,panel,p,erase=erased_training)
                for p in POLICIES}
        erased=_policy_score(model,panel,"active_information",erase=True)
        active=policy["active_information"]
        results[panel]={"policies":policy,"erased_active":erased,
            "fifth_erasure_regret_penalty":
                erased["mean_regret"]-active["mean_regret"],
            "fifth_erasure_decision_flips":sum(x!=y for x,y in zip(
                active["choices"],erased["choices"]))}
    return results


def _weight_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _train(arm,seed,fit,samples):
    model=ProbeValueMLP(seed)
    initial=_weight_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    curve=[]
    start=time.perf_counter()
    for step,pairs in enumerate(samples,1):
        a,e,selected=_batch(fit,pairs)
        model.train()
        opt.zero_grad(set_to_none=True)
        value=model(a,e,erase_fifth=arm=="compact_erased_strong")
        loss=objective(value,selected,arm)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R35_INVALID_TRAINING_LOSS")
        loss.backward()
        if (not any(p.grad is not None and bool(torch.any(p.grad!=0))
                    for p in model.parameters()) or
            any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                for p in model.parameters())):
            raise ValueError("F8R35_ZERO_OR_NONFINITE_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in (128,256,512):
            curve.append({"updates":step,
                "last_train_pair_loss":float(loss.detach()),
                "heldout_fit_mse_on_objective_scale":_validation(model,fit,arm)})
    final=_weight_hash(model)
    if final==initial or len(curve)!=3:
        raise ValueError("F8R35_NO_ACTUAL_LEARNING")
    scores=_evaluate(model,arm)
    restored=ProbeValueMLP(seed)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if _weight_hash(restored)!=final or any(
       _candidate_choices(restored,p,"active_information",
           erase=arm=="compact_erased_strong")!=
       scores[p]["policies"]["active_information"]["choices"]
       for p in PANELS):
        raise ValueError("F8R35_COLD_RESTORE_POLICY_DRIFT")
    return {"arm":arm,"seed":seed,"optimizer_updates":STEPS,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_weight_sha256":initial,"trained_weight_sha256":final,
            "training_wall_seconds":time.perf_counter()-start,
            "checkpoints":curve,"cold_restore_actions_equal":True,
            "score":scores}


def _bootstrap_lower(prev,new):
    rng=random.Random(BOOT_SEED)
    draws=[]
    for _ in range(DRAW_COUNT):
        ix=[rng.randrange(12) for __ in range(12)]
        draws.append(statistics.fmean(prev[i]-new[i] for i in ix))
    return sorted(draws)[int(.025*DRAW_COUNT)]


def _judge(rows,frozen):
    th=frozen["thresholds"]
    by={(r["arm"],r["seed"]):r for r in rows}
    erased=by[("compact_erased_strong",SEEDS[0])]
    negative={p:{"regret":erased["score"][p]["policies"]
                        ["active_information"]["mean_regret"],
                 "sign_pair_flips":erased["score"][p]["policies"]
                        ["active_information"]["sign_pair_flips"]}
              for p in PANELS}
    negative_ok=all(v["regret"]>=th["training_erased_control_regret_min_each"]
        and v["sign_pair_flips"]<=th["erased_control_sign_flip_count_max_each"]
        for v in negative.values())
    verdict=[]
    for seed in SEEDS:
        weak=by[("compact_weak",seed)]
        strong=by[("compact_strong",seed)]
        panels={}
        for p in PANELS:
            s=strong["score"][p]
            active=s["policies"]["active_information"]
            wa=weak["score"][p]["policies"]["active_information"]
            f=min((s["policies"][k] for k in ("fixed_probe0","fixed_probe1")),
                  key=lambda x:x["mean_regret"])
            vsweak=wa["mean_regret"]-active["mean_regret"]
            gain=f["mean_regret"]-active["mean_regret"]
            lower=_bootstrap_lower(wa["pergroup_regret"],
                                   active["pergroup_regret"])
            accepted=bool(active["mean_regret"]<=th["regret_active_max_each"]
                and vsweak>=th["active_gain_vs_weak_same_seed_min_each"]
                and gain>=th["active_gain_vs_best_fixed_min_each"]
                and s["fifth_erasure_regret_penalty"]>=
                    th["erasure_regret_penalty_min_each"]
                and active["sign_pair_flips"]>=
                    th["active_sign_pair_flip_count_min_each"]
                and lower>th["paired_bootstrap_lower_vs_weak_gt"])
            panels[p]={"strong_active_regret":active["mean_regret"],
                "weak_active_regret":wa["mean_regret"],
                "strong_gain_vs_weak":vsweak,
                "strong_gain_vs_best_fixed":gain,
                "erasure_regret_penalty":s["fifth_erasure_regret_penalty"],
                "sign_pair_flips":active["sign_pair_flips"],
                "paired_bootstrap_lower":lower,
                "panel_development_pass":accepted}
        verdict.append({"seed":seed,"panels":panels,
                        "both_panel_pass":all(v["panel_development_pass"]
                                              for v in panels.values())})
    wins=sum(x["both_panel_pass"] for x in verdict)
    return {"successful_strong_seeds":wins,"negative_control":negative,
        "erased_training_negative_pass":negative_ok,
        "source_neutral_development_pass":negative_ok and
             wins>=th["seed_success_min_both_panels"],
        "by_seed":verdict,"independent_scientific_pass":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    if _factual_pair_integrity(fit)!=1536:
        raise ValueError("F8R35_INCOMPLETE_FACTORIAL_TRAINING")
    generator=torch.Generator(device="cpu")
    generator.manual_seed(20949999)
    samples=torch.randint(0,FIT_PAIRS,(STEPS,PAIR_BATCH),
                          generator=generator).tolist()
    rows=[_train(arm,seed,fit,samples) for arm,seed in ARMS]
    result={"schema":SCHEMA,"classification":freeze["classification"],
            "freeze_blob":FREEZE_BLOB,"frozen_parent":freeze["parent"]["head"],
            "factual_selected_FIT_episodes":len(fit),
            "factual_training_pairs":FIT_PAIRS,
            "factual_holdout_pairs":384,
            "neural_model_instances":len(rows),
            "actual_optimizer_updates":len(rows)*STEPS,
            "models":rows,"judgment":_judge(rows,freeze),
            "matched_architecture_samples_updates":True,
            "learned_native_hybrid":False,
            "independent_mechanism_transfer":False,
            "independent_scientific_pass":False,
            "original_external_effects":0,"spent_F7_replay":False,
            "deployment":"NONE"}
    return result


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r35_raw_probe_value_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R35_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R35_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
