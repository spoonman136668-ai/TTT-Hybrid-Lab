"""F8-R36: frozen balanced factual-return choice learning, source-neutral only.

Only two separately selected, fully observed FIT return consequences label
each binary choice. The *current* return effect is never an input to the
learner. Every scored alternative return effect is evaluator-only.

These DEV/OOD panels reuse a known quartic mechanism: no independent science,
novel-law transfer, AGI demonstration, or architecture deployment is implied.
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
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r35_raw_probe_value as r35

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r36_balanced_choice_freeze.json"
FREEZE_BLOB="4c6d4904cf57f9269917ba092b6fd2f537433276"
SCHEMA="ttt-hybrid.f8r36-balanced-factual-choice.v1"
SEEDS=(985201,985203,985207)
ARMS=tuple([(arm,seed) for arm in ("raw","explicit","native")
            for seed in SEEDS]+[("history",985201),("erased",985201)])
PANELS=("dev","ood")
POLICIES=("active_information","fixed_probe0","fixed_probe1","seeded_random")
GROUPS=16
BATCH=16
STEPS=512
TRAIN_PAIRS=1152
DRAW=10000
BOOT_SEED=21079999
SALTS={"dev":21050000,"ood":21060000}
STATES={"dev":(.59,.71),"ood":(.57,.75)}
QABS={"dev":1.05,"ood":.92}
B_STEPS={"dev":8,"ood":16}


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R36_FREEZE_BLOB_DRIFT")
    p=json.loads(raw)
    d,a,t,s,u=(p["data"],p["arms"],p["train"],
                p["development"],p["authority"])
    if (p["schema"]!=SCHEMA or
        p["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULTS" or
        p["classification"]!="SOURCE_NEUTRAL_NEURAL_FACTUAL_CHOICE_DEV_ONLY" or
        p["parent"]["sha"]!="99dcb05cc0e16d9da33facd5482daaa9f5e2221d" or
        p["parent"]["r35_freeze_blob"]!=r35.FREEZE_BLOB or
        p["parent"]["r35_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r35_raw_probe_value.py").read_bytes()) or
        p["parent"]["r33_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r33_probe_calibration.py").read_bytes()) or
        d["FIT_episodes"]!=3072 or
        d["train_factual_choice_pairs"]!=TRAIN_PAIRS or
        d["validation_factual_choice_pairs"]!=384 or
        d["class0_pairs"]!=576 or d["class1_pairs"]!=576 or
        d["informative_fifth_pairs"]!=576 or
        d["noninformative_fifth_pairs"]!=576 or
        d["score_panels"]!=list(PANELS) or
        d["groups_per_panel"]!=GROUPS or
        d["worlds_per_panel"]!=4*GROUPS or
        d["dev"]!={"return_states_by_group_parity":[.59,.71],
                   "q_abs":1.05,"B_steps":8} or
        d["ood"]!={"return_states_by_group_parity":[.57,.75],
                   "q_abs":.92,"B_steps":16} or
        a["raw_seeds"]!=list(SEEDS) or
        a["explicit_seeds"]!=list(SEEDS) or
        a["native_seeds"]!=list(SEEDS) or
        a["total_instances"]!=len(ARMS) or
        t["updates_each"]!=STEPS or
        t["optimizer_updates_total"]!=len(ARMS)*STEPS or
        t["pairs_per_batch"]!=BATCH or
        t["sampler_seed"]!=21049999 or
        t["checkpoints"]!=[128,256,512] or
        s["bootstrap_draws"]!=DRAW or
        s["bootstrap_seed"]!=BOOT_SEED or
        s["raw_seed_pass_min"]!=2 or
        s["explicit_seed_pass_min"]!=2 or
        s["native_seed_pass_min"]!=2 or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in u.values())):
        raise ValueError("F8R36_PROTOCOL_OR_AUTHORITY_DRIFT")
    r33.locked()
    r34.locked()
    r35.locked()
    return p


def fit_pairs(fit):
    if len(fit)!=3072:
        raise ValueError("F8R36_UNQUALIFIED_FACTUAL_FIT")
    winners=[]
    by_probe=[0,0]
    for i in range(0,len(fit),2):
        low,high=fit[i],fit[i+1]
        if (low["action"].shape!=(14,16) or
            high["action"].shape!=(14,16) or
            not torch.equal(low["action"][:-1],high["action"][:-1]) or
            not torch.equal(low["effect"][:-1],high["effect"][:-1]) or
            torch.equal(low["action"][-1],high["action"][-1])):
            raise ValueError("F8R36_UNPAIRED_SELECTED_RETURN")
        # Factual cost, not an evaluator-only alternative-action consequence.
        first=float(low["direct_target"])
        second=float(high["direct_target"])
        if not math.isfinite(first+second) or first==second:
            raise ValueError("F8R36_NONIDENTIFIABLE_FACTUAL_PAIR")
        winners.append(0 if first<second else 1)
        # Public P80 action key context and fifth choice only.
        ctx=bool(float(low["action"][-1,2])>0)
        fifth=bool(float(low["action"][4,6])>float(low["action"][4,5]))
        by_probe[int(ctx==fifth)]+=1
    if (len(winners)!=1536 or
        winners[:TRAIN_PAIRS].count(0)!=576 or
        winners[:TRAIN_PAIRS].count(1)!=576 or
        by_probe[:]!=[768,768]):
        raise ValueError("F8R36_IMBALANCED_FACTUAL_CLASS_OR_PROBE")
    return winners


def batch(fit,winners,ids):
    if (len(fit)!=3072 or len(ids)!=BATCH or
        any(type(i) is not int or i<0 or i>=TRAIN_PAIRS for i in ids)):
        raise ValueError("F8R36_UNREGISTERED_TRAIN_BATCH")
    rows=[fit[2*i+j] for i in ids for j in (0,1)]
    a=torch.stack([row["action"] for row in rows])
    e=torch.stack([row["effect"] for row in rows])
    y=torch.tensor([winners[i] for i in ids],dtype=torch.long)
    return a,e,y


class ExplicitGateMLP(nn.Module):
    """Hand-designed public action equality indicators: positive control only."""
    def __init__(self,seed):
        super().__init__()
        if seed not in SEEDS:
            raise ValueError("F8R36_BAD_EXPLICIT_SEED")
        torch.manual_seed(seed+21030000)
        self.layers=nn.Sequential(nn.Linear(51,64),nn.GELU(),
                                  nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,actions,effects):
        if (actions.ndim!=3 or actions.shape!=effects.shape or
            actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
            raise ValueError("F8R36_INVALID_EXPLICIT_INPUT")
        ctx=actions[:,-1,2]>0
        fifth=actions[:,4,6]>actions[:,4,5]
        ret=actions[:,-1,6]>actions[:,-1,5]
        fifth_match=(ctx==fifth).to(dtype=actions.dtype)
        ret_match=(ctx==ret).to(dtype=actions.dtype)
        joint=fifth_match*ret_match
        public=torch.stack([fifth_match,ret_match,joint],dim=-1)
        return self.layers(torch.cat((actions[:,4,:],effects[:,4,:],
                                     actions[:,-1,:],public),dim=-1)).squeeze(-1)


def model_for(arm,seed):
    if (arm,seed) not in ARMS:
        raise ValueError("F8R36_UNKNOWN_ARM")
    if arm in ("raw","erased"):
        return r35.ProbeValueMLP(seed)
    if arm=="explicit":
        return ExplicitGateMLP(seed)
    return r34.FactualMemory(
        "native_factual_memory" if arm=="native" else "history_only",seed)


def preaction(model,arm,actions,effects,*,erase=False,zero_backbone=False):
    if (actions.ndim!=3 or actions.shape!=effects.shape or
        actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
        raise ValueError("F8R36_UNQUALIFIED_PREACTION_RECORD")
    # A current return effect is unavailable at decision time, always.
    masked=effects.clone()
    masked[:,-1,:].zero_()
    if erase or arm=="erased":
        masked[:,4,:].zero_()
    if arm=="explicit":
        return model(actions,masked)
    if arm in ("raw","erased"):
        return model(actions,masked)
    return model(actions,masked,ablate_backbone=zero_backbone)


def _world_B(panel,group):
    if panel not in PANELS or type(group) is not int or not 0<=group<GROUPS:
        raise ValueError("F8R36_INVALID_WORLD_B")
    return tuple((t%2,round(.021*math.sin((SALTS[panel]+group)*.19+t*.47)
                   +.013*((t+group)%3)-.018,8))
                 for t in range(B_STEPS[panel]))


def world(panel,group,context,sign):
    if (panel not in PANELS or type(group) is not int or
        not 0<=group<GROUPS or context not in (0,1) or sign not in (-1,1)):
        raise ValueError("F8R36_INVALID_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":sign*QABS[panel],
            "return_state":STATES[panel][group%2],"goal":1.,
            "b_history":_world_B(panel,group)}


def worlds(panel):
    if panel not in PANELS:
        raise ValueError("F8R36_INVALID_PANEL")
    return [world(panel,g,c,z) for g in range(GROUPS)
            for c in (0,1) for z in (-1,1)]


def _probe(w,policy):
    if policy not in POLICIES:
        raise ValueError("F8R36_INVALID_PROBE_POLICY")
    if policy=="active_information":
        return w["public_context"]
    if policy=="fixed_probe0":
        return 0
    if policy=="fixed_probe1":
        return 1
    payload=f'f8r36/{w["panel"]}/{w["group"]}/{w["public_context"]}'.encode()
    return hashlib.sha256(payload).digest()[0]&1


def candidate_inputs(w,policy,*,erase=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    fifth=_probe(w,policy)
    # Generator samples only the actually selected fifth action; it never
    # fabricates any unselected future return effect for the predictor.
    early=[r33.b._record(s,a,r33.r32.after(s,a,c,q),c)
           for s,a in r33.r32.EARLY]
    early.append(r33.b._record(r33.PROBE,fifth,
                              r33.r32.after(r33.PROBE,fifth,c,q),c))
    early.extend(r33.b._record(0.,a,eff,c)
                 for a,eff in w["b_history"])
    actions=torch.stack([x[0] for x in early])
    effects=torch.stack([x[1] for x in early])
    if erase:
        effects[4].zero_()
    items=[]
    for a in (0,1):
        key=r33.b._record(w["return_state"],a,w["return_state"],c)[0]
        items.append((torch.cat((actions,key.unsqueeze(0))),
                      torch.cat((effects,torch.zeros((1,16))))))
    return torch.stack([x[0] for x in items]),torch.stack([x[1] for x in items])


def regret_evaluator_only(ww,choices):
    """This is the sole function computing unchosen first-return outcomes."""
    if len(ww)!=4*GROUPS or len(choices)!=4*GROUPS:
        raise ValueError("F8R36_SCORE_WORLD_MISMATCH")
    costs=[]
    for w,a in zip(ww,choices):
        possible=[(r33.r32.after(w["return_state"],candidate,
                                 w["public_context"],
                                 w["q_evaluator_only"])-w["goal"])**2
                  for candidate in (0,1)]
        costs.append(max(0.,possible[a]-min(possible)))
    return costs


def _choices(model,arm,panel,policy,*,erase=False,zero_backbone=False):
    ww=worlds(panel)
    rows=[candidate_inputs(w,policy,erase=erase) for w in ww]
    actions=torch.cat([x[0] for x in rows])
    effects=torch.cat([x[1] for x in rows])
    model.eval()
    with torch.no_grad():
        pred=preaction(model,arm,actions,effects,erase=erase,
                       zero_backbone=zero_backbone)
    if pred.shape!=(2*len(ww),) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R36_BAD_DECISION_LOGITS")
    logits=pred.tolist()
    return [min((0,1),key=lambda a:(logits[2*i+a],a))
            for i in range(len(ww))]


def _score(model,arm,panel,policy,*,erase=False,zero_backbone=False):
    chosen=_choices(model,arm,panel,policy,erase=erase,
                    zero_backbone=zero_backbone)
    regrets=regret_evaluator_only(worlds(panel),chosen)
    return {"choices":chosen,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[4*g:4*(g+1)])
                               for g in range(GROUPS)],
            "sign_pair_flips":sum(chosen[i]!=chosen[i+1]
                                  for i in range(0,len(chosen),2)),
            "chosen_action1":sum(chosen)}


def score_all(model,arm):
    output={}
    for panel in PANELS:
        policies={p:_score(model,arm,panel,p) for p in POLICIES}
        erased=_score(model,arm,panel,"active_information",erase=True)
        backbone=_score(model,arm,panel,"active_information",
                        zero_backbone=True) if arm=="native" else None
        active=policies["active_information"]
        output[panel]={"policies":policies,"erased_active":erased,
            "fifth_erasure_regret_penalty":
                erased["mean_regret"]-active["mean_regret"],
            "fifth_erasure_decision_flips":
                sum(x!=y for x,y in zip(erased["choices"],active["choices"])),
            "backbone_zeroed_active":backbone,
            "backbone_zeroed_regret_penalty":
                backbone["mean_regret"]-active["mean_regret"] if backbone else None}
    return output


def _validation(model,arm,fit,winners):
    model.eval()
    good=0
    costs=[]
    with torch.no_grad():
        for start in range(TRAIN_PAIRS,1536,64):
            part=[fit[2*i+j] for i in range(start,start+64) for j in (0,1)]
            a=torch.stack([x["action"] for x in part])
            e=torch.stack([x["effect"] for x in part])
            pred=preaction(model,arm,a,e).reshape(64,2)
            labels=torch.tensor(winners[start:start+64],dtype=torch.long)
            costs.append(float(F.cross_entropy(-pred,labels)))
            good+=int((pred.argmin(dim=1)==labels).sum())
    return {"heldout_factual_pair_accuracy":good/384,
            "heldout_factual_choice_CE":statistics.fmean(costs),
            "heldout_factual_pairs":384}


def _weight_hash(model):
    return r35._weight_hash(model)


def train_arm(arm,seed,fit,winners,samples):
    model=model_for(arm,seed)
    before=_weight_hash(model)
    opt=torch.optim.AdamW(model.parameters(),
                          lr=.001 if arm in ("native","history") else .002,
                          weight_decay=.01)
    trace=[]
    start=time.perf_counter()
    for step,ids in enumerate(samples,1):
        a,e,y=batch(fit,winners,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        logits=preaction(model,arm,a,e)
        if logits.shape!=(2*BATCH,):
            raise ValueError("F8R36_TRAIN_PREDICTION_SHAPE")
        loss=F.cross_entropy(-logits.reshape(BATCH,2),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R36_NONFINITE_FACTUAL_CE")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0))
                    for g in grads) or
            any(g is not None and not bool(torch.isfinite(g).all()) for g in grads)):
            raise ValueError("F8R36_NO_VALID_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in (128,256,512):
            trace.append({"updates":step,"train_choice_CE":float(loss.detach()),
                          **_validation(model,arm,fit,winners)})
    after=_weight_hash(model)
    if after==before or len(trace)!=3:
        raise ValueError("F8R36_WEIGHTS_DID_NOT_UPDATE")
    scores=score_all(model,arm)
    restored=model_for(arm,seed)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if _weight_hash(restored)!=after or any(
        _choices(restored,arm,p,"active_information")!=
        scores[p]["policies"]["active_information"]["choices"]
        for p in PANELS):
        raise ValueError("F8R36_COLD_RESTORE_ACTION_DRIFT")
    return {"arm":arm,"seed":seed,"optimizer_updates":STEPS,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_sha256":before,"trained_sha256":after,
            "train_wall_seconds":time.perf_counter()-start,
            "checkpoints":trace,"cold_restore_equal":True,
            "panels":scores}


def _bootstrap_lower(active,fixed):
    rng=random.Random(BOOT_SEED)
    draws=[]
    for _ in range(DRAW):
        ix=[rng.randrange(GROUPS) for __ in range(GROUPS)]
        draws.append(statistics.fmean(fixed[i]-active[i] for i in ix))
    return sorted(draws)[int(.025*DRAW)]


def _factual_control(fit):
    control=r33._fit_model(fit,48,"indicator")
    output={}
    for panel in PANELS:
        ww=worlds(panel)
        chosen=[]
        for w in ww:
            actions,effects=candidate_inputs(w,"active_information")
            losses=[]
            for act in (0,1):
                mean,variance,_=r33._estimate("indicator",control,
                              actions[act],effects[act],True)
                losses.append((mean-w["goal"])**2+variance)
            chosen.append(min((0,1),key=lambda a:(losses[a],a)))
        errors=regret_evaluator_only(ww,chosen)
        output[panel]={"mean_regret":statistics.fmean(errors),
                       "choices":chosen}
    return {"fit_validation_mse":control["fit_validation_mse"],
            "status":"HAND_ENGINEERED_CONVENTIONAL_REFERENCE_ONLY",
            "panels":output}


def judge(rows,freeze):
    s=freeze["development"]
    by={(x["arm"],x["seed"]):x for x in rows}
    erased=by["erased",SEEDS[0]]
    er_controls={p:{"regret":erased["panels"][p]["policies"]
                       ["active_information"]["mean_regret"],
                   "flips":erased["panels"][p]["policies"]
                       ["active_information"]["sign_pair_flips"]}
                 for p in PANELS}
    neg=all(x["regret"]>=s["erased_raw_regret_min_each"] and
            x["flips"]<=s["erased_raw_sign_flips_max_each"]
            for x in er_controls.values())
    verdict={}
    for arm in ("raw","explicit","native","history"):
        cells=[]
        for seed in (SEEDS if arm!="history" else SEEDS[:1]):
            row=by[arm,seed]
            panels={}
            for panel in PANELS:
                record=row["panels"][panel]
                pol=record["policies"]
                active=pol["active_information"]
                best=min((pol[k] for k in ("fixed_probe0","fixed_probe1")),
                         key=lambda z:z["mean_regret"])
                gain=best["mean_regret"]-active["mean_regret"]
                lower=_bootstrap_lower(active["pergroup_regret"],
                                       best["pergroup_regret"])
                ok=bool(active["mean_regret"]<=s["model_active_regret_max_each"]
                  and active["sign_pair_flips"]>=
                        s["model_sign_pair_flips_min_each"]
                  and record["fifth_erasure_regret_penalty"]>=
                        s["model_erasure_regret_penalty_min_each"]
                  and gain>=s["model_gain_over_best_fixed_min_each"]
                  and lower>s["model_bootstrap_lower_vs_best_fixed_gt"])
                panels[panel]={"active_regret":active["mean_regret"],
                    "active_sign_pair_flips":active["sign_pair_flips"],
                    "erasure_regret_penalty":record["fifth_erasure_regret_penalty"],
                    "gain_over_best_fixed":gain,"bootstrap_2p5_lower":lower,
                    "frozen_panel_pass":ok}
            cells.append({"seed":seed,"panels":panels,
                          "both_panel_pass":all(x["frozen_panel_pass"]
                                                for x in panels.values())})
        verdict[arm]={"seed_results":cells,
                      "winning_seeds":sum(v["both_panel_pass"] for v in cells)}
    return {"erased_training_negative_pass":neg,
            "erased_training_control":er_controls,
            "arms":verdict,
            "raw_development_pass":neg and
                verdict["raw"]["winning_seeds"]>=s["raw_seed_pass_min"],
            "explicit_development_pass":neg and
                verdict["explicit"]["winning_seeds"]>=s["explicit_seed_pass_min"],
            "native_development_pass":neg and
                verdict["native"]["winning_seeds"]>=s["native_seed_pass_min"],
            "independent_scientific_pass":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    winners=fit_pairs(fit)
    gen=torch.Generator(device="cpu")
    gen.manual_seed(21049999)
    samples=torch.randint(0,TRAIN_PAIRS,(STEPS,BATCH),
                          generator=gen).tolist()
    rows=[train_arm(arm,seed,fit,winners,samples) for arm,seed in ARMS]
    reference=_factual_control(fit)
    return {"schema":SCHEMA,"classification":freeze["classification"],
        "freeze_blob":FREEZE_BLOB,"freeze_parent":freeze["parent"]["sha"],
        "factual_episodes":len(fit),"training_pairs":TRAIN_PAIRS,
        "validation_pairs":384,
        "balanced_train_labels":{"action0":winners[:TRAIN_PAIRS].count(0),
                                 "action1":winners[:TRAIN_PAIRS].count(1)},
        "optimizer_updates":len(rows)*STEPS,"model_instances":len(rows),
        "models":rows,"factual_indicator_reference":reference,
        "judgment":judge(rows,freeze),"independent_scientific_pass":False,
        "new_external_effects":0,"spent_F7_replay":False,
        "new_mechanism_transfer":False,
        "capacity_FLOPs_matched_across_model_families":False,
        "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r36_balanced_choice_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R36_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R36_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
