"""F8-R37: prospective public identifiability-gated factual neural comparisons.

Only P80-visible fifth action/context may select informative FIT pairs. Factual
return-choice labels come from TWO independently selected FIT return episodes,
never counterfactual evaluator information. The donor-shuffle arm deliberately
breaks factual alignment and is negative-control data, not a real trajectory.
R37 is source-neutral development on a known quartic law, not science first-use.
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

from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r35_raw_probe_value as r35
from experiments import p85mr_f8r36_balanced_choice as r36

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/"experiments/p85mr_f8r37_informative_strata_freeze.json"
FREEZE_BLOB = "87216d66fb5ac4193b1b4ddccf95d316277ad710"
AMENDMENT = ROOT/"experiments/p85mr_f8r37_pretrain_score_amendment.json"
AMENDMENT_BLOB = "9f69abec4d497920fa64aa88b3b3b5ce4f617175"
SCHEMA="ttt-hybrid.f8r37-informative-stratified-contrast.v1"
SEEDS=(985201,985203,985207)
ARMS=tuple([(arm,seed) for arm in ("raw_mixed","raw_informative",
                                    "gated_mixed","gated_informative")
            for seed in SEEDS]+[("gated_informative_shuffled",SEEDS[0])])
PANELS=("dev","ood")
POLICIES=("active_information","fixed_probe0","fixed_probe1","seeded_random")
GROUPS=16
TRAIN_PAIRS=1152
FIT_PAIRS=1536
BATCH=16
STEPS=512
DRAW=10000
BOOT_SEED=21179999
SALT={"dev":21150000,"ood":21160000}
STATES={"dev":(.61,.69),"ood":(.55,.77)}
QABS={"dev":1.,"ood":.88}
B_STEPS={"dev":8,"ood":16}


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R37_ORIGINAL_FROZEN_PROTOCOL_DRIFT")
    amend_raw=AMENDMENT.read_bytes()
    if _blob(amend_raw)!=AMENDMENT_BLOB:
        raise ValueError("F8R37_PRETRAIN_SCORE_AMENDMENT_DRIFT")
    f=json.loads(raw)
    amend=json.loads(amend_raw)
    d,a,t,s,u=(f["factual_source"],f["arms"],f["training"],
               f["frozen_judgment"],f["authority"])
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_CAUSAL_INFORMATION_IDENTIFIABILITY_DEV_ONLY" or
        f["parent"]["head"]!="c3fbd1c1ca5bbe474ad2d7f2e9c8d3cbd6c262f9" or
        f["parent"]["r36_freeze_blob"]!=r36.FREEZE_BLOB or
        f["parent"]["r36_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r36_balanced_choice.py").read_bytes()) or
        d["train_pair_full_count"]!=TRAIN_PAIRS or
        d["train_pair_informative_count"]!=576 or
        d["train_informative_class0"]!=288 or
        d["train_informative_class1"]!=288 or
        d["heldout_pair_informative_count"]!=192 or
        a["total_models"]!=len(ARMS) or
        any(a[arm]!=list(SEEDS) for arm in
            ("raw_mixed","raw_informative","gated_mixed","gated_informative")) or
        a["gated_informative_shuffled"]!=[SEEDS[0]] or
        t["optimizer"]!="AdamW lr 0.002, weight_decay 0.01, gradient_norm_clip 1" or
        t["updates_each"]!=STEPS or t["total_optimizer_steps"]!=STEPS*len(ARMS) or
        t["pairs_per_update"]!=BATCH or
        t["sampler_seed"]!=21149999 or
        t["shuffled_donor_seed"]!=21149998 or
        t["shuffled_donor_max_abs_sign_phi"]!=.1 or
        t["checkpoints"]!=[128,256,512] or
        s["seed_wins_both_panels_at_least"]!=2 or
        s["each_panel_active_vs_best_fixed_regret_gain_min"]!=.14 or
        s["each_panel_sign_pair_decision_flips_min"]!=30 or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in u.values())):
        raise ValueError("F8R37_FROZEN_PROTOCOL_OR_AUTHORITY_DRIFT")
    if (amend["schema"]!="ttt-hybrid.f8r37-pretrain-ood-gain-amendment.v1"
        or amend["stage"]!="FROZEN_BEFORE_R37_IMPLEMENTATION_TRAINING_OR_RESULTS"
        or amend["parent"]["base_freeze_blob"]!=FREEZE_BLOB
        or amend["score_override"]["original_value"]!=.14
        or amend["score_override"]["dev_min"]!=.14
        or amend["score_override"]["ood_min"]!=.12
        or any(v is not False for v in amend["authority"].values())):
        raise ValueError("F8R37_UNQUALIFIED_PRETRAIN_THRESHOLD")
    r36.locked()
    return f,amend


def informative_id(fit,pair_id):
    """Public selected fifth action equals public context; no hidden sign."""
    if not 0<=pair_id<FIT_PAIRS or len(fit)!=3072:
        raise ValueError("F8R37_UNREGISTERED_PAIR")
    row=fit[2*pair_id]
    key=row["action"]
    ctx=bool(float(key[-1,2])>0.)
    selected=bool(float(key[4,6])>float(key[4,5]))
    return int(ctx==selected)


def selection_pools(fit,winners):
    if len(winners)!=FIT_PAIRS:
        raise ValueError("F8R37_FIT_LABEL_SIZE")
    informative=[i for i in range(TRAIN_PAIRS) if informative_id(fit,i)]
    noninformative=[i for i in range(TRAIN_PAIRS) if not informative_id(fit,i)]
    val_inf=[i for i in range(TRAIN_PAIRS,FIT_PAIRS)
             if informative_id(fit,i)]
    val_other=[i for i in range(TRAIN_PAIRS,FIT_PAIRS)
               if not informative_id(fit,i)]
    if (len(informative)!=576 or len(noninformative)!=576
        or len(val_inf)!=192 or len(val_other)!=192
        or [sum(winners[i]==cls for i in informative)
            for cls in (0,1)]!=[288,288]):
        raise ValueError("F8R37_INCORRECT_PUBLIC_INFORMATION_BALANCE")
    return {"all":list(range(TRAIN_PAIRS)),
            "informative":informative,
            "noninformative":noninformative,
            "heldout_informative":val_inf,
            "heldout_noninformative":val_other}


def fit_only_offset(fit,pools):
    """Only actually selected safe/noninformative fifth effects."""
    values=[r31._physical_delta(fit[2*i]["effect"][4,0])
            for i in pools["noninformative"]]
    return statistics.fmean(values)


def _donor_map(fit,pools):
    """Deterministic same-public-context factual shuffling, negative only."""
    info=pools["informative"]
    contexts={c:[i for i in info
                 if int(float(fit[2*i]["action"][-1,2])>0.)==c]
              for c in (0,1)}
    rng=random.Random(21149998)
    out={}
    for c in (0,1):
        donors=contexts[c].copy()
        rng.shuffle(donors)
        out.update(zip(contexts[c],donors))
    if set(out)!=set(info):
        raise ValueError("F8R37_DONOR_CONTEXT_MAPPING_INCOMPLETE")
    # Sign labels derived from the known author-generated FIT enumeration
    # ONLY here for the original negative-control audit, never passed to input
    # or sample selection: each 24-pair group has c(12), sign(6), mag/fifth.
    def generator_sign(pair_index):
        return -1 if pair_index%12<6 else 1
    phi=statistics.fmean(generator_sign(i)*generator_sign(out[i])
                          for i in info)
    if abs(phi)>.1:
        raise ValueError("F8R37_SHUFFLED_DONOR_CORRELATED_SIGN")
    return out,phi


class PublicGateMLP(nn.Module):
    def __init__(self,seed,offset):
        super().__init__()
        if seed not in SEEDS or not math.isfinite(offset):
            raise ValueError("F8R37_INVALID_PUBLIC_GATE")
        torch.manual_seed(seed+21130000)
        self.offset=float(offset)
        self.net=nn.Sequential(nn.Linear(52,64),nn.GELU(),
                               nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,actions,effects,*,erase_fifth=False):
        if (actions.ndim!=3 or effects.shape!=actions.shape or
            actions.shape[1] not in (14,22) or actions.shape[-1]!=16):
            raise ValueError("F8R37_INVALID_P80_GATE_INPUT")
        factual=effects[:,4,:]
        if erase_fifth:
            factual=torch.zeros_like(factual)
        context=actions[:,-1,2]>0
        fifth=actions[:,4,6]>actions[:,4,5]
        ret=actions[:,-1,6]>actions[:,-1,5]
        yes=(fifth==context).to(dtype=actions.dtype)
        return_yes=(ret==context).to(dtype=actions.dtype)
        joint=yes*return_yes
        physical=2.4*torch.atanh(torch.clamp(factual[:,0],-.979,.979))
        centered=physical-self.offset
        public=torch.stack((yes,return_yes,joint,joint*centered),dim=-1)
        features=torch.cat((actions[:,4,:],factual,actions[:,-1,:],public),
                            dim=-1)
        return self.net(features).squeeze(-1)


def model_for(arm,seed,offset):
    if (arm,seed) not in ARMS:
        raise ValueError("F8R37_UNREGISTERED_MODEL")
    if arm.startswith("raw"):
        return r35.ProbeValueMLP(seed)
    return PublicGateMLP(seed,offset)


def _pred(model,arm,actions,effects,*,erase=False):
    masked=effects.clone()
    masked[:,-1,:].zero_()
    return model(actions,masked,erase_fifth=erase)


def _batch(fit,winners,ids,*,donor_map=None):
    a,e,y=r36.batch(fit,winners,ids)
    if donor_map is not None:
        changed=e.clone()
        for k,i in enumerate(ids):
            if i not in donor_map:
                raise ValueError("F8R37_SHUFFLE_REQUIRES_INFORMATIVE")
            donor=fit[2*donor_map[i]]["effect"][4,:]
            changed[2*k,4,:]=donor
            changed[2*k+1,4,:]=donor
        e=changed
    return a,e,y


def _B(panel,group):
    if panel not in PANELS or not 0<=group<GROUPS:
        raise ValueError("F8R37_INVALID_NUISANCE_GROUP")
    return tuple((t%2,round(.021*math.sin((SALT[panel]+group)*.19+t*.47)
                  +.013*((t+group)%3)-.018,8))
                 for t in range(B_STEPS[panel]))


def world(panel,group,context,sign):
    if (panel not in PANELS or not 0<=group<GROUPS or
        context not in (0,1) or sign not in (-1,1)):
        raise ValueError("F8R37_INVALID_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":sign*QABS[panel],
            "return_state":STATES[panel][group%2],"goal":1.,
            "b_history":_B(panel,group)}


def worlds(panel):
    if panel not in PANELS:
        raise ValueError("F8R37_INVALID_SCORE_PANEL")
    return [world(panel,g,c,z) for g in range(GROUPS)
            for c in (0,1) for z in (-1,1)]


def _probe(world_,policy):
    if policy not in POLICIES:
        raise ValueError("F8R37_UNREGISTERED_PROBE_POLICY")
    if policy=="active_information":
        return world_["public_context"]
    if policy=="fixed_probe0":
        return 0
    if policy=="fixed_probe1":
        return 1
    raw=(f'f8r37/{world_["panel"]}/{world_["group"]}/'
         f'{world_["public_context"]}').encode()
    return hashlib.sha256(raw).digest()[0]&1


def candidate_inputs(w,policy,*,erase=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    fifth=_probe(w,policy)
    early=[r33.b._record(s,a,r33.r32.after(s,a,c,q),c)
           for s,a in r33.r32.EARLY]
    early.append(r33.b._record(r33.PROBE,fifth,
                              r33.r32.after(r33.PROBE,fifth,c,q),c))
    early.extend(r33.b._record(0.,a,e,c) for a,e in w["b_history"])
    a=torch.stack([r[0] for r in early])
    e=torch.stack([r[1] for r in early])
    if erase:
        e[4].zero_()
    options=[]
    for act in (0,1):
        key=r33.b._record(w["return_state"],act,w["return_state"],c)[0]
        options.append((torch.cat((a,key.unsqueeze(0))),
                        torch.cat((e,torch.zeros((1,16))))))
    return (torch.stack([z[0] for z in options]),
            torch.stack([z[1] for z in options]))


def _choices(model,arm,panel,policy,*,erase=False):
    w=worlds(panel)
    candidates=[candidate_inputs(x,policy,erase=erase) for x in w]
    a=torch.cat([x[0] for x in candidates])
    e=torch.cat([x[1] for x in candidates])
    model.eval()
    with torch.no_grad():
        vals=_pred(model,arm,a,e,erase=erase).tolist()
    if len(vals)!=2*len(w) or not all(math.isfinite(x) for x in vals):
        raise ValueError("F8R37_INVALID_PREDICTED_ACTIONS")
    return [min((0,1),key=lambda k:(vals[2*i+k],k)) for i in range(len(w))]


def score_policy(model,arm,panel,policy,*,erase=False):
    w=worlds(panel)
    actions=_choices(model,arm,panel,policy,erase=erase)
    regret=r36.regret_evaluator_only(w,actions)
    return {"choices":actions,"mean_regret":statistics.fmean(regret),
            "pergroup_regret":[statistics.fmean(regret[4*g:4*g+4])
                                for g in range(GROUPS)],
            "sign_pair_flips":sum(actions[i]!=actions[i+1]
                                  for i in range(0,4*GROUPS,2))}


def score_arm(model,arm):
    scores={}
    for p in PANELS:
        policy={k:score_policy(model,arm,p,k) for k in POLICIES}
        erased=score_policy(model,arm,p,"active_information",erase=True)
        act=policy["active_information"]
        scores[p]={"policies":policy,"erased_active":erased,
            "erasure_regret_penalty":erased["mean_regret"]-act["mean_regret"]}
    return scores


def _validation(model,arm,fit,winners,pools):
    model.eval()
    results={}
    for label,ids in (("informative",pools["heldout_informative"]),
                      ("noninformative",pools["heldout_noninformative"])):
        losses=[]
        accurate=0
        with torch.no_grad():
            for start in range(0,len(ids),32):
                j=ids[start:start+32]
                rows=[fit[2*i+k] for i in j for k in (0,1)]
                a=torch.stack([r["action"] for r in rows])
                e=torch.stack([r["effect"] for r in rows])
                out=_pred(model,arm,a,e).reshape(len(j),2)
                target=torch.tensor([winners[i] for i in j],
                                     dtype=torch.long)
                losses.append((len(j),float(F.cross_entropy(-out,target))))
                accurate+=int((out.argmin(dim=1)==target).sum())
        results[label]={"accuracy":accurate/len(ids),
                        "choice_CE":sum(n*v for n,v in losses)/len(ids),
                        "pairs":len(ids)}
    return results


def _hash(model):
    return r35._weight_hash(model)


def train_arm(arm,seed,fit,winners,pools,donors,offset,samples):
    model=model_for(arm,seed,offset)
    first=_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,
                                weight_decay=.01)
    trace=[]
    start=time.perf_counter()
    for update,keys in enumerate(samples,1):
        if arm.endswith("_mixed"):
            ids=keys
        else:
            informative=pools["informative"]
            ids=[informative[x%len(informative)] for x in keys]
        a,e,target=_batch(fit,winners,ids,
                          donor_map=donors if arm.endswith("_shuffled") else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred=_pred(model,arm,a,e)
        if pred.shape!=(2*BATCH,):
            raise ValueError("F8R37_INVALID_PAIR_PREDICTION")
        loss=F.cross_entropy(-pred.reshape(BATCH,2),target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R37_NONFINITE_CHOICE_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0))
                    for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R37_INVALID_FACTUAL_GRADIENT")
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if update in (128,256,512):
            trace.append({"updates":update,"train_choice_ce":float(loss.detach()),
                         "heldout":_validation(model,arm,fit,winners,pools)})
    last=_hash(model)
    if first==last or len(trace)!=3:
        raise ValueError("F8R37_MODEL_NOT_UPDATED")
    score=score_arm(model,arm)
    restored=model_for(arm,seed,offset)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if _hash(restored)!=last or any(
        _choices(restored,arm,p,"active_information")!=
        score[p]["policies"]["active_information"]["choices"]
        for p in PANELS):
        raise ValueError("F8R37_COLD_RESTORE_DRIFT")
    return {"arm":arm,"seed":seed,"updates":STEPS,
            "parameters":sum(p.numel() for p in model.parameters()),
            "initial_sha256":first,"trained_sha256":last,
            "train_wall_seconds":time.perf_counter()-start,
            "checkpoints":trace,"cold_restore":True,
            "panels":score}


def _paired_lower(a,b):
    rng=random.Random(BOOT_SEED)
    values=[]
    for _ in range(DRAW):
        sample=[rng.randrange(GROUPS) for j in range(GROUPS)]
        values.append(statistics.fmean(b[i]-a[i] for i in sample))
    return sorted(values)[int(.025*DRAW)]


def judge(rows,freeze,amendment):
    rules=freeze["frozen_judgment"]
    by={(r["arm"],r["seed"]):r for r in rows}
    shuffled=by[("gated_informative_shuffled",SEEDS[0])]
    shuffle_control={}
    for panel in PANELS:
        rec=shuffled["panels"][panel]["policies"]["active_information"]
        shuffle_control[panel]={"regret":rec["mean_regret"],
              "flips":rec["sign_pair_flips"],
              "pass":rec["mean_regret"]>=rules["shuffled_control_regret_min"]
               and rec["sign_pair_flips"]<=rules["shuffled_control_sign_flips_max"]}
    native=[]
    for seed in SEEDS:
        record=by[("gated_informative",seed)]
        mixed=by[("gated_mixed",seed)]
        perpanel={}
        for panel in PANELS:
            scores=record["panels"][panel]
            active=scores["policies"]["active_information"]
            best=min((scores["policies"][p] for p in ("fixed_probe0","fixed_probe1")),
                     key=lambda x:x["mean_regret"])
            mixed_regret=mixed["panels"][panel]["policies"]["active_information"]["mean_regret"]
            gain=best["mean_regret"]-active["mean_regret"]
            lower=_paired_lower(active["pergroup_regret"],best["pergroup_regret"])
            target=(amendment["score_override"]["dev_min"] if panel=="dev"
                    else amendment["score_override"]["ood_min"])
            passed=bool(
                active["mean_regret"]<=rules["each_panel_active_regret_max"]
                and active["sign_pair_flips"]>=rules["each_panel_sign_pair_decision_flips_min"]
                and scores["erasure_regret_penalty"]>=
                    rules["each_panel_erasure_regret_penalty_min"]
                and gain>=target
                and lower>rules["each_panel_paired_bootstrap_lower_vs_fixed_gt"]
                and (mixed_regret-active["mean_regret"])>=
                    rules["each_panel_gain_over_gated_mixed_same_seed_min"])
            perpanel[panel]={"active_regret":active["mean_regret"],
                "gain_vs_best_fixed":gain,"bootstrap_lower":lower,
                "gain_vs_gated_mixed":mixed_regret-active["mean_regret"],
                "erasure_penalty":scores["erasure_regret_penalty"],
                "sign_pair_flips":active["sign_pair_flips"],
                "pretrain_amended_fixed_gain_threshold":target,
                "frozen_pass":passed}
        native.append({"seed":seed,"panels":perpanel,
                       "both_panel_pass":all(p["frozen_pass"]
                                            for p in perpanel.values())})
    wins=sum(x["both_panel_pass"] for x in native)
    return {"gated_informative_seeds_both_pass":wins,
        "gated_informative_development_pass":wins>=rules["seed_wins_both_panels_at_least"]
                                           and all(x["pass"] for x in shuffle_control.values()),
        "gated_informative":native,"shuffled_negative_control":shuffle_control,
        "independent_scientific_pass":False}


def run():
    freeze,amendment=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    winners=r36.fit_pairs(fit)
    pools=selection_pools(fit,winners)
    offset=fit_only_offset(fit,pools)
    donor,phi=_donor_map(fit,pools)
    g=torch.Generator(device="cpu")
    g.manual_seed(21149999)
    samples=torch.randint(0,TRAIN_PAIRS,(STEPS,BATCH),
                          generator=g).tolist()
    rows=[train_arm(a,seed,fit,winners,pools,donor,offset,samples)
          for a,seed in ARMS]
    return {"schema":SCHEMA,"classification":freeze["classification"],
        "freeze_blob":FREEZE_BLOB,"amendment_blob":AMENDMENT_BLOB,
        "frozen_parent":freeze["parent"]["head"],
        "fit_factual_episodes":len(fit),
        "train_factual_pairs":TRAIN_PAIRS,
        "train_informative_pairs":len(pools["informative"]),
        "train_noninformative_pairs":len(pools["noninformative"]),
        "validation_informative_pairs":len(pools["heldout_informative"]),
        "validation_noninformative_pairs":len(pools["heldout_noninformative"]),
        "fitted_safe_probe_physical_offset":offset,
        "shuffled_negative_signed_phi_audit":phi,
        "model_instances":len(rows),"optimizer_updates":len(rows)*STEPS,
        "models":rows,"judgment":judge(rows,freeze,amendment),
        "independent_scientific_pass":False,
        "original_external_effects":0,"spent_F7_replay":False,
        "new_mechanism_transfer":False,"deployment":"NONE"}


def retain_ci_receipt(x):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r37_informative_strata_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R37_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R37_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
