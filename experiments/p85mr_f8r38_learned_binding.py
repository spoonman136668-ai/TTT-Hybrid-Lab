"""F8-R38: frozen learned causal-binding, raw/additive and native comparisons.

Only actually selected fifth-action consequences and two independently selected
FIT return-action consequences enter training. A learned trilinear binder, unlike
R37, receives no hard-coded action==public_context indicator. Novel cubic law
is synthetic source-neutral only; original external scientific first-use forbidden.
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
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r38_learned_binding_freeze.json"
FREEZE_BLOB="3ca539e10cdf81e287d003453deaaba9ac33555d"
AMENDMENT=ROOT/"experiments/p85mr_f8r38_pretrain_cubic_amendment.json"
AMENDMENT_BLOB="33917d31d63c491312702a4a77c7838f1012dc7c"
SCHEMA="ttt-hybrid.f8r38-learned-trilinear-causal-binding.v1"
SEEDS=(985201,985203,985207)
ARMS=tuple([(name,seed) for name in ("binder","additive","raw")
            for seed in SEEDS]+[(name,SEEDS[0]) for name in
            ("native","gru","ssm","erased","shuffled","engineered")])
PANELS=("dev","ood","synthetic_cubic_transfer")
POLICIES=("active_information","fixed_probe0","fixed_probe1","seeded_random")
SALT={"dev":21250000,"ood":21260000,"synthetic_cubic_transfer":21270000}
STATES={"dev":(.61,.73),"ood":(.58,.76),"synthetic_cubic_transfer":(.61,.73)}
QABS={"dev":1.,"ood":.93,"synthetic_cubic_transfer":1.}
B_STEPS={"dev":8,"ood":16,"synthetic_cubic_transfer":16}
GROUPS=16
TRAIN_PAIRS=1152
FIT_PAIRS=1536
BATCH=16
STEPS=512
BOOT_DRAW=10000
BOOT_SEED=21279999


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    amendment_raw=AMENDMENT.read_bytes()
    if _blob(raw)!=FREEZE_BLOB or _blob(amendment_raw)!=AMENDMENT_BLOB:
        raise ValueError("F8R38_FROZEN_BLOB_OR_AMENDMENT_DRIFT")
    f=json.loads(raw)
    a=json.loads(amendment_raw)
    d,arch,t,s,auth=(f["factual_training"],f["architectures"],
                    f["training"],f["criterion"],f["authority"])
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULT" or
        f["classification"]!="SOURCE_NEUTRAL_NEURAL_BINDING_MECHANISM_DEVELOPMENT_ONLY" or
        f["parent"]["head"]!="a5bbb361387bb49c7b2fabeb820a1216c56e6ef7" or
        f["parent"]["r37_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r37_informative_strata.py").read_bytes()) or
        f["parent"]["r37_freeze_blob"]!=r37.FREEZE_BLOB or
        f["parent"]["r37_pretrain_amendment_blob"]!=r37.AMENDMENT_BLOB or
        d["sampler_seed"]!=21249999 or
        arch["instances"]!=len(ARMS) or
        arch["three_seed_arms"]!=list(SEEDS) or
        t["updates_each"]!=STEPS or
        t["total_updates"]!=STEPS*len(ARMS) or
        t["pairs_per_step"]!=BATCH or
        t["validation_updates"]!=[128,256,512] or
        f["evaluation"]["panels"]!=list(PANELS) or
        f["evaluation"]["bootstrap_seed"]!=BOOT_SEED or
        f["evaluation"]["bootstrap_draws"]!=BOOT_DRAW or
        s["binder_seeds_both_original_law_panels_min"]!=2 or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in auth.values())):
        raise ValueError("F8R38_FROZEN_PROTOCOL_OR_AUTHORITY_DRIFT")
    if (a["schema"]!="ttt-hybrid.f8r38-pretrain-cubic-null-early-amendment.v1" or
        a["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_OR_ANY_R38_TRAINING_OR_RESULTS" or
        a["parent"]["R38_freeze_blob"]!=FREEZE_BLOB or
        a["score_only_novel_law"]!=
        "baseline(s)+2.7*q*s*(s^2-0.15^2) if selected action==context" or
        any(v is not False for v in a["authority"].values())):
        raise ValueError("F8R38_PRETRAIN_CUBIC_PHYSICS_DRIFT")
    r37.locked()
    return f,a


class TrainableBinding(nn.Module):
    """Train triple product of raw P80 factual key/measurement/action embeddings."""

    def __init__(self,seed,variant="factorized"):
        super().__init__()
        if seed not in SEEDS or variant not in ("factorized","additive"):
            raise ValueError("F8R38_INVALID_BINDING_VARIANT")
        self.variant=variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+21230000)
            self.fifth=nn.Linear(16,16)
            self.return_key=nn.Linear(16,16)
            self.measurement=nn.Linear(16,16)
            self.value=nn.Sequential(nn.Linear(64,64),nn.GELU(),
                       nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,actions,effects):
        if (actions.ndim!=3 or actions.shape!=effects.shape or
            actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
            raise ValueError("F8R38_INVALID_RAW_P80_FACTS")
        f=torch.tanh(self.fifth(actions[:,4,:]))
        r=torch.tanh(self.return_key(actions[:,-1,:]))
        o=torch.tanh(self.measurement(effects[:,4,:]))
        # Learned embeddings, NOT hard-coded public action equality or q sign.
        binding=f*r*o if self.variant=="factorized" else f+r+o
        return self.value(torch.cat((f,r,o,binding),dim=-1)).squeeze(-1)


class SubstratePlusBinding(nn.Module):
    """Reuse current native, GRU or SSM R34 substrate without cross-project edits."""

    def __init__(self,arm,seed):
        super().__init__()
        mapping={"native":"native_factual_memory",
                 "gru":"gru_factual_memory",
                 "ssm":"ssm_factual_memory"}
        if arm not in mapping or seed!=SEEDS[0]:
            raise ValueError("F8R38_INVALID_NATIVE_CONTROL")
        self.substrate=r34.FactualMemory(mapping[arm],seed)
        self.binder=TrainableBinding(seed)

    def forward(self,actions,effects,*,zero_backbone=False,
                zero_substrate=False,zero_binding=False):
        binder=torch.zeros(actions.shape[0],dtype=actions.dtype,device=actions.device)
        substrate=binder
        if not zero_binding:
            binder=self.binder(actions,effects)
        if not zero_substrate:
            substrate=self.substrate(actions,effects,
                                      ablate_backbone=zero_backbone)
        return substrate+binder


def model_for(arm,seed,offset):
    if (arm,seed) not in ARMS:
        raise ValueError("F8R38_UNREGISTERED_TRAIN_ARM")
    if arm=="binder" or arm in ("erased","shuffled"):
        return TrainableBinding(seed)
    if arm=="additive":
        return TrainableBinding(seed,"additive")
    if arm=="raw":
        return r35.ProbeValueMLP(seed)
    if arm in ("native","gru","ssm"):
        return SubstratePlusBinding(arm,seed)
    if arm=="engineered":
        return r37.PublicGateMLP(seed,offset)
    raise ValueError("F8R38_UNKNOWN_MODEL")


def predict(model,arm,actions,effects,*,erase=False,
            zero_backbone=False,zero_substrate=False,zero_binding=False):
    if (actions.ndim!=3 or actions.shape!=effects.shape or
        actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
        raise ValueError("F8R38_INVALID_PREACTION_FACTS")
    clean=effects.clone()
    clean[:,-1,:].zero_()
    if erase or arm=="erased":
        clean[:,4,:].zero_()
    if arm in ("native","gru","ssm"):
        return model(actions,clean,zero_backbone=zero_backbone,
                     zero_substrate=zero_substrate,zero_binding=zero_binding)
    return model(actions,clean)


def selected_batch(fit,winners,ids,*,donors=None):
    a,e,target=r36.batch(fit,winners,ids)
    if donors is not None:
        changed=e.clone()
        for j,index in enumerate(ids):
            if index not in donors:
                raise ValueError("F8R38_SHUFFLE_NONINFORMATIVE")
            selected=fit[2*donors[index]]["effect"][4,:]
            changed[2*j,4,:]=selected
            changed[2*j+1,4,:]=selected
        e=changed
    return a,e,target


def _base(s):
    return .25*s-.08+.05*math.sin(3*s)


def physical(s,action,context,q,panel):
    if (panel not in PANELS or action not in (0,1) or
        context not in (0,1) or not math.isfinite(q)):
        raise ValueError("F8R38_INVALID_PHYSICAL_SCORE")
    if panel!="synthetic_cubic_transfer":
        return r33.r32.after(s,action,context,q)
    # PRETRAIN-AMENDED score-only cubic law with BOTH ±.15 early states null.
    return _base(s)+(2.7*q*s*(s*s-.15**2) if action==context else 0.)


def _B(panel,group):
    if panel not in PANELS or not 0<=group<GROUPS:
        raise ValueError("F8R38_UNREGISTERED_NUISANCE")
    return tuple((t%2,round(.021*math.sin((SALT[panel]+group)*.19+t*.47)
                    +.013*((t+group)%3)-.018,8))
                 for t in range(B_STEPS[panel]))


def world(panel,group,context,sign):
    if (panel not in PANELS or not 0<=group<GROUPS or
        context not in (0,1) or sign not in (-1,1)):
        raise ValueError("F8R38_UNREGISTERED_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":sign*QABS[panel],
            "return_state":STATES[panel][group%2],"goal":1.,
            "b_history":_B(panel,group)}


def worlds(panel):
    if panel not in PANELS:
        raise ValueError("F8R38_UNREGISTERED_SCORE_PANEL")
    return [world(panel,g,c,z) for g in range(GROUPS)
            for c in (0,1) for z in (-1,1)]


def _probe(w,policy):
    if policy not in POLICIES:
        raise ValueError("F8R38_UNREGISTERED_POLICY")
    if policy=="active_information":
        return w["public_context"]
    if policy=="fixed_probe0":
        return 0
    if policy=="fixed_probe1":
        return 1
    payload=(f'f8r38/{w["panel"]}/{w["group"]}/'
             f'{w["public_context"]}').encode()
    return hashlib.sha256(payload).digest()[0]&1


def candidate_inputs(w,policy,*,erase=False):
    panel,c,q=w["panel"],w["public_context"],w["q_evaluator_only"]
    probe=_probe(w,policy)
    # World generator produces only actually selected factual events.
    first=[r33.b._record(s,act,physical(s,act,c,q,panel),c)
           for s,act in r33.r32.EARLY]
    first.append(r33.b._record(r33.PROBE,probe,
                              physical(r33.PROBE,probe,c,q,panel),c))
    first.extend(r33.b._record(0.,a,obs,c) for a,obs in w["b_history"])
    actions=torch.stack([x[0] for x in first])
    effects=torch.stack([x[1] for x in first])
    if erase:
        effects[4].zero_()
    results=[]
    for act in (0,1):
        key=r33.b._record(w["return_state"],act,w["return_state"],c)[0]
        results.append((torch.cat((actions,key.unsqueeze(0))),
                        torch.cat((effects,torch.zeros((1,16))))))
    return torch.stack([x[0] for x in results]),torch.stack([x[1] for x in results])


def regret_evaluator_only(rows,choices):
    """Only evaluator accesses unchosen action-consequence effects."""
    if len(rows)!=len(choices):
        raise ValueError("F8R38_EVALUATOR_SHAPE")
    out=[]
    for w,choice in zip(rows,choices):
        costs=[(physical(w["return_state"],a,w["public_context"],
                         w["q_evaluator_only"],w["panel"])-w["goal"])**2
               for a in (0,1)]
        out.append(max(0.,costs[choice]-min(costs)))
    return out


def _actions(model,arm,panel,policy,*,erase=False,
             zero_backbone=False,zero_substrate=False,zero_binding=False):
    ww=worlds(panel)
    facts=[candidate_inputs(w,policy,erase=erase) for w in ww]
    actions=torch.cat([a for a,e in facts])
    effects=torch.cat([e for a,e in facts])
    model.eval()
    with torch.no_grad():
        pred=predict(model,arm,actions,effects,erase=erase,
                     zero_backbone=zero_backbone,zero_substrate=zero_substrate,
                     zero_binding=zero_binding)
    if pred.shape!=(2*len(ww),) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R38_INVALID_DECISION_VALUE")
    logits=pred.tolist()
    return [min((0,1),key=lambda act:(logits[2*i+act],act))
            for i in range(len(ww))]


def score_policy(model,arm,panel,policy,*,erase=False,
                 zero_backbone=False,zero_substrate=False,zero_binding=False):
    ww=worlds(panel)
    choices=_actions(model,arm,panel,policy,erase=erase,
                     zero_backbone=zero_backbone,
                     zero_substrate=zero_substrate,
                     zero_binding=zero_binding)
    regrets=regret_evaluator_only(ww,choices)
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[4*g:4*g+4])
                               for g in range(GROUPS)],
            "sign_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,4*GROUPS,2))}


def score_arm(model,arm):
    results={}
    for panel in PANELS:
        policies={p:score_policy(model,arm,panel,p) for p in POLICIES}
        erased=score_policy(model,arm,panel,"active_information",erase=True)
        backbone=score_policy(model,arm,panel,"active_information",
                  zero_backbone=True) if arm in ("native","gru","ssm") else None
        binder_only=score_policy(model,arm,panel,"active_information",
                  zero_substrate=True) if backbone else None
        substrate_only=score_policy(model,arm,panel,"active_information",
                  zero_binding=True) if backbone else None
        active=policies["active_information"]
        results[panel]={"policies":policies,"erased_active":erased,
                        "zeroed_backbone":backbone,"only_binding":binder_only,
                        "only_substrate":substrate_only,
                        "erasure_penalty":
                            erased["mean_regret"]-active["mean_regret"],
                        "zero_backbone_regret_increase":
                            backbone["mean_regret"]-active["mean_regret"]
                            if backbone else None,
                        "no_substrate_regret_increase":
                            binder_only["mean_regret"]-active["mean_regret"]
                            if binder_only else None}
    return results


def validation(model,arm,fit,winners,ids):
    values=[]
    good=0
    model.eval()
    with torch.no_grad():
        for i in range(0,len(ids),32):
            sample=ids[i:i+32]
            rows=[fit[2*k+j] for k in sample for j in (0,1)]
            a=torch.stack([r["action"] for r in rows])
            e=torch.stack([r["effect"] for r in rows])
            truth=torch.tensor([winners[k] for k in sample],dtype=torch.long)
            pred=predict(model,arm,a,e).reshape(len(sample),2)
            values.append((len(sample),float(F.cross_entropy(-pred,truth))))
            good+=int((pred.argmin(dim=1)==truth).sum())
    return {"pairs":len(ids),"accuracy":good/len(ids),
            "choice_CE":sum(n*v for n,v in values)/len(ids)}


def _hash(model):
    return r35._weight_hash(model)


def train_arm(arm,seed,fit,winners,pools,donors,offset,samples):
    model=model_for(arm,seed,offset)
    before=_hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),
            lr=.001 if arm in ("native","gru","ssm") else .002,
            weight_decay=.01)
    trace=[]
    t0=time.perf_counter()
    info=pools["informative"]
    for update,keys in enumerate(samples,1):
        selected=[info[i] for i in keys]
        a,e,y=selected_batch(fit,winners,selected,
                  donors=donors if arm=="shuffled" else None)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits=predict(model,arm,a,e)
        if logits.shape!=(2*BATCH,):
            raise ValueError("F8R38_INVALID_TRAIN_LOGIT_SHAPE")
        loss=F.cross_entropy(-logits.reshape(BATCH,2),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R38_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0))
                    for g in grads) or
            any(g is not None and not bool(torch.isfinite(g).all())
                for g in grads)):
            raise ValueError("F8R38_ZERO_OR_NONFINITE_GRADIENT")
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if update in (128,256,512):
            trace.append({"updates":update,"last_loss":float(loss.detach()),
                "heldout_informative":validation(
                    model,arm,fit,winners,pools["heldout_informative"]),
                "heldout_noninformative":validation(
                    model,arm,fit,winners,pools["heldout_noninformative"])})
    final=_hash(model)
    if final==before or len(trace)!=3:
        raise ValueError("F8R38_NONLEARNING_MODEL")
    scores=score_arm(model,arm)
    restored=model_for(arm,seed,offset)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if _hash(restored)!=final or any(
        _actions(restored,arm,p,"active_information") !=
        scores[p]["policies"]["active_information"]["choices"]
        for p in PANELS):
        raise ValueError("F8R38_COLD_RESTORE_DRIFT")
    return {"arm":arm,"seed":seed,"optimizer_updates":STEPS,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_weight_sha256":before,"trained_weight_sha256":final,
            "train_wall_seconds":time.perf_counter()-t0,
            "cold_restore_actions_equal":True,
            "training_checkpoints":trace,"panels":scores}


def _bootstrap_lower(active,fixed):
    rand=random.Random(BOOT_SEED)
    deltas=[]
    for _ in range(BOOT_DRAW):
        sample=[rand.randrange(GROUPS) for j in range(GROUPS)]
        deltas.append(statistics.fmean(fixed[i]-active[i] for i in sample))
    return sorted(deltas)[int(.025*BOOT_DRAW)]


def judge(models,freeze):
    rules=freeze["criterion"]
    by={(r["arm"],r["seed"]):r for r in models}
    seed_results=[]
    transfer=[]
    for seed in SEEDS:
        record=by["binder",seed]
        raw=by["raw",seed]
        row={}
        for panel in PANELS:
            vals=record["panels"][panel]
            active=vals["policies"]["active_information"]
            fixed=min((vals["policies"][p] for p in
                       ("fixed_probe0","fixed_probe1")),
                      key=lambda x:x["mean_regret"])
            lower=_bootstrap_lower(active["pergroup_regret"],
                                   fixed["pergroup_regret"])
            gain=fixed["mean_regret"]-active["mean_regret"]
            better_raw=raw["panels"][panel]["policies"]["active_information"]["mean_regret"]-active["mean_regret"]
            if panel=="synthetic_cubic_transfer":
                passed=bool(
                    active["mean_regret"]<=rules["synthetic_cubic_active_regret_max"] and
                    active["sign_pair_flips"]>=rules["synthetic_cubic_sign_flips_min"] and
                    vals["erasure_penalty"]>=rules["synthetic_cubic_erasure_penalty_min"])
            else:
                passed=bool(
                    active["mean_regret"]<=rules["binder_dev_ood_active_regret_max"] and
                    active["sign_pair_flips"]>=rules["binder_sign_flips_min_each"] and
                    vals["erasure_penalty"]>=rules["binder_fifth_erasure_regret_penalty_min_each"] and
                    gain>=rules["binder_gain_vs_best_fixed_min_each"] and
                    lower>rules["binder_paired_bootstrap_lower_gt"] and
                    better_raw>=rules["binder_gain_over_raw_same_seed_min_each"])
            row[panel]={"regret":active["mean_regret"],
                  "sign_pair_flips":active["sign_pair_flips"],
                  "erasure_penalty":vals["erasure_penalty"],
                  "best_fixed_gain":gain,"bootstrap_2p5_lower":lower,
                  "gain_vs_raw":better_raw,"frozen_pass":passed}
        seed_results.append({"seed":seed,"panels":row,
                             "dev_ood_pass":row["dev"]["frozen_pass"] and
                                            row["ood"]["frozen_pass"],
                             "cubic_pass":row["synthetic_cubic_transfer"]["frozen_pass"]})
    positive=sum(s["dev_ood_pass"] for s in seed_results)
    transferred=sum(s["cubic_pass"] for s in seed_results)
    negative={}
    for name in ("erased","shuffled"):
        controls={}
        for panel in PANELS:
            active=by[name,SEEDS[0]]["panels"][panel]["policies"]["active_information"]
            ok=(active["sign_pair_flips"]<=rules[
                    "erased_sign_flips_max_each" if name=="erased"
                    else "shuffled_sign_flips_max_each"] and
                (name!="shuffled" or active["mean_regret"]>=
                     rules["shuffled_regret_min_each"]))
            controls[panel]={"regret":active["mean_regret"],
                             "sign_pair_flips":active["sign_pair_flips"],
                             "frozen_negative_pass":ok}
        negative[name]=controls
    control_pass=all(v["frozen_negative_pass"] for z in negative.values()
                     for v in z.values())
    return {"binder_seed_results":seed_results,"binder_dev_ood_winning_seeds":positive,
            "source_neutral_binder_development_pass":
                 positive>=rules["binder_seeds_both_original_law_panels_min"] and control_pass,
            "cubic_winning_seeds":transferred,
            "synthetic_cubic_transfer_development_pass":
                 transferred>=rules["synthetic_cubic_transfer_seeds_both_panels_min"],
            "negative_controls":negative,"independent_scientific_pass":False}


def run():
    freeze,amend=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    winners=r36.fit_pairs(fit)
    pools=r37.selection_pools(fit,winners)
    offset=r37.fit_only_offset(fit,pools)
    donors,phi=r37._donor_map(fit,pools)
    generator=torch.Generator(device="cpu")
    generator.manual_seed(21249999)
    samples=torch.randint(0,len(pools["informative"]),(STEPS,BATCH),
                          generator=generator).tolist()
    models=[train_arm(arm,seed,fit,winners,pools,donors,offset,samples)
            for arm,seed in ARMS]
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "pretrain_amendment_blob":AMENDMENT_BLOB,
            "classification":freeze["classification"],
            "frozen_parent":freeze["parent"]["head"],
            "fit_factual_episodes":len(fit),
            "training_informative_factual_pairs":len(pools["informative"]),
            "heldout_informative_factual_pairs":len(pools["heldout_informative"]),
            "heldout_noninformative_factual_pairs":len(pools["heldout_noninformative"]),
            "shuffled_donor_signed_phi":phi,
            "model_instances":len(models),"optimizer_updates":len(models)*STEPS,
            "original_external_effects":0,"spent_F7_replay":False,
            "capacity_FLOPs_matched_across_architectures":False,
            "independent_scientific_pass":False,"deployment":"NONE",
            "models":models,"judgment":judge(models,freeze)}


def retain_ci_receipt(data):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r38_learned_binding_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R38_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R38_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
