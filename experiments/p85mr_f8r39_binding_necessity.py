"""F8-R39: frozen R38-weight replay, learned-product necessity and novel-law transfer.

This is a source-neutral *postfit mechanistic ablation* experiment. The exact
R38 training data, optimizer, seeds, and selected factual pair labels are
replayed verbatim; never fit on new law panels. Only the evaluator knows q.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r39_binding_necessity_freeze.json"
FREEZE_BLOB="f9f406ce6c0b49a71ab7d2d9b46857583d0502d0"
SCHEMA="ttt-hybrid.f8r39-binding-channel-necessity-and-transfer.v1"
SEEDS=(985201,985203,985207)
STEPS=512
BATCH=16
GROUPS=16
SALT={"oscillatory_even":21350000,"nonlinear_odd":21360000}
QABS={"oscillatory_even":1.,"nonlinear_odd":.86}
STATES={"oscillatory_even":(.63,.71),"nonlinear_odd":(.55,.79)}
B_STEPS={"oscillatory_even":8,"nonlinear_odd":16}
PANELS=("oscillatory_even","nonlinear_odd")
ORIGINAL_PANELS=("dev","ood","synthetic_cubic_transfer")
# SHA-256 of bytewise action vectors from both independently checked original
# R38 Windows/Ubuntu G4 exact-SHA artifacts, NOT post-R39 performance targets.
REFERENCE_CHOICES={985201:"010f3da83819e16ab936600e3f15eb85b272fa68119717a201d72be3fc712304",
                   985203:"7c8975e1e60a5c8337f28edf8c33c3b180360b7279644a9bc1af3c51e6220bf5",
                   985207:"010f3da83819e16ab936600e3f15eb85b272fa68119717a201d72be3fc712304"}


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R39_FROZEN_SOURCE_BLOB_DRIFT")
    f=json.loads(raw)
    p=f["parent"]
    t=f["immutable_training"]
    m=f["mechanism_attribution"]
    x=f["new_synthetic_transfer"]
    c=f["frozen_judgment"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R39_IMPLEMENTATION_TRAINING_AND_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_FROZEN_R38_MECHANISTIC_ABLATION_AND_SYNTHETIC_TRANSFER_DEV_ONLY" or
        p["head"]!="99a2fa617ece5529534b977f198f4e7d7d9c69f4" or
        p["r38_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r38_learned_binding.py").read_bytes()) or
        p["r38_freeze_blob"]!=r38.FREEZE_BLOB or
        p["r38_pretrain_amendment_blob"]!=r38.AMENDMENT_BLOB or
        t["train_seeds"]!=list(SEEDS) or
        t["updates_per_seed"]!=STEPS or
        t["total_replayed_updates"]!=STEPS*len(SEEDS) or
        t["pair_batch"]!=BATCH or
        m["positive_quartic_seed_reference"]!=[985201,985207] or
        x["groups"]!=GROUPS or
        x["states"]!={k:list(v) for k,v in STATES.items()} or
        x["q_magnitude"]!=QABS or
        x["score_seeds"]!=SALT or
        c["mechanism_product_necessary_seeds_min"]!=2 or
        c["transfer_both_new_laws_seed_success_min"]!=2 or
        c["independent_scientific_pass"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise ValueError("F8R39_PROTOCOL_OR_AUTHORITY_DRIFT")
    r38.locked()
    return f


def _generator_law(s,candidate,context,q,panel):
    """Only the synthetic data generator/evaluator can see q."""
    if panel not in PANELS or candidate not in (0,1) or context not in (0,1):
        raise ValueError("F8R39_UNREGISTERED_GENERATOR")
    baseline=.25*s-.08+.05*math.sin(3*s)
    if candidate!=context:
        return baseline
    if panel=="oscillatory_even":
        shift=2.9*q*(s*s-.15**2)*(.75+.5*math.cos(2*s))
    else:
        shift=2.2*q*s*(s*s-.15**2)*(1.25-.25*math.sin(6*s))
    return baseline+shift


def _B(panel,g):
    if panel not in PANELS or type(g) is not int or not 0<=g<GROUPS:
        raise ValueError("F8R39_UNREGISTERED_B_STREAM")
    return tuple((t%2,round(.017*math.sin((SALT[panel]+g)*.23+t*.61)
                 +.012*((t+g)%3)-.016,8)) for t in range(B_STEPS[panel]))


def world(panel,g,context,sign):
    if (panel not in PANELS or type(g) is not int or not 0<=g<GROUPS or
        context not in (0,1) or sign not in (-1,1)):
        raise ValueError("F8R39_UNREGISTERED_WORLD")
    return {"panel":panel,"group":g,"public_context":context,
            "q_evaluator_only":sign*QABS[panel],
            "return_state":STATES[panel][g%2],"goal":1.,
            "b_history":_B(panel,g)}


def worlds(panel):
    if panel not in PANELS:
        raise ValueError("F8R39_UNREGISTERED_PANEL")
    return [world(panel,g,c,sign) for g in range(GROUPS)
            for c in (0,1) for sign in (-1,1)]


def candidate_inputs(w,*,erase=False):
    panel,c,q=w["panel"],w["public_context"],w["q_evaluator_only"]
    # Only selected factual early and informative fifth intervention.
    early=[r33.b._record(s,a,_generator_law(s,a,c,q,panel),c)
           for s,a in r33.r32.EARLY]
    early.append(r33.b._record(r33.PROBE,c,
                              _generator_law(r33.PROBE,c,c,q,panel),c))
    early.extend(r33.b._record(0.,a,eff,c) for a,eff in w["b_history"])
    actions=torch.stack([v[0] for v in early])
    effects=torch.stack([v[1] for v in early])
    if erase:
        effects[4].zero_()
    choices=[]
    for act in (0,1):
        current_key=r33.b._record(w["return_state"],act,w["return_state"],c)[0]
        choices.append((torch.cat((actions,current_key.unsqueeze(0))),
                        torch.cat((effects,torch.zeros((1,16))))))
    return torch.stack([v[0] for v in choices]),torch.stack([v[1] for v in choices])


def _prediction(model,a,e,*,zero_product=False):
    if a.ndim!=3 or e.shape!=a.shape or a.shape[-1]!=16:
        raise ValueError("F8R39_EXPECT_RAW_P80")
    effect=e.clone()
    effect[:,-1,:].zero_()
    f=torch.tanh(model.fifth(a[:,4,:]))
    r=torch.tanh(model.return_key(a[:,-1,:]))
    measured=torch.tanh(model.measurement(effect[:,4,:]))
    # Mechanistic ablation: no parameter changes, no retraining, same readout.
    joint=f*r*measured
    if zero_product:
        joint=torch.zeros_like(joint)
    return model.value(torch.cat((f,r,measured,joint),dim=-1)).squeeze(-1)


def _score_new(model,panel,*,erase=False,zero_product=False):
    ww=worlds(panel)
    factual=[candidate_inputs(w,erase=erase) for w in ww]
    a=torch.cat([z[0] for z in factual])
    e=torch.cat([z[1] for z in factual])
    model.eval()
    with torch.no_grad():
        pred=_prediction(model,a,e,zero_product=zero_product)
        if not erase and not zero_product:
            identical=r38.predict(model,"binder",a,e)
            if not torch.allclose(pred,identical,atol=1e-6,rtol=1e-6):
                raise ValueError("F8R39_ABLATION_FORWARD_NOT_EXACT")
    if pred.shape!=(128,) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R39_INVALID_NEW_SCORE_PREDICTIONS")
    logits=pred.tolist()
    chosen=[min((0,1),key=lambda act:(logits[2*i+act],act))
            for i in range(len(ww))]
    regrets=[]
    for w,act in zip(ww,chosen):
        cc=[(_generator_law(w["return_state"],a,w["public_context"],
                            w["q_evaluator_only"],panel)-1.)**2
            for a in (0,1)]
        regrets.append(max(0.,cc[act]-min(cc)))
    return {"choices":chosen,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[4*g:4*g+4])
                               for g in range(GROUPS)],
            "sign_pair_flips":sum(chosen[i]!=chosen[i+1]
                                  for i in range(0,len(chosen),2))}


def _score_original(model,panel,*,erase=False,zero_product=False):
    ww=r38.worlds(panel)
    input_pairs=[r38.candidate_inputs(w,"active_information",erase=erase)
                 for w in ww]
    a=torch.cat([r[0] for r in input_pairs])
    e=torch.cat([r[1] for r in input_pairs])
    model.eval()
    with torch.no_grad():
        pred=_prediction(model,a,e,zero_product=zero_product)
        if not erase and not zero_product:
            reference=r38.predict(model,"binder",a,e)
            if not torch.allclose(pred,reference,atol=1e-6,rtol=1e-6):
                raise ValueError("F8R39_ORIGINAL_FORWARD_DRIFT")
    if pred.shape!=(128,) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R39_INVALID_SOURCE_SCORE")
    logits=pred.tolist()
    choices=[min((0,1),key=lambda v:(logits[2*i+v],v))
             for i in range(64)]
    regrets=r38.regret_evaluator_only(ww,choices)
    return {"choices":choices,
            "action_vector_sha256":hashlib.sha256(bytes(choices)).hexdigest(),
            "mean_regret":statistics.fmean(regrets),
            "sign_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,64,2))}


def replay_original_seed(seed,fit,winners,pools,samples):
    """Exact R38 source training objective / optimizer / model, no new learning."""
    model=r38.TrainableBinding(seed)
    before=r38._hash(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    info=pools["informative"]
    start=time.perf_counter()
    for keys in samples:
        selected=[info[i] for i in keys]
        a,e,y=r38.selected_batch(fit,winners,selected)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred=r38.predict(model,"binder",a,e)
        loss=F.cross_entropy(-pred.reshape(BATCH,2),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R39_NONFINITE_REPLAY_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R39_INVALID_REPLAY_GRADIENT")
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    if r38._hash(model)==before:
        raise ValueError("F8R39_NO_TRAINED_WEIGHT_CHANGE")
    restored=r38.TrainableBinding(seed)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if r38._hash(restored)!=r38._hash(model):
        raise ValueError("F8R39_RELOAD_HASH_DRIFT")
    for panel in ORIGINAL_PANELS:
        original=_score_original(model,panel)
        clone=_score_original(restored,panel)
        if (original["choices"]!=clone["choices"] or
            original["action_vector_sha256"]!=REFERENCE_CHOICES[seed]):
            raise ValueError("F8R39_SOURCE_REPLAY_ACTION_DRIFT")
    return model,{"seed":seed,"initial_weight_hash":before,
                  "trained_weight_hash":r38._hash(model),
                  "optimizer_updates":STEPS,
                  "training_wall_seconds":time.perf_counter()-start,
                  "replayed_R38_decision_fingerprints_match":True,
                  "cold_restored_action_identity":True}


def _judge(rows,freeze):
    t=freeze["frozen_judgment"]
    original_need=[]
    transfer=[]
    byseed={}
    for rec in rows:
        seed=rec["seed"]
        old=rec["original"]
        new=rec["new_laws"]
        mechanism=all(
            old[p]["normal"]["mean_regret"]<=t["mechanism_original_dev_ood_full_regret_max_each"]
            and old[p]["normal"]["sign_pair_flips"]>=t["mechanism_original_dev_ood_full_sign_pair_flips_min_each"]
            and (old[p]["zero_product"]["mean_regret"]-
                 old[p]["normal"]["mean_regret"])>=t["mechanism_original_dev_ood_zero_product_regret_increase_min_each"]
            and (old[p]["normal"]["sign_pair_flips"]-
                 old[p]["zero_product"]["sign_pair_flips"])>=t["mechanism_original_dev_ood_zero_product_sign_flip_drop_min_each"]
            for p in ("dev","ood"))
        moved=all(
            new[p]["normal"]["mean_regret"]<=t["transfer_new_law_active_regret_max_each"]
            and new[p]["normal"]["sign_pair_flips"]>=t["transfer_new_law_sign_pair_flips_min_each"]
            and (new[p]["erased"]["mean_regret"]-
                 new[p]["normal"]["mean_regret"])>=t["transfer_new_law_fifth_erasure_regret_penalty_min_each"]
            for p in PANELS)
        byseed[seed]={"product_necessary_both_old_panels":mechanism,
                       "both_new_laws_transfer":moved}
        original_need.append(mechanism)
        transfer.append(moved)
    return {"product_channel_necessary_seed_count":sum(original_need),
            "product_channel_necessity_development_pass":
                sum(original_need)>=t["mechanism_product_necessary_seeds_min"],
            "both_new_laws_transfer_seed_count":sum(transfer),
            "two_new_synthetic_laws_development_pass":
                sum(transfer)>=t["transfer_both_new_laws_seed_success_min"],
            "seed_results":byseed,
            "independent_scientific_pass":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    winners=r36.fit_pairs(fit)
    pools=r37.selection_pools(fit,winners)
    gen=torch.Generator(device="cpu")
    gen.manual_seed(21249999)
    samples=torch.randint(0,len(pools["informative"]),(STEPS,BATCH),
                          generator=gen).tolist()
    output=[]
    for seed in SEEDS:
        model,training=replay_original_seed(seed,fit,winners,pools,samples)
        old={}
        for panel in ORIGINAL_PANELS:
            old[panel]={"normal":_score_original(model,panel),
                        "zero_product":_score_original(model,panel,zero_product=True),
                        "erased":_score_original(model,panel,erase=True)}
        novel={}
        for panel in PANELS:
            novel[panel]={"normal":_score_new(model,panel),
                          "zero_product":_score_new(model,panel,zero_product=True),
                          "erased":_score_new(model,panel,erase=True)}
        output.append({**training,"original":old,"new_laws":novel})
    return {"schema":SCHEMA,"classification":freeze["classification"],
            "freeze_blob":FREEZE_BLOB,"frozen_parent":freeze["parent"]["head"],
            "original_R38_training_replays":len(SEEDS),
            "replayed_original_optimizer_updates":len(SEEDS)*STEPS,
            "new_law_training_updates":0,
            "evaluated_new_law_worlds_per_panel":4*GROUPS,
            "original_external_effects":0,"spent_F7_replay":False,
            "independent_scientific_pass":False,"deployment":"NONE",
            "models":output,"judgment":_judge(output,freeze)}


def retain_ci_receipt(output):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r39_binding_necessity_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R39_RECEIPT_OVERWRITE_DENIED")
    path.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R39_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
