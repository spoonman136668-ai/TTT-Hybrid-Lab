"""F8-R24: source-neutral *online* B learning followed by cold A revisit.

The original R22 A weights are actually trained on factual FIT examples.
Subsequent B-only updates use new factual B action effects under an unfamiliar
two-factor law, with independent B/C signs and a mid-episode reversal.
No hidden signs, original external source, unchosen effects or test A labels
enter model training. A-return is evaluated before its first new outcome.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments import p85mr_f8r23_novel_causal as r23
from experiments import p85mr_f8r22_objective_matched as r22

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/"experiments/p85mr_f8r24_online_interference_freeze.json"
FREEZE_BLOB = "fd51e60133969db2d778cd1db5272ae4e1cb483b"
SCHEMA = "ttt-hybrid.f8r24-online-B-interference-factual.v1"
ARMS = ("native_focus20","gru_focus20","ssm_focus20")
SEEDS = (985201,985203,985207)
SPLITS = {"B_train":(19910000,16),"B_holdout":(19920000,8),
          "A_score":(19930000,12)}
CONDITIONS = ("no_update","plain_B_update","anchored_B_update")
B_STEPS = 16
PRETRAIN_UPDATES = 384
ADAPT_UPDATES = 24
BATCH = 8
ADAPT_LR = 0.0005
ANCHOR_LAMBDA = 500.
BOOT = 10000
BOOT_SEED = 19939999
GOALS = (-.2,.2)


def locked():
    raw=FREEZE.read_bytes()
    digest=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if digest!=FREEZE_BLOB:
        raise ValueError("F8R24_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    w,m,s,a=f["world"],f["models"],f["score"],f["authority"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS"
        or f["classification"]!="SOURCE_NEUTRAL_ONLINE_GRADIENT_ADAPTATION_DEVELOPMENT_ONLY"
        or f["parent"]["sha"]!="98930824927845efd6164559926f1dba2f8eba10"
        or f["parent"]["r23_freeze_blob"]!="b47b3a143f123b2337f97fdb79ed5dfb66a2f836"
        or f["parent"]["r23_source_blob"]!="5ca960ea653aa2200e5e1ec76f04844c3ac77b86"
        or f["parent"]["r22_source_blob"]!="57e5977555f3c0afee7a987a4aee80be994cc082"
        or w["B_selected_action"]!=
           "behavior(s) XOR int((seed+5*t)%3==0); no hidden b/c used by behavior policy"
        or w["A_score_episodes_before_first_return"]!=96
        or w["A_score_observed_factual_effects"]!=1632
        or w["B_steps"]!=B_STEPS
        or {k:(v["seed_base"],v["n"]) for k,v in w["groups"].items()}!=SPLITS
        or m["arms"]!=list(ARMS)
        or m["seeds"]!=list(SEEDS)
        or m["adaptation_conditions"]!=list(CONDITIONS)
        or m["pretraining_updates"]!=3456
        or m["fitted_models"]!=9
        or m["adaptation_steps"]!=ADAPT_UPDATES
        or m["adaptation_batch_size"]!=BATCH
        or m["adaptation_lr"]!=ADAPT_LR
        or m["anchor_loss_lambda"]!=ANCHOR_LAMBDA
        or m["actual_B_update_steps"]!=432
        or s["paired_bootstrap_groups"]!=12
        or s["bootstrap_draws"]!=BOOT
        or s["bootstrap_seed"]!=BOOT_SEED
        or s["scientific_pass"] is not False
        or any(v is not False for v in a.values())):
        raise ValueError("F8R24_PROTOCOL_OR_AUTHORITY_DRIFT")
    r23.locked()
    return f


def b_law(s:float,action:int,b:int,c:int)->float:
    if (not math.isfinite(s) or type(action) is not int or action not in (0,1)
        or type(b) is not int or b not in (-1,1)
        or type(c) is not int or c not in (-1,1)):
        raise ValueError("F8R24_INVALID_TWO_FACTOR_LAW")
    return .23*math.tanh(1.9*s)+.18*b*(2*action-1)+.09*b*c*action


def episode(split:str,group:int,b:int,c:int,a:int|None=None)->dict:
    if (split not in SPLITS or type(group) is not int
        or not 0<=group<SPLITS[split][1]
        or type(b) is not int or b not in (-1,1)
        or type(c) is not int or c not in (-1,1)
        or (split=="A_score" and (type(a) is not int or a not in (-1,1)))
        or (split!="A_score" and a is not None)):
        raise ValueError("F8R24_INVALID_WORLD_OR_FACTUAL_SPLIT")
    seed=SPLITS[split][0]+group
    actions,effects=[],[]
    def append(before,chosen,after):
        actions.append(r22.r18.r5._action_key((before,0.,0.,0.),chosen))
        effects.append(r22.r18._observed(before,after))
    if split=="A_score":
        append(.2,0,r22.r20.r3.synthetic_effect(.2,0,a))
    state=0.
    for t in range(B_STEPS):
        if t==8:
            state=0.
        sign=b if t<8 else -b
        chosen=r22.r20.r3.behavior(state)^int((seed+5*t)%3==0)
        following=b_law(state,chosen,sign,c)
        append(state,chosen,following)
        state=following
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32),
            "group":group,"split":split,"z_a":a,"z_b":b,"z_c":c,
            "first_return_state":0.}


def cohort(split:str)->list[dict]:
    locked()
    if split=="A_score":
        return [episode(split,g,b,c,a) for g in range(SPLITS[split][1])
                for b in (-1,1) for c in (-1,1) for a in (-1,1)]
    return [episode(split,g,b,c) for g in range(SPLITS[split][1])
            for b in (-1,1) for c in (-1,1)]


def factual(x:dict)->dict:
    return {"action":x["action"],"effect":x["effect"]}


def audit(train,holdout,score):
    if (len(train),len(holdout),len(score))!=(64,32,96):
        raise ValueError("F8R24_WRONG_FACTUAL_PANEL_COUNT")
    for split,eps in (("B_train",train),("B_holdout",holdout),
                      ("A_score",score)):
        for e in eps:
            n=17 if split=="A_score" else 16
            if (e["action"].shape!=(n,16) or e["effect"].shape!=(n,16)
                or not bool(torch.isfinite(e["action"]).all())
                or not bool(torch.isfinite(e["effect"]).all())):
                raise ValueError("F8R24_INVALID_FACTUAL_EPISODE")
    if ({(e["z_b"],e["z_c"]) for e in train} !=
        {(b,c) for b in (-1,1) for c in (-1,1)}):
        raise ValueError("F8R24_MISSING_BC_FACTORIAL_SUPPORT")
    for i in range(0,len(score),2):
        n,p=score[i:i+2]
        if (n["z_a"]!=-1 or p["z_a"]!=1
            or n["group"]!=p["group"]
            or n["z_b"]!=p["z_b"] or n["z_c"]!=p["z_c"]
            or not torch.equal(n["action"],p["action"])
            or not torch.equal(n["effect"][1:],p["effect"][1:])
            or torch.equal(n["effect"][0],p["effect"][0])):
            raise ValueError("F8R24_UNPAIRED_EARLY_A_MEMORY")
    return {"B_train_factual_episodes":64,"B_holdout_factual_episodes":32,
            "A_score_matched_A_pairs":48,"A_score_factual_effects":1632,
            "B_observations_identical_within_A_pair":True}


def _delta_mse(arm,model,data,valid):
    pred=r22.predict(arm,model,data)
    if pred.shape!=data["effect"].shape:
        raise ValueError("F8R24_INVALID_PREQUENTIAL_OUTPUT")
    se=(pred[:,:,:4]-data["effect"][:,:,:4]).square().mean(dim=-1)
    mask=valid.to(dtype=se.dtype)
    if mask.shape!=se.shape or not bool(mask.sum()>0):
        raise ValueError("F8R24_INVALID_CAUSAL_TRAIN_MASK")
    return (se*mask).sum()/mask.sum()


def b_mse(arm,model,episodes):
    model.eval()
    all_err=[]
    with torch.no_grad():
        for start in range(0,len(episodes),BATCH):
            data,valid=r22.r18.r5.f4._padded(
                [factual(e) for e in episodes[start:start+BATCH]])
            all_err.append(float(_delta_mse(arm,model,data,valid)))
    score=float(np.mean(all_err))
    if not math.isfinite(score):
        raise ValueError("F8R24_B_HELDOUT_NONFINITE")
    return score


def _predicted_return(arm,model,episodes,*,erase_early=False):
    a,e=[],[]
    for ep in episodes:
        for action in (0,1):
            key=r22.r18.r5._action_key((0.,0.,0.,0.),action)
            action_hist=torch.cat((
                ep["action"],torch.tensor([key],dtype=torch.float32)),dim=0)
            effect_hist=torch.cat((
                ep["effect"],torch.zeros((1,16),dtype=torch.float32)),dim=0)
            if erase_early:
                effect_hist[0]=0.
            a.append(action_hist);e.append(effect_hist)
    model.eval()
    with torch.no_grad():
        outputs=r22.predict(arm,model,{
            "action":torch.stack(a),"effect":torch.stack(e)})
    if outputs.shape!=(len(episodes)*2,18,16) or not bool(torch.isfinite(outputs).all()):
        raise ValueError("F8R24_INVALID_COLD_RETURN_MODEL_QUERY")
    return np.asarray([2.4*math.atanh(max(-.98,min(.98,float(z))))
                       for z in outputs[:,-1,0].cpu().numpy()],
                       dtype=np.float64).reshape(len(episodes),2)


def score_A(arm,model,episodes):
    preds=_predicted_return(arm,model,erase_early=False,episodes=episodes)
    erased=_predicted_return(arm,model,erase_early=True,episodes=episodes)
    scores,negative,actions=[],[],[]
    for i,ep in enumerate(episodes):
        for goal in GOALS:
            selected=min((0,1),key=lambda q:((preds[i,q]-goal)**2,q))
            erased_selected=min((0,1),key=lambda q:((erased[i,q]-goal)**2,q))
            # The return-A oracle is evaluator-only after choice.
            scores.append(r22.r20.regret(ep["z_a"],goal,selected))
            negative.append(r22.r20.regret(ep["z_a"],goal,erased_selected))
            actions.append(selected)
    bygroup=[float(np.mean(scores[g*16:g*16+16])) for g in range(12)]
    contrast=sum(actions[2*i+j]!=actions[2*(i+1)+j]
                 for i in range(0,len(episodes),2) for j in (0,1))
    return {"mean_regret":float(np.mean(scores)),
            "erased_early_A_mean_regret":float(np.mean(negative)),
            "pergroup_regret":bygroup,
            "selected_actions":actions,
            "A_sign_action_contrasts":contrast,
            "A_sign_contrast_opportunities":96,
            "synthetic_first_return_actions":len(actions)}


def _clone_pretrained(arm,seed,model):
    cloned=r22.model_for(arm,seed)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    return cloned.cpu()


def adapt(arm,model,train,indices,condition,original):
    if condition not in CONDITIONS:
        raise ValueError("F8R24_INVALID_B_ADAPTATION_POLICY")
    before=r22.r18.r5.f5.hash_weights(model)
    if condition=="no_update":
        return {"updates":0,"pretrain_sha":before,"adapted_sha":before,
                "train_first4_loss":None,"train_last4_loss":None}
    if len(indices)!=ADAPT_UPDATES:
        raise ValueError("F8R24_ADAPTATION_BATCH_BUDGET_DRIFT")
    anchor={name:v.detach().clone() for name,v in original.named_parameters()}
    opt=torch.optim.AdamW(model.parameters(),lr=ADAPT_LR,weight_decay=.01)
    losses=[]
    for choices in indices:
        data,valid=r22.r18.r5.f4._padded([factual(train[i]) for i in choices])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=_delta_mse(arm,model,data,valid)
        if condition=="anchored_B_update":
            drift=sum((p-anchor[name]).square().sum()
                       for name,p in model.named_parameters())
            total=sum(p.numel() for p in model.parameters())
            loss=loss+ANCHOR_LAMBDA*drift/total
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R24_NONFINITE_B_ADAPTATION_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads)):
            raise ValueError("F8R24_B_ADAPTATION_MISSING_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        losses.append(float(loss.detach()))
    after=r22.r18.r5.f5.hash_weights(model)
    if after==before:
        raise ValueError("F8R24_ONLINE_B_WEIGHTS_DID_NOT_CHANGE")
    return {"updates":ADAPT_UPDATES,"pretrain_sha":before,
            "adapted_sha":after,
            "train_first4_loss":float(np.mean(losses[:4])),
            "train_last4_loss":float(np.mean(losses[-4:]))}


def fit_one(arm,seed,fit,dev,B_train,B_holdout,A_score,pre_batches,B_batches):
    model=r22.model_for(arm,seed).cpu()
    initial=r22.r18.r5.f5.hash_weights(model)
    optimizer=torch.optim.AdamW(model.parameters(),
               lr=.002 if arm.startswith("native") else .001,weight_decay=.01)
    t0=time.perf_counter()
    for choices in pre_batches:
        data,valid=r22.r18.r5.f4._padded([r22.factual(fit[i]) for i in choices])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=r22.objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R24_BAD_ORIGINAL_A_LEARNING")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    pretrained_sha=r22.r18.r5.f5.hash_weights(model)
    if initial==pretrained_sha:
        raise ValueError("F8R24_A_PRETRAIN_NOT_EXECUTED")
    dev_score=r22.score(arm,model,dev,"dev")["mean_regret"]
    fixed_A=score_A(arm,model,A_score)
    original_B_mse=b_mse(arm,model,B_holdout)
    conditions={}
    for condition in CONDITIONS:
        clone=_clone_pretrained(arm,seed,model)
        if r22.r18.r5.f5.hash_weights(clone)!=pretrained_sha:
            raise ValueError("F8R24_NOT_SAME_INITIAL_CHECKPOINT")
        started=time.perf_counter()
        receipt=adapt(arm,clone,B_train,B_batches,condition,model)
        retained=score_A(arm,clone,A_score)
        heldout_B=b_mse(arm,clone,B_holdout)
        train_B=b_mse(arm,clone,B_train)
        copied=_clone_pretrained(arm,seed,clone)
        replay=score_A(arm,copied,A_score)
        if (r22.r18.r5.f5.hash_weights(copied)!=receipt["adapted_sha"]
            or replay["selected_actions"]!=retained["selected_actions"]
            or replay["pergroup_regret"]!=retained["pergroup_regret"]):
            raise ValueError("F8R24_COLD_RESTORE_ACTION_REPLAY_FAILED")
        conditions[condition]={**receipt,"A_score":retained,
            "B_heldout_mse":heldout_B,"B_train_mse":train_B,
            "adaptation_and_score_seconds":time.perf_counter()-started,
            "cold_restore_exact_A_action_parity":True}
    if conditions["no_update"]["A_score"]["selected_actions"]!=fixed_A["selected_actions"]:
        raise ValueError("F8R24_CONTROL_MODEL_REINITIALIZATION_DRIFT")
    return {"arm":arm,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "initial_sha":initial,"A_pretrained_sha":pretrained_sha,
            "original_DEV_A_mean_regret":dev_score,
            "original_A_no_B_update_regret":fixed_A["mean_regret"],
            "original_B_heldout_mse":original_B_mse,
            "conditions":conditions,"elapsed_seconds":time.perf_counter()-t0}


def _fixed_A_controls(A_score,fit):
    fixed=[[],[]]
    for ep in A_score:
        for goal in GOALS:
            for act in (0,1):
                fixed[act].append(r22.r20.regret(ep["z_a"],goal,act))
    means={f"fixed{act}":float(np.mean(fixed[act])) for act in (0,1)}
    if not math.isclose(min(means.values()),.015,abs_tol=1e-12):
        raise ValueError("F8R24_BAD_FIXED_A_REGRET_REFERENCE")
    ridge=r22.fit_ridge(fit,use_early=True)
    weights=np.asarray(ridge["coef"],dtype=np.float64)
    ridge_losses=[]
    for ep in A_score:
        fact=r22.r19._physical_delta(ep["effect"][0])
        estimate=(float(weights@np.asarray([1.,fact])),ridge["action1_mean"])
        for goal in GOALS:
            selected=min((0,1),key=lambda x:((estimate[x]-goal)**2,x))
            ridge_losses.append(r22.r20.regret(ep["z_a"],goal,selected))
    result=float(np.mean(ridge_losses))
    if result>=.001:
        raise ValueError("F8R24_EARLY_A_FIT_ONLY_MEMORY_NOT_LEARNABLE")
    return {"fixed":means,"FIT_only_early_A_ridge_mean_regret":result}


def decision(rows):
    rng=np.random.default_rng(BOOT_SEED)
    draw=rng.integers(0,12,size=(BOOT,12))
    candidates=[]
    for row in rows:
        if row["arm"]!="native_focus20":
            continue
        conditions=row["conditions"]
        base=conditions["no_update"];plain=conditions["plain_B_update"]
        anchored=conditions["anchored_B_update"]
        before=base["A_score"]["mean_regret"]
        after=anchored["A_score"]["mean_regret"]
        plain_mean=plain["A_score"]["mean_regret"]
        bbase=base["B_heldout_mse"];ba=anchored["B_heldout_mse"]
        gain=np.asarray(plain["A_score"]["pergroup_regret"])-np.asarray(
            anchored["A_score"]["pergroup_regret"])
        lower=float(np.percentile(gain[draw].mean(axis=1),2.5))
        passed=bool(ba<=.95*bbase and after<=.0135
              and after<=before+.0025 and after<=plain_mean-.005 and lower>0)
        candidates.append({"seed":row["seed"],
            "A_before_B_regret":before,
            "A_after_plain_B_regret":plain_mean,
            "A_after_anchored_B_regret":after,
            "heldout_B_mse_before":bbase,
            "heldout_B_mse_after_anchor":ba,
            "bootstrap_2p5_plain_minus_anchor_gain":lower,
            "native_anchor_DEV_win":passed})
    return {"native_anchor_seed_gates":candidates,
            "frozen_DEV_gate_pass":sum(int(x["native_anchor_DEV_win"])
                for x in candidates)>=2,"scientific_pass":False}


def run():
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    fit=r22.cohort("fit")
    dev=r22.cohort("dev")
    bt=cohort("B_train");bh=cohort("B_holdout");asc=cohort("A_score")
    receipt=audit(bt,bh,asc)
    controls=_fixed_A_controls(asc,fit)
    g=torch.Generator(device="cpu");g.manual_seed(985299)
    batches=torch.randint(0,len(fit),(PRETRAIN_UPDATES,BATCH),generator=g).tolist()
    bgen=torch.Generator(device="cpu");bgen.manual_seed(19919999)
    bchoices=torch.randint(0,len(bt),(ADAPT_UPDATES,BATCH),generator=bgen).tolist()
    rows=[fit_one(arm,seed,fit,dev,bt,bh,asc,batches,bchoices)
          for arm in ARMS for seed in SEEDS]
    return {"schema":SCHEMA,
        "classification":"SOURCE_NEUTRAL_ONLINE_GRADIENT_ADAPTATION_DEVELOPMENT_ONLY",
        "frozen_blob":FREEZE_BLOB,
        "identity_receipt":receipt,
        "original_R22_A_pretraining_factual_effects":len(fit)*10,
        "new_B_train_factual_selected_effects":len(bt)*B_STEPS,
        "new_B_holdout_factual_selected_effects":len(bh)*B_STEPS,
        "new_score_factual_A_and_B_effects":len(asc)*(B_STEPS+1),
        "actual_model_count":len(rows),
        "original_pretrain_update_count":len(rows)*PRETRAIN_UPDATES,
        "actual_online_B_gradient_update_count":len(rows)*2*ADAPT_UPDATES,
        "fixed_controls":controls,
        "models":rows,"decision":decision(rows),
        "independent_conventional_convergence_qualified":False,
        "original_external_effects":0,"F7_replayed":False,
        "independent_scientific_pass":False,"deployment":"NONE",
        "process_elapsed_seconds":time.perf_counter()-t0}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r24_online_B_interference_dev.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R24_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_scientific_firstuse(*_a,**_k):
    raise PermissionError("F8R24_EXTERNAL_SCIENTIFIC_FIRSTUSE_NOT_AUTHORIZED")
