"""G7-S23: Hysteretic persistent-change detection for unlabeled numerical memory banks.

Frozen experiment. All routing/novelty decisions use ONLY the observed
history (sensor, actuator, numerical dt, measured sensor_next). The true
family name is only used by synthetic episode generation and by separate
read-only scoring of the heldout route counts; never sent into the router.

Specialized weights are stored in up to three distinct 2→32→12 feature
banks, each with a per-world continuous ridge reader. A bank is allocated
when cross-validated context prediction surprise exceeds the source-only
training-range frozen novelty threshold. At test time a bank is selected
by the lowest per-world prediction error on last-eight seen transitions,
after fitting its ridge coefficients on first 24 transitions.

Four-phase A(source)→B(sine)→C(rational)→D(source). Baselines: matched
source-initialized shared finetune, randomly routed three-bank capacity
control, 25%-old training-world rehearsal (extra information), privileged
label-routed three-bank upper bound, trained GRU, CAL-only RBF-KRR,
untrained feature map, and exact synthetic physical oracle.

No tokenizer, vocabulary, symbol IDs, attention or next-token loss.
Digital numerical time sampling and Euler residual prior remain.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s21_orthogonal_memory as previous
import torch
from torch import nn
from torch.nn import functional as F

s18=previous.s18
s20=previous.s20

class Blocked(Exception):
    pass

def require(ok,label):
    if not ok:
        raise Blocked(label)

FAMILIES=("source","sine","rational")
STAGES=("A","B","C","D")
NAMED_ARMS=("self_organizing","naive_S22","shared_no_replay","random_three_bank",
            "rehearsal","oracle_label_routed","GRU","KRR","untrained","static")

def frozen(p):
    t=p["training"];cal=p["calibration"];z=p["frozen_thresholds"]
    d=p["unconventional"]["detection"];ev=p["evaluation"]
    require(
        p.get("schema")==
         "ttt-hybrid.g7-s23.hysteretic-changepoint-banks.v1"
        and p["predecessor"]["s22_science_candidate_sha"]==
         "065e8bdfd79b4624e2c3d8fed23bbba7b12453de"
        and p["predecessor"]["s22_run"]==42408137211
        and p["predecessor"]["s22_science_blob"]==
         "519a1d92832d2b356d50c41b4fe6584c4b995ec7"
        and p["predecessor"]["s20_world_blob"]==
         "ac3f9ddd5b879e53e62030707fb6419adaf732ca"
        and p["predecessor"]["s18_model_blob"]==
         "cfc1959b8812e9abad581018462302bc89a31d59"
        and t["fit_seeds"]==[1901,2011,2131]
        and t["batch_size"]==48
        and [x["steps"] for x in t["phases"]]==[240,96,96,96]
        and [x["family"] for x in t["phases"]]==[
            "source","sine","rational","source"]
        and [x["seed_offset"] for x in t["phases"]]==[
            2401000,2501000,2601000,2701000]
        and t["total_updates_per_trainable_arm"]==528
        and t["worlds_per_trainable_arm"]==25344
        and p["unconventional"]["context_fit_first_n"]==24
        and p["unconventional"]["context_score_last_n"]==8
        and p["unconventional"]["max_banks"]==3
        and d["window_length"]==6
        and d["required_high"]==5
        and d["relative_change_multiplier"]==1.35
        and d["ema_alpha"]==.10
        and d["post_allocation_cooldown_updates"]==36
        and d["source_only_negative_duration_updates"]==48
        and d["negative_train_episode_seed_formula"]==
             "fit_seed*100000+3101000+step*37"
        and d["no_phase_boundary_reset"] is True
        and p["unconventional"]["training_router_task_label_used"] is False
        and p["unconventional"]["evaluation_router_task_label_used"] is False
        and cal["router_source_seeds"]==[
            4103,4111,4127,4133,4153,4163,4177,4187,
            4199,4231,4241,4253,4261,4271,4283,4297]
        and cal["kernel_source_seeds"]==[4201,4211,4229]
        and ev["heldout_ood_seeds"]=={
            "source":[4303,4319,4337],
            "sine":[4409,4421,4441],
            "rational":[4507,4519,4523]}
        and z["bank_source_C_relative_gain_vs_naive_S22_min"]==.10
        and z["required_banks_by_phase_B"]==2
        and z["required_banks_by_phase_C"]==3
        and z["maximum_banks_by_phase_B"]==2
        and z["max_stationary_false_allocations"]==0
        and z["minimum_route_accuracy_per_seed_after_C"]==.60
        and z["all_three_seeds_must_pass"] is True
        and z["wall_seconds_max"]==240
        and p["environment"]["no_tokenizer"] is True
        and p["environment"]["no_attention"] is True
        and p["authority"]["cpu_only"] is True,
        "FROZEN_S23_PROTOCOL_OR_AUTHORITY_DRIFT")

def episode(seed,b,regime):
    return s20.partial_world(seed,b,regime)

def context_scores(models,history):
    """No query target or hidden task IDs; prefix-24 fit, suffix-8 score."""
    prefix=history[:,:24,:]
    tail=history[:,24:,:]
    q=torch.stack((tail[:,:,0],tail[:,:,1],tail[:,:,2]),dim=-1)
    with torch.no_grad():
        scores=[]
        for model in models:
            model.eval()
            predicted=s18.learned_ridge(model,prefix,q)
            error=(predicted-tail[:,:,3]).square().mean(dim=1)
            require(bool(torch.isfinite(error).all()),
                    "NONFINITE_CONTEXT_SCORE")
            scores.append(error)
    return torch.stack(scores,dim=-1)

def contextual_predict(models,history,query):
    """Bank choice from context only. Hidden family label not an argument."""
    score=context_scores(models,history)
    chosen=score.argmin(dim=-1)
    with torch.no_grad():
        predictions=torch.stack([
            s18.learned_ridge(model,history,query) for model in models],
            dim=1)
        gathered=predictions[torch.arange(history.shape[0]),chosen]
    return gathered,chosen,score

def bank_train(model,optimizer,history,query,target):
    model.train()
    return previous.train_step(model,optimizer,history,query,target)

def GRU_train(model,optimizer,history,query,target):
    model.train()
    return previous.train_gru(model,optimizer,history,query,target)

def random_route(history,number_banks):
    """Only numerical context is hashed. No family/phase/target or seed."""
    with torch.no_grad():
        a=torch.round(
            (history[:,12:,0].mean(dim=1)+2.5)*10007).to(torch.int64)
        b=torch.round(
            (history[:,5:,1].mean(dim=1)+2.0)*65537).to(torch.int64)
        return torch.remainder(a*1664525+b*1013904223,number_banks)

def random_banked_predict(models,history,query):
    choice=random_route(history,len(models))
    with torch.no_grad():
        predictions=torch.stack([
            s18.learned_ridge(model,history,query)
            for model in models],dim=1)
        selected=predictions[torch.arange(history.shape[0]),choice]
    return selected,choice

def train_source(model,gru,fopt,gopt,p,seed,start):
    phase=p["training"]["phases"][0]
    seconds={"source_bank":0.,"GRU":0.}
    for step in range(phase["steps"]):
        h,q,y=episode(seed*100000+phase["seed_offset"]+step*37,
                      p["training"]["batch_size"],"train_switch")
        _,_,_,duration=bank_train(model,fopt,h,q,y)
        _,gru_seconds=GRU_train(gru,gopt,h,q,y)
        seconds["source_bank"]+=duration
        seconds["GRU"]+=gru_seconds
        require(time.monotonic()-start<p["frozen_thresholds"]["wall_seconds_max"],
                "SOURCE_TRAIN_TIME_EXCEEDED")
    return seconds

def kernel_calibration(p,start):
    cfg=p["calibration"]
    losses={}
    candidates=list(itertools.product(
        cfg["gamma"],cfg["decay"],cfg["ridge"]))
    data=[episode(seed,cfg["worlds_per_seed"],"train_switch")
          for seed in cfg["kernel_source_seeds"]]
    with torch.no_grad():
        for candidate in candidates:
            losses[candidate]=mean([
                float(F.mse_loss(
                    s18.kernel_ridge(h,q,*candidate),y))
                for h,q,y in data])
            require(time.monotonic()-start<
                    p["frozen_thresholds"]["wall_seconds_max"],
                    "CALIBRATION_TIME_EXCEEDED")
    best=min(candidates,key=lambda x:(
        losses[x],x[0],-x[1],-x[2]))
    return best,{str(k):v for k,v in losses.items()}

def novelty_threshold(model,p):
    cfg=p["calibration"]
    errors=[]
    for seed in cfg["router_source_seeds"]:
        h,_,_=episode(seed,cfg["router_worlds_per_calibration_batch"],
                      "train_switch")
        score=context_scores([model],h)
        errors.append(float(score[:,0].mean()))
    threshold=float(torch.quantile(
        torch.tensor(errors),.9))
    require(math.isfinite(threshold) and threshold>0,
            "INVALID_TRAIN_ONLY_NOVELTY_CALIBRATION")
    return threshold,errors

class PersistentChangeDetector:
    """Causal training-time novelty accumulator: NO task labels or phase reset."""
    def __init__(self,p,source_threshold,calibration_errors):
        self.d=p["unconventional"]["detection"]
        self.source_threshold=float(source_threshold)
        self.reference=float(mean(calibration_errors))
        require(math.isfinite(self.reference) and self.reference>0,
                "INVALID_SOURCE_ERROR_REFERENCE")
        self.high=[]
        self.cooldown=0
        self.total_above=0
        self.calls=0

    def observe(self,error):
        require(math.isfinite(error) and error>=0,
                "NONFINITE_NOVELTY_SIGNAL")
        d=self.d
        self.calls+=1
        threshold=max(self.source_threshold,
                      self.reference*d["relative_change_multiplier"])
        high=error>threshold
        if self.cooldown>0:
            # Bank stabilization: absorb its own post-creation error scale,
            # do not allocate another from the same unlearned regime.
            self.reference=(1-d["ema_alpha"])*self.reference+(
                d["ema_alpha"]*error)
            self.cooldown-=1
            self.high.clear()
            return False,threshold,high
        if high:
            self.total_above+=1
        else:
            self.reference=(1-d["ema_alpha"])*self.reference+(
                d["ema_alpha"]*error)
        self.high.append(high)
        self.high=self.high[-d["window_length"]:]
        triggered=(len(self.high)==d["window_length"]
                   and sum(self.high)>=d["required_high"])
        if triggered:
            self.cooldown=d["post_allocation_cooldown_updates"]
            self.high.clear()
            # Do not zero the long-term reference; it is adjusted
            # progressively using observed scores during refractory time.
        return triggered,threshold,high

def stationary_false_alarm_check(model,seed,p,threshold,calibration_errors):
    """Separate, never trained-on source-only control streams, dry-run only."""
    detector=PersistentChangeDetector(p,threshold,calibration_errors)
    scores=[]
    triggers=[]
    for step in range(p["unconventional"]["detection"][
            "source_only_negative_duration_updates"]):
        h,_,_=episode(seed*100000+3101000+step*37,
                      p["training"]["batch_size"],"train_switch")
        value=float(context_scores([model],h)[:,0].mean())
        fired,_,_=detector.observe(value)
        scores.append(value)
        if fired:
            triggers.append(step)
    return {"source_only_batches":len(scores),
            "false_allocation_steps":triggers,
            "mean_context_mse":mean(scores),
            "source_only_calibration_frozen":True,
            "no_task_labels_or_heldout_targets_used":True}

def clone_optimizer(model,state,p):
    opt=torch.optim.AdamW(model.parameters(),
        lr=p["training"]["feature_lr"],
        weight_decay=p["training"]["weight_decay"])
    if state is not None:
        opt.load_state_dict(copy.deepcopy(state))
    return opt

def rehearse(seed,phase,step,h,q,y):
    if phase=="A":return h,q,y
    old_seed=seed*100000+2201000+step*37
    if phase in ("B","C"):
        oh,oq,oy=episode(old_seed,12,"source_train")
        return (torch.cat((h[:36],oh),dim=0),
                torch.cat((q[:36],oq),dim=0),
                torch.cat((y[:36],oy),dim=0))
    sh,sq,sy=episode(old_seed,6,"sine_train")
    rh,rq,ry=episode(seed*100000+2301000+step*37,
                     6,"rational_train")
    return (torch.cat((h[:36],sh,rh),dim=0),
            torch.cat((q[:36],sq,rq),dim=0),
            torch.cat((y[:36],sy,ry),dim=0))

def train_later(state,stage,seed,p,start,threshold):
    """Only generator and privileged ORACLE know true stage/family.
    Persistent and naïve routers receive observed contexts, never labels.
    """
    spec=next(x for x in p["training"]["phases"] if x["id"]==stage)
    timing={key:0. for key in (
        "self_organizing","naive_S22","shared_no_replay",
        "random_three_bank","rehearsal","oracle_label_routed","GRU")}
    selected=[]
    naive_selected=[]
    novelty_alloc=[]
    naive_alloc=[]
    anomaly_count=0
    for step in range(spec["steps"]):
        h,q,y=episode(seed*100000+spec["seed_offset"]+step*37,
                      p["training"]["batch_size"],
                      spec["family"]+"_train")
        # Only observed sensor/action/consequence streams enter this gate.
        all_scores=context_scores(state["banks"],h).mean(dim=0)
        best=int(all_scores.argmin())
        surprise=float(all_scores[best])
        trigger,required,high=state["detector"].observe(surprise)
        anomaly_count+=int(high)
        if (trigger and len(state["banks"])<
                p["unconventional"]["max_banks"]):
            state["banks"].append(copy.deepcopy(state["banks"][best]))
            state["bankopts"].append(
                clone_optimizer(state["banks"][-1],None,p))
            best=len(state["banks"])-1
            novelty_alloc.append({
                "global_update":state["global_update"],
                "training_update":step,
                "bank_id":best,
                "source_only_quantile_threshold":threshold,
                "persistent_reference_at_event":state["detector"].reference,
                "trigger_threshold":required,
                "observed_context_surprise":surprise,
                "consecutive_window_required":5,
                "phase_for_posthoc_receipt_only":stage})
        selected.append(best)
        _,_,_,secs=bank_train(
            state["banks"][best],state["bankopts"][best],h,q,y)
        timing["self_organizing"]+=secs
        # Exact S22 single-batch novelty method on same NEW worlds.
        naive_scores=context_scores(state["naive_banks"],h).mean(dim=0)
        naive_best=int(naive_scores.argmin())
        naive_surprise=float(naive_scores[naive_best])
        if (naive_surprise>threshold and len(state["naive_banks"])<
                p["unconventional"]["max_banks"]):
            state["naive_banks"].append(
                copy.deepcopy(state["naive_banks"][naive_best]))
            state["naive_opts"].append(
                clone_optimizer(state["naive_banks"][-1],None,p))
            naive_best=len(state["naive_banks"])-1
            naive_alloc.append({
                "global_update":state["global_update"],
                "training_update":step,"bank_id":naive_best,
                "observed_context_surprise":naive_surprise,
                "phase_for_posthoc_receipt_only":stage})
        naive_selected.append(naive_best)
        _,_,_,secs=bank_train(
            state["naive_banks"][naive_best],
            state["naive_opts"][naive_best],h,q,y)
        timing["naive_S22"]+=secs
        _,_,_,secs=bank_train(
            state["shared"],state["shared_opt"],h,q,y)
        timing["shared_no_replay"]+=secs
        # Equal bank capacity, randomized TRAINING bank updates, deterministic
        # history-hash at evaluation; family label never enters this control.
        gen=torch.Generator(device="cpu").manual_seed(
            seed*10000+step*41+17)
        random_id=int(torch.randint(0,3,(1,),generator=gen))
        _,_,_,secs=bank_train(
            state["random_banks"][random_id],
            state["random_opts"][random_id],h,q,y)
        timing["random_three_bank"]+=secs
        rh,rq,ry=rehearse(seed,stage,step,h,q,y)
        _,_,_,secs=bank_train(
            state["rehearsal"],state["rehearsal_opt"],rh,rq,ry)
        timing["rehearsal"]+=secs
        # PRIVILEGED oracle reference only, never consumed by either router.
        oracle_id={"B":1,"C":2,"D":0}[stage]
        _,_,_,secs=bank_train(
            state["oracle_banks"][oracle_id],
            state["oracle_opts"][oracle_id],h,q,y)
        timing["oracle_label_routed"]+=secs
        _,secs=GRU_train(state["gru"],state["gru_opt"],h,q,y)
        timing["GRU"]+=secs
        state["global_update"]+=1
        require(time.monotonic()-start<
                p["frozen_thresholds"]["wall_seconds_max"],
                "MULTIBANK_TRAINING_BUDGET_EXCEEDED")
    return {
        "phase":stage,"updates":spec["steps"],
        "active_banks":len(state["banks"]),
        "naive_active_banks":len(state["naive_banks"]),
        "allocation_events":novelty_alloc,
        "naive_allocation_events":naive_alloc,
        "anomalies_over_threshold":anomaly_count,
        "persistent_detector_cooldown_remaining":
            state["detector"].cooldown,
        "persistent_detector_ewma_reference":
            state["detector"].reference,
        "blind_training_route_counts":[selected.count(i)
                                        for i in range(3)],
        "naive_training_route_counts":[naive_selected.count(i)
                                       for i in range(3)],
        "training_seconds_by_arm":timing,
        "source_task_label_passed_to_router":False,
        "old_training_episodes_used_by_self_organizing":False}

def evaluate(state,index,p,kcfg):
    results={}
    route_counts={}
    for family in FAMILIES:
        h,q,y=episode(
            p["evaluation"]["heldout_ood_seeds"][family][index],
            p["evaluation"]["worlds_per_seed_per_family"],
            family+"_ood")
        values={}
        with torch.no_grad():
            tick=time.monotonic()
            chosen,route,score=contextual_predict(
                state["banks"],h,q)
            values["self_organizing_mse"]=float(F.mse_loss(chosen,y))
            values["self_organizing_infer_seconds"]=time.monotonic()-tick
            values["blind_context_min_error_mean"]=float(
                score.min(dim=-1).values.mean())
            route_counts[family]=[
                int((route==n).sum())
                for n in range(3)]
            naive_t=time.monotonic()
            naive_pred,naive_route,_=contextual_predict(
                state["naive_banks"],h,q)
            values["naive_S22_mse"]=float(F.mse_loss(naive_pred,y))
            values["naive_S22_infer_seconds"]=time.monotonic()-naive_t
            start=time.monotonic()
            random_pred,random_route_ids=random_banked_predict(
                state["random_banks"],h,q)
            values["random_three_bank_mse"]=float(
                F.mse_loss(random_pred,y))
            values["random_three_bank_infer_seconds"]=time.monotonic()-start
            oracle_id={"source":0,"sine":1,"rational":2}[family]
            candidates={
                "shared_no_replay":lambda:s18.learned_ridge(
                    state["shared"],h,q),
                "rehearsal":lambda:s18.learned_ridge(
                    state["rehearsal"],h,q),
                "oracle_label_routed":lambda:s18.learned_ridge(
                    state["oracle_banks"][oracle_id],h,q),
                "GRU":lambda:state["gru"](h,q),
                "KRR":lambda:s18.kernel_ridge(h,q,*kcfg),
                "untrained":lambda:s18.learned_ridge(
                    state["untrained"],h,q),
                "static":lambda:q[:,:,0],
                "oracle":lambda:y
            }
            for name,fn in candidates.items():
                time0=time.monotonic()
                pred=fn()
                values[name+"_mse"]=float(F.mse_loss(pred,y))
                values[name+"_infer_seconds"]=time.monotonic()-time0
            if family=="source":
                censored=torch.cat((h[:,:20,:],h[:,:12,:]),dim=1)
                c_pred,c_route,_=contextual_predict(
                    state["banks"],censored,q)
                values["blind_bank_censored_history_mse"]=float(
                    F.mse_loss(c_pred,y))
                values["KRR_censored_history_mse"]=float(
                    F.mse_loss(s18.kernel_ridge(censored,q,*kcfg),y))
                values["censored_route_different_fraction"]=float(
                    (c_route!=route).float().mean())
            require(all(math.isfinite(float(x)) for x in values.values()),
                    "NONFINITE_HELDOUT_OR_ROUTING_METRIC")
        results[family]=values
    return results,route_counts

def checkpoint_all(state,seed):
    h,q,_=episode(seed*100000+844999,4,"source_ood")
    measured={}
    groups={
        "blind_banks":state["banks"],
        "naive_banks":state["naive_banks"],
        "random_banks":state["random_banks"],
        "oracle_banks":state["oracle_banks"],
        "singletons":[state["shared"],state["rehearsal"],
                      state["gru"]]}
    for group,models in groups.items():
        measured[group]=[]
        for model in models:
            sha,delta=s18.checkpoint_cold(model,h,q)
            require(delta==0.,"COLD_RESTORE_MEASUREMENT_DRIFT")
            measured[group].append({"sha256":sha,"max_abs_delta":delta})
    return measured

def routing_accuracy(counts,bank_count):
    """Labels used ONLY here, after heldout predictions for scoring."""
    permutations=itertools.permutations(range(3))
    total=sum(sum(c) for c in counts.values())
    best=max(sum(counts[family][assignment[index]]
                 for index,family in enumerate(FAMILIES))
             for assignment in permutations)
    return best/max(total,1)

def repeat_controls_not_changed(stage_rows):
    for family in FAMILIES:
        base=stage_rows["A"]["heldout"][family]
        for stage in ("B","C","D"):
            other=stage_rows[stage]["heldout"][family]
            for arm in ("KRR","untrained","static","oracle"):
                require(base[arm+"_mse"]==other[arm+"_mse"],
                        "UNCHANGED_CONTROL_PREDICTIONS_DRIFT_"+arm)

def run(a):
    raw=a.protocol.read_bytes()
    p=json.loads(raw)
    frozen(p)
    require(custody.git(ROOT,"hash-object",
                        "scripts/g7_s22_surprise_banks.py")==
            p["predecessor"]["s22_science_blob"] and
            custody.git(ROOT,"hash-object",
                        "scripts/g7_s20_hidden_multicycle.py")==
            p["predecessor"]["s20_world_blob"] and
            custody.git(ROOT,"hash-object",
                        "scripts/g7_s18_causal_feature_replication.py")==
            p["predecessor"]["s18_model_blob"],
            "PINNED_OLD_SCIENCE_IMPLEMENTATION_DRIFT")
    manifest,packages=custody.manifest()
    require(manifest==custody.R3_MANIFEST_SHA256
            and packages.get("torch")==custody.TORCH
            and packages.get("triton")==custody.TRITON,
            "ORIGINAL_PINNED_R3_ENVIRONMENT_CHANGED")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(),
            "CPU_ONLY_RESEARCH_CUDA_USED")
    start=time.monotonic()
    kcfg,calibration=kernel_calibration(p,start)
    results=[]
    for seed_index,seed in enumerate(p["training"]["fit_seeds"]):
        torch.manual_seed(seed)
        source=s18.RawFeatureBank()
        untrained=copy.deepcopy(source)
        GRU=s18.RawGRU()
        sourceopt=clone_optimizer(source,None,p)
        gruopt=torch.optim.AdamW(GRU.parameters(),
            lr=p["training"]["GRU_lr"],
            weight_decay=p["training"]["weight_decay"])
        source_times=train_source(
            source,GRU,sourceopt,gruopt,p,seed,start)
        threshold,calibration_errors=novelty_threshold(source,p)
        false_alarm_control=stationary_false_alarm_check(
            source,seed,p,threshold,calibration_errors)
        source_opt_state=copy.deepcopy(sourceopt.state_dict())
        state={
            "banks":[copy.deepcopy(source)],
            "bankopts":[],
            "naive_banks":[copy.deepcopy(source)],
            "naive_opts":[],
            "detector":PersistentChangeDetector(
                p,threshold,calibration_errors),
            "global_update":240,
            "shared":copy.deepcopy(source),
            "shared_opt":None,
            "random_banks":[copy.deepcopy(source) for _ in range(3)],
            "random_opts":[],
            "oracle_banks":[copy.deepcopy(source) for _ in range(3)],
            "oracle_opts":[],
            "rehearsal":copy.deepcopy(source),
            "rehearsal_opt":None,
            "gru":GRU,"gru_opt":gruopt,
            "untrained":untrained}
        state["bankopts"]=[
            clone_optimizer(state["banks"][0],source_opt_state,p)]
        state["naive_opts"]=[
            clone_optimizer(state["naive_banks"][0],source_opt_state,p)]
        state["shared_opt"]=clone_optimizer(
            state["shared"],source_opt_state,p)
        state["random_opts"]=[
            clone_optimizer(bank,source_opt_state,p)
            for bank in state["random_banks"]]
        state["oracle_opts"]=[
            clone_optimizer(bank,source_opt_state,p)
            for bank in state["oracle_banks"]]
        state["rehearsal_opt"]=clone_optimizer(
            state["rehearsal"],source_opt_state,p)
        require(all(torch.equal(
             previous.vector(state["banks"][0]),previous.vector(model))
             for model in (
                state["shared"],state["rehearsal"],
                state["naive_banks"][0],
                *state["random_banks"],*state["oracle_banks"])),
             "POST_SOURCE_BASELINE_WEIGHTS_NOT_IDENTICAL")
        stage_rows={}
        allocations=[]
        for stage in STAGES:
            if stage=="A":
                training={"phase":"A","updates":240,"active_banks":1,
                          "training_seconds_by_arm":source_times,
                          "allocation_events":[]}
            else:
                training=train_later(
                    state,stage,seed,p,start,threshold)
                allocations+=training["allocation_events"]
            measured,counts=evaluate(state,seed_index,p,kcfg)
            cold=checkpoint_all(state,seed)
            stage_rows[stage]={
                "training":training,
                "heldout":measured,
                "heldout_route_counts":counts,
                "permutation_audit_routing_accuracy":
                    routing_accuracy(counts,len(state["banks"])),
                "cold_checkpoints":cold,
                "number_of_active_blind_banks":len(state["banks"])}
        repeat_controls_not_changed(stage_rows)
        def mse(phase,family,arm="self_organizing"):
            return stage_rows[phase]["heldout"][family][arm+"_mse"]
        ratio=lambda x,y:x/max(y,1e-12)
        computed={
            "source_C_over_A":ratio(mse("C","source"),
                                    mse("A","source")),
            "source_D_over_A":ratio(mse("D","source"),
                                    mse("A","source")),
            "sine_C_over_B":ratio(mse("C","sine"),
                                 mse("B","sine")),
            "sine_gain_B_over_A":1-ratio(mse("B","sine"),
                                        mse("A","sine")),
            "rational_gain_C_over_B":1-ratio(
                mse("C","rational"),mse("B","rational")),
            "source_C_gain_vs_shared":1-ratio(
                mse("C","source"),mse("C","source","shared_no_replay")),
            "source_C_gain_vs_random_banks":1-ratio(
                mse("C","source"),mse("C","source","random_three_bank")),
            "source_C_gain_vs_original_naive_S22":1-ratio(
                mse("C","source"),mse("C","source","naive_S22")),
            "source_C_causal_history_error_ratio":ratio(
                stage_rows["C"]["heldout"]["source"][
                    "blind_bank_censored_history_mse"],
                mse("C","source")),
            "bank_count_after_B":
                stage_rows["B"]["number_of_active_blind_banks"],
            "bank_count_after_C":
                stage_rows["C"]["number_of_active_blind_banks"],
            "stationary_false_allocation_count":
                len(false_alarm_control["false_allocation_steps"]),
            "route_accuracy_after_C":
                stage_rows["C"]["permutation_audit_routing_accuracy"]
        }
        results.append({
            "seed":seed,
            "novelty_threshold":threshold,
            "stationary_negative_control":false_alarm_control,
            "calibration_source_mean_context_errors":calibration_errors,
            "stages":stage_rows,
            "observed_only_allocations":allocations,
            "frozen_ratio_metrics":computed,
            "no_task_family_label_passed_to_blind_router":True,
            "no_prior_training_worlds_replayed_in_blind_arm":True})
        require(time.monotonic()-start<
                p["frozen_thresholds"]["wall_seconds_max"],
                "TOTAL_FROZEN_WALL_CAP")
    t=p["frozen_thresholds"]
    oracle_controls=all(
        r["stages"][phase]["heldout"][fam]["oracle_mse"]<=
            t["oracle_mse_max"]
        for r in results for phase in STAGES for fam in FAMILIES)
    krr_controls=all(
        (lambda z: z["KRR_mse"]<=t["KRR_source_ood_MSE_max"] and
          (z["static_mse"]-z["KRR_mse"])/
            max(z["static_mse"],1e-12)>=
          t["KRR_source_gain_vs_static_min"] and
          z["KRR_censored_history_mse"]/
            max(z["KRR_mse"],1e-12)>=
          t["KRR_source_causal_ablation_ratio_min"])(
            r["stages"]["A"]["heldout"]["source"])
        for r in results)
    checks=[]
    for r in results:
        z=r["frozen_ratio_metrics"]
        checks.append(
            z["source_C_over_A"]<=
                t["bank_source_retention_C_over_A_MSE_max"] and
            z["source_D_over_A"]<=
                t["bank_source_recovery_D_over_A_MSE_max"] and
            z["sine_C_over_B"]<=
                t["bank_sine_retention_C_over_B_MSE_max"] and
            z["sine_gain_B_over_A"]>=
                t["bank_sine_learning_B_over_A_relative_gain_min"] and
            z["rational_gain_C_over_B"]>=
                t["bank_rational_learning_C_over_B_relative_gain_min"] and
            z["source_C_gain_vs_shared"]>=
                t["bank_source_C_relative_gain_vs_shared_min"] and
            z["source_C_gain_vs_random_banks"]>=
                t["bank_source_C_relative_gain_vs_random_three_bank_min"] and
            z["source_C_gain_vs_original_naive_S22"]>=
                t["bank_source_C_relative_gain_vs_naive_S22_min"] and
            z["stationary_false_allocation_count"]<=
                t["max_stationary_false_allocations"] and
            z["bank_count_after_B"]==
                t["required_banks_by_phase_B"] and
            z["bank_count_after_B"]<=
                t["maximum_banks_by_phase_B"] and
            z["bank_count_after_C"]==
                t["required_banks_by_phase_C"] and
            z["route_accuracy_after_C"]>=
                t["minimum_route_accuracy_per_seed_after_C"] and
            z["source_C_causal_history_error_ratio"]>=
                t["min_source_causal_ablation_ratio"]
        )
    status=(
        "ASSAY_INVALID_KERNEL_OR_ORACLE"
        if not (oracle_controls and krr_controls) else
        "PRELIMINARY_PERSISTENT_UNLABELED_CHANGE_MEMORY_NOT_SCIENTIFIC_PASS"
        if all(checks) else
        "PERSISTENT_CHANGEPOINT_BANK_ROUTING_OR_RETENTION_NULL"
    )
    aggregated={
        phase:{fam:{
            arm:mean(r["stages"][phase]["heldout"][fam][arm+"_mse"]
                     for r in results)
            for arm in NAMED_ARMS}
            for fam in FAMILIES}
        for phase in STAGES}
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(peak<=t["cpu_rss_bytes_max"],"PEAK_CPU_RSS_CAP")
    require(not torch.cuda.is_initialized(),"UNAUTHORIZED_GPU_LAUNCH")
    return {
        "schema":"ttt-hybrid.g7-s23.hysteretic-bank-evidence.v1",
        "candidate_sha":a.expected_sha,
        "status":status,
        "protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "pinned_r3_manifest_sha256":manifest,
        "tokenizer_used":False,
        "attention_used":False,
        "task_labels_passed_to_bank_router":False,
        "heldout_targets_used_for_bank_router":False,
        "source_only_novelty_calibration":True,
        "persistent_change_detection_used":True,
        "stationary_train_stream_negative_test_used":True,
        "S22_original_single_step_novelty_matched_control":True,
        "persistent_detector_never_reset_on_task_boundary":True,
        "old_train_worlds_replayed_in_blind_bank":False,
        "oracle_bank_uses_privileged_labels":True,
        "rehearsal_uses_old_train_worlds":True,
        "rehearsal_equal_information":False,
        "memory_capacity_matched_to_random_three_bank":True,
        "inference_compute_matched_to_random_bank":False,
        "source_kernel_positive_control_pass":krr_controls,
        "oracle_positive_control_pass":oracle_controls,
        "frozen_pass_by_seed":checks,
        "frozen_development_positive":all(checks) and krr_controls and oracle_controls,
        "per_seed":results,
        "aggregated_stage_family_mse":aggregated,
        "kernel_calibration_choice":{"gamma":kcfg[0],"decay":kcfg[1],
                                      "ridge":kcfg[2]},
        "kernel_calibration_losses":calibration,
        "updates_per_trainable_arm_per_seed":528,
        "worlds_per_trainable_arm_per_seed":25344,
        "cold_restores_all_models_exact":True,
        "whole_process_seconds":round(time.monotonic()-start,3),
        "peak_process_rss_bytes":peak,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    a=parser.parse_args()
    protocol=ROOT/"experiments/g7_s23_frozen_persistent_change.json"
    receipt=ROOT/"local_runs/g7_s23_persistent_banks_evidence.json"
    result={
        "schema":"ttt-hybrid.g7-s23.hysteretic-bank-evidence.v1",
        "candidate_sha":a.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "deployment":"NOT_AUTHORIZED"}
    rc=2
    try:
        require(a.protocol.resolve()==protocol.resolve() and
                a.receipt.resolve()==receipt.resolve(),
                "SOURCE_CUSTODY_FROZEN_PATH_DRIFT")
        result=run(a)
        rc=2 if result["status"]==(
            "ASSAY_INVALID_KERNEL_OR_ORACLE") else 0
    except Exception as ex:
        result["status"]="ENGINEERING_BLOCKED_"+type(ex).__name__
        result["error_type"]=type(ex).__name__
    finally:
        if a.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(
                json.dumps(result,sort_keys=True,indent=2)+"\n",
                encoding="utf-8")
        print("G7_S23_STATUS="+result["status"],flush=True)
        print("G7_S23_NO_TOKENIZER=TRUE",flush=True)
        print("G7_S23_NO_GPU=TRUE",flush=True)
    return rc

if __name__=="__main__":
    raise SystemExit(main())
