"""G7-S21: unconventional orthogonal episodic gradient sketch for tokenless memory.

S20 partial-observation source and S18 model implementations are immutable,
exact-blobs checked. In B/C/D a stored basis of past TRAINING gradients
projects the actual AdamW parameter step to avoid modifying directions
important to preceding tasks. No former episodes replay in that arm.

Controlled arms: ordinary no-replay finetune, RANDOM rank-matched step
projection, and a 25% historical-data rehearsal arm with disclosure of
greater former-task information. GRU receives the same new-family data
and optimizer update count, but has a different parameter/FLOP budget.

This is a bounded experiment in synthetic sampled real-valued state/action/
time/observation, without tokenizer, vocabulary or attention. Gradient
projection and rehearsal have prior art. No general cognition claim.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
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
import g7_s20_hidden_multicycle as s20
import torch
from torch import nn
from torch.nn import functional as F

class Blocked(Exception):
    pass

def require(ok,msg):
    if not ok:
        raise Blocked(msg)

s18=s20.s18
PHASES=("A","B","C","D")
FAMILIES=("source","sine","rational")
FEATURE_ARMS=("protected","fine_tune","random","rehearsal")

def verify_frozen(p):
    t=p["training"];z=p["thresholds"]
    require(
        p.get("schema")==
          "ttt-hybrid.g7-s21.frozen-orthogonal-continuous-memory.v1"
        and p["predecessor"]["s20_sha"]==
          "dfdf6706a69097027490d580659be6c0f6401aea"
        and p["predecessor"]["s20_run"]==42400091370
        and p["predecessor"]["s20_source_blob"]==
          "ac3f9ddd5b879e53e62030707fb6419adaf732ca"
        and p["predecessor"]["s18_model_blob"]==
          "cfc1959b8812e9abad581018462302bc89a31d59"
        and t["seeds"]==[1361,1429,1531]
        and t["batch"]==48
        and [x["updates"] for x in t["phases"]]==[240,96,96,96]
        and [x["family"] for x in t["phases"]]==[
          "source","sine","rational","source"]
        and [x["seed_offset"] for x in t["phases"]]==[
          1201000,1301000,1401000,1501000]
        and t["total_updates_per_model"]==528
        and t["worlds_per_model"]==25344
        and p["mechanism"]["capture_last_steps_per_family"]==12
        and p["mechanism"]["svd_relative_threshold"]==.001
        and p["mechanism"]["max_total_rank"]==36
        and p["training"]["rehearsal"]["B"]=={
          "sine":36,"source":12}
        and p["training"]["rehearsal"]["C"]=={
          "rational":36,"source":12}
        and p["training"]["rehearsal"]["D"]=={
          "source":36,"sine":6,"rational":6}
        and p["calibration"]["seeds"]==[2819,2833,2843]
        and p["evaluation"]["seeds"]=={
          "source":[2903,2917,2927],
          "sine":[3001,3011,3023],
          "rational":[3109,3121,3137]}
        and p["evaluation"]["stages"]==["A","B","C","D"]
        and z["protected_source_C_gain_vs_finetune_min"]==.15
        and z["protected_source_C_gain_vs_random_min"]==.10
        and z["protected_source_C_over_A_max"]==1.5
        and z["protected_sine_C_over_B_max"]==1.35
        and z["wall_seconds_max"]==240
        and p["environment"]["tokenizer"] is False
        and p["environment"]["attention"] is False
        and p["authority"]["cpu_only"] is True,
        "FROZEN_G7_S21_PROTOCOL_OR_SCOPE_DRIFT")

def model_params(model):
    return tuple(model.parameters())

def vector(model):
    return torch.cat([x.detach().reshape(-1) for x in model_params(model)])

def gradient_vector(model):
    return torch.cat([
        (x.grad.detach() if x.grad is not None else torch.zeros_like(x))
        .reshape(-1) for x in model_params(model)
    ]).clone()

def set_vector(model,v):
    with torch.no_grad():
        offset=0
        for param in model_params(model):
            count=param.numel()
            param.copy_(v[offset:offset+count].reshape(param.shape))
            offset+=count
        require(offset==v.numel(),"VECTOR_PARAMETER_NUMBER_MISMATCH")

def orthobasis(gradients,rel):
    require(len(gradients)>0,"NO_PAST_TASK_GRADIENTS")
    m=torch.stack(gradients,dim=1)
    require(bool(torch.isfinite(m).all()),"NONFINITE_GRADIENT_SKETCH")
    q,s,_=torch.linalg.svd(m,full_matrices=False)
    require(bool(torch.isfinite(s).all()) and bool((s>=0).all()),
            "NONFINITE_GRADIENT_SINGULAR_VALUES")
    rank=int((s>s[0]*rel).sum().item()) if float(s[0])>0 else 0
    require(rank>0,"PAST_TASK_GRADIENT_SKETCH_ZERO_RANK")
    return q[:,:rank].contiguous(),rank

def random_basis(n,rank,seed):
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    sample=torch.randn((n,rank),generator=gen)
    q,_=torch.linalg.qr(sample,mode="reduced")
    return q.contiguous()

def train_step(model,optimizer,h,q,y,projection=None):
    t0=time.monotonic()
    pred=s18.learned_ridge(model,h,q)
    loss=F.mse_loss(pred,y)
    require(bool(torch.isfinite(loss)),"NONFINITE_FEATURE_TRAINING_LOSS")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
    require(bool(torch.isfinite(norm)),"NONFINITE_FEATURE_TRAINING_GRAD")
    sketch=gradient_vector(model)
    before=vector(model) if projection is not None else None
    optimizer.step()
    removed_fraction=0.
    if projection is not None:
        delta=vector(model)-before
        removal=projection@(projection.T@delta)
        safe=delta-removal
        set_vector(model,before+safe)
        removed_fraction=float(removal.norm()/(delta.norm()+1e-12))
    return float(loss.detach()),sketch,removed_fraction,time.monotonic()-t0

def train_gru(gru,opt,h,q,y):
    start=time.monotonic()
    out=gru(h,q)
    loss=F.mse_loss(out,y)
    require(bool(torch.isfinite(loss)),"NONFINITE_GRU_LOSS")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm=nn.utils.clip_grad_norm_(gru.parameters(),1.)
    require(bool(torch.isfinite(norm)),"NONFINITE_GRU_GRAD")
    opt.step()
    return float(loss.detach()),time.monotonic()-start

def calibrate(p,start):
    c=p["calibration"]
    worlds=[
        s20.partial_world(seed,c["batch"],"train_switch")
        for seed in c["seeds"]]
    grid=list(itertools.product(c["gamma"],c["decay"],c["ridge"]))
    losses={}
    with torch.no_grad():
        for cfg in grid:
            scores=[float(F.mse_loss(
                s18.kernel_ridge(h,q,*cfg),y))
                for h,q,y in worlds]
            losses[cfg]=mean(scores)
            require(time.monotonic()-start<p["thresholds"]["wall_seconds_max"],
                    "CALIBRATION_EXCEEDED_TOTAL_TIME")
    winner=min(grid,key=lambda x:(losses[x],x[0],-x[1],-x[2]))
    return winner,{str(x):losses[x] for x in grid}

def rehearsal_batch(seed,phase,step,h,q,y,p):
    if phase=="A":
        return h,q,y
    b=36
    idx=seed*100000+1601000+step*37
    ah,aq,ay=h[:b],q[:b],y[:b]
    if phase in ("B","C"):
        # Historical task TRAIN distribution, never prior heldout worlds.
        old_h,old_q,old_y=s20.partial_world(
            idx,12,"source_train")
        return (torch.cat((ah,old_h),dim=0),
                torch.cat((aq,old_q),dim=0),
                torch.cat((ay,old_y),dim=0))
    # Phase D: fixed two memory-rehearsal categories.
    sh,sq,sy=s20.partial_world(idx,6,"sine_train")
    rh,rq,ry=s20.partial_world(
        seed*100000+1701000+step*37,6,"rational_train")
    return (torch.cat((ah,sh,rh),dim=0),
            torch.cat((aq,sq,rq),dim=0),
            torch.cat((ay,sy,ry),dim=0))

def measured_heldout(feature_models,random_init,gru,kcfg,seed,p,idx):
    res={}
    for family in FAMILIES:
        h,q,y=s20.partial_world(
            p["evaluation"]["seeds"][family][idx],
            p["evaluation"]["batch"],family+"_ood")
        values={}
        with torch.no_grad():
            for arm in FEATURE_ARMS:
                start=time.monotonic()
                pred=s18.learned_ridge(feature_models[arm],h,q)
                values[arm+"_mse"]=float(F.mse_loss(pred,y))
                values[arm+"_infer_s"]=time.monotonic()-start
            for name,func in (
                ("untrained",lambda:s18.learned_ridge(random_init,h,q)),
                ("GRU",lambda:gru(h,q)),
                ("KRR",lambda:s18.kernel_ridge(h,q,*kcfg)),
                ("static",lambda:q[:,:,0]),
                ("oracle",lambda:y)
            ):
                t0=time.monotonic()
                pred=func()
                values[name+"_mse"]=float(F.mse_loss(pred,y))
                values[name+"_infer_s"]=time.monotonic()-t0
            # Same source history for all arms, but last 12 post-switch
            # sensor/action/consequence observations are replaced with the
            # first twelve, not hidden state or privileged latent variables.
            if family=="source":
                censored=torch.cat((h[:,:20,:],h[:,:12,:]),dim=1)
                values["KRR_source_censored_mse"]=float(F.mse_loss(
                    s18.kernel_ridge(censored,q,*kcfg),y))
                values["protected_source_censored_mse"]=float(F.mse_loss(
                    s18.learned_ridge(
                        feature_models["protected"],censored,q),y))
            require(all(math.isfinite(v) for v in values.values()),
                    "NONFINITE_HELDOUT_MEASURED")
        res[family]=values
    return res

def checkpoint(model,h,q):
    return s18.checkpoint_cold(model,h,q)

def fit_source(model,gru,optf,optg,seed,p,start):
    training=p["training"]["phases"][0]
    grads=[]
    mtime=0.
    gtime=0.
    for step in range(training["updates"]):
        h,q,y=s20.partial_world(
            seed*100000+training["seed_offset"]+step*37,
            p["training"]["batch"],"train_switch")
        _,g,_,dt=train_step(model,optf,h,q,y)
        mtime+=dt
        _,dtgru=train_gru(gru,optg,h,q,y)
        gtime+=dtgru
        if step>=training["updates"]-12:
            grads.append(g)
        require(time.monotonic()-start<p["thresholds"]["wall_seconds_max"],
                "PHASE_A_TIME_CAP")
    require(len(grads)==12,"SOURCE_GRADIENT_SKETCH_INCOMPLETE")
    return grads,{"feature_training_seconds":mtime,
                  "gru_training_seconds":gtime,"steps":240}

def train_stage(models,optimizers,gru,gru_opt,seed,p,stage,start,sketches):
    spec=next(x for x in p["training"]["phases"] if x["id"]==stage)
    source_grads=sketches["source"]
    all_previous=[*source_grads]
    if stage in ("C","D"):
        all_previous+=sketches["sine"]
    if stage=="D":
        all_previous+=sketches["rational"]
    basis,rank=orthobasis(
        all_previous,p["mechanism"]["svd_relative_threshold"])
    require(0<rank<=p["mechanism"]["max_total_rank"],
            "PAST_GRADIENT_MEMORY_RANK_INVALID")
    rand=random_basis(vector(models["random"]).numel(),
                      rank,seed+7300+{"B":1,"C":2,"D":3}[stage])
    protect_training=[]
    times={arm:0. for arm in FEATURE_ARMS}
    times["GRU"]=0.
    source_delta_discarded=[]
    for step in range(spec["updates"]):
        h,q,y=s20.partial_world(
            seed*100000+spec["seed_offset"]+step*37,
            p["training"]["batch"],spec["family"]+"_train")
        rh,rq,ry=rehearsal_batch(seed,stage,step,h,q,y,p)
        for arm in FEATURE_ARMS:
            context=(rh,rq,ry) if arm=="rehearsal" else (h,q,y)
            projection=(basis if arm=="protected" else
                        rand if arm=="random" else None)
            _,g,removed,elapsed=train_step(
                models[arm],optimizers[arm],*context,
                projection=projection)
            times[arm]+=elapsed
            if arm=="protected":
                if step>=spec["updates"]-12:
                    protect_training.append(g)
                source_delta_discarded.append(removed)
        _,dtgru=train_gru(gru,gru_opt,h,q,y)
        times["GRU"]+=dtgru
        require(time.monotonic()-start<p["thresholds"]["wall_seconds_max"],
                "MULTICYCLE_TIME_CAP")
    require(len(protect_training)==12,"NOVEL_GRADIENT_SKETCH_INCOMPLETE")
    return protect_training,{
        "phase":stage,"family":spec["family"],"steps":spec["updates"],
        "projection_rank":rank,
        "subspace_memory_bytes":int(basis.numel()*basis.element_size()),
        "random_control_memory_bytes":int(rand.numel()*rand.element_size()),
        "mean_discarded_step_fraction":mean(source_delta_discarded),
        "train_seconds_by_arm":{k:round(v,5) for k,v in times.items()},
        "rehearsal_old_samples_per_step":12,
        "no_old_samples_for_protected_finetune_random_GRU":True
    }

def run(a):
    raw=a.protocol.read_bytes()
    p=json.loads(raw)
    verify_frozen(p)
    require(custody.git(ROOT,"hash-object",
            "scripts/g7_s20_hidden_multicycle.py")==
            p["predecessor"]["s20_source_blob"] and
            custody.git(ROOT,"hash-object",
            "scripts/g7_s18_causal_feature_replication.py")==
            p["predecessor"]["s18_model_blob"],
            "ORIGINAL_HIDDEN_STATE_OR_MODEL_IMPLEMENTATION_DRIFT")
    manifest,packages=custody.manifest()
    require(manifest==custody.R3_MANIFEST_SHA256
        and packages.get("torch")==custody.TORCH
        and packages.get("triton")==custody.TRITON,
        "PINNED_ORIGINAL_R3_ENVIRONMENT_DRIFT")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(),"CPU_STUDY_MUST_NOT_USE_CUDA")
    start=time.monotonic()
    kcfg,kdetail=calibrate(p,start)
    rows=[]
    for index,seed in enumerate(p["training"]["seeds"]):
        torch.manual_seed(seed)
        primary=s18.RawFeatureBank()
        never_trained=copy.deepcopy(primary)
        gru=s18.RawGRU()
        optf=torch.optim.AdamW(primary.parameters(),
            lr=p["training"]["feature_lr"],
            weight_decay=p["training"]["weight_decay"])
        optg=torch.optim.AdamW(gru.parameters(),
            lr=p["training"]["gru_lr"],
            weight_decay=p["training"]["weight_decay"])
        source_grad,a_time=fit_source(
            primary,gru,optf,optg,seed,p,start)
        models={name:copy.deepcopy(primary) for name in FEATURE_ARMS}
        state=copy.deepcopy(optf.state_dict())
        optimizers={}
        for name in FEATURE_ARMS:
            opt=torch.optim.AdamW(models[name].parameters(),
                lr=p["training"]["feature_lr"],
                weight_decay=p["training"]["weight_decay"])
            opt.load_state_dict(copy.deepcopy(state))
            optimizers[name]=opt
        require(all(torch.equal(vector(models["protected"]),
                                vector(models[name]))
                    for name in FEATURE_ARMS),
                "ARMS_NOT_IDENTICALLY_INITIALIZED_AFTER_SOURCE")
        sketch={"source":source_grad}
        history={}
        timing={"A":a_time}
        for stage in PHASES:
            if stage!="A":
                new_grads,stage_stats=train_stage(
                    models,optimizers,gru,optg,seed,p,stage,start,sketch)
                if stage=="B":sketch["sine"]=new_grads
                if stage=="C":sketch["rational"]=new_grads
                timing[stage]=stage_stats
            metric=measured_heldout(
                models,never_trained,gru,kcfg,seed,p,index)
            h,q,_=s20.partial_world(
                seed*100000+777999,4,"source_ood")
            restored={}
            for arm in FEATURE_ARMS:
                digest,delta=checkpoint(models[arm],h,q)
                require(delta==0.0,"FEATURE_COLD_CHECKPOINT_DRIFT")
                restored[arm]={"sha256":digest,"delta":delta}
            digest,delta=checkpoint(gru,h,q)
            require(delta==0.0,"RECURRENT_COLD_CHECKPOINT_DRIFT")
            restored["GRU"]={"sha256":digest,"delta":delta}
            history[stage]={"heldout":metric,"cold":restored}
        # All calibration-only baseline and random-init controls must be
        # mathematically identical across ALL four learning phases.
        for family in FAMILIES:
            for stage in ("B","C","D"):
                for name in ("KRR_mse","untrained_mse","static_mse",
                             "oracle_mse"):
                    require(history["A"]["heldout"][family][name]==
                            history[stage]["heldout"][family][name],
                            "FROZEN_CONVENTIONAL_OR_UNTRAINED_DRIFT")
        def mse(stage,family,arm="protected"):
            return history[stage]["heldout"][family][arm+"_mse"]
        ratio=lambda x,y:x/max(y,1e-12)
        row={
            "seed":seed,"stages":history,"training":timing,
            "source_C_over_A":ratio(mse("C","source"),
                                   mse("A","source")),
            "source_D_over_A":ratio(mse("D","source"),
                                   mse("A","source")),
            "sine_C_over_B":ratio(mse("C","sine"),
                                 mse("B","sine")),
            "sine_gain_B_over_A":1-ratio(
                mse("B","sine"),mse("A","sine")),
            "rational_gain_C_over_B":1-ratio(
                mse("C","rational"),mse("B","rational")),
            "source_C_protected_gain_vs_plain":1-ratio(
                mse("C","source"),mse("C","source","fine_tune")),
            "source_C_protected_gain_vs_random":1-ratio(
                mse("C","source"),mse("C","source","random")),
            "basis_max_rank":max(timing[phase]["projection_rank"]
                    for phase in ("B","C","D")),
            "old_world_replay_in_protected_arm":False,
            "cold_restored_all_arms_at_all_stages":True
        }
        rows.append(row)
        require(time.monotonic()-start<p["thresholds"]["wall_seconds_max"],
                "FROZEN_TOTAL_TIME_BUDGET")
    t=p["thresholds"]
    oracle_ok=all(
        row["stages"][st]["heldout"][family]["oracle_mse"]<=
            t["oracle_mse_max"]
        for row in rows for st in PHASES for family in FAMILIES)
    control_ok=all(
        row["stages"]["A"]["heldout"]["source"]["KRR_mse"]<=
            t["calibrated_KRR_source_mse_max"]
        and (row["stages"]["A"]["heldout"]["source"]["static_mse"]-
             row["stages"]["A"]["heldout"]["source"]["KRR_mse"])/
            max(row["stages"]["A"]["heldout"]["source"]["static_mse"],1e-12)>=
            t["KRR_source_gain_vs_static_min"]
        and row["stages"]["A"]["heldout"]["source"]["KRR_source_censored_mse"]/
            max(row["stages"]["A"]["heldout"]["source"]["KRR_mse"],1e-12)>=
            t["KRR_post_switch_ablation_ratio_min"]
        for row in rows)
    pass_by_seed=[
        row["source_C_over_A"]<=t["protected_source_C_over_A_max"] and
        row["source_D_over_A"]<=t["protected_source_D_over_A_max"] and
        row["sine_C_over_B"]<=t["protected_sine_C_over_B_max"] and
        row["rational_gain_C_over_B"]>=
          t["protected_rational_gain_C_over_B_min"] and
        row["sine_gain_B_over_A"]>=
          t["protected_sine_gain_B_over_A_min"] and
        row["source_C_protected_gain_vs_plain"]>=
          t["protected_source_C_gain_vs_finetune_min"] and
        row["source_C_protected_gain_vs_random"]>=
          t["protected_source_C_gain_vs_random_min"] and
        row["basis_max_rank"]>=t["minimum_source_basis_rank"] and
        row["basis_max_rank"]<=t["maximum_total_basis_rank"]
        for row in rows]
    positive=(oracle_ok and control_ok and all(pass_by_seed))
    status=("ASSAY_INVALID_SOURCE_CONTROL_OR_ORACLE"
            if not oracle_ok or not control_ok else
            "PRELIMINARY_ORTHOGONAL_TOKENLESS_RETENTION_GAIN_NOT_SCIENTIFIC_PASS"
            if positive else
            "ORTHOGONAL_MEMORY_FROZEN_RETENTION_OR_PLASTICITY_NULL")
    aggregate={
        stage:{family:{
            arm:mean(r["stages"][stage]["heldout"][family][arm+"_mse"]
                     for r in rows)
            for arm in ("protected","fine_tune","random","rehearsal",
                        "KRR","GRU","untrained","static")}
            for family in FAMILIES}
        for stage in PHASES
    }
    max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    require(max_rss<=t["cpu_rss_max_bytes"],"CPU_RSS_CAP_EXCEEDED")
    require(not torch.cuda.is_initialized(),"UNAUTHORIZED_GPU")
    return {
        "schema":"ttt-hybrid.g7-s21.orthogonal-memory-evidence.v1",
        "candidate_sha":a.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "r3_pinned_manifest_sha256":manifest,
        "tokenizer_used":False,"attention_used":False,
        "numeric_time_sampled":True,
        "same_initial_learned_weights_and_optimizer_state":True,
        "protected_no_old_episode_replay":True,
        "rehearsal_uses_old_train_episode_information":True,
        "rehearsal_new_family_fraction":.75,
        "all_trained_model_cold_restores_exact":True,
        "original_S18_S20_model_sources_exact":True,
        "source_oracle_positive_controls_pass":oracle_ok,
        "source_kernel_positive_controls_pass":control_ok,
        "per_seed":rows,
        "aggregate":aggregate,
        "calibration_only_kernel_choice":{
            "gamma":kcfg[0],"decay":kcfg[1],"ridge":kcfg[2]},
        "calibration_candidate_losses":kdetail,
        "frozen_success_per_seed":pass_by_seed,
        "frozen_science_development_criteria_met":positive,
        "train_updates_per_arm_per_seed":528,
        "train_worlds_per_arm_per_seed":25344,
        "process_wall_seconds":round(time.monotonic()-start,3),
        "peak_rss_bytes":max_rss,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    a=parser.parse_args()
    protocol=ROOT/"experiments/g7_s21_frozen_orthogonal_memory.json"
    receipt=ROOT/"local_runs/g7_s21_orthogonal_memory_evidence.json"
    result={
        "schema":"ttt-hybrid.g7-s21.orthogonal-memory-evidence.v1",
        "status":"ENGINEERING_UNVERIFIED",
        "candidate_sha":a.expected_sha,
        "deployment":"NOT_AUTHORIZED"
    }
    rc=2
    try:
        require(a.protocol.resolve()==protocol.resolve()
                and a.receipt.resolve()==receipt.resolve(),
                "FROZEN_PROTOCOL_OR_RECEIPT_PATH_DRIFT")
        result=run(a)
        rc=2 if result["status"]=="ASSAY_INVALID_SOURCE_CONTROL_OR_ORACLE" else 0
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["blocker_type"]=type(exc).__name__
    finally:
        if a.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                               encoding="utf-8")
        print("G7_S21_STATUS="+result["status"],flush=True)
        print("G7_S21_NO_TOKENIZER=TRUE",flush=True)
        print("G7_S21_CUDA=NOT_USED",flush=True)
    return rc

if __name__=="__main__":
    raise SystemExit(main())
