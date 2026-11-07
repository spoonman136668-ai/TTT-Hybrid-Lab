"""G7-S15 raw continuous action-consequence world modeling.

No tokenizer, discrete key/value symbols, learned token embeddings, text,
presegmented linguistic units, or next-token loss. Inputs are sampled real
sensor states, actuator commands and elapsed physical time.

Strong classical online system identification is the RLS control; a freshly
trained 32-unit GRU is a neural alternative. Neither method is novel. Time
sampling is not hidden; this is NOT proof of unsampled analog computing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
from statistics import mean
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

class Blocked(Exception):
    pass

def require(ok,reason):
    if not ok:
        raise Blocked(reason)

def make_world(seed,batch,length,nquery,regime):
    """Sample unfamiliar continuous scalar physical plant and consequences."""
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    if regime=="train" or regime=="id":
        damping=.3+1.3*torch.rand((batch,1),generator=generator)
        gain=.2+1.1*torch.rand((batch,1),generator=generator)
    elif regime=="ood":
        damping=1.65+.65*torch.rand((batch,1),generator=generator)
        gain=1.35+.45*torch.rand((batch,1),generator=generator)
    else:
        raise Blocked("UNKNOWN_PHYSICS_REGIME")
    state=-1+2*torch.rand((batch,1),generator=generator)
    observations=[]
    for _ in range(length):
        action=-1+2*torch.rand((batch,1),generator=generator)
        elapsed=.05+.15*torch.rand((batch,1),generator=generator)
        next_state=state+elapsed*(-damping*state+gain*action)
        measured=next_state+.003*torch.randn((batch,1),generator=generator)
        observations.append(torch.cat((state,action,elapsed,measured),dim=-1))
        state=next_state
    sensor_stream=torch.stack(observations,dim=1)
    qstate=state.expand(batch,nquery)
    qaction=-1+2*torch.rand((batch,nquery),generator=generator)
    qelapsed=.05+.15*torch.rand((batch,nquery),generator=generator)
    future=qstate+qelapsed*(-damping*qstate+gain*qaction)
    return (sensor_stream,
            torch.stack((qstate,qaction,qelapsed),dim=-1),
            future,torch.cat((damping,gain),dim=-1))

class AnalogGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory=nn.GRU(input_size=4,hidden_size=32,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,2))

    def forward(self,stream,interventions):
        state=self.memory(stream)[1][-1]
        coeff=self.readout(state)
        a=.1+2.5*torch.sigmoid(coeff[:,0:1])
        b=.05+2*torch.sigmoid(coeff[:,1:2])
        x,u,dt=interventions.unbind(-1)
        return x+dt*(-a*x+b*u)

def online_rls(stream,interventions,regularizer=.015):
    """Classical continuous system-identification baseline, not token memory."""
    x=stream[:,:,0]
    u=stream[:,:,1]
    dt=stream[:,:,2]
    measured_next=stream[:,:,3]
    features=torch.stack((-x,u),dim=-1)
    observed_derivative=(measured_next-x)/dt
    gram=torch.einsum("bti,btj->bij",features,features)
    rhs=torch.einsum("bti,bt->bi",features,observed_derivative)
    identity=torch.eye(2)[None].expand(stream.shape[0],-1,-1)
    estimate=torch.linalg.solve(gram+regularizer*identity,rhs)
    qx,qu,qdt=interventions.unbind(-1)
    return qx+qdt*(-estimate[:,0:1]*qx+estimate[:,1:2]*qu),estimate

def global_static(interventions):
    x,u,dt=interventions.unbind(-1)
    return x+dt*(-.95*x+.75*u)

def measure(model,seed,batch,regime):
    from torch.nn import functional as F
    model.eval()
    with torch.no_grad():
        stream,query,target,latent=make_world(seed,batch,32,8,regime)
        gru=model(stream,query)
        rls,estimated=online_rls(stream,query)
        static=global_static(query)
        x,u,dt=query.unbind(-1)
        oracle=x+dt*(-latent[:,0:1]*x+latent[:,1:2]*u)
        swapped_indices=torch.randperm(
            batch,generator=torch.Generator().manual_seed(seed+777))
        wrong_gru=model(stream[swapped_indices],query)
        wrong_rls,_=online_rls(stream[swapped_indices],query)
        flipped=query.clone()
        flipped[:,:,1]=-query[:,:,1]
        flipped_oracle=x+dt*(-latent[:,0:1]*x-latent[:,1:2]*u)
        flipped_gru=model(stream,flipped)
        values={
            "gru_mse":float(F.mse_loss(gru,target)),
            "rls_mse":float(F.mse_loss(rls,target)),
            "global_static_mse":float(F.mse_loss(static,target)),
            "gru_swapped_history_mse":float(F.mse_loss(wrong_gru,target)),
            "rls_swapped_history_mse":float(F.mse_loss(wrong_rls,target)),
            "oracle_mse":float(F.mse_loss(oracle,target)),
            "gru_counterfactual_action_mse":float(
                F.mse_loss(flipped_gru,flipped_oracle)),
            "rls_latent_damping_gain_mae":float(
                (estimated-latent).abs().mean()),
        }
    require(all(math_is_finite(x) for x in values.values()),
            "NONFINITE_HELDOUT_METRIC")
    return values

def math_is_finite(x):
    import math
    return math.isfinite(x)

def frozen_check(spec):
    require(spec.get("schema")=="ttt-hybrid.g7-s15.raw-analog-dynamics.v1",
            "FROZEN_PROTOCOL_SCHEMA")
    tr=spec["training"];te=spec["evaluation"];s=spec["thresholds"]
    inp=spec["numerical_input"];budget=spec["resource_limits"]
    require(tr["steps"]==240 and tr["batch"]==48
            and tr["seeds"]==[101,211,307]
            and tr["context_steps"]==32
            and te["seeds"]==[149,271,397]
            and te["ood_seeds"]==[557,613,719]
            and te["batch"]==64 and te["query_count"]==8
            and tr["lr"]==.004
            and budget["seconds_max"]==180
            and budget["cpu_only"] is True
            and inp["tokenizer"] is False
            and inp["vocabulary"] is False
            and inp["discrete_key_ids"] is False
            and inp["event_markers"] is False
            and s["rls_gain_vs_static_min"]==.85
            and s["gru_id_gain_vs_static_min"]==.1
            and s["gru_ood_gain_vs_static_min"]==.1,
            "FROZEN_PARAMETERS_OR_TOKENLESS_CONTRACT_DRIFT")

def run(args):
    raw=args.protocol.read_bytes()
    spec=json.loads(raw)
    frozen_check(spec)
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(),
            "CPU_ONLY_SOURCE_ACCESS_BREACH")
    started=time.monotonic()
    train=spec["training"]
    testing=spec["evaluation"]
    per_seed=[]
    from torch.nn import functional as F
    for index,seed in enumerate(train["seeds"]):
        torch.manual_seed(seed)
        model=AnalogGRU()
        optimizer=torch.optim.AdamW(model.parameters(),lr=train["lr"])
        losses=[]
        model.train()
        for step in range(train["steps"]):
            stream,query,target,_=make_world(
                seed*100000+17000+step*37,train["batch"],
                train["context_steps"],testing["query_count"],"train")
            prediction=model(stream,query)
            loss=F.mse_loss(prediction,target)
            require(bool(torch.isfinite(loss)),"NONFINITE_TRAIN_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(bool(torch.isfinite(norm)),"NONFINITE_TRAIN_GRADIENT")
            optimizer.step()
            losses.append(float(loss.detach()))
            require(time.monotonic()-started<
                    spec["resource_limits"]["seconds_max"],
                    "BOUNDED_TRAINING_TIME_EXCEEDED")
        id_measure=measure(model,testing["seeds"][index],
                           testing["batch"],"id")
        ood_measure=measure(model,testing["ood_seeds"][index],
                            testing["batch"],"ood")
        cold_stream,cold_query,_,_=make_world(
            seed*100000+92371,4,train["context_steps"],
            testing["query_count"],"ood")
        model.eval()
        with torch.no_grad():
            pre=model(cold_stream,cold_query)
        buf=io.BytesIO()
        torch.save(model.state_dict(),buf)
        checkpoint_sha=hashlib.sha256(buf.getvalue()).hexdigest()
        restored=AnalogGRU()
        restored.load_state_dict(torch.load(
            io.BytesIO(buf.getvalue()),weights_only=True))
        restored.eval()
        with torch.no_grad():
            restored_prediction=restored(cold_stream,cold_query)
            cold_delta=float((pre-restored_prediction).abs().max())
        require(cold_delta==0.0,"TRAINED_GRU_COLD_RESTORE_DRIFT")
        per_seed.append({
            "fit_seed":seed,
            "model_parameters":sum(x.numel() for x in model.parameters()),
            "updates":len(losses),
            "trained_worlds":len(losses)*train["batch"],
            "training_first_loss":losses[0],
            "training_last32_loss":mean(losses[-32:]),
            "id_seed":testing["seeds"][index],
            "ood_seed":testing["ood_seeds"][index],
            "id":id_measure,
            "ood":ood_measure,
            "trained_model_checkpoint_sha256":checkpoint_sha,
            "cold_restore_output_max_abs_delta":cold_delta,
        })
        require(time.monotonic()-started<spec["resource_limits"]["seconds_max"],
                "BOUNDED_MEASUREMENT_TIME_EXCEEDED")
    t=spec["thresholds"]
    oracle_valid=all(
        row[split]["oracle_mse"]<=t["oracle_mse_max"]
        for row in per_seed for split in ("id","ood"))
    rls_valid=all(
        row["id"]["rls_mse"]<=t["rls_id_mse_max"] and
        row["ood"]["rls_mse"]<=t["rls_ood_mse_max"] and
        (row[split]["global_static_mse"]-row[split]["rls_mse"])/
            row[split]["global_static_mse"]>=
            t["rls_gain_vs_static_min"] and
        row[split]["rls_swapped_history_mse"]/
            row[split]["rls_mse"]>=t["rls_shuffled_ratio_min"]
        for row in per_seed for split in ("id","ood"))
    gru_id_gain=all(
        (row["id"]["global_static_mse"]-row["id"]["gru_mse"])/
            row["id"]["global_static_mse"]>=
            t["gru_id_gain_vs_static_min"] and
        row["id"]["gru_swapped_history_mse"]/
            row["id"]["gru_mse"]>=t["gru_shuffled_ratio_min"]
        for row in per_seed)
    gru_ood_gain=all(
        (row["ood"]["global_static_mse"]-row["ood"]["gru_mse"])/
            row["ood"]["global_static_mse"]>=
            t["gru_ood_gain_vs_static_min"] and
        row["ood"]["gru_swapped_history_mse"]/
            row["ood"]["gru_mse"]>=t["gru_shuffled_ratio_min"]
        for row in per_seed)
    metrics={
        split:{
            key:mean(row[split][key] for row in per_seed)
            for key in per_seed[0][split]
        }
        for split in ("id","ood")
    }
    status=(
        "ASSAY_INVALID_ORACLE_OR_RLS_CONTROL"
        if not oracle_valid or not rls_valid else
        "TRAINABLE_CONTINUOUS_GRU_ADAPTATION_SIGNAL"
        if gru_id_gain and gru_ood_gain else
        "ANALOG_RLS_PASS_GRU_TRANSFER_NULL_OR_NEGATIVE")
    result={
        "schema":"ttt-hybrid.g7-s15.raw-analog-dynamics-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "direct_measurement_channels":[
            "observed_state","applied_action","elapsed_time",
            "measured_next_state"],
        "tokenizer_used":False,
        "vocabulary_used":False,
        "discrete_key_ids_used":False,
        "event_marker_ids_used":False,
        "next_token_loss_used":False,
        "numerical_time_sampling_used":True,
        "parameters_hidden_from_model":True,
        "trained_continuous_gru":True,
        "online_rls_adapts_within_each_world":True,
        "three_fresh_independent_training_seeds":True,
        "per_seed":per_seed,
        "aggregated_metrics":metrics,
        "oracle_controls_pass":oracle_valid,
        "strong_rls_adaptation_controls_pass":rls_valid,
        "trainable_gru_id_frozen_gain_met":gru_id_gain,
        "trainable_gru_ood_frozen_gain_met":gru_ood_gain,
        "training_updates_per_seed":train["steps"],
        "training_worlds_per_seed":train["steps"]*train["batch"],
        "heldout_worlds_per_seed_per_split":testing["batch"],
        "whole_process_wall_seconds":round(time.monotonic()-started,3),
        "process_peak_rss_bytes":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }
    require(result["process_peak_rss_bytes"]<=
            spec["resource_limits"]["peak_cpu_rss_bytes_max"],
            "CPU_RSS_BUDGET_EXCEEDED")
    require(not torch.cuda.is_initialized(),
            "CPU_EXPERIMENT_UNEXPECTED_CUDA")
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    expected_receipt=ROOT/"local_runs/g7_s15_raw_analog_results.json"
    expected_protocol=ROOT/"experiments/g7_s15_frozen_raw_analog_dynamics.json"
    result={
        "schema":"ttt-hybrid.g7-s15.raw-analog-dynamics-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }
    code=2
    try:
        require(args.receipt.resolve()==expected_receipt.resolve()
                and args.protocol.resolve()==expected_protocol.resolve(),
                "FROZEN_PROTOCOL_OR_OUTPUT_PATH_DENIED")
        result=run(args)
        code=0 if result["status"]!="ASSAY_INVALID_ORACLE_OR_RLS_CONTROL" else 2
    except (Blocked,RuntimeError,ValueError,AssertionError) as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["failure_class"]=type(exc).__name__
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["failure_class"]=type(exc).__name__
    finally:
        if args.receipt.resolve()==expected_receipt.resolve():
            args.receipt.parent.mkdir(parents=True,exist_ok=True)
            args.receipt.write_text(
                json.dumps(result,sort_keys=True,indent=2)+"\n",
                encoding="utf-8")
        print("G7_S15_STATUS="+result["status"],flush=True)
        print("G7_S15_GRADIENT_TRAINING=GRU_ONLY",flush=True)
        print("G7_S15_NO_TOKENIZER=TRUE",flush=True)
        print("G7_S15_CUDA=NOT_USED",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
