"""G7-S16: token-free nonlinear action-consequence learning in switched worlds.

Observes continuous sensor x, action u, elapsed dt, and measured x_next.
No tokenizer, vocabulary, linguistic event symbols, attention or next-token loss.

New candidate: a four-parameter surprise-dependent forgetting gate that
learns how to maintain nonlinear sufficient statistics under a world switch.
Controls: fully trained 32-unit GRU, calibration-only selected fixed-decay
ridge identification, full-history and linear RLS, static prior, exact oracle,
and unrelated-world swapped histories. Correct four-term physical basis is
explicitly supplied to estimators; do NOT claim general system discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import io
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

class Blocked(Exception):
    pass

def check(ok,msg):
    if not ok:
        raise Blocked(msg)

def basis(torch,x,u,kind="nonlinear"):
    if kind=="linear":
        return torch.stack((-x,u),dim=-1)
    return torch.stack((-x,u,torch.tanh(2*x),x*u),dim=-1)

def coeff_prior(torch,batch):
    return torch.tensor([.95,.9,0.,0.],dtype=torch.float32)[None].expand(batch,-1)

def consequence(torch,q,coef,function="known"):
    x,u,dt=q.unbind(-1)
    feature=torch.stack((
        -x,u,
        torch.tanh(3*x) if function=="novel" else torch.tanh(2*x),
        x*u),dim=-1)
    return x+dt*(feature*coef[:,None,:]).sum(-1)

def world(torch,seed,b,t,q,regime):
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    # In-regime and unseen-regime latent coefficients never appear in inputs.
    def parameters(ood):
        if ood:
            a=1.55+.70*torch.rand(b,1,generator=gen)
            gain=1.55+.55*torch.rand(b,1,generator=gen)
            c=-.8+1.6*torch.rand(b,1,generator=gen)
            d=-.65+1.3*torch.rand(b,1,generator=gen)
        else:
            a=.35+1.05*torch.rand(b,1,generator=gen)
            gain=.35+1.1*torch.rand(b,1,generator=gen)
            c=-.45+.9*torch.rand(b,1,generator=gen)
            d=-.35+.7*torch.rand(b,1,generator=gen)
        return torch.cat((a,gain,c,d),dim=-1)
    old=parameters(ood=regime=="ood_stable")
    changed=parameters(ood=regime in ("ood_switch","unseen_function","ood_stable"))
    if regime in ("train_stable","ood_stable"):
        changed=old
    check(regime in (
        "train_switch","train_stable","id_switch",
        "ood_switch","ood_stable","unseen_function"),
        "INVALID_FROZEN_WORLD_REGIME")
    x=-1+2*torch.rand(b,1,generator=gen)
    stream=[]
    for step in range(t):
        act=-1+2*torch.rand(b,1,generator=gen)
        dt=.05+.11*torch.rand(b,1,generator=gen)
        query=torch.stack((x.squeeze(-1),act.squeeze(-1),dt.squeeze(-1)),dim=-1)[:,None,:]
        after=step>=20 and regime not in ("train_stable","ood_stable")
        coef=changed if after else old
        func="novel" if after and regime=="unseen_function" else "known"
        nx=consequence(torch,query,coef,func)
        measured=nx+.002*torch.randn(b,1,generator=gen)
        stream.append(torch.cat((x,act,dt,measured),dim=-1))
        x=nx
    stream=torch.stack(stream,dim=1)
    qx=x.expand(-1,q)
    qu=-1+2*torch.rand(b,q,generator=gen)
    qdt=.05+.11*torch.rand(b,q,generator=gen)
    interventions=torch.stack((qx,qu,qdt),dim=-1)
    func="novel" if regime=="unseen_function" else "known"
    future=consequence(torch,interventions,changed,func)
    old_future=consequence(torch,interventions,old,"known")
    return stream,interventions,future,old_future,changed

def fixed_ridge(torch,stream,q,lam=1.,kind="nonlinear"):
    from torch import einsum
    x,u,dt,nx=stream.unbind(-1)
    phi=basis(torch,x,u,kind)
    y=(nx-x)/dt
    t=stream.shape[1]
    weights=torch.pow(
        torch.tensor(float(lam),dtype=stream.dtype),
        torch.arange(t-1,-1,-1,dtype=stream.dtype))
    gram=einsum("bti,btj,t->bij",phi,phi,weights)
    rhs=einsum("bti,bt,t->bi",phi,y,weights)
    d=phi.shape[-1]
    coeff=torch.linalg.solve(
        gram+.04*torch.eye(d)[None],rhs.unsqueeze(-1)).squeeze(-1)
    if kind=="linear":
        coeff=torch.cat([coeff,coeff.new_zeros(coeff.shape[0],2)],dim=1)
    return consequence(torch,q,coeff),coeff

import torch
from torch import nn

class SurpriseGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate=nn.Parameter(torch.tensor([2.0,-.5,0.,0.]))

    def forward(self,history,query,return_gate=False):
        x,u,dt,nx=history.unbind(-1)
        feat=basis(torch,x,u)
        derivative=(nx-x)/dt
        b,t,d=feat.shape
        gram=feat.new_zeros(b,d,d)
        rhs=feat.new_zeros(b,d)
        decays=[]
        for i in range(t):
            f=feat[:,i]
            y=derivative[:,i]
            diag=gram.diagonal(dim1=-2,dim2=-1)
            approximate=rhs/(diag+.2)
            innovation=(y-(f*approximate).sum(-1)).abs().clamp(max=10)
            raw=(self.gate[0]+self.gate[1]*innovation+
                 self.gate[2]*u[:,i].abs()+self.gate[3]*dt[:,i])
            decay=.4+.599*torch.sigmoid(raw)
            gram=decay[:,None,None]*gram+f[:,:,None]*f[:,None,:]
            rhs=decay[:,None]*rhs+f*y[:,None]
            decays.append(decay)
        coeff=torch.linalg.solve(
            gram+.04*torch.eye(d)[None],
            rhs.unsqueeze(-1)).squeeze(-1)
        prediction=consequence(torch,query,coeff)
        if return_gate:
            return prediction,coeff,torch.stack(decays,dim=-1).mean()
        return prediction

class ContinuousGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(input_size=4,hidden_size=32,batch_first=True)
        self.coeff=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,4))

    def forward(self,stream,query):
        latent=self.gru(stream)[1][-1]
        c=self.coeff(latent)
        calibrated=torch.stack((
            .1+2.5*torch.sigmoid(c[:,0]),
            .1+2.5*torch.sigmoid(c[:,1]),
            torch.tanh(c[:,2]),.8*torch.tanh(c[:,3])
        ),dim=-1)
        return consequence(torch,query,calibrated)

def frozen_check(p):
    check(p.get("schema")==
          "ttt-hybrid.g7-s16.tokenless-nonlinear-switch-adaptation.v1",
          "FROZEN_S16_SCHEMA")
    tr=p["train"];ev=p["evaluation"];cal=p["calibration"];lims=p["frozen_thresholds"]
    check(p["predecessor"]["s15_patch_sha"]==
          "1e634322f4668b6eab2ff41b1e6fd50e2c02e9e6"
          and tr["fit_seeds"]==[101,211,307]
          and tr["optimizer_steps_per_seed"]==240
          and tr["examples_per_optimizer_step"]==48
          and tr["histories_per_world"]==32
          and ev["splits"][1]["seeds"]==[557,613,719]
          and [s["id"] for s in ev["splits"]]==[
            "id_switch","ood_switch","ood_stable","unseen_function"]
          and cal["seeds"]==[811,823,839]
          and cal["fixed_decay_candidates"]==[
              .65,.78,.87,.93,.97,1.0]
          and lims["learned_gate_min_relative_ood_switch_gain_vs_best_fixed"]==.05
          and lims["learned_gate_min_absolute_ood_switch_gain_vs_best_fixed"]==.00002
          and p["budgets"]["cpu_only"] is True
          and lims["max_runtime_seconds"]==200,
          "FROZEN_S16_DESIGN_OR_LIMITS_DRIFT")

def trial(torch,models,history,query,target,old_target):
    from torch.nn import functional as F
    gate,gru=models
    gate.eval();gru.eval()
    with torch.no_grad():
        learned,coeff,mean_decay=gate(history,query,return_gate=True)
        neural=gru(history,query)
        no_forgetting,_=fixed_ridge(torch,history,query,1.)
        linear,_=fixed_ridge(torch,history,query,1.,"linear")
        static=consequence(torch,query,coeff_prior(torch,query.shape[0]))
        oracle=target
        swapped=history.roll(shifts=1,dims=0)
        swapped_gate=gate(swapped,query)
        flipped_query=query.clone()
        flipped_query[:,:,1]=-flipped_query[:,:,1]
        flipped_result=gate(history,flipped_query)
        updated_world_action_effect=(flipped_result-learned).abs().mean()
        old_regime_error=F.mse_loss(learned,old_target)
        result={
            "surprise_gate_mse":float(F.mse_loss(learned,target)),
            "gru_mse":float(F.mse_loss(neural,target)),
            "full_history_rls_mse":float(F.mse_loss(no_forgetting,target)),
            "linear_rls_mse":float(F.mse_loss(linear,target)),
            "static_prior_mse":float(F.mse_loss(static,target)),
            "oracle_mse":float(F.mse_loss(oracle,target)),
            "swapped_gate_mse":float(F.mse_loss(swapped_gate,target)),
            "learned_gate_mean_decay":float(mean_decay),
            "learned_gate_old_regime_mse":float(old_regime_error),
            "flipping_action_changes_prediction_mean_abs":float(updated_world_action_effect)
        }
        check(all(math.isfinite(v) for v in result.values()),
              "NONFINITE_EVALUATION")
    return result

def calibrated_decay(torch,p):
    from torch.nn import functional as F
    estimates={}
    for decay in p["calibration"]["fixed_decay_candidates"]:
        observations=[]
        for seed in p["calibration"]["seeds"]:
            h,q,y,_,_=world(
                torch,seed,64,32,8,"train_switch")
            with torch.no_grad():
                prediction,_=fixed_ridge(torch,h,q,decay)
                observations.append(float(F.mse_loss(prediction,y)))
        estimates[decay]=mean(observations)
    best=min(estimates,key=lambda x:(estimates[x],-x))
    return best,{str(k):v for k,v in estimates.items()}

def cold_exact(model,history,query,cls):
    model.eval()
    with torch.no_grad():
        before=model(history,query)
    raw=io.BytesIO()
    torch.save(model.state_dict(),raw)
    sha=hashlib.sha256(raw.getvalue()).hexdigest()
    other=cls()
    other.load_state_dict(torch.load(
        io.BytesIO(raw.getvalue()),weights_only=True))
    other.eval()
    with torch.no_grad():
        after=other(history,query)
    return sha,float((before-after).abs().max())

def run(args):
    from torch.nn import functional as F
    protocol_bytes=args.protocol.read_bytes()
    p=json.loads(protocol_bytes)
    frozen_check(p)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    check(not torch.cuda.is_initialized(),"CUDA_USED_IN_CPU_ONLY_STUDY")
    manifest,pkgs=custody.manifest()
    check(manifest==custody.R3_MANIFEST_SHA256
          and pkgs.get("torch")==custody.TORCH
          and pkgs.get("triton")==custody.TRITON,
          "EXACT_R3_PACKAGE_MANIFEST_UNQUALIFIED")
    t0=time.monotonic()
    calibrated,calibration=calibrated_decay(torch,p)
    results=[]
    for i,seed in enumerate(p["train"]["fit_seeds"]):
        torch.manual_seed(seed)
        gate=SurpriseGate()
        gate_optimizer=torch.optim.AdamW(
            gate.parameters(),lr=p["train"]["lr_gate"],weight_decay=0)
        torch.manual_seed(seed)
        gru=ContinuousGRU()
        gru_optimizer=torch.optim.AdamW(
            gru.parameters(),lr=p["train"]["lr_gru"],weight_decay=0)
        losses={"surprise_gate":[],"fully_trained_gru":[]}
        for step in range(p["train"]["optimizer_steps_per_seed"]):
            regime=p["train"]["regime_schedule"][step%2]
            h,q,y,_,_=world(
                torch,seed*100000+164000+step*37,
                p["train"]["examples_per_optimizer_step"],32,8,regime)
            for model,optimizer,key in (
                (gate,gate_optimizer,"surprise_gate"),
                (gru,gru_optimizer,"fully_trained_gru")
            ):
                model.train()
                pred=model(h,q)
                loss=F.mse_loss(pred,y)
                check(bool(torch.isfinite(loss)),"NONFINITE_TRAINING_LOSS")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad=nn.utils.clip_grad_norm_(model.parameters(),1.0)
                check(bool(torch.isfinite(grad)),
                      "NONFINITE_TRAINING_GRADIENT")
                optimizer.step()
                losses[key].append(float(loss.detach()))
            check(time.monotonic()-t0<p["frozen_thresholds"]["max_runtime_seconds"],
                  "TRAINING_RUNTIME_BUDGET_EXCEEDED")
        measured={}
        for split in p["evaluation"]["splits"]:
            h,q,y,old_y,_=world(
                torch,split["seeds"][i],
                p["evaluation"]["unseen_worlds_per_split_seed"],
                32,p["evaluation"]["counterfactual_queries_per_world"],
                split["id"])
            m=trial(torch,(gate,gru),h,q,y,old_y)
            with torch.no_grad():
                baseline,_=fixed_ridge(torch,h,q,calibrated)
                swapped,_=fixed_ridge(torch,h.roll(shifts=1,dims=0),
                                      q,calibrated)
                m["best_calibrated_fixed_mse"]=float(F.mse_loss(baseline,y))
                m["swapped_fixed_mse"]=float(F.mse_loss(swapped,y))
            check(all(math.isfinite(v) for v in m.values()),
                  "NONFINITE_SPLIT_METRIC")
            measured[split["id"]]=m
        cold_h,cold_q,_,_,_=world(
            torch,seed*100000+98777,4,32,8,"ood_switch")
        g_sha,g_delta=cold_exact(gate,cold_h,cold_q,SurpriseGate)
        gru_sha,gru_delta=cold_exact(gru,cold_h,cold_q,ContinuousGRU)
        check(g_delta==0 and gru_delta==0,
              "TRAINED_CONTINUOUS_STATE_COLD_CHECKPOINT_DRIFT")
        results.append({
            "training_seed":seed,
            "gate_parameter_count":sum(x.numel() for x in gate.parameters()),
            "gru_parameter_count":sum(x.numel() for x in gru.parameters()),
            "fit_steps":len(losses["surprise_gate"]),
            "worlds_per_model":len(losses["surprise_gate"])*
                p["train"]["examples_per_optimizer_step"],
            "fit_first_losses":{k:v[0] for k,v in losses.items()},
            "fit_last32_losses":{k:mean(v[-32:]) for k,v in losses.items()},
            "learned_gate_parameters":gate.gate.detach().tolist(),
            "calibrated_fixed_decay":calibrated,
            "splits":measured,
            "cold_gate_sha256":g_sha,
            "cold_gru_sha256":gru_sha,
            "cold_gate_max_abs_delta":g_delta,
            "cold_gru_max_abs_delta":gru_delta
        })
        check(time.monotonic()-t0<p["frozen_thresholds"]["max_runtime_seconds"],
              "MEASUREMENT_RUNTIME_BUDGET_EXCEEDED")
    agg={split["id"]:{
        k:mean(row["splits"][split["id"]][k] for row in results)
        for k in results[0]["splits"][split["id"]]
    } for split in p["evaluation"]["splits"]}
    t=p["frozen_thresholds"]
    oracle=all(r["splits"][s]["oracle_mse"]<=t["oracle_max_mse"]
               for r in results for s in agg)
    conv_valid=all(
        r["splits"]["ood_switch"]["best_calibrated_fixed_mse"]<=
            t["calibrated_fixed_ood_switch_mse_max"] and
        r["splits"]["ood_switch"]["static_prior_mse"]>=
            t["static_prior_min_ood_switch_mse"] and
        (r["splits"]["ood_switch"]["static_prior_mse"]-
         r["splits"]["ood_switch"]["best_calibrated_fixed_mse"])/
            r["splits"]["ood_switch"]["static_prior_mse"]>=
            t["calibrated_fixed_vs_static_ood_switch_fractional_gain_min"] and
        r["splits"]["ood_switch"]["swapped_fixed_mse"]/
            max(r["splits"]["ood_switch"]["best_calibrated_fixed_mse"],1e-12)>=
            t["calibrated_fixed_swapped_history_ratio_min"]
        for r in results)
    ood=agg["ood_switch"]
    comparison=(ood["best_calibrated_fixed_mse"]-
                ood["surprise_gate_mse"])
    ratio=comparison/max(ood["best_calibrated_fixed_mse"],1e-12)
    per_seed_better=[row["splits"]["ood_switch"]["surprise_gate_mse"]<
                     row["splits"]["ood_switch"]["best_calibrated_fixed_mse"]
                     for row in results]
    gate_swap_ok=all(
        row["splits"]["ood_switch"]["swapped_gate_mse"]/
        max(row["splits"]["ood_switch"]["surprise_gate_mse"],1e-12)>=
        t["learned_gate_swapped_history_ratio_min"]
        for row in results)
    learned_positive=(oracle and conv_valid and gate_swap_ok and
        ratio>=t["learned_gate_min_relative_ood_switch_gain_vs_best_fixed"]
        and comparison>=t["learned_gate_min_absolute_ood_switch_gain_vs_best_fixed"]
        and all(per_seed_better))
    gru_better=all(
        row["splits"]["ood_switch"]["gru_mse"]<
        row["splits"]["ood_switch"]["static_prior_mse"]
        for row in results)
    gru_gain=(ood["static_prior_mse"]-ood["gru_mse"])/max(
        ood["static_prior_mse"],1e-12)
    gru_positive=(oracle and conv_valid and gru_better and
        gru_gain>=t["gru_min_relative_gain_vs_static_ood_switch"])
    status=("ASSAY_INVALID_ORACLE_OR_CONVENTIONAL_CONTROL_FAILED"
            if not oracle or not conv_valid else
            "PRELIMINARY_TOKENLESS_SURPRISE_PLASTICITY_GAIN_NOT_INDEPENDENTLY_QUALIFIED"
            if learned_positive else
            "TOKENLESS_LEARNED_PLASTICITY_ADVANTAGE_NULL_ON_FROZEN_NONLINEAR_SWITCH_TEST")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    check(rss<=t["max_cpu_rss_bytes"],"CPU_RESIDENT_MEMORY_CAP_EXCEEDED")
    check(not torch.cuda.is_initialized(),"CUDA_WRONGFUL_USE")
    return {
        "schema":"ttt-hybrid.g7-s16.nonlinear-switch-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(protocol_bytes).hexdigest(),
        "pinned_package_manifest_sha256":manifest,
        "tokenizer_used":False,
        "vocabulary_used":False,
        "next_token_objective_used":False,
        "has_continuous_sensor_action_consequence_stream":True,
        "physical_time_is_sampled":True,
        "no_attention_or_transformer":True,
        "uses_privileged_known_nonlinear_feature_basis":True,
        "preselected_calibrated_decay":calibrated,
        "calibration_only_losses":calibration,
        "per_seed":results,
        "aggregate_by_split":agg,
        "oracle_controls_pass":oracle,
        "conventional_positive_controls_pass":conv_valid,
        "learned_gate_swapped_history_controls_pass":gate_swap_ok,
        "learned_gate_ood_relative_gain_over_best_calibrated_fixed":ratio,
        "learned_gate_ood_absolute_mse_improvement":comparison,
        "learned_gate_each_seed_better":per_seed_better,
        "learned_gate_frozen_thresholds_met":learned_positive,
        "trained_gru_ood_static_relative_gain":gru_gain,
        "trained_gru_each_seed_better_than_static":gru_better,
        "trained_gru_frozen_thresholds_met":gru_positive,
        "trained_updates_per_seed_per_model":240,
        "trained_worlds_per_seed_per_model":11520,
        "cold_restoration_all_models_exact":True,
        "wall_seconds":round(time.monotonic()-t0,3),
        "process_peak_rss_bytes":rss,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    args=parser.parse_args()
    expected_receipt=ROOT/"local_runs/g7_s16_nonlinear_switch_evidence.json"
    expected_protocol=ROOT/"experiments/g7_s16_frozen_nonlinear_switch.json"
    result={
        "schema":"ttt-hybrid.g7-s16.nonlinear-switch-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    exit_code=2
    try:
        check(args.receipt.resolve()==expected_receipt.resolve() and
              args.protocol.resolve()==expected_protocol.resolve(),
              "FROZEN_INPUT_OR_OUTPUT_PATH_DENIED")
        result=run(args)
        exit_code=0 if result["status"]!=(
            "ASSAY_INVALID_ORACLE_OR_CONVENTIONAL_CONTROL_FAILED") else 2
    except (Blocked,RuntimeError,ValueError,AssertionError) as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["blocked_type"]=type(exc).__name__
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["blocked_type"]=type(exc).__name__
    finally:
        if args.receipt.resolve()==expected_receipt.resolve():
            expected_receipt.parent.mkdir(parents=True,exist_ok=True)
            expected_receipt.write_text(
                json.dumps(result,sort_keys=True,indent=2)+"\n",
                encoding="utf-8")
        print("G7_S16_STATUS="+result["status"],flush=True)
        print("G7_S16_TOKENIZER=NOT_USED",flush=True)
        print("G7_S16_GPU=NOT_USED",flush=True)
    return exit_code

if __name__=="__main__":
    raise SystemExit(main())
