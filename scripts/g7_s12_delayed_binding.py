"""G7-S12 frozen delayed-retrieval diagnostic.

Reuses byte-identical G7-S11 *training* and source-custody code. Hooks only
the end-of-training cold checks to measure new heldout delay/interference;
does not change model, optimizer, training data, weights, or prior thresholds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s11_attention_replacement as s11
import g7_s6_trained_conventional_baseline as conventional
import g7_s9_permutation_transfer as permutation

class Blocked(Exception):
    pass

PHASE="NOT_STARTED"

def check(value,reason):
    if not value:
        raise Blocked(reason)

def tokens_with_delay(torch,k,v,q,gap,decoys,seed):
    base=permutation.native_tokens(torch,k,v,q)
    if gap==0:
        check(not decoys,"DECOY_WITHOUT_DELAY")
        return base
    check(gap in (16,48),"UNFROZEN_DELAY_LENGTH")
    b=k.shape[0]
    middle=torch.zeros(b,gap,64,dtype=torch.float32)
    if decoys:
        gen=torch.Generator(device="cpu").manual_seed(int(seed))
        n=gap//2
        fake_keys=torch.randint(0,2,(b,n,16),generator=gen,
                                dtype=torch.int64).float()*2-1
        fake_values=torch.randint(0,2,(b,n,16),generator=gen,
                                  dtype=torch.int64).float()*2-1
        for bit in range(16):
            collision=(fake_keys[:,:,None,:]==k[:,None,:,:]).all(-1).any(-1)
            if not bool(collision.any()):
                break
            fake_keys[:,:,bit]=torch.where(collision,
                                          -fake_keys[:,:,bit],
                                          fake_keys[:,:,bit])
        check(not bool((fake_keys[:,:,None,:]==k[:,None,:,:]).all(-1).any()),
              "DISTRACTOR_KEY_EQUALS_GENUINE")
        middle[:,0::2,:16]=fake_keys
        middle[:,1::2,16:32]=fake_values
        middle[:,0::2,32]=1
        middle[:,1::2,33]=1
    result=torch.cat([base[:,:8],middle,base[:,8:]],dim=1)
    check(tuple(result.shape)==(b,16+gap,64)
          and bool((result[:,-8:,16:32]==0).all()),
          "DELAYED_INPUT_SHAPE_OR_LABEL_LEAK")
    return result

def heldout(torch,model,seed,arm_id,protocol,device):
    from torch.nn import functional as F
    cases=protocol["heldout"]["cases"]
    names=[x["id"] for x in cases]
    metrics={x:[] for x in names+[
        "oracle_correct_mse","oracle_corrupted_mse","zero_predictor_mse"]}
    model.eval()
    is_native=arm_id!="conventional"
    with torch.no_grad():
        for batch in range(protocol["heldout"]["batches_per_seed"]):
            epi=seed*10000000+9980000+batch*17
            k,v,q,y=conventional.get_batch(
                torch,epi,protocol["heldout"]["examples_per_batch"],
                [0,1,2,3,3,2,1,0])
            qp,yp,kp,vp=permutation.permuted_support_queries(
                torch,k,v,q,y,epi)
            matches=(qp[:,:,None,:]==kp[:,None,:,:]).all(-1)
            check(bool((matches.sum(-1)==1).all()),
                  "AMBIGUOUS_EPISODIC_LOOKUP")
            lookup=matches.float()
            oracle=torch.einsum("bqi,biv->bqv",lookup,vp)
            wrong=torch.einsum("bqi,biv->bqv",lookup,torch.roll(vp,1,dims=1))
            check(bool(torch.equal(oracle,yp)),
                  "POSITIVE_ORACLE_OR_TARGET_MISMATCH")
            metrics["oracle_correct_mse"].append(
                float(F.mse_loss(oracle,yp).item()))
            metrics["oracle_corrupted_mse"].append(
                float(F.mse_loss(wrong,yp).item()))
            metrics["zero_predictor_mse"].append(
                float(yp.square().mean().item()))
            if not is_native:
                val=float(F.mse_loss(model(kp,vp,qp),yp).item())
                for case in cases:
                    metrics[case["id"]].append(val)
                continue
            for case in cases:
                label=case["id"]
                values=(torch.roll(vp,1,dims=1)
                        if label=="decoy64_corrupted_true_support" else vp)
                sample=tokens_with_delay(
                    torch,kp,values,qp,
                    gap=case["gap_tokens"],
                    decoys=label.startswith("decoy"),
                    seed=epi+451231)
                predictions=[]
                chunk=protocol["budget"]["heldout_native_chunk_size"]
                for idx in range(0,sample.shape[0],chunk):
                    inp=sample[idx:idx+chunk].to(device)
                    out=model(inp)[:,-8:,:16].cpu()
                    check(bool(torch.isfinite(out).all()),
                          "NONFINITE_DELAYED_RECALL")
                    predictions.append(out)
                pred=torch.cat(predictions,dim=0)
                check(tuple(pred.shape)==tuple(yp.shape),
                      "INVALID_DELAYED_RECALL_SHAPE")
                metrics[label].append(float(F.mse_loss(pred,yp).item()))
                torch.cuda.synchronize(device)
                check(torch.cuda.max_memory_allocated(device)<=
                      protocol["budget"]["torch_peak_allocated_bytes_max"],
                      "GPU_DELAY_VRAM_LIMIT")
    return {key:statistics.mean(vals) for key,vals in metrics.items()}

def cold_delay(torch,model,official,arm,seed,device):
    with tempfile.TemporaryDirectory(prefix="g7-s12-checkpoint-") as d:
        path=Path(d)/"temporary.pt"
        torch.save(model.state_dict(),path)
        restored,_=s11.instantiate(torch,official,arm,seed)
        restored.load_state_dict(torch.load(
            path,map_location="cpu",weights_only=True))
        restored=restored.to(device=device,dtype=torch.float32).eval()
        model.eval()
        with torch.no_grad():
            k,v,q,_=conventional.get_batch(
                torch,seed*10000000+9989900,2,[0,1,2,3,3,2,1,0])
            x=tokens_with_delay(
                torch,k,v,q,48,True,seed*10000000+9989900+451231).to(device)
            before=model(x)[:,-8:,:16]
            after=restored(x)[:,-8:,:16]
            delta=float((before-after).abs().max().item())
        del restored
    return delta

def run(args):
    global PHASE
    import torch
    PHASE="PRE_REGISTERED_PROTOCOL"
    raw=args.protocol.read_bytes()
    p=json.loads(raw)
    check(p.get("schema")=="ttt-hybrid.g7-s12.delayed-distractor-binding.v1",
          "FROZEN_PROTOCOL_SCHEMA_DRIFT")
    check(p["predecessor"]["s11_run"]==42383988759 and
          p["predecessor"]["qualified_generator_git_blob"]==
          "72ba66eebcebd3856d5251ccc765ea52ddf84500" and
          p["training"]["seeds"]==[101,211,307] and
          p["training"]["fit_steps"]==512 and
          p["training"]["batch_size"]==64 and
          p["heldout"]["batches_per_seed"]==4 and
          p["heldout"]["examples_per_batch"]==64 and
          p["budget"]["max_wall_seconds"]==900 and
          p["budget"]["heldout_native_chunk_size"]==16 and
          [c["id"] for c in p["heldout"]["cases"]]==[
              "base16","blank32","decoy32","blank64","decoy64",
              "decoy64_corrupted_true_support"],
          "FROZEN_RESEARCH_DESIGN_DRIFT")
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s11_attention_replacement.py")==
          "fafb993ec3389b1b5668622d86ecd015ea659138",
          "S11_VERIFIED_TRAINING_SOURCE_CHANGED")
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s6_trained_conventional_baseline.py")==
          "72ba66eebcebd3856d5251ccc765ea52ddf84500",
          "S6_VERIFIED_GENERATOR_CHANGED")
    PHASE="ORIGINAL_S11_TRAINING_WITH_HELDOUT_HOOK"
    original_native=s11.cold_native
    original_conventional=permutation.baseline_cold
    readings={}
    cold_deltas={}
    def native_hook(torch_,model,official,arm,seed,device):
        global PHASE
        PHASE="NEW_DELAYED_EVAL_"+arm["id"]
        readings.setdefault(seed,{})[arm["id"]]=heldout(
            torch_,model,seed,arm["id"],p,device)
        PHASE="NEW_DELAYED_COLD_RESTORATION_"+arm["id"]
        cold_deltas.setdefault(seed,{})[arm["id"]]=cold_delay(
            torch_,model,official,arm,seed,device)
        return original_native(torch_,model,official,arm,seed,device)
    def conventional_hook(torch_,model,seed,order):
        global PHASE
        PHASE="NEW_STRUCTURED_POSITIVE_REFERENCE"
        readings.setdefault(seed,{})["conventional"]=heldout(
            torch_,model,seed,"conventional",p,None)
        return original_conventional(torch_,model,seed,order)
    s11.cold_native=native_hook
    permutation.baseline_cold=conventional_hook
    start=time.monotonic()
    try:
        original=s11.run(SimpleNamespace(
            expected_sha=args.expected_sha,
            protocol=ROOT/"experiments/g7_s11_frozen_attention_replacement.json"))
    finally:
        s11.cold_native=original_native
        permutation.baseline_cold=original_conventional
    check(original["oracle_controls_pass"] is True
          and original["conventional_positive_control_pass"] is True
          and original["full_static_positive_control_pass"] is True,
          "ORIGINAL_S11_SCIENCE_CONTROLS_NOT_REPRODUCED")
    check(set(readings)==set(p["training"]["seeds"]) and
          all(set(readings[seed])==set(x["id"] for x in p["arms"])
              for seed in p["training"]["seeds"]) and
          all(len(cold_deltas.get(seed,{}))==3 for seed in p["training"]["seeds"])
          and all(value==0.0 for d in cold_deltas.values() for value in d.values()),
          "MISSING_ARM_OR_LONG_DELAY_COLD_RESTORE_DRIFT")
    PHASE="AGGREGATE_NEW_DELAY_AND_DISTRACTOR_RESULTS"
    arm_names=[x["id"] for x in p["arms"]]
    metrics={arm:{name:statistics.mean(
        readings[seed][arm][name] for seed in p["training"]["seeds"])
        for name in readings[p["training"]["seeds"][0]][arm]}
        for arm in arm_names}
    t=p["thresholds"]
    oracle_valid=all(
        abs(m["zero_predictor_mse"]-t["zero_predictor_mse_exact"])<1e-12
        and m["oracle_correct_mse"]<=t["oracle_correct_mse_max"]
        and m["oracle_corrupted_mse"]>=t["oracle_corrupted_mse_min"]
        for m in metrics.values())
    positive_reference=all(
        readings[seed]["conventional"]["base16"]<=
        t["conventional_structured_mse_max"] for seed in p["training"]["seeds"])
    full=metrics["full_static"]
    attention=metrics["attention_replaces_mamba"]
    base_valid=full["base16"]<=t["full_base16_mse_max"] and \
        attention["base16"]<=t["attention_base16_mse_max"]
    check(full["base16"]>0 and attention["base16"]>0
          and full["decoy64"]>0 and attention["decoy64"]>0,
          "INVALID_MSE_DENOMINATOR")
    full_corruption_harm=(
        full["decoy64_corrupted_true_support"]-full["decoy64"]
    )/full["decoy64"]
    full_seed_gains=[]
    mamba_each_seed=[]
    for seed in p["training"]["seeds"]:
        a=readings[seed]["full_static"]
        b=readings[seed]["attention_replaces_mamba"]
        check(a["base16"]>0,"PER_SEED_BASELINE_ZERO")
        full_seed_gains.append(
            a["decoy64"]<=t["full_decoy64_absolute_mse_max"]
            and a["decoy64"]/a["base16"]<=
            t["full_decoy64_over_base16_mse_ratio_max"])
        mamba_each_seed.append(a["decoy64"]<b["decoy64"])
    assay_valid=oracle_valid and positive_reference and base_valid
    retained=(assay_valid and all(full_seed_gains)
              and full_corruption_harm>=
              t["decoy64_corrupt_true_support_relative_error_increase_min"])
    advantage=(assay_valid and
               full["decoy64"]/attention["decoy64"]<=
               t["mamba_vs_replacement_decoy64_ratio_max"] and
               attention["decoy64"]-full["decoy64"]>=
               t["mamba_vs_replacement_abs_mse_advantage_min"]
               and all(mamba_each_seed))
    status=("ASSAY_INVALID_CONTROL_FAILURE" if not assay_valid else
            "PRELIMINARY_NATIVE_DELAYED_BINDING_SIGNAL_REQUIRES_REPLICATION"
            if retained else "UNFAMILIAR_DELAY_OR_DECOY_INTERFERENCE_NOT_RESOLVED")
    return {
        "schema":"ttt-hybrid.g7-s12.delayed-binding-scientific-evidence.v1",
        "candidate_sha":args.expected_sha,"status":status,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "reused_exact_s11_training_git_blob":
            "fafb993ec3389b1b5668622d86ecd015ea659138",
        "previous_s11_reproduced_science_status":original["status"],
        "previous_s11_reproduced_science_controls_pass":True,
        "per_seed_arms":readings,
        "aggregate_heldout_means":metrics,
        "per_seed_cold_delayed64_deltas":cold_deltas,
        "oracle_controls_pass":oracle_valid,
        "conventional_positive_control_pass":positive_reference,
        "native_familiar_baselines_pass":base_valid,
        "full_decoy64_over_base16_mse_ratio":full["decoy64"]/full["base16"],
        "replacement_decoy64_over_base16_mse_ratio":
            attention["decoy64"]/attention["base16"],
        "mamba_decoy64_over_replacement_mse_ratio":
            full["decoy64"]/attention["decoy64"],
        "mamba_decoy64_absolute_mse_advantage":
            attention["decoy64"]-full["decoy64"],
        "mamba_better_each_seed":mamba_each_seed,
        "mamba_exploratory_delay_advantage_thresholds_met":advantage,
        "full_decoy64_corruption_relative_harm":full_corruption_harm,
        "full_all_seeds_robust_to_delay":full_seed_gains,
        "full_delayed_retrieval_thresholds_met":retained,
        "all_new_delayed_cold_restores_exact":True,
        "training_episodes_per_seed_per_arm":32768,
        "fresh_heldout_episodes_per_seed_per_arm":
            p["heldout"]["episodes_per_seed_arm"],
        "conventional_temporal_comparator":"STRUCTURED_REFERENCE_ONLY",
        "equal_trained_source_and_episodes":True,
        "equal_compute_flops_or_energy":False,
        "max_torch_allocated_bytes":original["max_torch_allocated_bytes"],
        "elapsed_seconds":round(time.monotonic()-start,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    destination=ROOT/"local_runs/g7_s12_delayed_binding.json"
    result={
        "schema":"ttt-hybrid.g7-s12.delayed-binding-scientific-evidence.v1",
        "candidate_sha":a.expected_sha,"status":"ENGINEERING_UNVERIFIED",
        "phase":"NOT_STARTED","independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    code=2
    try:
        check(a.receipt.resolve()==destination.resolve() and
              a.protocol.resolve()==
                  (ROOT/"experiments/g7_s12_frozen_delayed_binding.json").resolve(),
              "FROZEN_INPUT_OR_RECEIPT_PATH_DENIED")
        result=run(a)
        code=0 if result["status"] in (
            "PRELIMINARY_NATIVE_DELAYED_BINDING_SIGNAL_REQUIRES_REPLICATION",
            "UNFAMILIAR_DELAY_OR_DECOY_INTERFERENCE_NOT_RESOLVED") else 2
    except (Blocked,custody.Blocked,s11.Blocked) as exc:
        result["status"]="ENGINEERING_BLOCKED_"+str(exc)
        result["phase"]=PHASE
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["phase"]=PHASE
        result["error_class"]=type(exc).__name__
    finally:
        if a.receipt.resolve()==destination.resolve():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                                   encoding="utf-8")
        print("G7_S12_STATUS="+result["status"],flush=True)
        print("G7_S12_PHASE="+result.get("phase","COMPLETE"),flush=True)
        print("G7_S12_INDEPENDENT=NOT_RUN",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
