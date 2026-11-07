"""G7-S13: frozen mixed-delay *training* versus fixed-delay Mamba on OOD128.

Runs original source-pinned S11 Mamba and matched-attention modules unmodified.
All native arms share initial native weights, K/V examples, optimizer-step
count, and heldout split. Frozen manipulation is training delay exposure.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s6_trained_conventional_baseline as conventional
import g7_s7_native_extended_binding as native_source
import g7_s9_permutation_transfer as permutation
import g7_s11_attention_replacement as s11
import g7_s12_delayed_binding as s12

class Blocked(Exception):
    pass
PHASE="NOT_STARTED"

def check(ok,reason):
    if not ok:
        raise Blocked(reason)

def phase(phase_name):
    global PHASE
    PHASE=phase_name

def delayed_tokens(torch,k,v,q,gap,decoys,seed):
    """Same G7-S12 inputs at <=64 tokens; newly extend to 96/128."""
    if gap in (0,16,48):
        return s12.tokens_with_delay(torch,k,v,q,gap,decoys,seed)
    check(gap in (80,112),"UNFROZEN_LONG_DELAY")
    base=permutation.native_tokens(torch,k,v,q)
    b=k.shape[0]
    middle=torch.zeros(b,gap,64,dtype=torch.float32)
    if decoys:
        gen=torch.Generator(device="cpu").manual_seed(int(seed))
        n=gap//2
        fk=torch.randint(0,2,(b,n,16),generator=gen,
                         dtype=torch.int64).float()*2-1
        fv=torch.randint(0,2,(b,n,16),generator=gen,
                         dtype=torch.int64).float()*2-1
        for bit in range(16):
            match=(fk[:,:,None,:]==k[:,None,:,:]).all(-1).any(-1)
            if not bool(match.any()):
                break
            fk[:,:,bit]=torch.where(match,-fk[:,:,bit],fk[:,:,bit])
        check(not bool((fk[:,:,None,:]==k[:,None,:,:]).all(-1).any()),
              "DECOY_KEY_MATCHES_TRUE_SUPPORT")
        middle[:,0::2,:16]=fk
        middle[:,1::2,16:32]=fv
        middle[:,0::2,32]=1
        middle[:,1::2,33]=1
    x=torch.cat([base[:,:8],middle,base[:,8:]],dim=1)
    check(tuple(x.shape)==(b,16+gap,64)
          and bool((x[:,-8:,16:32]==0).all())
          and bool((x[:,-8:,34]==1).all()),
          "LONG_INPUT_OR_QUERY_TARGET_LEAK")
    return x

def freeze_check(p):
    check(p.get("schema")=="ttt-hybrid.g7-s13.frozen-mixed-delay-causal-retrieval.v1"
          and p["lineage"]["s12_run"]==42385309942
          and p["lineage"]["s12_sha"]==
              "7ae34b4cc234a9802c44eecf95254880ed6799a6"
          and p["lineage"]["verified_s11_training_source_blob"]==
              "fafb993ec3389b1b5668622d86ecd015ea659138"
          and p["lineage"]["verified_s6_generator_git_blob"]==
              "72ba66eebcebd3856d5251ccc765ea52ddf84500",
          "FROZEN_LINEAGE_DRIFT")
    check([arm["id"] for arm in p["arms"]]==[
        "conventional","mamba_fixed16","mamba_mixed","attention_mixed"]
          and p["training"]["seeds"]==[101,211,307]
          and p["training"]["steps_per_seed_arm"]==512
          and p["training"]["batch_size"]==32
          and p["training"]["gap_schedule"]==[
              "base16","blank32","decoy32","blank64","decoy64"]
          and p["heldout"]["batches_per_seed"]==4
          and p["heldout"]["batch_size"]==64
          and [v["id"] for v in p["heldout"]["variants"]]==[
              "base16","decoy64","blank96","decoy96","blank128",
              "decoy128","decoy128_corrupt_true"]
          and p["budget"]["wall_seconds_max"]==1100
          and p["budget"]["torch_peak_bytes_max"]==1073741824,
          "FROZEN_DATA_OR_BUDGET_DRIFT")
    t=p["frozen_thresholds"]
    check(t["mixed_vs_fixed_decoy128_relative_error_reduction_min"]==0.10
          and t["mixed_vs_fixed_decoy128_absolute_error_reduction_min"]==0.02
          and t["mixed_ood_decoy128_mse_max"]==0.25
          and t["mixed_ood_decoy128_over_decoy64_mse_ratio_max"]==4.0,
          "PRECOMMITTED_THRESHOLDS_DRIFT")

def evaluate(torch,model,arm_id,seed,p,device,t0):
    from torch.nn import functional as F
    names=[v["id"] for v in p["heldout"]["variants"]]
    measures={n:[] for n in names+[
        "oracle_exact_mse","oracle_wrong_mse","zero_reference_mse"]}
    is_native=arm_id!="conventional"
    model.eval()
    with torch.no_grad():
        for batch in range(p["heldout"]["batches_per_seed"]):
            episode_seed=seed*10000000+9990000+batch*17
            k,v,q,y=conventional.get_batch(
                torch,episode_seed,p["heldout"]["batch_size"],
                [0,1,2,3,3,2,1,0])
            qp,yp,kp,vp=permutation.permuted_support_queries(
                torch,k,v,q,y,episode_seed)
            mask=(qp[:,:,None,:]==kp[:,None,:,:]).all(-1)
            check(bool((mask.sum(-1)==1).all()),"AMBIGUOUS_OOD_KEYS")
            oracle=torch.einsum("bqi,biv->bqv",mask.float(),vp)
            oracle_bad=torch.einsum(
                "bqi,biv->bqv",mask.float(),torch.roll(vp,1,dims=1))
            check(bool(torch.equal(oracle,yp)),"OOD_ORACLE_TARGET_BROKEN")
            measures["oracle_exact_mse"].append(
                float(F.mse_loss(oracle,yp).item()))
            measures["oracle_wrong_mse"].append(
                float(F.mse_loss(oracle_bad,yp).item()))
            measures["zero_reference_mse"].append(
                float(yp.square().mean().item()))
            for case in p["heldout"]["variants"]:
                label=case["id"]
                if is_native:
                    altered=(torch.roll(vp,1,dims=1)
                             if case.get("cyclically_rotate_only_true_support_values")
                             else vp)
                    inputs=delayed_tokens(
                        torch,kp,altered,qp,case["gap"],case["decoys"],
                        episode_seed+451231)
                    chunks=[]
                    step=p["budget"]["native_holdout_chunk"]
                    for i in range(0,inputs.shape[0],step):
                        out=model(inputs[i:i+step].to(device))[:,-8:,:16].cpu()
                        check(bool(torch.isfinite(out).all()),
                              "NONFINITE_OOD_OUTPUT")
                        chunks.append(out)
                    predictions=torch.cat(chunks,dim=0)
                    torch.cuda.synchronize(device)
                    check(int(torch.cuda.max_memory_allocated(device))<=
                          p["budget"]["torch_peak_bytes_max"],
                          "OOD_VRAM_BUDGET")
                else:
                    # Structured task positive control, not an OOD time-gap arm.
                    predictions=model(kp,vp,qp)
                check(tuple(predictions.shape)==tuple(yp.shape),
                      "OOD_TENSOR_SHAPE_DRIFT")
                measures[label].append(
                    float(F.mse_loss(predictions,yp).item()))
                check(time.monotonic()-t0 < p["budget"]["wall_seconds_max"],
                      "OOD_TIME_BUDGET")
    return {
        "mean":{k:statistics.mean(v) for k,v in measures.items()},
        "batch_mse":measures,
    }

def cold128(torch,model,official,arm,seed,device):
    """Serialize trained weights and compare exact 128-token decoy predictions."""
    with tempfile.TemporaryDirectory(prefix="g7-s13-cold-") as path:
        file=Path(path)/"temporary-state.pt"
        torch.save(model.state_dict(),file)
        hashval=hashlib.sha256(file.read_bytes()).hexdigest()
        restored,_=s11.instantiate(torch,official,arm,seed)
        restored.load_state_dict(torch.load(
            file,map_location="cpu",weights_only=True))
        restored=restored.to(device=device,dtype=torch.float32).eval()
        model.eval()
        with torch.no_grad():
            k,v,q,_=conventional.get_batch(
                torch,seed*10000000+9999900,2,[0,1,2,3,3,2,1,0])
            x=delayed_tokens(torch,k,v,q,112,True,
                             seed*10000000+9999900+451231).to(device)
            delta=float((model(x)[:,-8:,:16]-
                         restored(x)[:,-8:,:16]).abs().max().item())
        del restored
    return hashval,delta

def run(args):
    import torch
    from torch.nn import functional as F
    phase("FROZEN_PROTOCOL_AND_CUSTODY")
    raw=args.protocol.read_bytes()
    p=json.loads(raw)
    freeze_check(p)
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s11_attention_replacement.py")==
          p["lineage"]["verified_s11_training_source_blob"],
          "ORIGINAL_MODEL_SOURCE_DRIFT")
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s6_trained_conventional_baseline.py")==
          p["lineage"]["verified_s6_generator_git_blob"],
          "QUALIFIED_CONVENTIONAL_SOURCE_DRIFT")
    official,manifest=native_source.check_environment(
        torch,{"budget":{"required_free_cuda_bytes_min":
                          p["budget"]["cuda_free_min"]}},args)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda:0")
    t0=time.monotonic()
    records=[]
    max_allocated=0
    for seed in p["training"]["seeds"]:
        arms={}
        initial_hashes={}
        for arm in p["arms"]:
            arm_id=arm["id"]
            phase("INITIALIZE_"+arm_id)
            native=arm_id!="conventional"
            if native:
                model,initial_hash=s11.instantiate(torch,official,arm,seed)
                initial_hashes[arm_id]=initial_hash
                model=model.to(device=device,dtype=torch.float32)
                # Same authentic full model as previously qualified S11.
                if arm["kind"]=="full":
                    comparison_arm=dict(arm,id="full_static")
                    s11.verify_full_forward_exact(
                        torch,official,model,comparison_arm,seed,device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                torch.manual_seed(seed)
                model=conventional.new_model(torch)
            full_count=sum(x.numel() for x in model.parameters())
            live=[x for x in model.parameters() if x.requires_grad]
            live_count=sum(x.numel() for x in live)
            check(full_count==arm["model_parameters_total"] and
                  live_count==arm["trainable_parameters"],
                  "NATIVE_OR_CONVENTIONAL_CAPACITY_DRIFT")
            optim=torch.optim.AdamW(live,lr=arm["train_lr"],weight_decay=0)
            model.train()
            steps=[]
            by_case={}
            training_start=time.monotonic()
            for step in range(p["training"]["steps_per_seed_arm"]):
                episode_seed=seed*10000000+step*17
                k,v,q,y=conventional.get_batch(
                    torch,episode_seed,p["training"]["batch_size"],
                    [0,1,2,3,3,2,1,0])
                qp,yp,kp,vp=permutation.permuted_support_queries(
                    torch,k,v,q,y,episode_seed)
                selected=("base16" if arm_id in
                          ("mamba_fixed16","conventional") else
                          p["training"]["gap_schedule"][step%5])
                by_case[selected]=by_case.get(selected,0)+1
                if native:
                    if selected=="base16": gap,decoys=0,False
                    elif selected=="blank32": gap,decoys=16,False
                    elif selected=="decoy32": gap,decoys=16,True
                    elif selected=="blank64": gap,decoys=48,False
                    elif selected=="decoy64": gap,decoys=48,True
                    else: raise Blocked("UNFROZEN_TRAINING_CASE")
                    sequence=delayed_tokens(
                        torch,kp,vp,qp,gap,decoys,episode_seed+451231).to(device)
                    target=yp.to(device)
                    output=model(sequence)[:,-8:,:16]
                else:
                    output=model(kp,vp,qp)
                    target=yp
                check(tuple(output.shape)==tuple(target.shape),
                      "TRAIN_TARGET_SHAPE_DRIFT")
                loss=F.mse_loss(output,target)
                check(bool(torch.isfinite(loss)),"NONFINITE_TRAIN_LOSS")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(live,1.0)
                check(bool(torch.isfinite(grad_norm)),"NONFINITE_TRAIN_GRAD")
                optim.step()
                steps.append(float(loss.detach().item()))
                if native:
                    torch.cuda.synchronize(device)
                    current=int(torch.cuda.max_memory_allocated(device))
                    check(current<=p["budget"]["torch_peak_bytes_max"],
                          "TRAIN_VRAM_BUDGET_EXCEEDED")
                    max_allocated=max(max_allocated,current)
                check(time.monotonic()-t0<p["budget"]["wall_seconds_max"],
                      "TRAIN_WALL_BUDGET_EXCEEDED")
            fit_time=time.monotonic()-training_start
            phase("FRESH_LONGER_DELAY_HELDOUT_"+arm_id)
            eval_start=time.monotonic()
            evaluated=evaluate(torch,model,arm_id,seed,p,device,t0)
            if native:
                phase("COLD_RESTORE_LONG128_"+arm_id)
                check_sha,delta=cold128(torch,model,official,arm,seed,device)
                check(delta==0,"LONG_COLD_RESTORE_NOT_EXACT")
                max_allocated=max(max_allocated,
                    int(torch.cuda.max_memory_allocated(device)))
            else:
                check_sha,delta=permutation.baseline_cold(
                    torch,model,seed,[0,1,2,3,3,2,1,0])
                check(delta==0,"CONVENTIONAL_COLD_RESTORE_NOT_EXACT")
            arm_evaluation_seconds=time.monotonic()-eval_start
            arms[arm_id]={
                "total_parameters":full_count,
                "active_trainable_parameters":live_count,
                "training_updates":len(steps),
                "training_episodes":len(steps)*p["training"]["batch_size"],
                "training_case_update_counts":by_case,
                "train_first_loss":steps[0],
                "train_last32_loss":statistics.mean(steps[-32:]),
                "train_seconds":round(fit_time,3),
                "eval_and_cold_seconds":round(arm_evaluation_seconds,3),
                "cold_checkpoint_sha256":check_sha,
                "cold128_max_abs_delta":delta,
                "heldout":evaluated,
            }
            del model,optim
            if native:
                torch.cuda.empty_cache()
            check(time.monotonic()-t0<p["budget"]["wall_seconds_max"],
                  "POST_EVAL_WALL_BUDGET")
        check(len(set(initial_hashes.values()))==1,
              "NATIVE_INITIAL_WEIGHT_MISMATCH")
        check(all(a["training_updates"]==512 and
                  a["training_episodes"]==16384 for a in arms.values()),
              "UNEQUAL_DATA_EXPOSURE_OR_STEPS")
        records.append({"seed":seed,
                        "native_common_initial_sha256":
                            next(iter(initial_hashes.values())),
                        "arms":arms})
    phase("AGGREGATE_FROZEN_OOD_THRESHOLDS")
    names=[arm["id"] for arm in p["arms"]]
    means={n:{k:statistics.mean(
        seed["arms"][n]["heldout"]["mean"][k] for seed in records)
        for k in records[0]["arms"][n]["heldout"]["mean"]}
        for n in names}
    t=p["frozen_thresholds"]
    oracle_ok=all(
        abs(m["zero_reference_mse"]-t["zero_mse_exact"])<1e-12
        and m["oracle_exact_mse"]<=t["oracle_exact_mse_max"]
        and m["oracle_wrong_mse"]>=t["wrong_oracle_mse_min"]
        for m in means.values())
    familiar_ok=(
        means["conventional"]["base16"]<=t["conventional_base_mse_max"]
        and means["mamba_fixed16"]["base16"]<=t["mamba_fixed_base_mse_max"]
        and means["mamba_mixed"]["base16"]<=t["mamba_mixed_base_mse_max"]
        and means["attention_mixed"]["base16"]<=t["attention_mixed_base_mse_max"])
    fixed=means["mamba_fixed16"]["decoy128"]
    mixed=means["mamba_mixed"]["decoy128"]
    attention=means["attention_mixed"]["decoy128"]
    check(fixed>0 and mixed>0 and attention>0 and
          means["mamba_mixed"]["decoy64"]>0,
          "NONPOSITIVE_ERROR_DENOMINATOR")
    gain=(fixed-mixed)/fixed
    gain_abs=fixed-mixed
    seed_gain=[row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy128"]<
               row["arms"]["mamba_fixed16"]["heldout"]["mean"]["decoy128"]
               for row in records]
    ratio=mixed/means["mamba_mixed"]["decoy64"]
    ood_seeds=[
        row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy128"]<=
            t["mixed_ood_decoy128_mse_max"]
        and row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy64"]>0
        and row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy128"]/
            row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy64"]<=
            t["mixed_ood_decoy128_over_decoy64_mse_ratio_max"]
        for row in records]
    corrupted=means["mamba_mixed"]["decoy128_corrupt_true"]
    corruption_harm=(corrupted-mixed)/mixed
    accepted=oracle_ok and familiar_ok
    training_gain=(accepted
        and gain>=t["mixed_vs_fixed_decoy128_relative_error_reduction_min"]
        and gain_abs>=t["mixed_vs_fixed_decoy128_absolute_error_reduction_min"]
        and all(seed_gain))
    retention=(accepted and all(ood_seeds)
        and corruption_harm>=t["mixed_corrupt_true_support_relative_harm_min"])
    vs_attention=(attention-mixed)/attention
    attention_abs=attention-mixed
    vs_attention_seeds=[
        row["arms"]["mamba_mixed"]["heldout"]["mean"]["decoy128"]<
        row["arms"]["attention_mixed"]["heldout"]["mean"]["decoy128"]
        for row in records]
    mamba_contribution=(accepted
        and vs_attention>=t["mamba_vs_attention128_relative_error_reduction_min"]
        and attention_abs>=t["mamba_vs_attention128_absolute_error_reduction_min"]
        and all(vs_attention_seeds))
    state=("ASSAY_INVALID_CONTROL_FAILURE" if not accepted else
           "PRELIMINARY_MIXED_DELAY_GAIN_REQUIRES_REPLICATION"
           if training_gain else "NULL_OR_NEGATIVE_MIXED_DELAY_GAIN")
    return {
        "schema":"ttt-hybrid.g7-s13.mixed-delay-ood-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":state,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "s11_original_model_source_blob":
            p["lineage"]["verified_s11_training_source_blob"],
        "pinned_runtime_manifest_sha256":manifest,
        "per_seed":records,
        "aggregate_means":means,
        "oracle_controls_pass":oracle_ok,
        "familiar_native_baselines_pass":familiar_ok,
        "training_gain_relative":gain,
        "training_gain_absolute_mse":gain_abs,
        "training_gain_each_seed":seed_gain,
        "mixed_training_gain_frozen_thresholds_met":training_gain,
        "mixed_decoy128_over_decoy64_ratio":ratio,
        "mixed_ood_robust_each_seed":ood_seeds,
        "mixed_corrupted_true_support_relative_harm":corruption_harm,
        "mixed_ood_robust_frozen_thresholds_met":retention,
        "mamba_vs_attention128_relative_mse_gain":vs_attention,
        "mamba_vs_attention128_abs_mse_gain":attention_abs,
        "mamba_vs_attention128_each_seed":vs_attention_seeds,
        "exploratory_mamba_vs_attention128_thresholds_met":mamba_contribution,
        "training_updates_per_seed_arm":512,
        "training_episodes_per_seed_arm":16384,
        "heldout_episodes_per_seed_arm":256,
        "same_key_value_training_information_across_arms":True,
        "training_intervention_schedule_differs_by_design":True,
        "equal_active_parameters_within_mamba_training_arms":True,
        "near_parameter_matched_attention_not_flop_matched":True,
        "conventional_is_structured_reference_only":True,
        "new_128token_cold_restoration_exact":True,
        "max_torch_gpu_allocation_bytes":max_allocated,
        "whole_science_wall_seconds":round(time.monotonic()-t0,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    receipt=ROOT/"local_runs/g7_s13_mixed_delay_evidence.json"
    report={
        "schema":"ttt-hybrid.g7-s13.mixed-delay-ood-evidence.v1",
        "status":"ENGINEERING_UNVERIFIED","candidate_sha":args.expected_sha,
        "phase":"NOT_STARTED","independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    code=2
    try:
        check(args.receipt.resolve()==receipt.resolve()
              and args.protocol.resolve()==
              (ROOT/"experiments/g7_s13_frozen_mixed_delay.json").resolve(),
              "UNAUTHORIZED_PROTOCOL_OR_RECEIPT_PATH")
        report=run(args)
        code=0 if report["status"] in (
            "PRELIMINARY_MIXED_DELAY_GAIN_REQUIRES_REPLICATION",
            "NULL_OR_NEGATIVE_MIXED_DELAY_GAIN") else 2
    except (Blocked,custody.Blocked,s11.Blocked,native_source.Blocked) as exc:
        report["status"]="ENGINEERING_BLOCKED_"+str(exc)
        report["phase"]=PHASE
    except Exception as exc:
        report["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        report["phase"]=PHASE
        report["error_class"]=type(exc).__name__
    finally:
        if args.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                               encoding="utf-8")
        print("G7_S13_STATUS="+report["status"],flush=True)
        print("G7_S13_PHASE="+report.get("phase","COMPLETE"),flush=True)
        print("G7_S13_INDEPENDENT=NOT_RUN",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
