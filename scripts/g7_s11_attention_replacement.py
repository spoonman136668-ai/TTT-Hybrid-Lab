"""G7-S11: original pinned Mamba2 versus near-active-parameter-matched attention replacement.

All inputs, training steps and episode seeds are identical across four arms.
Removing an active module changes effective trainable parameter count and
compute, so the result is *not* a capacity- or FLOP-matched superiority claim.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody
import g7_s6_trained_conventional_baseline as conventional_source
import g7_s7_native_extended_binding as native_source
import g7_s9_permutation_transfer as prior_transfer

class Blocked(Exception):
    pass

PHASE = "NOT_STARTED"

def check(value, reason):
    if not value:
        raise Blocked(reason)

def phase(value):
    global PHASE
    PHASE = value

def instantiate(torch, official, arm, seed):
    from authentic_hybrid import AuthenticHybrid
    from model import CausalAttentionBlock
    from torch import nn
    class AblatedNative(AuthenticHybrid):
        def __init__(self):
            super().__init__(official_classes=official, width=64, heads=4,
                             mini_batch=8, mamba_mem_eff_path=False,
                             use_ttt=True, use_depth=True)
            self.active_mamba = arm["kind"] == "full"
            self.active_attention = True
            self.active_outer_ttt = True
            self.ttt.config.ttt_base_lr = 0.0
            # Construct extra ordinary causal attention only AFTER creating
            # the identical seeded native base. Two blocks are +33,536
            # trainable parameters, within 1% of Mamba's 34,438 active params.
            self.extra_attn = nn.ModuleList([
                CausalAttentionBlock(64, 4) for _ in range(2)
            ] if arm["kind"] == "attention_replace" else [])
            with torch.no_grad():
                self.depth.router.weight.zero_()
                self.depth.router.bias.fill_(1.0)
            for p in self.depth.router.parameters():
                p.requires_grad_(False)
            if not self.active_mamba:
                for module in (self.norm_in, self.mamba):
                    for p in module.parameters():
                        p.requires_grad_(False)

        def forward(self, x):
            if x.ndim != 3 or x.shape[2] != self.width or x.shape[1] < 1:
                raise ValueError("expected nonempty [batch,time,width]")
            b, t, _ = x.shape
            h = x + self.mamba(self.norm_in(x)) if self.active_mamba else x
            for layer in self.extra_attn:
                h = layer(h)
            h = self.attn(h)
            if self.active_outer_ttt:
                positions = torch.arange(t, device=x.device).unsqueeze(0).expand(b,-1)
                h = h + self.ttt(
                    self.norm_ttt(h), position_ids=positions,cache_params=None)
            h = self.depth(h)
            return self.output(h)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=AblatedNative()
    # Compare initial *shared* base tensors across all three native arms.
    # Exclude only additional attention replacement parameters.
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if not name.startswith("extra_attn."):
            h.update(name.encode("utf-8")+b"\0")
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return model, h.hexdigest()

def verify_full_forward_exact(torch, official, model, arm, seed, device):
    from authentic_hybrid import AuthenticHybrid
    if arm["id"] != "full_static":
        return None
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reference=AuthenticHybrid(
        official_classes=official,width=64,heads=4,mini_batch=8,
        mamba_mem_eff_path=False,use_ttt=True,use_depth=True)
    with torch.no_grad():
        reference.depth.router.weight.zero_()
        reference.depth.router.bias.fill_(1.0)
    reference.ttt.config.ttt_base_lr = 0.0
    check(native_source.state_hash(reference)==native_source.state_hash(model),
          "ABLATED_FULL_MODULE_INITIALIZATION_NOT_ORIGINAL")
    reference=reference.to(device=device,dtype=torch.float32).eval()
    model.eval()
    with torch.no_grad():
        k,v,q,_=conventional_source.get_batch(
            torch,seed*10000000+9670000,2,[0,1,2,3,3,2,1,0])
        tokens=prior_transfer.native_tokens(torch,k,v,q).to(device)
        delta=float((model(tokens)-reference(tokens)).abs().max().item())
    del reference
    check(delta==0.0,"ABLATED_FULL_FORWARD_DIFFERS_FROM_ORIGINAL")
    return delta

def cold_native(torch,model,official,arm,seed,device):
    phase("COLD_RESTORE_"+arm["id"])
    with tempfile.TemporaryDirectory(prefix="g7-s10-cold-") as d:
        ck=Path(d)/"weights.pt"
        torch.save(model.state_dict(),ck)
        sha=hashlib.sha256(ck.read_bytes()).hexdigest()
        restored,_=instantiate(torch,official,arm,seed)
        restored.load_state_dict(torch.load(
            ck,map_location="cpu",weights_only=True))
        restored=restored.to(device=device,dtype=torch.float32).eval()
        model.eval()
        with torch.no_grad():
            k,v,q,_=conventional_source.get_batch(
                torch,seed*10000000+9650000,2,[0,1,2,3,3,2,1,0])
            x=prior_transfer.native_tokens(torch,k,v,q).to(device)
            delta=float((model(x)-restored(x)).abs().max().item())
        del restored
    return sha,delta

def freeze_check(spec):
    check(spec.get("schema")=="ttt-hybrid.g7-s11.native-vs-capacity-matched-attention.v1",
          "FROZEN_PROTOCOL_SCHEMA_DRIFT")
    arms=spec["design"]["arms"]
    check([a["id"] for a in arms]==[
        "conventional","full_static","no_mamba_one_attention",
        "attention_replaces_mamba"],
        "ARM_ENUMERATION_DRIFT")
    tr=spec["training"];ev=spec["evaluation"];b=spec["budget"]
    check(tr["seeds"]==[101,211,307] and
          tr["steps"]==512 and tr["batch_size"]==64 and
          ev["batches_per_seed"]==4 and ev["batch_size"]==128 and
          b["max_wall_seconds"]==900 and
          b["torch_peak_bytes_max"]==1073741824 and
          spec["predecessor"]["s10_run"]==42378398424 and
          spec["predecessor"]["conventional_source_blob"]==
          "72ba66eebcebd3856d5251ccc765ea52ddf84500" and
          spec["design"]["parameter_tolerance_fraction"]==0.02,
          "FROZEN_SEEDS_STEPS_OR_BUDGET_DRIFT")
    t=spec["frozen_thresholds"]
    check(t["replacement_relative_to_full_mse_max_for_recovery"]==2.0 and
          t["distinct_mamba_contribution_ratio_min"]==2.0 and
          t["distinct_mamba_contribution_absolute_mse_min"]==0.01,
          "FROZEN_THRESHOLDS_DRIFT")

def run(args):
    phase("READ_FROZEN_PROTOCOL")
    raw=args.protocol.read_bytes()
    frozen=json.loads(raw)
    freeze_check(frozen)
    import torch
    from torch.nn import functional as F
    # Source custody matches the existing qualified native runner.
    phase("PIN_SOURCE_RUNTIME_DRIVER")
    official,manifest=native_source.check_environment(
        torch,{"budget":{"required_free_cuda_bytes_min":
                             frozen["budget"]["min_free_cuda_bytes"]}},args)
    actual_git_blob=custody.git(
        ROOT,"hash-object","scripts/g7_s6_trained_conventional_baseline.py")
    check(actual_git_blob==
          frozen["predecessor"]["conventional_source_blob"],
          "S6_CONVENTIONAL_MODEL_OR_GENERATOR_DRIFT")
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device=torch.device("cuda:0")
    started=time.monotonic()
    order=[0,1,2,3,3,2,1,0]
    per_seed=[]
    max_allocated=0
    for seed in frozen["training"]["seeds"]:
        results={}
        initial_native={}
        for arm in frozen["design"]["arms"]:
            arm_id=arm["id"]
            phase("INITIALIZE_"+arm_id)
            is_native=arm_id!="conventional"
            if is_native:
                model,initsha=instantiate(torch,official,arm,seed)
                initial_native[arm_id]=initsha
                model=model.to(device=device,dtype=torch.float32)
                exact_delta=verify_full_forward_exact(
                    torch,official,model,arm,seed,device)
                torch.cuda.reset_peak_memory_stats(device)
            else:
                torch.manual_seed(seed)
                model=conventional_source.new_model(torch)
                exact_delta=None
            model_parameter_count=sum(p.numel() for p in model.parameters())
            if not is_native:
                check(model_parameter_count==1025,
                      "CONVENTIONAL_MODEL_PARAMETER_DRIFT")
            else:
                check(model_parameter_count in (90259,123795),
                      "NATIVE_MODEL_TOTAL_PARAMETER_DRIFT")
            trainable=[p for p in model.parameters() if p.requires_grad]
            check(len(trainable)>0,"NO_TRAINABLE_PARAMETERS")
            trainable_count=sum(p.numel() for p in trainable)
            if is_native:
                check(trainable_count==arm["expected_trainable"],
                      "ACTIVE_TRAINABLE_PARAMETER_DRIFT")
            optim=torch.optim.AdamW(
                trainable,lr=arm["train_lr"],weight_decay=0.0)
            phase("FIT_"+arm_id)
            model.train()
            arm_started=time.monotonic()
            losses=[]
            for step in range(frozen["training"]["steps"]):
                k,v,q,y=conventional_source.get_batch(
                    torch,seed*10000000+step*17,
                    frozen["training"]["batch_size"],order)
                if is_native:
                    x=prior_transfer.native_tokens(torch,k,v,q).to(device)
                    y=y.to(device)
                    predicted=model(x)[:,8:,:16]
                else:
                    predicted=model(k,v,q)
                check(tuple(predicted.shape)==tuple(y.shape),"FIT_TARGET_SHAPE")
                loss=F.mse_loss(predicted,y)
                check(bool(torch.isfinite(loss)),"NONFINITE_FIT_LOSS")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(trainable,1.0)
                check(bool(torch.isfinite(grad_norm)),"NONFINITE_FIT_GRADIENT")
                optim.step()
                losses.append(float(loss.detach().item()))
                if is_native:
                    torch.cuda.synchronize(device)
                    peak=int(torch.cuda.max_memory_allocated(device))
                    check(peak<=frozen["budget"]["torch_peak_bytes_max"],
                          "NATIVE_VRAM_LIMIT")
                    max_allocated=max(max_allocated,peak)
                check(time.monotonic()-started<
                      frozen["budget"]["max_wall_seconds"],"TIME_BUDGET")
            fit_seconds=time.monotonic()-arm_started
            # New heldout distribution, never used for training.
            phase("UNSEEN_JOINT_PERMUTATION_"+arm_id)
            model.eval()
            metric_keys=[
                "fixed_layout","joint_permuted",
                "joint_corrupted_values",
                "oracle_correct_mse","oracle_corrupted_mse","zero_predictor_mse"
            ]
            batches={metric:[] for metric in metric_keys}
            eval_started=time.monotonic()
            with torch.no_grad():
                for batch in range(frozen["evaluation"]["batches_per_seed"]):
                    epi=seed*10000000+9970000+batch*17
                    k,v,q,y=conventional_source.get_batch(
                        torch,epi,frozen["evaluation"]["batch_size"],
                        order)
                    qp,yp,kp,vp=prior_transfer.permuted_support_queries(
                        torch,k,v,q,y,epi)
                    variants={
                        "fixed_layout":(k,v,q,y),
                        "joint_permuted":(kp,vp,qp,yp),
                        "joint_corrupted_values":
                            (kp,torch.roll(vp,1,dims=1),qp,yp)
                    }
                    for variant,(keys,values,queries,labels) in variants.items():
                        matches=(queries[:,:,None,:]==
                                 keys[:,None,:,:]).all(dim=-1)
                        check(bool((matches.sum(dim=-1)==1).all()),
                              "AMBIGUOUS_HELDOUT_KEYS")
                        oracle=torch.einsum(
                            "bqi,biv->bqv",matches.float(),values)
                        if variant!="joint_corrupted_values":
                            check(bool(torch.equal(oracle,labels)),
                                  "POSITIVE_ORACLE_TARGET_FAIL")
                        if is_native:
                            output=[]
                            for begin in range(0,keys.shape[0],16):
                                x=prior_transfer.native_tokens(
                                    torch,keys[begin:begin+16],
                                    values[begin:begin+16],
                                    queries[begin:begin+16]).to(device)
                                output.append(model(x)[:,8:,:16].cpu())
                            prediction=torch.cat(output,dim=0)
                        else:
                            prediction=model(keys,values,queries)
                        check(bool(torch.isfinite(prediction).all()),
                              "NONFINITE_HELDOUT_OUTPUT")
                        batches[variant].append(
                            float(F.mse_loss(prediction,labels).item()))
                        if variant=="fixed_layout":
                            batches["zero_predictor_mse"].append(
                                float(labels.square().mean().item()))
                        elif variant=="joint_permuted":
                            batches["oracle_correct_mse"].append(
                                float(F.mse_loss(oracle,labels).item()))
                        else:
                            if variant=="joint_corrupted_values":
                                batches["oracle_corrupted_mse"].append(
                                    float(F.mse_loss(oracle,labels).item()))
            if is_native:
                ck_sha,cold_delta=cold_native(
                    torch,model,official,arm,seed,device)
                torch.cuda.synchronize(device)
                arm_peak=int(torch.cuda.max_memory_allocated(device))
                max_allocated=max(max_allocated,arm_peak)
            else:
                ck_sha,cold_delta=prior_transfer.baseline_cold(
                    torch,model,seed,order)
                arm_peak=0
            check(cold_delta==0.0,"COLD_RESTORATION_NOT_EXACT")
            results[arm_id]={
                "initial_sha256":initial_native.get(arm_id),
                "full_original_forward_max_abs_delta":exact_delta,
                "model_parameters_total":model_parameter_count,
                "model_parameters_trainable":trainable_count,
                "optimizer_steps":len(losses),
                "training_episodes":len(losses)*
                    frozen["training"]["batch_size"],
                "first_training_loss":losses[0],
                "last_32_training_loss":statistics.mean(losses[-32:]),
                "fresh_heldout_batch_metrics":batches,
                "fresh_heldout_means":{
                    key:statistics.mean(vals) for key,vals in batches.items()},
                "fit_seconds":round(fit_seconds,3),
                "heldout_restore_seconds":round(time.monotonic()-eval_started,3),
                "peak_torch_allocated_bytes":arm_peak,
                "cold_checkpoint_sha256":ck_sha,
                "cold_output_max_abs_delta":cold_delta,
            }
            del model,optim
            if is_native:
                torch.cuda.empty_cache()
            check(time.monotonic()-started<
                  frozen["budget"]["max_wall_seconds"],"TIME_AFTER_EVALUATION")
        check(len(set(initial_native.values()))==1,
              "NATIVE_ARM_INITIAL_WEIGHTS_NOT_IDENTICAL")
        check(all(item["optimizer_steps"]==512 and
                  item["training_episodes"]==32768
                  for item in results.values()),"UNMATCHED_EXPOSURE")
        per_seed.append({"seed":seed,"arms":results})
    phase("AGGREGATE_FROZEN_ATTENTION_REPLACEMENT")
    means={}
    ids=[a["id"] for a in frozen["design"]["arms"]]
    for arm_id in ids:
        means[arm_id]={
            key:statistics.mean(
                r["arms"][arm_id]["fresh_heldout_means"][key]
                for r in per_seed)
            for key in per_seed[0]["arms"][arm_id]["fresh_heldout_means"]
        }
    t=frozen["frozen_thresholds"]
    oracle_valid=all(
        abs(item["zero_predictor_mse"]-t["zero_mse_exact"])<1e-12 and
        item["oracle_correct_mse"]<=t["oracle_mse_max"] and
        item["oracle_corrupted_mse"]>=t["oracle_corrupt_mse_min"]
        for item in means.values())
    conventional_ok=(
        means["conventional"]["joint_permuted"]<=t["conventional_joint_mse_max"]
        and all(row["arms"]["conventional"]["fresh_heldout_means"]
                ["joint_permuted"]<=t["conventional_joint_mse_max"]
                for row in per_seed))
    full=means["full_static"]["joint_permuted"]
    full_ok=full>0 and full<=t["full_joint_mse_max"]
    replacement=means["attention_replaces_mamba"]["joint_permuted"]
    ratio=replacement/full if full>0 else None
    per_seed_ratios=[]
    per_seed_replacement_worse=[]
    for row in per_seed:
        a=row["arms"]["attention_replaces_mamba"]["fresh_heldout_means"]["joint_permuted"]
        b=row["arms"]["full_static"]["fresh_heldout_means"]["joint_permuted"]
        check(b>0,"ZERO_DENOMINATOR_NATIVE_SEED")
        per_seed_ratios.append(a/b)
        per_seed_replacement_worse.append(a>b)
    corrupted=means["attention_replaces_mamba"]["joint_corrupted_values"]
    damage=(corrupted-replacement)/replacement if replacement>0 else None
    all_controls_valid=oracle_valid and conventional_ok and full_ok
    active_full=per_seed[0]["arms"]["full_static"]["model_parameters_trainable"]
    active_replace=per_seed[0]["arms"]["attention_replaces_mamba"]["model_parameters_trainable"]
    parameter_difference=abs(active_full-active_replace)/active_full
    check(parameter_difference<=t["active_parameter_difference_tolerance"],
          "ACTIVE_PARAM_CAPACITY_DIFF_TOO_LARGE")
    recovery=(all_controls_valid and
              replacement<=t["replacement_joint_mse_max_for_recovered_learning"]
              and ratio<=t["replacement_relative_to_full_mse_max_for_recovery"]
              and all(v<=t["replacement_each_seed_joint_mse_ratio_max"]
                      for v in per_seed_ratios)
              and damage>=0.03)
    contribution=(all_controls_valid and
                  ratio>=t["distinct_mamba_contribution_ratio_min"]
                  and replacement-full>=
                      t["distinct_mamba_contribution_absolute_mse_min"]
                  and all(per_seed_replacement_worse))
    classification=(
        "ASSAY_INVALID_CONVENTIONAL_OR_ORACLE" if not all_controls_valid else
        "ATTENTION_REPLACEMENT_RECOVERS_BINDING_ON_FROZEN_TASK" if recovery else
        "MAMBA_ADVANTAGE_PERSISTS_VS_NEAR_PARAMETER_MATCHED_ATTENTION"
            if contribution else
        "INCONCLUSIVE_CAPACITY_MATCHED_ATTENTION_CONTROL")
    return {
        "schema":"ttt-hybrid.g7-s11.near-parameter-attention-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":classification,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "s6_conventional_generator_blob":actual_git_blob,
        "runtime_manifest_sha256":manifest,
        "seeds":per_seed,
        "aggregate_heldout_means":means,
        "oracle_controls_pass":oracle_valid,
        "conventional_positive_control_pass":conventional_ok,
        "full_static_positive_control_pass":full_ok,
        "replacement_over_full_mse_ratio":ratio,
        "replacement_minus_full_absolute_mse":replacement-full,
        "replacement_relative_mse_damage_under_corruption":damage,
        "replacement_over_full_ratio_each_seed":per_seed_ratios,
        "replacement_worse_each_seed":per_seed_replacement_worse,
        "recovery_thresholds_met":recovery,
        "distinct_mamba_contribution_thresholds_met":contribution,
        "full_native_active_params":active_full,
        "replacement_active_params":active_replace,
        "relative_active_parameter_difference":parameter_difference,
        "equal_exposure_and_optimizer_steps":True,
        "equal_training_flops":False,
        "energy_measured":False,
        "train_steps_per_seed_per_arm":512,
        "training_episodes_per_seed_per_arm":32768,
        "new_heldout_episodes_per_seed_per_arm":512,
        "max_torch_allocated_bytes":max_allocated,
        "elapsed_seconds":round(time.monotonic()-started,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    destination=ROOT/"local_runs/g7_s11_capacity_matched_attention.json"
    report={
        "schema":"ttt-hybrid.g7-s11.near-parameter-attention-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "phase":"NOT_STARTED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    code=2
    try:
        check(args.receipt.resolve()==destination.resolve() and
              args.protocol.resolve()==
              (ROOT/"experiments/g7_s11_frozen_attention_replacement.json").resolve(),
              "RECEIPT_OR_FROZEN_PROTOCOL_PATH_DENIED")
        report=run(args)
        code=(0 if report["status"] in (
              "ATTENTION_REPLACEMENT_RECOVERS_BINDING_ON_FROZEN_TASK",
              "MAMBA_ADVANTAGE_PERSISTS_VS_NEAR_PARAMETER_MATCHED_ATTENTION",
              "INCONCLUSIVE_CAPACITY_MATCHED_ATTENTION_CONTROL") else 2)
    except (Blocked,custody.Blocked,native_source.Blocked) as exc:
        report["status"]="ENGINEERING_BLOCKED_"+str(exc)
        report["phase"]=PHASE
    except Exception as exc:
        report["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        report["phase"]=PHASE
        report["error_class"]=type(exc).__name__
    finally:
        if args.receipt.resolve()==destination.resolve():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                                   encoding="utf-8")
        print("G7_S11_STATUS="+report["status"],flush=True)
        print("G7_S11_PHASE="+report.get("phase","COMPLETE"),flush=True)
        print("G7_S11_INDEPENDENT=NOT_RUN",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
