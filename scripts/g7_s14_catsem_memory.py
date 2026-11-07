"""G7-S14 CATSEM: collision-aware tri-sketch episodic associative memory.

Experimental algorithmic alternative to attention. Uses binary key/event
markers and bounded write/read sketches; no learned parameters and no
token-token attention. Hash associative memory has prior art. This is NOT
a claim of scientific model superiority or general learning.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s6_trained_conventional_baseline as original_dataset
import g7_s9_permutation_transfer as original_permutations

class Blocked(Exception):
    pass

PHASE="NOT_STARTED"
PRIMES=(1021,1031,1033)
SEEDS=(0x9e3779b9,0x85ebca6b,0xc2b2ae35)

def check(ok,label):
    if not ok:
        raise Blocked(label)

def phase(name):
    global PHASE
    PHASE=name

def dpkg_hash():
    fmt="-f=" + chr(36)+"{binary:Package}\t"+chr(36)+"{Status}\t"+chr(36)+"{Version}\n"
    p=subprocess.run(["/usr/bin/dpkg-query","-W",fmt],
                     cwd=ROOT,capture_output=True,text=True,
                     timeout=30,check=False)
    check(p.returncode==0,"DPKG_READ_ONLY_INVENTORY_UNAVAILABLE")
    return hashlib.sha256(p.stdout.encode()).hexdigest()

def freeze_check(p):
    check(p.get("schema")==
          "ttt-hybrid.g7-s14.catsem-attention-free-exploration.v1",
          "FROZEN_SCHEMA_DRIFT")
    q=p["frozen_experiment"];t=p["controls"]
    check(p["lineage"]["s13_run"]==42386864471
          and p["lineage"]["s13_sha"]==
              "d6f6854930dad85f99ee5efbed2ade03facfb2a9"
          and p["lineage"]["source_s6_generator_git_blob"]==
              "72ba66eebcebd3856d5251ccc765ea52ddf84500"
          and q["seeds"]==[137,257,389]
          and q["heldout_batches_per_seed"]==3
          and q["heldout_batch_size"]==64
          and [x["id"] for x in q["cases"]]==[
              "base16","decoy128","decoy512","decoy1024",
              "decoy1024_corrupted_true"]
          and [x["total_tokens"] for x in q["cases"]]==[
              16,128,512,1024,1024]
          and t["tri_sketch_decoy1024_mse_max"]==0.08
          and t["tri_vs_single_decoy1024_relative_error_max"]==0.5
          and t["wall_seconds_cap"]==180,
          "FROZEN_CONTROLS_OR_SAMPLE_DRIFT")
    check(p["mechanism"]["name"].startswith("CATSEM"),
          "FROZEN_MECHANISM_DRIFT")

def make_tokens(torch,k,v,q,gap,decoys,seed):
    from g7_s9_permutation_transfer import native_tokens
    base=native_tokens(torch,k,v,q)
    if gap==0:
        check(not decoys,"UNFROZEN_GAPLESS_DISTRACTOR")
        return base
    check(gap in (112,496,1008) and decoys,
          "UNFROZEN_SEQUENCE_LENGTH_OR_DISTRACTOR")
    b=k.shape[0]
    filler=torch.zeros(b,gap,64,dtype=torch.float32)
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    n=gap//2
    keys=torch.randint(0,2,(b,n,16),generator=gen,
                       dtype=torch.int64).float()*2-1
    vals=torch.randint(0,2,(b,n,16),generator=gen,
                       dtype=torch.int64).float()*2-1
    for bit in range(16):
        collisions=(keys[:,:,None,:]==k[:,None,:,:]).all(-1).any(-1)
        if not bool(collisions.any()):
            break
        keys[:,:,bit]=torch.where(collisions,
                                 -keys[:,:,bit],keys[:,:,bit])
    check(not bool((keys[:,:,None,:]==k[:,None,:,:]).all(-1).any()),
          "DECOY_KEY_MATCHED_GENUINE")
    filler[:,0::2,:16]=keys
    filler[:,1::2,16:32]=vals
    filler[:,0::2,32]=1.
    filler[:,1::2,33]=1.
    sequence=torch.cat((base[:,:8],filler,base[:,8:]),dim=1)
    check(tuple(sequence.shape)==(b,gap+16,64)
          and bool((sequence[:,-8:,16:32]==0).all())
          and bool((sequence[:,-8:,34]==1).all()),
          "QUERY_LEAKAGE_OR_SHAPE_DRIFT")
    return sequence

def extract_events(torch,tokens):
    """Assert and decode strict support key/value writes and final reads."""
    b,t,w=tokens.shape
    check(w==64 and t in (16,128,512,1024),"UNEXPECTED_TOKEN_SHAPE")
    key_mark=tokens[:,:,32]==1.
    val_mark=tokens[:,:,33]==1.
    query_mark=tokens[:,:,34]==1.
    check(bool((key_mark==key_mark[:1]).all())
          and bool((val_mark==val_mark[:1]).all())
          and bool((query_mark==query_mark[:1]).all())
          and int(key_mark[0].sum())==int(val_mark[0].sum())
          and bool(query_mark[:,t-8:].all())
          and not bool(query_mark[:,:t-8].any())
          and not bool(key_mark[:,t-8:].any())
          and not bool(val_mark[:,t-8:].any()),
          "UNEXPECTED_EPISODIC_EVENT_MARKERS")
    indexes=torch.where(key_mark[0])[0]
    check(bool((indexes+1<t-8).all()) and
          bool(val_mark[:,indexes+1].all())
          and bool((indexes%2==0).all())
          and not bool((key_mark&val_mark).any()),
          "KEY_VALUE_WRITE_PAIR_NOT_ADJACENT")
    key=tokens[:,indexes,:16]
    val=tokens[:,indexes+1,16:32]
    query=tokens[:,-8:,:16]
    check(bool(((key.abs()==1).all()) and
               ((val.abs()==1).all()) and
               ((query.abs()==1).all())),
          "UNEXPECTED_BINARY_EVENT_CODE")
    return key,val,query

def sketch_hashes(torch,key):
    """Independent fixed-seed 32-bit avalanching of 16-bit signed key code."""
    check(key.shape[-1]==16,"NOT_16_BIT_KEY")
    weights=(1 << torch.arange(16,dtype=torch.int64))
    x=((key>0).to(dtype=torch.int64)*weights).sum(-1)
    results=[]
    for seed,prime in zip(SEEDS,PRIMES):
        h=(x ^ seed) & 0xffffffff
        h=(h ^ (h >> 16)) & 0xffffffff
        h=(h*0x7feb352d) & 0xffffffff
        h=(h ^ (h >> 15)) & 0xffffffff
        h=(h*0x846ca68b) & 0xffffffff
        h=(h ^ (h >> 16)) & 0xffffffff
        results.append(h % prime)
    return torch.stack(results,dim=-1)

def retrieve(torch,keys,values,queries,streams=3):
    """Write episode-local bounded state; select least-collided read route."""
    b,n,w=keys.shape
    check(w==16 and values.shape==(b,n,16)
          and queries.shape[0]==b
          and queries.shape[2]==16
          and streams in (1,3),"BAD_ASSOCIATIVE_INPUT")
    writes=sketch_hashes(torch,keys)
    reads=sketch_hashes(torch,queries)
    candidate_reads=[]
    candidate_occupancies=[]
    state_bytes=0
    for i in range(streams):
        size=PRIMES[i]
        state=torch.zeros((b,size,16),dtype=torch.float32)
        occupancies=torch.zeros((b,size,1),dtype=torch.float32)
        state.scatter_add_(1,writes[:,:,i,None].expand(-1,-1,16),values)
        occupancies.scatter_add_(1,writes[:,:,i,None],
                                  torch.ones((b,n,1),dtype=torch.float32))
        qindex=reads[:,:,i,None]
        read_value=state.gather(1,qindex.expand(-1,-1,16))
        count=occupancies.gather(1,qindex)
        check(bool((count>0).all()),"QUERY_KEY_NOT_WRITTEN")
        candidate_reads.append(read_value/count)
        candidate_occupancies.append(count.squeeze(-1))
        state_bytes+=(state.numel()+occupancies.numel())*4
    collision=torch.stack(candidate_occupancies,dim=-1)
    route=collision.argmin(dim=-1)
    options=torch.stack(candidate_reads,dim=-2)
    output=options.gather(2,route[:,:,None,None].expand(-1,-1,1,16)).squeeze(2)
    no_collision_rate=float((collision.min(dim=-1).values==1).float().mean().item())
    return output,{"state_bytes":state_bytes,
                   "collision_free_read_fraction":no_collision_rate,
                   "mean_minimum_bucket_occupancy":float(
                       collision.min(dim=-1).values.float().mean().item())}

def check_runtime(torch,args,p):
    phase("READ_ONLY_PINNED_SOURCE_AND_ENVIRONMENT")
    check(os.geteuid()==1000 and
          "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower(),
          "NORMAL_WSL_USER_REQUIRED")
    check(custody.git(ROOT,"rev-parse","HEAD")==args.expected_sha
          and not custody.git(ROOT,"status","--porcelain",
                              "--untracked-files=all"),
          "SOURCE_SHA_OR_TREE_DIRTY")
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s6_trained_conventional_baseline.py")==
          p["lineage"]["source_s6_generator_git_blob"],
          "ORIGINAL_S6_GENERATOR_DRIFT")
    check(custody.git(ROOT,"hash-object",
                      "scripts/g7_s9_permutation_transfer.py")==
          "893355fb7eac22dc16053db9320a1dac4130ca9f",
          "SOURCE_S9_PERMUTATION_DRIFT")
    check(custody.sha256(ROOT/"source_pins.json")==
          custody.R3_PINS_SHA256,
          "SOURCE_PIN_DRIFT")
    manifest,packages=custody.manifest()
    check(manifest==custody.R3_MANIFEST_SHA256
          and packages.get("torch")==custody.TORCH
          and packages.get("triton")==custody.TRITON
          and torch.__version__==custody.TORCH
          and sys.version_info[:2]==(3,12),
          "PINNED_PYTHON_RUNTIME_DRIFT")
    check(not torch.cuda.is_initialized(),"CUDA_USED_BY_CPU_ONLY_EXPERIMENT")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return manifest

def execute(args):
    import torch
    from torch.nn import functional as F
    raw=args.protocol.read_bytes()
    p=json.loads(raw)
    freeze_check(p)
    manifest=check_runtime(torch,args,p)
    start=time.monotonic()
    data=p["frozen_experiment"]
    all_seeds=[]
    max_state_bytes=0
    for seed in data["seeds"]:
        phase("CPU_HELDOUT_SEED_"+str(seed))
        lists={c["id"]:[] for c in data["cases"]}
        mono=[]
        metrics={c["id"]:[] for c in data["cases"]}
        oracle_correct=[]
        oracle_bad=[]
        zero=[]
        erased=[]
        reset=[]
        causal=[]
        for batch in range(data["heldout_batches_per_seed"]):
            es=seed*10000000+9910000+batch*17
            k,v,q,y=original_dataset.get_batch(
                torch,es,data["heldout_batch_size"],
                data["query_order"])
            qp,yp,kp,vp=original_permutations.permuted_support_queries(
                torch,k,v,q,y,es)
            match=(qp[:,:,None,:]==kp[:,None,:,:]).all(-1)
            check(bool((match.sum(-1)==1).all()),
                  "POSITIVE_LOOKUP_NOT_UNIQUE")
            exact=torch.einsum("bqi,biv->bqv",match.float(),vp)
            corrupted_oracle=torch.einsum(
                "bqi,biv->bqv",match.float(),torch.roll(vp,1,dims=1))
            check(bool(torch.equal(exact,yp)),
                  "POSITIVE_ORACLE_NOT_EXACT")
            oracle_correct.append(float(F.mse_loss(exact,yp).item()))
            oracle_bad.append(float(F.mse_loss(corrupted_oracle,yp).item()))
            zero.append(float(yp.square().mean().item()))
            erased.append(float(yp.square().mean().item()))
            for case in data["cases"]:
                label=case["id"]
                tokens=make_tokens(
                    torch,kp,
                    torch.roll(vp,1,dims=1)
                    if case.get("corrupt_original_values") else vp,
                    qp,case["total_tokens"]-16,
                    case["decoys"],es+451231)
                keys,values,queries=extract_events(torch,tokens)
                prediction,info=retrieve(torch,keys,values,queries,streams=3)
                check(bool(torch.isfinite(prediction).all()),
                      "NONFINITE_SKETCH_RETRIEVAL")
                lists[label].append(float(F.mse_loss(prediction,yp).item()))
                metrics[label].append(info)
                max_state_bytes=max(max_state_bytes,info["state_bytes"])
                if label=="decoy1024":
                    one,one_info=retrieve(
                        torch,keys,values,queries,streams=1)
                    mono.append(float(F.mse_loss(one,yp).item()))
                    # Changing *future query keys* must not affect an
                    # earlier query answer (no query writes or suffix reads).
                    changed=tokens.clone()
                    # Replace final query with a DIFFERENT KNOWN key, so
                    # the independent, earlier queries remain valid and
                    # cannot be contaminated by a future read.
                    changed[:,-1,:16]=keys[:,0,:]
                    _,_,qchanged=extract_events(torch,changed)
                    changed_out,_=retrieve(
                        torch,keys,values,qchanged,streams=3)
                    causal.append(float((
                        prediction[:,:-1]-changed_out[:,:-1]
                    ).abs().max().item()))
                    # Full state reinitialized for every call/episode;
                    # repeated identical retrieval must be bitwise exact.
                    again,_=retrieve(
                        torch,keys,values,queries,streams=3)
                    reset.append(float((
                        prediction-again).abs().max().item()))
                check(time.monotonic()-start < p["controls"]["wall_seconds_cap"],
                      "TIME_BUDGET_EXCEEDED")
                check(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss*1024 <=
                    p["controls"]["cpu_peak_rss_bytes_cap"],
                    "CPU_RSS_BUDGET_EXCEEDED")
        mean={k:statistics.mean(v) for k,v in lists.items()}
        one_mean=statistics.mean(mono)
        check(one_mean>0,"ONE_SKETCH_ZERO_MSE_DENOMINATOR")
        all_seeds.append({
            "seed":seed,
            "episodes":data["heldout_batches_per_seed"]*
                       data["heldout_batch_size"],
            "tri_sketch_heldout_mse":mean,
            "single_sketch_decoy1024_mse":one_mean,
            "tri_over_single_decoy1024_mse_ratio":
                mean["decoy1024"]/one_mean,
            "corrupted_true_support_oracle_mse":
                statistics.mean(oracle_bad),
            "exact_oracle_mse":statistics.mean(oracle_correct),
            "zero_predictor_mse":statistics.mean(zero),
            "erased_writer_mse":statistics.mean(erased),
            "suffix_query_causality_max_delta":max(causal),
            "episode_memory_reset_max_delta":max(reset),
            "collision_free_read_fraction_by_case":{
                case:statistics.mean(
                    x["collision_free_read_fraction"]
                    for x in entries)
                for case,entries in metrics.items()},
            "state_bytes_max_by_case":{
                case:max(x["state_bytes"] for x in entries)
                for case,entries in metrics.items()},
        })
    t=p["controls"]
    tests=[]
    validity=[]
    for entry in all_seeds:
        m=entry["tri_sketch_heldout_mse"]
        valid=(
            entry["exact_oracle_mse"]<=t["positive_oracle_correct_mse_max"]
            and entry["corrupted_true_support_oracle_mse"]>=
                t["corrupted_oracle_mse_min"]
            and abs(entry["zero_predictor_mse"]-
                    t["zero_predictor_mse_exact"])<=1e-12
            and abs(entry["erased_writer_mse"]-
                    t["erasable_writer_zero_mse_exact"])<=1e-12
        )
        validity.append(valid)
        ok=(valid
            and m["decoy128"]<=t["tri_sketch_decoy128_mse_max"]
            and m["decoy512"]<=t["tri_sketch_decoy512_mse_max"]
            and m["decoy1024"]<=t["tri_sketch_decoy1024_mse_max"]
            and entry["single_sketch_decoy1024_mse"]>=
                t["single_sketch_decoy1024_mse_min"]
            and entry["tri_over_single_decoy1024_mse_ratio"]<=
                t["tri_vs_single_decoy1024_relative_error_max"]
            and m["decoy1024_corrupted_true"]>=
                t["tri_sketch_decoy1024_corrupted_true_mse_min"]
            and entry["suffix_query_causality_max_delta"]<=
                t["prefix_causality_output_delta_max"]
            and entry["episode_memory_reset_max_delta"]<=
                t["cross_episode_reset_max_delta"])
        tests.append(ok)
    check(max_state_bytes<=t["state_footprint_including_count_bytes_cap"],
          "STATE_MEMORY_BUDGET_EXCEEDED")
    check(not torch.cuda.is_initialized(),
          "CPU_ONLY_EXPERIMENT_TOUCHED_CUDA")
    status=(
        "ASSAY_INVALID_OR_ENVIRONMENT_BLOCKED" if not all(validity) else
        "ATTENTION_FREE_EPISODIC_SKETCH_RETRIEVAL_POSITIVE_ENGINEERING_EXPLORATION"
        if all(tests) else
        "ATTENTION_FREE_SKETCH_RETRIEVAL_FROZEN_LIMITS_NOT_MET"
    )
    mean_mse={c["id"]:statistics.mean(
        x["tri_sketch_heldout_mse"][c["id"]] for x in all_seeds)
        for c in data["cases"]}
    return {
        "schema":"ttt-hybrid.g7-s14.catsem-evidence.v1",
        "status":status,
        "candidate_sha":args.expected_sha,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "pinned_package_manifest_sha256":manifest,
        "algorithm":"CATSEM three independent 32bit hash sketches, occupancy-ranked",
        "does_pairwise_softmax_attention":False,
        "does_mamba_or_ttt_or_cuda_execution":False,
        "learned_parameters":0,
        "no_training":True,
        "marker_and_binary_key_task_prior":True,
        "per_seed":all_seeds,
        "aggregate_tri_sketch_mse_by_case":mean_mse,
        "aggregate_one_sketch_decoy1024_mse":statistics.mean(
            x["single_sketch_decoy1024_mse"] for x in all_seeds),
        "all_oracle_causality_reset_controls_valid":all(validity),
        "all_oracle_causality_reset_controls_valid_per_seed":validity,
        "all_frozen_limits_met_per_seed":tests,
        "all_frozen_limits_met":all(tests),
        "heldout_episodes_per_seed":
            data["heldout_batches_per_seed"]*data["heldout_batch_size"],
        "peak_allocated_sketch_state_bytes":max_state_bytes,
        "process_peak_rss_bytes":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "wall_seconds":round(time.monotonic()-start,3),
        "explicit_intelligence_or_learning_claim":"NONE",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    destination=ROOT/"local_runs/g7_s14_catsem_evidence.json"
    result={
        "schema":"ttt-hybrid.g7-s14.catsem-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "phase":PHASE,
        "deployment":"NOT_AUTHORIZED",
    }
    code=2
    before=None
    try:
        check(args.receipt.resolve()==destination.resolve() and
              args.protocol.resolve()==
              (ROOT/"experiments/g7_s14_frozen_trisketch.json").resolve(),
              "FROZEN_PROTOCOL_OR_RECEIPT_PATH_DENIED")
        phase("READ_ONLY_DPKG_BEFORE")
        before=dpkg_hash()
        result=execute(args)
        code=0 if result["status"]!="ASSAY_INVALID_OR_ENVIRONMENT_BLOCKED" else 2
    except (Blocked,custody.Blocked) as exc:
        result["status"]="ENGINEERING_BLOCKED_"+str(exc)
        result["phase"]=PHASE
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["phase"]=PHASE
        result["error_type"]=type(exc).__name__
    finally:
        if before is not None:
            try:
                after=dpkg_hash()
                result["dpkg_before_sha256"]=before
                result["dpkg_after_sha256"]=after
                result["installed_packages_changed"]=(after!=before)
                if after!=before:
                    result["status"]="ENGINEERING_BLOCKED_PACKAGE_INVENTORY_DRIFT"
                    code=2
            except Exception:
                result["status"]="ENGINEERING_BLOCKED_POST_DPKG_INVENTORY"
                code=2
        if args.receipt.resolve()==destination.resolve():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(
                json.dumps(result,sort_keys=True,indent=2)+"\n",
                encoding="utf-8")
        print("G7_S14_STATUS="+result["status"],flush=True)
        print("G7_S14_GPU=NOT_RUN",flush=True)
        print("G7_S14_DPKG="+str(result.get(
            "installed_packages_changed","UNVERIFIED")),flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
