"""Prospectively frozen P104: fresh-source replication of immutable P101 active acquisition.

New SHA256 world generator (never P101 original FIT/CAL), strongest FIT passive
and conventional A-opt control. Developer-only, no protected first-use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86
from experiments import p101_information_interventions_dev as p101

FREEZE = "4b16585d437bc3903bdb28c4188e065679681a38"
FREEZE_BLOB = "3fcc57f26809f92d058ddcaface599c7002fb56c"
D, TASKS, STEPS, CHOICES, PROBES = 16, 2, 8, 8, 12
FAMILIES = ("isotropic", "anisotropic", "correlated")
SEEDS = {
    "isotropic": {"FIT": (1040111,1040113,1040119),
                  "CAL": (1040127,1040131,1040133)},
    "anisotropic": {"FIT": (1040211,1040217,1040223),
                    "CAL": (1040227,1040229,1040233)},
    "correlated": {"FIT": (1040311,1040317,1040323),
                   "CAL": (1040329,1040331,1040337)},
}
COUNTS = {"FIT":18,"CAL":12}
PASSIVES = ("hash_random","cyclic","max_norm")
CONVENTIONAL = PASSIVES + ("a_opt",)
POLICIES = ("max_variance",) + CONVENTIONAL
PINNED = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p101_information_interventions_dev.py":
        "a4375102e64492f179826b8775855b63b1c315fd",
}


def digest(seed, wid, tag, i, j, mod):
    key=f"p104-validation-v1|{seed}|{wid}|{tag}|{i}|{j}".encode("ascii")
    return int.from_bytes(hashlib.sha256(key).digest()[:4],"little") % mod


def frozen():
    root=Path(__file__).resolve().parents[1]
    path=root/"experiments/p104_independent_active_design_freeze.json"
    if p86.blob(path)!=FREEZE_BLOB:
        raise ValueError("P104_FROZEN_BLOB_DRIFT")
    spec=json.loads(path.read_text(encoding="utf-8"))
    expected={f:{s:list(v) for s,v in splits.items()} for f,splits in SEEDS.items()}
    expected["OOD_RESERVED"]=[1040401,1040403]
    expected["NEGATIVE_RESERVED"]=[1040501,1040503]
    if (spec["schema"]!="ttt-hybrid.p104-fresh-active-design-replication.v1"
        or spec["status"]!="FROZEN_PROSPECTIVE_BEFORE_NEW_DATA_AND_IMPLEMENTATION"
        or spec["identity"]["parent_sha"]!="fcf260b966e8e6dd12a2dc1594103f5a5cc44725"
        or spec["source"]["seeds"]!=expected
        or spec["source"]["worlds_per_seed"]!=COUNTS
        or spec["source"]["interventions_per_task"]!=STEPS
        or spec["source"]["choices_per_intervention"]!=CHOICES
        or spec["source"]["novel_queries_per_task"]!=PROBES
        or spec["acceptance"]["wrong_action_credit_MSE_ratio_min"]!=1.05
        or spec["acceptance"]["whole_process_CPU_candidate_ratio_vs_FIT_passive_max"]!=3.
        or spec["acceptance"]["whole_process_CPU_candidate_ratio_vs_Aopt_max"]!=1.5):
        raise ValueError("P104_PROTOCOL_DRIFT")
    for rel,blob in PINNED.items():
        if p86.blob(root/rel)!=blob:
            raise ValueError("P104_ORIGINAL_BLOB_DRIFT")
    return spec


def rank4(seed,wid,tag):
    return torch.tensor([[digest(seed,wid,tag,j,k,7)-3 for k in range(4)]
                         for j in range(D)],dtype=torch.float64)


def raw_action(seed,wid,family,tag,n):
    common=digest(seed,wid,tag+"-shared",n,0,193)-96
    a=[]
    for j in range(D):
        z=digest(seed,wid,tag,n,j,193)-96
        if family=="anisotropic":
            z=z//(2**(j%4))
        elif family=="correlated" and j%4==0:
            z+=2*common
        a.append(z/128.)
    return a


def build_world(family,split,index,wid):
    if (family not in FAMILIES or split not in COUNTS
        or type(index) is not int or index not in (0,1,2)
        or type(wid) is not int or not 0<=wid<COUNTS[split]):
        raise ValueError("P104_PROTECTED_OR_UNKNOWN_SOURCE_DENIED")
    seed=SEEDS[family][split][index]
    maps,actions,queries=[],[],[]
    for task in range(TASKS):
        u=rank4(seed,wid,f"p104-u{task}")
        v=rank4(seed,wid,f"p104-v{task}")
        maps.append((v@u.T)/256.)
        actions.append(torch.tensor([
            [raw_action(seed,wid,family,f"p104-action-{task}",t*CHOICES+opt)
             for opt in range(CHOICES)] for t in range(STEPS)],dtype=torch.float64))
        queries.append([raw_action(seed,wid,family,f"p104-probe-{task}",q)
                        for q in range(PROBES)])
    raw=torch.stack(actions)
    query=torch.tensor(queries,dtype=torch.float64)
    if (not bool((raw.norm(dim=-1)>1e-8).all())
        or not bool((query.norm(dim=-1)>1e-8).all())):
        raise ValueError("P104_ZERO_SOURCE_ACTION")
    return p101.World(tuple(maps),raw,F.normalize(raw,dim=-1),
                      F.normalize(query,dim=-1))


def source_hash(family,split,index):
    h=hashlib.sha256()
    for wid in range(COUNTS[split]):
        w=build_world(family,split,index,wid)
        for z in (*w.maps,w.raw,w.pool,w.queries):
            h.update(z.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def public_covariance(family):
    if family=="isotropic":
        return torch.eye(D,dtype=torch.float64)
    if family=="anisotropic":
        return torch.diag(torch.tensor([2.**(-2*(j%4)) for j in range(D)],
                                       dtype=torch.float64))
    if family=="correlated":
        c=torch.tensor([float(j%4==0) for j in range(D)],dtype=torch.float64)
        return torch.eye(D,dtype=torch.float64)+4*torch.outer(c,c)
    raise ValueError("P104_UNQUALIFIED_COVARIANCE")


def aoptimal_choice(state,pool,family):
    if (pool.shape!=(CHOICES,D) or pool.dtype!=torch.float64
        or not bool(torch.isfinite(pool).all())):
        raise ValueError("P104_AOPT_BAD_POOL")
    p=state.precision
    g=pool@p.T
    sigma=public_covariance(family)
    reduction=((g@sigma)*g).sum(-1) / (1.+(g*pool).sum(-1))
    return int(torch.argmax(reduction))


def aoptimal_world(world,seed,wid,family):
    states,metrics,journal=[],[],[]
    task0_snapshot=None
    task0_before=None
    for task in range(TASKS):
        state=p101.initial()
        for t in range(STEPS):
            choices=world.pool[task,t]
            index=aoptimal_choice(state,choices,family)
            chosen=choices[index].detach().clone()
            journal.append((task,t,index))
            # True effect born ONLY after candidate is committed.
            effect=(world.maps[task]@chosen).detach().clone()
            state=p101.confirmed_write(state,chosen,effect)
        states.append(p101.RLS(state.weights.detach().clone(),
                               state.precision.detach().clone()))
        if task==0:
            task0_snapshot=p101.RLS(states[0].weights.clone(),states[0].precision.clone())
            task0_before=world.queries[0]@task0_snapshot.weights.T
        if task==1 and (
            not torch.equal(world.queries[0]@task0_snapshot.weights.T,task0_before)
            or not torch.equal(states[0].weights,task0_snapshot.weights)
            or not torch.equal(states[0].precision,task0_snapshot.precision)):
            raise ValueError("P104_AOPT_COLD_DRIFT")
        target=world.queries[task]@world.maps[task].T
        prediction=world.queries[task]@state.weights.T
        metrics.append(float((target-prediction).square().mean()))
    if len(journal)!=TASKS*STEPS:
        raise ValueError("P104_AOPT_EFFECT_COUNT")
    return {
        "mse_all":statistics.mean(metrics),"mse_task0":metrics[0],
        "mse_task1":metrics[1],"cold_task0_bitwise":True,
        "n_confirmed_effects":len(journal),"persistent_floats":1024,
        "selected_indices_sha256":hashlib.sha256(
            json.dumps(journal,separators=(",",":")).encode()).hexdigest(),
        "wrong_credit_mse":None,
    }


def one_world(world,policy,seed,wid,family,wrong_credit=False):
    if policy=="a_opt":
        return aoptimal_world(world,seed,wid,family)
    # Unmodified P101 chooser/online RLS and its negative-control implementation.
    return p101.run_world(world,policy,seed,wid,wrong_credit=wrong_credit)


def evaluate(family,split,index,policy,wrong_credit=False,count=None):
    if (family not in FAMILIES or split not in COUNTS
        or type(index) is not int or index not in (0,1,2)
        or policy not in POLICIES):
        raise ValueError("P104_PROTECTED_EVALUATION")
    n=COUNTS[split] if count is None else count
    if type(n) is not int or not 1<=n<=COUNTS[split]:
        raise ValueError("P104_NONFROZEN_WORLD_SCOPE")
    seed=SEEDS[family][split][index]
    rows=[one_world(build_world(family,split,index,w),policy,seed,w,family,
                    wrong_credit=wrong_credit) for w in range(n)]
    out={
        k:statistics.mean(row[k] for row in rows)
        for k in ("mse_all","mse_task0","mse_task1")
    }
    out.update({
        "world_count":n,"total_confirmed_outcomes":
            sum(row["n_confirmed_effects"] for row in rows),
        "persistent_floats":1024,
        "cold_bitwise":all(row["cold_task0_bitwise"] for row in rows),
        "action_journal_sha256":hashlib.sha256("".join(
            row["selected_indices_sha256"] for row in rows).encode()).hexdigest(),
        "wrong_credit_mse":statistics.mean(row["wrong_credit_mse"]
            for row in rows) if wrong_credit else None,
    })
    return out


def bench(family,index,policy):
    seed=SEEDS[family]["CAL"][index]
    worlds=[build_world(family,"CAL",index,w) for w in range(8)]
    def work():
        return [one_world(worlds[w],policy,seed,w,family) for w in range(8)]
    with torch.no_grad():
        for _ in range(2): work()
        timings=[]
        for _ in range(7):
            start=time.perf_counter();work()
            timings.append((time.perf_counter()-start)*1000)
    return statistics.median(timings)


def study():
    frozen()  # fail closed before new synthetic source birth
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    began=time.perf_counter()
    receipt={
        "schema":"ttt-hybrid.p104-independent-active-design-dev-result.v1",
        "status":"UNVERIFIED","freeze_commit":FREEZE,
        "freeze_blob":FREEZE_BLOB,"scientific_pass":False,
        "author_synthetic_only":True,"external_source_used":False,
        "reserved_OOD_generated":False,"reserved_negative_generated":False,
        "original_model_modified":False,"native_hybrid_trained":False,
        "source_sha256":{},"families":{},
    }
    for fam in FAMILIES:
        for split in COUNTS:
            for index in range(3):
                receipt["source_sha256"][f"{fam}_{split}_{index}"]=source_hash(fam,split,index)
    if len(set(receipt["source_sha256"].values()))!=18:
        raise ValueError("P104_SOURCE_HASH_COLLISION")
    with torch.no_grad():
        for family in FAMILIES:
            family_rows=[]
            for index in range(3):
                fit={p:evaluate(family,"FIT",index,p) for p in POLICIES}
                selected_passive=min(PASSIVES,key=lambda k:fit[k]["mse_all"])
                selected_classical=min(CONVENTIONAL,key=lambda k:fit[k]["mse_all"])
                cal={p:evaluate(family,"CAL",index,p,
                        wrong_credit=(p=="max_variance")) for p in POLICIES}
                active=cal["max_variance"]
                ratios={name:{
                    k:active["mse_"+k]/max(cal[baseline]["mse_"+k],1e-18)
                    for k in ("all","task0","task1")}
                    for name,baseline in (
                        ("FIT_passive",selected_passive),
                        ("FIT_classical",selected_classical),
                        ("A_opt","a_opt"))}
                negative=active["wrong_credit_mse"]/max(active["mse_all"],1e-18)
                if (negative<1.05
                    or not all(row["cold_bitwise"] for row in cal.values())
                    or not all(row["total_confirmed_outcomes"]==COUNTS["CAL"]*TASKS*STEPS
                               and row["persistent_floats"]==1024
                               for row in cal.values())):
                    raise ValueError("P104_REQUIRED_NEGATIVE_OR_PARITY_FAILED")
                cpu={p:bench(family,index,p) for p in
                     dict.fromkeys(("max_variance",selected_passive,"a_opt"))}
                cpu_passive=cpu["max_variance"]/max(cpu[selected_passive],1e-12)
                cpu_aopt=cpu["max_variance"]/max(cpu["a_opt"],1e-12)
                guards={
                    "passive_all":ratios["FIT_passive"]["all"]<=.85,
                    "passive_task0":ratios["FIT_passive"]["task0"]<=.95,
                    "passive_task1":ratios["FIT_passive"]["task1"]<=.95,
                    "strong_classical":ratios["FIT_classical"]["all"]<=1.,
                    "cpu_passive":cpu_passive<=3.,
                    "cpu_a_opt":cpu_aopt<=1.5,
                    "wrong_credit":negative>=1.05,
                }
                replicate=all(guards[k] for k in (
                    "passive_all","passive_task0","passive_task1",
                    "cpu_passive","cpu_a_opt","wrong_credit"))
                family_rows.append({
                    "index":index,"FIT_all_policies":fit,
                    "FIT_selected_passive":selected_passive,
                    "FIT_selected_classical":selected_classical,
                    "CAL_all_policies":cal,"ratios":ratios,
                    "wrong_credit_ratio":negative,
                    "cpu_median_ms":cpu,
                    "cpu_ratio_vs_passive":cpu_passive,
                    "cpu_ratio_vs_a_opt":cpu_aopt,
                    "gates":guards,"replicates_P101":replicate,
                    "beats_strong_classical":replicate and guards["strong_classical"],
                })
            md_passive=statistics.median(
                x["ratios"]["FIT_passive"]["all"] for x in family_rows)
            md_classical=statistics.median(
                x["ratios"]["FIT_classical"]["all"] for x in family_rows)
            replicated=sum(x["replicates_P101"] for x in family_rows)>=2 and md_passive<=.85
            robust=replicated and (
                sum(x["beats_strong_classical"] for x in family_rows)>=2
                and md_classical<=1.)
            receipt["families"][family]={
                "seeds":family_rows,"median_ratio_vs_passive":md_passive,
                "median_ratio_vs_classical":md_classical,
                "replication_family_pass":replicated,
                "robust_classical_family_pass":robust,
            }
    b=receipt["families"]["anisotropic"]
    receipt["status"]=(
        "DEV_ANISOTROPIC_REPLICATION_PLUS_CLASSICAL_PASS_NOT_SCIENTIFIC"
        if b["robust_classical_family_pass"] else
        "DEV_ANISOTROPIC_PASSIVE_REPLICATION_ONLY_NOT_CLASSICAL"
        if b["replication_family_pass"] else
        "DEV_FAIL_ANISOTROPIC_P101_REPLICATION_OR_COST")
    receipt["process_wall_seconds"]=time.perf_counter()-began
    return receipt


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    r=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":r["status"],"scientific_pass":False,
                      "family_replication":{f:d["replication_family_pass"]
                                           for f,d in r["families"].items()}}))


if __name__=="__main__":
    main()
