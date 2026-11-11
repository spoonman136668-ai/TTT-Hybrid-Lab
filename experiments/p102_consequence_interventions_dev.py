"""P102: frozen confirmed-consequence-conditioned intervention acquisition.

P101's data-acquisition baseline is kept intact; P102 policy sees only its own
confirmed RLS state and the current equal action pool. DEV synthetic only.
Freeze-before-new-source: 16dab90c352af558ae1db152eaf292065652745d.
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

D, STEPS, CHOICES, PROBES = 16, 8, 8, 12
FREEZE = "16dab90c352af558ae1db152eaf292065652745d"
FREEZE_BLOB = "4f50b8bcfbc6429a675765b89a3b33abbc56d1dc"
FAMILIES = ("isotropic", "mildly_anisotropic", "anisotropic")
SEEDS = {
    "isotropic": {"FIT": (1020111, 1020113, 1020119),
                  "CAL": (1020127, 1020131, 1020133)},
    "mildly_anisotropic": {"FIT": (1020311, 1020317, 1020323),
                           "CAL": (1020329, 1020331, 1020337)},
    "anisotropic": {"FIT": (1020211, 1020217, 1020223),
                    "CAL": (1020227, 1020229, 1020233)},
}
EXPONENTS = {"isotropic": (0, 0, 0, 0),
             "mildly_anisotropic": (0, 0, 1, 1),
             "anisotropic": (0, 1, 2, 3)}
COUNT = {"FIT": 24, "CAL": 12}
BETAS = (.25, 1., 4.)
PASSIVE = ("hash_random", "cyclic", "max_norm")
ARMS = ("impact", "geometry", "hash_random", "cyclic", "max_norm",
        "impact_misaligned")
PARENTS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p101_information_interventions_dev.py":
        "a4375102e64492f179826b8775855b63b1c315fd",
}


def frozen():
    root = Path(__file__).resolve().parents[1]
    file = root / "experiments/p102_consequence_interventions_frozen.json"
    if p86.blob(file) != FREEZE_BLOB:
        raise ValueError("P102_FROZEN_BLOB_DRIFT")
    doc = json.loads(file.read_text(encoding="utf-8"))
    canonical = {f:{k:list(v) for k,v in splits.items()}
                 for f,splits in SEEDS.items()}
    canonical["OOD_RESERVED"] = [1020401, 1020403]
    canonical["NEGATIVE_RESERVED"] = [1020501, 1020503]
    if (doc["schema"] !=
        "ttt-hybrid.p102-consequence-conditioned-interventions-dev-freeze.v1"
        or doc["identity"]["parent_sha"] !=
           "fcf260b966e8e6dd12a2dc1594103f5a5cc44725"
        or doc["source"]["seeds"] != canonical
        or doc["parameter_selection"]["beta_grid"] != list(BETAS)
        or doc["source"]["FIT_worlds_per_seed"] != COUNT["FIT"]
        or doc["source"]["CAL_worlds_per_seed"] != COUNT["CAL"]
        or doc["source"]["candidate_actions_per_opportunity"] != CHOICES
        or doc["source"]["intervention_opportunities_per_task"] != STEPS
        or doc["source"]["new_heldout_readonly_query_actions_per_task"] != PROBES
        or doc["learner"]["shared"].find("prior=0.5") == -1
        or doc["acceptance"]["full_episode_cpu_candidate_ratio_vs_geometric_max"] != 1.6
        or doc["acceptance"]["wrong_action_credit_MSE_ratio_min"] != 1.05):
        raise ValueError("P102_PROTOCOL_DRIFT")
    for path, blob in PARENTS.items():
        if p86.blob(root/path) != blob:
            raise ValueError("P102_ORIGINAL_BLOB_DRIFT")
    return doc


def build_world(family: str, split: str, index: int, wid: int) -> p101.World:
    if (family not in FAMILIES or split not in COUNT
        or type(index) is not int or not 0 <= index < 3
        or type(wid) is not int or not 0 <= wid < COUNT[split]):
        raise ValueError("P102_PROTECTED_OR_UNKNOWN_SOURCE_DENIED")
    seed = SEEDS[family][split][index]
    exp = EXPONENTS[family]
    maps, pool, probes = [], [], []
    for task in range(2):
        u = torch.tensor(p86.basis(seed, wid, "p102-u"+str(task)),
                         dtype=torch.float64)
        v = torch.tensor(p86.basis(seed, wid, "p102-v"+str(task)),
                         dtype=torch.float64)
        maps.append(v@u.T/256.)
        actions=[]
        for t in range(STEPS):
            options=[]
            for opt in range(CHOICES):
                coords=[]
                for j in range(D):
                    z=p86.digest_int(seed,wid,"p102-action-"+str(task),
                                      t*CHOICES+opt,j,193)-96
                    coords.append((z//(2**exp[j%4]))/128.)
                options.append(coords)
            actions.append(options)
        raw=torch.tensor(actions,dtype=torch.float64)
        if not bool((raw.norm(dim=-1)>1e-9).all()):
            raise ValueError("P102_INVALID_ACTION")
        pool.append(raw)
        qa=[]
        for t in range(PROBES):
            coords=[]
            for j in range(D):
                z=p86.digest_int(seed,wid,"p102-probe-"+str(task),t,j,193)-96
                coords.append((z//(2**exp[j%4]))/128.)
            qa.append(coords)
        probes.append(qa)
    raw_tensor=torch.stack(pool)
    return p101.World(tuple(maps), raw_tensor,
                      F.normalize(raw_tensor,dim=-1),
                      F.normalize(torch.tensor(probes,dtype=torch.float64),dim=-1))


def source_sha(family: str, split: str, index: int):
    h=hashlib.sha256()
    for i in range(COUNT[split]):
        w=build_world(family,split,index,i)
        for z in (*w.maps,w.raw,w.pool,w.queries):
            h.update(z.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def choice(arm: str, state: p101.RLS, options: torch.Tensor,
           raw: torch.Tensor, seed: int, wid: int, task: int,
           t: int, beta: float=1.):
    """A chooser cannot receive hidden maps, unchosen outcomes or queries."""
    if (arm not in ARMS or options.shape!=(CHOICES,D)
        or raw.shape!=options.shape or beta not in BETAS
        or options.dtype!=torch.float64 or raw.dtype!=torch.float64
        or not bool(torch.isfinite(options).all())
        or not bool(torch.isfinite(raw).all())):
        raise ValueError("P102_INVALID_CHOICE")
    if arm=="hash_random":
        return int(p86.digest_int(seed,wid,"p102-random",task*STEPS+t,0,CHOICES))
    if arm=="cyclic":
        return t%CHOICES
    if arm=="max_norm":
        return int(torch.argmax(raw.square().sum(-1)))
    variance=((options@state.precision)*options).sum(-1)
    if arm=="geometry":
        return int(torch.argmax(variance))
    # Only effects of THIS chooser's previously confirmed actions set W.
    W=state.weights
    if arm=="impact_misaligned":
        W=W[:,torch.arange(D-1,-1,-1)]
    impact=(options@W.T).square().sum(dim=-1)
    score=variance*(1.+beta*impact/(impact.mean()+1e-9))
    return int(torch.argmax(score))


def run_world(world: p101.World, arm: str, seed: int, wid: int,
              beta: float=1., wrong_credit: bool=False):
    states, ledger, journal=[],[],[]
    before_cold=None
    task0_snapshot=None
    task0_first_read=None
    for task in range(2):
        s=p101.initial()
        history=[]
        for t in range(STEPS):
            opts,raw=world.pool[task,t],world.raw[task,t]
            idx=choice(arm,s,opts,raw,seed,wid,task,t,beta)
            action=opts[idx].detach().clone()
            # COMMIT the action before the hidden environment produces
            # the sole admissible outcome of this opportunity.
            ledger.append((task,t,idx))
            consequence=(world.maps[task]@action).detach().clone()
            history.append((action,consequence))
            s=p101.confirmed_write(s,action,consequence)
        states.append(p101.RLS(s.weights.detach().clone(),
                              s.precision.detach().clone()))
        journal.append(history)
        if task==0:
            task0_snapshot=p101.RLS(s.weights.detach().clone(),
                                   s.precision.detach().clone())
            task0_first_read=world.queries[0]@s.weights.T
    errors=[]
    zero_errors=[]
    for task in range(2):
        q=world.queries[task]
        truth=q@world.maps[task].T  # evaluator ONLY, after factual writes
        predicted=q@states[task].weights.T
        errors.append(float((predicted-truth).square().mean()))
        zero_errors.append(float(truth.square().mean()))
    cold=world.queries[0]@states[0].weights.T
    restored=world.queries[0]@task0_snapshot.weights.T
    if not (torch.equal(task0_first_read,cold)
            and torch.equal(restored,cold)
            and torch.equal(states[0].weights,task0_snapshot.weights)
            and torch.equal(states[0].precision,task0_snapshot.precision)):
        raise ValueError("P102_COLD_STATE_DRIFT")
    wrong=None
    if wrong_credit:
        scores=[]
        for task in range(2):
            s=p101.initial()
            prev=journal[task][-1][0]
            for action,effect in journal[task]:
                s=p101.confirmed_write(s,prev,effect)
                prev=action
            q=world.queries[task]
            scores.append(float(((q@s.weights.T)-(q@world.maps[task].T)).square().mean()))
        wrong=statistics.mean(scores)
    digest=hashlib.sha256(json.dumps(ledger,separators=(",",":")).encode()).hexdigest()
    if len(ledger)!=2*STEPS:
        raise ValueError("P102_EFFECT_BUDGET_MISMATCH")
    return {
        "mse_all":statistics.mean(errors),"mse_task0":errors[0],
        "mse_task1":errors[1],"zero_mse":statistics.mean(zero_errors),
        "confirmed_effects":len(ledger),"persistent_float64_state":1024,
        "cold_bitwise":True,"action_journal_sha256":digest,
        "wrong_credit_mse":wrong,
    }


def evaluate(family: str, split: str, index: int, arm: str,
             beta: float=1., wrong_credit: bool=False,
             count: int|None=None):
    n=COUNT[split] if count is None else count
    if type(n) is not int or not 1 <= n <= COUNT[split]:
        raise ValueError("P102_NONFROZEN_SCOPE")
    seed=SEEDS[family][split][index]
    rows=[run_world(build_world(family,split,index,w),arm,seed,w,beta,
                    wrong_credit=wrong_credit) for w in range(n)]
    result={
        "world_count":n,"mse_all":statistics.mean(z["mse_all"] for z in rows),
        "mse_task0":statistics.mean(z["mse_task0"] for z in rows),
        "mse_task1":statistics.mean(z["mse_task1"] for z in rows),
        "zero_mse":statistics.mean(z["zero_mse"] for z in rows),
        "total_confirmed_outcomes":sum(z["confirmed_effects"] for z in rows),
        "persistent_float64_state":1024,
        "cold_bitwise":all(z["cold_bitwise"] for z in rows),
        "action_journal_sha256":hashlib.sha256("".join(z["action_journal_sha256"]
                                          for z in rows).encode()).hexdigest(),
        "wrong_credit_mse":(statistics.mean(z["wrong_credit_mse"] for z in rows)
                            if wrong_credit else None),
    }
    return result


def bench(family: str, index: int, arm: str, beta: float):
    seed=SEEDS[family]["CAL"][index]
    worlds=[build_world(family,"CAL",index,w) for w in range(8)]
    def operation():
        return [run_world(w,arm,seed,i,beta) for i,w in enumerate(worlds)]
    with torch.no_grad():
        for _ in range(2):
            operation()
        xs=[]
        for _ in range(7):
            t=time.perf_counter()
            operation()
            xs.append((time.perf_counter()-t)*1000)
    return statistics.median(xs)


def study():
    frozen()  # exact original protocol/custody verified BEFORE data birth
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    receipt={
        "schema":"ttt-hybrid.p102-causal-consequence-intervention-dev-result.v1",
        "status":"NOT_STARTED","freeze_sha":FREEZE,
        "freeze_blob":FREEZE_BLOB,"scientific_pass":False,
        "original_model_edited":False,"native_hybrid_trained":False,
        "protected_OOD_generated":False,"protected_NEGATIVE_generated":False,
        "external_source_used":False,"source_sha256":{},"families":{},
    }
    for family in FAMILIES:
        for split in COUNT:
            for i in range(3):
                key=f"{family}_{split}_{i}"
                receipt["source_sha256"][key]=source_sha(family,split,i)
    if len(set(receipt["source_sha256"].values()))!=18:
        raise ValueError("P102_SOURCE_HASH_COLLISION")
    with torch.no_grad():
        for family in FAMILIES:
            rows=[]
            for i in range(3):
                fit_passive={arm:evaluate(family,"FIT",i,arm) for arm in PASSIVE}
                selected_passive=min(PASSIVE,key=lambda a:fit_passive[a]["mse_all"])
                beta_grid={str(beta):evaluate(family,"FIT",i,"impact",beta)
                           for beta in BETAS}
                selected_beta=min(BETAS,key=lambda b:beta_grid[str(b)]["mse_all"])
                fit_geometric=evaluate(family,"FIT",i,"geometry")
                active=evaluate(family,"CAL",i,"impact",selected_beta,
                                wrong_credit=True)
                geo=evaluate(family,"CAL",i,"geometry")
                control=evaluate(family,"CAL",i,selected_passive)
                misaligned=evaluate(family,"CAL",i,"impact_misaligned",
                                    selected_beta)
                zeros=active["zero_mse"]
                measures={}
                for name,base in (("geometry",geo),("FIT_selected_passive",control)):
                    measures[name]={k:active["mse_"+k]/max(base["mse_"+k],1e-18)
                                    for k in ("all","task0","task1")}
                wrong_ratio=active["wrong_credit_mse"]/max(active["mse_all"],1e-18)
                cost={arm:bench(family,i,arm,selected_beta)
                      for arm in ("impact","geometry",selected_passive)}
                cpu_geom=cost["impact"]/max(cost["geometry"],1e-12)
                cpu_control=cost["impact"]/max(cost[selected_passive],1e-12)
                gates={
                    "beats_geometry_all":measures["geometry"]["all"]<=.90,
                    "beats_passive_all":measures["FIT_selected_passive"]["all"]<=.90,
                    "beats_geometry_task0_cold":measures["geometry"]["task0"]<=.95,
                    "beats_geometry_task1":measures["geometry"]["task1"]<=.95,
                    "beats_passive_task0_cold":measures["FIT_selected_passive"]["task0"]<=.95,
                    "beats_passive_task1":measures["FIT_selected_passive"]["task1"]<=.95,
                    "wrong_credit_negative":wrong_ratio>=1.05,
                    "all_controls_outcome_budget":all(z["total_confirmed_outcomes"]
                        ==COUNT["CAL"]*2*STEPS for z in (active,geo,control,misaligned)),
                    "equal_learner_capacity":all(z["persistent_float64_state"]
                        ==1024 for z in (active,geo,control,misaligned)),
                    "cold_exact":all(z["cold_bitwise"] for z in (active,geo,control,misaligned)),
                    "zero_qualified":active["mse_all"]<zeros and
                        geo["mse_all"]<zeros and control["mse_all"]<zeros,
                    "cpu_geometry":cpu_geom<=1.6,
                    "cpu_passive":cpu_control<=3.,
                }
                rows.append({
                    "seed_index":i,"FIT_selected_beta":selected_beta,
                    "FIT_impact_grid":beta_grid,"FIT_geometric":fit_geometric,
                    "FIT_passives":fit_passive,"FIT_selected_passive":selected_passive,
                    "CAL_impact":active,"CAL_geometric":geo,
                    "CAL_selected_passive":control,"CAL_misaligned_ablation":misaligned,
                    "ratios_impact_vs":measures,"wrong_credit_ratio":wrong_ratio,
                    "cpu_median_ms":cost,"cpu_ratio_vs_geometry":cpu_geom,
                    "cpu_ratio_vs_passive":cpu_control,"frozen_gates":gates,
                    "seed_pass":all(gates.values()),
                })
            family_out={
                "seeds":rows,"pass_2_of_3":sum(r["seed_pass"] for r in rows)>=2,
                "median_ratios_all":{
                    name:statistics.median(r["ratios_impact_vs"][name]["all"] for r in rows)
                    for name in ("geometry","FIT_selected_passive")
                },
            }
            family_out["family_pass"]=(
                family_out["pass_2_of_3"] and
                all(z<=.90 for z in family_out["median_ratios_all"].values()))
            receipt["families"][family]=family_out
    receipt["status"]=(
        "DEV_CAL_ONLY_POSITIVE_NONPROMOTING_INDEPENDENT_FIRSTUSE_REQUIRED"
        if all(receipt["families"][f]["family_pass"]
               for f in ("isotropic","anisotropic"))
        else "DEV_FAIL_CONSEQUENCE_CONDITIONED_ACQUISITION_OR_COST")
    receipt["whole_study_wall_seconds"]=time.perf_counter()-start
    return receipt


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    report=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2,
                                      allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],
                      "family_pass":{f:row["family_pass"]
                                     for f,row in report["families"].items()},
                      "scientific_pass":False}))


if __name__=="__main__":
    main()
