"""P43: preregistered DEV-only actual-action nuisance/OOD training comparison.

Runs only authored, openly specified analog worlds. No protected source, no
scientific first use, no external runner, no FORGE research-loop integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HISTORY = 8
FIT_N = 4096
VALID_N = 512
OOD_N = 512
SOURCE_SEEDS = {"fit": 710001, "validation": 710101, "ood_dev": 711001}
INIT_SEEDS = (9101, 9113, 9127)
UPDATES = 400
BATCH_SIZE = 128
SAMPLE_SEED = 720001
FREQUENCIES = {"fit": (.45, .95), "validation": (.45, .95), "ood_dev": (1.15, 1.65)}
SCHEMA = "ttt-hybrid.p43.factual-neural-nuisance-ood-dev.v1"


def authored(n: int, cohort: str) -> dict[str, torch.Tensor | str]:
    """One actual query action/source outcome for training; heldout both only scoring."""
    if cohort not in SOURCE_SEEDS or n != {"fit": FIT_N, "validation": VALID_N, "ood_dev": OOD_N}[cohort]:
        raise ValueError("P43_NONFROZEN_SOURCE_COHORT")
    gen = np.random.default_rng(SOURCE_SEEDS[cohort])
    fmin, fmax = FREQUENCIES[cohort]
    b0 = gen.uniform(-.4, .4, (n, 1))
    b1 = gen.uniform(-.5, .5, (n, 1))
    b2 = gen.uniform(-.2, .2, (n, 1))
    b3 = gen.uniform(-.2, .2, (n, 1))
    phase = gen.uniform(-math.pi, math.pi, (n, 1))
    frequency = gen.uniform(fmin, fmax, (n, 1))
    beta = gen.choice(np.array([-1., 1.]), (n, 1)) * gen.uniform(.15, .45, (n, 1))
    observed_actions = gen.choice(np.array([-1., 1.]), (n, HISTORY))
    factual_query_action = gen.choice(np.array([-1., 1.]), (n, 1))
    t = (np.arange(HISTORY, dtype=np.float64) / HISTORY)[None, :]
    tq = 1.0
    hist_background = b0 + b1*t + b2*t*t + b3*np.sin(2*np.pi*frequency*t + phase)
    query_background = b0 + b1*tq + b2*tq*tq + b3*np.sin(2*np.pi*frequency*tq + phase)
    history_observed = hist_background + beta*observed_actions + gen.normal(0,.01,(n,HISTORY))
    query_observed = query_background + beta*factual_query_action + gen.normal(0,.01,(n,1))
    # Evaluator-only: two independent physical responses on a heldout world's
    # same pre-action state; not visible to any trainable learner during fitting.
    eval_actions = np.array([-1., 1.], dtype=np.float64)[None, :]
    both_physically_executed = (query_background + beta*eval_actions
                                 + gen.normal(0,.01,(n,2)))
    history = np.stack((np.broadcast_to(t,(n,HISTORY)), observed_actions,
                        history_observed), axis=-1).astype(np.float32)
    h, a, y, both, b = (torch.tensor(z,dtype=torch.float32) for z in (
        history, factual_query_action, query_observed, both_physically_executed, beta))
    receipt = hashlib.sha256(b"".join(z.numpy().tobytes() for z in (h,a,y,both,b))).hexdigest()
    return {"history":h,"query_action":a,"factual_query":y,"scored_both":both,"beta":b,
            "source_trace_sha256":receipt,"cohort":cohort}


class FactorizedHistoryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(HISTORY*3,32), nn.GELU(),
                                     nn.Linear(32,24), nn.GELU())
        self.drift = nn.Linear(24,1)
        self.causal = nn.Linear(24,1)

    def forward(self, history, action):
        state=self.encoder(history.flatten(1))
        return self.drift(state) + self.causal(state)*action


class ConventionalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor=nn.Sequential(nn.Linear(HISTORY*3+1,32),nn.GELU(),
                                     nn.Linear(32,24),nn.GELU(),nn.Linear(24,1))

    def forward(self, history, action):
        return self.predictor(torch.cat((history.flatten(1), action),dim=-1))


class ConventionalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(3,18,batch_first=True)
        self.query=nn.Sequential(nn.Linear(19,24),nn.GELU(),nn.Linear(24,1))

    def forward(self, history, action):
        _, h=self.encoder(history)
        return self.query(torch.cat((h[-1],action),dim=-1))


def wrong_credit_history(history):
    wrong=history.clone()
    wrong[:,:,1]=history[:,:,1].roll(1,dims=1)
    return wrong


def params(model):
    return sum(p.numel() for p in model.parameters())


def fit_model(model, fit, valid, draws, *, wrong_credit=False):
    opt=torch.optim.AdamW(model.parameters(),lr=.002)
    history=wrong_credit_history(fit["history"]) if wrong_credit else fit["history"]
    valid_history=wrong_credit_history(valid["history"]) if wrong_credit else valid["history"]
    trajectory=[]
    started=time.perf_counter()
    for step in range(UPDATES):
        model.train()
        ix=draws[step]
        prediction=model(history[ix],fit["query_action"][ix])
        loss=(prediction-fit["factual_query"][ix]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1)%100 == 0:
            model.eval()
            with torch.no_grad():
                mse=(model(valid_history,valid["query_action"])-valid["factual_query"]).square().mean()
            trajectory.append(float(mse))
    elapsed=time.perf_counter()-started
    return {"optimizer_updates":UPDATES,"validation_curve":trajectory,
            "train_wall_seconds":elapsed,
            "validation_plateau_last_two_relative":abs(trajectory[-1]-trajectory[-2])/max(trajectory[-2],1e-9)}


def evaluate(model, cohort, *, wrong_credit=False):
    model.eval()
    h=wrong_credit_history(cohort["history"]) if wrong_credit else cohort["history"]
    n=h.shape[0]
    started=time.perf_counter()
    with torch.no_grad():
        minus=model(h,torch.full((n,1),-1.,dtype=h.dtype))
        plus=model(h,torch.full((n,1),1.,dtype=h.dtype))
    scored=torch.cat((minus,plus),dim=1)
    err=scored-cohort["scored_both"]
    effect=(plus-minus)*.5
    opt=0
    for goal in (-.35,.35):
        preferred=(scored-goal).square().argmin(1)
        actual=(cohort["scored_both"]-goal).square().argmin(1)
        opt += int((preferred==actual).sum())
    return {"two_action_mse":float(err.square().mean()),
            "inferred_causal_beta_mae":float((effect-cohort["beta"]).abs().mean()),
            "goal_optimal_fraction":opt/(2*n),
            "inference_wall_seconds":time.perf_counter()-started}


def polynomial_ridge(cohort):
    """Strong conventional per-world 4-feature factual fit; no source frequency."""
    h=cohort["history"].double()
    n=h.shape[0]
    times=h[:,:,0]
    actions=h[:,:,1]
    X=torch.stack((torch.ones_like(times),times,times.square(),actions),dim=-1)
    y=h[:,:,2,None]
    rank=torch.linalg.matrix_rank(X)
    ridge=torch.eye(4,dtype=torch.float64).expand(n,-1,-1).clone()*0.001
    ridge[:,0,0]=0.0
    started=time.perf_counter()
    mat=X.transpose(1,2)@X+ridge
    coeff=torch.linalg.solve(mat,X.transpose(1,2)@y).squeeze(-1)
    # Flag truly unidentifiable cases, don't invent action effect.
    identifiable=rank==4
    coeff[~identifiable,3]=0.0
    minus=coeff[:,0]+coeff[:,1]+coeff[:,2]-coeff[:,3]
    plus=coeff[:,0]+coeff[:,1]+coeff[:,2]+coeff[:,3]
    scored=torch.stack((minus,plus),dim=1).float()
    err=scored-cohort["scored_both"]
    effect=(scored[:,1]-scored[:,0])*.5
    opt=0
    for goal in (-.35,.35):
        opt+=int(((scored-goal).square().argmin(1)==
                  (cohort["scored_both"]-goal).square().argmin(1)).sum())
    return {"two_action_mse":float(err.square().mean()),
            "inferred_causal_beta_mae":float((effect[:,None]-cohort["beta"]).abs().mean()),
            "goal_optimal_fraction":opt/(2*n),"rank_sufficient":int(identifiable.sum()),
            "inference_wall_seconds":time.perf_counter()-started}


def last_effect_baseline(cohort):
    h=cohort["history"]
    beta=h[:,-1,2]/h[:,-1,1]
    return {"inferred_causal_beta_mae":float((beta[:,None]-cohort["beta"]).abs().mean())}


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    cohorts={k:authored(n,k) for k,n in (("fit",FIT_N),("validation",VALID_N),("ood_dev",OOD_N))}
    params_by_model={name:params(factory()) for name,factory in (
        ("factorized",FactorizedHistoryMLP),("mlp",ConventionalMLP),("gru",ConventionalGRU),
        ("wrong_credit",FactorizedHistoryMLP))}
    draws=torch.randint(0,FIT_N,(UPDATES,BATCH_SIZE),generator=torch.Generator().manual_seed(SAMPLE_SEED))
    shared_draw_sha=hashlib.sha256(draws.numpy().tobytes()).hexdigest()
    results=[]
    for seed in INIT_SEEDS:
        arms={}
        for name,factory in (("factorized",FactorizedHistoryMLP),("mlp",ConventionalMLP),
                             ("gru",ConventionalGRU),("wrong_credit",FactorizedHistoryMLP)):
            torch.manual_seed(seed)
            model=factory()
            corrupt=name=="wrong_credit"
            training=fit_model(model,cohorts["fit"],cohorts["validation"],draws,wrong_credit=corrupt)
            arms[name]={"training":training,"val":evaluate(model,cohorts["validation"],wrong_credit=corrupt),
                        "ood":evaluate(model,cohorts["ood_dev"],wrong_credit=corrupt)}
        results.append({"seed":seed,"arms":arms})
    fitted=polynomial_ridge(cohorts["ood_dev"])
    heuristic=last_effect_baseline(cohorts["ood_dev"])
    prespecified=[]
    for x in results:
        a=x["arms"]
        c=a["factorized"]["ood"]
        controls=[a["mlp"]["ood"]["two_action_mse"],a["gru"]["ood"]["two_action_mse"],fitted["two_action_mse"]]
        prespecified.append({"seed":x["seed"],
            "mse_beats_all_strong_controls_by_10pct":c["two_action_mse"] <= .9*min(controls),
            "effect_mae_beats_both_neural_baselines":all(c["inferred_causal_beta_mae"]<a[k]["ood"]["inferred_causal_beta_mae"] for k in ("mlp","gru")),
            "wrong_credit_mse_margin_20pct":c["two_action_mse"] <= .8*a["wrong_credit"]["ood"]["two_action_mse"]})
    receipt={"schema":SCHEMA,"source_type":"AUTHORED_PUBLIC_DEVELOPMENT_ONLY",
             "scientific_first_use":"NOT_RUN","windows_wsl_full_regression":"NOT_RUN",
             "source_sha256":{k:v["source_trace_sha256"] for k,v in cohorts.items()},
             "matched_source_sample_index_sha256":shared_draw_sha,
             "model_params":params_by_model,"updates_each":UPDATES,
             "seeds":list(INIT_SEEDS),"fitted_polynomial_ridge_ood":fitted,
             "zero_train_last_actual_effect_ood":heuristic,"model_seed_results":results,
             "prespecified_thresholds":prespecified,
             "all_three_seed_gates_pass":all(all(row[k] for k in ("mse_beats_all_strong_controls_by_10pct",
                                                "effect_mae_beats_both_neural_baselines",
                                                "wrong_credit_mse_margin_20pct")) for row in prespecified),
             "overall_wall_seconds":time.perf_counter()-started}
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"source_sha256":result["source_sha256"],"params":result["model_params"],
                      "source_index_sha256":result["matched_source_sample_index_sha256"],
                      "strong_ridge":result["fitted_polynomial_ridge_ood"],
                      "gates":result["prespecified_thresholds"],
                      "all_three_seed_gates_pass":result["all_three_seed_gates_pass"],
                      "wall_seconds":result["overall_wall_seconds"]},indent=2))


if __name__=="__main__":
    main()
