"""F8-R18: actual native training on paired-identical-B synthetic A→B→A.

This is *author-generated SOURCE-NEUTRAL development*, NOT external source first use.
The learner sees a factual early A probe, 8 identical B effects per matched pair,
then must act before any new A observation. A/B labels remain evaluator-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments import p85mr_f8r3_counterpolicy_identifiability as r3
from experiments import p85mr_f8r5_native_policy as r5
from p85mr_dynamics_baselines import new_baseline, prequential as baseline_predict

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r18_matched_aba_freeze.json"
FREEZE_BLOB = "2b82e54e5be7f515fd75fc659b2074b4e42db270"
SCHEMA = "ttt-hybrid.f8r18-identified-matched-aba-native-development.v1"
ARMS = ("bypass_hard", "bypass_surrogate", "linear7", "mlp7", "ssm256")
SEEDS = (985201, 985203, 985207)
GROUPS = {"fit": (19450000, 32), "dev": (19460000, 12),
          "score": (19470000, 16)}
GOALS = (-.2, .2)
UPDATES, BATCH, BOOT, BOOT_SEED = 128, 8, 10000, 19479999
B_STEPS = 8


def locked() -> dict:
    raw = FREEZE.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F8R18_FROZEN_BLOB_DRIFT")
    f = json.loads(raw)
    w, tr, sc, auth = f["world"], f["training"], f["score"], f["authority"]
    if (f.get("schema") != SCHEMA
        or f["stage"] != "FROZEN_BEFORE_CODE_TRAINING_RESULTS"
        or f["classification"] != "SOURCE_NEUTRAL_TRAINED_NATIVE_DEV_ONLY"
        or f["parent"]["head"] != "28f2999821975762b114b375a338adec3e9fa324"
        or f["parent"]["original_f8r3_blob"] !=
            "9a7077ece06dfbc685232bbc606b0c2dd33a7895"
        or f["parent"]["original_f8r5_native_blob"] !=
            "9ac9fc17d6f82d6bf2dc7b05458939d5bdb6ea92"
        or tuple(w["per_group_signs"]) != (-1, 1)
        or any(w["group_counts"][part] != n or
               w["group_seed_base"][part] != seed
               for part, (seed, n) in GROUPS.items())
        or w["B_phase"]["steps"] != B_STEPS
        or w["A_probe"]["initial_state"] != .2
        or w["A_probe"]["selected_action"] != 0
        or w["first_return_A"]["reset_public_state_to_zero"] is not True
        or w["first_return_A"]["no_factual_return_A_effect_in_dev_or_score_before_action"] is not True
        or w["test_goal_crossed_with_A_and_B"] is not True
        or tuple(tr["arms"]) != ARMS
        or tuple(tr["seeds"]) != SEEDS
        or tr["updates_per_model"] != UPDATES
        or tr["episodes_per_batch"] != BATCH
        or tr["total_models"] != 15
        or tr["total_optimizer_updates"] != 1920
        or tr["shared_sampler_seed"] != 985299
        or tr["native_lr"] != .002 or tr["conventional_lr"] != .001
        ):
        raise ValueError("F8R18_FROZEN_IDENTITY_DRIFT")
    if (tuple(w["first_return_A"]["goals"]) != GOALS
        or sc["bootstrap"]["resamples"] != BOOT
        or sc["bootstrap"]["seed"] != BOOT_SEED
        or sc["primary"].find("16 matched B groups") == -1
        or sc["candidate"] != "bypass_surrogate"
        or sc["independent_scientific_pass"] is not False
        or any(value is not False for value in auth.values())):
        raise ValueError("F8R18_FROZEN_SCOPE_OR_AUTHORITY_DRIFT")
    r5.locked()
    r3.locked()
    return f


def _observed(before: float, after: float, *, final: bool = False):
    return r5._observed_effect((before, 0., 0., 0.),
                               (after, 0., 0., 0.),
                               1. - min(1., (after-.2)**2),
                               final, False)


def episode(split: str, group: int, sign_a: int) -> dict:
    if (split not in GROUPS or type(group) is not int
        or not 0 <= group < GROUPS[split][1]
        or type(sign_a) is not int or sign_a not in (-1,1)):
        raise ValueError("F8R18_UNREGISTERED_PAIR_WORLD")
    seed = GROUPS[split][0] + group
    sign_b = 1 if group % 2 == 0 else -1
    actions, effects = [], []
    # One genuinely selected early-A physical action exposes the A mechanism
    # only through its actually observed consequence. No sign is in the key.
    s, act = .2, 0
    a_effect = r3.synthetic_effect(s, act, sign_a)
    actions.append(r5._action_key((s,0.,0.,0.),act))
    effects.append(_observed(s,a_effect))
    # Deliberate observable B anchor reset: B start and full B effects are
    # identical between +A and -A members. B sign is not derived from A.
    s = 0.
    for step in range(B_STEPS):
        act = r3.behavior(s) ^ int((seed + 7*step)%3 == 0)
        following = r3.synthetic_effect(s, act, sign_b)
        actions.append(r5._action_key((s,0.,0.,0.),act))
        effects.append(_observed(s,following))
        s = following
    if split == "fit":
        # Distinct *factual* FIT return-A actions across groups. The learner
        # never receives both counterpolicy labels within the same session.
        selected = group % 2
        following = r3.synthetic_effect(0.,selected,sign_a)
        actions.append(r5._action_key((0.,0.,0.,0.),selected))
        effects.append(_observed(0.,following,final=True))
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32),
            "group":group,"sign_a":sign_a,"sign_b":sign_b,
            "goals":GOALS,"source_seed":seed,"return_state":0.}


def pair_cohort(split: str) -> list[dict]:
    locked()
    return [episode(split,i,sign)
            for i in range(GROUPS[split][1]) for sign in (-1,1)]


def factual(ep: dict) -> dict:
    return {"action":ep["action"],"effect":ep["effect"]}


def _model(arm: str, seed: int):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("F8R18_UNREGISTERED_MODEL_OR_SEED")
    return new_baseline("mlp7",seed) if arm=="mlp7" else r5.model_for(arm,seed)


def _predict(arm: str, model, data: dict) -> torch.Tensor:
    return baseline_predict(model,data) if arm=="mlp7" else r5.predict(arm,model,data)


def _objective(arm: str,model,data: dict,valid: torch.Tensor):
    return r5.f4._physical_loss(model,data,valid)[0] if arm=="mlp7" else r5.objective(arm,model,data,valid)


def _state_predictions(arm: str,model,eps: list[dict],*,erase_first_A=False):
    """Batched two-action hypothetical inference, before any return-A effect.

    Current effects are zeros (masked); no unchosen oracle or outcome is sent
    to the model. Score-world metadata signs remain outside the batch.
    """
    actions, effects = [], []
    for ep in eps:
        for action in (0,1):
            current = torch.tensor([r5._action_key((0.,0.,0.,0.),action)],dtype=torch.float32)
            a = torch.cat((ep["action"],current),dim=0)
            e = torch.cat((ep["effect"],torch.zeros((1,16),dtype=torch.float32)),dim=0)
            if erase_first_A:
                e[0] = torch.zeros_like(e[0])
            actions.append(a)
            effects.append(e)
    with torch.no_grad():
        outputs=_predict(arm,model,{"action":torch.stack(actions),
                                    "effect":torch.stack(effects)})
    if outputs.shape != (len(eps)*2,10,16) or not bool(torch.isfinite(outputs).all()):
        raise ValueError("F8R18_UNQUALIFIED_PREACTION_PREDICTION")
    delta=outputs[:,-1,0].detach().cpu().numpy().astype(np.float64)
    proposed=np.asarray([2.4*math.atanh(max(-.98,min(.98,float(q))))
                          for q in delta],dtype=np.float64).reshape(len(eps),2)
    return proposed


def _regret(sign: int,goal: float,act: int) -> float:
    """Evaluator-only synthetic selected effect and post-choice oracle."""
    if type(sign) is not int or sign not in (-1,1) or act not in (0,1):
        raise ValueError("F8R18_INVALID_EVALUATOR_INPUT")
    y=r3.synthetic_effect(0.,act,sign)
    oracle=min((r3.synthetic_effect(0.,x,sign)-goal)**2 for x in (0,1))
    return max(0.,float((y-goal)**2-oracle))


def _fixed_controls(eps: list[dict]) -> dict:
    controls={"fixed0":[],"fixed1":[],"goal_positive":[],"goal_negative":[]}
    for group in range(len(eps)//2):
        by_sign=eps[2*group:2*group+2]
        if [p["sign_a"] for p in by_sign] != [-1,1]:
            raise ValueError("F8R18_MISMATCHED_PAIR_ORDER")
        for name in controls:
            r=[]
            for ep in by_sign:
                for goal in GOALS:
                    chosen={"fixed0":0,"fixed1":1,
                            "goal_positive":int(goal>0),
                            "goal_negative":int(goal<0)}[name]
                    r.append(_regret(ep["sign_a"],goal,chosen))
            controls[name].append(float(math.fsum(r)/4))
    means={k:float(np.mean(v)) for k,v in controls.items()}
    if (not math.isclose(means["fixed1"],.015,rel_tol=0,abs_tol=1e-12)
        or not math.isclose(means["fixed0"],.105,rel_tol=0,abs_tol=1e-12)):
        raise ValueError("F8R18_BROKEN_IDENTIFIABILITY_LOWER_BOUND")
    return {"means":means,"per_group":controls,
            "best_fixed":min(means,key=lambda k:(means[k],k)),
            "analytic_A_memory_oracle_mean_regret":0.}


def score(arm: str,model,eps: list[dict]) -> dict:
    model.eval()
    started=time.perf_counter()
    if len(eps)%2 or not eps or [p["sign_a"] for p in eps[::2]] != [-1]*(len(eps)//2):
        raise ValueError("F8R18_UNQUALIFIED_EVAL_PAIR_PANEL")
    for i in range(0,len(eps),2):
        minus,plus=eps[i:i+2]
        if (minus["group"] != plus["group"]
            or minus["sign_b"] != plus["sign_b"]
            or not torch.equal(minus["action"][1:],plus["action"][1:])
            or not torch.equal(minus["effect"][1:],plus["effect"][1:])
            or not torch.equal(minus["action"],plus["action"])
            or torch.equal(minus["effect"][0],plus["effect"][0])):
            raise ValueError("F8R18_MISSING_IDENTICAL_B_OR_A_CONTRAST")
    pred=_state_predictions(arm,model,eps)
    erased=_state_predictions(arm,model,eps,erase_first_A=True)
    regrets,erased_regrets,choices,erased_choices=[],[],[],[]
    for row, ep in enumerate(eps):
        for goal in GOALS:
            action=min((0,1),key=lambda a:((pred[row,a]-goal)**2,a))
            erased_action=min((0,1),key=lambda a:((erased[row,a]-goal)**2,a))
            choices.append(action); erased_choices.append(erased_action)
            regrets.append(_regret(ep["sign_a"],goal,action))
            erased_regrets.append(_regret(ep["sign_a"],goal,erased_action))
    grouped=[float(math.fsum(regrets[4*i:4*i+4])/4) for i in range(len(eps)//2)]
    grouped_erased=[float(math.fsum(erased_regrets[4*i:4*i+4])/4)
                    for i in range(len(eps)//2)]
    contrasts=sum(choices[4*i+j]!=choices[4*i+2+j]
                  for i in range(len(eps)//2) for j in (0,1))
    n_groups=len(eps)//2
    return {"mean_regret":float(np.mean(grouped)),
            "per_group_regret":grouped,
            "erased_initial_A_mean_regret":float(np.mean(grouped_erased)),
            "per_group_erased_regret":grouped_erased,
            "A_sign_action_contrast_count":contrasts,
            "A_sign_action_contrast_pairs":n_groups*2,
            "model_selected_synthetic_actions":4*n_groups,
            "hypothetical_queries":len(eps)*2*2,
            "score_seconds":time.perf_counter()-started,
            "selected_actions":choices,
            "zeroed_probe_selected_actions":erased_choices}


def train_one(arm: str, seed: int, fit:list[dict],dev:list[dict],
              test:list[dict], sampler:list[list[int]]) -> dict:
    model=_model(arm,seed).cpu()
    initial=r5.f5.hash_weights(model)
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("bypass_") else .001, weight_decay=.01)
    t0=time.perf_counter()
    losses=[]
    for indices in sampler:
        batch, valid=r5.f4._padded([factual(fit[i]) for i in indices])
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=_objective(arm,model,batch,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R18_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(v is not None and bool((v!=0).any()) for v in grads)
            or any(v is not None and not bool(torch.isfinite(v).all()) for v in grads)):
            raise ValueError("F8R18_BAD_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    elapsed=time.perf_counter()-t0
    trained=r5.f5.hash_weights(model)
    if initial==trained:
        raise ValueError("F8R18_MODEL_NOT_TRAINED")
    model.eval()
    dev_score=score(arm,model,dev)
    test_score=score(arm,model,test)
    restored=_model(arm,seed).cpu()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    restored.eval()
    restored_score=score(arm,restored,test)
    if (r5.f5.hash_weights(restored)!=trained
        or not np.allclose(test_score["per_group_regret"],
                           restored_score["per_group_regret"],atol=1e-12,rtol=0)
        or test_score["selected_actions"]!=restored_score["selected_actions"]):
        raise ValueError("F8R18_COLD_MODEL_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"model_parameters":sum(p.numel() for p in model.parameters()),
            "updates":UPDATES,"first16_training_loss":float(np.mean(losses[:16])),
            "last16_training_loss":float(np.mean(losses[-16:])),
            "original_weight_hash":initial,"trained_weight_hash":trained,
            "train_seconds":elapsed,
            "dev":dev_score,"test":test_score,
            "cold_model_restore_exact_action_parity":True}


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=pair_cohort("fit")
    dev=pair_cohort("dev")
    test=pair_cohort("score")
    # Prove matched recent B is identical before any model sees the data.
    for group in range(GROUPS["score"][1]):
        minus,plus=test[2*group:2*group+2]
        if (not torch.equal(minus["action"],plus["action"])
            or not torch.equal(minus["effect"][1:],plus["effect"][1:])):
            raise ValueError("F8R18_IDENTIFIABILITY_MATCH_FAILED")
    ctrl=_fixed_controls(test)
    sampler=torch.Generator(device="cpu")
    sampler.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=sampler).tolist()
    t0=time.perf_counter()
    rows=[train_one(a,s,fit,dev,test,batches)
          for a in ARMS for s in SEEDS]
    baselines=[r for r in rows if r["arm"] in ("linear7","mlp7","ssm256")]
    strongest=min(baselines,key=lambda r:
                  (r["dev"]["mean_regret"],r["arm"],r["seed"]))
    best_fixed=min(ctrl["means"].values())
    control_score=min(best_fixed,strongest["test"]["mean_regret"])
    # Same 16 group draws for all arms. Each contains both A signs and both
    # independent target goals. Never bootstrap 64 correlated decisions.
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,16,size=(BOOT,16))
    winners=[]
    conventional=strongest["test"]["per_group_regret"]
    fixed=ctrl["per_group"][ctrl["best_fixed"]]
    chosen_reference=conventional if strongest["test"]["mean_regret"] < best_fixed else fixed
    for row in rows:
        if row["arm"]!="bypass_surrogate":continue
        improvement=np.asarray(chosen_reference)-np.asarray(row["test"]["per_group_regret"])
        lower=float(np.percentile(improvement[draws].mean(axis=1),2.5))
        base=float(np.mean(chosen_reference))
        mean=float(row["test"]["mean_regret"])
        winner=bool(base>1e-12 and mean<=.9*base and lower>0)
        winners.append({"seed":row["seed"],
            "mean_action_regret":mean,"relative_improvement_vs_best_control":
            (base-mean)/base if base>0 else -1.,
            "bootstrap_2p5_absolute_gain":lower,"candidate_dev_win":winner})
    return {"schema":SCHEMA,"classification":"SOURCE_NEUTRAL_TRAINED_NATIVE_DEV_ONLY",
            "freeze_blob":FREEZE_BLOB,"models":rows,"model_count":len(rows),
            "optimizer_updates":len(rows)*UPDATES,
            "synthetic_fit_factual_effects":len(fit)*10,
            "synthetic_dev_observed_early_A_and_B_effects":len(dev)*9,
            "synthetic_score_observed_early_A_and_B_effects":len(test)*9,
            "score_model_selected_synthetic_actions":len(rows)*len(test)*2,
            "new_external_effects":0,"original_F7_replay":False,
            "score_identical_B_paired_groups":GROUPS["score"][1],
            "score_goals_independently_crossed":list(GOALS),
            "fixed_controls":ctrl,
            "strongest_dev_selected_conventional":{
                "arm":strongest["arm"],"seed":strongest["seed"],
                "dev_mean_regret":strongest["dev"]["mean_regret"],
                "test_mean_regret":strongest["test"]["mean_regret"]},
            "comparison_reference_mean_regret":control_score,
            "candidate_seed_decisions":winners,
            "candidate_dev_gate_pass":sum(r["candidate_dev_win"] for r in winners)>=2,
            "conventional_training_convergence_qualified":False,
            "independent_scientific_pass":False,
            "deployment":"NONE","elapsed_dev_cpu_seconds":time.perf_counter()-t0}


def retain_ci_receipt(result: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r18_matched_aba_dev.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R18_REFUSE_EVIDENCE_REPLAY_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R18_ORIGINAL_FIRSTUSE_NOT_AUTHORIZED")
