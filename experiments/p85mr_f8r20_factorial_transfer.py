"""R20: frozen factorial matched-B causal memory, author-synthetic DEV only.

Cross A sign, independent B sign and actually selected return action in FIT.
The score decision occurs BEFORE any new A outcome. Never run the external
CausalGym source, replay F7, or claim independent scientific validation.
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

from experiments import p85mr_f8r18_matched_aba as r18
from experiments import p85mr_f8r19_balanced_return as r19
from experiments import p85mr_f8r3_counterpolicy_identifiability as r3

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r20_factorial_transfer_freeze.json"
FREEZE_BLOB = "8a181c6fd71c9dc6d4c4d640396d19920eb74175"
R18_BLOB = "affdbf248d114cad5a012cfc07c60308a525665f"
R19_FOCUS_BLOB = "5f4418436f592672d5acf1e1444e15e561a77755"
SCHEMA = "ttt-hybrid.f8r20-factorial-long-B-transfer-dev.v1"
ARMS = ("native_uniform", "native_return_focus", "native_return_only",
        "linear_return_only", "mlp_return_only", "ssm_return_only")
MAPPING = {"native_uniform":"native_uniform", "native_return_focus":"native_return_focus",
           "native_return_only":"native_return_focus",
           "linear_return_only":"linear_return_focus",
           "mlp_return_only":"mlp_return_focus",
           "ssm_return_only":"ssm_return_focus"}
SEEDS = (985201, 985203, 985207)
GROUPS = {"fit": (19610000, 24, 8), "dev": (19620000, 8, 8),
          "id": (19630000, 12, 8), "ood": (19640000, 12, 12)}
B_SIGNS = (-1, 1)
A_SIGNS = (-1, 1)
GOALS = (-.2, .2)
UPDATES = 384
BATCH = 8
CHECKPOINTS = (128, 256, 384)
BOOT = 10000
BOOT_SEED = 19649999


def git_blob(content: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(content)).encode("ascii") + b"\0" + content
    ).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R20_FROZEN_PROTOCOL_DRIFT")
    if git_blob((ROOT / "experiments/p85mr_f8r18_matched_aba.py").read_bytes()) != R18_BLOB:
        raise ValueError("F8R20_PARENT_MODEL_SOURCE_DRIFT")
    if git_blob((ROOT / "experiments/p85mr_f8r19_balanced_return.py").read_bytes()) != R19_FOCUS_BLOB:
        raise ValueError("F8R20_PARENT_FOCUS_SOURCE_DRIFT")
    f = json.loads(raw)
    p, w, m, s, a = (f["parent"], f["synthetic_world"], f["model"],
                      f["score"], f["authority"])
    if (f["schema"] != SCHEMA
            or f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_AND_NUMERIC_RESULTS"
            or f["classification"] != "SOURCE_NEUTRAL_AUTHOR_GENERATED_DEVELOPMENT_ONLY"
            or p["sha"] != "c460ffd06a6a69d141de00d04c1982ef7f0a319b"
            or p["r18_source_blob"] != R18_BLOB
            or p["r19_focus_source_blob"] != R19_FOCUS_BLOB
            or p["original_r3_blob"] != "9a7077ece06dfbc685232bbc606b0c2dd33a7895"
            or w["group_count"] != {k: v[1] for k, v in GROUPS.items()}
            or w["seed_base"] != {k: v[0] for k, v in GROUPS.items()}
            or w["FIT_B_steps"] != 8 or w["DEV_B_steps"] != 8
            or w["ID_B_steps"] != 8 or w["OOD_B_steps"] != 12
            or w["FIT_sessions"] != 192 or w["FIT_factual_effects"] != 1920
            or w["A_signs"] != list(A_SIGNS)
            or w["B_signs"] != list(B_SIGNS)
            or w["goals"] != list(GOALS)
            or w["fit_return_actions"] != [0, 1]
            or not w["full_factorial_FIT_each_A_B_return_action_cell"]
            or not w["scoring_before_first_return_A_effect"]
            or m["arms"] != list(ARMS) or m["seeds"] != list(SEEDS)
            or m["underlying"] != [r19.UNDERLYING[MAPPING[arm]] for arm in ARMS]
            or m["total_fits"] != 18
            or m["updates_each"] != UPDATES or m["batch_worlds"] != BATCH
            or m["total_optimizer_updates"] != 6912
            or m["checkpoints"] != list(CHECKPOINTS)
            or m["sampler_seed"] != 985299
            or m["objective"] !=
               "uniform=existing physical, focus=existing physical+20xreturn factual delta, return_only=last factual return delta MSE channels0..3"
            or m["conventional_independently_converged"] is not False
            or s["bootstrap_resamples"] != BOOT or s["bootstrap_seed"] != BOOT_SEED
            or s["known_history_free_best_mean_regret"] != .015
            or s["candidate"] != "native_return_focus"
            or s["independent_sci_pass"] is not False
            or any(value is not False for value in a.values())):
        raise ValueError("F8R20_PINNED_SCOPE_OR_BUDGET_DRIFT")
    r18.locked()
    r19.locked()
    r3.locked()
    return f


def episode(split: str, group: int, z_b: int, z_a: int,
            factual_return: int | None = None) -> dict:
    if (split not in GROUPS or type(group) is not int
            or not 0 <= group < GROUPS[split][1]):
        raise ValueError("F8R20_UNREGISTERED_GROUP")
    if type(z_a) is not int or z_a not in A_SIGNS or type(z_b) is not int or z_b not in B_SIGNS:
        raise ValueError("F8R20_UNREGISTERED_LATENT_SIGN")
    if split == "fit":
        if type(factual_return) is not int or factual_return not in (0, 1):
            raise ValueError("F8R20_FIT_REQUIRES_FACTUAL_RETURN_ACTION")
    elif factual_return is not None:
        raise ValueError("F8R20_SCORE_MUST_NOT_OBSERVE_RETURN_EFFECT")
    seed, _, b_steps = GROUPS[split]
    seed += group
    actions, effects = [], []

    def record(before: float, action: int, after: float, final: bool = False):
        actions.append(r18.r5._action_key((before, 0., 0., 0.), action))
        effects.append(r18._observed(before, after, final=final))

    # The learner sees the result of one actually selected A probe, never z_a.
    record(.2, 0, r3.synthetic_effect(.2, 0, z_a))
    # Public reset makes B trajectories independent of the earlier A sign.
    state = 0.
    for t in range(b_steps):
        act = r3.behavior(state) ^ int((seed + 5*t + int(z_b == 1)) % 3 == 0)
        nxt = r3.synthetic_effect(state, act, z_b)
        record(state, act, nxt)
        state = nxt
    # Separate FIT sessions supply one factual action each. Never give a
    # learner an evaluator-only unchosen outcome within the same session.
    if split == "fit":
        record(0., factual_return, r3.synthetic_effect(0., factual_return, z_a),
               final=True)
    return {
        "action": torch.tensor(actions, dtype=torch.float32),
        "effect": torch.tensor(effects, dtype=torch.float32),
        "group": group, "z_a": z_a, "z_b": z_b, "split": split,
        "factual_return": factual_return, "seed_evaluator_only": seed
    }


def factual(ep: dict) -> dict:
    return {"action": ep["action"], "effect": ep["effect"]}


def cohort(split: str) -> list[dict]:
    locked()
    if split == "fit":
        return [episode(split, g, b, a, action)
                for g in range(GROUPS[split][1])
                for b in B_SIGNS for a in A_SIGNS for action in (0, 1)]
    return [episode(split, g, b, a)
            for g in range(GROUPS[split][1])
            for b in B_SIGNS for a in A_SIGNS]


def verify_matched(eps: list[dict], split: str) -> dict:
    if split not in GROUPS:
        raise ValueError("F8R20_INVALID_COHORT")
    if split == "fit":
        expected = 8 * GROUPS[split][1]
        if len(eps) != expected:
            raise ValueError("F8R20_FIT_FACTORIAL_COUNT")
        counts = {(a, b, q): 0 for a in A_SIGNS for b in B_SIGNS for q in (0, 1)}
        for ep in eps:
            counts[(ep["z_a"], ep["z_b"], ep["factual_return"])] += 1
            if ep["action"].shape != (10, 16):
                raise ValueError("F8R20_FIT_TENSOR_SHAPE")
        if set(counts.values()) != {GROUPS["fit"][1]}:
            raise ValueError("F8R20_FIT_MISSING_FACTORIAL_CELL")
        return {"coverage_per_A_B_return_cell": min(counts.values()),
                "factorial_cells": len(counts), "episodes": len(eps)}
    if len(eps) != 4 * GROUPS[split][1]:
        raise ValueError("F8R20_SCORE_PANEL_COUNT")
    for i in range(0, len(eps), 2):
        neg, pos = eps[i:i+2]
        if (neg["group"] != pos["group"] or neg["z_a"] != -1
                or pos["z_a"] != 1 or neg["z_b"] != pos["z_b"]
                or not torch.equal(neg["action"][1:], pos["action"][1:])
                or not torch.equal(neg["effect"][1:], pos["effect"][1:])
                or not torch.equal(neg["action"], pos["action"])
                or torch.equal(neg["effect"][0], pos["effect"][0])):
            raise ValueError("F8R20_B_PAIR_OR_PROBE_IDENTIFIABILITY_DRIFT")
        if neg["action"].shape != (1 + GROUPS[split][2], 16):
            raise ValueError("F8R20_PANEL_HORIZON_DRIFT")
    return {"identical_B_paired": True, "groups": GROUPS[split][1],
            "score_episodes": len(eps)}


def regret(z_a: int, goal: float, act: int) -> float:
    if (type(z_a) is not int or z_a not in A_SIGNS
            or type(goal) is not float or goal not in GOALS
            or type(act) is not int or act not in (0, 1)):
        raise ValueError("F8R20_BAD_EVALUATOR_ACTION")
    effect = r3.synthetic_effect(0., act, z_a)
    ref = min((r3.synthetic_effect(0., a, z_a) - goal) ** 2 for a in (0, 1))
    return max(0., (effect-goal)**2 - ref)


def analytic_controls(eps: list[dict], split: str) -> dict:
    verify_matched(eps, split)
    by_group = {name: [] for name in
                ("always0", "always1", "goal_positive", "goal_negative", "A_probe_oracle")}
    for group in range(GROUPS[split][1]):
        members = [ep for ep in eps if ep["group"] == group]
        for name in by_group:
            outcomes = []
            for ep in members:
                # This inverse normalization is evaluator-only: a positive
                # first-probe delta means z_A=-1, not fed to a learned arm.
                raw_delta = 2.4 * math.atanh(float(ep["effect"][0, 0]))
                remembered = 1 if raw_delta < -.1 else -1
                if remembered != ep["z_a"]:
                    raise ValueError("F8R20_PROBE_SIGN_NOT_IDENTIFIED")
                for goal in GOALS:
                    actions = {
                        "always0": 0, "always1": 1,
                        "goal_positive": int(goal > 0),
                        "goal_negative": int(goal < 0),
                    }
                    if name == "A_probe_oracle":
                        act = min((0, 1), key=lambda a: (
                            (r3.synthetic_effect(0., a, remembered) - goal)**2, a))
                    else:
                        act = actions[name]
                    outcomes.append(regret(ep["z_a"], goal, act))
            by_group[name].append(float(np.mean(outcomes)))
    means = {k: float(np.mean(v)) for k, v in by_group.items()}
    if (not math.isclose(means["always1"], .015, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(means["always0"], .105, rel_tol=0, abs_tol=1e-12)
            or any(x != 0. for x in by_group["A_probe_oracle"])):
        raise ValueError("F8R20_NO_IDENTIFIABILITY_GAP")
    return {"means": means, "pergroup": by_group,
            "best_history_free": min(
                ("always0", "always1", "goal_positive", "goal_negative"),
                key=lambda name: (means[name], name)),
            "oracle_early_A": means["A_probe_oracle"]}


def _predict_actions(arm: str, model, eps: list[dict], erase_A=False):
    keys = [torch.tensor(r18.r5._action_key((0.,0.,0.,0.),a),
                         dtype=torch.float32) for a in (0,1)]
    actions, effects = [], []
    for ep in eps:
        for act in (0,1):
            aa = torch.cat((ep["action"], keys[act].unsqueeze(0)))
            ee = torch.cat((ep["effect"], torch.zeros((1,16),dtype=torch.float32)))
            if erase_A:
                ee[0] = 0.
            actions.append(aa)
            effects.append(ee)
    model.eval()
    with torch.no_grad():
        pred = r19._pred(MAPPING[arm], model, {
            "action":torch.stack(actions),
            "effect":torch.stack(effects)})
    expected = (2*len(eps), 2+GROUPS[eps[0]["split"]][2], 16)
    if pred.shape != expected or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R20_BAD_PREACTION_PREDICTION")
    arr = pred[:, -1, 0].detach().cpu().numpy().astype(np.float64)
    return np.asarray([
        2.4 * math.atanh(max(-.98, min(.98, float(t)))) for t in arr
    ]).reshape(len(eps),2)


def model_score(arm: str, model, eps: list[dict], split: str) -> dict:
    verify_matched(eps, split)
    t0 = time.perf_counter()
    normal = _predict_actions(arm, model, eps)
    erased = _predict_actions(arm, model, eps, erase_A=True)
    costs, masked, choices, masked_choices = [], [], [], []
    n_contrast = 0
    for i, ep in enumerate(eps):
        for goal in GOALS:
            action = min((0,1), key=lambda a: ((normal[i,a]-goal)**2,a))
            erased_action = min((0,1), key=lambda a: ((erased[i,a]-goal)**2,a))
            costs.append(regret(ep["z_a"], goal, action))
            masked.append(regret(ep["z_a"], goal, erased_action))
            choices.append(action)
            masked_choices.append(erased_action)
    # episode order: per group, B-, A-/A+, B+, A-/A+
    for g in range(GROUPS[split][1]):
        for b in range(2):
            for goal_idx in range(2):
                first = 2 * (4*g + 2*b) + goal_idx
                second = first + 2
                n_contrast += int(choices[first] != choices[second])
    grouped = [float(np.mean(costs[8*g:8*g+8]))
               for g in range(GROUPS[split][1])]
    paired_erased = [float(np.mean(masked[8*g:8*g+8]))
                     for g in range(GROUPS[split][1])]
    return {
        "mean_regret":float(np.mean(grouped)),
        "mean_probe_erased_regret":float(np.mean(paired_erased)),
        "pergroup_regret":grouped,"pergroup_probe_erased":paired_erased,
        "model_selected_actions":len(choices),
        "two_action_queries":2*len(eps),
        "two_action_queries_probe_erased":2*len(eps),
        "A_sign_action_contrast":n_contrast,
        "A_sign_action_contrast_opportunities":4*GROUPS[split][1],
        "chosen_actions":choices,"probe_erased_chosen_actions":masked_choices,
        "inference_seconds":time.perf_counter()-t0
    }


def fit_one(arm: str, seed: int, fit: list[dict], dev: list[dict],
            id_score: list[dict], ood_score: list[dict],
            sampler: list[list[int]]) -> dict:
    model=r19._model(MAPPING[arm],seed).cpu()
    initial=r18.r5.f5.hash_weights(model)
    opt=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("native_") else .001, weight_decay=.01)
    start=time.perf_counter()
    diagnostics=[]
    losses=[]
    for update, indices in enumerate(sampler, start=1):
        batch, valid=r18.r5.f4._padded([factual(fit[i]) for i in indices])
        if valid.shape != (BATCH,10) or not bool(valid[:,-1].all()):
            raise ValueError("F8R20_MALFORMED_PREQUENTIAL_FIT_BATCH")
        model.train()
        opt.zero_grad(set_to_none=True)
        if arm in ("native_uniform", "native_return_focus"):
            loss=r19._objective(MAPPING[arm],model,batch,valid)
        else:
            pred=r19._pred(MAPPING[arm],model,batch)
            target=batch["effect"][:,-1,:4]
            loss=(pred[:,-1,:4]-target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R20_NONFINITE_TRAIN_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool((g!=0).any()) for g in gradients)
                or any(g is not None and not bool(torch.isfinite(g).all())
                       for g in gradients)):
            raise ValueError("F8R20_INVALID_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if update in CHECKPOINTS:
            d=model_score(arm,model,dev,"dev")
            diagnostics.append({"updates":update,
                                "fit_loss_last32":float(np.mean(losses[-32:])),
                                "dev_mean_regret":d["mean_regret"],
                                "dev_A_sign_action_contrast":d["A_sign_action_contrast"]})
    elapsed=time.perf_counter()-start
    trained=r18.r5.f5.hash_weights(model)
    if trained==initial:
        raise ValueError("F8R20_UNTRAINED_CHECKPOINT")
    score_id=model_score(arm,model,id_score,"id")
    score_ood=model_score(arm,model,ood_score,"ood")
    restored=r19._model(MAPPING[arm],seed).cpu()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    restored.eval()
    cold_id=model_score(arm,restored,id_score,"id")
    cold_ood=model_score(arm,restored,ood_score,"ood")
    if (r18.r5.f5.hash_weights(restored)!=trained
            or score_id["chosen_actions"]!=cold_id["chosen_actions"]
            or score_ood["chosen_actions"]!=cold_ood["chosen_actions"]
            or any(abs(a-b)>1e-12 for sc, cold in
                   ((score_id,cold_id),(score_ood,cold_ood)) for a,b in
                   zip(sc["pergroup_regret"],cold["pergroup_regret"]))):
        raise ValueError("F8R20_COLD_RESTORE_DIVERGENCE")
    return {"arm":arm,"seed":seed,"updates":UPDATES,
            "parameters":sum(p.numel() for p in model.parameters()),
            "first32_fit_loss":float(np.mean(losses[:32])),
            "last32_fit_loss":float(np.mean(losses[-32:])),
            "diagnostic_dev_learning_curve":diagnostics,
            "original_weights_sha256":initial,"trained_weights_sha256":trained,
            "train_seconds":elapsed,
            "cold_restored_exact_action_parity":True,
            "id":score_id,"ood":score_ood}


def _decisions(rows: list[dict], controls: dict, seed: int) -> dict:
    classical=[r for r in rows if r["arm"] in ("linear_return_only","mlp_return_only","ssm_return_only")]
    chosen=min(classical,key=lambda r:(
        r["diagnostic_dev_learning_curve"][-1]["dev_mean_regret"],r["arm"],r["seed"]))
    fixed=controls["best_history_free"]
    all_wins=[]
    draws=np.random.default_rng(BOOT_SEED).integers(0,12,size=(BOOT,12))
    for row in rows:
        if row["arm"] != "native_return_focus":
            continue
        metrics={}
        both=True
        for split in ("id","ood"):
            reference=(chosen[split]["pergroup_regret"]
                       if chosen[split]["mean_regret"] < controls["reference"][split]["means"][fixed]
                       else controls["reference"][split]["pergroup"][fixed])
            base=float(np.mean(reference))
            c=row[split]
            cost=float(c["mean_regret"])
            gain=np.asarray(reference)-np.asarray(c["pergroup_regret"])
            lower=float(np.percentile(gain[draws].mean(axis=1),2.5))
            erased=float(c["mean_probe_erased_regret"])-cost
            win=bool(base>1e-12 and cost <= .9*base
                     and lower>0 and erased>=.003 and c["A_sign_action_contrast"]>0)
            both=both and win
            metrics[split]={"candidate_regret":cost,
                            "reference_regret":base,"paired_bootstrap_2p5_gain":lower,
                            "probe_erasure_regret_increase":erased,
                            "A_sign_action_contrast":c["A_sign_action_contrast"],
                            "passes_fixed_DEV_threshold":win}
        all_wins.append({"seed":row["seed"],"two_horizon_win":both,"details":metrics})
    return {"dev_selected_conventional":{"arm":chosen["arm"],"seed":chosen["seed"],
                "dev_regret":chosen["diagnostic_dev_learning_curve"][-1]["dev_mean_regret"]},
            "candidate_seeds":all_wins,
            "candidate_dev_gate_pass":sum(x["two_horizon_win"] for x in all_wins)>=2}


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,dev,ids,ood=(cohort(key) for key in ("fit","dev","id","ood"))
    identity={name:verify_matched(panel,name)
              for name,panel in (("fit",fit),("dev",dev),("id",ids),("ood",ood))}
    controls={name:analytic_controls(panel,name)
              for name,panel in (("id",ids),("ood",ood))}
    g=torch.Generator(device="cpu")
    g.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=g).tolist()
    start=time.perf_counter()
    rows=[fit_one(arm,seed,fit,dev,ids,ood,batches)
          for arm in ARMS for seed in SEEDS]
    decision=_decisions(rows,{"reference":controls,
                         "best_history_free":controls["id"]["best_history_free"]},985299)
    return {"schema":SCHEMA,"classification":
            "SOURCE_NEUTRAL_AUTHOR_GENERATED_DEVELOPMENT_ONLY",
            "freeze_blob":FREEZE_BLOB,"source_pin":R18_BLOB,
            "factual_FIT_sessions":len(fit),
            "factual_FIT_effects":sum(len(e["effect"]) for e in fit),
            "synthetic_score_observed_B_and_probe_effects":
              sum(len(e["effect"]) for e in ids+ood),
            "training_optimizer_updates":len(rows)*UPDATES,
            "synthetic_model_selected_decisions":len(rows)*len(ids)*len(GOALS)*2,
            "model_count":len(rows),"models":rows,"identity_checks":identity,
            "analytical_controls":controls,"decision":decision,
            "unfamiliar_B_length_tested":12,"trained_B_length":8,
            "original_external_source_effects":0,"F7_replayed":False,
            "conventional_independent_convergence":False,
            "independent_scientific_pass":False,"deployment":"NONE",
            "development_train_and_score_seconds":time.perf_counter()-start}


def retain_ci_receipt(result: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path=ROOT/"local_runs/p85mr_f8r20_factorial_transfer_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R20_NO_RESULT_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_source_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R20_ORIGINAL_SCIENTIFIC_FIRSTUSE_NOT_AUTHORIZED")
