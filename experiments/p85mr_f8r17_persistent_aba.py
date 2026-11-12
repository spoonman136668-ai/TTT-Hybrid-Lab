"""F8-R17: actual native Hybrid training on SOURCE-NEUTRAL persistent A→B→A.

Only established author-generated F8-R3 mathematical recurrence is evaluated.
No CausalGym import, F7 original replay, sealed test/source, GPU or owner action.
All training sees only factual selected state/action and past physical effects.
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

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r17_persistent_aba_freeze.json"
FREEZE_BLOB = "8017634ac7f5da4f7092b4c86e86f0030c941ac7"
ARMS = ("bypass_hard", "bypass_surrogate", "linear7", "ssm256")
SEEDS = (985201, 985203, 985207)
SPLITS = {"fit": (19350000, 24, (4, 8)),
          "diagnostic": (19360000, 12, (4, 8)),
          "test": (19370000, 12, (3, 7))}
UPDATES, BATCH, BOOT, BOOT_SEED = 96, 8, 10000, 19379999
SCHEMA = "ttt-hybrid.f8r17-native-persistent-aba-development.v1"


def locked():
    raw = FREEZE.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F8R17_FROZEN_PROTOCOL_BLOB_DRIFT")
    f = json.loads(raw)
    m, w, s, a = f["models"], f["world"], f["outcome"], f["authority"]
    if (f["schema"] != SCHEMA
            or f["classification"] != "SOURCE_NEUTRAL_NATIVE_TRAINED_DEV_ONLY"
            or f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_AND_NUMERICAL_RESULTS"
            or f["parent"]["sha"] != "3373068410eaf604bba0e74bd0ae5c62930e3803"
            or f["parent"]["original_f8r3_world_blob"] != r3.locked()["source_parent"] if False else False):
        raise ValueError("F8R17_PROTOCOL_IDENTITY_DRIFT")
    if (f["parent"]["original_f8r3_world_blob"] !=
            "9a7077ece06dfbc685232bbc606b0c2dd33a7895"
            or f["parent"]["original_f8r5_native_policy_blob"] !=
            "9ac9fc17d6f82d6bf2dc7b05458939d5bdb6ea92"
            or m["arms"] != list(ARMS)
            or m["seeds"] != list(SEEDS)
            or m["max_updates_each"] != UPDATES
            or m["mini_batch_worlds"] != BATCH
            or m["total_model_fits"] != 12
            or m["total_optimizer_updates"] != 1152
            or w["steps_per_world"] != 12
            or any(w["worlds"][k] != v[1] or
                   w["seed_base"][k] != v[0] or
                   w["switch_times"][k] != list(v[2])
                   for k,v in SPLITS.items())
            or s["world_cluster_bootstrap"] != BOOT
            or s["bootstrap_seed"] != BOOT_SEED
            or s["score_worlds"] != 12
            or not all(value is False for value in a.values())):
        raise ValueError("F8R17_SOURCE_BUDGET_OR_AUTHORITY_DRIFT")
    r5.locked()
    r3.locked()
    return f


def sign_at(initial: int, step: int, switches: tuple[int, int]) -> int:
    if (type(initial) is not int or initial not in (-1, 1)
            or type(step) is not int or not 0 <= step < 12
            or type(switches) is not tuple or len(switches) != 2
            or not all(type(v) is int for v in switches)
            or not 0 < switches[0] < switches[1] < 12):
        raise ValueError("F8R17_INVALID_HIDDEN_EVALUATOR_REGIME")
    return initial if step < switches[0] or step >= switches[1] else -initial


def episode(split: str, index: int) -> dict:
    if split not in SPLITS or type(index) is not int or not 0 <= index < SPLITS[split][1]:
        raise ValueError("F8R17_UNREGISTERED_WORLD")
    base, _, switches = SPLITS[split]
    seed = base + index
    initial = 1 if index % 2 == 0 else -1
    state = .42 * math.sin(seed*.013) + .19 * math.cos(seed*.021)
    actions, effects, modes = [], [], []
    for step in range(12):
        regime = sign_at(initial, step, switches)
        modes.append(regime)
        natural = r3.behavior(state)
        action = (1-natural if step in (0, *switches)
                  else natural ^ int((seed+7*step)%3 == 0))
        factual = r3.synthetic_effect(state, action, regime)
        before, after = (state,0.,0.,0.), (factual,0.,0.,0.)
        actions.append(r5._action_key(before, action))
        effects.append(r5._observed_effect(
            before, after, 1.-min(1.,(factual-.2)**2),
            False, step == 11))
        state = factual
    # 'regimes' and 'initial_sign' belong only to the evaluator; training
    # receives exclusively the two tensors returned by factual().
    return {"action": torch.tensor(actions, dtype=torch.float32),
            "effect": torch.tensor(effects, dtype=torch.float32),
            "regimes": tuple(modes),
            "initial_sign": initial,
            "final_public_state": float(state),
            "switches": switches, "seed": seed}


def factual(ep: dict) -> dict:
    return {"action": ep["action"], "effect": ep["effect"]}


def cohort(split: str) -> list[dict]:
    locked()
    return [episode(split, i) for i in range(SPLITS[split][1])]


def predictions(arm: str, model, ep: dict, *, extra_delay=False) -> torch.Tensor:
    a = ep["action"].unsqueeze(0)
    y = ep["effect"].unsqueeze(0)
    if extra_delay:
        y = torch.cat((torch.zeros_like(y[:, :1]), y[:, :-1]), dim=1)
    with torch.no_grad():
        pred = r5.predict(arm, model, {"action":a, "effect":y})
    if pred.shape != (1, 12, 16) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R17_INVALID_PREQUENTIAL_PREDICTIONS")
    return pred[0]


def predict_action(arm: str, model, ep: dict, action: int) -> float:
    """One hypothetical model query with only historical factual effects."""
    if type(action) is not int or action not in (0,1):
        raise ValueError("F8R17_INVALID_HYPOTHETICAL_ACTION")
    s = ep["final_public_state"]
    key = r5._action_key((s,0.,0.,0.),action)
    actions = torch.cat((ep["action"], torch.tensor([key],dtype=torch.float32)),dim=0)
    effects = torch.cat((ep["effect"], torch.zeros((1,16),dtype=torch.float32)),dim=0)
    with torch.no_grad():
        pred = r5.predict(arm,model,
                 {"action":actions.unsqueeze(0),"effect":effects.unsqueeze(0)})
    if pred.shape != (1,13,16) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R17_INVALID_MODEL_ACTION_PREDICTION")
    delta = float(pred[0,-1,0])
    return s + 2.4 * math.atanh(max(-.98,min(.98,delta)))


def model_selected_toy_action(arm: str,model,ep: dict,world_index: int):
    if type(world_index) is not int or not 0 <= world_index < 12:
        raise ValueError("F8R17_UNREGISTERED_SCORE_WORLD")
    goal = .2 if world_index % 2 == 0 else -.2
    predictions = [predict_action(arm,model,ep,act) for act in (0,1)]
    chosen = min((0,1),key=lambda act: ((predictions[act]-goal)**2,act))
    # Only after selecting action is the hidden original A sign used to
    # evaluate the realized synthetic action and a non-training oracle.
    s, hidden = ep["final_public_state"], ep["initial_sign"]
    realized = r3.synthetic_effect(s,chosen,hidden)
    oracle = min((r3.synthetic_effect(s,a,hidden)-goal)**2 for a in (0,1))
    regret = max(0.,(realized-goal)**2-oracle)
    return int(chosen), float(regret)


def evaluate(arm: str, model, episodes: list[dict]) -> dict:
    model.eval()
    t0=time.perf_counter()
    losses, negatives, regrets, chosen = [], [], [], []
    phases = {"initial_A":[], "switch_B":[], "return_A":[]}
    for i,ep in enumerate(episodes):
        pred = predictions(arm,model,ep)
        bad = predictions(arm,model,ep,extra_delay=True)
        target=ep["effect"]
        err=(pred[:,:4]-target[:,:4]).square().mean(-1)
        neg=(bad[:,:4]-target[:,:4]).square().mean(-1)
        if not bool(torch.isfinite(err).all()) or not bool(torch.isfinite(neg).all()):
            raise ValueError("F8R17_NONFINITE_PHYSICAL_SCORE")
        first,second=ep["switches"]
        losses.append(float(err[1:].mean()))
        negatives.append(float(neg[1:].mean()))
        for name, part in (("initial_A",err[1:first]),
                           ("switch_B",err[first:second]),
                           ("return_A",err[second:])):
            if part.numel()==0: raise ValueError("F8R17_EMPTY_CAUSAL_PHASE")
            phases[name].append(float(part.mean()))
        action,regret=model_selected_toy_action(arm,model,ep,i)
        chosen.append(action); regrets.append(regret)
    elapsed=time.perf_counter()-t0
    return {"mean_physical_mse":float(np.mean(losses)),
            "perworld_mse":losses,
            "delayed_history_negative_mse":float(np.mean(negatives)),
            "phases":{k:float(np.mean(v)) for k,v in phases.items()},
            "perworld_action_regret":regrets,
            "mean_action_regret":float(np.mean(regrets)),
            "selected_actions":chosen,
            "model_selected_toy_actions":len(chosen),
            "hypothetical_model_queries":2*len(chosen),
            "inference_seconds":elapsed}


def train_one(arm: str, seed: int, fit: list[dict], dev: list[dict],
              test: list[dict], sampler: list[list[int]]) -> dict:
    model = r5.model_for(arm,seed).cpu()
    initial = r5.f5.hash_weights(model)
    opt = torch.optim.AdamW(model.parameters(),
          lr=.002 if arm.startswith("bypass_") else .001,weight_decay=.01)
    start=time.perf_counter()
    for indices in sampler:
        batch, valid = r5.f4._padded([factual(fit[i]) for i in indices])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = r5.objective(arm,model,batch,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R17_NONFINITE_NATIVE_TRAIN_LOSS")
        loss.backward()
        if not any(p.grad is not None and bool(torch.any(p.grad!=0))
                   for p in model.parameters()):
            raise ValueError("F8R17_UNTRAINED_MODEL_GRADIENTS")
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in model.parameters()):
            raise ValueError("F8R17_NONFINITE_MODEL_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    secs=time.perf_counter()-start
    trained = r5.f5.hash_weights(model)
    if initial == trained:
        raise ValueError("F8R17_WEIGHTS_DID_NOT_CHANGE")
    dev_score=evaluate(arm,model,dev)
    score=evaluate(arm,model,test)
    restored=r5.model_for(arm,seed).cpu()
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    model.eval();restored.eval()
    restored_score=evaluate(arm,restored,test)
    if (trained!=r5.f5.hash_weights(restored)
            or any(abs(a-b)>1e-10 for a,b in
                   zip(score["perworld_action_regret"],
                       restored_score["perworld_action_regret"]))
            or any(abs(a-b)>1e-10 for a,b in
                   zip(score["perworld_mse"],restored_score["perworld_mse"]))):
        raise ValueError("F8R17_COLD_RESTORE_RECOMPUTATION_DRIFT")
    return {"arm":arm, "seed":seed,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "optimizer_updates":UPDATES,
            "initial_weight_sha256":initial,"trained_weight_sha256":trained,
            "train_seconds":secs,
            "dev_mean_mse":dev_score["mean_physical_mse"],
            "score":score,"cold_restore_parity":True}


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,dev,test=cohort("fit"),cohort("diagnostic"),cohort("test")
    sampler=torch.Generator(device="cpu")
    sampler.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=sampler).tolist()
    start=time.perf_counter()
    rows=[train_one(arm,seed,fit,dev,test,batches)
          for arm in ARMS for seed in SEEDS]
    baselines=[r for r in rows if r["arm"] in ("linear7","ssm256")]
    chosen_baseline=min(baselines,key=lambda r:
                      (r["score"]["mean_action_regret"],r["arm"],r["seed"]))
    base=chosen_baseline["score"]["perworld_action_regret"]
    bootstrap=np.random.default_rng(BOOT_SEED).integers(0,12,size=(BOOT,12))
    wins=[]
    for row in rows:
        if row["arm"]!="bypass_surrogate":continue
        costs=row["score"]["perworld_action_regret"]
        gains=np.asarray(base)-np.asarray(costs)
        lower=float(np.percentile(gains[bootstrap].mean(axis=1),2.5))
        mean_gain=float(np.mean(gains))
        eligible=bool(chosen_baseline["score"]["mean_action_regret"]>1e-8)
        win=bool(eligible and
            row["score"]["mean_action_regret"] <=
            .90*chosen_baseline["score"]["mean_action_regret"]
            and lower>0)
        wins.append({"seed":row["seed"],"mean_absolute_gain":mean_gain,
             "bootstrap_2p5_absolute_gain":lower,
             "candidate_win":win,"conventional_regret_nonzero":eligible})
    native_gate=sum(x["candidate_win"] for x in wins)>=2
    return {"schema":SCHEMA,"classification":"SOURCE_NEUTRAL_NATIVE_TRAINED_DEV_ONLY",
            "frozen_blob":FREEZE_BLOB,"model_count":len(rows),
            "total_optimizer_updates":len(rows)*UPDATES,
            "synthetic_fit_selected_effects":len(fit)*12,
            "synthetic_dev_selected_effects":len(dev)*12,
            "synthetic_test_factual_effects":len(test)*12,
            "score_unfamiliar_switches":[3,7],
            "score_model_selected_synthetic_actions":len(rows)*len(test),
            "original_source_effects":0,"spent_F7_replayed":False,
            "trained_controls_convergence_independently_qualified":False,
            "scientific_pass":False,"deployment":"NONE",
            "best_conventional":{"arm":chosen_baseline["arm"],
                                 "seed":chosen_baseline["seed"],
                                 "mean_action_regret":chosen_baseline["score"]["mean_action_regret"]},
            "candidate_wins":wins,"candidate_native_dev_gate_pass":bool(native_gate),
            "models":rows,"total_seconds":time.perf_counter()-start}


def retain_ci_receipt(result: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":return
    path=ROOT/"local_runs/p85mr_f8r17_native_aba_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R17_REFUSE_OVERWRITE_EXISTING_RECEIPT")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_source_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R17_ORIGINAL_SCIENTIFIC_FIRSTUSE_NOT_AUTHORIZED")
