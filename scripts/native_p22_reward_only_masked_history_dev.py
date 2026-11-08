"""P22: preregistered reward-only velocity-blind historical control DEV study.

The frozen protocol is a separate earlier Git commit. Reuse P20's unchanged
PPO and external source admission, not new learning/optimizer machinery.
Source only, public DEV: no sealed science first use, model promotion or FORGE.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import native_p20_reward_only_ppo_dev as p20

PROTOCOL = ROOT / "experiments" / "native_p22_reward_only_masked_history_frozen.json"
SCHEMA = "ttt-hybrid.native-p22.reward-only-masked-history-dev.v1"
SEEDS = (3359, 3371, 3383)
EVAL_SEEDS = (94011, 94017, 94023, 94029, 94035, 94041, 94047, 94053)
CONDITIONS = ("original", "short_pole_x0.7", "long_pole_x1.3")
VIEWS = {
    "masked_full_history_reward_GRU": ("reward_gru", "masked_full"),
    "masked_current_only_reward_GRU": ("reward_gru", "masked_current"),
    "masked_two_frame_reward_GRU": ("reward_gru", "masked_two"),
    "masked_wrong_actor_credit_GRU": ("wrong_credit_gru", "masked_full"),
    "unmasked_full_history_reward_GRU": ("reward_gru", "full_4d"),
}
OTHER = ("masked_finite_difference", "seeded_random")
MAX_SECONDS = 600

def frozen():
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (value.get("schema") != "ttt-hybrid.native-p22.reward-only-masked-history-frozen.v1"
        or value["status"] != "FROZEN_PROTOCOL_ONLY_NO_IMPLEMENTATION_OR_FIRST_USE"
        or value["source"]["scientific_first_use"] is not False
        or tuple(value["study"]["model_initializations"]) != SEEDS
        or tuple(value["study"]["eval_seeds"]) != EVAL_SEEDS
        or value["study"]["total_real_training_decisions"] != 737280
        or value["study"]["total_optimizer_minibatches"] != 8640):
        raise RuntimeError("P22_FROZEN_PROTOCOL_MISMATCH")
    return value

def require_dev():
    if os.environ.get("NATIVE_P22_DEV_ONLY", "1") != "1":
        raise RuntimeError("P22_SEALED_SCIENCE_FIRST_USE_DENIED")
    return frozen()

def tensor_for_view(history, view):
    if view not in {"masked_full", "masked_current", "masked_two", "full_4d"}:
        raise ValueError("P22_UNREGISTERED_OBSERVATION_VIEW")
    rows = [list(x) for x in history[-p20.HISTORY:]]
    if not rows or any(len(x) != 6 for x in rows):
        raise ValueError("P22_MALFORMED_FACTUAL_HISTORY")
    if view != "full_4d":
        for x in rows:
            x[1] = 0.0
            x[3] = 0.0
    out = p20.scaled_history(rows)
    if view == "masked_current":
        out[:, :-1, :] = 0
    elif view == "masked_two":
        out[:, :-2, :] = 0
    return out

@contextmanager
def isolated_ppo_view(view):
    # P20 is unmodified; its sequential CPU trainer calls its actor_tensor
    # for the current and next actual states. No asynchronous or parallel run.
    original_tensor = p20.actor_tensor
    original_seeds = p20.SEEDS
    original_reset = p20.TRAIN_SEED_BASE
    try:
        p20.actor_tensor = lambda history: tensor_for_view(history, view)
        p20.SEEDS = SEEDS
        p20.TRAIN_SEED_BASE = 84000
        yield
    finally:
        p20.actor_tensor = original_tensor
        p20.SEEDS = original_seeds
        p20.TRAIN_SEED_BASE = original_reset

def train_all(gym):
    models, training = {}, {}
    for seed in SEEDS:
        models[seed] = {}
        training[seed] = {}
        for name, (p20_arm, view) in VIEWS.items():
            with isolated_ppo_view(view):
                outcome, receipt = p20.train_ppo(gym, p20_arm, seed)
            if (receipt["actual_factual_environment_actions"] != 49152
                or receipt["actor_critic_training_update_batches"] != 576
                or not receipt["trained_weights_changed"]
                or not receipt["in_process_fresh_weights_exact_logits"]):
                raise RuntimeError("P22_TRAINING_CUSTODY_FAILED")
            models[seed][name] = outcome["trained"]
            training[seed][name] = receipt
    return models, training

def finite_difference(hist):
    theta = float(hist[-1][2])
    velocity = 0.0 if len(hist) == 1 else (
        theta - float(hist[-2][2])) / 0.02
    return int(theta + 0.5 * velocity >= 0.0)

def run_episode(gym, model, view, seed, condition, init, name):
    if seed not in EVAL_SEEDS or condition not in CONDITIONS or init not in SEEDS:
        raise ValueError("P22_UNREGISTERED_EVALUATION")
    env = gym.make("CartPole-v1", render_mode=None)
    try:
        dynamics = {"original":"original", "short_pole_x0.7":"short_pole",
                    "long_pole_x1.3":"long_pole"}[condition]
        p20.set_public_environment_dynamics(env, dynamics)
        obs, _ = env.reset(seed=seed)
        hist = [p20.factual_row(obs, 0, 0.)]
        first = list(hist[0])
        digest = hashlib.sha256()
        returns = []
        actions = []
        rng = random.Random(seed + 7919 * init)
        for _ in range(p20.EVAL_HORIZON):
            if name == "masked_finite_difference":
                action = finite_difference(hist)
            elif name == "seeded_random":
                action = rng.randrange(2)
            else:
                with torch.no_grad():
                    logits, _ = model(tensor_for_view(hist, view))
                    action = int(logits.argmax(dim=-1).item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            row = p20.factual_row(next_obs, 2 * action - 1, float(reward))
            digest.update(json.dumps([hist[-1], action, row,
                                      bool(terminated), bool(truncated)],
                                     allow_nan=False, separators=(",", ":")).encode())
            hist.append(row)
            actions.append(action)
            returns.append(float(reward))
            if terminated or truncated:
                break
        return {
            "initial_observation": first,
            "mean_reward": sum(returns),
            "actual_factual_steps": len(actions),
            "factual_actions": actions,
            "factual_step_reward_sha256": digest.hexdigest(),
            "online_eval_gradient_steps": 0,
            "velocity_masked_for_model": view != "full_4d",
        }
    finally:
        env.close()

def run(s29_root, source_root, cloudpickle_root):
    policy = require_dev()
    if (p20.ROLLOUTS, p20.NUM_ENVS, p20.STEPS_PER_ROLLOUT,
        p20.PPO_EPOCHS, p20.MINIBATCH) != (96, 8, 64, 3, 256):
        raise RuntimeError("P22_ORIGINAL_PPO_ALGORITHM_DRIFT")
    torch.set_num_threads(1)
    begin = time.perf_counter()
    gym, upstream = p20.admit_gymnasium(s29_root, source_root, cloudpickle_root)
    models, training = train_all(gym)
    all_reports = []
    for init in SEEDS:
        before = {name:p20.model_sha(model) for name,model in models[init].items()}
        worlds = []
        for condition in CONDITIONS:
            for seed in EVAL_SEEDS:
                observations = []
                results = {}
                for name in tuple(VIEWS) + OTHER:
                    model = models[init].get(name)
                    view = VIEWS[name][1] if name in VIEWS else "masked_full"
                    episode = run_episode(gym, model, view, seed, condition, init, name)
                    results[name] = episode
                    observations.append(tuple(episode["initial_observation"]))
                if len(set(observations)) != 1:
                    raise RuntimeError("P22_PAIRED_WORLD_INITIAL_OBSERVATION_MISMATCH")
                worlds.append({"condition":condition,"seed":seed,"arms":results})
        after = {name:p20.model_sha(model) for name,model in models[init].items()}
        if before != after:
            raise RuntimeError("P22_EVALUATION_UPDATED_TRAINED_WEIGHTS")
        means = {c:{name:sum(w["arms"][name]["mean_reward"]
                    for w in worlds if w["condition"] == c) / len(EVAL_SEEDS)
                    for name in tuple(VIEWS) + OTHER} for c in CONDITIONS}
        s = means["original"]
        fg = s["masked_full_history_reward_GRU"]
        gates = {
            "masked_GRU_at_least_150":fg >= 150,
            "over_trained_current_only_by_15":fg >= s["masked_current_only_reward_GRU"] + 15,
            "over_trained_two_frame_by_10":fg >= s["masked_two_frame_reward_GRU"] + 10,
            "over_wrong_reward_credit_by_40":fg >= s["masked_wrong_actor_credit_GRU"] + 40,
            "short_pole_at_least_100":
                means["short_pole_x0.7"]["masked_full_history_reward_GRU"] >= 100,
            "long_pole_at_least_100":
                means["long_pole_x1.3"]["masked_full_history_reward_GRU"] >= 100,
            "unmasked_calibration_at_least_150":
                s["unmasked_full_history_reward_GRU"] >= 150,
        }
        all_reports.append({"init":init,"means":means,"frozen_gates":gates,
                            "vs_strong_two_frame_finite_difference":
                                fg-s["masked_finite_difference"],
                            "training":training[init],
                            "eval_worlds":worlds,
                            "all_frozen_models_unchanged_during_eval":True})
    wall=time.perf_counter()-begin
    integrity = (len(all_reports)==3 and wall<=MAX_SECONDS and
        sum(r["training"][name]["actual_factual_environment_actions"]
            for r in all_reports for name in VIEWS) == 737280 and
        sum(r["training"][name]["actor_critic_training_update_batches"]
            for r in all_reports for name in VIEWS) == 8640)
    support = all(all(r["frozen_gates"].values()) for r in all_reports)
    return {
      "schema":SCHEMA,"engineering":"PASS" if integrity else "FAIL",
      "development_frozen_primary_support":support,
      "source":upstream,
      "preregistration_parent":policy["exact_parent"],
      "training_factual_steps":737280,
      "training_ppo_optimizer_minibatches":8640,
      "worlds_per_init":len(CONDITIONS)*len(EVAL_SEEDS),
      "original_science_first_use":"NOT_RUN",
      "novel_native_hybrid_advantage_proven":False,
      "strong_finite_difference_superiority_claim":False,
      "P20_original_results_reinterpreted":False,
      "other_research_branches_changed":False,
      "whole_process_seconds":wall,
      "records":all_reports,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",required=True,type=Path)
    parser.add_argument("--source-root",required=True,type=Path)
    parser.add_argument("--cloudpickle-root",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    args=parser.parse_args()
    result=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({
        "engineering":result["engineering"],
        "frozen_development_support":result["development_frozen_primary_support"],
        "science":result["original_science_first_use"],
        "wall":result["whole_process_seconds"],
        "summary":[{"seed":r["init"],"means":r["means"],
                    "rules":r["frozen_gates"]} for r in result["records"]]
    }, sort_keys=True))
    return 0 if result["engineering"]=="PASS" else 1

if __name__=="__main__":
    raise SystemExit(main())
