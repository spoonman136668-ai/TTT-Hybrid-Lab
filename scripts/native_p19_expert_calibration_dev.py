"""P19: teacher-supervised capability calibration on factual MIT CartPole, DEV ONLY.

This is an independent diagnostic after frozen P18 failure, NOT a continuation
of P18 source/weights, an RL-only comparison, sealed science, or FORGE activation.
Teacher targets are deliberately labelled expert supervision. During evaluation
all policies act from their own public sensor/history with no teacher access.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ttt-hybrid.native-p19.expert-calibration-dev.v1"
TRAIN_SEEDS = tuple(range(73000, 73048))
VAL_SEEDS = tuple(range(74000, 74008))
EVAL_SEEDS = (75011,75012,75013,75014,75015,75016,75017,75018)
MODEL_SEEDS = (2741,2753,2767)
KINDS = ("history_gru","history_mlp","linear_observation")
ARMS = ("history_gru","history_mlp","linear_observation",
        "untrained_gru","reversed_history_gru","teacher","seeded_random")
MAX_STEPS = 200
EPOCHS = 32
BATCH = 512
LR = 0.008
MAX_SECONDS = 240


def require_dev():
    if os.environ.get("NATIVE_P19_DEV_ONLY", "0") != "1":
        raise RuntimeError("P19_SCIENCE_FIRST_USE_NOT_AUTHORIZED")


def frozen():
    obj = json.loads((ROOT/"experiments/native_p19_expert_calibration_frozen.json").read_text())
    if (obj["schema"] != "ttt-hybrid.native-p19.expert-calibration-frozen.v1"
        or obj["science"]["authorized"] or obj["science"]["first_use_consumed"]
        or obj["source"]["S29_original_owner_sha"] != p18.S29_SHA
        or tuple(obj["design"]["evaluation_reset_seeds"]) != EVAL_SEEDS
        or tuple(obj["design"]["train_initializations"]) != MODEL_SEEDS
        or obj["design"]["batch_size"] != BATCH
        or obj["design"]["epochs"] != EPOCHS
        or obj["design"]["lr"] != LR):
        raise RuntimeError("P19_FROZEN_PROTOCOL_DRIFT")
    return obj


class LinearObservation(nn.Module):
    """A learned, 10-parameter, public-current-sensor conventional control."""
    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(4, 2)
    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (8,6):
            raise ValueError("P19_LINEAR_INPUT_SHAPE")
        return self.actor(x[:, -1, :4]), None


def new_model(kind):
    if kind == "history_gru":
        return p18.HistoryGRU()
    if kind == "history_mlp":
        return p18.FullHistoryMLP()
    if kind == "linear_observation":
        return LinearObservation()
    raise ValueError("P19_UNKNOWN_MODEL")


def expert_dataset(gym, seeds):
    """Collect actual source-action trajectories. Only present sensor labels."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("P19_BAD_DEMONSTRATION_SEEDS")
    inputs, labels = [], []
    digest = hashlib.sha256()
    total_steps, total_rewards = 0, 0.0
    for seed in seeds:
        env = gym.make("CartPole-v1", render_mode=None)
        try:
            obs, _ = env.reset(seed=seed)
            hist = [p18.factual_row(obs,0,0.)]
            for _ in range(MAX_STEPS):
                x = p18.history_tensor(hist)[0]
                action = p18.handcoded_action(obs)
                inputs.append(x)
                labels.append(action)
                nxt, reward, terminated, truncated, _ = env.step(action)
                reward = float(reward)
                next_row = p18.factual_row(nxt,2*action-1,reward)
                digest.update(json.dumps(
                    [seed, hist[-1], action, next_row, bool(terminated), bool(truncated)],
                    separators=(",",":"),allow_nan=False).encode())
                hist.append(next_row)
                obs = nxt
                total_steps += 1
                total_rewards += reward
                if terminated or truncated:
                    break
        finally:
            env.close()
    return (torch.stack(inputs), torch.tensor(labels,dtype=torch.long),
            {"actual_teacher_actions":total_steps,
             "actual_teacher_rewards":total_rewards,
             "factual_action_outcome_sha256":digest.hexdigest(),
             "world_reset_seeds":list(seeds),
             "teacher_labels_from_present_public_sensor_only":True})


@torch.no_grad()
def accuracy(net, x, y):
    net.eval()
    correct = 0
    for st in range(0,len(y),BATCH):
        logits, _ = net(x[st:st+BATCH])
        correct += int((logits.argmax(dim=1)==y[st:st+BATCH]).sum())
    return correct/len(y)


def fit(kind, seed, train_x, train_y, val_x, val_y, epochs=EPOCHS):
    if kind not in KINDS or seed not in MODEL_SEEDS or not 1<=epochs<=EPOCHS:
        raise ValueError("P19_UNREGISTERED_FIT")
    torch.manual_seed(seed + (KINDS.index(kind) * 50000))
    net = new_model(kind)
    before = p18.model_sha(net)
    original = copy.deepcopy(net)
    opt = torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.)
    generator = torch.Generator().manual_seed(seed+710000)
    trace = []
    updates = 0
    for epoch in range(epochs):
        net.train()
        permutation = torch.randperm(len(train_y),generator=generator)
        weighted_loss = 0.0
        for st in range(0,len(permutation),BATCH):
            idx = permutation[st:st+BATCH]
            logits, _ = net(train_x[idx])
            loss = F.cross_entropy(logits,train_y[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P19_NONFINITE_SUPERVISED_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
            weighted_loss += float(loss.detach())*len(idx)
            updates += 1
        val_accuracy = accuracy(net,val_x,val_y)
        trace.append({"epoch":epoch+1,
                      "mean_train_cross_entropy":weighted_loss/len(train_y),
                      "heldout_teacher_agreement":val_accuracy})
    net.eval()
    store = io.BytesIO()
    torch.save(net.state_dict(),store)
    store.seek(0)
    torch.manual_seed(seed+999999)
    restored = new_model(kind)
    restored.load_state_dict(torch.load(store,weights_only=True,map_location="cpu"),
                             strict=True)
    restored.eval()
    sample = val_x[:min(BATCH,len(val_x))]
    with torch.no_grad():
        before_logits,_ = net(sample)
        after_logits,_ = restored(sample)
    exact = bool(torch.equal(before_logits,after_logits))
    return {"trained":restored,"untrained":original},{
        "kind":kind,"initialization":seed,
        "trained_model_original_sha256":before,
        "trained_model_final_sha256":p18.model_sha(restored),
        "trained_weight_updated":before!=p18.model_sha(restored),
        "checkpoint_bytes_sha256":hashlib.sha256(store.getvalue()).hexdigest(),
        "fresh_object_state_only_restore_exact":exact,
        "real_process_death_cold_restore_proven":False,
        "optimizer_updates":updates,
        "epochs_completed":epochs,
        "validation_agreement":trace[-1]["heldout_teacher_agreement"],
        "epoch_train_and_heldout_curves":trace,
        "teacher_supervision_not_reward_only":True}


@torch.no_grad()
def eval_episode(gym, arm, models, world_seed, model_seed):
    if arm not in ARMS or world_seed not in EVAL_SEEDS or model_seed not in MODEL_SEEDS:
        raise ValueError("P19_UNREGISTERED_EVALUATION")
    env = gym.make("CartPole-v1",render_mode=None)
    trace = hashlib.sha256()
    try:
        obs,_ = env.reset(seed=world_seed)
        initial = p18.factual_row(obs,0,0.)
        hist = [initial]
        actions, rewards = [], []
        rng = random.Random(world_seed+model_seed*31+97)
        for _ in range(MAX_STEPS):
            if arm == "teacher":
                act = p18.handcoded_action(obs)
            elif arm == "seeded_random":
                act = rng.randrange(2)
            else:
                net = models["history_gru"] if arm=="reversed_history_gru" else models[arm]
                xhist = hist
                if arm == "reversed_history_gru":
                    window = hist[-8:]
                    xhist = list(reversed(window[:-1])) + window[-1:]
                logits,_ = net(p18.history_tensor(xhist))
                act = int(logits.argmax(dim=-1).item())
            nxt, reward, terminated, truncated,_ = env.step(act)
            reward = float(reward)
            nextrow = p18.factual_row(nxt,2*act-1,reward)
            trace.update(json.dumps(
                [hist[-1],act,nextrow,bool(terminated),bool(truncated)],
                separators=(",",":"),allow_nan=False).encode())
            actions.append(act)
            rewards.append(reward)
            hist.append(nextrow)
            obs = nxt
            if terminated or truncated:
                break
        return {"real_episode_reward":sum(rewards),"real_episode_actions":len(actions),
                "actual_action_choices":actions,"trace_sha256":trace.hexdigest(),
                "initial_observation":initial,"real_source_step_only":True,
                "eval_optimizer_updates":0}
    finally:
        env.close()


def run(s29_root, gym_root, cloud_root):
    require_dev()
    frozen_protocol=frozen()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    gym, identity = p18.admit_gymnasium(s29_root,gym_root,cloud_root)
    tx,ty,training = expert_dataset(gym,TRAIN_SEEDS)
    vx,vy,validation = expert_dataset(gym,VAL_SEEDS)
    if set(TRAIN_SEEDS)&set(VAL_SEEDS) or set(TRAIN_SEEDS)&set(EVAL_SEEDS) or set(VAL_SEEDS)&set(EVAL_SEEDS):
        raise RuntimeError("P19_WORLD_SPLIT_LEAK")
    per_init=[]
    for seed in MODEL_SEEDS:
        trained={}
        fit_receipts={}
        for kind in KINDS:
            trained[kind],fit_receipts[kind]=fit(kind,seed,tx,ty,vx,vy)
        models={k:trained[k]["trained"] for k in KINDS}
        models["untrained_gru"]=trained["history_gru"]["untrained"]
        hashes_before={k:p18.model_sha(m) for k,m in models.items()}
        evals = []
        for world in EVAL_SEEDS:
            arms={arm:eval_episode(gym,arm,models,world,seed) for arm in ARMS}
            initial={tuple(v["initial_observation"]) for v in arms.values()}
            if len(initial)!=1:
                raise RuntimeError("P19_SOURCE_RESET_MISMATCH")
            evals.append({"reset_seed":world,"arms":arms})
        hashes_after={k:p18.model_sha(m) for k,m in models.items()}
        means={arm:sum(q["arms"][arm]["real_episode_reward"] for q in evals)/len(evals)
               for arm in ARMS}
        checks={
            "all_actor_weights_updated":all(r["trained_weight_updated"] for r in fit_receipts.values()),
            "all_in_process_fresh_restores_exact":all(r["fresh_object_state_only_restore_exact"] for r in fit_receipts.values()),
            "all_model_weights_immutable_during_eval":hashes_before==hashes_after,
            "all_evaluation_real_source_steps":all(
                1<=v["real_episode_actions"]<=MAX_STEPS and v["eval_optimizer_updates"]==0
                for case in evals for v in case["arms"].values()),
            "same_factual_demonstration_supervision":True}
        per_init.append({
            "model_seed":seed,
            "engineering_integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,
            "model_fits":fit_receipts,
            "mean_real_episode_reward":means,
            "linear_baseline_teacher_agreement":fit_receipts["linear_observation"]["validation_agreement"],
            "linear_learned_baseline_competent":
                means["linear_observation"]>=180 and
                fit_receipts["linear_observation"]["validation_agreement"]>=.98,
            "real_environment_evaluations":evals})
    wall=time.perf_counter()-t0
    engineering=(all(r["engineering_integrity"]=="PASS" for r in per_init)
                 and wall<=MAX_SECONDS)
    competent=sum(bool(r["linear_learned_baseline_competent"]) for r in per_init)
    return {
        "schema":SCHEMA,
        "status":"PASS_DEV_ENGINEERING_NOT_SCIENCE" if engineering else "FAIL_DEV_ENGINEERING",
        "real_external_mit_cartpole_run":True,
        "frozen_protocol":frozen_protocol["schema"],
        "source_identity":identity,
        "parent_P18_failed_trial_unchanged":True,
        "original_source_S29_modified":False,
        "scientific_first_use":"NOT_RUN",
        "protected_scientific_worlds_consumed":[],
        "teacher_labels_used_for_supervised_training":True,
        "reward_only_competence_proven":False,
        "match_P18_reward_information_budget":False,
        "actual_teacher_demonstration_train":training,
        "actual_teacher_demonstration_validation":validation,
        "training_worlds":list(TRAIN_SEEDS),
        "validation_worlds":list(VAL_SEEDS),
        "evaluation_worlds":list(EVAL_SEEDS),
        "three_initialization_learned_linear_competence_support":competent,
        "all_three_learned_linear_competence":competent==len(MODEL_SEEDS),
        "per_initialization":per_init,
        "elapsed_whole_experiment_seconds":wall,
        "full_accepted_science_or_promotion":False,
        "FORGE_research_loop_activated":False,
        "limitations":"Expert-derived labels use stronger supervision than P18 reward-only GRU. This is an architecture/control-capability calibration on known public MIT CartPole, not a causal memory, novel-world, independent transfer, plateau, or scientific superiority proof. In-process state tensors restored, true crash-cold retention unverified; original S26/S27 untouched."}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s29-root",required=True,type=Path)
    ap.add_argument("--source-root",required=True,type=Path)
    ap.add_argument("--cloudpickle-root",required=True,type=Path)
    ap.add_argument("--receipt",required=True,type=Path)
    args=ap.parse_args()
    report=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":report["status"],
        "linear_competence_support":report["three_initialization_learned_linear_competence_support"],
        "mean_rewards":[x["mean_real_episode_reward"] for x in report["per_initialization"]],
        "elapsed":report["elapsed_whole_experiment_seconds"],
        "science":report["scientific_first_use"]},sort_keys=True))
    return 0 if report["status"]=="PASS_DEV_ENGINEERING_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
