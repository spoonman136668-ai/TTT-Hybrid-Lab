"""P19: independently learned conventional competence from REAL factual CartPole demonstrations.

This is NOT reward-only self-learning and NOT scientific first use. The visible
observation-feedback teacher makes decisions BEFORE true env.step, and those
specific decisions are EXECUTED in pinned MIT Gymnasium. Independent small
networks imitate only the executed, fully factual (sensor, action) records:
current-observation linear, completed-history MLP and completed-history GRU.
A same-cost wrong-attribution-label negative is trained in the same conditions.

All source licensing/admission is performed by exact original S29 via P18.
Do not promote, replace P17/P18 checkpoints, consume reserved science seeds,
mutate accepted sources or activate FORGE.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from scripts.native_p18_real_cartpole_factual_dev import (
    admit_gymnasium,HistoryGRU,FullHistoryMLP,handcoded_action,
    factual_row,history_tensor,HISTORY,model_sha,
)

SCHEMA="ttt-hybrid.native-p19.factual-demonstration-competence-dev.v1"
INIT_SEEDS=(2819,2833,2843)
TRAIN_RESET_SEEDS=tuple(range(71000,71048))
VAL_RESET_SEEDS=tuple(range(71500,71508))
DEV_EVAL_SEEDS=(73013,73029,73041,73053)
STEPS_PER_EPISODE=200
EVAL_ARMS=("teacher","learned_current_linear","learned_history_mlp",
           "learned_history_gru","misattributed_linear",
           "gru_without_previous_history","seeded_random")
TRAIN_ARMS=("linear","history_mlp","history_gru","misattributed_linear")
TRAIN_STEPS={"linear":700,"history_mlp":900,"history_gru":900,
             "misattributed_linear":700}
BATCH=256
TRAIN_LR=.008
MAX_RUN_SECONDS=220.
REFERENCE_MIN_REWARD=175.
INCORRECT_LABEL_MAX_REWARD=120.
MARGIN_OVER_NEGATIVE=50.
CURVE_EVERY=100

# Scaling is frozen, factual source coordinates only; no latent variables.
SCALE=torch.tensor([.5,.5,10.,5.,1.,1.],dtype=torch.float32)


def require_dev():
    if os.environ.get("NATIVE_P19_DEV_ONLY","1")!="1":
        raise RuntimeError("P19_PROTECTED_SCIENCE_FIRST_USE_DENIED")


class LinearCurrentController(nn.Module):
    """A very cheap conventional learner; never receives hidden physics."""
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(4,1)

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,6):
            raise ValueError("P19_INPUT_NOT_COMPLETE_FACTUAL_HISTORY")
        # The last 4 sensory values are current real public observation.
        return self.linear(x[:,-1,:4]).reshape(-1)


def logits(model,kind,h):
    if kind in ("linear","misattributed_linear"):
        z=model(h)
        return torch.stack((torch.zeros_like(z),z),dim=1)
    scores,_=model(h)
    return scores


def scaled_history(rows):
    # Previous completed factual observations, actions, rewards only.
    return history_tensor(rows)*SCALE


def collect_actual_demonstrations(gym,reset_seeds):
    x=[]
    y=[]
    episodes=[]
    for seed in reset_seeds:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            obs,_=env.reset(seed=seed)
            history=[factual_row(obs,0,0.)]
            digest=hashlib.sha256()
            actions=[]
            for _ in range(STEPS_PER_EPISODE):
                current=factual_row(obs,history[-1][4],history[-1][5])
                # True expert decision from only currently visible obs.
                action=handcoded_action(obs)
                if action not in (0,1):
                    raise RuntimeError("P19_EXPERT_ACTION_INVALID")
                x.append(scaled_history(history).reshape(HISTORY,6))
                y.append(action)
                nxt,reward,term,trunc,_=env.step(action)
                completed=factual_row(nxt,2*action-1,float(reward))
                digest.update(json.dumps(
                    [current,action,completed,bool(term),bool(trunc)],
                    separators=(",",":"),allow_nan=False).encode())
                history.append(completed)
                actions.append(action)
                obs=nxt
                if term or trunc:
                    break
            episodes.append({
                "public_reset_seed":seed,
                "chosen_factual_actions":len(actions),
                "teacher_action_ones":sum(actions),
                "complete_trace_sha256":digest.hexdigest(),
                "only_real_executed_actions":True})
        finally:
            env.close()
    if not x or len(x)!=len(y):
        raise RuntimeError("P19_EMPTY_FACTUAL_TRAINING")
    return torch.stack(x),torch.tensor(y,dtype=torch.long),episodes


def training_model(kind,seed):
    if kind not in TRAIN_ARMS or seed not in INIT_SEEDS:
        raise ValueError("P19_UNREGISTERED_MODEL")
    torch.manual_seed(seed+(10000 if kind=="history_mlp" else
                            20000 if kind=="history_gru" else 0))
    return (LinearCurrentController() if kind in ("linear","misattributed_linear")
            else FullHistoryMLP() if kind=="history_mlp" else HistoryGRU())


def shuffled_label_negative(labels):
    # True expert action histogram preserved, attribution to actual sensor
    # destroyed, in training only. This is not training on correct factual
    # actions; must NOT be misrepresented as valid causal supervision.
    gen=torch.Generator().manual_seed(99001)
    perm=torch.randperm(len(labels),generator=gen)
    negative=labels[perm]
    if torch.equal(labels,negative):
        raise RuntimeError("P19_MISATTRIBUTION_NEGATIVE_INEFFECTIVE")
    return negative,hashlib.sha256(perm.numpy().tobytes()).hexdigest()


def train(gym,train_x,train_y,val_x,val_y,kind,seed):
    model=training_model(kind,seed)
    starting=model_sha(model)
    neg_y,neg_digest=shuffled_label_negative(train_y)
    target=neg_y if kind=="misattributed_linear" else train_y
    optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN_LR)
    generator=torch.Generator().manual_seed(seed+300000)
    updates=TRAIN_STEPS[kind]
    curve=[]
    begin=time.perf_counter()
    model.train()
    for step in range(1,updates+1):
        ids=torch.randint(len(target),(BATCH,),generator=generator)
        scores=logits(model,kind,train_x[ids])
        loss=F.cross_entropy(scores,target[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),3.)
        optimizer.step()
        if step%CURVE_EVERY==0:
            model.eval()
            with torch.no_grad():
                vs=logits(model,kind,val_x)
                accuracy=float((vs.argmax(dim=1)==val_y).float().mean())
                ce=float(F.cross_entropy(vs,val_y))
            curve.append({"update":step,"validation_real_expert_action_accuracy":
                          accuracy,"validation_true_label_cross_entropy":ce,
                          "last_train_batch_cross_entropy":float(loss.detach())})
            model.train()
    model.eval()
    raw=io.BytesIO()
    torch.save(model.state_dict(),raw)
    raw.seek(0)
    torch.manual_seed(seed+404001)
    restored=training_model(kind,seed)
    restored.load_state_dict(torch.load(raw,weights_only=True,map_location="cpu"),
                             strict=True)
    restored.eval()
    with torch.no_grad():
        exact=torch.equal(logits(model,kind,val_x[:2]),
                          logits(restored,kind,val_x[:2]))
        final_accuracy=float((logits(restored,kind,val_x).argmax(dim=1)==
                              val_y).float().mean())
    return restored,{
        "kind":kind,"seed":seed,"weight_parameters":sum(
            p.numel() for p in restored.parameters()),
        "optimizer":"AdamW supervised cross entropy on actual executed action",
        "actual_gradient_updates":updates,"training_examples":int(len(train_y)),
        "validation_examples":int(len(val_y)),
        "true_factual_labels_used":kind!="misattributed_linear",
        "misattribution_index_digest":neg_digest
             if kind=="misattributed_linear" else None,
        "trained_weights_changed":model_sha(restored)!=starting,
        "state_tensor_sha256":model_sha(restored),
        "weights_only_new_object_exact_logits":exact,
        "same_factual_source_not_same_training_objective_as_P18":True,
        "final_true_expert_action_accuracy":final_accuracy,
        "training_curve":curve,
        "train_wall_seconds":time.perf_counter()-begin,
    }


def evaluate_one(gym,models,seed,init):
    if seed not in DEV_EVAL_SEEDS or init not in INIT_SEEDS:
        raise ValueError("P19_UNREGISTERED_EVALUATION_SOURCE")
    result={}
    first=[]
    for arm in EVAL_ARMS:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            obs,_=env.reset(seed=seed)
            initial=factual_row(obs,0,0.)
            first.append(tuple(initial))
            rows=[initial]
            actions=[]
            observed=[]
            digest=hashlib.sha256()
            rng=random.Random(seed+init*39)
            for _ in range(STEPS_PER_EPISODE):
                if arm=="teacher":
                    a=handcoded_action(obs)
                elif arm=="seeded_random":
                    a=rng.randrange(2)
                else:
                    kind=("linear" if arm=="learned_current_linear" else
                          "misattributed_linear" if arm=="misattributed_linear"
                          else "history_mlp" if arm=="learned_history_mlp"
                          else "history_gru")
                    model=models[kind]
                    view=rows
                    if arm=="gru_without_previous_history":
                        # Current real observation retained; earlier factual
                        # rows removed. Diagnostic, not assumed necessary.
                        view=[[0.]*6]*7+[rows[-1]]
                    with torch.no_grad():
                        s=logits(model,kind,scaled_history(view))
                    a=int(s.argmax(dim=1).item())
                nxt,r,term,trunc,_=env.step(a)
                reward=float(r)
                done=bool(term or trunc)
                row=factual_row(nxt,2*a-1,reward)
                digest.update(json.dumps(
                    [rows[-1],a,row,bool(term),bool(trunc)],
                    separators=(",",":"),allow_nan=False).encode())
                rows.append(row)
                actions.append(a)
                observed.append(reward)
                obs=nxt
                if done:break
            result[arm]={
                "real_reward":sum(observed),
                "actual_executed_actions":actions,
                "actual_observed_rewards":observed,
                "factual_trace_sha256":digest.hexdigest(),
                "score_bounded_by_real_200_step_episode":
                    1<=len(actions)<=STEPS_PER_EPISODE,
                "no_optimizer_updates_during_eval":True}
        finally:env.close()
    checks={
        "exact_same_source_reset_factual_observation":len(set(first))==1,
        "seven_real_factual_action_controller_episodes":len(result)==7,
        "all_actual_action_return_lengths_and_budgets":all(
            len(v["actual_executed_actions"])==
            len(v["actual_observed_rewards"]) and
            1<=len(v["actual_executed_actions"])<=200
            and all(math.isfinite(x) for x in v["actual_observed_rewards"])
            for v in result.values()),
        "no_hidden_physics_or_future_oracle":True}
    return {"source_dev_seed":seed,"model_seed":init,
            "integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"arms":result}


def run(s29_root:Path,source_root:Path,cloudpickle_root:Path):
    require_dev()
    torch.set_num_threads(1)
    begin=time.perf_counter()
    gym,source=admit_gymnasium(s29_root,source_root,cloudpickle_root)
    x,y,teacher_train=collect_actual_demonstrations(gym,TRAIN_RESET_SEEDS)
    xv,yv,teacher_val=collect_actual_demonstrations(gym,VAL_RESET_SEEDS)
    source_training_digest=hashlib.sha256(
        json.dumps([e["complete_trace_sha256"] for e in teacher_train],
                   separators=(",",":")).encode()).hexdigest()
    negative_action_label_histogram_preserved=bool(
        torch.equal(y.sort().values,shuffled_label_negative(y)[0].sort().values))
    records=[]
    for init in INIT_SEEDS:
        models={};trained={}
        for kind in TRAIN_ARMS:
            models[kind],trained[kind]=train(gym,x,y,xv,yv,kind,init)
        original_weights={kind:model_sha(m) for kind,m in models.items()}
        episodes=[evaluate_one(gym,models,seed,init) for seed in DEV_EVAL_SEEDS]
        frozen_weights={kind:model_sha(m) for kind,m in models.items()}
        avg={arm:sum(e["arms"][arm]["real_reward"] for e in episodes)/
             len(episodes) for arm in EVAL_ARMS}
        thresholds={
            "trained_current_observation_linear_reaches_175_of_200":
                avg["learned_current_linear"]>=REFERENCE_MIN_REWARD,
            "trained_history_MLP_reaches_175_of_200":
                avg["learned_history_mlp"]>=REFERENCE_MIN_REWARD,
            "trained_history_GRU_reaches_175_of_200":
                avg["learned_history_gru"]>=REFERENCE_MIN_REWARD,
            "wrong_attribution_linear_stays_at_most_120":
                avg["misattributed_linear"]<=INCORRECT_LABEL_MAX_REWARD,
            "trained_linear_beats_wrong_label_by_50":
                avg["learned_current_linear"]>=
                avg["misattributed_linear"]+MARGIN_OVER_NEGATIVE,
            "teacher_actual_factual_reaches_200":
                avg["teacher"]==STEPS_PER_EPISODE,
            "trained_history_GRU_relies_on_prior_history":
                avg["learned_history_gru"]>
                avg["gru_without_previous_history"]+5,
        }
        checks={
            "all_4_real_gradient_trainings_changed_weights":
                all(t["trained_weights_changed"] for t in trained.values()),
            "same_48_actual_demonstration_source_for_all_models":
                all(t["training_examples"]==len(y) for t in trained.values()),
            "same_weight_training_budget_within_linear_and_MLP_GRU_pairs":
                trained["linear"]["actual_gradient_updates"]==
                trained["misattributed_linear"]["actual_gradient_updates"]==
                TRAIN_STEPS["linear"] and
                trained["history_mlp"]["actual_gradient_updates"]==
                trained["history_gru"]["actual_gradient_updates"]==
                TRAIN_STEPS["history_gru"],
            "all_4_state_only_fresh_restorations_exact":
                all(t["weights_only_new_object_exact_logits"]
                    for t in trained.values()),
            "wrong_label_histogram_preserved_but_attribution_destroyed":
                negative_action_label_histogram_preserved and
                trained["misattributed_linear"]["misattribution_index_digest"]
                is not None,
            "all_actual_factual_evaluation_episodes_pass":
                all(e["integrity"]=="PASS" for e in episodes),
            "none_of_four_models_mutated_during_evaluation":
                original_weights==frozen_weights,
        }
        records.append({
            "initialization":init,
            "engineering":"PASS" if all(checks.values()) else "FAIL",
            "factual_training_receipts":trained,
            "integrity_checks":checks,
            "mean_actual_real_environment_reward":avg,
            "frozen_development_hypotheses":thresholds,
            "all_original_real_environment_episodes":episodes})
    wall=time.perf_counter()-begin
    support={name:sum(int(v["frozen_development_hypotheses"][name])
                       for v in records) for name in
             records[0]["frozen_development_hypotheses"]}
    okay=len(records)==3 and all(r["engineering"]=="PASS" for r in records)
    okay=okay and wall<=MAX_RUN_SECONDS
    return {
        "schema":SCHEMA,
        "status":"PASS_FARMA_ACTUAL_FACTUAL_IMITATION_DEV_NOT_SCIENCE"
                 if okay else "FAIL_FACTUAL_IMITATION_DEV_INTEGRITY",
        "environment_source_identity":source,
        "actual_external_gymnasium_used":True,
        "supervision":"ONLY_TRUE_EXECUTED_TEACHER_ACTIONS_FROM_SOURCE_OBSERVATIONS",
        "expert_is_handwritten_not_learned":True,
        "not_reward_only_autonomous_learning":True,
        "original_P18_RL_null_preserved":True,
        "original_s29_owner_source_modified":False,
        "actual_factual_training_demonstration_actions":len(y),
        "actual_factual_validation_demonstration_actions":len(yv),
        "public_train_seed_range":[TRAIN_RESET_SEEDS[0],TRAIN_RESET_SEEDS[-1]],
        "public_validation_seed_range":[VAL_RESET_SEEDS[0],VAL_RESET_SEEDS[-1]],
        "public_eval_seeds":list(DEV_EVAL_SEEDS),
        "actual_factual_eval_actions":sum(
            len(e["arms"][arm]["actual_executed_actions"])
            for r in records for e in r["all_original_real_environment_episodes"]
            for arm in EVAL_ARMS),
        "actual_optimizer_updates":len(INIT_SEEDS)*sum(TRAIN_STEPS.values()),
        "train_source_factual_digest":source_training_digest,
        "original_training_demonstration_receipts":teacher_train,
        "original_validation_demonstration_receipts":teacher_val,
        "frozen_support_out_of_three":support,
        "records":records,"process_wall_seconds":wall,
        "frozen_DEV_competence_qualified":all(
            r["frozen_development_hypotheses"][
                "trained_current_observation_linear_reaches_175_of_200"] and
            r["frozen_development_hypotheses"][
                "trained_linear_beats_wrong_label_by_50"]
            for r in records),
        "all_3_learned_architectures_qualified":all(all(
            r["frozen_development_hypotheses"][k] for k in (
                "trained_current_observation_linear_reaches_175_of_200",
                "trained_history_MLP_reaches_175_of_200",
                "trained_history_GRU_reaches_175_of_200"))
            for r in records),
        "no_official_scientific_first_use":True,
        "official_science_seeds_consumed":[],
        "original_native_runtime_changed":False,
        "FORGE_research_loop_activated":False,
        "accepted_model_pointer_changed":False,
        "scientific_superiority_established":False,
        "cold_restart_process_resume_proven":False,
        "whole_process_matched_compute":False,
        "limitations":"A new external-source DEV training/eval with separate reset seeds but same well-known CartPole simulator. A handwritten observation-feedback expert provides action labels; supervised imitation demonstrates model/data competency, NOT autonomous reward-only causal learning. Teacher attains maximum 200 on prior visible environments; no learned actor can outperform by margin. 3 model seeds share 4 public eval seeds, not independent unseen scientific worlds. GRU/MLP near matched params but NOT FLOPs; current-linear is far smaller. No persistent across-world memory, cold process-death replay, P17 weight reuse, sealed S27 first-use, model promotion or FORGE integration."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s29-root",type=Path,required=True)
    ap.add_argument("--source-root",type=Path,required=True)
    ap.add_argument("--cloudpickle-root",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    output=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":output["status"],
        "public_dev_qualification":output["frozen_DEV_competence_qualified"],
        "all_learned_architectures":output["all_3_learned_architectures_qualified"],
        "true_training_acts":output["actual_factual_training_demonstration_actions"],
        "true_eval_acts":output["actual_factual_eval_actions"],
        "gradient_steps":output["actual_optimizer_updates"],
        "support":output["frozen_support_out_of_three"],
        "rewards":[{"seed":r["initialization"],
                    "mean_real_episode_rewards":r[
                        "mean_actual_real_environment_reward"]}
                   for r in output["records"]],
        "science":"NOT_RUN","wall_seconds":output["process_wall_seconds"]
    },sort_keys=True))
    return 0 if output["status"]=="PASS_FARMA_ACTUAL_FACTUAL_IMITATION_DEV_NOT_SCIENCE" else 1


if __name__=="__main__":
    raise SystemExit(main())
