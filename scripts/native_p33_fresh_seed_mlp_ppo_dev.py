"""P33: fresh-seed reward-only frozen-recurrence replication against trained MLP.

Reuses the EXACT qualified P32 real-physics PPO training and validation loop;
the MLP invokes the full-training arm with a scoped alternate module class.
Both policies receive identical 8x10 public past-only continuous histories.
No original P20/P30/P32 changes, no teacher decisions, no protected first use.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p32_component_damage_dev as p32

SCHEMA="ttt-hybrid.native-p33.fresh-seed-mlp-reward-replication-dev.v1"
PROTOCOL=ROOT/"experiments/native_p33_fresh_seed_mlp_replication_frozen.json"
SEEDS=(4903,4919,4931)
TRAIN_BASE=164000
EVAL_SEEDS=(165011,165019,165029,165037,165049,165059,165071,165089)
PROBE_SEEDS=range(166000,166064)
CONDITIONS=("original","short_pole_x0.7","long_pole_x1.3")
TRAIN_ARMS=("full_actor_recurrent_ppo","actor_head_only_ppo",
            "conventional_mlp_full_ppo","critic_only_frozen_policy")
EVAL_ARMS=("initial_no_update",)+TRAIN_ARMS+(
          "finite_difference_public_two_frame","seeded_random")
EXPECTED_TRAIN_ACTIONS=3*4*49152
EXPECTED_UPDATES=3*4*576

class FullyTrainedHistoryMLP(nn.Module):
    """Conventional nonrecurrent 591-param MLP sees the same FULL 8x10 history."""
    def __init__(self):
        super().__init__()
        # Field called recurrent solely to use the original-source P32
        # unchanged PPO parameter-partition and cold-audit code. It is a
        # standard flattened MLP (NO recurrence, hidden state, or RNN).
        self.recurrent=nn.Sequential(nn.Flatten(),nn.Linear(80,7),nn.Tanh())
        self.actor=nn.Linear(7,2)
        self.critic=nn.Linear(7,1)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(8,10):
            raise ValueError("P33_CONVENTIONAL_MLP_FULL_SOURCE_SHAPE")
        z=self.recurrent(x)
        return self.actor(z),self.critic(z).reshape(-1)

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p["schema"]!="ttt-hybrid.native-p33.fresh-seed-conventional-neural-ppo-replication-frozen.v1"
        or p["phase"]!="PREIMPLEMENTATION_FROZEN_ORIGINAL_PUBLIC_SOURCE_FIRST_USE_UNCONSUMED"
        or p["parent"]["sha"]!="7bc49166b83788efc7bfb9da82d39fc420630c46"
        or p["source"]["no_protected_science_first_use"] is not True
        or p["training"]["new_model_seeds"]!=list(SEEDS)
        or p["training"]["new_public_train_source_reset_base"]!=TRAIN_BASE
        or p["training"]["arms"]!=list(TRAIN_ARMS)
        or p["training"]["MLP_total_parameters"]!=591
        or p["training"]["GRU_total_parameters"]!=597
        or p["evaluation"]["new_original_heldout_reset_seeds"]!=list(EVAL_SEEDS)
        or p["evaluation"]["eval_arms"]!=list(EVAL_ARMS)
        or p["evaluation"]["new_factual_source_probe_seed_range"]!=
            [PROBE_SEEDS.start,PROBE_SEEDS.stop-1]):
        raise RuntimeError("P33_ORIGINAL_FROZEN_REPLICATION_PROTOCOL_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P33_DEV_ONLY")!="1":
        raise RuntimeError("P33_PROTECTED_FIRST_USE_DENIED")
    p=frozen()
    if (p20.ROLLOUTS,p20.NUM_ENVS,p20.STEPS_PER_ROLLOUT,
        p20.PPO_EPOCHS,p20.MINIBATCH,p20.PPO_LR,p20.GAMMA,
        p20.GAE_LAMBDA,p20.PPO_CLIP,p20.ENTROPY,p20.VALUE_COEF,
        p20.MAX_GRAD_NORM)!=(96,8,64,3,256,.0015,.99,.95,.2,.015,.5,1.):
        raise RuntimeError("P33_ORIGINAL_PPO_BUDGET_OR_OPTIMIZER_DRIFT")
    n=sum(z.numel() for z in FullyTrainedHistoryMLP().parameters())
    g=sum(z.numel() for z in p30.RewardGRU10().parameters())
    if (n,g)!=(591,597):
        raise RuntimeError("P33_MLP_VS_GRU_PARAMETER_MATCH_DRIFT")
    if (p30.H,p30.DT,p30.MOTION_SCALE)!=(8,.02,(.5,1.0,.05,.05)):
        raise RuntimeError("P33_ORIGINAL_MOTION_INPUT_SOURCE_DRIFT")
    return p

@contextmanager
def scoped_original_p32():
    """Only sequential local adapter, always restore original frozen owners."""
    old=(p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,
         p32.PROBE_SEEDS,p30.SEEDS)
    try:
        p32.SEEDS=SEEDS
        p32.TRAIN_BASE=TRAIN_BASE
        p32.EVAL_SEEDS=EVAL_SEEDS
        p32.PROBE_SEEDS=PROBE_SEEDS
        p30.SEEDS=SEEDS
        yield
    finally:
        (p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,
         p32.PROBE_SEEDS,p30.SEEDS)=old

@contextmanager
def scoped_conventional_mlp():
    """P32 same *full* PPO loss/rollouts and cold serialization on MLP."""
    old=p30.RewardGRU10
    try:
        p30.RewardGRU10=FullyTrainedHistoryMLP
        yield
    finally:
        p30.RewardGRU10=old

def model_for_eval(seed,arm):
    if seed not in SEEDS or arm not in TRAIN_ARMS:
        raise ValueError("P33_UNREGISTERED_COLD_SOURCE_MODEL")
    if arm=="conventional_mlp_full_ppo":
        torch.manual_seed(seed)
        model=FullyTrainedHistoryMLP()
    else:
        model=p32.make_initial(seed)
    model.eval()
    return model

def original_eval(gym,model,arm,seed,condition,model_seed):
    if arm not in EVAL_ARMS:
        raise ValueError("P33_UNREGISTERED_SOURCE_POLICY_EVAL")
    original_alias=("full_actor_recurrent_ppo"
                   if arm=="conventional_mlp_full_ppo" else arm)
    return p32.evaluate_episode(
        gym,model,seed,condition,model_seed,original_alias)

def record_original_worlds(gym,seed,models):
    """Exactly paired physical source resets, no model updates."""
    rows=[]
    for condition in CONDITIONS:
        for source_seed in EVAL_SEEDS:
            values={}
            first=[]
            for arm in EVAL_ARMS:
                result=original_eval(gym,models.get(arm),arm,source_seed,
                                     condition,seed)
                values[arm]=result
                first.append(tuple(result["source_original_initial_observation"]))
            if len(set(first))!=1:
                raise RuntimeError("P33_REAL_SOURCE_EVALUATION_RESET_NOT_PAIRED")
            if (values["initial_no_update"]["source_real_action_consequence_sha256"]!=
                values["critic_only_frozen_policy"]["source_real_action_consequence_sha256"]):
                raise RuntimeError("P33_CRITIC_ONLY_ACTOR_IDENTITY_FAILED")
            rows.append({"source_condition":condition,
                         "source_reset_seed":source_seed,
                         "factual_original_source_episodes":values})
    return rows

def reward_means(rows):
    return {c:{arm:sum(
        x["factual_original_source_episodes"][arm]["actual_source_reward_return"]
        for x in rows if x["source_condition"]==c)/len(EVAL_SEEDS)
        for arm in EVAL_ARMS} for c in CONDITIONS}

def frozen_arm_gates(means):
    spec=frozen()["frozen_thresholds"]
    out={}
    for condition in CONDITIONS:
        a=means[condition]["actor_head_only_ppo"]
        f=means[condition]["full_actor_recurrent_ppo"]
        b=means[condition]["conventional_mlp_full_ppo"]
        cutoff=(spec["per_seed_actor_only_original_reward_min"]
                if condition=="original"
                else spec["per_seed_actor_only_short_and_long_reward_min_each"])
        out[condition]={
            "reward_min":a>=cutoff,
            "actor_outperforms_full_at_least_25":
                a>=f+spec[
                  "per_seed_actor_only_over_full_PPO_original_and_both_shifted_min_margin"],
            "actor_within_15_of_fully_trained_MLP":
                a>=b-spec[
                  "per_seed_actor_only_noninferior_to_fully_trained_MLP_original_and_both_shifted_within_points"]
        }
    return out

def train(gym,checkpoint_dir):
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    if any(checkpoint_dir.iterdir()):
        raise RuntimeError("P33_COLD_CHECKPOINT_ROOT_ALREADY_CONSUMED")
    started=time.perf_counter()
    with scoped_original_p32():
        physical_probe,probe_receipt=p32.fixed_history_probe(gym)
        records=[]
        for seed in SEEDS:
            baseline=p32.make_initial(seed)
            init_sha=p20.model_sha(baseline)
            trained={}
            training={}
            cold_files={}
            for arm in TRAIN_ARMS:
                if arm=="conventional_mlp_full_ppo":
                    with scoped_conventional_mlp():
                        model,initial,receipt=p32.train_arm(
                            gym,seed,"full_actor_recurrent_ppo")
                    receipt["actual_ppo_training_source_alias"]="full_actor_recurrent_ppo"
                    receipt["architecture"]="standard_flatten_history_MLP_591_params"
                else:
                    model,initial,receipt=p32.train_arm(gym,seed,arm)
                    receipt["architecture"]="GRU_597_params"
                if (receipt["real_original_train_source_actions"]!=49152
                    or receipt["original_ppo_minibatch_updates"]!=576
                    or not receipt["cold_exact_logits_and_values"]
                    or not receipt["learning_executed_real_actions_without_expert"]):
                    raise RuntimeError("P33_ORIGINAL_REWARD_ONLY_ACTION_BUDGET_OR_COLD_RECEIPT_FAIL")
                if arm!="conventional_mlp_full_ppo":
                    if p20.model_sha(initial)!=init_sha:
                        raise RuntimeError("P33_NEW_GRU_INITIAL_WEIGHTS_NOT_EQUAL")
                else:
                    if receipt["all_model_parameters"]!=591:
                        raise RuntimeError("P33_FULLY_TRAINED_MATCHED_MLP_PARAMETER_MISMATCH")
                filename=f"source_seed_{seed}_{arm}.pt"
                path=checkpoint_dir/filename
                if path.exists():
                    raise RuntimeError("P33_ALREADY_TRAINED_CHECKPOINT_REFUSE_REPLAY")
                torch.save(model.state_dict(),path)
                data=path.read_bytes()
                loaded=model_for_eval(seed,arm)
                loaded.load_state_dict(torch.load(
                    path,map_location="cpu",weights_only=True),strict=True)
                loaded.eval()
                if p20.model_sha(loaded)!=p20.model_sha(model):
                    raise RuntimeError("P33_INPROCESS_COLD_CHECKPOINT_MODEL_DIFFERS")
                cold_files[arm]={
                  "relative_name":filename,
                  "checkpoint_file_sha256":hashlib.sha256(data).hexdigest(),
                  "state_weight_sha256":p20.model_sha(model),
                  "original_architecture":receipt["architecture"],
                  "actual_source_training_model_seed":seed
                }
                receipt["source_study_arm"]=arm
                trained[arm]=model
                training[arm]=receipt
            gru_first={training[arm]["first_512_real_source_actions_trace_sha256"]
                for arm in TRAIN_ARMS if arm!="conventional_mlp_full_ppo"}
            if len(gru_first)!=1:
                raise RuntimeError("P33_GRU_FIRST_SOURCE_ROLLOUT_NOT_IDENTICAL")
            if not all(
                len(training[arm]["original_source_reset_stream_sha256"])==64
                and len(training[arm]["original_actual_action_consequence_sha256"])==64
                for arm in TRAIN_ARMS):
                raise RuntimeError("P33_BAD_REAL_SOURCE_RESET_OR_ACTION_TRACE")
            if any(training[arm]["initial_source_actor_weights_sha256"]!=init_sha
                   for arm in TRAIN_ARMS if arm!="conventional_mlp_full_ppo"):
                raise RuntimeError("P33_GRU_SOURCE_INITIAL_SHA_MISMATCH")
            models={"initial_no_update":baseline,**trained}
            rows=record_original_worlds(gym,seed,models)
            means=reward_means(rows)
            gates=frozen_arm_gates(means)
            drift={arm:p32.policy_drift(baseline,model,physical_probe,seed)
                   for arm,model in trained.items()
                   if arm!="conventional_mlp_full_ppo"}
            if not drift["critic_only_frozen_policy"]["bit_exact_original_logits"]:
                raise RuntimeError("P33_CRITIC_ONLY_FRESH_SOURCE_LOGITS_CHANGED")
            records.append({
              "new_model_seed":seed,
              "historically_selected_positive_control":False,
              "initial_GRU_sha256":init_sha,
              "matching_initial_GRU_real_source_trace_sha256":next(iter(gru_first)),
              "initial_MLP_sha256":training["conventional_mlp_full_ppo"][
                  "initial_source_actor_weights_sha256"],
              "actual_original_source_train_receipts":training,
              "checkpoint_manifest":cold_files,
              "actual_source_heldout_episodes":rows,
              "real_physics_reward_means":means,
              "pre_frozen_actor_only_replication_rules":gates,
              "real_public_source_actor_drift":drift,
              "no_eval_updates_or_model_selection":True
            })
        actual_actions=sum(
           x["real_original_train_source_actions"]
           for rec in records
           for x in rec["actual_original_source_train_receipts"].values())
        updates=sum(
           x["original_ppo_minibatch_updates"]
           for rec in records
           for x in rec["actual_original_source_train_receipts"].values())
        all_gates=all(all(all(g.values())
                          for g in rec["pre_frozen_actor_only_replication_rules"].values())
                      for rec in records)
        walls=time.perf_counter()-started
        allowed=frozen()["qualification"]["max_train_and_cold_source_seconds"]
        good=(actual_actions==EXPECTED_TRAIN_ACTIONS
              and updates==EXPECTED_UPDATES
              and walls<=allowed)
        return {
          "schema":SCHEMA,
          "engineering":"PASS_ORIGINAL_PUBLIC_DEV_ONLY" if good else "FAIL_P33_ENGINEERING",
          "protected_scientific_first_use":"NOT_RUN",
          "original_P20_P30_P32_source_modified":False,
          "original_other_research_lanes_untouched":True,
          "new_model_seeds_all_disjoint_from_P32":True,
          "P33_fresh_seed_PPO_frozen_replication_all_three":all_gates,
          "novel_non_CartPole_task_generalization_proven":False,
          "no_merge_model_promotion_or_FORGE_activation":True,
          "source_physical_probe":probe_receipt,
          "original_factual_train_actions":actual_actions,
          "original_PPO_optimizer_minibatches":updates,
          "original_source_eval_actions":sum(
              item["actual_source_actions"] for rec in records
              for row in rec["actual_source_heldout_episodes"]
              for item in row["factual_original_source_episodes"].values()),
          "train_and_source_eval_wall_seconds":walls,
          "records":records
        }

def cold_verify(gym,receipt_path,checkpoint_dir):
    result=json.loads(receipt_path.read_text(encoding="utf-8"))
    if (result["schema"]!=SCHEMA
        or result["engineering"]!="PASS_ORIGINAL_PUBLIC_DEV_ONLY"
        or result["original_factual_train_actions"]!=EXPECTED_TRAIN_ACTIONS
        or result["original_PPO_optimizer_minibatches"]!=EXPECTED_UPDATES
        or len(result["records"])!=len(SEEDS)):
        raise RuntimeError("P33_COLD_DIFFERENT_FROZEN_ORIGINAL_RUN")
    expected_files=len(SEEDS)*len(TRAIN_ARMS)
    if len(list(checkpoint_dir.glob("*.pt")))!=expected_files:
        raise RuntimeError("P33_COLD_CHECKPOINT_SET_INCOMPLETE")
    compared=0
    actions=0
    with scoped_original_p32(),torch.no_grad():
        for record in result["records"]:
            seed=record["new_model_seed"]
            if seed not in SEEDS:
                raise RuntimeError("P33_COLD_DIFFERENT_MODEL_SEED")
            for arm in TRAIN_ARMS:
                manifest=record["checkpoint_manifest"][arm]
                path=checkpoint_dir/manifest["relative_name"]
                if not path.is_file() or path.parent.resolve()!=checkpoint_dir.resolve():
                    raise RuntimeError("P33_COLD_CHECKPOINT_PATH_ESCAPE_OR_MISSING")
                content=path.read_bytes()
                if hashlib.sha256(content).hexdigest()!=manifest[
                        "checkpoint_file_sha256"]:
                    raise RuntimeError("P33_COLD_CHECKPOINT_FILE_SHA_DRIFT")
                model=model_for_eval(seed,arm)
                model.load_state_dict(torch.load(path,map_location="cpu",
                                             weights_only=True),strict=True)
                model.eval()
                if p20.model_sha(model)!=manifest["state_weight_sha256"]:
                    raise RuntimeError("P33_COLD_CHECKPOINT_WEIGHT_SHA_DRIFT")
                prior=p20.model_sha(model)
                for row in record["actual_source_heldout_episodes"]:
                    world=row["source_reset_seed"]
                    condition=row["source_condition"]
                    expected=row["factual_original_source_episodes"][arm]
                    fresh=original_eval(gym,model,arm,world,condition,seed)
                    if (fresh["source_real_action_consequence_sha256"]!=
                        expected["source_real_action_consequence_sha256"]
                        or fresh["actual_source_reward_return"]!=
                            expected["actual_source_reward_return"]
                        or fresh["source_executed_action_choices"]!=
                            expected["source_executed_action_choices"]):
                        raise RuntimeError("P33_COLD_FRESH_PROCESS_SOURCE_ACTION_TRACE_PARITY_FAIL")
                    compared+=1
                    actions+=fresh["actual_source_actions"]
                if p20.model_sha(model)!=prior:
                    raise RuntimeError("P33_COLD_VERIFICATION_MODIFIED_MODEL")
    if compared!=len(SEEDS)*len(TRAIN_ARMS)*len(CONDITIONS)*len(EVAL_SEEDS):
        raise RuntimeError("P33_COLD_ORIGINAL_WORLD_COVERAGE_INCOMPLETE")
    return {
      "schema":"ttt-hybrid.native-p33.fresh-process-cold-reload.v1",
      "status":"PASS_BIT_EXACT_ORIGINAL_SOURCE_TRACE_AND_REWARD_PARITY",
      "model_seeds":list(SEEDS),
      "trained_model_count":expected_files,
      "independent_fresh_process_source_episodes":compared,
      "independent_fresh_process_source_actions":actions,
      "no_optimizer_or_training_performed":True,
      "full_architecture_MLP_and_GRU_weights_read_back_sha256":True,
      "protected_scientific_first_use":"NOT_RUN"
    }

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--s29-root",type=Path,required=True)
    arg.add_argument("--source-root",type=Path,required=True)
    arg.add_argument("--cloudpickle-root",type=Path,required=True)
    arg.add_argument("--receipt",type=Path,required=True)
    arg.add_argument("--checkpoint-dir",type=Path,required=True)
    arg.add_argument("--mode",choices=("train","cold"),required=True)
    args=arg.parse_args()
    require_dev()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not args.checkpoint_dir.is_absolute():
        raise RuntimeError("P33_COLD_CHECKPOINT_ROOT_MUST_BE_ABSOLUTE")
    if not args.receipt.is_absolute():
        raise RuntimeError("P33_RECEIPT_PATH_MUST_BE_ABSOLUTE")
    gym,source=p20.admit_gymnasium(args.s29_root,args.source_root,
                                   args.cloudpickle_root)
    if args.mode=="train":
        if args.receipt.exists():
            raise RuntimeError("P33_EXPERIMENT_RESULT_ALREADY_CONSUMED_REFUSE_REPLAY")
        result=train(gym,args.checkpoint_dir)
        result["source_custody"]=source
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                                encoding="utf-8")
        print("P33_ORIGINAL_FROZEN_REPLICATION_RESULT="+json.dumps({
          "engineering":result["engineering"],
          "all_three_actor_only_replication":result[
            "P33_fresh_seed_PPO_frozen_replication_all_three"],
          "original_real_training_actions":result["original_factual_train_actions"],
          "original_real_PPO_minibatches":result["original_PPO_optimizer_minibatches"],
          "source_eval_actions":result["original_source_eval_actions"],
          "whole_process_seconds":result["train_and_source_eval_wall_seconds"],
          "new_seed_results":[{
             "model_seed":rec["new_model_seed"],
             "reward_means_by_source_condition":rec["real_physics_reward_means"],
             "pre_frozen_actor_only_rules":rec[
                "pre_frozen_actor_only_replication_rules"],
             "real_training_source_trace_sha256":{
                arm:x["original_actual_action_consequence_sha256"]
                for arm,x in rec["actual_original_source_train_receipts"].items()}
          } for rec in result["records"]]
        },sort_keys=True))
        return 0 if result["engineering"]=="PASS_ORIGINAL_PUBLIC_DEV_ONLY" else 1
    else:
        result=cold_verify(gym,args.receipt,args.checkpoint_dir)
        out=args.receipt.parent/"native_p33_cold_audit.json"
        if out.exists():
            raise RuntimeError("P33_COLD_AUDIT_ALREADY_DONE_REFUSE_REPLAY")
        out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                       encoding="utf-8")
        print("P33_INDEPENDENT_FRESH_PROCESS_COLD_REPLAY="+json.dumps(result,sort_keys=True))
        return 0

if __name__=="__main__":
    raise SystemExit(main())
