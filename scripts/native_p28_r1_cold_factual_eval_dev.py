"""P28-R1: cold-only original checkpoint readback on NEW, source-admitted DEV seeds.

Original P28 training data/weights and P26 source/guards remain unchanged.
The original P28 one-use job failed because P26's correct source seed guard
rejected P28's evaluation worlds. This separate evaluator owns its own
frozen seeds, never calls P26.evaluate_episode, and NEVER trains or refits.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p20_velocity_blind_history_dev as p20
from scripts import native_p23_exact_training_resume_gravity_dev as p23
from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p28_context_task_order_dev as p28

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p28.r1-cold-factual-evaluation.v1"
TRAIN_SHA="36f3afaa76f13c59490979de5343e81a08052d95"
SOURCE_PARENT="3c94ab1d9a5dd82eac268d94880dc535cafc446f"
TRAIN_BLOB="34c6a05aee2b8b734ba23ccf7e987627d68c5609"
MODEL_SEEDS=p28.SEEDS
NORMAL_EVAL=(137011,137012,137013,137014,137015,137016,137017,137018)
REVERSE_EVAL=(138011,138012,138013,138014,138015,138016,138017,138018)
LEARNED=p28.ARMS
EVALUATE=LEARNED+("frozen_old_half_gru_plus_XOR","joint_gru_reversed_task_cue",
                  "cue_aware_two_frame_strong_conventional","full_public_sensor_teacher")


def require_dev():
    if os.environ.get("NATIVE_P28_R1_DEV_ONLY")!="1":
        raise RuntimeError("P28_R1_SCIENCE_FIRST_USE_NOT_AUTHORIZED")


def exact_source():
    x=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if not re.fullmatch("[0-9a-f]{40}",x) or x==TRAIN_SHA:
        raise RuntimeError("P28_R1_DISTINCT_EVALUATION_SOURCE_SHA_REQUIRED")
    return x


def frozen():
    q=json.loads((ROOT/"experiments/native_p28_r1_eval_scope_recovery_frozen.json").read_text())
    original=p28.frozen()
    if (q["schema"]!="ttt-hybrid.native-p28.r1-cold-eval-scope-recovery-frozen.v1"
        or q["original"]["candidate_SHA"]!=TRAIN_SHA
        or q["original"]["original_P28_script_blob"]!=TRAIN_BLOB
        or q["original"]["training_zip_id"]!=13641906162
        or q["original"]["training_zip_SHA256"]!="695e0df9ad1095823ad04e97eea0ddbbbd68c24a6d87d28489d7369f6c4ac32e"
        or q["recovery"]["only_reuse_existing_15_verified_checkpoints"] is not True
        or q["recovery"]["no_new_training_or_optimizer_steps"] is not True
        or q["recovery"]["no_scientific_first_use"] is not True
        or q["recovery"]["FORGE_activation"] is not False
        or tuple(q["recovery"]["new_eval_normal"])!=NORMAL_EVAL
        or tuple(q["recovery"]["new_eval_reverse"])!=REVERSE_EVAL
        or q["recovery"]["frozen_joint_both_tasks_min_return"]!=original["preregistered"]["old_and_reverse_min_reward_each"]
        or q["recovery"]["frozen_order_gain_new_task_min"]!=25
        or q["recovery"]["frozen_order_old_task_tolerance"]!=20
        or q["recovery"]["frozen_old_XOR_min_both"]!=175):
        raise RuntimeError("P28_R1_FROZEN_RECOVERY_OR_SOURCE_DRIFT")
    if (not set(NORMAL_EVAL).isdisjoint(REVERSE_EVAL)
        or not set(NORMAL_EVAL+REVERSE_EVAL).isdisjoint(p28.EVAL_OLD+p28.EVAL_NEW)
        or not set(NORMAL_EVAL+REVERSE_EVAL).isdisjoint(p26.EVAL_OLD+p26.EVAL_NEW)):
        raise RuntimeError("P28_R1_OLD_SEED_REUSE_DENIED")
    return q,original


def read_manifest(path):
    m=json.loads(path.read_text(encoding="utf-8"))
    if (m.get("schema")!=p28.SCHEMA or m.get("mode")!="train"
        or m.get("candidate_sha")!=TRAIN_SHA
        or m.get("parent_sha")!=SOURCE_PARENT
        or len(m.get("entries",[]))!=15
        or len(m.get("training",[]))!=12
        or m.get("same_indices_exactly_between_all_arms") is not True
        or m.get("same_old_reverse_sample_counts_and_steps") is not True
        or m.get("science_first_use")!="NOT_RUN"
        or m.get("protected_science_worlds_consumed")!=[]):
        raise RuntimeError("P28_R1_ORIGINAL_TRAINING_RECEIPT_DRIFT")
    return m


def load_original_model(directory,item,kind,seed,manifest):
    valid=kind in LEARNED or kind=="frozen_old_half_gru_plus_XOR"
    filename=(f"p28_old_anchor_{seed}.pt" if kind=="frozen_old_half_gru_plus_XOR"
              else f"p28_{kind}_{seed}.pt")
    if (not valid or seed not in MODEL_SEEDS
        or item.get("kind")!=kind or item.get("seed")!=seed
        or item.get("file")!=filename
        or item.get("data_SHA256")!=manifest["factual_source"]["source_data_SHA256"]):
        raise RuntimeError("P28_R1_MODEL_SEED_OR_DATA_IDENTITY_DENIED")
    raw=(directory/filename).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item.get("archive_sha256"):
        raise RuntimeError("P28_R1_ORIGINAL_CHECKPOINT_CORRUPT")
    state=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    if (state.get("schema")!=p28.SCHEMA
        or state.get("kind")!=kind or state.get("seed")!=seed
        or state.get("candidate_sha")!=TRAIN_SHA
        or state.get("parent_sha")!=SOURCE_PARENT
        or state.get("stream_sha")!=item.get("stream_SHA256")
        or state.get("data_sha")!=item.get("data_SHA256")
        or (kind in LEARNED and (state.get("updates")!=p28.STEPS
                                 or not state.get("optimizer",{}).get("state")))
        or (kind=="frozen_old_half_gru_plus_XOR"
            and state.get("half_updates")!=p28.HALF)):
        raise RuntimeError("P28_R1_ORIGINAL_FULL_STATE_SOURCE_DRIFT")
    if p23.semantic_digest(state)!=item["semantic_SHA256"]:
        raise RuntimeError("P28_R1_CHECKPOINT_SEMANTIC_DIGEST_DRIFT")
    net=p19.new_model("history_mlp" if kind=="joint_history_mlp" else "history_gru").eval()
    net.load_state_dict(state["model"],strict=True)
    if p23.semantic_digest(net.state_dict())!=item["model_SHA256"]:
        raise RuntimeError("P28_R1_RESTORED_MODEL_CHANGED")
    return net


@torch.no_grad()
def actual_episode(gym,arm,model,reverse,seed):
    if arm not in EVALUATE or seed not in (REVERSE_EVAL if reverse else NORMAL_EVAL):
        raise ValueError("P28_R1_NEW_SOURCE_EVAL_SEED_OR_ARM_DENIED")
    if arm not in ("full_public_sensor_teacher","cue_aware_two_frame_strong_conventional") and model is None:
        raise ValueError("P28_R1_LEARNED_MODEL_MISSING")
    env=gym.make("CartPole-v1",render_mode=None)
    trace=hashlib.sha256()
    try:
        obs,_=env.reset(seed=seed)
        history=[p18.factual_row(obs,0,0.)]
        initial=history[0]
        selected_actions=[]
        physical_actions=[]
        rewards=[]
        for _ in range(200):
            if arm=="cue_aware_two_frame_strong_conventional":
                physical=p20.finite_difference_action(history)
                selected=p26.physical_action(physical,reverse)
            elif arm=="full_public_sensor_teacher":
                physical=p18.handcoded_action(obs)
                selected=p26.physical_action(physical,reverse)
            elif arm=="frozen_old_half_gru_plus_XOR":
                x=p26.source_visible(history,False)
                if reverse:x[:,:,4]*=-1.
                logits,_=model(x)
                physical=int(logits.argmax(dim=1).item())
                selected=p26.physical_action(physical,reverse)
            else:
                cue=(not reverse if arm=="joint_gru_reversed_task_cue" else reverse)
                x=p26.source_visible(history,cue)
                if arm=="joint_cue_blind_gru":x[:,:,1]=1.
                logits,_=model(x)
                selected=int(logits.argmax(dim=1).item())
                physical=p26.physical_action(selected,reverse)
            nxt,reward,done,trunc,_=env.step(physical)
            reward=float(reward)
            after=p18.factual_row(nxt,2*selected-1,reward)
            trace.update(json.dumps(
                [history[-1],selected,physical,after,bool(done),bool(trunc)],
                allow_nan=False,separators=(",",":")).encode())
            history.append(after)
            obs=nxt
            selected_actions.append(selected)
            physical_actions.append(physical)
            rewards.append(reward)
            if done or trunc:break
        return {"reward":sum(rewards),"real_action_count":len(physical_actions),
                "source_fact_sha256":trace.hexdigest(),
                "selected_actions":selected_actions,
                "actual_physical_actions":physical_actions,
                "initial_observation":initial,
                "online_optimizer_updates":0}
    finally:env.close()


def evaluate(gym,archive,manifest):
    meta=read_manifest(manifest)
    lookup={(e.get("kind"),e.get("seed")):e for e in meta["entries"]}
    if len(lookup)!=15:
        raise RuntimeError("P28_R1_ORIGINAL_CHECKPOINT_SET_NOT_UNIQUE")
    bad={}
    sample=lookup[("joint_history_gru",MODEL_SEEDS[0])]
    original_bytes=bytearray((archive/sample["file"]).read_bytes())
    original_bytes[len(original_bytes)//2]^=1
    bad["archive_corruption_rejected"]=hashlib.sha256(original_bytes).hexdigest()!=sample["archive_sha256"]
    try:load_original_model(archive,{**sample,"seed":MODEL_SEEDS[1]},"joint_history_gru",MODEL_SEEDS[0],meta)
    except RuntimeError:bad["seed_swap_rejected"]=True
    try:load_original_model(archive,{**sample,"stream_SHA256":"0"*64},"joint_history_gru",MODEL_SEEDS[0],meta)
    except RuntimeError:bad["index_stream_tamper_rejected"]=True
    if len(bad)!=3 or not all(bad.values()):
        raise RuntimeError("P28_R1_COLD_NEGATIVE_ADMISSION_NOT_PROVEN")
    per=[]
    for seed in MODEL_SEEDS:
        models={}
        before={}
        for kind in LEARNED+("frozen_old_half_gru_plus_XOR",):
            item=lookup[(kind,seed)]
            models[kind]=load_original_model(archive,item,kind,seed,meta)
            before[kind]=p23.semantic_digest(models[kind].state_dict())
        tasks={}
        for name,reverse in (("old",False),("inverted",True)):
            worlds=REVERSE_EVAL if reverse else NORMAL_EVAL
            cases=[]
            for world in worlds:
                arms={}
                for arm in EVALUATE:
                    net=(models["joint_history_gru"] if arm=="joint_gru_reversed_task_cue"
                         else models.get(arm))
                    arms[arm]=actual_episode(gym,arm,net,reverse,world)
                if len({tuple(v["initial_observation"]) for v in arms.values()})!=1:
                    raise RuntimeError("P28_R1_INITIAL_REAL_SOURCE_RESET_DRIFT")
                if not all(v["online_optimizer_updates"]==0 and 1<=v["real_action_count"]<=200
                           for v in arms.values()):
                    raise RuntimeError("P28_R1_EVAL_UPDATED_STATE_OR_MISSING_REAL_ACTIONS")
                cases.append({"world_seed":world,"arms":arms})
            means={arm:sum(c["arms"][arm]["reward"] for c in cases)/len(cases)
                   for arm in EVALUATE}
            tasks[name]={"mean_reward":means,"source_worlds":cases}
        if before!={k:p23.semantic_digest(v.state_dict()) for k,v in models.items()}:
            raise RuntimeError("P28_R1_EVAL_MODEL_WEIGHT_MUTATION")
        old=tasks["old"]["mean_reward"]
        new=tasks["inverted"]["mean_reward"]
        per.append({
            "seed":seed,"tasks":tasks,
            "joint_GRU_both_tasks_competent":old["joint_history_gru"]>=150 and new["joint_history_gru"]>=150,
            "joint_MLP_both_tasks_competent":old["joint_history_mlp"]>=150 and new["joint_history_mlp"]>=150,
            "joint_GRU_order_gain":old["joint_history_gru"]>=150 and new["joint_history_gru"]>=150
                and new["joint_history_gru"]>=new["blocked_history_gru"]+25
                and old["joint_history_gru"]>=old["blocked_history_gru"]-20,
            "frozen_old_explicit_XOR_both_competent":old["frozen_old_half_gru_plus_XOR"]>=175
                and new["frozen_old_half_gru_plus_XOR"]>=175,
            "cue_effect_reverse":new["joint_history_gru"]-new["joint_gru_reversed_task_cue"]})
    counts=lambda field:sum(bool(row[field]) for row in per)
    joint=counts("joint_GRU_both_tasks_competent")
    mlp=counts("joint_MLP_both_tasks_competent")
    return {"schema":SCHEMA,"status":"PASS_PUBLIC_DEV_ENGINEERING_NOT_SCIENCE",
            "evaluation_candidate_sha":exact_source(),
            "original_training_candidate_sha":TRAIN_SHA,
            "original_training_artifact_id":13641906162,
            "original_train_state_checkpoint_count":15,
            "original_training_not_reexecuted":True,
            "original_training_sample_indices_and_weights_unchanged":True,
            "source_environment_only_new_seed_scope":True,
            "scientific_first_use":"NOT_RUN",
            "protected_science_seed_consumption":[],
            "reward_only_learning_proven":False,
            "architectural_superiority":False,"FORGE_owner_activation":False,
            "source_negative_controls":bad,
            "joint_GRU_dual_support_of_three":joint,
            "joint_MLP_dual_support_of_three":mlp,
            "joint_order_gain_support_of_three":counts("joint_GRU_order_gain"),
            "frozen_old_explicit_XOR_support_of_three":counts("frozen_old_explicit_XOR_both_competent"),
            "cue_intervention_support_of_three":sum(r["cue_effect_reverse"]>=25 for r in per),
            "conventional_but_not_GRU_warning":mlp>=2 and joint<2,
            "per_seed":per,
            "limitations":"Original P28 expert-supervised models and same matched train indices reused exactly; R1 only fixes new frozen public evaluation source admissions. Still capped-200 real MIT CartPole authored action inversion, not reward-only/independent-world causal discovery. Transparent controller may saturate 200. No sealed science, merge or FORGE activation."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s29-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--cloudpickle-root",type=Path,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--train-receipt",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    require_dev()
    frozen()
    exact_source()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(a.s29_root,a.source_root,a.cloudpickle_root)
    out=evaluate(gym,a.checkpoint_dir,a.train_receipt)
    p28.receipt(a.receipt,out)
    print(json.dumps({
        "status":out["status"],"original_training_not_reexecuted":True,
        "joint_GRU":out["joint_GRU_dual_support_of_three"],
        "conventional_MLP":out["joint_MLP_dual_support_of_three"],
        "order_effect":out["joint_order_gain_support_of_three"],
        "explicit_XOR":out["frozen_old_explicit_XOR_support_of_three"],
        "cue_test":out["cue_intervention_support_of_three"],
        "means":[{"seed":r["seed"],"old":r["tasks"]["old"]["mean_reward"],
                  "reverse":r["tasks"]["inverted"]["mean_reward"]}
                 for r in out["per_seed"]],
        "scientific_first_use":out["scientific_first_use"]},sort_keys=True))


if __name__=="__main__":
    raise SystemExit(main())
