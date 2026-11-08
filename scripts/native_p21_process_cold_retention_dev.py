"""P21 DEV: train checkpoint once; restore and evaluate in fresh OS Python processes.

Preserves P18/P19/P20 immutable sources, P20 original positive result, other
P20 PPO sibling, S29 licensing and protected sealed scientific first-use.
This proves a new P20-equivalent model cohort's *inference-state* persistence,
not original P20 weight restoration or continuation of optimizer/RNG training.
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
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p20_velocity_blind_history_dev as p20

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p21.process-death-cold-retention-dev.v1"
P20_PARENT="ad69bcd06e973f0c05546c3e5782b770a0ce0e6e"
TRAIN_SEEDS=p20.TRAIN_SEEDS
VAL_SEEDS=p20.VAL_SEEDS
MODEL_SEEDS=p20.MODEL_SEEDS
EVAL_SEEDS=(79011,79012,79013,79014,79015,79016,79017,79018)
KINDS=("full_history_gru","current_only_gru")
EVAL_ARMS=("full_history_gru","current_only_gru","history_reversed_gru",
           "finite_difference_public_history","full_observation_expert")
MAX_ACTIONS=200


def require_dev():
    if os.environ.get("NATIVE_P21_DEV_ONLY","0")!="1":
        raise RuntimeError("P21_SCIENCE_FIRST_USE_DENIED")


def frozen():
    j=json.loads((ROOT/"experiments/native_p21_cold_retention_frozen.json").read_text())
    if (j["schema"]!="ttt-hybrid.native-p21.process-death-cold-retention-frozen.v1"
        or j["parent"]["source_sha"]!=P20_PARENT
        or j["parent"]["original_P20_script_blob"]!="99ebe68151642a5c7a9f5efd95923e4450086155"
        or j["science"]["first_use_authorized"] is not False
        or j["science"]["sealed_original_science_source_consumed"] is not False
        or tuple(j["eval"]["seeds"])!=EVAL_SEEDS
        or tuple(j["train"]["model_initializations"])!=MODEL_SEEDS
        or tuple(j["train"]["trained_actors"])!=KINDS
        or j["train"]["epochs"]!=p19.EPOCHS):
        raise RuntimeError("P21_SOURCE_OR_FROZEN_PROTOCOL_DRIFT")
    if len(set(EVAL_SEEDS))!=8 or not set(EVAL_SEEDS).isdisjoint(p20.EVAL_SEEDS):
        raise RuntimeError("P21_REUSED_PREVIOUS_EVALUATION_SEEDS")
    return j


def sha256(raw:bytes):
    return hashlib.sha256(raw).hexdigest()


def sentinel(kind):
    rows=[p18.factual_row([.01*k,500.+k,.008*k,100.+k],1 if k%2 else -1,
                          float(k%2)) for k in range(1,9)]
    return p20.visible_rows(rows,current_only=kind=="current_only_gru")


@torch.no_grad()
def probe_sha(model,kind):
    model.eval()
    logits,value=model(sentinel(kind))
    return sha256(logits.detach().cpu().contiguous().numpy().tobytes()+
                  value.detach().cpu().contiguous().numpy().tobytes())


def checkpoint_name(kind,seed):
    if kind not in KINDS or seed not in MODEL_SEEDS:
        raise ValueError("P21_UNKNOWN_MODEL_IDENTITY")
    return f"p21_{kind}_{seed}.pt"


def require_digest(raw,expected):
    if not isinstance(expected,str) or len(expected)!=64 or sha256(raw)!=expected:
        raise RuntimeError("P21_CHECKPOINT_ARCHIVE_DIGEST_MISMATCH")


def check_entry(entry,kind,seed):
    if (entry.get("architecture")!=kind or entry.get("model_seed")!=seed
        or entry.get("source_sha")!=P20_PARENT
        or entry.get("filename")!=checkpoint_name(kind,seed)
        or entry.get("checkpoint_state_sha256") is None
        or entry.get("checkpoint_archive_sha256") is None
        or entry.get("sentinel_logits_sha256") is None):
        raise RuntimeError("P21_CHECKPOINT_CUSTODY_MISMATCH")


def model_from_checkpoint(checkpoints:Path,entry:dict,kind:str,seed:int):
    check_entry(entry,kind,seed)
    path=checkpoints/entry["filename"]
    raw=path.read_bytes()
    require_digest(raw,entry["checkpoint_archive_sha256"])
    model=p19.new_model("history_gru").eval()
    tensors=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    if not isinstance(tensors,dict) or set(tensors)!=set(model.state_dict()):
        raise RuntimeError("P21_CHECKPOINT_TENSORS_INVALID")
    for key,ref in model.state_dict().items():
        tensor=tensors[key]
        if (not isinstance(tensor,torch.Tensor)
            or tensor.shape!=ref.shape or tensor.dtype!=ref.dtype
            or not bool(torch.isfinite(tensor).all())):
            raise RuntimeError("P21_CHECKPOINT_TENSOR_SHAPE_NONFINITE")
    model.load_state_dict(tensors,strict=True)
    model.eval()
    if p18.model_sha(model)!=entry["checkpoint_state_sha256"]:
        raise RuntimeError("P21_CHECKPOINT_MODEL_DIGEST_DRIFT")
    if probe_sha(model,kind)!=entry["sentinel_logits_sha256"]:
        raise RuntimeError("P21_COLD_SENTINEL_LOGITS_CHANGED")
    return model


def receipt_write(path:Path,obj:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(obj,sort_keys=True,indent=2)+"\n")


def train(gym,checkpoints:Path):
    if checkpoints.exists():
        raise RuntimeError("P21_CHECKPOINT_DESTINATION_ALREADY_EXISTS")
    checkpoints.mkdir(parents=True,exist_ok=False)
    tx,ty,training=p19.expert_dataset(gym,TRAIN_SEEDS)
    vx,vy,validation=p19.expert_dataset(gym,VAL_SEEDS)
    full_train=p20.visible_history(tx)
    full_val=p20.visible_history(vx)
    current_train=p20.visible_history(tx,current_only=True)
    current_val=p20.visible_history(vx,current_only=True)
    entries=[]
    for seed in MODEL_SEEDS:
        for kind in KINDS:
            is_current=kind=="current_only_gru"
            x=current_train if is_current else full_train
            v=current_val if is_current else full_val
            actors,fit=p19.fit("history_gru",seed,x,ty,v,vy)
            model=actors["trained"]
            file_name=checkpoint_name(kind,seed)
            path=checkpoints/file_name
            with path.open("xb") as fout:
                torch.save(model.state_dict(),fout)
            raw=path.read_bytes()
            entries.append({
                "architecture":kind,
                "model_seed":seed,
                "source_sha":P20_PARENT,
                "filename":file_name,
                "checkpoint_archive_sha256":sha256(raw),
                "checkpoint_state_sha256":p18.model_sha(model),
                "sentinel_logits_sha256":probe_sha(model,kind),
                "original_model_updated":fit["trained_weight_updated"],
                "original_train_optimizer_updates":fit["optimizer_updates"],
                "original_training_epochs":fit["epochs_completed"],
                "in_process_source_restore_exact":fit["fresh_object_state_only_restore_exact"],
                "retains_optimizer_state":False,
                "train_heldout_teacher_agreement":fit["validation_agreement"]})
    if (len(entries)!=6
        or not all(e["original_model_updated"] and e["in_process_source_restore_exact"] for e in entries)):
        raise RuntimeError("P21_NEW_TRAINING_COHORT_INTEGRITY_FAIL")
    return {
        "schema":SCHEMA,
        "mode":"TRAIN_NEW_P20_EQUIVALENT_COHORT_ON_PUBLIC_DEV",
        "P20_original_weights_copied_or_restored":False,
        "original_P20_source_unchanged":True,
        "parent_source_sha":P20_PARENT,
        "source_only_factual_teacher":True,
        "expert_supervision_not_reward_only":True,
        "training_worlds":list(TRAIN_SEEDS),
        "validation_worlds":list(VAL_SEEDS),
        "training_source_factual_receipt":training,
        "validation_source_factual_receipt":validation,
        "checkpoints":entries,
        "checkpoint_count":len(entries),
        "science_first_use":"NOT_RUN",
        "protected_science_seeds_consumed":[],
        "cold_restoration_not_yet_tested":True}


@torch.no_grad()
def rollout(gym,arm,models,world,model_seed):
    if arm not in EVAL_ARMS or world not in EVAL_SEEDS or model_seed not in MODEL_SEEDS:
        raise ValueError("P21_UNREGISTERED_EVALUATION")
    env=gym.make("CartPole-v1",render_mode=None)
    trace=hashlib.sha256()
    try:
        obs,_=env.reset(seed=world)
        hist=[p18.factual_row(obs,0,0.)]
        initial=hist[0]
        acts=[]
        reward_sum=0.
        for _ in range(MAX_ACTIONS):
            if arm=="finite_difference_public_history":
                action=p20.finite_difference_action(hist)
            elif arm=="full_observation_expert":
                action=p18.handcoded_action(obs)
            else:
                net=models["full_history_gru"] if arm=="history_reversed_gru" else models[arm]
                x=p20.visible_rows(hist,
                                   current_only=arm=="current_only_gru",
                                   reversed_history=arm=="history_reversed_gru")
                logits,_=net(x)
                action=int(logits.argmax(dim=-1).item())
            nxt,reward,terminated,truncated,_=env.step(action)
            reward=float(reward)
            if not math.isfinite(reward):
                raise RuntimeError("P21_REAL_SOURCE_NONFINITE_REWARD")
            row=p18.factual_row(nxt,2*action-1,reward)
            trace.update(json.dumps(
                [hist[-1],action,row,bool(terminated),bool(truncated)],
                separators=(",",":"),allow_nan=False).encode())
            acts.append(action)
            reward_sum+=reward
            hist.append(row)
            obs=nxt
            if terminated or truncated:
                break
        return {"actual_reward":reward_sum,"real_source_actions":len(acts),
                "chosen_actions":acts,"factual_sha256":trace.hexdigest(),
                "original_reset_observation":initial,"online_training_updates":0}
    finally:
        env.close()


def evaluate(gym,checkpoints:Path,training_receipt:Path):
    training=json.loads(training_receipt.read_text(encoding="utf-8"))
    if (training.get("schema")!=SCHEMA or training.get("mode")!="TRAIN_NEW_P20_EQUIVALENT_COHORT_ON_PUBLIC_DEV"
        or training.get("parent_source_sha")!=P20_PARENT
        or training.get("checkpoint_count")!=6
        or training.get("science_first_use")!="NOT_RUN"
        or len(training.get("checkpoints",[]))!=6):
        raise RuntimeError("P21_TRAIN_COHORT_RECEIPT_INVALID")
    lookup={(e.get("architecture"),e.get("model_seed")):e for e in training["checkpoints"]}
    if len(lookup)!=6:
        raise RuntimeError("P21_DUPLICATE_OR_MISSING_CHECKPOINT")
    negatives=[]
    entry=lookup[("full_history_gru",MODEL_SEEDS[0])]
    raw=(checkpoints/entry["filename"]).read_bytes()
    corrupted=bytearray(raw)
    corrupted[len(corrupted)//2]^=0x01
    try:
        require_digest(bytes(corrupted),entry["checkpoint_archive_sha256"])
    except RuntimeError as e:
        negatives.append(str(e)=="P21_CHECKPOINT_ARCHIVE_DIGEST_MISMATCH")
    try:
        check_entry({**entry,"source_sha":"0000000000000000000000000000000000000000"},
                    "full_history_gru",MODEL_SEEDS[0])
    except RuntimeError as e:
        negatives.append(str(e)=="P21_CHECKPOINT_CUSTODY_MISMATCH")
    try:
        check_entry(entry,"full_history_gru",MODEL_SEEDS[1])
    except RuntimeError as e:
        negatives.append(str(e)=="P21_CHECKPOINT_CUSTODY_MISMATCH")
    if negatives!=[True,True,True]:
        raise RuntimeError("P21_NEGATIVE_ADMISSION_CONTROLS_FAILED")
    reports=[]
    for seed in MODEL_SEEDS:
        models={kind:model_from_checkpoint(checkpoints,lookup[(kind,seed)],kind,seed)
                for kind in KINDS}
        before={kind:p18.model_sha(model) for kind,model in models.items()}
        worlds=[]
        for world in EVAL_SEEDS:
            cases={arm:rollout(gym,arm,models,world,seed) for arm in EVAL_ARMS}
            if len({tuple(q["original_reset_observation"]) for q in cases.values()})!=1:
                raise RuntimeError("P21_EVAL_RESET_NOT_MATCHED")
            worlds.append({"public_dev_world_seed":world,"arms":cases})
        after={kind:p18.model_sha(model) for kind,model in models.items()}
        if before!=after:
            raise RuntimeError("P21_EVAL_WEIGHTS_MUTATED")
        means={arm:sum(x["arms"][arm]["actual_reward"] for x in worlds)/len(worlds)
               for arm in EVAL_ARMS}
        reports.append({
            "initialization":seed,
            "cold_loaded_weights_sha256":before,
            "mean_reward_public_development":means,
            "full_minus_current_only":means["full_history_gru"]-means["current_only_gru"],
            "full_minus_history_reversed":means["full_history_gru"]-means["history_reversed_gru"],
            "all_new_world_factual_actions":worlds,
            "no_eval_weights_modified":before==after})
    return {
        "schema":SCHEMA,
        "mode":"INDEPENDENT_FRESH_PROCESS_COLD_INFERENCE_ONLY",
        "status":"PASS_CHECKPOINT_INTEGRITY_NOT_SCIENCE",
        "original_P20_weights_restored":False,
        "P21_new_cohort_restored":True,
        "checkpoint_source_sha":P20_PARENT,
        "cold_loaded_models":6,
        "all_cold_tensor_and_probe_hashes_exact":True,
        "corrupt_archive_wrong_source_wrong_seed_denied":all(negatives),
        "all_training_absent_in_cold_process":True,
        "online_optimizer_updates":0,
        "public_development_evaluation_seeds":list(EVAL_SEEDS),
        "per_seed":reports,
        "scientific_first_use":"NOT_RUN",
        "protected_science_seeds_consumed":[],
        "no_optimizer_rng_state_cold_training_continuation":True,
        "strongest_2_frame_conventional_not_claimed_beaten":True,
        "FORGE_owner_activation":False,
        "limitations":"New P21 expert-supervised P20-equivalent checkpoint cohort, not original P20 tensors. Proves process-separated inference-state equality only if independent eval A/B receipts byte-match. All experiments share known MIT CartPole; no independent transfer, sealed scientific first-use, reward-only learning, or full training continuation."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("train","evaluate"),required=True)
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--train-receipt",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    require_dev()
    frozen()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(args.s29_root,args.source_root,args.cloudpickle_root)
    if args.mode=="train":
        out=train(gym,args.checkpoint_dir)
        receipt_write(args.train_receipt,out)
        if args.receipt!=args.train_receipt:
            raise ValueError("P21_TRAIN_RECEIPT_ARGUMENT_MUST_MATCH")
        print(json.dumps({"mode":args.mode,"status":"PASS_NEW_DEV_TRAIN_COHORT",
                         "checkpoint_count":out["checkpoint_count"],
                         "science":out["science_first_use"]},sort_keys=True))
    else:
        out=evaluate(gym,args.checkpoint_dir,args.train_receipt)
        receipt_write(args.receipt,out)
        print(json.dumps({"mode":args.mode,"status":out["status"],
                          "cold_models":out["cold_loaded_models"],
                          "mean_rewards":[v["mean_reward_public_development"] for v in out["per_seed"]],
                          "science":out["scientific_first_use"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
