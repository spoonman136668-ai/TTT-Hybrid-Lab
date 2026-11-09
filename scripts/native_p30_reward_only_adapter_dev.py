"""P30: learn a tiny action adapter from real paired reward, keeping old GRU frozen.

This is a public DEV original-source action-consequence experiment, NOT
independent novel-world intelligence. Original P26/P28/P29 and sealed
scientific-first-use research remain untouched.
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
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p20_velocity_blind_history_dev as p20
from scripts import native_p23_exact_training_resume_gravity_dev as p23
from scripts import native_p26_contradictory_action_retention_dev as p26
from scripts import native_p28_r1_cold_factual_eval_dev as p28r1

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p30.reward-only-context-adapter-dev.v1"
TRAIN_PARENT="57d89efde042c0a14f316e69a49a34f73e738340"
ORIGINAL_P28_TRAIN="36f3afaa76f13c59490979de5343e81a08052d95"
SEEDS=(2741,2753,2767)
TRAIN_NORMAL=(142011,142012,142013,142014,142015,142016,142017,142018,142019,142020,142021,142022)
TRAIN_REVERSE=(143011,143012,143013,143014,143015,143016,143017,143018,143019,143020,143021,143022)
EVAL_NORMAL=(144011,144012,144013,144014,144015,144016,144017,144018)
EVAL_REVERSE=(145011,145012,145013,145014,145015,145016,145017,145018)
LEARNED=("two_parameter_cue_logistic_adapter","cue_blind_two_parameter_logistic",
         "wrong_credited_two_parameter_logistic")
ALL_ARMS=LEARNED+("conventional_two_context_reward_table","unadapted_identity",
                  "manually_explicit_XOR_oracle","cue_aware_two_frame_finite_difference")
CAP=200
EPOCHS=48
LR=0.10


def require_dev():
    if os.environ.get("NATIVE_P30_DEV_ONLY")!="1":
        raise RuntimeError("P30_SCIENCE_FIRST_USE_NOT_AUTHORIZED")


def exact_sha():
    sha=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if not re.fullmatch(r"[a-f0-9]{40}",sha) or sha==TRAIN_PARENT:
        raise RuntimeError("P30_EXACT_NEW_ADAPTER_SOURCE_REQUIRED")
    return sha


def frozen():
    q=json.loads((ROOT/"experiments/native_p30_reward_only_action_adapter_frozen.json").read_text())
    if (q["schema"]!="ttt-hybrid.native-p30.source-reward-only-mapping-adapter-frozen.v1"
        or q["parent"]["P28_R1_qualified_SHA"]!=TRAIN_PARENT
        or q["parent"]["source_training_P28_SHA"]!=ORIGINAL_P28_TRAIN
        or q["parent"]["original_training_artifact_id"]!=13641906162
        or q["qualification"]["protected_scientific_first_use"] is not False
        or q["qualification"]["no_accepted_model_no_merge_no_FORGE_owner_activation"] is not True
        or tuple(q["training"]["base_GRU_initializations"])!=SEEDS
        or q["training"]["epochs"]!=EPOCHS or q["training"]["lr"]!=LR
        or tuple(q["factual_source"]["train_normal_seeds"])!=TRAIN_NORMAL
        or tuple(q["factual_source"]["train_reverse_seeds"])!=TRAIN_REVERSE
        or tuple(q["factual_source"]["eval_normal_seeds"])!=EVAL_NORMAL
        or tuple(q["factual_source"]["eval_reverse_seeds"])!=EVAL_REVERSE
        or tuple(q["training"]["arms"])!=ALL_ARMS
        or q["thresholds"]["learned_adapter_both_contexts_mean_real_return_each_at_least"]!=175
        or q["thresholds"]["reversed_gain_over_unadapted_identity_at_least"]!=100
        or q["thresholds"]["negative_wrong_credit_reversed_gap_at_least"]!=100):
        raise RuntimeError("P30_FROZEN_SOURCE_OR_HYPOTHESIS_DRIFT")
    sets=[TRAIN_NORMAL,TRAIN_REVERSE,EVAL_NORMAL,EVAL_REVERSE,
          p28r1.NORMAL_EVAL,p28r1.REVERSE_EVAL]
    if len(set().union(*[set(x) for x in sets]))!=sum(len(x) for x in sets):
        raise RuntimeError("P30_CONSUMED_OR_OVERLAPPING_PUBLIC_SEEDS")
    return q


def digest(blob:bytes)->str:
    return hashlib.sha256(blob).hexdigest()


def source_frozen_model(checkpoints,manifest,seed):
    if seed not in SEEDS:
        raise ValueError("P30_UNREGISTERED_MODEL_SEED")
    data=p28r1.read_manifest(manifest)
    entries={(e["kind"],e["seed"]):e for e in data["entries"]}
    if len(entries)!=15:
        raise RuntimeError("P30_ORIGINAL_15_MODEL_ARCHIVE_SET_DRIFT")
    model=p28r1.load_original_model(checkpoints,entries[("frozen_old_half_gru_plus_XOR",seed)],
                                    "frozen_old_half_gru_plus_XOR",seed,data)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,entries[("frozen_old_half_gru_plus_XOR",seed)]["model_SHA256"]


@torch.no_grad()
def source_episode(gym,base_model,task_is_reversed,candidate_bit,world):
    """Two independent real env worlds for interventions, no fake counterfactuals.

    Learner ONLY gets public cue and actual reward. The environment's hidden
    physical action remap is used here solely to EXECUTE and audit source
    actions, never passed as a label or prior to the trained adapter.
    """
    if not isinstance(task_is_reversed,bool) or candidate_bit not in (0,1):
        raise ValueError("P30_EXPLICIT_CUE_AND_TWO_PHYSICAL_MAPPING_CANDIDATES_REQUIRED")
    if world not in (TRAIN_REVERSE+EVAL_REVERSE if task_is_reversed else TRAIN_NORMAL+EVAL_NORMAL):
        raise ValueError("P30_UNREGISTERED_FACTUAL_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    trace=hashlib.sha256()
    try:
        obs,_=env.reset(seed=world)
        first=p18.factual_row(obs,0,0.)
        history=[first]
        selected_actions=[]
        physical_actions=[]
        rewards=[]
        for _ in range(CAP):
            # Old control expects past PHYSICAL actions, not the actuator
            # command. Candidate inference bit relabels previous commands;
            # this code does NOT use true task reversal for the policy input.
            x=p26.source_visible(history,False)
            if candidate_bit:
                x[:,:,4]*=-1.
            pred,_=base_model(x)
            desired_physical=int(pred.argmax(dim=1).item())
            selected=desired_physical^candidate_bit
            physical=selected^int(task_is_reversed)
            nxt,reward,terminated,truncated,_=env.step(physical)
            reward=float(reward)
            if not (0<=reward<=1 and torch.isfinite(torch.tensor(reward))):
                raise RuntimeError("P30_ACTUAL_SOURCE_NONFINITE_OR_UNEXPECTED_REWARD")
            after=p18.factual_row(nxt,2*selected-1,reward)
            trace.update(json.dumps(
                [history[-1],selected,physical,after,bool(terminated),bool(truncated)],
                separators=(",",":"),allow_nan=False).encode())
            history.append(after)
            selected_actions.append(selected)
            physical_actions.append(physical)
            rewards.append(reward)
            obs=nxt
            if terminated or truncated:break
        return {"source_return":sum(rewards),"factual_actions":len(rewards),
                "chosen_actions":selected_actions,"physically_executed_actions":physical_actions,
                "factual_consequence_SHA256":trace.hexdigest(),"original_initial_observation":first,
                "optimizer_updates":0}
    finally:env.close()


def training_pairs(gym,base_model,seed):
    if seed not in SEEDS:
        raise ValueError("P30_INVALID_ORIGINAL_BASE_SEED")
    pairs=[]
    for cue,reversed_task,worlds in ((1.,False,TRAIN_NORMAL),(-1.,True,TRAIN_REVERSE)):
        for world in worlds:
            bit0=source_episode(gym,base_model,reversed_task,0,world)
            bit1=source_episode(gym,base_model,reversed_task,1,world)
            if bit0["original_initial_observation"]!=bit1["original_initial_observation"]:
                raise RuntimeError("P30_PAIRED_SOURCE_RESET_MISMATCH")
            if bit0["physically_executed_actions"][0]==bit1["physically_executed_actions"][0]:
                raise RuntimeError("P30_NOT_ACTUALLY_OPPOSITE_FIRST_PHYSICAL_INTERVENTIONS")
            pairs.append({
                "cue":cue,"world":world,"episode_return_bit0":bit0["source_return"],
                "episode_return_bit1":bit1["source_return"],
                "two_truly_executed_real_source_episodes":[bit0,bit1]})
    if len(pairs)!=48:
        raise RuntimeError("P30_UNEXPECTED_SOURCE_PAIR_COUNT")
    return pairs


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit=nn.Linear(1,1)
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=1:
            raise ValueError("P30_CONTINUOUS_CUE_SHAPE")
        return self.logit(x).squeeze(1)


def adapter_actor(seed,kind):
    if seed not in SEEDS or kind not in LEARNED:
        raise ValueError("P30_UNREGISTERED_ADAPTER")
    torch.manual_seed(seed+900000)
    net=Adapter()
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.)
    return net,opt


def train_from_actual_rewards(pairs,seed,kind):
    if len(pairs)!=48 or kind not in LEARNED:
        raise RuntimeError("P30_SOURCE_TRAINING_PAIR_OR_ADAPTER_DRIFT")
    x=torch.tensor([[float(e["cue"])] for e in pairs],dtype=torch.float32)
    deltas=torch.tensor([float(e["episode_return_bit1"]-e["episode_return_bit0"]) for e in pairs],
                        dtype=torch.float32)
    if not all(bool((deltas[x[:,0]==cue]!=0).any()) for cue in (1.,-1.)):
        raise RuntimeError("P30_NO_DISCRIMINATING_FACTUAL_SOURCE_REWARDS")
    if kind=="cue_blind_two_parameter_logistic":
        x=torch.zeros_like(x)
    elif kind=="wrong_credited_two_parameter_logistic":
        deltas=-deltas
    y=(deltas>0).float()
    weights=deltas.abs()/CAP
    net,opt=adapter_actor(seed,kind)
    before=p23.semantic_digest(net.state_dict())
    for _ in range(EPOCHS):
        net.train()
        out=net(x)
        loss=(F.binary_cross_entropy_with_logits(out,y,reduction="none")*weights).sum()/weights.sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("P30_NONFINITE_REWARD_ONLY_LOGISTIC")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    net.eval()
    if before==p23.semantic_digest(net.state_dict()):
        raise RuntimeError("P30_ADAPTER_NOT_TRAINED")
    return net,opt,{"kind":kind,"optimizer_updates":EPOCHS,
                     "source_pair_count":len(pairs),"raw_real_reward_pairs":len(pairs),
                     "initial_sha":before,"final_sha":p23.semantic_digest(net.state_dict()),
                     "mean_abs_actual_reward_difference":float(deltas.abs().mean())}


def table_from_actual_rewards(pairs):
    if len(pairs)!=48:raise RuntimeError("P30_TABLE_SOURCE_PAIR_DRIFT")
    table={}
    for cue in (1.,-1.):
        entries=[e for e in pairs if e["cue"]==cue]
        if len(entries)!=24:raise RuntimeError("P30_NOT_EQUAL_REAL_SOURCE_EXPOSURE")
        means=[sum(e[f"episode_return_bit{bit}"] for e in entries)/len(entries) for bit in (0,1)]
        if means[0]==means[1]:raise RuntimeError("P30_TABLE_ACTION_AMBIGUITY")
        table[str(int(cue))]={"mapping_bit":int(means[1]>means[0]),
                              "real_reward_means_bit0_and_bit1":means,
                              "source_episodes_per_candidate":len(entries)}
    return table


def checkpoint(directory,seed,kind,model,opt,data_sha,base_sha):
    directory.mkdir(parents=True,exist_ok=True)
    name=f"p30_{kind}_{seed}.pt"
    state={"schema":SCHEMA,"source_sha":exact_sha(),"original_train_SHA":ORIGINAL_P28_TRAIN,
           "original_base_SHA":base_sha,"seed":seed,"kind":kind,
           "data_sha":data_sha,"epochs":EPOCHS,
           "model":model.state_dict(),"optimizer":opt.state_dict()}
    path=directory/name
    with path.open("xb") as f:torch.save(state,f)
    return {"kind":kind,"seed":seed,"file":name,
            "archive_sha256":digest(path.read_bytes()),
            "semantic_sha256":p23.semantic_digest(state),
            "model_sha256":p23.semantic_digest(model.state_dict()),
            "original_base_sha256":base_sha,"training_data_sha256":data_sha,
            "optimizer_steps":EPOCHS}


def read_checkpoint(directory,entry,kind,seed,data_sha,base_sha,train_sha):
    path=directory/f"p30_{kind}_{seed}.pt"
    if (entry.get("seed")!=seed or entry.get("kind")!=kind
        or entry.get("file")!=path.name or entry.get("training_data_sha256")!=data_sha
        or entry.get("original_base_sha256")!=base_sha):
        raise RuntimeError("P30_SOURCE_OR_ADAPTER_IDENTITY_DENIED")
    raw=path.read_bytes()
    if digest(raw)!=entry.get("archive_sha256"):
        raise RuntimeError("P30_ADAPTER_ARCHIVE_CORRUPT")
    state=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    if (state.get("schema")!=SCHEMA or state.get("source_sha")!=train_sha
        or state.get("original_train_SHA")!=ORIGINAL_P28_TRAIN
        or state.get("original_base_SHA")!=base_sha
        or state.get("seed")!=seed or state.get("kind")!=kind
        or state.get("data_sha")!=data_sha or state.get("epochs")!=EPOCHS
        or not state.get("optimizer",{}).get("state")
        or p23.semantic_digest(state)!=entry["semantic_sha256"]):
        raise RuntimeError("P30_ADAPTER_OPTIMIZER_OR_SOURCE_DRIFT")
    net=Adapter().eval()
    net.load_state_dict(state["model"],strict=True)
    if p23.semantic_digest(net.state_dict())!=entry["model_sha256"]:
        raise RuntimeError("P30_COLD_ADAPTER_MODEL_DRIFT")
    return net


def chosen_bit(arm,cue,model=None,table=None):
    if cue not in (1.,-1.):
        raise ValueError("P30_ONLY_FROZEN_PUBLIC_CONTEXTS")
    if arm=="manually_explicit_XOR_oracle":
        return int(cue<0)  # GOLD ONLY; never sent to trainable adapter.
    if arm=="unadapted_identity":
        return 0
    if arm=="conventional_two_context_reward_table":
        return int(table[str(int(cue))]["mapping_bit"])
    if arm=="cue_aware_two_frame_finite_difference":
        return int(cue<0)
    if arm not in LEARNED or model is None:
        raise ValueError("P30_UNKNOWN_OR_UNLOADED_ADAPTER")
    with torch.no_grad():
        x=torch.tensor([[cue if arm!="cue_blind_two_parameter_logistic" else 0.]],dtype=torch.float32)
        return int(model(x).item()>=0.)


def deterministic_control_episode(gym,reverse,world):
    if world not in (EVAL_REVERSE if reverse else EVAL_NORMAL):
        raise ValueError("P30_CONVENTIONAL_UNREGISTERED_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    trace=hashlib.sha256()
    try:
        obs,_=env.reset(seed=world)
        history=[p18.factual_row(obs,0,0.)]
        total=0.
        actions=0
        for _ in range(CAP):
            physical=p20.finite_difference_action(history)
            selected=p26.physical_action(physical,reverse)
            nxt,reward,done,trunc,_=env.step(physical)
            after=p18.factual_row(nxt,2*selected-1,float(reward))
            trace.update(json.dumps([history[-1],selected,physical,after,bool(done),bool(trunc)],
                                    separators=(",",":"),allow_nan=False).encode())
            history.append(after)
            obs=nxt
            total+=float(reward)
            actions+=1
            if done or trunc:break
        return {"source_return":total,"factual_actions":actions,
                "factual_consequence_SHA256":trace.hexdigest(),"optimizer_updates":0}
    finally:env.close()


def immutable_receipt(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True,indent=2)+"\n")


def train(gym,original_dir,original_receipt,output_dir,manifest):
    if output_dir.exists():raise RuntimeError("P30_NO_REPLAY_OF_SOURCE_EPISODES")
    frozen()
    old=p28r1.read_manifest(original_receipt)
    output_dir.mkdir(parents=True)
    entries=[]
    evidence=[]
    summaries=[]
    for seed in SEEDS:
        model,base_sha=source_frozen_model(original_dir,original_receipt,seed)
        original=p23.semantic_digest(model.state_dict())
        pairs=training_pairs(gym,model,seed)
        source_sha=p23.semantic_digest(pairs)
        table=table_from_actual_rewards(pairs)
        statuses=[]
        for kind in LEARNED:
            adapter,opt,status=train_from_actual_rewards(pairs,seed,kind)
            entries.append(checkpoint(output_dir,seed,kind,adapter,opt,source_sha,base_sha))
            statuses.append(status)
        if original!=p23.semantic_digest(model.state_dict()):
            raise RuntimeError("P30_FROZEN_OLD_GRU_MUTATED_DURING_ADAPTER_TRAINING")
        summaries.append({"seed":seed,"original_gru_sha":base_sha,
                          "factual_pair_data_sha256":source_sha,
                          "paired_actually_executed_episodes":len(pairs)*2,
                          "conventional_table":table,
                          "adapter_training":statuses})
        evidence.append({"seed":seed,"data":pairs,"source_sha256":source_sha})
    if len(entries)!=len(SEEDS)*len(LEARNED):
        raise RuntimeError("P30_ADAPTER_CHECKPOINT_SET_MISSING")
    result={"schema":SCHEMA,"mode":"train","candidate_sha":exact_sha(),
            "parent_R1_SHA":TRAIN_PARENT,"original_P28_train_SHA":ORIGINAL_P28_TRAIN,
            "original_model_checkpoint_count":15,"original_frozen_GRU_not_retrained":True,
            "source_true_pair_replay_not_inferred_counterfactual":True,
            "learned_adapter_source_is_true_rewards_not_teacher_labels":True,
            "entries":entries,"summaries":summaries,"source_evidence":evidence,
            "science_first_use":"NOT_RUN","FORGE_owner_activation":False}
    immutable_receipt(manifest,result)
    return result


def evaluate(gym,original_dir,original_receipt,adapter_dir,training_manifest):
    m=json.loads(training_manifest.read_text())
    if (m.get("schema")!=SCHEMA or m.get("mode")!="train"
        or m.get("candidate_sha")!=exact_sha()
        or m.get("parent_R1_SHA")!=TRAIN_PARENT
        or m.get("original_P28_train_SHA")!=ORIGINAL_P28_TRAIN
        or len(m.get("entries",[]))!=len(LEARNED)*len(SEEDS)
        or m.get("science_first_use")!="NOT_RUN"):
        raise RuntimeError("P30_REUSED_TRAINING_MANIFEST_SOURCE_DENIED")
    lookup={(e["kind"],e["seed"]):e for e in m["entries"]}
    if len(lookup)!=9:raise RuntimeError("P30_ADAPTER_ARCHIVE_DUPLICATE_OR_MISSING")
    source_rows={r["seed"]:r for r in m["summaries"]}
    if set(source_rows)!=set(SEEDS):raise RuntimeError("P30_ORIGINAL_MODEL_SEEDS_DRIFT")
    neg={}
    sample=lookup[(LEARNED[0],SEEDS[0])]
    damaged=bytearray((adapter_dir/sample["file"]).read_bytes())
    damaged[len(damaged)//2]^=1
    neg["byte_corrupted_archive_rejected"]=digest(damaged)!=sample["archive_sha256"]
    try:read_checkpoint(adapter_dir,{**sample,"seed":SEEDS[1]},LEARNED[0],SEEDS[0],
                        sample["training_data_sha256"],sample["original_base_sha256"],exact_sha())
    except RuntimeError:neg["wrong_model_seed_rejected"]=True
    try:read_checkpoint(adapter_dir,{**sample,"training_data_sha256":"0"*64},LEARNED[0],SEEDS[0],
                        sample["training_data_sha256"],sample["original_base_sha256"],exact_sha())
    except RuntimeError:neg["wrong_source_data_rejected"]=True
    if len(neg)!=3 or not all(neg.values()):
        raise RuntimeError("P30_FAIL_CLOSED_CHECKPOINT_NEGATIVE_FAILED")
    per_seed=[]
    for seed in SEEDS:
        base,base_sha=source_frozen_model(original_dir,original_receipt,seed)
        before=p23.semantic_digest(base.state_dict())
        mrow=source_rows[seed]
        if mrow["original_gru_sha"]!=base_sha:
            raise RuntimeError("P30_ORIGINAL_BASE_SHA_CHANGED")
        adapters={kind:read_checkpoint(adapter_dir,lookup[(kind,seed)],kind,seed,
                                       mrow["factual_pair_data_sha256"],base_sha,exact_sha())
                  for kind in LEARNED}
        before_adapters={kind:p23.semantic_digest(net.state_dict()) for kind,net in adapters.items()}
        tasks={}
        for name,reverse,worlds in (("normal",False,EVAL_NORMAL),("reversed",True,EVAL_REVERSE)):
            scores=[]
            for world in worlds:
                outcomes={}
                for arm in ALL_ARMS:
                    if arm=="cue_aware_two_frame_finite_difference":
                        result=deterministic_control_episode(gym,reverse,world)
                        chosen=None
                    else:
                        chosen=chosen_bit(arm,1. if not reverse else -1.,
                                         adapters.get(arm),mrow["conventional_table"])
                        result=source_episode(gym,base,reverse,chosen,world)
                    outcomes[arm]={"chosen_adapter_mapping_bit":chosen,**result}
                if not all(x["optimizer_updates"]==0 for x in outcomes.values()):
                    raise RuntimeError("P30_EVALUATION_OPTIMIZER_UPDATED")
                scores.append({"world":world,"actual_source_results":outcomes})
            means={arm:sum(x["actual_source_results"][arm]["source_return"] for x in scores)/len(scores)
                   for arm in ALL_ARMS}
            tasks[name]={"mean_actual_return":means,"real_source_worlds":scores}
        if before!=p23.semantic_digest(base.state_dict()) or before_adapters!={k:p23.semantic_digest(v.state_dict()) for k,v in adapters.items()}:
            raise RuntimeError("P30_COLD_RESTORED_MODEL_MUTATED_DURING_EVALUATION")
        old=tasks["normal"]["mean_actual_return"]
        new=tasks["reversed"]["mean_actual_return"]
        per_seed.append({
            "seed":seed,"tasks":tasks,
            "learned_both_reward_ge_175":old[LEARNED[0]]>=175 and new[LEARNED[0]]>=175,
            "learned_reverse_gain_over_identity_ge_100":new[LEARNED[0]]>=new["unadapted_identity"]+100,
            "learned_reverse_gain_over_wrong_credit_ge_100":new[LEARNED[0]]>=new[LEARNED[2]]+100,
            "oracle_both_reward_ge_175":old["manually_explicit_XOR_oracle"]>=175 and new["manually_explicit_XOR_oracle"]>=175,
            "trained_conventional_tabular_mean":mrow["conventional_table"],
            "learned_adapter_map":{kind:{name:chosen_bit(kind,cue,adapters[kind])
                                         for name,cue in (("normal",1.),("reversed",-1.))}
                                   for kind in LEARNED}})
    count=lambda key:sum(int(bool(x[key])) for x in per_seed)
    return {
        "schema":SCHEMA,"status":"PASS_ENGINEERING_SOURCE_EVAL_NOT_SCIENCE",
        "candidate_sha":exact_sha(),"original_train_SHA":ORIGINAL_P28_TRAIN,
        "original_15_checkpoint_artifact_id":13641906162,
        "original_frozen_source_GRU_reused_no_gradients":True,
        "actual_source_paired_reward_learning_only":True,
        "conventional_reward_table_same_real_episode_data":True,
        "source_negative_controls":neg,
        "learned_dual_support_of_three":count("learned_both_reward_ge_175"),
        "learned_reverse_gain_support_of_three":count("learned_reverse_gain_over_identity_ge_100"),
        "wrong_credit_negative_support_of_three":count("learned_reverse_gain_over_wrong_credit_ge_100"),
        "oracle_both_support_of_three":count("oracle_both_reward_ge_175"),
        "new_public_eval_seeds_only":list(EVAL_NORMAL+EVAL_REVERSE),
        "per_seed":per_seed,
        "science_first_use":"NOT_RUN","protected_sci_data_used":[],
        "scientific_superiority":False,"FORGE_owner_activation":False,
        "limitations":"Only a two-bit task mapping under an explicitly presented cue, trained from true paired external source reward episodes. Correct direction could also be inferred by a fully trained conventional empirical reward table; this is not novel causal intelligence or cross-world transfer. The old physical controller was originally teacher supervised."}


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--mode",choices=("train","evaluate"),required=True)
    arg.add_argument("--s29-root",type=Path,required=True)
    arg.add_argument("--source-root",type=Path,required=True)
    arg.add_argument("--cloudpickle-root",type=Path,required=True)
    arg.add_argument("--original-checkpoints",type=Path,required=True)
    arg.add_argument("--original-manifest",type=Path,required=True)
    arg.add_argument("--adapter-dir",type=Path,required=True)
    arg.add_argument("--train-receipt",type=Path,required=True)
    arg.add_argument("--receipt",type=Path,required=True)
    a=arg.parse_args()
    require_dev()
    frozen()
    exact_sha()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(a.s29_root,a.source_root,a.cloudpickle_root)
    if a.mode=="train":
        if a.receipt!=a.train_receipt:raise ValueError("P30_TRAIN_RECEIPT_MUST_MATCH")
        result=train(gym,a.original_checkpoints,a.original_manifest,a.adapter_dir,a.receipt)
        summary={"mode":"train","status":"PASS_PUBLIC_DEV_TRAINING_ONLY",
                 "actual_pair_trials":sum(e["paired_actually_executed_episodes"] for e in result["summaries"]),
                 "adapter_archives":len(result["entries"]),
                 "science":result["science_first_use"]}
    else:
        result=evaluate(gym,a.original_checkpoints,a.original_manifest,a.adapter_dir,a.train_receipt)
        immutable_receipt(a.receipt,result)
        summary={"mode":"evaluate","status":result["status"],
                 "learned_dual":result["learned_dual_support_of_three"],
                 "reversed_gain":result["learned_reverse_gain_support_of_three"],
                 "wrong_credit_control":result["wrong_credit_negative_support_of_three"],
                 "oracle":result["oracle_both_support_of_three"],
                 "means":[{"seed":e["seed"],"old":e["tasks"]["normal"]["mean_actual_return"],
                           "reverse":e["tasks"]["reversed"]["mean_actual_return"],
                           "bits":e["learned_adapter_map"]}
                          for e in result["per_seed"]],
                 "science":result["science_first_use"]}
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    raise SystemExit(main())
