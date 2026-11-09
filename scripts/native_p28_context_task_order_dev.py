"""P28 public DEV: task cue versus training-order bottleneck for contradictory control.

Frozen before implementation. Reuses actual, separately pinned MIT/S29 CartPole
source and original P26 action remap. All newly trained architectures receive
the same actual expert-controlled physical actions and same matched sampled
index streams. No reward-only learning or protected first-use.
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
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p23_exact_training_resume_gravity_dev as p23
from scripts import native_p26_contradictory_action_retention_dev as p26

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p28.context-versus-task-order-dev.v1"
PARENT="3c94ab1d9a5dd82eac268d94880dc535cafc446f"
SEEDS=(2741,2753,2767)
TRAIN_OLD=tuple(range(131000,131048))
TRAIN_NEW=tuple(range(132000,132048))
VALID_OLD=tuple(range(133000,133008))
VALID_NEW=tuple(range(134000,134008))
EVAL_OLD=(135011,135012,135013,135014,135015,135016,135017,135018)
EVAL_NEW=(136011,136012,136013,136014,136015,136016,136017,136018)
ARMS=("joint_history_gru","blocked_history_gru","joint_history_mlp","joint_cue_blind_gru")
EVAL_ARMS=ARMS+("frozen_old_half_gru_plus_XOR","joint_gru_reversed_task_cue",
                "cue_aware_two_frame_strong_conventional","full_public_sensor_teacher")
BATCH=512
STEPS=1216
HALF=STEPS//2
PER_TASK=STEPS*(BATCH//2)


def require_dev():
    if os.environ.get("NATIVE_P28_DEV_ONLY")!="1":
        raise RuntimeError("P28_PROTECTED_SCIENTIFIC_FIRST_USE_DENIED")


def exact_sha():
    s=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if not re.fullmatch("[a-f0-9]{40}",s):
        raise RuntimeError("P28_EXACT_SOURCE_SHA_REQUIRED")
    return s


def frozen():
    q=json.loads((ROOT/"experiments/native_p28_context_task_order_frozen.json").read_text())
    if (q["schema"]!="ttt-hybrid.native-p28.matched-task-order-and-cue-frozen.v1"
        or q["parent"]["P26_SHA"]!=PARENT
        or q["parent"]["P26_script_blob"]!="c6ba33c7d7df1a0c12ebde9e20308e723346dd41"
        or q["training"]["total_updates"]!=STEPS
        or q["training"]["per_task_labels_seen"]!=PER_TASK
        or q["training"]["batch_size"]!=BATCH
        or q["training"]["lr"]!=p23.LR
        or tuple(q["training"]["initialization_seeds"])!=SEEDS
        or tuple(q["task_source"]["evaluation_normal"])!=EVAL_OLD
        or tuple(q["task_source"]["evaluation_reverse"])!=EVAL_NEW
        or q["boundaries"]["scientific_first_use"] is not False
        or q["boundaries"]["FORGE_activation"] is not False):
        raise RuntimeError("P28_SOURCE_OR_FROZEN_PROTOCOL_DRIFT")
    tr=set(TRAIN_OLD)|set(TRAIN_NEW)|set(VALID_OLD)|set(VALID_NEW)
    if len(tr)!=112 or len(set(EVAL_OLD)|set(EVAL_NEW))!=16 or not tr.isdisjoint(set(EVAL_OLD)|set(EVAL_NEW)):
        raise RuntimeError("P28_TRAIN_EVAL_SOURCE_LEAK")
    return q


def sha(b):
    return hashlib.sha256(b).hexdigest()


def source_sets(gym):
    ox,oy,om=p26.factual_demonstrations(gym,TRAIN_OLD,False)
    nx,ny,nm=p26.factual_demonstrations(gym,TRAIN_NEW,True)
    vx,vy,vm=p26.factual_demonstrations(gym,VALID_OLD,False)
    wx,wy,wm=p26.factual_demonstrations(gym,VALID_NEW,True)
    if len(oy)!=len(ny) or len(vy)!=len(wy):
        raise RuntimeError("P28_SAME_TASK_DATA_LENGTH_REQUIRED")
    for x,y in ((ox,oy),(nx,ny),(vx,vy),(wx,wy)):
        if tuple(x.shape[1:])!=(8,6) or len(x)!=len(y) or not torch.isfinite(x).all():
            raise RuntimeError("P28_REAL_SOURCE_DATA_INVALID")
        if not bool(torch.count_nonzero(x[:,:,3])==0):
            raise RuntimeError("P28_ANGULAR_VELOCITY_LEAK")
    return (ox,oy,nx,ny),(vx,vy,wx,wy),{
        "normal_train":om,"reverse_train":nm,
        "normal_validation":vm,"reverse_validation":wm,
        "source_data_SHA256":p23.semantic_digest({"normal_inputs":ox,"normal_actions":oy,
                              "reverse_inputs":nx,"reverse_actions":ny})}


def stream(length,need,seed):
    g=torch.Generator().manual_seed(seed)
    pieces=[]
    remain=need
    while remain:
        ids=torch.randperm(length,generator=g)
        use=min(remain,length)
        pieces.append(ids[:use])
        remain-=use
    return torch.cat(pieces)


def matched_indices(old_count,new_count,seed,steps=STEPS):
    if steps<2 or steps%2:
        raise ValueError("P28_EVEN_UPDATES_REQUIRED")
    n=steps*(BATCH//2)
    old=stream(old_count,n,seed+900001)
    new=stream(new_count,n,seed+900002)
    digest=p23.semantic_digest({"normal_index_stream":old,"reverse_index_stream":new})
    return old,new,digest


def actor(kind,seed):
    if kind not in ARMS or seed not in SEEDS:
        raise ValueError("P28_UNREGISTERED_MODEL")
    torch.manual_seed(seed+(50000 if kind=="joint_history_mlp" else 0))
    m=p19.new_model("history_mlp" if kind=="joint_history_mlp" else "history_gru")
    return m,torch.optim.AdamW(m.parameters(),lr=p23.LR,weight_decay=0.)


def cue_blind(x):
    result=x.clone()
    result[:,:,1]=1.
    return result


def train_actor(kind,seed,old_x,old_y,new_x,new_y,old_ids,new_ids,steps=STEPS):
    if steps%2 or len(old_ids)!=len(new_ids) or len(old_ids)!=steps*(BATCH//2):
        raise RuntimeError("P28_MATCHED_ACTOR_INFORMATION_BUDGET_DRIFT")
    model,opt=actor(kind,seed)
    initial=p23.semantic_digest(model.state_dict())
    old_half=None
    updates=0
    for step in range(steps):
        model.train()
        if kind=="blocked_history_gru":
            pos=(step if step<steps//2 else step-steps//2)*BATCH
            if step<steps//2:
                idx=old_ids[pos:pos+BATCH]
                x,y=old_x[idx],old_y[idx]
            else:
                idx=new_ids[pos:pos+BATCH]
                x,y=new_x[idx],new_y[idx]
        else:
            start=step*(BATCH//2)
            a=old_ids[start:start+BATCH//2]
            b=new_ids[start:start+BATCH//2]
            x=torch.cat((old_x[a],new_x[b]),dim=0)
            y=torch.cat((old_y[a],new_y[b]),dim=0)
        if kind=="joint_cue_blind_gru":
            x=cue_blind(x)
        logits,_=model(x)
        loss=F.cross_entropy(logits,y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("P28_NONFINITE_REAL_SOURCE_TEACHER_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        opt.step()
        updates+=1
        if kind=="blocked_history_gru" and updates==steps//2:
            old_half={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.eval()
    if updates!=steps or initial==p23.semantic_digest(model.state_dict()):
        raise RuntimeError("P28_NO_REAL_PARAMETER_UPDATE")
    return model,opt,old_half,{
        "kind":kind,"seed":seed,"optimizer_updates":updates,"initial_state_SHA256":initial,
        "final_state_SHA256":p23.semantic_digest(model.state_dict()),
        "old_examples_seen":steps*(BATCH//2),
        "reverse_examples_seen":steps*(BATCH//2),
        "state_finite":all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())}


def checkpoint(directory,kind,seed,model,opt,stream_sha,source_sha,data_sha):
    directory.mkdir(parents=True,exist_ok=True)
    name=f"p28_{kind}_{seed}.pt"
    payload={"schema":SCHEMA,"kind":kind,"seed":seed,
             "candidate_sha":source_sha,"parent_sha":PARENT,
             "stream_sha":stream_sha,"data_sha":data_sha,
             "model":model.state_dict(),
             "optimizer":opt.state_dict(),"updates":STEPS}
    file=directory/name
    with file.open("xb") as f:torch.save(payload,f)
    return {"kind":kind,"seed":seed,"file":name,"archive_sha256":sha(file.read_bytes()),
            "semantic_SHA256":p23.semantic_digest(payload),
            "model_SHA256":p23.semantic_digest(payload["model"]),
            "data_SHA256":data_sha,"stream_SHA256":stream_sha,"optimizer_updates":STEPS}


def old_anchor(directory,seed,weights,stream_sha,source_sha,data_sha):
    if not isinstance(weights,dict) or not weights:
        raise RuntimeError("P28_FROZEN_OLD_HALF_ANCHOR_MISSING")
    name=f"p28_old_anchor_{seed}.pt"
    file=directory/name
    blob={"schema":SCHEMA,"kind":"frozen_old_half_gru_plus_XOR","seed":seed,
          "candidate_sha":source_sha,"parent_sha":PARENT,"half_updates":HALF,
          "stream_sha":stream_sha,"data_sha":data_sha,"model":weights}
    with file.open("xb") as f:torch.save(blob,f)
    return {"kind":"frozen_old_half_gru_plus_XOR","seed":seed,"file":name,
            "archive_sha256":sha(file.read_bytes()),"semantic_SHA256":p23.semantic_digest(blob),
            "model_SHA256":p23.semantic_digest(weights),"half_updates":HALF,
            "stream_SHA256":stream_sha,"data_SHA256":data_sha}


def receipt(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:f.write(json.dumps(obj,sort_keys=True,indent=2)+"\n")


def train(gym,directory,manifest):
    if directory.exists():
        raise RuntimeError("P28_NO_UNCERTAIN_EFFECT_TRAINING_REPLAY")
    data,valid,meta=source_sets(gym)
    old_x,old_y,new_x,new_y=data
    directory.mkdir(parents=True,exist_ok=False)
    entries=[]
    metrics=[]
    for seed in SEEDS:
        old_i,new_i,index_sha=matched_indices(len(old_y),len(new_y),seed)
        initial_gru_shas=[]
        for kind in ARMS:
            net,opt,anchor,result=train_actor(kind,seed,*data,old_i,new_i)
            if kind!="joint_history_mlp":
                initial_gru_shas.append(result["initial_state_SHA256"])
            entries.append(checkpoint(directory,kind,seed,net,opt,index_sha,exact_sha(),meta["source_data_SHA256"]))
            if anchor is not None:
                entries.append(old_anchor(directory,seed,anchor,index_sha,exact_sha(),meta["source_data_SHA256"]))
            metrics.append({**result,"index_stream_SHA256":index_sha})
        if len(set(initial_gru_shas))!=1:
            raise RuntimeError("P28_GRU_INITIAL_WEIGHT_NOT_MATCHED")
    if len(entries)!=len(SEEDS)*5:
        raise RuntimeError("P28_CHECKPOINT_SET_INCOMPLETE")
    result={"schema":SCHEMA,"mode":"train","candidate_sha":exact_sha(),"parent_sha":PARENT,
            "entries":entries,"training":metrics,"factual_source":meta,
            "same_indices_exactly_between_all_arms":True,
            "same_old_reverse_sample_counts_and_steps":True,
            "all_source_actions_real":True,"privileged_teacher_not_reward_only":True,
            "science_first_use":"NOT_RUN","protected_science_worlds_consumed":[]}
    receipt(manifest,result)
    return result


def load_model(directory,entry,kind,seed,manifest):
    if (entry.get("kind")!=kind or entry.get("seed")!=seed
        or entry.get("file")!= (f"p28_old_anchor_{seed}.pt" if kind=="frozen_old_half_gru_plus_XOR" else f"p28_{kind}_{seed}.pt")
        or entry.get("data_SHA256")!=manifest["factual_source"]["source_data_SHA256"]):
        raise RuntimeError("P28_COLD_MODEL_IDENTITY_DRIFT")
    b=(directory/entry["file"]).read_bytes()
    if sha(b)!=entry["archive_sha256"]:
        raise RuntimeError("P28_ARCHIVE_CORRUPTION_DENIED")
    state=torch.load(io.BytesIO(b),weights_only=True,map_location="cpu")
    if (state.get("schema")!=SCHEMA or state.get("kind")!=kind or state.get("seed")!=seed
        or state.get("candidate_sha")!=exact_sha() or state.get("parent_sha")!=PARENT
        or state.get("stream_sha")!=entry["stream_SHA256"]
        or state.get("data_sha")!=entry["data_SHA256"]
        or (kind!="frozen_old_half_gru_plus_XOR" and
            (state.get("updates")!=STEPS or not state.get("optimizer",{}).get("state")))
        or (kind=="frozen_old_half_gru_plus_XOR" and state.get("half_updates")!=HALF)):
        raise RuntimeError("P28_CHECKPOINT_FULL_OPTIMIZER_OR_SOURCE_CUSTODY_DRIFT")
    if p23.semantic_digest(state)!=entry["semantic_SHA256"]:
        raise RuntimeError("P28_COLD_STATE_SHA_MISMATCH")
    m=p19.new_model("history_mlp" if kind=="joint_history_mlp" else "history_gru").eval()
    m.load_state_dict(state["model"],strict=True)
    if p23.semantic_digest(m.state_dict())!=entry["model_SHA256"]:
        raise RuntimeError("P28_COLD_WEIGHT_HASH_CHANGED")
    return m


@torch.no_grad()
def old_XOR_rollout(gym,model,reverse,world):
    if world not in (EVAL_NEW if reverse else EVAL_OLD):
        raise ValueError("P28_FROZEN_XOR_BAD_EVALUATION_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    dig=hashlib.sha256()
    try:
        obs,_=env.reset(seed=world)
        hist=[p18.factual_row(obs,0,0.)]
        choices=[]
        physicals=[]
        total=0.
        for _ in range(200):
            # Model trained only for physical action under old normal task.
            # Reframe observed prior selected action into actual physical action,
            # without ever giving learner hidden source physics or future labels.
            x=p26.source_visible(hist,False)
            if reverse:
                x[:,:,4]*=-1.
            logits,_=model(x)
            phys=int(logits.argmax(dim=1).item())
            selected=p26.physical_action(phys,reverse)
            nxt,reward,terminated,truncated,_=env.step(phys)
            new=p18.factual_row(nxt,2*selected-1,float(reward))
            dig.update(json.dumps([hist[-1],selected,phys,new,bool(terminated),bool(truncated)],
                                   separators=(",",":"),allow_nan=False).encode())
            hist.append(new)
            choices.append(selected)
            physicals.append(phys)
            total+=float(reward)
            obs=nxt
            if terminated or truncated:break
        return {"reward":total,"actual_actions":len(physicals),"source_fact_SHA256":dig.hexdigest(),
                "chosen_model_actions":choices,"actual_physical_actions":physicals,
                "eval_optimizer_updates":0}
    finally:env.close()


def eval_model(gym,kind,model,reverse,world):
    if kind=="frozen_old_half_gru_plus_XOR":
        return old_XOR_rollout(gym,model,reverse,world)
    if kind=="joint_gru_reversed_task_cue":
        return p26.evaluate_episode(gym,kind,model,reverse,world,wrong_cue=True)
    if kind=="joint_cue_blind_gru":
        return p26.evaluate_episode(gym,kind,model,reverse,world,wrong_cue=reverse)
    if kind=="cue_aware_two_frame_strong_conventional" or kind=="full_public_sensor_teacher":
        return p26.evaluate_episode(gym,kind,None,reverse,world)
    return p26.evaluate_episode(gym,kind,model,reverse,world)


def evaluate(gym,directory,manifest):
    source=json.loads(manifest.read_text())
    if (source.get("schema")!=SCHEMA or source.get("mode")!="train"
        or source.get("candidate_sha")!=exact_sha() or source.get("parent_sha")!=PARENT
        or len(source.get("entries",[]))!=len(SEEDS)*5
        or source.get("science_first_use")!="NOT_RUN"):
        raise RuntimeError("P28_TRAINING_MANIFEST_NOT_QUALIFIED")
    lookup={(x["kind"],x["seed"]):x for x in source["entries"]}
    if len(lookup)!=len(SEEDS)*5:raise RuntimeError("P28_MISSING_OR_DUPLICATED_MODELS")
    negative={}
    anchor=lookup[("joint_history_gru",SEEDS[0])]
    raw=bytearray((directory/anchor["file"]).read_bytes())
    raw[len(raw)//2]^=1
    negative["corrupt_archive_denied"]=sha(raw)!=anchor["archive_sha256"]
    try:load_model(directory,{**anchor,"seed":SEEDS[1]},"joint_history_gru",SEEDS[0],source)
    except RuntimeError:negative["wrong_seed_denied"]=True
    try:load_model(directory,{**anchor,"stream_SHA256":"0"*64},"joint_history_gru",SEEDS[0],source)
    except RuntimeError:negative["wrong_sampler_denied"]=True
    if len(negative)!=3 or not all(negative.values()):
        raise RuntimeError("P28_COLD_NEGATIVE_CONTROL_NOT_PASSED")
    per_seed=[]
    for seed in SEEDS:
        models={}
        for kind in ARMS+("frozen_old_half_gru_plus_XOR",):
            models[kind]=load_model(directory,lookup[(kind,seed)],kind,seed,source)
        conditions={}
        for name,reverse in (("old",False),("inverted",True)):
            worlds=EVAL_NEW if reverse else EVAL_OLD
            episodes=[]
            for world in worlds:
                results={}
                for arm in EVAL_ARMS:
                    net=(models["joint_history_gru"] if arm=="joint_gru_reversed_task_cue"
                         else models.get(arm))
                    results[arm]=eval_model(gym,arm,net,reverse,world)
                if not all(v.get("eval_optimizer_updates",v.get("online_optimizer_updates",0))==0
                           for v in results.values()):
                    raise RuntimeError("P28_EVALUATION_UPDATED_MODEL")
                episodes.append({"seed":world,"arms":results})
            means={a:sum(w["arms"][a]["reward"] for w in episodes)/len(episodes)
                   for a in EVAL_ARMS}
            conditions[name]={"mean_reward":means,"factual_worlds":episodes}
        old=conditions["old"]["mean_reward"]
        new=conditions["inverted"]["mean_reward"]
        old_xor=old["frozen_old_half_gru_plus_XOR"]
        new_xor=new["frozen_old_half_gru_plus_XOR"]
        per_seed.append({
            "seed":seed,"tasks":conditions,
            "joint_GRU_both_tasks_competent":old["joint_history_gru"]>=150 and new["joint_history_gru"]>=150,
            "joint_MLP_both_tasks_competent":old["joint_history_mlp"]>=150 and new["joint_history_mlp"]>=150,
            "joint_GRU_order_gain":old["joint_history_gru"]>=150 and new["joint_history_gru"]>=150
                 and new["joint_history_gru"]>=new["blocked_history_gru"]+25
                 and old["joint_history_gru"]>=old["blocked_history_gru"]-20,
            "frozen_old_explicit_XOR_both_competent":old_xor>=175 and new_xor>=175,
            "task_cue_reverse_intervention_effect":new["joint_history_gru"]-
                new["joint_gru_reversed_task_cue"],
        })
    support=lambda field:sum(int(bool(r[field])) for r in per_seed)
    joint=support("joint_GRU_both_tasks_competent")
    mlp=support("joint_MLP_both_tasks_competent")
    return {
        "schema":SCHEMA,"status":"PASS_PUBLIC_DEV_ENGINEERING_NOT_SCIENCE",
        "candidate_sha":exact_sha(),"parent_sha":PARENT,
        "old_and_reverse_learner_information_and_updates_matched":True,
        "source_actions_real":True,
        "all_models_cold_restored_and_state_checked":True,
        "negative_controls":negative,
        "joint_GRU_dual_support_of_three":joint,
        "joint_MLP_dual_support_of_three":mlp,
        "joint_order_gain_support_of_three":support("joint_GRU_order_gain"),
        "conventional_but_not_GRU_warning":mlp>=2 and joint<2,
        "frozen_old_explicit_XOR_calibration_support_of_three":support("frozen_old_explicit_XOR_both_competent"),
        "cue_intervention_support_of_three":sum(int(r["task_cue_reverse_intervention_effect"]>=25) for r in per_seed),
        "source_GIT_SHA":exact_sha(),"per_seed":per_seed,
        "science_first_use":"NOT_RUN","protected_science_worlds_consumed":[],
        "teacher_supervised_not_reward_only":True,
        "independent_P27_untouched":True,
        "scientific_superiority":False,"FORGE_owner_activation":False,
        "limitations":"Public CartPole source tasks authored as contradictory actuator mapping; shared privileged full-public-state teacher actions. Context cue vs order and fully trained conventional MLP are DEV causal diagnoses, not independently unfamiliar environments or reward-only action learning. Strong two-frame controller has access to the known task mapping, and comparison is not scientific architecture superiority."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("train","evaluate"),required=True)
    p.add_argument("--s29-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--cloudpickle-root",type=Path,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--train-receipt",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    require_dev()
    frozen()
    exact_sha()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(a.s29_root,a.source_root,a.cloudpickle_root)
    if a.mode=="train":
        if a.receipt!=a.train_receipt:raise ValueError("P28_TRAIN_RECEIPT_MUST_MATCH")
        r=train(gym,a.checkpoint_dir,a.receipt)
        print(json.dumps({"status":"PASS_PUBLIC_DEV_TRAINING_ONLY","models":len(r["entries"]),
                          "updates_per_actor":STEPS,"scientific_first_use":r["science_first_use"]},sort_keys=True))
    else:
        r=evaluate(gym,a.checkpoint_dir,a.train_receipt)
        receipt(a.receipt,r)
        print(json.dumps({"status":r["status"],
                          "joint_GRU":r["joint_GRU_dual_support_of_three"],
                          "joint_MLP":r["joint_MLP_dual_support_of_three"],
                          "order_gain":r["joint_order_gain_support_of_three"],
                          "explicit_XOR":r["frozen_old_explicit_XOR_calibration_support_of_three"],
                          "cue_gain":r["cue_intervention_support_of_three"],
                          "means":[{"seed":z["seed"],"old":z["tasks"]["old"]["mean_reward"],
                                    "reverse":z["tasks"]["inverted"]["mean_reward"]}
                                   for z in r["per_seed"]],"science":r["science_first_use"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
