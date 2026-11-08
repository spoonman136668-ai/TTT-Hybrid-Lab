"""P23: exact restarted optimizer/RNG training and gravity-perturbed factual transfer DEV.

Four distinct Python processes: 32-epoch reference, 16-epoch split,
16-epoch cold continuation, and evaluation/comparison. P18/P19/P20/P21
original source unchanged. No protected science first-use or learner activation.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import struct
import sys
import torch
import numpy as np
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p20_velocity_blind_history_dev as p20

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p23.exact-resume-gravity-dev.v1"
P21_PARENT="c1587ad5012d66a1ee9d368e52d38a6c3e81573c"
MODEL_SEEDS=(2741,2753,2767)
KINDS=("full_history_gru","current_only_gru")
WORLD_SEEDS=(81011,81012,81013,81014,81015,81016,81017,81018)
GRAVITY=(("original",1.),("gravity_x0.75",.75),("gravity_x1.25",1.25))
EPOCHS=32
HALF=16
BATCH=512
LR=.008
MAX_ACTIONS=200


def require_dev():
    if os.environ.get("NATIVE_P23_DEV_ONLY","0")!="1":
        raise RuntimeError("P23_PROTECTED_SCIENCE_FIRST_USE_DENIED")


def candidate_sha():
    sha=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if not re.fullmatch(r"[a-f0-9]{40}",sha):
        raise RuntimeError("P23_EXACT_CANDIDATE_SOURCE_REQUIRED")
    return sha


def frozen():
    q=json.loads((ROOT/"experiments/native_p23_exact_resume_gravity_frozen.json").read_text())
    if (q["schema"]!="ttt-hybrid.native-p23.exact-resume-gravity-frozen.v1"
        or q["source"]["parent_P21_SHA"]!=P21_PARENT
        or q["custody"]["scientific_first_use_authorized"] is not False
        or q["custody"]["original_S26_S27_first_use"] is not False
        or tuple(q["training"]["model_initializations"])!=MODEL_SEEDS
        or tuple(q["training"]["trained_arms"])!=KINDS
        or q["training"]["total_epochs"]!=EPOCHS
        or q["training"]["interruption_after_epoch"]!=HALF
        or q["training"]["AdamW_learning_rate"]!=LR
        or q["training"]["batch_size"]!=BATCH
        or tuple(q["transfer"]["evaluation_world_seeds"])!=WORLD_SEEDS
        or [(c["name"],c["gravity_factor"]) for c in q["transfer"]["environment_conditions"]]!=list(GRAVITY)):
        raise RuntimeError("P23_FROZEN_PROTOCOL_OR_SOURCE_DRIFT")
    if len(set(WORLD_SEEDS))!=len(WORLD_SEEDS) or not set(WORLD_SEEDS).isdisjoint(p20.EVAL_SEEDS):
        raise RuntimeError("P23_EVALUATION_REUSE_DENIED")
    return q


def sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def semantic_digest(obj)->str:
    """Content digest independent of torch.save zip metadata or process identity."""
    h=hashlib.sha256()
    def add(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous()
            h.update(b"T");h.update(str(a.dtype).encode())
            h.update(json.dumps(list(a.shape)).encode())
            h.update(a.numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b"D")
            for key in sorted(v,key=lambda x:str(x)):
                add(key);add(v[key])
            h.update(b"d")
        elif isinstance(v,(list,tuple)):
            h.update(b"L" if isinstance(v,list) else b"U")
            h.update(str(len(v)).encode())
            for x in v:add(x)
        elif isinstance(v,bool):
            h.update(b"B1" if v else b"B0")
        elif isinstance(v,int):
            h.update(b"I");h.update(str(v).encode())
        elif isinstance(v,float):
            h.update(b"F");h.update(struct.pack("!d",v))
        elif isinstance(v,str):
            b=v.encode("utf-8");h.update(b"S");h.update(str(len(b)).encode());h.update(b)
        elif v is None:h.update(b"N")
        else:raise TypeError("P23_UNEXPECTED_STATE_TYPE:"+str(type(v)))
    add(obj)
    return h.hexdigest()


def snapshot_numpy_rng():
    state=np.random.get_state()
    if state[0]!="MT19937":
        raise RuntimeError("P23_WRONG_NUMPY_GENERATOR")
    return {"kind":state[0],"keys":[int(x) for x in state[1]],
            "position":int(state[2]),"has_gauss":int(state[3]),
            "cached_gaussian":float(state[4])}


def restore_numpy_rng(state):
    if (state.get("kind")!="MT19937" or len(state.get("keys",[]))!=624
        or not 0<=state.get("position",-1)<=624):
        raise RuntimeError("P23_NUMPY_STATE_MISSING_OR_INVALID")
    np.random.set_state((state["kind"],np.array(state["keys"],dtype=np.uint32),
                         state["position"],state["has_gauss"],state["cached_gaussian"]))


def input_data(gym):
    tx,ty,tmeta=p19.expert_dataset(gym,p20.TRAIN_SEEDS)
    vx,vy,vmeta=p19.expert_dataset(gym,p20.VAL_SEEDS)
    return (tx,ty,vx,vy,{"training":tmeta,"validation":vmeta})


def actor_input(raw,current):
    return p20.visible_history(raw,current_only=current)


def model_and_optimizer(kind,seed):
    if kind not in KINDS or seed not in MODEL_SEEDS:
        raise ValueError("P23_UNREGISTERED_ACTOR")
    torch.manual_seed(seed)
    random.seed(seed+1234)
    np.random.seed(seed+4321)
    net=p19.new_model("history_gru")
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.)
    sampler=torch.Generator().manual_seed(seed+710000)
    return net,opt,sampler


def run_epochs(net,opt,sampler,x,y,epoch_first,epoch_end,updates):
    for epoch in range(epoch_first,epoch_end):
        net.train()
        ids=torch.randperm(len(y),generator=sampler)
        for start in range(0,len(ids),BATCH):
            idx=ids[start:start+BATCH]
            logits,_=net(x[idx])
            loss=F.cross_entropy(logits,y[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P23_NONFINITE_OR_NONFACTUAL_OBJECTIVE")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
            updates+=1
    net.eval()
    return updates


def factual_input_digest(x,y):
    return semantic_digest({"x":x,"teacher_executed_actions":y,
                            "source_semantics":"P20_MASKED_FACTUAL_PUBLIC_DEVELOPMENT"})


def state_payload(net,opt,sampler,kind,seed,epoch,updates,data_digest):
    return {
        "schema":SCHEMA,"parent_P21_SHA":P21_PARENT,
        "candidate_SHA":candidate_sha(),"kind":kind,"model_seed":seed,
        "completed_epochs":epoch,"optimizer_updates":updates,
        "data_sha256":data_digest,
        "model":net.state_dict(),"optimizer":opt.state_dict(),
        "sampler_rng":sampler.get_state(),"torch_cpu_rng":torch.get_rng_state(),
        "python_rng":random.getstate(),"numpy_rng":snapshot_numpy_rng()}


def state_path(directory,kind,seed):
    return directory/f"p23_{kind}_{seed}.pt"


def save_state(directory,kind,seed,payload):
    path=state_path(directory,kind,seed)
    with path.open("xb") as f:
        torch.save(payload,f)
    archive=path.read_bytes()
    return {"kind":kind,"seed":seed,"filename":path.name,
            "archive_sha256":sha_bytes(archive),
            "semantic_sha256":semantic_digest(payload),
            "model_sha256":p18.model_sha_from_payload(payload) if hasattr(p18,"model_sha_from_payload") else semantic_digest(payload["model"]),
            "epoch":payload["completed_epochs"],"optimizer_updates":payload["optimizer_updates"],
            "data_sha256":payload["data_sha256"]}


def validate_payload(p,kind,seed,epoch,data_digest):
    if (not isinstance(p,dict) or p.get("schema")!=SCHEMA
        or p.get("parent_P21_SHA")!=P21_PARENT
        or p.get("candidate_SHA")!=candidate_sha()
        or p.get("kind")!=kind or p.get("model_seed")!=seed
        or p.get("completed_epochs")!=epoch
        or p.get("data_sha256")!=data_digest):
        raise RuntimeError("P23_SOURCE_SEED_CURSOR_OR_DATA_DRIFT")
    for field in ("model","optimizer","sampler_rng","torch_cpu_rng","python_rng","numpy_rng"):
        if field not in p:raise RuntimeError("P23_MISSING_FULL_RESUME_STATE:"+field)
    if not isinstance(p["optimizer"],dict) or not p["optimizer"].get("state"):
        raise RuntimeError("P23_OPTIMIZER_MOMENTS_MISSING")
    if not isinstance(p["sampler_rng"],torch.Tensor) or not isinstance(p["torch_cpu_rng"],torch.Tensor):
        raise RuntimeError("P23_SAMPLER_OR_TORCH_RNG_MISSING")
    if not isinstance(p["python_rng"],tuple) or not isinstance(p["numpy_rng"],dict):
        raise RuntimeError("P23_HOST_RNG_MISSING")


def load_state(directory,entry,kind,seed,epoch,data_digest):
    if (entry.get("kind")!=kind or entry.get("seed")!=seed
        or entry.get("filename")!=state_path(directory,kind,seed).name
        or entry.get("epoch")!=epoch or entry.get("data_sha256")!=data_digest):
        raise RuntimeError("P23_CHECKPOINT_MANIFEST_MISMATCH")
    raw=state_path(directory,kind,seed).read_bytes()
    if sha_bytes(raw)!=entry.get("archive_sha256"):
        raise RuntimeError("P23_CORRUPTED_CHECKPOINT_DENIED")
    p=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    validate_payload(p,kind,seed,epoch,data_digest)
    if semantic_digest(p)!=entry.get("semantic_sha256"):
        raise RuntimeError("P23_CHECKPOINT_STATE_DIGEST_DRIFT")
    return p


def new_file(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True,indent=2)+"\n")


def read_manifest(path,mode,epoch):
    j=json.loads(path.read_text())
    if (j.get("schema")!=SCHEMA or j.get("mode")!=mode
        or j.get("parent_P21_SHA")!=P21_PARENT
        or j.get("candidate_SHA")!=candidate_sha()
        or j.get("completed_epochs")!=epoch
        or len(j.get("entries",[]))!=6
        or len({(a.get("kind"),a.get("seed")) for a in j.get("entries",[])})!=6):
        raise RuntimeError("P23_MISSING_OR_UNQUALIFIED_MANIFEST")
    return j


def do_training(gym,mode,directory,manifest,prior_dir=None,prior_manifest=None):
    if directory.exists():
        raise RuntimeError("P23_DESTINATION_EXISTS_NO_REPLAY")
    directory.mkdir(parents=True)
    tx,ty,vx,vy,meta=input_data(gym)
    outputs=[]
    previous=None
    if mode=="resume":
        previous=read_manifest(prior_manifest,"split",HALF)
        originals={(r["kind"],r["seed"]):r for r in previous["entries"]}
    for seed in MODEL_SEEDS:
        for kind in KINDS:
            x=actor_input(tx,kind=="current_only_gru")
            digest=factual_input_digest(x,ty)
            net,opt,sampler=model_and_optimizer(kind,seed)
            if mode=="resume":
                prev=load_state(prior_dir,originals[(kind,seed)],kind,seed,HALF,digest)
                net.load_state_dict(prev["model"],strict=True)
                opt.load_state_dict(prev["optimizer"])
                sampler.set_state(prev["sampler_rng"])
                torch.set_rng_state(prev["torch_cpu_rng"])
                random.setstate(prev["python_rng"])
                restore_numpy_rng(prev["numpy_rng"])
                begin=HALF
                updates=prev["optimizer_updates"]
            else:
                begin=0
                updates=0
            end=HALF if mode=="split" else EPOCHS
            updates=run_epochs(net,opt,sampler,x,ty,begin,end,updates)
            expected=(len(ty)+BATCH-1)//BATCH * end
            if updates!=expected:
                raise RuntimeError("P23_SAMPLER_UPDATE_CURSOR_DRIFT")
            state=state_payload(net,opt,sampler,kind,seed,end,updates,digest)
            outputs.append(save_state(directory,kind,seed,state))
    end=HALF if mode=="split" else EPOCHS
    result={
        "schema":SCHEMA,"mode":mode,"parent_P21_SHA":P21_PARENT,
        "candidate_SHA":candidate_sha(),"completed_epochs":end,"entries":outputs,
        "training_teacher_actual_source_actions":meta["training"]["actual_teacher_actions"],
        "dev_only":True,"science_first_use":"NOT_RUN","protected_seed_use":[],
        "full_optimizer_and_cpu_rng_sampler_saved":True,
        "original_P20_P21_weights_not_reused":True}
    new_file(manifest,result)
    return result


def check_corruption_and_negative_controls(directory,manifest):
    entry=manifest["entries"][0]
    raw=state_path(directory,entry["kind"],entry["seed"]).read_bytes()
    broken=bytearray(raw);broken[len(broken)//2]^=1
    checks={
      "one_byte_corruption_rejected":sha_bytes(broken)!=entry["archive_sha256"],
      "wrong_source_rejected":False,
      "missing_optimizer_rejected":False,
      "missing_rng_rejected":False,
      "wrong_epoch_rejected":False,
    }
    p=torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")
    for label,mutation in [
        ("wrong_source_rejected",lambda q:q.update(candidate_SHA="f"*40)),
        ("missing_optimizer_rejected",lambda q:q.pop("optimizer")),
        ("missing_rng_rejected",lambda q:q.pop("sampler_rng")),
        ("wrong_epoch_rejected",lambda q:q.update(completed_epochs=HALF))]:
        q=dict(p);mutation(q)
        try:validate_payload(q,entry["kind"],entry["seed"],EPOCHS,entry["data_sha256"])
        except RuntimeError:checks[label]=True
    if not all(checks.values()):
        raise RuntimeError("P23_FAIL_CLOSED_NEGATIVE_CONTROL_FAILED")
    return checks


def full_actor_from_payload(payload):
    net=p19.new_model("history_gru").eval()
    net.load_state_dict(payload["model"],strict=True)
    return net


def rollout(gym,policy,gravity_factor,world,seed,net=None):
    if world not in WORLD_SEEDS or seed not in MODEL_SEEDS:
        raise ValueError("P23_UNREGISTERED_TRANSFER_WORLD")
    env=gym.make("CartPole-v1",render_mode=None)
    dig=hashlib.sha256()
    try:
        original=float(env.unwrapped.gravity)
        if not math_isfinite(original*gravity_factor):
            raise RuntimeError("P23_INVALID_GRAVITY_VARIANT")
        env.unwrapped.gravity=original*gravity_factor
        obs,_=env.reset(seed=world)
        hist=[p18.factual_row(obs,0,0.)]
        actions=[]
        rewards=[]
        for _ in range(MAX_ACTIONS):
            if policy=="two_frame":
                action=p20.finite_difference_action(hist)
            elif policy=="teacher":
                action=p18.handcoded_action(obs)
            else:
                tensor=p20.visible_rows(hist,current_only=policy=="current_only_gru")
                with torch.no_grad():
                    logits,_=net(tensor)
                action=int(logits.argmax(dim=1).item())
            nxt,reward,terminated,truncated,_=env.step(action)
            reward=float(reward)
            step=p18.factual_row(nxt,2*action-1,reward)
            dig.update(json.dumps([hist[-1],action,step,bool(terminated),bool(truncated)],
                                  separators=(",",":"),allow_nan=False).encode())
            hist.append(step)
            actions.append(action)
            rewards.append(reward)
            obs=nxt
            if terminated or truncated:break
        return {"actual_reward":sum(rewards),"actual_actions":len(actions),
                "chosen_actions":actions,"factual_SHA256":dig.hexdigest(),
                "online_optimizer_updates":0}
    finally:env.close()


def math_isfinite(x):
    import math
    return math.isfinite(x)


def evaluate(gym,reference_dir,reference_manifest,resumed_dir,resumed_manifest):
    ref=read_manifest(reference_manifest,"reference",EPOCHS)
    resumed=read_manifest(resumed_manifest,"resume",EPOCHS)
    controls=check_corruption_and_negative_controls(resumed_dir,resumed)
    references={(e["kind"],e["seed"]):e for e in ref["entries"]}
    resumed_entries={(e["kind"],e["seed"]):e for e in resumed["entries"]}
    if set(references)!=set(resumed_entries):
        raise RuntimeError("P23_REFERENCE_RESUME_MODEL_SET_DRIFT")
    per_seed=[]
    semantic_equal=[]
    trace_equal=[]
    for seed in MODEL_SEEDS:
        pair={}
        for kind in KINDS:
            r=references[(kind,seed)]
            q=resumed_entries[(kind,seed)]
            if r["data_sha256"]!=q["data_sha256"]:
                raise RuntimeError("P23_DIFFERENT_SOURCE_DATA")
            a=load_state(reference_dir,r,kind,seed,EPOCHS,r["data_sha256"])
            b=load_state(resumed_dir,q,kind,seed,EPOCHS,r["data_sha256"])
            matched=semantic_digest(a)==semantic_digest(b)
            semantic_equal.append(matched)
            pair[kind]=(full_actor_from_payload(a),full_actor_from_payload(b))
        variants=[]
        for name,factor in GRAVITY:
            controls_by_world=[]
            per_kind={kind:[] for kind in KINDS}
            for world in WORLD_SEEDS:
                results={}
                for kind in KINDS:
                    aa=rollout(gym,kind,factor,world,seed,pair[kind][0])
                    bb=rollout(gym,kind,factor,world,seed,pair[kind][1])
                    identical=aa==bb
                    trace_equal.append(identical)
                    results[kind]=aa
                    per_kind[kind].append(aa["actual_reward"])
                controls_by_world.append({
                    "world":world,
                    "two_frame":rollout(gym,"two_frame",factor,world,seed),
                    "teacher":rollout(gym,"teacher",factor,world,seed),
                    "trained":results})
            means={kind:sum(per_kind[kind])/len(WORLD_SEEDS) for kind in KINDS}
            means["two_frame"]=sum(v["two_frame"]["actual_reward"] for v in controls_by_world)/len(WORLD_SEEDS)
            means["teacher"]=sum(v["teacher"]["actual_reward"] for v in controls_by_world)/len(WORLD_SEEDS)
            variants.append({"condition":name,"gravity_factor":factor,"mean_actual_reward":means,
                             "full_minus_current_only":means["full_history_gru"]-means["current_only_gru"],
                             "all_original_source_action_outcomes":controls_by_world})
        per_seed.append({"model_seed":seed,"conditions":variants})
    thresholds=frozen()["transfer"]["development_history_gain_threshold"]
    history_positive=sum(all(
        condition["full_minus_current_only"]>=15
        for condition in seed["conditions"] if condition["condition"]!="original")
        for seed in per_seed)
    exact=all(semantic_equal) and all(trace_equal)
    if not exact:
        raise RuntimeError("P23_COLD_RESUMED_OPTIMIZER_OR_ACTUAL_ACTION_DRIFT")
    return {"schema":SCHEMA,"status":"PASS_EXACT_COLD_TRAINING_CONTINUATION_NOT_SCIENCE",
            "exact_reference_resume_state_count":len(semantic_equal),
            "bit_exact_final_model_optimizer_rng_sampler_and_action_states":exact,
            "byte_identical_actual_source_action_outcomes_count":len(trace_equal),
            "negative_controls":controls,"candidate_SHA":candidate_sha(),
            "new_public_development_worlds":list(WORLD_SEEDS),"conditions":list(GRAVITY),
            "prospective_history_support_in_three":history_positive,
            "scientific_first_use":"NOT_RUN","reward_only_learning_claim":False,
            "strongest_conventional_superiority_claim":False,
            "FORGE_owner_activation":False,
            "per_seed":per_seed,
            "limitations":"Exact full CPU optimizer and RNG resumption on freshly trained teacher-supervised P20-equivalent models; altered gravity is an authored variant of the original public MIT source, NOT independently unknown science. Full rollout cost and strongest conventionals reported, no reward-only/cold native online continual learning claim."}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--mode",choices=("reference","split","resume","evaluate"),required=True)
    a.add_argument("--s29-root",type=Path,required=True)
    a.add_argument("--source-root",type=Path,required=True)
    a.add_argument("--cloudpickle-root",type=Path,required=True)
    a.add_argument("--output-dir",type=Path)
    a.add_argument("--output-manifest",type=Path)
    a.add_argument("--split-dir",type=Path)
    a.add_argument("--split-manifest",type=Path)
    a.add_argument("--reference-dir",type=Path)
    a.add_argument("--reference-manifest",type=Path)
    a.add_argument("--resumed-dir",type=Path)
    a.add_argument("--resumed-manifest",type=Path)
    a.add_argument("--receipt",type=Path)
    args=a.parse_args()
    require_dev()
    frozen()
    candidate_sha()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(args.s29_root,args.source_root,args.cloudpickle_root)
    if args.mode in ("reference","split","resume"):
        result=do_training(gym,args.mode,args.output_dir,args.output_manifest,
                           prior_dir=args.split_dir,prior_manifest=args.split_manifest)
        print(json.dumps({"mode":args.mode,"completed_epochs":result["completed_epochs"],
                          "trained_models":len(result["entries"]),"science":result["science_first_use"]},sort_keys=True))
    else:
        result=evaluate(gym,args.reference_dir,args.reference_manifest,
                        args.resumed_dir,args.resumed_manifest)
        new_file(args.receipt,result)
        print(json.dumps({"status":result["status"],
                          "exact_state_models":result["exact_reference_resume_state_count"],
                          "identical_world_action_trajectories":result["byte_identical_actual_source_action_outcomes_count"],
                          "history_gravity_support":result["prospective_history_support_in_three"],
                          "rewards":[{v["condition"]:v["mean_actual_reward"] for v in e["conditions"]} for e in result["per_seed"]],
                          "science":result["scientific_first_use"]},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
