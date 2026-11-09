"""P26 DEV: preserve old control under conflicting source-grounded actuator remap.

Original MIT CartPole-v1 runs actual physical actions. Application-level inversion
changes selected-versus-physical actuator semantics; teacher action labels have
additional full-public-state information and DO NOT constitute reward-only learning.
New cohort only, original P18-P25 and sealed science untouched.
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
import sys
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import native_p18_real_cartpole_factual_dev as p18
from scripts import native_p19_expert_calibration_dev as p19
from scripts import native_p20_velocity_blind_history_dev as p20
from scripts import native_p23_exact_training_resume_gravity_dev as p23

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p26.contradictory-action-retention-dev.v1"
P23_SHA="2d6f452b64d47c873400b33aaa7737966f67d225"
SEEDS=(2741,2753,2767)
TRAIN_OLD=tuple(range(111000,111048))
TRAIN_NEW=tuple(range(112000,112048))
VALID_OLD=tuple(range(113000,113008))
VALID_NEW=tuple(range(114000,114008))
EVAL_OLD=(115011,115012,115013,115014,115015,115016,115017,115018)
EVAL_NEW=(116011,116012,116013,116014,116015,116016,116017,116018)
ARMS=("base_immutable","new_only","true_rehearsal","wrong_old_labels_rehearsal")
CONTEXTS=(("normal",False),("reversed",True))
EPOCHS=32
BATCH=512
MAX_ACTIONS=200


def require_dev():
    if os.environ.get("NATIVE_P26_DEV_ONLY")!="1":
        raise RuntimeError("P26_SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED")


def candidate_sha():
    x=os.environ.get("EXPECTED_CANDIDATE_SHA","")
    if not re.fullmatch("[a-f0-9]{40}",x):
        raise RuntimeError("P26_EXACT_SOURCE_SHA_REQUIRED")
    return x


def frozen():
    q=json.loads((ROOT/"experiments/native_p26_contradictory_action_retention_frozen.json").read_text())
    if (q["schema"]!="ttt-hybrid.native-p26.contradictory-action-retention-frozen.v1"
        or q["source"]["P23_exact_SHA"]!=P23_SHA
        or q["qualification"]["science_first_use_authorized"] is not False
        or q["qualification"]["protected_S26_S27_source_unconsumed"] is not True
        or tuple(q["training"]["model_initializations"])!=SEEDS
        or tuple(q["training"]["arms"])!=ARMS
        or q["training"]["base_epochs_normal_only"]!=EPOCHS
        or q["training"]["adaptation_epochs"]!=EPOCHS
        or q["training"]["batch_size"]!=BATCH
        or q["training"]["lr"]!=p23.LR
        or tuple(q["world"]["normal_evaluation_seeds"])!=EVAL_OLD
        or tuple(q["world"]["reverse_evaluation_seeds"])!=EVAL_NEW):
        raise RuntimeError("P26_FROZEN_SCIENCE_OR_SOURCE_DRIFT")
    training=set(TRAIN_OLD)|set(TRAIN_NEW)|set(VALID_OLD)|set(VALID_NEW)
    if (len(training)!=112 or len(set(EVAL_OLD)|set(EVAL_NEW))!=16
        or not training.isdisjoint(set(EVAL_OLD)|set(EVAL_NEW))
        or not set(EVAL_OLD).isdisjoint(p23.WORLD_SEEDS)):
        raise RuntimeError("P26_TRAIN_EVAL_LEAK")
    return q


def source_visible(history,reverse):
    if not isinstance(reverse,bool):
        raise ValueError("P26_TASK_CUE_MUST_BE_EXPLICIT")
    x=p20.visible_rows(history)
    # CartPole velocity 1 is removed and replaced by a task indicator;
    # angular velocity 3 remains masked for the entire genuine history.
    x[:,:,1]=-1. if reverse else 1.
    if torch.count_nonzero(x[:,:,3])!=0:
        raise RuntimeError("P26_HIDDEN_VELOCITY_LEAK")
    return x


def physical_action(model_action:int,reverse:bool):
    if model_action not in (0,1) or not isinstance(reverse,bool):
        raise ValueError("P26_UNREGISTERED_ACTUATOR_MAPPING")
    return model_action ^ int(reverse)


def factual_demonstrations(gym,seeds,reverse):
    if len(seeds)!=len(set(seeds)) or not seeds:
        raise ValueError("P26_DUPLICATE_FROZEN_SOURCE_SEEDS")
    inputs,labels=[],[]
    digest=hashlib.sha256()
    real_actions=0
    for seed in seeds:
        env=gym.make("CartPole-v1",render_mode=None)
        try:
            obs,_=env.reset(seed=int(seed))
            history=[p18.factual_row(obs,0,0.)]
            for _ in range(MAX_ACTIONS):
                x=source_visible(history,reverse)[0]
                expert_physical=p18.handcoded_action(obs)
                chosen=physical_action(expert_physical,reverse)
                assert physical_action(chosen,reverse)==expert_physical
                inputs.append(x)
                labels.append(chosen)
                nxt,reward,terminated,truncated,_=env.step(expert_physical)
                reward=float(reward)
                after=p18.factual_row(nxt,2*chosen-1,reward)
                digest.update(json.dumps(
                    [int(seed),history[-1],chosen,expert_physical,after,bool(terminated),bool(truncated)],
                    separators=(",",":"),allow_nan=False).encode())
                history.append(after)
                obs=nxt
                real_actions+=1
                if terminated or truncated:break
        finally:env.close()
    return torch.stack(inputs),torch.tensor(labels,dtype=torch.long),{
        "factual_physical_actions_executed":real_actions,
        "num_source_reset_episodes":len(seeds),
        "source_selected_and_physical_action_SHA256":digest.hexdigest(),
        "actuator_reversed":reverse,
        "expert_supervision_not_reward_only":True,
        "source_seeds":list(seeds)}


def archive_path(directory,kind,seed):
    if kind not in ARMS or seed not in SEEDS:
        raise ValueError("P26_INVALID_ARCHIVE_NAME")
    return directory/f"p26_{kind}_{seed}.pt"


def save_checkpoint(directory,kind,seed,model,opt,gen,epochs,updates,data_sha):
    payload={
        "schema":SCHEMA,"candidate_sha":candidate_sha(),"parent_P23_sha":P23_SHA,
        "kind":kind,"seed":seed,"epoch":epochs,"updates":updates,"data_sha256":data_sha,
        "model":model.state_dict(),"optimizer":opt.state_dict(),
        "sampler_rng":gen.get_state(),"torch_rng":torch.get_rng_state(),
        "python_rng":random.getstate(),"numpy_rng":p23.snapshot_numpy_rng()}
    path=archive_path(directory,kind,seed)
    with path.open("xb") as f:torch.save(payload,f)
    return {"kind":kind,"seed":seed,"file":path.name,"archive_sha256":p23.sha_bytes(path.read_bytes()),
            "semantic_sha256":p23.semantic_digest(payload),
            "model_sha256":p23.semantic_digest(payload["model"]),
            "data_sha256":data_sha,"epoch":epochs,"optimizer_updates":updates}


def load_checkpoint(directory,item,kind,seed,epochs,data_sha):
    if (item.get("kind")!=kind or item.get("seed")!=seed
        or item.get("file")!=archive_path(directory,kind,seed).name
        or item.get("epoch")!=epochs or item.get("data_sha256")!=data_sha):
        raise RuntimeError("P26_COLD_ARCHIVE_MANIFEST_MISMATCH")
    blob=archive_path(directory,kind,seed).read_bytes()
    if p23.sha_bytes(blob)!=item.get("archive_sha256"):
        raise RuntimeError("P26_COLD_ARCHIVE_CORRUPTION")
    payload=torch.load(io.BytesIO(blob),weights_only=True,map_location="cpu")
    if (payload.get("schema")!=SCHEMA
        or payload.get("candidate_sha")!=candidate_sha()
        or payload.get("parent_P23_sha")!=P23_SHA
        or payload.get("kind")!=kind or payload.get("seed")!=seed
        or payload.get("epoch")!=epochs or payload.get("data_sha256")!=data_sha
        or not payload.get("optimizer",{}).get("state")
        or not isinstance(payload.get("sampler_rng"),torch.Tensor)
        or not isinstance(payload.get("torch_rng"),torch.Tensor)
        or not isinstance(payload.get("python_rng"),tuple)
        or not isinstance(payload.get("numpy_rng"),dict)):
        raise RuntimeError("P26_COLD_SOURCE_OPTIMIZER_RNG_OR_CURSOR_MISMATCH")
    if p23.semantic_digest(payload)!=item.get("semantic_sha256"):
        raise RuntimeError("P26_COLD_STATE_SEMANTIC_DIGEST_MISMATCH")
    return payload


def new_receipt(path,record):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(json.dumps(record,sort_keys=True,indent=2)+"\n")


def read_receipt(path,mode,count):
    record=json.loads(path.read_text())
    if (record.get("schema")!=SCHEMA or record.get("mode")!=mode
        or record.get("candidate_sha")!=candidate_sha()
        or record.get("source_parent_P23_sha")!=P23_SHA
        or len(record.get("entries",[]))!=count):
        raise RuntimeError("P26_PRIOR_EXPERIMENT_RECEIPT_DRIFT")
    return record


def model_initial(seed):
    model,opt,sampler=p23.model_and_optimizer("full_history_gru",seed)
    return model,opt,sampler


def train_base(gym,directory,receipt):
    if directory.exists():raise RuntimeError("P26_NO_REPLAY_BASE_EXISTS")
    directory.mkdir(parents=True)
    x,y,meta=factual_demonstrations(gym,TRAIN_OLD,False)
    valid_x,valid_y,validation=factual_demonstrations(gym,VALID_OLD,False)
    data_sha=p23.semantic_digest({"old":x,"actions":y})
    entries=[]
    for seed in SEEDS:
        model,opt,gen=model_initial(seed)
        n=p23.run_epochs(model,opt,gen,x,y,0,EPOCHS,0)
        if n!=(len(y)+BATCH-1)//BATCH*EPOCHS:
            raise RuntimeError("P26_PRETRAIN_UPDATES_DRIFT")
        entries.append(save_checkpoint(directory,"base_immutable",seed,model,opt,gen,EPOCHS,n,data_sha))
    out={"schema":SCHEMA,"mode":"base","candidate_sha":candidate_sha(),
         "source_parent_P23_sha":P23_SHA,"entries":entries,"train_data_sha256":data_sha,
         "factual_train":meta,"factual_validation":validation,
         "science_first_use":"NOT_RUN","reward_only_control_learning":False}
    new_receipt(receipt,out)
    return out


def adapt_update(net,opt,new_x,new_y,old_x,old_y,seed,kind):
    if kind not in ("new_only","true_rehearsal","wrong_old_labels_rehearsal"):
        raise ValueError("P26_INVALID_ADAPTATION_ARM")
    if len(new_y)!=len(old_y):
        raise RuntimeError("P26_UNBALANCED_REAL_EXPERT_DEMONSTRATIONS")
    torch.manual_seed(seed+170000)
    gen=torch.Generator().manual_seed(seed+7100000)
    batch_count=(len(new_y)+BATCH-1)//BATCH
    updates=0
    for _ in range(EPOCHS):
        net.train()
        ni=torch.randperm(len(new_y),generator=gen)
        oi=torch.randperm(len(old_y),generator=gen)
        for b in range(batch_count):
            if kind=="new_only":
                idx=ni[b*BATCH:(b+1)*BATCH]
                inputs,labels=new_x[idx],new_y[idx]
            else:
                new_idx=ni[b*(BATCH//2):(b+1)*(BATCH//2)]
                old_idx=oi[b*(BATCH//2):(b+1)*(BATCH//2)]
                inputs=torch.cat((new_x[new_idx],old_x[old_idx]))
                old_label=old_y[old_idx]
                if kind=="wrong_old_labels_rehearsal":
                    old_label=1-old_label
                labels=torch.cat((new_y[new_idx],old_label))
            logits,_=net(inputs)
            loss=F.cross_entropy(logits,labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P26_NONFINITE_EXPERT_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
            updates+=1
    net.eval()
    return gen,updates


def adapt(gym,base_dir,base_receipt,directory,receipt):
    if directory.exists():raise RuntimeError("P26_NO_REPLAY_ADAPT_EXISTS")
    directory.mkdir(parents=True)
    base=read_receipt(base_receipt,"base",3)
    old_x,old_y,old_meta=factual_demonstrations(gym,TRAIN_OLD,False)
    new_x,new_y,new_meta=factual_demonstrations(gym,TRAIN_NEW,True)
    if old_x.shape[0]!=new_x.shape[0]:
        raise RuntimeError("P26_OLD_NEW_DEMOS_UNEQUAL")
    old_hash=p23.semantic_digest({"old":old_x,"actions":old_y})
    if old_hash!=base["train_data_sha256"]:
        raise RuntimeError("P26_OLD_FACTUAL_TRAIN_SET_DRIFT")
    new_hash=p23.semantic_digest({"new":new_x,"actions":new_y})
    old_base={e["seed"]:e for e in base["entries"]}
    entries=[]
    for seed in SEEDS:
        saved=load_checkpoint(base_dir,old_base[seed],"base_immutable",seed,EPOCHS,old_hash)
        for kind in ARMS[1:]:
            model,opt,sampler=model_initial(seed)
            model.load_state_dict(saved["model"],strict=True)
            opt.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            p23.restore_numpy_rng(saved["numpy_rng"])
            gen,added=adapt_update(model,opt,new_x,new_y,old_x,old_y,seed,kind)
            if added!=(len(new_y)+BATCH-1)//BATCH*EPOCHS:
                raise RuntimeError("P26_ADAPT_EQUAL_UPDATE_BUDGET_DRIFT")
            original_updates=saved["updates"]
            data_sha=p23.semantic_digest({"old":old_hash,"new":new_hash,"kind":kind,
                                         "samples_per_update":BATCH})
            entries.append(save_checkpoint(directory,kind,seed,model,opt,gen,EPOCHS+EPOCHS,
                                           original_updates+added,data_sha))
    if len(entries)!=9:raise RuntimeError("P26_ADAPT_MODEL_COUNT_WRONG")
    out={"schema":SCHEMA,"mode":"adapt","candidate_sha":candidate_sha(),
         "source_parent_P23_sha":P23_SHA,"entries":entries,"old_data_sha256":old_hash,
         "new_data_sha256":new_hash,"adaptation_updates_each":added,
         "same_new_and_old_sample_stream_for_rehearsal_and_wrong_label":True,
         "equal_optimizer_update_count_not_equal_new_information":True,
         "factual_old":old_meta,"factual_new":new_meta,
         "science_first_use":"NOT_RUN","reward_only_control_learning":False}
    new_receipt(receipt,out)
    return out


@torch.no_grad()
def evaluate_episode(gym,kind,model,reverse,seed,wrong_cue=False):
    if seed not in (EVAL_NEW if reverse else EVAL_OLD):
        raise ValueError("P26_UNREGISTERED_EVALUATION_RESET")
    env=gym.make("CartPole-v1",render_mode=None)
    trace=hashlib.sha256()
    try:
        obs,_=env.reset(seed=seed)
        hist=[p18.factual_row(obs,0,0.)]
        chosen_actions=[]
        physical_actions=[]
        rewards=[]
        for _ in range(MAX_ACTIONS):
            if kind=="cue_aware_two_frame_strong_conventional":
                physical=p20.finite_difference_action(hist)
                selected=physical_action(physical,reverse)
            elif kind=="full_public_sensor_teacher":
                physical=p18.handcoded_action(obs)
                selected=physical_action(physical,reverse)
            else:
                if model is None:raise ValueError("P26_TRAINED_MODEL_MISSING")
                x=source_visible(hist,(not reverse) if wrong_cue else reverse)
                logits,_=model(x)
                selected=int(logits.argmax(dim=1).item())
                physical=physical_action(selected,reverse)
            nxt,reward,terminated,truncated,_=env.step(physical)
            reward=float(reward)
            row=p18.factual_row(nxt,2*selected-1,reward)
            trace.update(json.dumps([hist[-1],selected,physical,row,bool(terminated),bool(truncated)],
                                    separators=(",",":"),allow_nan=False).encode())
            hist.append(row)
            chosen_actions.append(selected)
            physical_actions.append(physical)
            rewards.append(reward)
            obs=nxt
            if terminated or truncated:break
        return {"reward":sum(rewards),"chosen_model_actions":chosen_actions,
                "actual_physical_actions":physical_actions,
                "actual_external_steps":len(rewards),"source_fact_SHA256":trace.hexdigest(),
                "online_optimizer_updates":0}
    finally:env.close()


def evaluation(gym,base_dir,base_receipt,adapt_dir,adapt_receipt):
    base=read_receipt(base_receipt,"base",3)
    adapt=read_receipt(adapt_receipt,"adapt",9)
    old_hash=base["train_data_sha256"]
    lookup={(e["kind"],e["seed"]):e for e in adapt["entries"]}
    if len(lookup)!=9:raise RuntimeError("P26_ADAPT_MISSING_ARCHIVES")
    base_lookup={e["seed"]:e for e in base["entries"]}
    negatives={}
    item=base_lookup[SEEDS[0]]
    corrupted=bytearray((base_dir/item["file"]).read_bytes())
    corrupted[len(corrupted)//2]^=1
    negatives["changed_archive_hash_denied"]=p23.sha_bytes(corrupted)!=item["archive_sha256"]
    try:
        load_checkpoint(base_dir,{**item,"data_sha256":"X"*64},"base_immutable",SEEDS[0],EPOCHS,old_hash)
    except RuntimeError:negatives["wrong_source_data_denied"]=True
    try:source_visible([p18.factual_row([0.,0.,0.,0.],0,0.)],"bad")
    except ValueError:negatives["invalid_task_cue_denied"]=True
    if len(negatives)!=3 or not all(negatives.values()):
        raise RuntimeError("P26_NEGATIVE_GATES_FAILED")
    seed_reports=[]
    for seed in SEEDS:
        model={}
        for kind in ARMS:
            if kind=="base_immutable":
                e=base_lookup[seed]
                source=load_checkpoint(base_dir,e,kind,seed,EPOCHS,old_hash)
            else:
                e=lookup[(kind,seed)]
                data_sha=p23.semantic_digest({"old":old_hash,"new":adapt["new_data_sha256"],
                                               "kind":kind,"samples_per_update":BATCH})
                source=load_checkpoint(adapt_dir,e,kind,seed,2*EPOCHS,data_sha)
            n=p19.new_model("history_gru").eval()
            n.load_state_dict(source["model"],strict=True)
            model[kind]=n
        results={}
        for name,reverse in CONTEXTS:
            worlds=EVAL_NEW if reverse else EVAL_OLD
            episodes=[]
            for world in worlds:
                cases={kind:evaluate_episode(gym,kind,model[kind],reverse,world) for kind in ARMS}
                cases["cue_aware_two_frame_strong_conventional"]=evaluate_episode(gym,"cue_aware_two_frame_strong_conventional",None,reverse,world)
                cases["full_public_sensor_teacher"]=evaluate_episode(gym,"full_public_sensor_teacher",None,reverse,world)
                if not all(v["online_optimizer_updates"]==0 for v in cases.values()):
                    raise RuntimeError("P26_ONLINE_EVAL_TRAINING_FORBIDDEN")
                episodes.append({"world":world,"arms":cases})
            means={arm:sum(e["arms"][arm]["reward"] for e in episodes)/len(episodes)
                   for arm in (ARMS+("cue_aware_two_frame_strong_conventional","full_public_sensor_teacher"))}
            results[name]={"mean_real_episode_reward":means,
                           "real_factual_world_trajectories":episodes}
        o=results["normal"]["mean_real_episode_reward"]
        n=results["reversed"]["mean_real_episode_reward"]
        decision=(o["base_immutable"]>=175 and o["true_rehearsal"]>=150
                  and n["true_rehearsal"]>=150
                  and o["true_rehearsal"]-o["new_only"]>=20
                  and o["true_rehearsal"]-o["wrong_old_labels_rehearsal"]>=20)
        seed_reports.append({"model_seed":seed,"old_skill":results["normal"],
                             "new_conflicting_skill":results["reversed"],
                             "old_rehearsal_minus_new_only":o["true_rehearsal"]-o["new_only"],
                             "old_rehearsal_minus_wrong_old_labels":
                                 o["true_rehearsal"]-o["wrong_old_labels_rehearsal"],
                             "frozen_dual_skill_and_retention_supported":decision})
    support=sum(r["frozen_dual_skill_and_retention_supported"] for r in seed_reports)
    return {"schema":SCHEMA,"status":"PASS_PUBLIC_DEV_ENGINEERING_NOT_SCIENCE",
            "candidate_sha":candidate_sha(),"source_parent_P23_sha":P23_SHA,
            "frozen_two_of_three_hypothesis_supported":support>=2,
            "supporting_initializations_of_three":support,
            "source_actuator_intervention_real":True,
            "original_external_engine_unchanged":True,
            "teacher_supervised_not_reward_only":True,
            "matched_update_counts_but_rehearsal_new_information_half":True,
            "strongest_cue_aware_nonlearned_baseline_included":True,
            "new_cold_model_state_archive_provenance":True,
            "science_first_use":"NOT_RUN","sealed_original_science_seeds_consumed":[],
            "scientific_superiority":False,"FORGE_owner_activation":False,
            "negative_controls":negatives,"per_seed":seed_reports,
            "limitations":"New expert-supervised continuous GRU cohort. The reverse actuator task is explicitly authored, not independently unfamiliar-world science. Rehearsal uses half as many NEW examples per optimizer step as new-only. Strong source-state expert and conventional two-frame comparator may saturate 200. No reward-only learning, long-horizon autonomy, cross-source scientific generalization or formal SCIENTIFIC_PASS."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("base","adapt","evaluate"),required=True)
    p.add_argument("--s29-root",type=Path,required=True)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--cloudpickle-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--base-dir",type=Path)
    p.add_argument("--base-receipt",type=Path)
    p.add_argument("--adapt-dir",type=Path)
    p.add_argument("--adapt-receipt",type=Path)
    args=p.parse_args()
    require_dev()
    frozen()
    candidate_sha()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,_=p18.admit_gymnasium(args.s29_root,args.source_root,args.cloudpickle_root)
    if args.mode=="base":
        out=train_base(gym,args.output_dir,args.receipt)
        summary={"mode":args.mode,"models":len(out["entries"]),"science":out["science_first_use"]}
    elif args.mode=="adapt":
        out=adapt(gym,args.base_dir,args.base_receipt,args.output_dir,args.receipt)
        summary={"mode":args.mode,"models":len(out["entries"]),
                 "per_actor_updates":out["adaptation_updates_each"],
                 "science":out["science_first_use"]}
    else:
        out=evaluation(gym,args.base_dir,args.base_receipt,args.adapt_dir,args.adapt_receipt)
        new_receipt(args.receipt,out)
        summary={"mode":args.mode,"status":out["status"],
                 "retention_support":out["supporting_initializations_of_three"],
                 "scores":[{"seed":v["model_seed"],
                            "old":v["old_skill"]["mean_real_episode_reward"],
                            "reverse":v["new_conflicting_skill"]["mean_real_episode_reward"]}
                           for v in out["per_seed"]],"science":out["science_first_use"]}
    print(json.dumps(summary,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
