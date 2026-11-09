"""P36: immutable P35 models choose real MountainCar one-step interventions.
No training, optimizer, privileged velocity, force input, or long-horizon MPC.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p35_mountaincar_factual_aux_ppo_dev as p35
SCHEMA="ttt-hybrid.native-p36.original-source-intervention-readonly-dev.v1"
PROTOCOL=ROOT/"experiments/native_p36_cold_action_intervention_frozen.json"
SEEDS=(5209,5227,5237)
EVAL_SEEDS=(195011,195019,195029,195037,195049,195059,195071,195089)
PROBE_SEEDS=(196011,196019,196029,196037,196049,196059,196071,196089)
WORLDS=("original","low_force_x0.75","high_force_x1.25")
ARMS=("factual_aux_energy_choice","shuffled_aux_energy_choice",
      "permuted_factual_action_credit_energy_choice",
      "factual_original_trained_actor_greedy",
      "two_frame_public_momentum_rule","seeded_random")
MODEL_ARMS=("factual_action_aux_ppo","shuffled_action_aux_ppo")
ARCHIVE_SHA="4e0caf6566fb9365b8d98ba57a21de80b113e5e669a76443c7079624808f0138"

def frozen():
    d=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (d.get("schema")!="ttt-hybrid.native-p36.cold-action-intervention-frozen.v1"
        or d["phase"]!="FROZEN_BEFORE_NEW_SOURCE_FIRST_USE"
        or d["parent"]["sha"]!="6cbd0a9b390e793df3f24a09a9ccc66b5011ca5d"
        or d["parent"]["archive_sha256"]!=ARCHIVE_SHA
        or d["source"]["trained_model_seeds"]!=list(SEEDS)
        or d["source"]["mountaincar_blob"]!="5e7e11518715197974e8df46a842f1bc66ccd88c"
        or d["probe"]["seeds"]!=list(PROBE_SEEDS)
        or d["heldout"]["seeds"]!=list(EVAL_SEEDS)
        or d["intervention"]["arms"]!=list(ARMS)
        or d["safety"]["model_parameter_updates"]!=0
        or d["safety"]["protected_science_first_use"] is not False
        or d["thresholds"]["no_postresult_tuning"] is not True):
        raise RuntimeError("P36_ORIGINAL_SOURCE_FROZEN_DATA_AND_AUTHORITY_DRIFT")
    return d

def guard():
    if os.environ.get("NATIVE_P36_DEV_ONLY")!="1":
        raise RuntimeError("P36_PROTECTED_SCIENCE_FIRST_USE_NOT_AUTHORIZED")
    return frozen()

def sha_file(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def load_previous_owner(root):
    root=root.resolve()
    archive=root/"native_p35r2_owner_receipts.zip"
    priorfile=root/"native_p35r2_reward.json"
    if (not archive.is_file() or sha_file(archive)!=ARCHIVE_SHA
        or not priorfile.is_file()):
        raise RuntimeError("P36_P35_ORIGINAL_OWNER_ARCHIVE_UNAVAILABLE_OR_SHA_CHANGED")
    prev=json.loads(priorfile.read_text(encoding="utf-8"))
    if (prev.get("schema")!=p35.SCHEMA
        or prev.get("engineering")!="PASS_PUBLIC_ORIGINAL_SOURCE_DEV"
        or prev.get("real_source_training_actions")!=884736
        or prev.get("real_source_optimizer_minibatches")!=10368
        or len(prev.get("records",[]))!=3):
        raise RuntimeError("P36_ORIGINAL_P35_TRAINING_RECEIPT_UNQUALIFIED")
    weights=root/"native_p35r2_weights"
    models={};receipts={}
    for rec in prev["records"]:
        seed=rec["model_seed"]
        if seed not in SEEDS or seed in models:
            raise RuntimeError("P36_UNEXPECTED_PRIOR_MODEL_SEED")
        models[seed]={};receipts[seed]={}
        for arm in MODEL_ARMS:
            info=rec["checkpoints"][arm]
            path=weights/info["filename"]
            if (path.parent.resolve()!=weights.resolve()
                or not path.is_file()
                or sha_file(path)!=info["checkpoint_file_SHA256"]):
                raise RuntimeError("P36_PRIOR_CHECKPOINT_FILE_SHA_CHANGED")
            m=p35.make_model(seed,arm)
            m.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=True)
            m.eval()
            if p20.model_sha(m)!=info["trained_weights_SHA256"]:
                raise RuntimeError("P36_PRIOR_ORIGINAL_TRAINED_WEIGHT_SHA_CHANGED")
            models[seed][arm]=m
            receipts[seed][arm]=dict(checkpoint_sha256=sha_file(path),
                                     original_state_sha256=p20.model_sha(m))
    if set(models)!=set(SEEDS):
        raise RuntimeError("P36_PRIOR_MODEL_COHORT_INCOMPLETE")
    return models,receipts

@torch.no_grad()
def effects(m,history,flipped=False):
    x=p35.past_tensor(history)
    h,_=m.recurrent(x)
    labels=[2,1,0] if flipped else [0,1,2]
    action_onehot=F.one_hot(torch.tensor(labels),num_classes=3).float()
    z=m.aux_predictor(torch.cat((h[:,-1,:].expand(3,-1),action_onehot),dim=1))
    pred=tuple(float(v) for v in z.reshape(-1).tolist())
    if len(pred)!=3 or any(not math.isfinite(v) for v in pred):
        raise RuntimeError("P36_PREDICTED_SOURCE_INTERVENTION_NOT_FINITE")
    return pred

def action_select(m,history,flipped=False):
    x=history[-1][3]*.07
    candidates=effects(m,history,flipped)
    return max(range(3),key=lambda a:((x+candidates[a]*.0035)**2,a))

@torch.no_grad()
def episode(gym,seed,model_seed,world,arm,models):
    if (seed not in EVAL_SEEDS or model_seed not in SEEDS
        or world not in WORLDS or arm not in ARMS):
        raise ValueError("P36_EVAL_SOURCE_NOT_PRECOMMITTED")
    env=p35.source_env(gym,world)
    try:
        obs,_=env.reset(seed=seed)
        h=[p35.row(obs[0])]
        initial=h[0]
        rng=random.Random(seed+19991*model_seed)
        actions=[];reward_total=0.;goal=False;sha=hashlib.sha256()
        for step in range(200):
            if arm=="two_frame_public_momentum_rule":
                action=p35.rule(h)
            elif arm=="seeded_random":
                action=rng.randrange(3)
            elif arm=="factual_original_trained_actor_greedy":
                logits,_=models["factual_action_aux_ppo"](p35.past_tensor(h))
                action=int(logits.argmax(dim=1).item())
            else:
                name="shuffled_action_aux_ppo" if arm=="shuffled_aux_energy_choice" else "factual_action_aux_ppo"
                action=action_select(models[name],h,
                     flipped=(arm=="permuted_factual_action_credit_energy_choice"))
            nxt,r,terminated,truncated=p35.actual_step(env,h,action)
            sha.update(json.dumps([h[-1],action,nxt,terminated,truncated],
                       separators=(",",":"),allow_nan=False).encode())
            actions.append(action);reward_total+=r;h.append(nxt)
            if terminated or truncated:
                goal=bool(terminated)
                break
        return {"world":world,"source_seed":seed,"model_seed":model_seed,"arm":arm,
                "initial_public_source_observation":initial,
                "real_source_actions":len(actions),
                "source_actual_action_choices":actions,
                "true_real_source_reward":reward_total,
                "actual_original_goal_reached":goal,
                "real_source_action_consequence_sha256":sha.hexdigest(),
                "source_training_updates":0,
                "source_privileged_velocity_seen_by_policy":False}
    finally:
        env.close()

def paired_probe(gym,world,seed,index):
    """Three independent source instances; reset and replay same actual actions."""
    if world not in WORLDS or seed not in PROBE_SEEDS or index not in range(8):
        raise ValueError("P36_CAUSAL_PROBE_NOT_PRECOMMITTED")
    rng=random.Random(seed+1977)
    prefix=[rng.randrange(3) for _ in range(index)]
    observations=[];values=[];actions=0;trace=hashlib.sha256()
    for test_action in (0,1,2):
        env=p35.source_env(gym,world)
        try:
            first,_=env.reset(seed=seed)
            history=[p35.row(first[0])]
            for a in prefix:
                nxt,_,terminated,truncated=p35.actual_step(env,history,a)
                if terminated or truncated:
                    raise RuntimeError("P36_PAIRED_WORLD_PROBE_PREMATURE_TERMINATION")
                history.append(nxt);actions+=1
            before=tuple(history)
            old_delta=history[-1][3]*.07
            nxt,reward,term,trunc=p35.actual_step(env,history,test_action)
            observed=(nxt[3]*.07-old_delta)/.0035
            trace.update(json.dumps([before,test_action,nxt,reward,term,trunc],
                         separators=(",",":"),allow_nan=False).encode())
            observations.append(before)
            values.append(observed);actions+=1
        finally:
            env.close()
    if len(set(observations))!=1 or not values[0]<values[1]<values[2]:
        raise RuntimeError("P36_REAL_THREE_ACTION_SOURCE_INTERVENTIONS_NOT_IDENTIFIABLE")
    return {"world":world,"seed":seed,"state_index":index,
            "public_history":list(observations[0]),"real_physical_acceleration":values,
            "real_probe_actions":actions,"original_source_trace_sha256":trace.hexdigest()}

def probe_scores(probes,models):
    result={}
    for seed in SEEDS:
        result[seed]={}
        for arm in ARMS[:3]:
            m=models[seed]["shuffled_action_aux_ppo" if arm==ARMS[1] else "factual_action_aux_ppo"]
            flipped=(arm==ARMS[2])
            rank=0;error=0.
            for probe in probes:
                pred=effects(m,probe["public_history"],flipped)
                actual=probe["real_physical_acceleration"]
                rank+=int(sorted(range(3),key=lambda a:(pred[a],a))==
                          sorted(range(3),key=lambda a:(actual[a],a)))
                error+=sum((pred[a]-actual[a])**2 for a in range(3))
            result[seed][arm]={
               "paired_real_source_states":len(probes),
               "exact_three_action_order_fraction":rank/len(probes),
               "actual_original_source_action_effect_MSE":error/(len(probes)*3)}
    return result

def heldout_scores(records):
    out={}
    for rec in records:
        seed=rec["model_seed"]
        out[seed]={}
        for world in WORLDS:
            out[seed][world]={}
            worlds=[x for x in rec["original_world_episodes"] if x["world"]==world]
            for arm in ARMS:
                out[seed][world][arm]={
                    "true_source_reward_mean":sum(
                        x["episodes"][arm]["true_real_source_reward"]
                        for x in worlds)/len(worlds),
                    "physical_goal_successes_of_8":sum(int(
                        x["episodes"][arm]["actual_original_goal_reached"])
                        for x in worlds)}
    return out

def precommitted_gates(means,probes):
    t=frozen()["thresholds"]
    out={}
    for seed in SEEDS:
        original=means[seed]["original"]
        a=original[ARMS[0]];sham=original[ARMS[1]]
        permuted=original[ARMS[2]];rule=original["two_frame_public_momentum_rule"]
        ranks=probes[seed];rf=ranks[ARMS[0]]["exact_three_action_order_fraction"]
        rs=ranks[ARMS[1]]["exact_three_action_order_fraction"]
        rp=ranks[ARMS[2]]["exact_three_action_order_fraction"]
        out[seed]={
          "original_true_goals":a["physical_goal_successes_of_8"]>=t["factual_original_success_min_of_8"],
          "both_true_force_shift_goals":all(
             means[seed][w][ARMS[0]]["physical_goal_successes_of_8"]>=t["factual_each_shift_success_min_of_8"]
             for w in WORLDS if w!="original"),
          "better_true_reward_than_sham_and_permuted":all(
             a["true_source_reward_mean"]>=x["true_source_reward_mean"]+
             t["factual_original_reward_margin_over_sham_and_permuted"]
             for x in (sham,permuted)),
          "near_strong_conventional_reward":
             a["true_source_reward_mean"]>=rule["true_source_reward_mean"]-
             t["factual_original_reward_within_conventional_rule_points"],
          "real_three_action_physical_rank":rf>=t["factual_intervention_exact_rank_min"],
          "real_action_credit_rank_above_both_controls":all(
             rf>=x+t["factual_rank_advantage_over_sham_and_permuted_min"]
             for x in (rs,rp))}
    return out

def run(gym,original_owner):
    started=time.perf_counter()
    models,source_sha=load_previous_owner(original_owner)
    before={seed:{arm:p20.model_sha(m) for arm,m in dd.items()}
            for seed,dd in models.items()}
    probes=[paired_probe(gym,w,seed,i) for w in WORLDS for seed in PROBE_SEEDS for i in range(8)]
    counterfactual=probe_scores(probes,models)
    record=[]
    for model_seed in SEEDS:
        episodes=[]
        for w in WORLDS:
            for seed in EVAL_SEEDS:
                data={arm:episode(gym,seed,model_seed,w,arm,models[model_seed]) for arm in ARMS}
                if len({tuple(d["initial_public_source_observation"]) for d in data.values()})!=1:
                    raise RuntimeError("P36_EVALUATION_SOURCE_RESET_NOT_PAIRED")
                episodes.append({"world":w,"seed":seed,"episodes":data})
        record.append({"model_seed":model_seed,"original_world_episodes":episodes})
    if any(p20.model_sha(m)!=before[seed][arm] for seed,dd in models.items()
           for arm,m in dd.items()):
        raise RuntimeError("P36_READONLY_SOURCE_EXPERIMENT_MUTATED_P35_CHECKPOINT")
    means=heldout_scores(record)
    gates=precommitted_gates(means,counterfactual)
    passing=sum(int(all(x.values())) for x in gates.values())
    return {"schema":SCHEMA,"engineering":"PASS_READ_ONLY_ORIGINAL_P35_MODELS_DEV",
      "protected_scientific_first_use":"NOT_RUN",
      "actual_new_training_source_actions":0,"actual_optimizer_updates":0,
      "original_P35_archive_sha256":ARCHIVE_SHA,
      "original_P35_model_checkpoint_custody":source_sha,
      "actual_new_original_heldout_source_episodes":432,
      "actual_new_source_eval_actions":sum(d["real_source_actions"]
             for x in record for y in x["original_world_episodes"] for d in y["episodes"].values()),
      "actually_executed_real_physical_probe_states":len(probes),
      "real_source_physical_probe_actions":sum(x["real_probe_actions"] for x in probes),
      "prospective_factual_utility_2_of_3":passing>=frozen()["thresholds"]["model_seeds_required"],
      "prospective_model_seeds_passing":passing,
      "real_source_action_rank_accuracy_and_MSE":counterfactual,
      "real_source_heldout_goal_reward_summary":means,
      "prospective_rules":gates,
      "paired_source_physical_probe_records":probes,
      "original_heldout_source_episodes":record,
      "real_owner_source_wall_seconds":time.perf_counter()-started}

def cold_replay(gym,owner,receipt):
    original=json.loads(receipt.read_text(encoding="utf-8"))
    if (original.get("schema")!=SCHEMA
        or original.get("engineering")!="PASS_READ_ONLY_ORIGINAL_P35_MODELS_DEV"
        or original.get("actual_optimizer_updates")!=0
        or original.get("actual_new_original_heldout_source_episodes")!=432
        or len(original.get("original_heldout_source_episodes",[]))!=3):
        raise RuntimeError("P36_ORIGINAL_COLD_SOURCE_RECEIPT_NOT_QUALIFIED")
    models,_=load_previous_owner(owner)
    count=0;actions=0
    for rec in original["original_heldout_source_episodes"]:
        model_seed=rec["model_seed"]
        for world in rec["original_world_episodes"]:
            for arm in ARMS:
                was=world["episodes"][arm]
                now=episode(gym,world["seed"],model_seed,world["world"],arm,models[model_seed])
                if (was["source_actual_action_choices"]!=now["source_actual_action_choices"]
                    or was["true_real_source_reward"]!=now["true_real_source_reward"]
                    or was["actual_original_goal_reached"]!=now["actual_original_goal_reached"]
                    or was["real_source_action_consequence_sha256"]!=now["real_source_action_consequence_sha256"]):
                    raise RuntimeError("P36_FRESH_PROCESS_ORIGINAL_ACTION_CONSEQUENCE_SHA_DIFFERS")
                actions+=now["real_source_actions"];count+=1
    if count!=432:raise RuntimeError("P36_COLD_432_ORIGINAL_SOURCE_EPISODES_INCOMPLETE")
    return {"schema":"ttt-hybrid.native-p36.fresh-process-physical-actions.v1",
      "status":"PASS_EXACT_SOURCE_ACTION_REWARD_GOAL_SHA",
      "new_python_process":True,"episodes":count,"real_actions":actions,
      "actual_model_optimizer_updates":0,"original_paired_probe_not_reused":True,
      "protected_scientific_first_use":"NOT_RUN"}

def main():
    ap=argparse.ArgumentParser()
    for x in ("s29-root","source-root","cloudpickle-root","original-owner-root","receipt"):
        ap.add_argument("--"+x,type=Path,required=True)
    ap.add_argument("--mode",choices=("eval","cold"),required=True)
    args=ap.parse_args()
    guard()
    if not args.receipt.is_absolute() or not args.original_owner_root.is_absolute():
        raise RuntimeError("P36_OWNER_AND_RESULT_PATH_ABSOLUTE_REQUIRED")
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    gym,source=p20.admit_gymnasium(args.s29_root,args.source_root,args.cloudpickle_root)
    if args.mode=="eval":
        if args.receipt.exists():raise RuntimeError("P36_CONSUMED_SOURCE_EVAL_NO_REPLAY")
        report=run(gym,args.original_owner_root)
        report["original_source_custody"]=source
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        print("P36_REAL_ACTION_INTERVENTION_RESULT="+json.dumps({
           "engineering":report["engineering"],"science":report["prospective_factual_utility_2_of_3"],
           "passing":report["prospective_model_seeds_passing"],
           "heldout_episodes":report["actual_new_original_heldout_source_episodes"],
           "source_eval_actions":report["actual_new_source_eval_actions"],
           "physical_probe_states":report["actually_executed_real_physical_probe_states"],
           "physical_probe_actions":report["real_source_physical_probe_actions"],
           "rank":report["real_source_action_rank_accuracy_and_MSE"],
           "goals":report["real_source_heldout_goal_reward_summary"],
           "gates":report["prospective_rules"],
           "wall_seconds":report["real_owner_source_wall_seconds"]},sort_keys=True))
        return 0
    audit=cold_replay(gym,args.original_owner_root,args.receipt)
    out=args.receipt.parent/"native_p36_cold_intervention_audit.json"
    if out.exists():raise RuntimeError("P36_COLD_AUDIT_ALREADY_EXECUTED")
    out.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("P36_INDEPENDENT_COLD_SOURCE_REPLAY="+json.dumps(audit,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
