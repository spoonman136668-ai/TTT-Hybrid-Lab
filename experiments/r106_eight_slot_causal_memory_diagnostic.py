"""R106 independent UNIT_ONLY native causal eight-slot microfeedback experiment.

R106 uses freshly seeded R85 synthetic physics but never opens any prior
scored R101/R102/R103 cohort or loads old trained weights.  The in-episode
8-slot model is differentiable and operates on the PUBLIC role tags, noisy
factor probes, actual factual reward and chosen support/query actions only.
All original scientific first-use, GPU, merge, model promotion are denied.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r88_causal_event_access as r88
from experiments import r104_operator_identifiability_diagnostic as r104
from experiments import r105_native_role_binding_diagnostic as r105

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r106_eight_slot_causal_memory_freeze.json"
FREEZE_BLOB="42ae966e06ac11a8be862f46491423ea22cef631"
SCHEMA="ttt-hybrid.r106.eight-slot-causal-memory.unit-only-development-freeze.v1"
SEEDS=(1006011,1006023,1006037)
TRAIN=(0xca,0x96,0x80,0xfe,0x7f,0xe8,0x16,0x68,0x81,0x87,0xe1)
UNSEEN=(0x1d,0x27,0x3a,0x4b,0x71,0x93,0xa9,0xc5)
EXCLUDED_R104=(0x1b,0x66,0x2d,0x55,0x8e,0xa3,0xb4,0xc9)
EXCLUDED_R105=(0x0f,0x33,0x3c,0x59,0x65,0x95,0x9a,0xa6)
SALTS={"UNIT_FIT":26601101,"UNIT_VALID":26601107,
       "UNIT_UNSEEN":26601113,"UNIT_BATCH":26601119,
       "UNIT_SHUFFLE":26601123}
COUNTS={"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
SUPPORT=16
ARMS=("R106_NATIVE_CONTRADICTION_EIGHT_SLOT","R106_GATE_BLIND",
      "R106_OUTCOME_SHUFFLED","R106_NO_FEEDBACK","R106_MISSING_ROLES",
      "R105_SCRATCH_ROLE_SIMILARITY","ROLE_AWARE_GRU64",
      "ROLE_AWARE_MLP64","BAYES256_PUBLIC_FIXED")
NATIVE,BLIND,SHUFFLED,ZERO,NO_ROLE,R105,GRU,MLP,BAYES=ARMS
TRAINABLE=ARMS[:-1]
RECEIPT=ROOT/"local_runs/r106_unit_only_eight_slot_causal_memory.json"


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R106_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    src=f["source"];training=f["training"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PREIMPLEMENTATION_FROZEN_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
        or tuple(src["seeds"])!=SEEDS
        or tuple(int(x,16) for x in src["known_fit_operator_masks_hex"])!=TRAIN
        or tuple(int(x,16) for x in src["unseen_heldout_masks_hex"])!=UNSEEN
        or tuple(int(x,16) for x in src["disjoint_from_R104_evaluator_hex"])!=EXCLUDED_R104
        or tuple(int(x,16) for x in src["disjoint_from_R105_unseen_hex"])!=EXCLUDED_R105
        or src["panel_salts"]!=SALTS
        or (src["FIT_episodes_per_seed"],src["VALID_episodes_per_seed"],
            src["UNIT_UNSEEN_episodes_per_seed"],
            src["chosen_factual_supports_per_episode"])!=(176,88,128,16)
        or tuple(f["arms"])!=ARMS
        or (training["trainable_arm_count"],training["total_arm_count"],
            training["steps_per_trainable_arm"],
            training["batch_size"],training["seed_max_CPU_wall_seconds"])
            !=(8,9,512,16,240)
        or f["authority"]["protected_original_scientific_first_use"] is not False
        or not f["authority"]["unit_only_CPU_software_engineering"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="unit_only_CPU_software_engineering")):
        raise PermissionError("R106_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f


def _deadline(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError("R106_FROZEN_240S_UNIT_ONLY_SEED_BUDGET_EXCEEDED")


def static_audit():
    f=frozen()
    if (set(TRAIN)&set(UNSEEN) or set(EXCLUDED_R104)&set(UNSEEN)
        or set(EXCLUDED_R105)&set(UNSEEN)
        or not all(s not in r105.SEEDS and s not in r104.SEEDS for s in SEEDS)):
        raise RuntimeError("R106_SPENT_OR_PREVIOUSLY_USED_SOURCE_REJECTED")
    dist=lambda x,y:(x^y).bit_count()
    train_dist={f"{u:02x}":min(dist(u,t) for t in TRAIN) for u in UNSEEN}
    new_dist={f"{u:02x}":min(dist(u,v) for v in UNSEEN if v!=u) for u in UNSEEN}
    if (min(train_dist.values())<f["source"]["min_unseen_hamming_from_train"]
        or min(new_dist.values())<f["source"]["min_unseen_pair_hamming"]):
        raise RuntimeError("R106_UNSEEN_HAMMING_NOVELTY_DRIFT")
    return {"new_heldout_masks":[f"{x:02x}" for x in UNSEEN],
            "min_hamming_from_training":train_dist,
            "min_heldout_pairwise_hamming":new_dist,
            "prior_scored_cohorts_reused":False,"science_pass":False}


def _private_truth(factors,roles,mask):
    if mask not in TRAIN+UNSEEN or len(roles)!=3 or len(set(roles))!=3:
        raise PermissionError("R106_PRIVATE_TRUTH_MASK_OR_ROLE_INVALID")
    value=0
    for role in roles:
        if int(role) not in range(4):
            raise ValueError("R106_PRIVATE_PHYSICAL_ROLE_INVALID")
        value=2*value+int(factors[int(role)]>0)
    return 1 if (mask>>value)&1 else -1


def _unit_world(panel,seed,index,event,mask):
    if (panel not in COUNTS or seed not in SEEDS or not isinstance(index,int)
        or not 0<=index<COUNTS.get(panel,0)
        or not isinstance(event,int) or not 0<=event<=SUPPORT):
        raise PermissionError("R106_PROTECTED_OR_UNREGISTERED_PHYSICAL_SOURCE_DENIED")
    families=UNSEEN if panel=="UNIT_UNSEEN" else TRAIN
    if mask!=families[index%len(families)]:
        raise PermissionError("R106_PRIVATE_FAMILY_ASSIGNMENT_DRIFT")
    rng=np.random.default_rng(np.random.SeedSequence(
        [SALTS[panel],seed,index,event]))
    pivot=(index//len(families))%2
    physical_index=(2*(index%(r85.N//2))+pivot)%r85.N
    world=r85._make_synthetic_world(physical_index,rng)
    roles=tuple(int(x) for x in rng.permutation(4)[:3])
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r106_private_roles"]=roles
    world["law_a"]=_private_truth(factors,roles,mask)
    return world


def _view(world,action):
    prefix,reward=r86._view(world,action)
    public=r88._public_slots(prefix).copy()
    role_tags=np.zeros(4,np.float32)
    for order,physical in enumerate(world["r106_private_roles"],start=1):
        role_tags[physical]=order/4.
    public[:4,8]=role_tags
    if public.shape!=(5,13) or not np.isfinite(public).all():
        raise RuntimeError("R106_PUBLIC_PHYSICAL_PROBE_INVALID")
    return public,float(reward)


def episode(panel,seed,index,*,deadline=None):
    frozen()
    if panel not in COUNTS or seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNTS.get(panel,0):
        raise PermissionError("R106_NONFROZEN_ORIGINAL_EPISODE_DENIED")
    families=UNSEEN if panel=="UNIT_UNSEEN" else TRAIN
    mask=families[index%len(families)]
    slots=[];rewards=[];actions=[]
    for event in range(SUPPORT):
        _deadline(deadline)
        action=r85.CALIBRATE if event%2==0 else r85.EXPLOIT_POS
        x,reward=_view(_unit_world(panel,seed,index,event,mask),action)
        slots.append(x);rewards.append(reward);actions.append(action)
    _deadline(deadline)
    qworld=_unit_world(panel,seed,index,SUPPORT,mask)
    q0,r0=_view(qworld,r85.CALIBRATE)
    q2,r2=_view(qworld,r85.EXPLOIT_POS)
    if not np.array_equal(q0[:4],q2[:4]) or not np.array_equal(q0[-1,4:],q2[-1,4:]):
        raise RuntimeError("R106_PREACTION_COUNTERFACTUAL_PUBLIC_LEAK")
    true_opt=r85.CALIBRATE if qworld["law_a"]>0 else r85.EXPLOIT_POS
    optimal=r85.CALIBRATE if r0>r2 else r85.EXPLOIT_POS
    if true_opt!=optimal:
        raise RuntimeError("R106_PHYSICAL_CAUSAL_REWARD_LAW_MISMATCH")
    support=np.stack(slots).astype(np.float32)
    observed=np.asarray(rewards,np.float32)
    actions=np.asarray(actions,np.int8)
    public={"support_slots":support,"support_rewards":observed,
            "support_actions":actions,
            "support_rewards_shuffled":r104._shuffle_action_strata(observed,actions)}
    if panel=="UNIT_UNSEEN":
        public.update({"query0":q0,"query2":q2})
    else:
        # Predict ONLY one actually chosen query return per FIT/VALID case.
        chosen=(r85.CALIBRATE if (index//len(families)+SEEDS.index(seed))%2==0
                else r85.EXPLOIT_POS)
        public["query"]=q0 if chosen==r85.CALIBRATE else q2
        public["target"]=r0 if chosen==r85.CALIBRATE else r2
    return {"public":public,"evaluator_only":{
        "private_mask":mask,"private_law":int(qworld["law_a"]),
        "reward0":r0,"reward2":r2,"optimal_action":true_opt,
        "pivot":(index//len(families))%2}}


def source(panel,seed,*,deadline=None):
    if panel not in COUNTS or seed not in SEEDS:
        raise PermissionError("R106_ORIGINAL_PROTECTED_SOURCE_DENIED")
    results=[]
    for index in range(COUNTS[panel]):
        _deadline(deadline)
        results.append(episode(panel,seed,index,deadline=deadline))
    keys=("support_slots","support_rewards","support_actions","support_rewards_shuffled")
    if panel=="UNIT_UNSEEN":
        keys+=("query0","query2")
    else:
        keys+=("query","target")
    data={key:np.asarray([r["public"][key] for r in results]) for key in keys}
    if panel=="UNIT_UNSEEN":
        for key in ("private_mask","private_law","reward0","reward2","optimal_action","pivot"):
            data[key]=np.asarray([r["evaluator_only"][key] for r in results])
    elif any(x in data for x in ("private_mask","private_law","reward0","reward2","query0","query2","optimal_action")):
        raise RuntimeError("R106_PRIVATE_QUERY_TARGET_LEAKED_TO_TRAINING")
    return data


class NativeEightSlotValue(nn.Module):
    """Trainable public triple-role patterns, factual eight-slot local memory.

    Eight addresses are the eight sign combinations of three observed
    noisy factors; their learned return values are NOT fixed truth-table
    labels, Boolean rules or trained weights from prior experiments.
    """
    def __init__(self,mode="gated"):
        super().__init__()
        if mode not in ("gated","blind"):
            raise ValueError("R106_LOCAL_CONTRADICTION_MODE_INVALID")
        self.mode=mode
        self.log_sharpness=nn.Parameter(torch.tensor(4.5))
        self.log_pattern_temp=nn.Parameter(torch.tensor(.8))
        self.local_gate=nn.Sequential(nn.Linear(2,12),nn.Tanh(),nn.Linear(12,1))
        self.evidence=nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,4))
        self.readout=nn.Sequential(nn.Linear(8*5+8+3+2+4,64),
                                   nn.GELU(),nn.Linear(64,32),nn.GELU(),
                                   nn.Linear(32,1))
        combos=torch.tensor([[float((i>>2)&1),float((i>>1)&1),float(i&1)]
                             for i in range(8)],dtype=torch.float32)
        self.register_buffer("slot_signs",2*combos-1,persistent=False)
    def bind(self,slots):
        if slots.ndim!=4 or slots.shape[-2:]!=(5,13):
            raise RuntimeError("R106_PUBLIC_ROLE_SLOT_SHAPE_INVALID")
        tags=slots[:,:,:4,8];observed=slots[:,:,:4,10]
        target=slots.new_tensor([.25,.5,.75])
        logits=-(tags.unsqueeze(-1)-target).square()*self.log_sharpness.exp().clamp(4,128)
        weights=torch.softmax(logits,dim=-2)
        return (weights*observed.unsqueeze(-1)).sum(dim=-2)
    def addresses(self,role_factors):
        probs=torch.sigmoid(role_factors.unsqueeze(-2)
                            *self.slot_signs*self.log_pattern_temp.exp().clamp(.5,20))
        mass=probs.prod(dim=-1)
        return mass/mass.sum(dim=-1,keepdim=True).clamp_min(1e-8)
    def forward(self,support_slots,rewards,query):
        if (support_slots.ndim!=4 or support_slots.shape[1:]!=(SUPPORT,5,13)
            or rewards.shape!=support_slots.shape[:2]
            or query.shape!=(len(support_slots),5,13)):
            raise RuntimeError("R106_NATIVE_PUBLIC_FACTUAL_SHAPE_INVALID")
        role=self.bind(support_slots)
        qrole=self.bind(query.unsqueeze(1)).squeeze(1)
        addr=self.addresses(role);qaddr=self.addresses(qrole)
        action=support_slots[:,:,-1,[0,2]]
        bounded_reward=torch.tanh(rewards/5)
        density=addr.sum(dim=1)
        total=torch.einsum("bsk,bs->bk",addr,bounded_reward)
        peer=(total.unsqueeze(1)-addr*bounded_reward.unsqueeze(-1))/(
            density.unsqueeze(1)-addr).clamp_min(.02)
        pred_peer=(addr*peer).sum(dim=-1)
        contradiction=(bounded_reward-pred_peer).abs()
        if self.mode=="gated":
            gate=torch.sigmoid(self.local_gate(torch.stack(
                (contradiction,bounded_reward.abs()),dim=-1))).squeeze(-1)
        else:
            gate=addr.new_full(addr.shape[:2],.5)
        evidence=self.evidence(torch.cat(
            (bounded_reward.unsqueeze(-1),action),dim=-1))
        writes=addr*gate.unsqueeze(-1)
        count=writes.sum(dim=1).clamp_min(1e-4)
        memory=torch.einsum("bsk,bsd->bkd",writes,evidence)/count.unsqueeze(-1)
        retrieval=torch.sum(qaddr.unsqueeze(-1)*memory,dim=1)
        qp=query[:,-1,[0,2]]
        features=torch.cat((
            torch.cat((memory,torch.log1p(count).unsqueeze(-1)),dim=-1)
                .reshape(len(query),8*5),
            qaddr,qrole,qp,retrieval),dim=-1)
        return self.readout(features)[:,0]


def make_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R106_UNREGISTERED_FRESH_SEED")
    models={}
    for name in (NATIVE,BLIND,SHUFFLED,ZERO,NO_ROLE):
        torch.manual_seed(seed+106)
        models[name]=NativeEightSlotValue("blind" if name==BLIND else "gated")
    torch.manual_seed(seed+105)
    models[R105]=r105.NativeRoleKeyValue()
    torch.manual_seed(seed+505)
    models[GRU]=r105.RoleHistoryValue("gru")
    torch.manual_seed(seed+506)
    models[MLP]=r105.RoleHistoryValue("mlp")
    if set(models)!=set(TRAINABLE):
        raise RuntimeError("R106_FULL_MODEL_ARM_COVERAGE_INVALID")
    original=dict(models[NATIVE].named_parameters())
    for name in (BLIND,SHUFFLED,ZERO,NO_ROLE):
        compare=dict(models[name].named_parameters())
        if compare.keys()!=original.keys() or any(
            not torch.equal(compare[k],v) for k,v in original.items()):
            raise RuntimeError("R106_MATCHED_NATIVE_INITIALIZATION_VIOLATION")
    return models


def _inputs(data,indices,arm,action=None):
    if arm not in TRAINABLE:
        raise ValueError("R106_UNKNOWN_MODEL_ARM")
    x=torch.from_numpy(data["support_slots"][indices].astype(np.float32))
    key="support_rewards_shuffled" if arm==SHUFFLED else "support_rewards"
    reward=torch.from_numpy(data[key][indices].astype(np.float32))
    if arm==ZERO:
        reward=torch.zeros_like(reward)
    qkey=("query" if action is None
          else "query0" if action==r85.CALIBRATE else "query2")
    q=torch.from_numpy(data[qkey][indices].astype(np.float32))
    if arm==NO_ROLE:
        x=x.clone();q=q.clone()
        x[:,:,:4,8]=0;q[:,:4,8]=0
    return x,reward,q


def fit_arm(model,arm,fit,valid,minibatches,deadline):
    train=frozen()["training"]
    if (arm not in TRAINABLE
        or minibatches.shape!=(512,16)
        or len(fit.get("target",[]))!=COUNTS["UNIT_FIT"]
        or set(fit)&{"private_mask","private_law","reward0","reward2",
                     "query0","query2","optimal_action"}):
        raise PermissionError("R106_UNIT_FIT_UNFROZEN_OR_PRIVILEGED")
    optimizer=torch.optim.AdamW(model.parameters(),lr=train["learning_rate"],
                                 weight_decay=train["weight_decay"])
    model.train()
    start=time.perf_counter()
    for ix in minibatches:
        _deadline(deadline)
        x,r,q=_inputs(fit,ix,arm)
        target=torch.from_numpy(fit["target"][ix].astype(np.float32))
        pred=model(x,r,q)
        loss=F.mse_loss(pred,target)
        if pred.shape!=target.shape or not bool(torch.isfinite(loss)):
            raise RuntimeError("R106_TRAINING_NONFINITE")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),train["gradient_clip"])
        optimizer.step()
    _deadline(deadline)
    return {"updates":len(minibatches),
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                       if p.requires_grad),
            "fit_wall_seconds":time.perf_counter()-start}


def predict(model,arm,data,action,deadline):
    model.eval()
    output=[]
    with torch.no_grad():
        for start in range(0,len(data["query0"]),32):
            _deadline(deadline)
            ix=np.arange(start,min(start+32,len(data["query0"])))
            x,r,q=_inputs(data,ix,arm,action)
            value=model(x,r,q)
            if value.shape!=(len(ix),) or not bool(torch.isfinite(value).all()):
                raise RuntimeError("R106_HELDOUT_PREDICTION_NONFINITE")
            output.append(value.cpu().numpy())
    return np.concatenate(output)


def evaluate(models,data,deadline):
    truth=data["optimal_action"]
    out={}
    for arm in TRAINABLE:
        _deadline(deadline)
        v0=predict(models[arm],arm,data,r85.CALIBRATE,deadline)
        v2=predict(models[arm],arm,data,r85.EXPLOIT_POS,deadline)
        out[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    bayes=[]
    for index in range(len(truth)):
        _deadline(deadline)
        p={"support_slots":data["support_slots"][index],
           "support_actions":data["support_actions"][index],
           "actual_factual_rewards":data["support_rewards"][index],
           "query_slots":data["query0"][index]}
        bayes.append(r104.posterior_decision(p,supports=SUPPORT,mode="factual"))
    out[BAYES]=np.asarray(bayes,np.int8)
    if set(out)!=set(ARMS):
        raise RuntimeError("R106_EVALUATION_ARM_MISSING")
    return {arm:{"unknown_action_accuracy":float(np.mean(pred==truth)),
                 "correct_actions":int(np.sum(pred==truth)),
                 "realized_reward_mean":float(np.mean(np.where(
                      pred==r85.CALIBRATE,data["reward0"],data["reward2"]))),
                 "oracle_reward_gap":float(np.mean(
                     np.maximum(data["reward0"],data["reward2"])-np.where(
                         pred==r85.CALIBRATE,data["reward0"],data["reward2"])))}
            for arm,pred in out.items()}


def frozen_checks(evals):
    threshold=frozen()["evaluation"]["thresholds_all_three_seeds_required_for_DEV_only"]
    a=lambda arm:evals[arm]["unknown_action_accuracy"]
    values={
        "native_unseen_action_accuracy_min":a(NATIVE),
        "native_minus_strongest_fully_trained_conventional_accuracy_min":
            a(NATIVE)-max(a(GRU),a(MLP)),
        "native_minus_fresh_R105_native_accuracy_min":a(NATIVE)-a(R105),
        "native_minus_gate_blind_accuracy_min":a(NATIVE)-a(BLIND),
        "native_minus_shuffled_accuracy_min":a(NATIVE)-a(SHUFFLED),
        "native_minus_zero_feedback_accuracy_min":a(NATIVE)-a(ZERO),
        "native_minus_missing_roles_accuracy_min":a(NATIVE)-a(NO_ROLE),
        "Bayes256_unseen_accuracy_min":a(BAYES)}
    if set(values)!=set(threshold):
        raise RuntimeError("R106_POSTRESULT_THRESHOLD_MAPPING_DRIFT")
    checks={k:bool(v>=threshold[k]) for k,v in values.items()}
    return {"values":values,"checks":checks,
            "all_unit_only_checks_pass":all(checks.values()),
            "scientific_pass":False}


def run_seed(seed):
    f=frozen()
    if seed not in SEEDS:
        raise PermissionError("R106_UNREGISTERED_UNIT_ONLY_SEED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    deadline=start+f["training"]["seed_max_CPU_wall_seconds"]
    fit=source("UNIT_FIT",seed,deadline=deadline)
    valid=source("UNIT_VALID",seed,deadline=deadline)
    rng=np.random.default_rng(np.random.SeedSequence([SALTS["UNIT_BATCH"],seed]))
    minibatches=rng.integers(0,len(fit["target"]),size=(512,16))
    models=make_models(seed)
    trained={}
    for arm in TRAINABLE:
        _deadline(deadline)
        trained[arm]=fit_arm(models[arm],arm,fit,valid,minibatches,deadline)
    if any(trained[a]["updates"]!=512 for a in TRAINABLE):
        raise RuntimeError("R106_HELDOUT_BEFORE_FULL_EIGHT_ARM_TRAINING")
    # Fresh UNIT_ONLY heldout generation occurs only after all eight models fit.
    _deadline(deadline)
    heldout=source("UNIT_UNSEEN",seed,deadline=deadline)
    metrics=evaluate(models,heldout,deadline)
    checks=frozen_checks(metrics)
    _deadline(deadline)
    status=(f["evaluation"]["pass_status"] if checks["all_unit_only_checks_pass"]
            else f["evaluation"]["fail_status"])
    return {"seed":seed,"status":status,"all_eight_arms_fully_trained":True,
            "model_training":trained,"metrics":metrics,
            "prespecified_checks":checks,"seconds":time.perf_counter()-start,
            "protected_scored_source_accessed":False,"scientific_pass":False}


def run_all():
    static=static_audit()
    if (ROOT/"experiments/r106_cpu_firstuse_authorization.json").exists():
        raise PermissionError("R106_UNIT_ONLY_CANNOT_ADMIT_ORIGINAL_SCIENCE")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R106_UNIT_ONLY_EXACT_SHA_DRIFT")
    by_seed={str(seed):run_seed(seed) for seed in SEEDS}
    qualified=all(x["prespecified_checks"]["all_unit_only_checks_pass"]
                  for x in by_seed.values())
    return {"schema":"ttt-hybrid.r106.unit-only-eight-slot-development-result.v1",
            "exact_source_sha":git,"freeze_blob":FREEZE_BLOB,
            "status":(frozen()["evaluation"]["pass_status"] if qualified
                      else frozen()["evaluation"]["fail_status"]),
            "source_audit":static,"by_seed":by_seed,
            "all_three_unit_only_seed_checks_pass":qualified,
            "prior_scored_source_reused":False,
            "original_scored_firstuse":False,"scientific_pass":False,
            "model_weights_promoted":False,"deployment":"HOLD"}


def main():
    result=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R106_UNIT_ONLY_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                       encoding="utf-8")
    print("R106_UNIT_ONLY_EIGHT_SLOT "+json.dumps({
        "sha":result["exact_source_sha"],"status":result["status"],
        "per_seed":{k:{"accuracy":{arm:round(d["unknown_action_accuracy"],6)
                                     for arm,d in v["metrics"].items()},
                        "checks":v["prespecified_checks"]["checks"],
                        "seconds":round(v["seconds"],2)} for k,v in result["by_seed"].items()},
        "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
