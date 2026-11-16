"""R105 tokenless trainable role-aligned causal memory: independent UNIT_ONLY.

No scored R101/R102/R103 samples or fitted weights. No R104 unit episodes
are reused; R104's 256-hypothesis public inference implementation serves
only as a strong independently defined comparator. All factual training
rewards arise from actually chosen R85 actions on R105 fresh synthetic worlds.
Development-unit results are NEVER independent scientific acceptance.
"""
from __future__ import annotations

import json
import math
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
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r102_contradiction_evidence_diagnostic as r102
from experiments import r103_relational_binding_diagnostic as r103
from experiments import r104_operator_identifiability_diagnostic as r104

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r105_native_role_binding_freeze.json"
FREEZE_BLOB="914118cb2ca1cb173a0b230bd5a50d4372cb5eb2"
SCHEMA="ttt-hybrid.r105.native-role-binding.unit-only-development-freeze.v1"
SEEDS=(1005011,1005023,1005037)
TRAIN=(0xca,0x96,0x80,0xfe,0x7f,0xe8,0x16,0x68,0x81,0x87,0xe1)
UNSEEN=(0x0f,0x33,0x3c,0x59,0x65,0x95,0x9a,0xa6)
EXCLUDED_R104=(0x1b,0x66,0x2d,0x55,0x8e,0xa3,0xb4,0xc9)
PANELS={"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
SUPPORT=16
ARMS=("NATIVE_ROLE_ALIGNED","NATIVE_REWARD_SHUFFLED","NATIVE_NO_FEEDBACK",
      "NATIVE_MISSING_ROLES","SCRATCH_R103_NATIVE","CONVENTIONAL_GRU64",
      "CONVENTIONAL_MLP64","BAYES256_PUBLIC_DIAGNOSTIC")
(NATIVE,SHUFFLED,ZERO,NO_ROLE,R103,GRU,MLP,BAYES)=ARMS
TRAINABLE=ARMS[:-1]
RECEIPT=ROOT/"local_runs/r105_unit_only_native_role_binding.json"


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R105_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    data=f["data"];train=f["training"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVELY_FROZEN_BEFORE_IMPLEMENTATION_OR_ANY_UNIT_ONLY_MODEL_FIT"
        or tuple(data["seeds"])!=SEEDS
        or tuple(int(x,16) for x in data["train_truth_masks_hex"])!=TRAIN
        or tuple(int(x,16) for x in data["unseen_truth_masks_hex"])!=UNSEEN
        or tuple(int(x,16) for x in data["exclude_prior_R104_unit_only_hidden_masks_hex"])!=EXCLUDED_R104
        or data["panel_salts"]!={"UNIT_FIT":26501101,"UNIT_VALID":26501107,
                                "UNIT_UNSEEN":26501113,"UNIT_BATCH":26501119,
                                "UNIT_SHUFFLE":26501123}
        or (data["train_episodes_per_seed"],data["valid_episodes_per_seed"],
            data["heldout_episodes_per_seed"],data["episode_support_count"])
           !=(176,88,128,16)
        or train["steps_per_trainable_arm"]!=512
        or train["batch_size"]!=16 or train["trainable_arms"]!=7
        or train["total_arms"]!=8
        or train["one_seed_max_wall_seconds"]!=240
        or f["authority"]["protected_original_scientific_first_use"] is not False
        or not f["authority"]["unit_only_CPU_software_engineering"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="unit_only_CPU_software_engineering")):
        raise PermissionError("R105_SOURCE_OR_AUTHORITY_FREEZE_DRIFT")
    return f


def _budget(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError("R105_FROZEN_UNIT_ONLY_240S_SEED_CAP_EXCEEDED")


def static_source_audit():
    f=frozen()
    if set(TRAIN)&set(UNSEEN) or set(EXCLUDED_R104)&set(UNSEEN):
        raise RuntimeError("R105_HELDOUT_OPERATOR_PREVIOUSLY_USED_OR_TRAINED")
    if len(set(TRAIN))!=11 or len(set(UNSEEN))!=8 or len(set(f["data"]["panel_salts"].values()))!=5:
        raise RuntimeError("R105_UNFROZEN_FAMILY_SALT_UNIQUENESS")
    pop=lambda x:int(x).bit_count()
    near={f"{m:02x}":min(pop(m^t) for t in TRAIN) for m in UNSEEN}
    pairs={f"{m:02x}":min(pop(m^x) for x in UNSEEN if x!=m) for m in UNSEEN}
    if min(near.values())<f["data"]["heldout_min_hamming_from_any_train_rule"] or min(pairs.values())<f["data"]["heldout_min_pair_hamming"]:
        raise RuntimeError("R105_HELDOUT_TRUTH_NOVELTY_FREEZE_FAILED")
    return {"stage":"R105_PROSPECTIVE_UNIT_ONLY_ENGINEERING",
            "train_rule_count":len(TRAIN),"unseen_rule_count":len(UNSEEN),
            "fresh_seeds":SEEDS,"nearest_known_Hamming":near,
            "nearest_unseen_Hamming":pairs,"original_scored_opened":False,
            "scientific_pass":False}


def _truth(factors,roles,mask):
    if mask not in TRAIN+UNSEEN:
        raise PermissionError("R105_PRIVATE_TRUTH_MASK_NOT_REGISTERED")
    if len(roles)!=3 or len(set(roles))!=3:
        raise ValueError("R105_PRIVATE_ROLES_INVALID")
    pattern=0
    for i in roles:
        if int(i) not in range(4):raise ValueError("R105_PRIVATE_FACTOR_ROLE_INVALID")
        pattern=2*pattern+int(factors[int(i)]>0)
    return 1 if (mask>>pattern)&1 else -1


def _unit_world(panel,seed,index,event,mask):
    """NEW R105 physical RNG and salt; never call any protected _world."""
    f=frozen()
    if (panel not in PANELS or seed not in SEEDS or not isinstance(index,int)
        or not 0<=index<PANELS.get(panel,0)
        or not isinstance(event,int) or not 0<=event<=SUPPORT):
        raise PermissionError("R105_ORIGINAL_OR_UNREGISTERED_WORLD_DENIED")
    fam=TRAIN if panel!="UNIT_UNSEEN" else UNSEEN
    if mask!=fam[index%len(fam)]:
        raise PermissionError("R105_PRIVATE_OPERATOR_ASSIGNMENT_DRIFT")
    rng=np.random.default_rng(np.random.SeedSequence(
        [f["data"]["panel_salts"][panel],seed,index,event]))
    parity=(index//len(fam))%2
    physical_index=(2*(index%(r85.N//2))+parity)%r85.N
    w=r85._make_synthetic_world(physical_index,rng)
    roles=tuple(int(x) for x in rng.permutation(4)[:3])
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    w["factors_a"]=factors
    w["r105_private_roles"]=roles
    w["law_a"]=_truth(factors,roles,mask)
    return w


def _view(w,action):
    prefix,reward=r86._view(w,action)
    x=r88._public_slots(prefix).copy()
    labels=np.zeros(4,np.float32)
    for order,i in enumerate(w["r105_private_roles"],1):
        labels[i]=order/4.
    x[:4,8]=labels
    if x.shape!=(5,13) or not np.isfinite(x).all():
        raise RuntimeError("R105_PUBLIC_FACTUAL_INPUT_INVALID")
    return x,float(reward)


def episode(panel,seed,index,*,deadline=None):
    f=frozen()
    if panel not in PANELS or seed not in SEEDS or not isinstance(index,int) or not 0<=index<PANELS.get(panel,0):
        raise PermissionError("R105_PROTECTED_OR_UNFROZEN_EPISODE_DENIED")
    fam=TRAIN if panel!="UNIT_UNSEEN" else UNSEEN
    mask=fam[index%len(fam)]
    slots=[];actual=[];actions=[]
    for event in range(SUPPORT):
        _budget(deadline)
        choice=r85.CALIBRATE if event%2==0 else r85.EXPLOIT_POS
        x,reward=_view(_unit_world(panel,seed,index,event,mask),choice)
        slots.append(x);actual.append(reward);actions.append(choice)
    _budget(deadline)
    query_world=_unit_world(panel,seed,index,SUPPORT,mask)
    q0,r0=_view(query_world,r85.CALIBRATE)
    q2,r2=_view(query_world,r85.EXPLOIT_POS)
    if not np.array_equal(q0[:4],q2[:4]) or not np.array_equal(q0[-1,4:],q2[-1,4:]):
        raise RuntimeError("R105_UNCHOSEN_CAUSAL_QUERY_LEAK")
    optimal=r85.CALIBRATE if r0>r2 else r85.EXPLOIT_POS
    if optimal!=(r85.CALIBRATE if query_world["law_a"]>0 else r85.EXPLOIT_POS):
        raise RuntimeError("R105_PHYSICAL_REWARD_LAW_ACTION_INCONSISTENT")
    supports=np.stack(slots).astype(np.float32)
    returns=np.asarray(actual,np.float32)
    actions=np.asarray(actions,np.int8)
    shuffled=r104._shuffle_action_strata(returns,actions)
    pub={"support_slots":supports,"support_rewards":returns,
         "support_actions":actions,"support_rewards_shuffled":shuffled}
    if panel!="UNIT_UNSEEN":
        chosen=(r85.CALIBRATE if (index//len(fam)+SEEDS.index(seed))%2==0
                else r85.EXPLOIT_POS)
        pub.update({"query":q0 if chosen==r85.CALIBRATE else q2,
                    "target":r0 if chosen==r85.CALIBRATE else r2})
    else:
        pub.update({"query0":q0,"query2":q2})
    return {"public":pub,"evaluator_only":{
        "mask":mask,"private_law":int(query_world["law_a"]),
        "reward0":r0,"reward2":r2,"optimal_action":optimal,
        "pivot":(index//len(fam))%2}}


def source(panel,seed,*,deadline=None):
    if panel not in PANELS or seed not in SEEDS:
        raise PermissionError("R105_ORIGINAL_PROTECTED_SOURCE_DENIED")
    records=[]
    for i in range(PANELS[panel]):
        _budget(deadline)
        records.append(episode(panel,seed,i,deadline=deadline))
    keys=("support_slots","support_rewards","support_actions",
          "support_rewards_shuffled")
    if panel=="UNIT_UNSEEN":
        keys+=("query0","query2")
    else:keys+=("query","target")
    out={k:np.asarray([x["public"][k] for x in records])
         for k in keys}
    if panel=="UNIT_UNSEEN":
        for k in ("mask","private_law","reward0","reward2","optimal_action","pivot"):
            out[k]=np.asarray([x["evaluator_only"][k] for x in records])
    elif set(out)&{"mask","private_law","reward0","reward2","optimal_action","pivot"}:
        raise RuntimeError("R105_FIT_VALID_PRIVILEGED_FIELD_LEAK")
    return out


class NativeRoleKeyValue(nn.Module):
    """Trainable tokenless 3-role binding and support-to-query causal memory.

    A PUBLIC role tag carries which observed operand corresponds to role 1/2/3;
    no rule ID, Boolean truth table, oracle action or future outcome is encoded.
    """
    def __init__(self):
        super().__init__()
        self.log_sharpness=nn.Parameter(torch.tensor(4.5))
        self.key=nn.Sequential(nn.Linear(3,32),nn.GELU(),
                               nn.Linear(32,24))
        self.value=nn.Sequential(nn.Linear(25,32),nn.GELU(),
                                 nn.Linear(32,1))
        self.log_similarity=nn.Parameter(torch.tensor(0.4))
        self.readout=nn.Sequential(nn.Linear(24+3+1+2,48),nn.GELU(),
                                   nn.Linear(48,32),nn.GELU(),nn.Linear(32,1))
    def bind(self,slots):
        if slots.ndim!=4 or slots.shape[-2:]!=(5,13):
            raise RuntimeError("R105_PUBLIC_ROLE_SLOT_FORMAT_INVALID")
        tags=slots[:,:,:4,8]
        factors=slots[:,:,:4,10]
        target=slots.new_tensor([.25,.5,.75])
        logits=-(tags.unsqueeze(-1)-target).square()*self.log_sharpness.exp().clamp(4,128)
        weights=torch.softmax(logits,dim=-2)
        return torch.sum(weights*factors.unsqueeze(-1),dim=-2)
    def forward(self,support_slots,rewards,query):
        if (support_slots.ndim!=4 or support_slots.shape[1:]!=(SUPPORT,5,13)
            or rewards.shape!=support_slots.shape[:2]
            or query.shape!=(len(support_slots),5,13)):
            raise RuntimeError("R105_NATIVE_FACTUAL_SOURCE_SHAPE")
        support=self.bind(support_slots)
        q=self.bind(query.unsqueeze(1)).squeeze(1)
        keys=F.normalize(self.key(support),dim=-1,eps=1e-5)
        qkey=F.normalize(self.key(q),dim=-1,eps=1e-5)
        similarities=(keys*qkey.unsqueeze(1)).sum(-1)
        att=torch.softmax(similarities*self.log_similarity.exp().clamp(.25,16),dim=1)
        # Reward is the REAL actually observed consequence, not the hidden law.
        values=torch.tanh(self.value(torch.cat(
            (keys,torch.tanh(rewards.unsqueeze(-1)/5)),dim=-1))).squeeze(-1)
        memory=torch.sum(values*att,dim=1,keepdim=True)
        action_code=query[:,-1,[0,2]]
        return self.readout(torch.cat(
            (qkey,q,memory,action_code),dim=-1))[:,0]


class RoleHistoryValue(nn.Module):
    """Strong fully trained conventional history comparator, same public role cues.

    Input binding is independently learned and public, exactly like native.
    No private mask or operator label. Conventional GRU or flat MLP gets
    the FULL 16 factual support cues and observed rewards.
    """
    def __init__(self,mode):
        super().__init__()
        if mode not in ("gru","mlp"):
            raise ValueError("R105_CONVENTIONAL_MODE_INVALID")
        self.mode=mode
        self.log_sharpness=nn.Parameter(torch.tensor(4.5))
        if mode=="gru":
            self.history=nn.GRU(4,64,batch_first=True)
            self.head=nn.Sequential(nn.Linear(64+3+2,64),nn.GELU(),
                                    nn.Linear(64,1))
        else:
            self.head=nn.Sequential(nn.Linear(SUPPORT*4+3+2,64),
                                    nn.GELU(),nn.Linear(64,64),nn.GELU(),
                                    nn.Linear(64,1))
    def bind(self,slots):
        tags=slots[:,:,:4,8]
        obs=slots[:,:,:4,10]
        target=slots.new_tensor([.25,.5,.75])
        w=torch.softmax(-(tags.unsqueeze(-1)-target).square()
                        *self.log_sharpness.exp().clamp(4,128),dim=-2)
        return torch.sum(w*obs.unsqueeze(-1),dim=-2)
    def forward(self,slots,rewards,query):
        if (slots.ndim!=4 or slots.shape[1:]!=(SUPPORT,5,13)
            or rewards.shape!=slots.shape[:2]
            or query.shape!=(len(slots),5,13)):
            raise RuntimeError("R105_CONVENTIONAL_PUBLIC_FACTUAL_INPUT_INVALID")
        facts=self.bind(slots)
        x=torch.cat((facts,torch.tanh(rewards.unsqueeze(-1)/5)),dim=-1)
        q=self.bind(query.unsqueeze(1)).squeeze(1)
        action=query[:,-1,[0,2]]
        if self.mode=="gru":
            h=self.history(x)[1][-1]
        else:
            h=x.flatten(start_dim=1)
        return self.head(torch.cat((h,q,action),dim=-1))[:,0]


def make_models(seed):
    if seed not in SEEDS:raise PermissionError("R105_UNREGISTERED_NEW_MODEL_SEED")
    models={}
    for arm in (NATIVE,SHUFFLED,ZERO,NO_ROLE):
        torch.manual_seed(seed+105)
        models[arm]=NativeRoleKeyValue()
    torch.manual_seed(seed+105)
    models[R103]=r103.NativeRelationalValue(r95._native(seed),"gated")
    torch.manual_seed(seed+505)
    models[GRU]=RoleHistoryValue("gru")
    torch.manual_seed(seed+506)
    models[MLP]=RoleHistoryValue("mlp")
    if set(models)!=set(TRAINABLE):
        raise RuntimeError("R105_MODEL_ARM_COUNT_DRIFT")
    keys=[dict(models[n].named_parameters()) for n in (NATIVE,SHUFFLED,ZERO,NO_ROLE)]
    if any(any(not torch.equal(keys[0][k],other[k]) for k in keys[0])
           for other in keys[1:]):
        raise RuntimeError("R105_SAME_NATIVE_ROLE_INIT_VIOLATION")
    return models


def _model_input(data,indices,arm,action=None):
    if arm not in TRAINABLE:
        raise ValueError("R105_MODEL_ARM_UNREGISTERED")
    x=torch.from_numpy(data["support_slots"][indices].astype(np.float32))
    ykey=("support_rewards_shuffled" if arm==SHUFFLED else "support_rewards")
    reward=torch.from_numpy(data[ykey][indices].astype(np.float32))
    if arm==ZERO:reward=torch.zeros_like(reward)
    qkey=("query" if action is None else ("query0" if action==r85.CALIBRATE else "query2"))
    q=torch.from_numpy(data[qkey][indices].astype(np.float32))
    if arm==NO_ROLE:
        x=x.clone();q=q.clone()
        x[:,:,:4,8]=0;q[:,:4,8]=0
    return x,reward,q


def fit_arm(model,arm,fit,valid,minibatches,deadline):
    f=frozen();train=f["training"]
    if (arm not in TRAINABLE
        or set(fit)&{"mask","private_law","reward0","reward2","query0","query2","optimal_action"}
        or minibatches.shape!=(512,16)
        or len(fit["target"])!=PANELS["UNIT_FIT"]):
        raise PermissionError("R105_UNIT_ONLY_FIT_INVALID_OR_PRIVILEGED")
    optimizer=torch.optim.AdamW(model.parameters(),lr=train["learning_rate"],
                                 weight_decay=train["weight_decay"])
    model.train()
    started=time.perf_counter()
    for indices in minibatches:
        _budget(deadline)
        x,r,q=_model_input(fit,indices,arm)
        target=torch.from_numpy(fit["target"][indices].astype(np.float32))
        predicted=model(x,r,q)
        loss=F.mse_loss(predicted,target)
        if predicted.shape!=target.shape or not bool(torch.isfinite(loss)):
            raise RuntimeError("R105_NATIVE_TRAINED_VALUE_INVALID")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),train["gradient_clip"])
        optimizer.step()
    _budget(deadline)
    return {"optimizer_updates":len(minibatches),
            "whole_arm_fit_seconds":time.perf_counter()-started,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)}


def predict(model,arm,data,action,deadline):
    model.eval()
    out=[]
    with torch.no_grad():
        for start in range(0,len(data["query0"]),32):
            _budget(deadline)
            inds=np.arange(start,min(start+32,len(data["query0"])))
            x,r,q=_model_input(data,inds,arm,action)
            v=model(x,r,q)
            if v.shape!=(len(inds),) or not bool(torch.isfinite(v).all()):
                raise RuntimeError("R105_NONFINITE_QUERY_VALUE")
            out.append(v.cpu().numpy())
    return np.concatenate(out)


def score(models,data,deadline):
    _budget(deadline)
    n=len(data["reward0"])
    outcomes={};choices={}
    for arm in TRAINABLE:
        _budget(deadline)
        v0=predict(models[arm],arm,data,r85.CALIBRATE,deadline)
        v2=predict(models[arm],arm,data,r85.EXPLOIT_POS,deadline)
        choices[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    bayes=[]
    for i in range(n):
        _budget(deadline)
        p={"support_slots":data["support_slots"][i],
           "support_actions":data["support_actions"][i],
           "actual_factual_rewards":data["support_rewards"][i],
           "query_slots":data["query0"][i]}
        bayes.append(r104.posterior_decision(p,mode="factual",supports=SUPPORT))
    choices[BAYES]=np.asarray(bayes,np.int8)
    truth=data["optimal_action"]
    if set(choices)!=set(ARMS):
        raise RuntimeError("R105_SCORE_ARM_MISSING")
    for name,act in choices.items():
        reward=np.where(act==r85.CALIBRATE,data["reward0"],data["reward2"])
        outcomes[name]={"unknown_action_accuracy":float(np.mean(act==truth)),
                        "realized_reward_mean":float(np.mean(reward)),
                        "oracle_reward_gap":float(np.mean(
                            np.maximum(data["reward0"],data["reward2"])-reward))}
    return outcomes


def checks(outcomes):
    f=frozen()
    acc=lambda name:outcomes[name]["unknown_action_accuracy"]
    differences={
        "native_unseen_action_accuracy_min":acc(NATIVE),
        "native_minus_best_conventional_accuracy_min":
            acc(NATIVE)-max(acc(GRU),acc(MLP)),
        "native_minus_r103_native_accuracy_min":acc(NATIVE)-acc(R103),
        "native_minus_shuffled_accuracy_min":acc(NATIVE)-acc(SHUFFLED),
        "native_minus_zero_feedback_accuracy_min":acc(NATIVE)-acc(ZERO),
        "native_minus_missing_roles_accuracy_min":acc(NATIVE)-acc(NO_ROLE),
        "Bayes256_unseen_accuracy_min":acc(BAYES)
    }
    thresholds=f["evaluation"]["checks"]
    if set(thresholds)!=set(differences):
        raise RuntimeError("R105_POSTRESULT_METRIC_DRIFT")
    passed={k:bool(differences[k]>=thresholds[k]) for k in thresholds}
    return {"measured":differences,"checks":passed,
            "all_unit_only_checks_pass":all(passed.values()),
            "scientific_pass":False}


def run_seed(seed):
    f=frozen()
    if seed not in SEEDS:raise PermissionError("R105_UNREGISTERED_SEED")
    started=time.perf_counter()
    deadline=started+f["training"]["one_seed_max_wall_seconds"]
    torch.set_num_threads(1)
    fit=source("UNIT_FIT",seed,deadline=deadline)
    valid=source("UNIT_VALID",seed,deadline=deadline)
    _budget(deadline)
    rng=np.random.default_rng(np.random.SeedSequence([
        f["data"]["panel_salts"]["UNIT_BATCH"],seed]))
    steps=rng.integers(0,len(fit["target"]),size=(512,16))
    models=make_models(seed)
    trained={}
    for arm in TRAINABLE:
        trained[arm]=fit_arm(models[arm],arm,fit,valid,steps,deadline)
    _budget(deadline)
    # All seven fully trained BEFORE generating any UNIT_ONLY heldout world.
    heldout=source("UNIT_UNSEEN",seed,deadline=deadline)
    metrics=score(models,heldout,deadline)
    result=checks(metrics)
    return {"seed":seed,"status":(
                f["evaluation"]["success"] if result["all_unit_only_checks_pass"]
                else f["evaluation"]["failure"]),
            "all_seven_fully_trained":all(x["optimizer_updates"]==512 for x in trained.values()),
            "training":trained,"unit_only_scores":metrics,"frozen_unit_only_checks":result,
            "source_scored_protected_access":False,
            "seed_whole_process_wall_seconds":time.perf_counter()-started,
            "scientific_pass":False}


def run_all():
    f=frozen();static=static_source_audit()
    if (ROOT/"experiments/r105_cpu_firstuse_authorization.json").exists():
        raise PermissionError("R105_ORIGINAL_FIRSTUSE_UNEXPECTEDLY_AUTHORIZED")
    current=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,
                            capture_output=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=current:
        raise PermissionError("R105_UNIT_ONLY_EXACT_SOURCE_DRIFT")
    seeds={}
    for seed in SEEDS:
        seeds[str(seed)]=run_seed(seed)
    success=all(x["frozen_unit_only_checks"]["all_unit_only_checks_pass"]
                for x in seeds.values())
    return {"schema":"ttt-hybrid.r105.unit-only-native-transfer-result.v1",
            "source_sha":current,"freeze_blob":FREEZE_BLOB,
            "source_type":"UNIT_ONLY_DEV",
            "source_static_audit":static,
            "all_three_seed_unit_only_checks":success,
            "status":(f["evaluation"]["success"] if success else f["evaluation"]["failure"]),
            "seeds":seeds,"scientific_pass":False,
            "prior_scored_cohorts_reused":False,"trained_weights_promoted":False,
            "deployment_status":"HOLD"}


def main():
    report=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R105_UNIT_ONLY_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                       encoding="utf-8")
    print("R105_UNIT_ONLY_NATIVE_TRANSFER "+json.dumps({
        "sha":report["source_sha"],"status":report["status"],
        "per_seed":{k:{"status":v["status"],
                        "accuracies":{arm:round(x["unknown_action_accuracy"],6)
                                      for arm,x in v["unit_only_scores"].items()},
                        "checks":v["frozen_unit_only_checks"]["checks"],
                        "seconds":round(v["seed_whole_process_wall_seconds"],2)}
                    for k,v in report["seeds"].items()},
        "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
