"""R110 prospective UNIT_ONLY 2-context factual correction and retention study.

R110 never loads R101/R102/R103 original scored cases, prior model weights,
or R104–R108 heldout worlds. It trains thirteen fresh arms on physical
chosen-action factual returns only. Private truth is evaluator-only, withheld
until EVERY control finishes the frozen 512 training steps. This is not
protected original science, GPU training, deployment, or model promotion.
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r88_causal_event_access as r88
from experiments import r104_operator_identifiability_diagnostic as r104
from experiments import r110_public_custody as r110
from experiments import r110_public_contradiction_credit as credit

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1010011,1010023,1010037)
TRAIN = tuple(int(x,16) for x in ("ca","96","80","fe","7f","e8","16","68","81","87","e1"))
UNSEEN = tuple(int(x,16) for x in ("03","0c","30","5a","99","a5","aa","c6"))
SALTS = {"UNIT_FIT":27001101,"UNIT_VALID":27001107,
         "UNIT_UNSEEN":27001113,"UNIT_BATCH":27001119,
         "UNIT_SHUFFLE":27001123,"UNIT_COLD":27001129}
COUNTS = {"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
SUPPORT = 16
ARMS = ("R110_NATIVE_PUBLIC_CREDIT","R110_NO_AUX","R110_PERMUTED_CREDIT",
        "R110_ALWAYS_SHARED","R110_ALWAYS_PRIVATE","R110_FIXED_HALF",
        "R110_PHASE_ONLY","R110_NO_PUBLIC_CONTEXT","R110_SHUFFLED_FACTUAL",
        "R110_ZERO_FACTUAL","R110_MISSING_PUBLIC_ROLES",
        "ROLE_AWARE_PHASE_GRU","ROLE_AWARE_PHASE_MLP","BAYES256_PHASE_PUBLIC_FIXED")
NATIVE,NO_AUX,PLACEBO,SINGLE,TWO,STATIC,PHASE_ONLY,NO_CONTEXT,SHUFFLE,ZERO,NO_ROLES,GRU,MLP,BAYES=ARMS
TRAINABLE=ARMS[:-1]
RECEIPT=ROOT/"local_runs/r110_unit_only_public_contradiction_credit.json"


def frozen():
    info=r110.audit_freeze()
    f=json.loads(r110.FREEZE.read_text(encoding="utf-8"))
    src=f["source"];train=f["training"]
    if (info["scientific_pass"] is not False
        or tuple(src["seeds"])!=SEEDS
        or tuple(int(x,16) for x in src["known_fit_masks_hex"])!=TRAIN
        or tuple(int(x,16) for x in src["unseen_heldout_masks_hex"])!=UNSEEN
        or src["panel_salts"]!=SALTS
        or src["episodes_per_seed"]!=COUNTS
        or src["actual_chosen_support_consequences_per_episode"]!=SUPPORT
        or (train["updates_per_trainable_arm"],train["batch_size"],
            train["CPU_wall_budget_seconds_per_seed"])!=(512,16,300)
        or not src["no_unseen_world_generation_until_all_trainable_arms_finish"]
        or f["authority"]["protected_original_scientific_first_use"] is not False):
        raise PermissionError("R110_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f


def _deadline(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError("R110_FROZEN_300S_CPU_UNIT_ONLY_BUDGET_EXCEEDED")


def _law(factors,roles,mask):
    bits=0
    for role in roles:
        bits=(bits<<1)+int(factors[int(role)]>0)
    return 1 if (mask>>bits)&1 else -1


def _unit_world(panel,seed,index,event,context,mask):
    if (panel not in COUNTS or seed not in SEEDS
        or type(index) is not int or not 0<=index<COUNTS.get(panel,0)
        or type(event) is not int or not 0<=event<=SUPPORT
        or context not in (0,1)
        or mask not in (UNSEEN if panel=="UNIT_UNSEEN" else TRAIN)):
        raise PermissionError("R110_UNREGISTERED_OR_PROTECTED_SOURCE_DENIED")
    rng=np.random.default_rng(np.random.SeedSequence(
        [SALTS[panel],seed,index,event,context]))
    pivot=(index//(2*(8 if panel=="UNIT_UNSEEN" else 11)))%2
    physical_index=(2*(index%(r85.N//2))+pivot)%r85.N
    world=r85._make_synthetic_world(physical_index,rng)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    roles=tuple(int(x) for x in rng.permutation(4)[:3])
    world["factors_a"]=factors
    world["r110_private_roles"]=roles
    world["law_a"]=_law(factors,roles,mask)
    return world


def _view(world,action):
    prefix,reward=r86._view(world,action)
    slots=r88._public_slots(prefix).copy()
    for role,physical in enumerate(world["r110_private_roles"],start=1):
        slots[physical,8]=role/4.
    if slots.shape!=(5,13) or not np.isfinite(slots).all():
        raise RuntimeError("R110_PUBLIC_ROLE_OBSERVATION_INVALID")
    return slots,float(reward)


def _assignment(panel,index):
    families=UNSEEN if panel=="UNIT_UNSEEN" else TRAIN
    n=len(families)
    group=index//n
    old=families[index%n]
    switched=(group%2)==1
    new=(families[(index+3)%n] if switched else old)
    cutoff=(6,8,10)[index%3]
    return old,new,switched,cutoff


def _shuffle_by_public_stratum(rewards,actions,contexts):
    out=np.asarray(rewards,np.float32).copy()
    for phase in (0,1):
        for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
            indices=np.flatnonzero((contexts==phase)&(actions==action))
            if len(indices)<3:
                raise RuntimeError("R110_ACTION_CONTEXT_STRATUM_TOO_SMALL")
            out[indices]=rewards[np.roll(indices,1)]
    return out


def episode(panel,seed,index,*,deadline=None):
    frozen()
    if panel not in COUNTS or seed not in SEEDS or type(index) is not int or not 0<=index<COUNTS.get(panel,0):
        raise PermissionError("R110_INVALID_PROTECTED_ORIGINAL_PANEL")
    old,new,switched,cutoff=_assignment(panel,index)
    slots=[];rewards=[];actions=[];phases=[]
    for event in range(SUPPORT):
        _deadline(deadline)
        phase=0 if event<cutoff else 1
        mask=old if phase==0 else new
        action=r85.CALIBRATE if event%2==0 else r85.EXPLOIT_POS
        world=_unit_world(panel,seed,index,event,phase,mask)
        view,reward=_view(world,action)
        slots.append(view);rewards.append(reward);actions.append(action);phases.append(phase)
    slots=np.stack(slots).astype(np.float32)
    rewards=np.asarray(rewards,np.float32)
    actions=np.asarray(actions,np.int64)
    phases=np.asarray(phases,np.int64)
    q0=[];q2=[];rq0=[];rq2=[];truth=[]
    for phase,mask in ((0,old),(1,new)):
        world=_unit_world(panel,seed,index,SUPPORT,phase,mask)
        v0,r0=_view(world,r85.CALIBRATE)
        v2,r2=_view(world,r85.EXPLOIT_POS)
        if not np.array_equal(v0[:4],v2[:4]):
            raise RuntimeError("R110_UNCHOSEN_COUNTERFACTUAL_PUBLIC_CUE_LEAK")
        optimal=(r85.CALIBRATE if world["law_a"]>0 else r85.EXPLOIT_POS)
        if optimal!=(r85.CALIBRATE if r0>r2 else r85.EXPLOIT_POS):
            raise RuntimeError("R110_PHYSICAL_CAUSAL_REWARD_MISMATCH")
        q0.append(v0);q2.append(v2);rq0.append(r0);rq2.append(r2);truth.append(optimal)
    public={"support_slots":slots,"support_rewards":rewards,
            "support_actions":actions,"support_context":phases,
            "support_rewards_shuffled":_shuffle_by_public_stratum(rewards,actions,phases)}
    if panel=="UNIT_UNSEEN":
        public["query0"]=np.stack(q0).astype(np.float32)
        public["query2"]=np.stack(q2).astype(np.float32)
    else:
        chosen=[r85.CALIBRATE if (index+seed+phase)%2==0 else r85.EXPLOIT_POS
                for phase in (0,1)]
        public["query"]=np.stack([q0[p] if chosen[p]==r85.CALIBRATE else q2[p]
                                  for p in (0,1)]).astype(np.float32)
        public["target"]=np.asarray([rq0[p] if chosen[p]==r85.CALIBRATE else rq2[p]
                                     for p in (0,1)],np.float32)
    return {"public":public,"evaluator_only":{
        "old_private_mask":old,"new_private_mask":new,
        "private_switched":bool(switched),"switch_after":cutoff,
        "old_optimal":truth[0],"new_optimal":truth[1],
        "reward0":np.asarray(rq0,np.float32),
        "reward2":np.asarray(rq2,np.float32)}}


def source(panel,seed,*,deadline=None):
    if panel not in COUNTS or seed not in SEEDS:
        raise PermissionError("R110_NO_ORIGINAL_OR_NONUNIT_SOURCE")
    rows=[]
    for i in range(COUNTS[panel]):
        _deadline(deadline)
        rows.append(episode(panel,seed,i,deadline=deadline))
    names=("support_slots","support_rewards","support_actions",
           "support_context","support_rewards_shuffled")
    names+=(("query0","query2") if panel=="UNIT_UNSEEN" else ("query","target"))
    data={k:np.stack([x["public"][k] for x in rows]) for k in names}
    if panel=="UNIT_UNSEEN":
        # Evaluator-only source is generated exclusively after all 13 arms fit.
        for name in ("old_private_mask","new_private_mask","private_switched",
                     "switch_after","old_optimal","new_optimal","reward0","reward2"):
            data[name]=np.asarray([x["evaluator_only"][name] for x in rows])
    elif any(k in data for k in ("old_private_mask","new_private_mask","reward0",
                                 "reward2","query0","query2","old_optimal","new_optimal")):
        raise RuntimeError("R110_TRAIN_QUERY_PRIVILEGE_VIOLATION")
    return data


class NativeIdentifiableValue(nn.Module):
    """Native eight-slot, public factual conditional routing with no head bypass.

    Every action value is gate*private_value + (1-gate)*shared_value.
    Both experts use the SAME trainable readout, initialized identically across
    all native variants. No side channel can read both banks directly.
    """
    def __init__(self,mode="adaptive"):
        super().__init__()
        if mode not in ("adaptive","single","two","static","phase"):
            raise ValueError("R110_GATE_MODE_INVALID")
        self.mode=mode
        self.log_sharpness=nn.Parameter(torch.tensor(4.5))
        self.log_temperature=nn.Parameter(torch.tensor(.8))
        self.evidence=nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,4))
        self.router=nn.Sequential(nn.Linear(16,16),nn.GELU(),nn.Linear(16,1))
        self.expert=nn.Sequential(nn.Linear(18,48),nn.GELU(),
                                  nn.Linear(48,16),nn.GELU(),nn.Linear(16,1))
        combos=torch.tensor([[float((i>>2)&1),float((i>>1)&1),float(i&1)]
                             for i in range(8)],dtype=torch.float32)
        self.register_buffer("patterns",2*combos-1,persistent=False)
        self.last_route_gate=None
        self.route_for_credit_loss=None
        self.last_expert_delta=None

    def bind(self,slots):
        tags=slots[:,:,:4,8]
        obs=slots[:,:,:4,10]
        target=slots.new_tensor([.25,.5,.75])
        weights=torch.softmax(-(tags.unsqueeze(-1)-target).square()
                              *self.log_sharpness.exp().clamp(4,128),dim=-2)
        return (weights*obs.unsqueeze(-1)).sum(dim=-2)

    def address(self,role):
        probabilities=torch.sigmoid(role.unsqueeze(-2)*self.patterns
                                   *self.log_temperature.exp().clamp(.5,20))
        mass=probabilities.prod(dim=-1)
        return mass/mass.sum(dim=-1,keepdim=True).clamp_min(1e-8)

    def forward(self,slots,rewards,phases,query,query_context,valid,*,forced_gate=None):
        if (slots.ndim!=4 or slots.shape[1:]!=(16,5,13)
            or rewards.shape!=slots.shape[:2] or phases.shape!=rewards.shape
            or valid.shape!=rewards.shape or query.shape!=(len(slots),5,13)
            or query_context.shape!=(len(slots),)
            or not bool(torch.isfinite(rewards).all())):
            raise RuntimeError("R110_PUBLIC_FACTUAL_MODEL_SHAPE_INVALID")
        roles=self.bind(slots)
        qrole=self.bind(query.unsqueeze(1)).squeeze(1)
        address=self.address(roles)
        qaddr=self.address(qrole)
        action=slots[:,:,-1,[0,2]]
        observed=self.evidence(torch.cat((
            torch.tanh(rewards.unsqueeze(-1)/5),action,
            F.one_hot(phases.long(),2).to(slots.dtype)),dim=-1))
        bank=F.one_hot(phases.long(),2).to(slots.dtype)
        writes=(bank.unsqueeze(-1)*address.unsqueeze(-2)
                *valid.unsqueeze(-1).unsqueeze(-1))
        phase_count=writes.sum(dim=1).clamp_min(1e-4)
        phase_memory=torch.einsum("bspk,bsd->bpkd",writes,observed)/phase_count.unsqueeze(-1)
        selector=F.one_hot(query_context.long(),2).to(slots.dtype)
        private_bank=torch.einsum("bp,bpkd->bkd",selector,phase_memory)
        private_count=torch.einsum("bp,bpk->bk",selector,phase_count)
        private=torch.einsum("bkd,bk->bd",private_bank,qaddr)
        global_w=address*valid.unsqueeze(-1)
        global_count=global_w.sum(dim=1).clamp_min(1e-4)
        global_memory=torch.einsum("bsk,bsd->bkd",global_w,observed)/global_count.unsqueeze(-1)
        shared=torch.einsum("bkd,bk->bd",global_memory,qaddr)
        context=F.one_hot(query_context.long(),2).to(slots.dtype)
        local_mass=torch.log1p((private_count*qaddr).sum(dim=-1,keepdim=True))
        shared_mass=torch.log1p((global_count*qaddr).sum(dim=-1,keepdim=True))
        gate_input=torch.cat((private,shared,torch.abs(private-shared),
                              local_mass,shared_mass,context),dim=-1)
        if self.mode=="phase":
            # Same parameterized router, no observed consequence in gate inputs.
            gate_input=torch.cat((torch.zeros_like(gate_input[:,:12]),
                                  gate_input[:,12:]),dim=-1)
        gate=torch.sigmoid(self.router(gate_input))
        if self.mode=="static":
            gate=gate*0.+.5
        elif self.mode=="single":
            gate=gate*0.
        elif self.mode=="two":
            gate=gate*0.+1.
        if forced_gate is not None:
            if self.mode!="adaptive" or type(forced_gate) is not float or not 0<=forced_gate<=1:
                raise PermissionError("R110_FORCED_GATE_DIAGNOSTIC_ONLY")
            gate=gate*0.+forced_gate
        self.last_route_gate=gate.detach()
        self.route_for_credit_loss=gate[:,0]
        action_query=query[:,-1,[0,2]]
        common=torch.cat((qaddr,qrole,action_query),dim=-1)
        private_input=torch.cat((private,common,local_mass),dim=-1)
        shared_input=torch.cat((shared,common,shared_mass),dim=-1)
        private_value=self.expert(private_input)[:,0]
        shared_value=self.expert(shared_input)[:,0]
        self.last_expert_delta=(private_value-shared_value).detach()
        # ONE output pathway: the gate cannot be skipped by a subsequent head.
        return gate[:,0]*private_value+(1.-gate[:,0])*shared_value


class ConventionalPhaseValue(nn.Module):
    """Newly trained public phase-aware GRU/MLP, all 16 factual action cues."""
    def __init__(self,mode,width):
        super().__init__()
        self.mode=mode
        self.width=width
        self.log_sharpness=nn.Parameter(torch.tensor(4.5))
        if mode=="gru":
            self.history=nn.GRU(8,width,batch_first=True)
            input_width=width+7
        elif mode=="mlp":
            input_width=SUPPORT*8+7
        else:
            raise ValueError("R110_UNKNOWN_CONVENTIONAL_ARCHITECTURE")
        self.head=nn.Sequential(nn.Linear(input_width,width),nn.GELU(),
                                nn.Linear(width,1))

    def bind(self,slots):
        tags=slots[:,:,:4,8]
        obs=slots[:,:,:4,10]
        target=slots.new_tensor([.25,.5,.75])
        probs=torch.softmax(-(tags.unsqueeze(-1)-target).square()
                             *self.log_sharpness.exp().clamp(4,128),dim=-2)
        return (probs*obs.unsqueeze(-1)).sum(dim=-2)

    def forward(self,slots,rewards,phases,query,query_context,valid):
        role=self.bind(slots)
        x=torch.cat((role,torch.tanh(rewards.unsqueeze(-1)/5),
            slots[:,:,-1,[0,2]],F.one_hot(phases.long(),2).to(slots.dtype)),dim=-1)
        x=x*valid.unsqueeze(-1)
        if self.mode=="gru":
            seq,_=self.history(x)
            lengths=(valid.sum(dim=1).long()-1).clamp(0,SUPPORT-1)
            h=seq[torch.arange(len(slots)),lengths]
        else:
            h=x.flatten(start_dim=1)
        q=self.bind(query.unsqueeze(1)).squeeze(1)
        features=torch.cat((h,q,query[:,-1,[0,2]],
                           F.one_hot(query_context.long(),2).to(slots.dtype)),dim=-1)
        return self.head(features)[:,0]


def make_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R110_UNREGISTERED_TRAINING_SEED")
    models={}
    modes={NATIVE:"adaptive",NO_AUX:"adaptive",PLACEBO:"adaptive",
           SINGLE:"single",TWO:"two",STATIC:"static",
           PHASE_ONLY:"phase",NO_CONTEXT:"adaptive",SHUFFLE:"adaptive",
           ZERO:"adaptive",NO_ROLES:"adaptive"}
    for name,mode in modes.items():
        torch.manual_seed(seed+110)
        models[name]=NativeIdentifiableValue(mode)
    ref=dict(models[NATIVE].named_parameters())
    for name in modes:
        if any(not torch.equal(ref[k],v) for k,v in models[name].named_parameters()):
            raise RuntimeError("R110_SAME_INIT_NEGATIVE_CONTROL_DRIFT")
    size=sum(p.numel() for p in models[NATIVE].parameters())
    counts={}
    for name,mode,seed_extra in ((GRU,"gru",507),(MLP,"mlp",508)):
        candidates=[(abs(sum(p.numel() for p in ConventionalPhaseValue(
            mode,width).parameters())-size),width) for width in (32,64,96)]
        width=min(candidates)[1]
        torch.manual_seed(seed+seed_extra)
        models[name]=ConventionalPhaseValue(mode,width)
        counts[name]={"hidden_width":width,
                      "parameters":sum(p.numel() for p in models[name].parameters())}
    return models,counts


def _inputs(data,indices,arm,phase,action=None,before=False):
    if arm not in TRAINABLE or phase not in (0,1):
        raise ValueError("R110_ARM_OR_PUBLIC_CONTEXT_UNKNOWN")
    slots=torch.from_numpy(data["support_slots"][indices].astype(np.float32))
    reward_key="support_rewards_shuffled" if arm==SHUFFLE else "support_rewards"
    reward=torch.from_numpy(data[reward_key][indices].astype(np.float32))
    if arm==ZERO:
        reward=torch.zeros_like(reward)
    contexts=torch.from_numpy(data["support_context"][indices].astype(np.int64))
    if action is None:
        query=torch.from_numpy(data["query"][indices,phase].astype(np.float32))
    else:
        key="query0" if action==r85.CALIBRATE else "query2"
        query=torch.from_numpy(data[key][indices,phase].astype(np.float32))
    valid=torch.ones_like(reward)
    if before:
        valid=valid*(contexts==0)
    queryphase=torch.full((len(indices),),phase,dtype=torch.long)
    if arm==NO_CONTEXT:
        contexts=torch.zeros_like(contexts)
        queryphase=torch.zeros_like(queryphase)
    if arm==NO_ROLES:
        slots=slots.clone();query=query.clone()
        slots[:,:,:4,8]=0
        query[:,:4,8]=0
    return slots,reward,contexts,query,queryphase,valid


def train_arm(model,arm,fit,minibatches,deadline,fit_credit):
    f=frozen()["training"]
    if (minibatches.shape!=(512,16) or len(fit.get("target",[]))!=176
        or fit_credit.shape!=(176,) or not np.isfinite(fit_credit).all()
        or any(k in fit for k in ("old_private_mask","new_private_mask",
                                  "reward0","reward2","query0","query2",
                                  "old_optimal","new_optimal"))):
        raise PermissionError("R110_TRAINING_SOURCE_LEAK_OR_BUDGET_DRIFT")
    optimizer=torch.optim.AdamW(model.parameters(),lr=f["learning_rate"],
                                weight_decay=f["weight_decay"])
    start=time.perf_counter()
    model.train()
    aux_total=0.
    for index in minibatches:
        _deadline(deadline)
        losses=[]
        aux=None
        for context in (0,1):
            inp=_inputs(fit,index,arm,context,before=(context==0))
            estimate=model(*inp)
            target=torch.from_numpy(fit["target"][index,context].astype(np.float32))
            if estimate.shape!=target.shape:
                raise RuntimeError("R110_TARGET_PUBLIC_CHOSEN_RETURN_MISMATCH")
            losses.append(F.mse_loss(estimate,target))
            if context==1 and arm in (NATIVE,PLACEBO):
                observed_target=torch.from_numpy(fit_credit[index].astype(np.float32))
                if arm==PLACEBO:
                    # Same marginal target distribution, deliberately wrong world.
                    observed_target=torch.roll(observed_target,1,0)
                gate=model.route_for_credit_loss
                if (gate is None or gate.shape!=observed_target.shape
                    or not gate.requires_grad):
                    raise RuntimeError("R110_ROUTER_CREDIT_GRADIENT_NOT_IDENTIFIABLE")
                aux=F.binary_cross_entropy(gate,observed_target)
        loss=(losses[0]+losses[1])/2
        if aux is not None:
            loss=loss+credit_weight()*aux
            aux_total+=float(aux.detach().item())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R110_NONFINITE_TRAINING_OBJECTIVE")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),f["gradient_clip"])
        optimizer.step()
    _deadline(deadline)
    return {"updates":len(minibatches),
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "fit_seconds":time.perf_counter()-start,
            "auxiliary_credit_loss_mean":aux_total/len(minibatches),
            "auxiliary_credit_enabled":arm in (NATIVE,PLACEBO)}


def credit_weight():
    f=frozen()["credit"]
    if (f["aux_weight"]!=.25 or f["likelihood_match"]!=.85
        or f["likelihood_mismatch"]!=.15 or f["change_prior"]!=.5
        or f["target_clamp_low"]!=.05 or f["target_clamp_high"]!=.95):
        raise PermissionError("R110_FROZEN_PUBLIC_CREDIT_SOURCE_DRIFT")
    return float(f["aux_weight"])


def _predict_actions(model,arm,data,phase,deadline,*,before=False,forced_gate=None):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(data["support_slots"]),32):
            _deadline(deadline)
            ix=np.arange(start,min(start+32,len(data["support_slots"])))
            i0=_inputs(data,ix,arm,phase,r85.CALIBRATE,before)
            i2=_inputs(data,ix,arm,phase,r85.EXPLOIT_POS,before)
            if forced_gate is None:
                a=model(*i0);b=model(*i2)
            elif arm==NATIVE:
                a=model(*i0,forced_gate=forced_gate)
                b=model(*i2,forced_gate=forced_gate)
            else:
                raise PermissionError("R110_GATE_OVERRIDE_ONLY_ON_NATIVE")
            if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
                raise RuntimeError("R110_NONFINITE_UNSEEN_ACTION_VALUE")
            result.append(np.where(a.cpu().numpy()>b.cpu().numpy(),
                                   r85.CALIBRATE,r85.EXPLOIT_POS))
    return np.concatenate(result)


def _bayes_public(data,index,phase,before=False):
    # All 256 private Boolean operators are considered, but NONE are shown.
    # Inference uses only physical role probes and actually chosen rewards.
    positions=(data["support_context"][index]==phase)
    if before:
        positions=positions & (data["support_context"][index]==0)
    slots=data["support_slots"][index][positions]
    returns=data["support_rewards"][index][positions]
    pattern=r104._patterns_from_public(slots)
    query=r104._patterns_from_public(data["query0"][index,phase][None])[0]
    choices=np.arange(256,dtype=np.uint16)
    guess=((choices[:,None]>>pattern[None,:])&1)>0
    ll=np.where(guess==(returns[None,:]>0),math.log(.85),math.log(.15)).sum(axis=1)
    weights=np.exp(ll-ll.max());weights/=weights.sum()
    predicted=float(weights@(2*((choices>>query)&1).astype(np.float64)-1))
    if abs(predicted)<1e-12:
        return r85.CALIBRATE if int(query)%2==0 else r85.EXPLOIT_POS
    return r85.CALIBRATE if predicted>0 else r85.EXPLOIT_POS


def evaluate(models,data,deadline):
    truth={"old_before":data["old_optimal"],
           "old_after":data["old_optimal"],
           "new_after":data["new_optimal"]}
    outputs={}
    for name in TRAINABLE:
        _deadline(deadline)
        outputs[name]={
            "old_before":_predict_actions(models[name],name,data,0,deadline,before=True),
            "old_after":_predict_actions(models[name],name,data,0,deadline),
            "new_after":_predict_actions(models[name],name,data,1,deadline)}
    outputs[BAYES]={}
    for stage,phase,before in (("old_before",0,True),
                               ("old_after",0,False),
                               ("new_after",1,False)):
        _deadline(deadline)
        outputs[BAYES][stage]=np.asarray([
            _bayes_public(data,i,phase,before)
            for i in range(len(data["support_slots"]))],np.int64)
    result={}
    for name,scores in outputs.items():
        result[name]={}
        for key,pred in scores.items():
            correct=int(np.sum(pred==truth[key]))
            result[name][key+"_accuracy"]=correct/len(pred)
            result[name][key+"_correct"]=correct
        for stage,idx in (("old_after",0),("new_after",1)):
            pred=scores[stage]
            returns=np.where(pred==r85.CALIBRATE,data["reward0"][:,idx],
                             data["reward2"][:,idx])
            result[name][stage+"_oracle_regret"]=float(np.mean(
                np.maximum(data["reward0"][:,idx],data["reward2"][:,idx])-returns))
        changed=data["private_switched"]
        result[name]["changed_law_new_after_accuracy"]=float(np.mean(
            scores["new_after"][changed]==truth["new_after"][changed]))
        result[name]["unchanged_law_new_after_accuracy"]=float(np.mean(
            scores["new_after"][~changed]==truth["new_after"][~changed]))
    return result


def cold_prediction_parity(model,data,deadline):
    """Exact cold prediction parity in a NEW isolated Python process.

    Fresh development state only: no owner, production or prior scored weights.
    Evaluate both actions and old/new contexts, including old-before-B state.
    """
    _deadline(deadline)
    model.eval()
    ix=np.arange(min(8,len(data["support_slots"])))
    cases=((0,r85.CALIBRATE,True),(0,r85.EXPLOIT_POS,True),
           (0,r85.CALIBRATE,False),(0,r85.EXPLOIT_POS,False),
           (1,r85.CALIBRATE,False),(1,r85.EXPLOIT_POS,False))
    samples=[_inputs(data,ix,NATIVE,context,action,before)
             for context,action,before in cases]
    with torch.no_grad():
        original=torch.stack([model(*sample) for sample in samples])
    stream=io.BytesIO()
    torch.save({"state":model.state_dict(),"samples":samples},stream)
    code=("import io,sys,torch\n"
          "from experiments.r110_contradiction_credit_diagnostic import NativeIdentifiableValue\n"
          "torch.set_num_threads(1)\n"
          "payload=torch.load(io.BytesIO(sys.stdin.buffer.read()),map_location='cpu',weights_only=True)\n"
          "model=NativeIdentifiableValue('adaptive')\n"
          "model.load_state_dict(payload['state']); model.eval()\n"
          "with torch.no_grad(): after=torch.stack([model(*s) for s in payload['samples']])\n"
          "torch.save(after,sys.stdout.buffer)\n")
    result=subprocess.run([sys.executable,"-c",code],
        cwd=ROOT,input=stream.getvalue(),capture_output=True,
        check=True,timeout=30)
    restored=torch.load(io.BytesIO(result.stdout),map_location="cpu",weights_only=True)
    if original.shape!=restored.shape:
        raise RuntimeError("R110_ISOLATED_COLD_RESTORE_SHAPE_DRIFT")
    _deadline(deadline)
    return int(torch.count_nonzero(original!=restored).item())


def frozen_checks(evals,cold_mismatch,intervention):
    thresholds=frozen()["evaluation"]["frozen_checks_all_seeds"]
    acc=lambda arm:evals[arm]["new_after_accuracy"]
    score=lambda arm,kind:evals[arm][kind]
    fixed=(SINGLE,TWO,STATIC)
    values={
        "native_new_after_accuracy_min":acc(NATIVE),
        "native_minus_no_aux_new_accuracy_min":acc(NATIVE)-acc(NO_AUX),
        "native_minus_placebo_aux_new_accuracy_min":acc(NATIVE)-acc(PLACEBO),
        "native_minus_best_fixed_route_new_accuracy_min":
            acc(NATIVE)-max(acc(k) for k in fixed),
        "native_minus_phase_only_new_accuracy_min":acc(NATIVE)-acc(PHASE_ONLY),
        "native_minus_context_blind_new_accuracy_min":acc(NATIVE)-acc(NO_CONTEXT),
        "native_minus_strongest_conventional_new_accuracy_min":
            acc(NATIVE)-max(acc(GRU),acc(MLP)),
        "native_minus_shuffle_new_accuracy_min":acc(NATIVE)-acc(SHUFFLE),
        "native_minus_zero_feedback_new_accuracy_min":acc(NATIVE)-acc(ZERO),
        "native_minus_missing_roles_new_accuracy_min":acc(NATIVE)-acc(NO_ROLES),
        "changed_law_native_minus_best_fixed_route_min":
            score(NATIVE,"changed_law_new_after_accuracy")-
            max(score(k,"changed_law_new_after_accuracy") for k in fixed),
        "unchanged_law_native_minus_best_fixed_route_min":
            score(NATIVE,"unchanged_law_new_after_accuracy")-
            max(score(k,"unchanged_law_new_after_accuracy") for k in fixed),
        "old_after_accuracy_min":evals[NATIVE]["old_after_accuracy"],
        "old_before_minus_old_after_accuracy_max":
            evals[NATIVE]["old_before_accuracy"]-evals[NATIVE]["old_after_accuracy"],
        "native_new_action_oracle_regret_max":score(NATIVE,"new_after_oracle_regret"),
        "gate_knockout_action_flip_count_min":intervention["gate_knockout_action_flip_count"],
        "factual_feedback_intervention_action_flip_count_min":
            intervention["feedback_action_flip_count"],
        "cold_reload_prediction_mismatch_max":cold_mismatch,
        "bayes256_new_after_accuracy_min":acc(BAYES)}
    if set(values)!=set(thresholds):
        raise RuntimeError("R110_FROZEN_THRESHOLD_MAPPING_DRIFT")
    checks={k:(v<=thresholds[k] if k.endswith("_max") else v>=thresholds[k])
            for k,v in values.items()}
    return {"values":values,"checks":checks,
            "all_unit_only_checks_pass":all(checks.values()),
            "scientific_pass":False}


def router_interventions(model,data,deadline):
    """Causal *input/model* sensitivity diagnostics, not new physics data.

    Public factual rewards are shuffled within each actually chosen
    action/context. The evaluator labels are never fed to this diagnostic.
    Also replace ONLY the gate by 0.5, preserving fitted expert weights.
    """
    original=_predict_actions(model,NATIVE,data,1,deadline)
    counter=dict(data)
    counter["support_rewards"]=data["support_rewards_shuffled"]
    factual=_predict_actions(model,NATIVE,counter,1,deadline)
    constant=_predict_actions(model,NATIVE,data,1,deadline,forced_gate=.5)
    return {"feedback_action_flip_count":int(np.count_nonzero(original!=factual)),
            "gate_knockout_action_flip_count":int(np.count_nonzero(original!=constant)),
            "queries":int(len(original))}


def public_router_bins(model,data,deadline):
    """Diagnostic gate distribution; evaluator labels used ONLY after inference."""
    model.eval();gates=[]
    with torch.no_grad():
        for start in range(0,len(data["support_slots"]),32):
            _deadline(deadline)
            ix=np.arange(start,min(start+32,len(data["support_slots"])))
            model(*_inputs(data,ix,NATIVE,1,r85.CALIBRATE))
            gates.extend(model.last_route_gate.reshape(-1).cpu().tolist())
    gates=np.asarray(gates)
    def bins(g):
        return {"share_lt_quarter":int(np.sum(g<.25)),
                "mixed_quarter_to_three_quarters":int(np.sum((g>=.25)&(g<.75))),
                "private_ge_three_quarters":int(np.sum(g>=.75))}
    changed=data["private_switched"]
    return {"changed":bins(gates[changed]),"unchanged":bins(gates[~changed]),
            "scored_query_count":int(len(gates))}


def run_seed(seed):
    if seed not in SEEDS:
        raise PermissionError("R110_UNREGISTERED_FRESH_SEED")
    frozen()
    torch.set_num_threads(1)
    t0=time.perf_counter()
    deadline=t0+frozen()["training"]["CPU_wall_budget_seconds_per_seed"]
    fit=source("UNIT_FIT",seed,deadline=deadline)
    valid=source("UNIT_VALID",seed,deadline=deadline)
    fit_credits=credit.fit_credit(fit)
    credit_weight()
    rng=np.random.default_rng(np.random.SeedSequence([SALTS["UNIT_BATCH"],seed]))
    minibatches=rng.integers(0,len(fit["target"]),size=(512,16))
    models,conventional_widths=make_models(seed)
    trained={}
    for name in TRAINABLE:
        _deadline(deadline)
        trained[name]=train_arm(models[name],name,fit,minibatches,deadline,fit_credits)
    if len(trained)!=13 or any(m["updates"]!=512 for m in trained.values()):
        raise RuntimeError("R110_PREMATURE_UNSEEN_OR_INCOMPLETE_CONTROL")
    # This is the very first access to R110 UNIT_ONLY unseen source.
    unseen=source("UNIT_UNSEEN",seed,deadline=deadline)
    counts={"phase_switch_positions":{str(x):int(np.sum(unseen["switch_after"]==x))
                                    for x in (6,8,10)},
            "changed_law":int(unseen["private_switched"].sum()),
            "unchanged_law":int((~unseen["private_switched"]).sum())}
    if (counts["changed_law"]!=64 or counts["unchanged_law"]!=64
        or max(counts["phase_switch_positions"].values())
            -min(counts["phase_switch_positions"].values())>1):
        raise RuntimeError("R110_PRIVATE_COHORT_BALANCING_PRECOMMITTED_FAILURE")
    evals=evaluate(models,unseen,deadline)
    cold=cold_prediction_parity(models[NATIVE],unseen,deadline)
    routing=public_router_bins(models[NATIVE],unseen,deadline)
    intervention=router_interventions(models[NATIVE],unseen,deadline)
    checks=frozen_checks(evals,cold,intervention)
    _deadline(deadline)
    return {"seed":seed,
            "status":(frozen()["evaluation"]["pass_status"]
                      if checks["all_unit_only_checks_pass"]
                      else frozen()["evaluation"]["fail_status"]),
            "all_thirteen_arms_fully_trained":True,
            "credit_fit_only_quantiles":{
               "min":float(fit_credits.min()),
               "p25":float(np.quantile(fit_credits,.25)),
               "median":float(np.median(fit_credits)),
               "p75":float(np.quantile(fit_credits,.75)),
               "max":float(fit_credits.max())},
            "trained":trained,"conventional_widths":conventional_widths,
            "cohort_balance":counts,"evaluation":evals,"router_bins":routing,"interventions":intervention,
            "frozen_checks":checks,"cold_restore_differences":cold,
            "seconds":time.perf_counter()-t0,
            "protected_original_firstuse":False,"scientific_pass":False}


def run_all():
    f=frozen()
    if (ROOT/"experiments/r110_cpu_firstuse_authorization.json").exists():
        raise PermissionError("R110_PROTECTED_SCIENCE_NOT_AUTHORIZED_IN_UNIT_ONLY")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R110_EXACT_CANDIDATE_SHA_DRIFT")
    results={str(seed):run_seed(seed) for seed in SEEDS}
    qualified=all(r["frozen_checks"]["all_unit_only_checks_pass"]
                  for r in results.values())
    return {"schema":"ttt-hybrid.r110.unit-only-public-contradiction-credit-result.v1",
            "exact_source_sha":git,"freeze_blob":r110.FREEZE_BLOB,
            "status":(f["evaluation"]["pass_status"] if qualified
                      else f["evaluation"]["fail_status"]),
            "by_seed":results,"all_three_seed_dev_gates_pass":qualified,
            "prior_original_scored_source_reused":False,
            "original_scientific_firstuse":False,"scientific_pass":False,
            "GPU_firstuse":False,"model_promoted":False,"deployment":"HOLD"}


def main():
    receipt=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R110_UNIT_ONLY_IMMUTABLE_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R110_UNIT_ONLY "+json.dumps({
        "sha":receipt["exact_source_sha"],"status":receipt["status"],
        "by_seed":{s:{"accuracy":{arm:{k:v for k,v in d.items()
                                        if k.endswith("_accuracy")}
                                  for arm,d in r["evaluation"].items()},
                       "interventions":r["interventions"],"checks":r["frozen_checks"]["checks"],
                       "seconds":round(r["seconds"],2)}
                   for s,r in receipt["by_seed"].items()},
        "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
