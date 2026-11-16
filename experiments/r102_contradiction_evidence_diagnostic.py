"""R102 new causal-operator evidence and contradiction-memory engineering ONLY.

New distinct source seeds/salts/operators, never R101 scored world/fixture/weights.
All protected FIT/DEV first use is permanently denied in this candidate.
Only R102 UNIT_ONLY examples may be constructed and evaluated in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r98_native_program_diagnostic as r98
from experiments import r101_unknown_operator_diagnostic as r101

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r102_contradiction_evidence_freeze.json"
FREEZE_BLOB = "9e4a74c4b903a8c205807f5ec0ba87d1ff028f82"
SCHEMA = "ttt-hybrid.r102-contradiction-gated-evidence.cpu-prospective-freeze.v1"
PARENT = "d6362f1ce587ea9a8ae00fcbe48aba2d5427e04e"
R101_SPENT = "ad21af19d342d239207629d562a0b4990e5fd163"
SEEDS = (1002711,1002723,1002737)
KNOWN = ("MUX3","PARITY3","AND3","OR3","NAND3","MAJORITY3","EXACTLY_ONE3")
UNSEEN = ("EXACTLY_TWO3","ALL_EQUAL3")
SUPPORT = 16
SIX = 6
UNIT_SALT = 25629991
ARMS = (
    "NATIVE16_CONTRADICTION_GATED",
    "NATIVE16_UNGATED_SAME_CAPACITY",
    "NATIVE16_SHUFFLED_FACTUAL_OUTCOMES",
    "NATIVE16_ZERO_OUTCOMES",
    "NATIVE16_RESET_EPISODE_MEMORY",
    "NATIVE6_CONTRADICTION_GATED",
    "EXTERNAL16_CONTRADICTION_GATED_MATCHED",
    "GRU64_16_FACTUAL_HISTORY",
    "ROLE_SET64_16_FACTUAL_HISTORY",
    "MLP64_16_FLATTENED",
    "STATIC_PRIOR16_NO_FEEDBACK",
)
(NATIVE,GATE_BLIND,SHUFFLED,NO_OUTCOME,RESET,SHORT,EXTERNAL,
 GRU,ROLE_SET,MLP,STATIC) = ARMS
PANEL_COUNTS = {"FIT":1024,"VALID":128,"DEV_FAMILIAR":128,
                "DEV_UNKNOWN":192,"OOD_SHIFT":128,"DELAYED_COLD":128,
                "REVERSAL":128}
LEARNED_PUBLIC_SHAPE = (5,13)
ROLENAMES = (0.25,0.5,0.75)


def frozen():
    """Identity and negative authority check, frozen at preimplementation SHA."""
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R102_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    if (f["schema"] != SCHEMA
        or f["stage"] != "PROSPECTIVE_FREEZE_ONLY_BEFORE_IMPLEMENTATION_AND_FIRST_USE"
        or f["lineage"]["parent_R101_sealed_SHA"] != PARENT
        or f["lineage"]["prior_R101_original_scored_SHA"] != R101_SPENT
        or f["lineage"]["prior_scored_sources_or_weights_reused"]
        or tuple(f["source"]["seeds"]) != SEEDS
        or f["source"]["counts"] != PANEL_COUNTS
        or tuple(f["source"]["known_train_families"]) != KNOWN
        or tuple(f["source"]["new_unseen_families"]) != UNSEEN
        or f["source"]["salts"]["UNIT_ONLY"] != UNIT_SALT
        or len(set(f["source"]["salts"].values())) != len(f["source"]["salts"])
        or f["source"]["support_interventions_total"] != SUPPORT
        or tuple(f["arms"]) != ARMS
        or f["training"]["steps_per_arm"] != 3072
        or f["training"]["batch_size"] != 16
        or f["evaluation"]["independent_seeds"] != len(SEEDS)
        or f["evaluation"]["independent_scientific_pass"] is not False
        or f["source"]["no_prior_cohort_replay"] is not True
        or f["source"]["one_first_use_per_seed"] is not True
        or f["authority"]["prospective_freeze_and_isolated_CPU_engineering"] is not True
        or any(bool(value) for name,value in f["authority"].items()
               if name != "prospective_freeze_and_isolated_CPU_engineering")):
        raise PermissionError("R102_FROZEN_AUTHORITY_OR_PROTOCOL_CHANGED")
    return f


def _admit(*_args,**_kwargs):
    frozen()
    raise PermissionError("R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


def _world(panel,seed,index,*_args,**_kwargs):
    f=frozen()
    if panel not in f["source"]["counts"]:
        raise PermissionError("R102_OLD_SPENT_OR_NONREGISTERED_SOURCE_DENIED")
    if (seed not in SEEDS or not isinstance(index,int)
        or not 0 <= index < f["source"]["counts"][panel]):
        raise PermissionError("R102_UNREGISTERED_SEED_OR_INDEX")
    raise PermissionError("R102_PROTECTED_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED")


def original_ci(*_args,**_kwargs):
    raise PermissionError("R102_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


def family_for_panel(panel,index,event=SUPPORT):
    f=frozen()
    if (panel not in PANEL_COUNTS or not isinstance(index,int)
        or not 0<=index<PANEL_COUNTS[panel] or not 0<=event<=SUPPORT):
        raise PermissionError("R102_FROZEN_SOURCE_IDENTITY_INVALID")
    if panel in ("FIT","VALID","DEV_FAMILIAR"):
        return KNOWN[index%len(KNOWN)]
    # Joint balance: public time is parity, chosen query action is index//4.
    # In each 4-index group, both families occur at both public pivots.
    chosen=(index//2)%2
    if panel=="REVERSAL" and event<SUPPORT//2:
        chosen=(chosen+1)%2
    return UNSEEN[chosen]


def query_action(seed,index):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R102_QUERY_SEED_INDEX_INVALID")
    return (r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0
            else r85.EXPLOIT_POS)


def support_action(event):
    if not isinstance(event,int) or not 0<=event<SUPPORT:
        raise PermissionError("R102_SUPPORT_ACTION_EVENT_INVALID")
    return r85.CALIBRATE if event%2==0 else r85.EXPLOIT_POS


def _operator_truth(factors, roles, family):
    """PRIVATE EVALUATOR RULES ONLY; no model calls or feature inheritance."""
    if family not in KNOWN+UNSEEN:
        raise PermissionError("R102_UNREGISTERED_CAUSAL_OPERATOR")
    a=np.asarray(factors)
    if (a.shape!=(4,) or not np.all(np.isin(a,[-1,1])) or len(roles)!=3
        or len(set(roles))!=3 or any(not 0<=int(k)<4 for k in roles)):
        raise ValueError("R102_INVALID_PRIVATE_PHYSICAL_LAW")
    x,y,z=(int(a[int(k)]>0) for k in roles)
    total=x+y+z
    if family=="MUX3": val=y if x else z
    elif family=="PARITY3": val=total%2==1
    elif family=="AND3": val=total==3
    elif family=="OR3": val=total>0
    elif family=="NAND3": val=total!=3
    elif family=="MAJORITY3": val=total>=2
    elif family=="EXACTLY_ONE3": val=total==1
    elif family=="EXACTLY_TWO3": val=total==2
    elif family=="ALL_EQUAL3": val=total in (0,3)
    else: raise RuntimeError("R102_OPERATOR_EVALUATOR_IMPL_MISSING")
    return 1 if val else -1


def _unit_world(index,event,family):
    """Independent R102-salt simulated physics, NOT a scored FIT/DEV cohort."""
    frozen()
    if (not isinstance(index,int) or not 0<=index<10
        or not isinstance(event,int) or not 0<=event<=SUPPORT
        or family not in KNOWN+UNSEEN):
        raise PermissionError("R102_UNIT_ONLY_SOURCE_NOT_REGISTERED")
    rng=np.random.default_rng(np.random.SeedSequence(
        [UNIT_SALT,index,event]))
    physics_index=(2*(index%(r85.N//2))+index%2)%r85.N
    w=r85._make_synthetic_world(physics_index,rng)
    roles=tuple(int(k) for k in rng.permutation(4)[:3])
    facts=rng.choice([-1,1],size=4).astype(np.int64)
    w["factors_a"]=facts
    w["r102_roles"]=roles
    w["law_a"]=_operator_truth(facts,roles,family)
    return w


def public_view(world,action):
    """R85/R86 pure physical action return; no preaction private truth."""
    prefix,reward=r86._view(world,action)
    slots=r88._public_slots(prefix).copy()
    roles=np.zeros(4,np.float32)
    for k,role in enumerate(world["r102_roles"],start=1):
        roles[role]=k/4.
    slots[:4,8]=roles
    if slots.shape!=LEARNED_PUBLIC_SHAPE or not np.isfinite(slots).all():
        raise RuntimeError("R102_PUBLIC_CUE_PHYSICS_SHAPE_INVALID")
    return slots,float(reward)


def unit_fixture(index):
    """One independent UNIT_ONLY episode with 16 factual action returns."""
    if not isinstance(index,int) or not 0<=index<10:
        raise PermissionError("R102_TEST_ONLY_FIXTURE_INDEX")
    fam=(KNOWN+UNSEEN)[index%len(KNOWN+UNSEEN)]
    events=[]; returns=[]; acts=[]
    for event in range(SUPPORT):
        action=support_action(event)
        w=_unit_world(index,event,fam)
        cues,ret=public_view(w,action)
        events.append(cues);returns.append(ret);acts.append(action)
    qworld=_unit_world(index,SUPPORT,fam)
    q0,r0=public_view(qworld,r85.CALIBRATE)
    q2,r2=public_view(qworld,r85.EXPLOIT_POS)
    if (not np.array_equal(q0[:4],q2[:4])
        or not np.array_equal(q0[-1,4:],q2[-1,4:])):
        raise RuntimeError("R102_EVALUATOR_QUERY_PREACTION_LEAK")
    return {"public":{
        "support_slots":np.stack(events).astype(np.float32),
        "support_rewards":np.asarray(returns,np.float32),
        "support_actions":np.asarray(acts,np.int8),
        "query0":q0,"query2":q2},
        "evaluator_only":{
            "private_family":fam,
            "private_law":int(qworld["law_a"]),
            "return0":float(r0),"return2":float(r2)}}


def public_support_shuffle(actions,slots,index):
    """Deterministic UNIT_ONLY action/time matched outcome derangement."""
    frozen()
    if (not isinstance(index,int) or not 0<=index<10
        or actions.shape!=(SUPPORT,) or slots.shape!=(SUPPORT,5,13)):
        raise PermissionError("R102_SHUFFLE_ONLY_IN_UNIT_FIXTURE")
    keys=[(int(actions[k]),float(slots[k,-1,12])) for k in range(SUPPORT)]
    order=np.arange(SUPPORT)
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index,1801]))
    for label in sorted(set(keys)):
        group=np.asarray([j for j in range(SUPPORT) if keys[j]==label])
        if len(group)<2:
            raise RuntimeError("R102_SOURCE_SHUFFLE_STRATUM_TOO_SMALL")
        order[group]=np.roll(group,int(rng.integers(1,len(group))))
    if (any(order[j]==j or keys[j]!=keys[order[j]] for j in range(SUPPORT))
        or not np.array_equal(np.sort(order),np.arange(SUPPORT))):
        raise RuntimeError("R102_SOURCE_SHUFFLE_NOT_DERANGED")
    return order


def non_scoring_design_audit():
    """No world construction; inspect only frozen combinatorics."""
    f=frozen()
    tables={}
    for fam in KNOWN+UNSEEN:
        tables[fam]=tuple(_operator_truth(
            np.array([2*((k>>2)&1)-1,2*((k>>1)&1)-1,
                      2*(k&1)-1,-1],np.int64),(0,1,2),fam)
            for k in range(8))
    if len(set(tables.values()))!=len(tables):
        raise RuntimeError("R102_PRIVATE_TRUTH_FAMILY_NOT_DISTINCT")
    comparisons={}
    for unseen in UNSEEN:
        differences={known:sum(t!=u for t,u in zip(tables[known],tables[unseen]))
                     for known in KNOWN}
        closest=min(differences.values())
        comparisons[unseen]={
            "nearest_known_hamming_distance_of_8":closest,
            "six_iid_uniform_ideal_feedback_ambiguity":((8-closest)/8)**SIX,
            "sixteen_iid_uniform_ideal_feedback_ambiguity":((8-closest)/8)**SUPPORT}
    balance={}
    for panel,count in PANEL_COUNTS.items():
        if panel in ("FIT","VALID","DEV_FAMILIAR"):
            families=KNOWN
        else:families=UNSEEN
        grids={}
        for seed in SEEDS:
            cells={}
            for index in range(count):
                pivot="COLD" if panel=="DELAYED_COLD" or index%2 else "EARLY"
                action=query_action(seed,index)
                group=f"{pivot}:{action}"
                cell=cells.setdefault(group,{name:0 for name in families})
                cell[family_for_panel(panel,index)]+=1
                if panel=="REVERSAL" and (family_for_panel(panel,index,0)
                                         ==family_for_panel(panel,index,SUPPORT)):
                    raise RuntimeError("R102_REVERSAL_HAS_NOT_CHANGED_HIDDEN_LAW")
            for cell in cells.values():
                if (any(v==0 for v in cell.values()) or
                    max(cell.values())-min(cell.values())>
                     (2 if len(families)>2 else 0)):
                    raise RuntimeError("R102_PRIVATE_FAMILY_PUBLIC_TIME_ACTION_LEAK")
            grids[str(seed)]=cells
        balance[panel]=grids
    return {"schema":"ttt-hybrid.r102.nonscoring-design-audit.v1",
            "original_scored_source_opened":False,
            "contradiction_efficacy_unverified":True,
            "R101_scored_cohort_reuse":False,
            "private_evaluator_truth_only":True,
            "combinatorial_family_distinguishability":comparisons,
            "public_pivot_action_family_cell_balance":balance,
            "independent_scientific_pass":False}


class EvidenceMemory(nn.Module):
    """Differentiable online state updates based on *factual* microfeedback.

    Uses publicly observed physical cue/action and actual selected return.
    Prediction residual is a LEARNED factual surprise; NOT private law.
    """
    def __init__(self,width=32,contradiction=True):
        super().__init__()
        self.width=width
        self.contradiction=contradiction
        self.public=r101.PublicEncoder()
        self.prediction=nn.Linear(width+32,1)
        self.proposal=nn.GRUCell(33,width)
        self.gate=nn.Sequential(nn.Linear(width+32+2,width),
                                nn.GELU(),nn.Linear(width,width))
    def forward(self,slots,rewards):
        if (slots.ndim!=4 or slots.shape[-2:]!=(5,13)
            or slots.shape[1] not in (SIX,SUPPORT)
            or rewards.shape!=slots.shape[:2]):
            raise RuntimeError("R102_MATCHED_FACTUAL_SUPPORT_SHAPE_INVALID")
        b,n=slots.shape[:2]
        cues=self.public(slots.reshape(-1,5,13)).reshape(b,n,32)
        state=cues.new_zeros((b,self.width))
        for i in range(n):
            cue=cues[:,i]
            reward=rewards[:,i:i+1]
            predicted=self.prediction(torch.cat((state,cue),-1))
            # For the contradiction-blind control, preserve same actively
            # trainable prediction head through zero-effect residual with
            # differentiable zero, but no source of contradiction.
            residual=(reward-predicted) if self.contradiction else \
                     predicted-predicted.detach()
            weights=torch.sigmoid(self.gate(torch.cat(
                (state,cue,reward,residual),-1)))
            proposal=self.proposal(torch.cat((cue,reward),-1),state)
            state=state+weights*(proposal-state)
        return state


class NativeEvidenceValue(nn.Module):
    def __init__(self,core,mode="gated"):
        super().__init__()
        if mode not in ("gated","blind","none","reset","short"):
            raise ValueError("R102_NATIVE_MODE_UNREGISTERED")
        self.core=core
        self.mode=mode
        self.support=EvidenceMemory(32,contradiction=(mode!="blind"))
        self.projection=nn.Linear(32,32)
    def forward(self,slots,rewards,query):
        if self.mode=="short":
            slots=slots[:,:SIX]
            rewards=rewards[:,:SIX]
        if self.mode=="none":
            rewards=torch.zeros_like(rewards)
        memory=self.support(slots,rewards)
        if self.mode=="reset":
            memory=torch.zeros_like(memory)
        stem=self.core.stem(query)
        ctx=self.projection(memory)
        stem=torch.cat((stem[:,:-1],stem[:,-1:]+ctx[:,None]),1)
        return r98._native_trunk(self.core,stem)


class ExternalEvidenceValue(nn.Module):
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.support=EvidenceMemory(32,contradiction=True)
        self.value=nn.Sequential(nn.Linear(32,32),nn.GELU(),
                                 nn.Linear(32,1))
    def forward(self,slots,rewards,query):
        z=self.support(slots,rewards)
        return self.core(query)[:,-1,0]+self.value(z)[:,0]


class ConventionalEvidenceValue(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ("gru","set","flat","static"):
            raise ValueError("R102_CONVENTIONAL_MODE_UNREGISTERED")
        self.mode=mode
        self.query=r101.PublicEncoder()
        if mode!="static":
            self.public=r101.PublicEncoder()
        if mode=="gru":
            self.history=nn.GRU(33,64,batch_first=True)
        if mode=="set":
            self.history=nn.Sequential(nn.Linear(33,64),nn.GELU(),
                                       nn.Linear(64,64))
        if mode=="flat":
            self.flat=nn.Sequential(nn.Linear(SUPPORT*33+32,64),nn.GELU(),
                                    nn.Linear(64,64),nn.GELU(),
                                    nn.Linear(64,1))
        else:
            self.head=nn.Sequential(nn.Linear((0 if mode=="static" else 64)+32,64),
                                    nn.GELU(),nn.Linear(64,1))
    def forward(self,slots,rewards,query):
        q=self.query(query)
        if self.mode=="static":
            return self.head(q)[:,0]
        if (slots.ndim!=4 or slots.shape[1:]!=(SUPPORT,5,13)
            or rewards.shape!=slots.shape[:2]):
            raise RuntimeError("R102_CONVENTIONAL_FACTUAL_SOURCE_SHAPE")
        b=len(slots)
        ev=self.public(slots.reshape(-1,5,13)).reshape(b,SUPPORT,32)
        record=torch.cat((ev,rewards.unsqueeze(-1)),-1)
        if self.mode=="flat":
            return self.flat(torch.cat((record.reshape(b,-1),q),-1))[:,0]
        memory=(self.history(record)[1][-1] if self.mode=="gru"
                else self.history(record).mean(dim=1))
        return self.head(torch.cat((memory,q),-1))[:,0]


def make_models(seed):
    """Fresh R102 only: equal source, common init, no prior R101 weights."""
    if seed not in SEEDS:
        raise PermissionError("R102_UNREGISTERED_FRESH_MODEL_SEED")
    models={}
    for name,mode in ((NATIVE,"gated"),(GATE_BLIND,"blind"),
                      (SHUFFLED,"gated"),(NO_OUTCOME,"none"),
                      (RESET,"reset"),(SHORT,"short")):
        core=r95._native(seed)
        torch.manual_seed(seed+102)
        models[name]=NativeEvidenceValue(core,mode)
    core=r95._native(seed)
    torch.manual_seed(seed+102)
    models[EXTERNAL]=ExternalEvidenceValue(core)
    for i,(name,mode) in enumerate(((GRU,"gru"),(ROLE_SET,"set"),
                                    (MLP,"flat"),(STATIC,"static"))):
        torch.manual_seed(seed+501+i)
        models[name]=ConventionalEvidenceValue(mode)
    # Native and EXTERNAL share exact same initial native core, but distinct
    # paths for writing factual bounded memory.
    cores=[dict(models[name].core.named_parameters()) for name in
           (NATIVE,GATE_BLIND,SHUFFLED,NO_OUTCOME,RESET,SHORT,EXTERNAL)]
    if any(cores[0].keys()!=c.keys() for c in cores[1:]) or any(
        not torch.equal(cores[0][k],c[k]) for c in cores[1:] for k in cores[0]):
        raise RuntimeError("R102_NATIVE_CORE_SAME_INIT_MISMATCH")
    adapt=lambda m:sum(p.numel() for name,p in m.named_parameters()
                      if p.requires_grad and not name.startswith("core."))
    na,ea=adapt(models[NATIVE]),adapt(models[EXTERNAL])
    if not (na>0 and ea>0 and abs(na-ea)<=.10*max(na,ea)):
        raise RuntimeError("R102_NATIVE_EXTERNAL_ADAPTER_CAPACITY_UNMATCHED")
    return models


def unit_value(model,fixture,action=r85.CALIBRATE,shuffle=False):
    """No scoring; exercise only R102 UNIT_ONLY physically factual data."""
    p=fixture["public"]
    rewards=p["support_rewards"].copy()
    if shuffle:
        rewards=rewards[public_support_shuffle(
            p["support_actions"],p["support_slots"],0)]
    x=torch.from_numpy(p["support_slots"])[None]
    y=torch.from_numpy(rewards)[None]
    q=torch.from_numpy(p["query0" if action==r85.CALIBRATE else "query2"])[None]
    return model(x,y,q)
