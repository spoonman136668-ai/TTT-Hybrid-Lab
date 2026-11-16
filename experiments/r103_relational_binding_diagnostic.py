"""R103 fresh non-token relational microfeedback: UNIT_ONLY engineering proposal.

R102 scored FIT/DEV cohorts are CONSUMED; no R102 _world, scored source
records, saved models or source receipts are imported. New R103 physical
world construction is permitted ONLY on the distinct UNIT_ONLY salt.
Every original R103 FIT/VALID/DEV route fails closed until separately
authorized and exact-source qualified scientific first-use.
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
from experiments import r102_contradiction_evidence_diagnostic as r102

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r103_relational_binding_freeze.json"
FREEZE_BLOB="e9a12e3e5224abf162ae00e82b0781987c00484d"
SCHEMA="ttt-hybrid.r103-relational-binding.cpu-prospective-freeze.v1"
PARENT="4c4d41697fcab5b27c7c6912f613ef392aad4233"
R102_SPENT="745ce5c9ba2c7bb0dbfa3d44433e635a079a0ce2"
SEEDS=(1003011,1003023,1003037)
KNOWN=("MUX3","PARITY3","AND3","OR3","NAND3","MAJORITY3",
       "EXACTLY_ONE3","EXACTLY_TWO3","ALL_EQUAL3")
UNSEEN=("A_XNOR_B_AND_C","A_XNOR_B_OR_C")
ARMS=(
    "NATIVE16_RELATIONAL_CONTRADICTION",
    "NATIVE16_RELATIONAL_UNGATED",
    "NATIVE16_RELATIONAL_SHUFFLED",
    "NATIVE16_RELATIONAL_NO_OUTCOMES",
    "NATIVE16_RELATIONAL_RESET",
    "NATIVE16_R102_SCALAR_GATE",
    "EXTERNAL16_RELATIONAL_MATCHED",
    "GRU64_16_FACTUAL_HISTORY",
    "ROLE_SET64_16_FACTUAL_HISTORY",
    "MLP64_16_FLATTENED",
    "STATIC_PRIOR16_NO_FEEDBACK",
)
(NATIVE,UNGATED,SHUFFLED,ZERO,RESET,SCALAR,EXTERNAL,GRU,ROLE_SET,MLP,STATIC)=ARMS
SUPPORT=16
UNIT_SALT=26339991
PANEL_COUNTS={"FIT":1024,"VALID":128,"DEV_FAMILIAR":128,"DEV_UNKNOWN":192,
              "OOD_SHIFT":128,"DELAYED_COLD":128,"REVERSAL":128}
SCORING_PANELS=("DEV_FAMILIAR","DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL")


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R103_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    if (f["schema"]!=SCHEMA or
        f["stage"]!="PROSPECTIVE_FREEZE_ONLY_BEFORE_IMPLEMENTATION_AND_FIRST_USE"
        or f["lineage"]["base_parent_R102_sealed_SHA"]!=PARENT
        or f["lineage"]["prior_R102_original_scored_SHA"]!=R102_SPENT
        or f["lineage"]["prior_R102_scored_source_or_weights_reused"]
        or tuple(f["source"]["seeds"])!=SEEDS
        or f["source"]["salts"]["UNIT_ONLY"]!=UNIT_SALT
        or len(set(f["source"]["salts"].values()))!=len(f["source"]["salts"])
        or f["source"]["counts"]!=PANEL_COUNTS
        or tuple(f["source"]["known_train_families"])!=KNOWN
        or tuple(f["source"]["new_unseen_families"])!=UNSEEN
        or f["source"]["support_interventions_total"]!=SUPPORT
        or tuple(f["arms"])!=ARMS
        or f["training"]["steps_per_arm"]!=3072
        or f["training"]["batch_size"]!=16
        or f["evaluation"]["full_seed_count"]!=3
        or f["evaluation"]["independent_scientific_pass"] is not False
        or any(bool(v) for k,v in f["authority"].items()
               if k!="prospective_freeze_and_isolated_CPU_engineering")
        or f["authority"]["prospective_freeze_and_isolated_CPU_engineering"] is not True):
        raise PermissionError("R103_FROZEN_SOURCE_OR_OWNER_SCOPE_DRIFT")
    return f


def _admit(*_args,**_kw):
    frozen()
    raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


def _world(panel,seed,index,sha=None,*,event=0,family=None):
    """Original physical cohort prohibited, including FIT, VALID and DEV."""
    f=frozen()
    if panel not in PANEL_COUNTS:
        raise PermissionError("R103_PRIOR_SPENT_OR_UNKNOWN_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<PANEL_COUNTS[panel]:
        raise PermissionError("R103_UNREGISTERED_SEED_OR_EPISODE")
    raise PermissionError("R103_PROTECTED_ORIGINAL_PHYSICAL_SOURCE_NOT_ADMITTED")


def original_ci(*_args,**_kwargs):
    raise PermissionError("R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


def family_for_panel(panel,index,event=SUPPORT):
    if (panel not in PANEL_COUNTS or not isinstance(index,int)
        or not 0<=index<PANEL_COUNTS[panel] or not isinstance(event,int)
        or not 0<=event<=SUPPORT):
        raise PermissionError("R103_FROZEN_FAMILY_INDEX_OR_EVENT_INVALID")
    if panel in ("FIT","VALID","DEV_FAMILIAR"):
        return KNOWN[index%len(KNOWN)]
    kind=(index//2)%2
    if panel=="REVERSAL" and event<SUPPORT//2:
        kind=1-kind
    return UNSEEN[kind]


def query_action(seed,index):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R103_UNREGISTERED_QUERY_ACTION_SEED")
    return (r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0
            else r85.EXPLOIT_POS)


def support_action(event):
    if not isinstance(event,int) or not 0<=event<SUPPORT:
        raise PermissionError("R103_UNREGISTERED_FACTUAL_INTERVENTION")
    return r85.CALIBRATE if event%2==0 else r85.EXPLOIT_POS


def _operator_truth(factors,roles,family):
    """PRIVATE EVALUATOR truth; forbidden as model input or training label."""
    if family not in KNOWN+UNSEEN:
        raise PermissionError("R103_EVALUATOR_OPERATOR_NOT_REGISTERED")
    factors=np.asarray(factors)
    if (factors.shape!=(4,) or not np.all(np.isin(factors,[-1,1]))
        or len(roles)!=3 or len(set(roles))!=3
        or any(not 0<=int(k)<4 for k in roles)):
        raise ValueError("R103_INVALID_PRIVATE_PHYSICS")
    a,b,c=(int(factors[int(k)]>0) for k in roles)
    n=a+b+c
    if family=="MUX3": out=b if a else c
    elif family=="PARITY3":out=n%2
    elif family=="AND3":out=n==3
    elif family=="OR3":out=n>0
    elif family=="NAND3":out=n!=3
    elif family=="MAJORITY3":out=n>=2
    elif family=="EXACTLY_ONE3":out=n==1
    elif family=="EXACTLY_TWO3":out=n==2
    elif family=="ALL_EQUAL3":out=n in (0,3)
    elif family=="A_XNOR_B_AND_C":out=1-(a^(b&c))
    elif family=="A_XNOR_B_OR_C":out=1-(a^(b|c))
    else:raise RuntimeError("R103_EVALUATOR_FAMILY_MISSING")
    return 1 if out else -1


def _unit_world(index,event,family):
    """Fresh R103-only UNIT salt; no scored panel, no R102 fixture calls."""
    frozen()
    if (not isinstance(index,int) or not 0<=index<11
        or not isinstance(event,int) or not 0<=event<=SUPPORT
        or family not in KNOWN+UNSEEN):
        raise PermissionError("R103_UNIT_ONLY_SOURCE_SCOPE_DENIED")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index,event]))
    physics_index=(2*(index%(r85.N//2))+index%2)%r85.N
    world=r85._make_synthetic_world(physics_index,rng)
    roles=tuple(int(k) for k in rng.permutation(4)[:3])
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r103_roles"]=roles
    world["law_a"]=_operator_truth(factors,roles,family)
    return world


def public_view(world,action):
    """Public R85 physical cues and actually returned chosen reward only."""
    rows,reward=r86._view(world,action)
    slots=r88._public_slots(rows).copy()
    role_tags=np.zeros(4,np.float32)
    for role_num,role_idx in enumerate(world["r103_roles"],start=1):
        role_tags[role_idx]=role_num/4.
    slots[:4,8]=role_tags
    if slots.shape!=(5,13) or not np.isfinite(slots).all():
        raise RuntimeError("R103_PUBLIC_SLOTS_SHAPE_OR_NAN")
    return slots,float(reward)


def unit_fixture(index):
    """Independent 16-choice UNIT_ONLY causal episode, never a scored panel."""
    if not isinstance(index,int) or not 0<=index<11:
        raise PermissionError("R103_UNIT_FIXTURE_OUTSIDE_REGISTERED_SCOPE")
    family=(KNOWN+UNSEEN)[index]
    views=[];rewards=[];actions=[]
    for event in range(SUPPORT):
        action=support_action(event)
        w=_unit_world(index,event,family)
        slots,actual=public_view(w,action)
        views.append(slots);rewards.append(actual);actions.append(action)
    query=_unit_world(index,SUPPORT,family)
    q0,r0=public_view(query,r85.CALIBRATE)
    q2,r2=public_view(query,r85.EXPLOIT_POS)
    if (not np.array_equal(q0[:4],q2[:4])
        or not np.array_equal(q0[-1,4:],q2[-1,4:])):
        raise RuntimeError("R103_TEST_PRIVATE_QUERY_LEAK")
    return {"public":{
        "support_slots":np.stack(views).astype(np.float32),
        "support_rewards":np.asarray(rewards,np.float32),
        "support_actions":np.asarray(actions,np.int8),
        "query0":q0,"query2":q2,
    },"evaluator_only":{
        "operator":family,"private_law":int(query["law_a"]),
        "return0":float(r0),"return2":float(r2)
    }}


def public_shuffled_order(actions,slots,index):
    """UNIT_ONLY factual-outcome derangement inside public action/time strata."""
    if (not isinstance(index,int) or not 0<=index<11 or
        actions.shape!=(SUPPORT,) or slots.shape!=(SUPPORT,5,13)):
        raise PermissionError("R103_UNIT_SHUFFLE_SCOPE_DENIED")
    groups=[(int(actions[j]),float(slots[j,-1,12])) for j in range(SUPPORT)]
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index,1117]))
    order=np.arange(SUPPORT)
    for k in sorted(set(groups)):
        members=np.asarray([j for j in range(SUPPORT) if groups[j]==k])
        if len(members)<2:
            raise RuntimeError("R103_SHUFFLE_UNDERPOPULATED_PUBLIC_STRATUM")
        order[members]=np.roll(members,int(rng.integers(1,len(members))))
    if (not np.array_equal(np.sort(order),np.arange(SUPPORT)) or
        any(order[j]==j or groups[j]!=groups[order[j]] for j in range(SUPPORT))):
        raise RuntimeError("R103_FACTUAL_SHUFFLE_FAILED")
    return order


def non_scoring_design_audit():
    """Prospective static private-truth and identity balance only; no world."""
    f=frozen()
    truth={}
    for family in KNOWN+UNSEEN:
        truth[family]=tuple(_operator_truth(np.asarray([
            2*((j>>2)&1)-1,2*((j>>1)&1)-1,2*(j&1)-1,-1]),
            (0,1,2),family) for j in range(8))
    if len(set(truth.values()))!=len(truth):
        raise RuntimeError("R103_HELDOUT_OR_KNOWN_OPERATOR_DUPLICATE")
    distances={u:min(sum(x!=y for x,y in zip(truth[u],truth[k]))
                     for k in KNOWN) for u in UNSEEN}
    pair=sum(x!=y for x,y in zip(truth[UNSEEN[0]],truth[UNSEEN[1]]))
    if min(distances.values())<f["source"]["unseen_truth_functions_min_hamming_distance_from_any_known_8_cases"] or pair!=f["source"]["unseen_pair_hamming_distance_of_8"]:
        raise RuntimeError("R103_TRUTH_DISTINGUISHABILITY_FROZEN_FAIL")
    panels={}
    for panel in ("DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL"):
        seed_cells={}
        for seed in SEEDS:
            groups={}
            for j in range(PANEL_COUNTS[panel]):
                pivot="COLD" if panel=="DELAYED_COLD" or j%2 else "EARLY"
                label=f"{pivot}|{query_action(seed,j)}"
                distribution=groups.setdefault(label,{u:0 for u in UNSEEN})
                family=family_for_panel(panel,j)
                distribution[family]+=1
                if panel=="REVERSAL" and family_for_panel(panel,j,0)==family:
                    raise RuntimeError("R103_SOURCE_REVERSAL_NO_PRIVATE_CHANGE")
            if any(not set(v.values())=={len(v) and next(iter(v.values()))} or
                   0 in v.values() for v in groups.values()):
                raise RuntimeError("R103_PUBLIC_TIME_ACTION_OPERATOR_IDENTITY_LEAK")
            seed_cells[str(seed)]=groups
        panels[panel]=seed_cells
    return {"schema":"ttt-hybrid.r103.nonscoring-identity-audit.v1",
            "original_scored_source_opened":False,
            "prior_scored_cohorts_reused":False,
            "independent_scientific_pass":False,
            "nearest_known_truth_hamming":distances,
            "unseen_pair_hamming":pair,
            "public_pivot_action_family_balance":panels}


class RelationalMemory(nn.Module):
    """Within-episode locally updated 16x16 associative matrix, NO tokens.

    Each factual support cue yields a learned 16D key and value; the
    *actually observed* chosen reward corrects its predicted consequence.
    No operator truth, private factor, future query or counterfactual is seen.
    Outer fitted weights get gradient through the 16 factual writes.
    """
    def __init__(self,contradiction=True):
        super().__init__()
        self.contradiction=contradiction
        self.public=r101.PublicEncoder()
        self.key=nn.Linear(32,16)
        self.value=nn.Linear(34,16)
        self.pred=nn.Sequential(nn.Linear(48,32),nn.GELU(),nn.Linear(32,1))
        self.gate=nn.Sequential(nn.Linear(50,32),nn.GELU(),nn.Linear(32,1))
    def encode_key(self,public):
        cue=self.public(public)
        return cue,F.normalize(torch.tanh(self.key(cue)),p=2,dim=-1,eps=1e-5)
    def forward(self,slots,rewards,query):
        if (slots.ndim!=4 or slots.shape[1:]!=(SUPPORT,5,13)
            or rewards.shape!=slots.shape[:2]
            or query.shape!=(len(slots),5,13)):
            raise RuntimeError("R103_RELATIONAL_FEEDBACK_SHAPES_OR_SUPPORT_COUNT")
        b=len(slots)
        mat=query.new_zeros((b,16,16))
        cues=self.public(slots.reshape(-1,5,13)).reshape(b,SUPPORT,32)
        keys=F.normalize(torch.tanh(self.key(cues)),p=2,dim=-1,eps=1e-5)
        for i in range(SUPPORT):
            cue=cues[:,i]
            key=keys[:,i]
            reward=rewards[:,i:i+1]
            read=torch.bmm(key.unsqueeze(1),mat).squeeze(1)
            predicted=self.pred(torch.cat((cue,read),dim=-1))
            residual=reward-predicted if self.contradiction else torch.zeros_like(reward)
            gate=torch.sigmoid(self.gate(torch.cat(
                (cue,read,reward,residual),dim=-1)))
            # The blind control uses identical trainable prediction/proposal
            # heads but gates on factual rewards, NOT prediction mismatch.
            value=torch.tanh(self.value(torch.cat((cue,reward,predicted),dim=-1)))
            strength=gate*(residual if self.contradiction else reward)
            mat=.95*mat+strength.unsqueeze(-1)*torch.bmm(
                key.unsqueeze(-1),value.unsqueeze(1))
        qcue,qkey=self.encode_key(query)
        result=torch.bmm(qkey.unsqueeze(1),mat).squeeze(1)
        return result,mat


class NativeRelationalValue(nn.Module):
    def __init__(self,core,mode="gated"):
        super().__init__()
        if mode not in ("gated","blind","none","reset"):
            raise ValueError("R103_NATIVE_RELATIONAL_MODE_NOT_REGISTERED")
        self.core=core
        self.mode=mode
        self.memory=RelationalMemory(contradiction=(mode!="blind"))
        self.projection=nn.Linear(16,32)
    def forward(self,slots,rewards,query):
        if self.mode=="none":
            rewards=torch.zeros_like(rewards)
        retrieved,_matrix=self.memory(slots,rewards,query)
        if self.mode=="reset":
            retrieved=torch.zeros_like(retrieved)
        stem=self.core.stem(query)
        ctx=self.projection(retrieved)
        stem=torch.cat((stem[:,:-1],stem[:,-1:]+ctx[:,None]),dim=1)
        return r98._native_trunk(self.core,stem)


class ExternalRelationalValue(nn.Module):
    """Same fresh native core + same learned relation memory, external read."""
    def __init__(self,core):
        super().__init__()
        self.core=core
        self.memory=RelationalMemory(contradiction=True)
        self.projection=nn.Linear(16,32)
        self.head=nn.Linear(32,1)
    def forward(self,slots,rewards,query):
        retrieved,_matrix=self.memory(slots,rewards,query)
        return self.core(query)[:,-1,0]+self.head(
            self.projection(retrieved))[:,0]


def make_models(seed):
    """All 11 FROM SCRATCH with common native core initialization."""
    if seed not in SEEDS:
        raise PermissionError("R103_MODEL_SEED_NOT_REGISTERED")
    models={}
    for name,mode in ((NATIVE,"gated"),(UNGATED,"blind"),
                      (SHUFFLED,"gated"),(ZERO,"none"),(RESET,"reset")):
        core=r95._native(seed)
        torch.manual_seed(seed+103)
        models[name]=NativeRelationalValue(core,mode)
    core=r95._native(seed)
    torch.manual_seed(seed+103)
    models[EXTERNAL]=ExternalRelationalValue(core)
    core=r95._native(seed)
    torch.manual_seed(seed+103)
    models[SCALAR]=r102.NativeEvidenceValue(core,"gated")
    for i,(name,mode) in enumerate(((GRU,"gru"),(ROLE_SET,"set"),
                                    (MLP,"flat"),(STATIC,"static"))):
        torch.manual_seed(seed+503+i)
        models[name]=r102.ConventionalEvidenceValue(mode)
    if set(models)!=set(ARMS):
        raise RuntimeError("R103_FROZEN_ARM_CREATION_MISMATCH")
    cores=[dict(models[n].core.named_parameters()) for n in
           (NATIVE,UNGATED,SHUFFLED,ZERO,RESET,SCALAR,EXTERNAL)]
    if any(set(cores[0])!=set(c) or any(not torch.equal(cores[0][k],c[k])
                                      for k in cores[0]) for c in cores[1:]):
        raise RuntimeError("R103_NATIVE_CORE_FRESH_SAME_INIT_MISMATCH")
    adapter=lambda m:sum(p.numel() for n,p in m.named_parameters()
                         if p.requires_grad and not n.startswith("core."))
    a,b=adapter(models[NATIVE]),adapter(models[EXTERNAL])
    if a==0 or abs(a-b)>.1*max(a,b):
        raise RuntimeError("R103_NATIVE_EXTERNAL_TRAINABLE_CAPACITY_UNMATCHED")
    if any(adapter(models[name])!=a for name in
           (UNGATED,SHUFFLED,ZERO,RESET)):
        raise RuntimeError("R103_INTERNAL_NEGATIVE_CONTROL_CAPACITY_MISMATCH")
    return models


def unit_value(model,fixture,action=r85.CALIBRATE,shuffled=False):
    public=fixture["public"]
    rewards=public["support_rewards"].copy()
    if shuffled:
        rewards=rewards[public_shuffled_order(
            public["support_actions"],public["support_slots"],0)]
    x=torch.from_numpy(public["support_slots"][None])
    r=torch.from_numpy(rewards[None])
    q=torch.from_numpy(public["query0" if action==r85.CALIBRATE else "query2"][None])
    return model(x,r,q)
