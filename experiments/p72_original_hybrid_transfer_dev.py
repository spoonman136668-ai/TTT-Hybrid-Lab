"""P72 preregistered ORIGINAL four-block HybridPilot gate transfer DEV.

FROZEN BEFORE SOURCE: experiments/p72_original_hybrid_transfer_frozen.json
Commit efe41d31030a5c61f9bd8ab1bd421c95793da3f8.
Isolated authored synthetic evaluation. No physical first-use, model promotion or
scientific PASS. Original P71 and all model source modules remain unchanged.
"""
from __future__ import annotations
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from model import HybridPilot, Configuration
from native_p71_gated_hybrid_stream import GatedStreamingHybridPilot

D = 16
N_TARGET = 8
FIT_STEPS = 120
BATCH = 8
VALID_EVERY = 20
SEEDS = {'fit': 720111, 'val': 720113, 'ood_dev': 720117, 'negative': 720119}
COUNTS = {'fit': 96, 'val': 48, 'ood_dev': 48, 'negative': 32}
MODEL_SEEDS = (720201,720203,720207)
POLICY_SEED = 1000203
MAX_SECONDS = 300


def digest_tensor(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def make_world(split, n=None):
    if split not in SEEDS: raise ValueError('P72_UNKNOWN_SOURCE')
    n = COUNTS[split] if n is None else n
    if not 1 <= n <= COUNTS[split]: raise ValueError('P72_BAD_N')
    g = torch.Generator().manual_seed(SEEDS[split])
    ood = split == 'ood_dev'
    reps, nuis = (2,16) if ood else (2,8)
    basis=torch.linalg.qr(torch.randn(n,D,D,generator=g)).Q
    targets=basis[:,:,:N_TARGET].transpose(1,2).contiguous()
    outcomes=torch.tanh(torch.randn(n,N_TARGET,D,generator=g)*(1.2 if ood else 1.0))
    action=targets.repeat(1,reps,1)
    effect=outcomes.repeat(1,reps,1)
    distractor_keys=F.normalize(torch.randn(n,nuis,D,generator=g),dim=-1)
    distractor_effects=torch.tanh(torch.randn(n,nuis,D,generator=g))
    action=torch.cat((action,distractor_keys),dim=1)
    effect=torch.cat((effect,distractor_effects),dim=1)
    role=torch.cat((torch.ones(n,N_TARGET*reps,dtype=torch.bool),torch.zeros(n,nuis,dtype=torch.bool)),dim=1)
    order=torch.rand(n,action.shape[1],generator=g).argsort(1)
    action=action.gather(1,order[:,:,None].expand_as(action))
    effect=effect.gather(1,order[:,:,None].expand_as(effect))
    role=role.gather(1,order)
    chosen=torch.randint(N_TARGET,(n,),generator=g)
    query_action=targets[torch.arange(n),chosen]
    label=outcomes[torch.arange(n),chosen]
    return {'facts':torch.cat((action,effect),dim=-1),
            'query':torch.cat((query_action,torch.zeros_like(query_action)),dim=-1)[:,None,:],
            'label':label, 'role_audit':role}


def source_hash(w):
    return hashlib.sha256(b''.join(w[k].contiguous().cpu().numpy().tobytes()
          for k in ('facts','query','label','role_audit'))).hexdigest()


def position_audit(w):
    roles=w['role_audit']
    n,length=roles.shape
    fractions=roles.float().reshape(n,4,length//4).mean((0,2))
    imbalance=float((fractions-roles.float().mean()).abs().max())
    train, held=roles[:n//2], roles[n//2:]
    # Balanced accuracy rejects a constant-negative classifier when OOD target
    # role prevalence changes. No roles are exposed to any model.
    guess=(train.float().mean(0)>=.5)[None].expand_as(held)
    tpr=(guess & held).sum()/held.sum().clamp_min(1)
    tnr=((~guess)&(~held)).sum()/(~held).sum().clamp_min(1)
    ba=float((tpr+tnr)/2)
    return {'quartile_departure':imbalance,'heldout_position_balanced_accuracy':ba,
            'pass':bool(imbalance<=.05 and ba<=.55)}


def sampled(w, idx):
    return {key:value[idx] for key,value in w.items()}


def cache_prediction(w):
    """Frequency-protected 8 x (16 key +16 consequence) = 256 float state.

    Per slot action-key magnitude is its observed hit count. No role, clock,
    future query, or counter outside the 256 floats is persisted.
    """
    fact=w['facts']
    key=F.normalize(fact[:,:,:D],dim=-1)
    effect=fact[:,:,D:]
    b,t,_=key.shape
    keys=key.new_zeros((b,N_TARGET,D)); vals=keys.clone()
    rows=torch.arange(b)
    for step in range(t):
        current=key[:,step]; observed=effect[:,step]
        norms=keys.norm(dim=-1)
        sims=(F.normalize(keys,dim=-1)*current[:,None]).sum(-1)
        best,ix=sims.max(-1)
        hit=best>=.98
        hits=norms[rows,ix]
        newkey=(hits+1).clamp(max=48)[:,None]*current
        newval=(vals[rows,ix]*hits[:,None]+observed)/(hits[:,None]+1)
        minnorm,weakest=norms.min(-1)
        permitted=hit|(minnorm<=1.01)
        at=torch.where(hit,ix,weakest)
        rr,cc=rows[permitted],at[permitted]
        keys[rr,cc]=torch.where(hit[permitted,None],newkey[permitted],current[permitted])
        vals[rr,cc]=torch.where(hit[permitted,None],newval[permitted],observed[permitted])
    q=F.normalize(w['query'][:,0,:D],dim=-1)
    match,at=(F.normalize(keys,dim=-1)*q[:,None]).sum(-1).max(-1)
    return vals[rows,at]*(match>=.98)[:,None]


def losses(pred,w):
    return (pred-w['label']).square().mean(-1)


def new_core(seed):
    torch.manual_seed(seed)
    return HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,
                            attention=True,ttt=True,depth=True))


def native_predict(model,w,mode):
    """Full original SSM, causal-attention, P71 factual TTT, depth and head.

    Rebind the adapter to exact current model weights on every call. This avoids
    reusing a stale factual state across optimizer updates.
    """
    facts=w['facts']
    adapter=GatedStreamingHybridPilot(model,seed=POLICY_SEED,mode=mode,max_confirmed_events=32)
    state=adapter.initial_state(facts[:,0,:])
    _,state=adapter.ingest_confirmed(facts,state,
                         torch.ones(facts.shape[:2],dtype=torch.bool,device=facts.device))
    return adapter.preview(w['query'],state)[:,0,:]


def train_arm(seed,mode,fit,val,indices,deadline):
    core=new_core(seed)
    init_hash=hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes()
                       for v in core.state_dict().values())).hexdigest()
    opt=torch.optim.AdamW(core.parameters(),lr=.002,weight_decay=.01)
    best=float('inf'); best_step=None; best_weights=None
    checkpoints=[]; started=time.perf_counter()
    for step in range(1,FIT_STEPS+1):
        if time.perf_counter()>deadline:
            return core,{'status':'TIME_BUDGET_EXCEEDED','seed':seed,'mode':mode,'steps':step-1}
        core.train()
        world=sampled(fit,indices[step-1])
        opt.zero_grad(set_to_none=True)
        prediction=native_predict(core,world,mode)
        loss=losses(prediction,world).mean()
        if not bool(torch.isfinite(loss)): return core,{'status':'NONFINITE_LOSS','steps':step,'seed':seed,'mode':mode}
        loss.backward()
        nn.utils.clip_grad_norm_(core.parameters(),1.)
        opt.step()
        if step%VALID_EVERY==0:
            core.eval()
            with torch.no_grad():
                val_mse=float(losses(native_predict(core,val,mode),val).mean())
            checkpoints.append({'step':step,'validation_mse':val_mse})
            if val_mse<best:
                best=val_mse;best_step=step;best_weights=copy.deepcopy(core.state_dict())
    core.load_state_dict(best_weights)
    return core,{'status':'DONE','mode':mode,'seed':seed,'init_sha256':init_hash,
          'parameters':sum(p.numel() for p in core.parameters()),
          'updates':FIT_STEPS,'best_step':best_step,'best_val_mse':best,
          'validation_checkpoints':checkpoints,'train_wall_seconds':time.perf_counter()-started}


def bootstrap_lower(candidate,reference,reps=2000,seed=720401):
    ix=torch.randint(len(candidate),(reps,len(candidate)),generator=torch.Generator().manual_seed(seed))
    return float(torch.quantile(1-candidate[ix].mean(-1)/reference[ix].mean(-1).clamp_min(1e-9),.025))


def study():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();deadline=started+MAX_SECONDS
    fit,val=make_world('fit'),make_world('val')
    r={'schema':'ttt-hybrid.p72-original-four-block-transfer-dev-result.v1',
       'classification':'AUTHORED_DEV_ONLY_NOT_INDEPENDENT_SCIENCE',
       'frozen_commit':'efe41d31030a5c61f9bd8ab1bd421c95793da3f8',
       'source_hashes':{'fit':source_hash(fit),'val':source_hash(val)},
       'source_audits':{'fit':position_audit(fit),'val':position_audit(val)},
       'OOD_outcome_scored':False,'original_hybrid_source_modified':False,
       'scientific_pass':False,'full_cross_platform':'NOT_RUN','gpu':'NOT_RUN',
       'original_model':'HybridPilot_all_four','gate_policy_seed':POLICY_SEED,
       'model_seeds':MODEL_SEEDS,'status':'NOT_STARTED'}
    if not all(x['pass'] for x in r['source_audits'].values()):
        r['status']='DEV_FAIL_SOURCE_LEAKAGE';return r
    with torch.no_grad():
        zero_mse=float(val['label'].square().mean())
        cache_mse=float(losses(cache_prediction(val),val).mean())
    r['val_controls']={'zero_mse':zero_mse,'frequency_cache_mse':cache_mse,
        'frequency_competent':cache_mse<=.75*zero_mse,
        'frequency_cache_state_floats':256}
    if not r['val_controls']['frequency_competent']:
        r['status']='DEV_FAIL_CLASSICAL_BASELINE_COMPETENCE';return r
    idx=torch.randint(len(fit['facts']),(FIT_STEPS,BATCH),generator=torch.Generator().manual_seed(720299))
    r['sample_indices_sha256']=digest_tensor(idx)
    final=[]
    for seed in MODEL_SEEDS:
        arm={}
        for mode in ('all_write','learned'):
            trained,meta=train_arm(seed,mode,fit,val,idx,deadline)
            arm[mode]=meta
            if meta['status']!='DONE':
                r['status']='DEV_FAIL_TRAINING_'+meta['status'];r['completed_training']=final+[arm];r['seconds']=time.perf_counter()-started;return r
            arm[mode+'_model']=trained
        arm['same_initial_weights']=arm['learned']['init_sha256']==arm['all_write']['init_sha256']
        final.append(arm)
    r['training_rows']=[{'seed':seed,'all_write':{k:v for k,v in arm['all_write'].items()},
         'learned':{k:v for k,v in arm['learned'].items()},
         'same_initial_weights':arm['same_initial_weights']}
         for seed,arm in zip(MODEL_SEEDS,final)]
    r['seed_identity_pass']=all(x['same_initial_weights'] for x in final)
    r['seed_independence_pass']=len({x['learned']['init_sha256'] for x in final})==3
    r['validation_competence_pass']=all(
       x[mode]['best_val_mse']<=.85*zero_mse and x[mode]['best_step']<FIT_STEPS
       for x in final for mode in ('all_write','learned'))
    if not (r['seed_identity_pass'] and r['seed_independence_pass'] and r['validation_competence_pass']):
        r['status']='DEV_FAIL_OR_UNQUALIFIED_TRAINED_VALIDATION';r['seconds']=time.perf_counter()-started;return r
    if time.perf_counter()>deadline:
        r['status']='DEV_FAIL_CPU_BUDGET';r['seconds']=time.perf_counter()-started;return r
    # OOD generated exactly once, after all training and VAL eligibility gates.
    ood=make_world('ood_dev');r['source_hashes']['ood_dev']=source_hash(ood)
    r['source_audits']['ood_dev']=position_audit(ood)
    if not r['source_audits']['ood_dev']['pass']:
        r['status']='DEV_FAIL_OOD_POSITION_LEAKAGE';r['seconds']=time.perf_counter()-started;return r
    with torch.no_grad():
        conventional=losses(cache_prediction(ood),ood)
        zeros=ood['label'].square().mean(-1)
        for seed,arm in zip(MODEL_SEEDS,final):
            allw=losses(native_predict(arm['all_write_model'],ood,'all_write'),ood)
            gate=losses(native_predict(arm['learned_model'],ood,'learned'),ood)
            eligible={'all_write_trained':allw,'frequency_cache':conventional,'zero':zeros}
            strongest=min(eligible,key=lambda k:float(eligible[k].mean()))
            row=next(x for x in r['training_rows'] if x['seed']==seed)
            row['ood']={'candidate_mse':float(gate.mean()),
                'reference':strongest,'reference_mse':float(eligible[strongest].mean()),
                'all_write_mse':float(allw.mean()),'cache_mse':float(conventional.mean()),
                'zero_mse':float(zeros.mean()),
                'lower_bootstrap_gain':bootstrap_lower(gate,eligible[strongest])}
            row['ood']['pass']=bool(row['ood']['candidate_mse']<=.9*row['ood']['reference_mse']
                   and row['ood']['lower_bootstrap_gain']>0)
    r['OOD_outcome_scored']=True
    # On evaluated OOD no further data/parameter fitting; wrong-credit remains
    # optional post-primary negative if the primary is already impossible.
    r['status']='DEV_PASS_PRIMARY_ONLY_NEGATIVES_PENDING' if all(x['ood']['pass'] for x in r['training_rows']) else 'DEV_FAIL_STRONGEST_BASELINE'
    r['seconds']=time.perf_counter()-started
    return r


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    result=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'OOD_outcome_scored':result['OOD_outcome_scored'],
          'seconds':result.get('seconds'),'scientific_pass':False}))
