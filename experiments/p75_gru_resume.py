"""P75 GRU FIT/VAL DEV only; deterministic 50-update restartable chunks.

No OOD selectors, science runner, new worker, external dispatch or scheduler.
The checkpoint is an exact source/sampler/recipe keyed local engineering cache.
Do not re-run a partially successful chunk until validating prior state first.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch
from p75_factual_source import make_batch,source_integrity
from p75_prequential_baselines import model_from_p75_recipe,frozen_sample_indices,losses
from p69_trainable_competitors import validation_mse,zero_mse

LOCKED = {
  'p75_factual_source.py':'c56485dc394260861ce48d7e4cb884ddf4b7a6ca',
  'p75_prequential_baselines.py':'197946f52c7a23fb4847628696dd71802bec4e30',
  'p69_trainable_competitors.py':'ad0d3712e4eff289ee1ab782820e4c3fc80a95aa',
}
WORLD_DIGESTS = {
  'fit':'aadab7833c4aa9c97d740809978adadf075b59abd14b231da7945bdf00a9212b',
  'val':'a152a8b49287a73615c53a9101ba8363a61c2a5fbf36f849387f9d18af44748a',
}


def blob(path):
    data=path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def checksum_parameters(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def run_chunk(seed,mode,outdir):
    if seed not in (975201,975203,975207) or mode not in ('final_only','matched_prequential'):
        raise ValueError('P75_UNREGISTERED_TRAINING')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parent
    if {n:blob(root/n) for n in LOCKED} != LOCKED:
        raise RuntimeError('P75_SOURCE_HASH_DRIFT')
    # Verify source integrity on the registered development split only.
    fit,val=make_batch('fit',512),make_batch('val',192)
    for s,b in (('fit',fit),('val',val)):
        a=source_integrity(b)
        if not all(a[k] for k in ('target_quartile_pass','index_target_pass','changed_role_index_pass','query_duplicate_pass')):
            raise RuntimeError('P75_SOURCE_INTEGRITY_FAIL_'+s)
    zero=zero_mse(val)
    ids=frozen_sample_indices(512,600)
    sample_hash=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    outdir=Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    stem=f'p75_gru_{seed}_{mode}'
    statefile=outdir/(stem+'.pt')
    model=model_from_p75_recipe('gru',seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    init=checksum_parameters(model)
    if statefile.exists():
        c=torch.load(statefile,map_location='cpu',weights_only=True)
        if (c.get('schema')!='p75-gru-frozen-resume-v1' or
            c.get('seed')!=seed or c.get('mode')!=mode or
            c.get('source_blobs')!=LOCKED or c.get('source_digests')!=WORLD_DIGESTS or
            c.get('sample_indices_sha256')!=sample_hash or c.get('init_sha256')!=init):
            raise RuntimeError('P75_UNQUALIFIED_OR_DRIFTED_CHECKPOINT')
        model.load_state_dict(c['model'])
        opt.load_state_dict(c['optimizer'])
    else:
        c={'schema':'p75-gru-frozen-resume-v1','seed':seed,'mode':mode,
           'step':0,'best':float('inf'),'best_update':0,'bad':0,
           'checkpoints':[],'best_state':None,'stopped_early':False,
           'init_sha256':init,'source_blobs':LOCKED.copy(),'source_digests':WORLD_DIGESTS.copy(),
           'sample_indices_sha256':sample_hash,'model':model.state_dict(),
           'optimizer':opt.state_dict(),'wall_seconds_cumulative':0.0}
    if c['stopped_early'] or c['step']==600:
        return {'status':'ALREADY_COMPLETE_NO_REPLAY','step':c['step'],
                'best_val_mse':c['best'],'seed':seed,'mode':mode,
                'competent':bool(c['best'] <= .75*zero)}
    start=c['step'];stop=min(start+50,600)
    if start%50!=0: raise RuntimeError('P75_NONBOUNDARY_CUSTODY')
    started=time.perf_counter()
    for ix in ids[start:stop]:
        facts={k:fit[k][ix] for k in ('action','effect')}
        q=fit['query_action'][ix]; y=fit['observed_query_effect'][ix]
        opt.zero_grad(set_to_none=True)
        objective=losses(model,facts,q,y,mode)['loss']
        if not bool(torch.isfinite(objective)):raise RuntimeError('P75_NONFINITE_FIT_LOSS')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
    mse=validation_mse(model,val)
    c['checkpoints'].append({'step':stop,'mse':mse})
    if mse<c['best']:
        c['best']=mse;c['best_update']=stop;c['bad']=0
        c['best_state']=copy.deepcopy(model.state_dict())
    else:c['bad']+=1
    last=[r['mse'] for r in c['checkpoints'][-3:]]
    c['converged']=len(last)==3 and max(last)-min(last)<=.01
    c['stopped_early']=bool(stop>=200 and c['bad']>=4 and c['converged'])
    c['step']=stop
    c['wall_seconds_cumulative']+=time.perf_counter()-started
    c['model']=model.state_dict()
    c['optimizer']=opt.state_dict()
    tmp=statefile.with_suffix('.part')
    torch.save(c,tmp)
    tmp.replace(statefile)
    summary={'schema':'p75-gru-frozen-resume-dev-receipt.v1','seed':seed,'mode':mode,
             'step':stop,'best_val_mse':c['best'],'best_update':c['best_update'],
             'last_val_mse':mse,'checks':c['checkpoints'],
             'stopped_early':c['stopped_early'],'complete':c['stopped_early'] or stop==600,
             'converged':c['converged'],'zero':zero,'competence_limit':zero*.75,
             'competent':bool((c['stopped_early'] or stop==600) and c['converged'] and c['best']<=.75*zero),
             'static_params':sum(p.numel() for p in model.parameters()),'memory_state_floats':256,
             'source_blobs':LOCKED,'source_digests':WORLD_DIGESTS,
             'sample_indices_sha256':sample_hash,'init_sha256':init,
             'cumulative_training_wall_seconds':c['wall_seconds_cumulative'],
             'scientific_pass':False,'p75_ood_generated':False,'p75_ood_scored':False,
             'full_project_regression':'UNVERIFIED','runner_or_owner_acted':False}
    (outdir/(stem+'.json')).write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=[975201,975203,975207],required=True)
    p.add_argument('--mode',choices=['final_only','matched_prequential'],required=True)
    p.add_argument('--outdir',type=Path,required=True)
    a=p.parse_args()
    result=run_chunk(a.seed,a.mode,a.outdir)
    print(json.dumps({k:result.get(k) for k in ('seed','mode','step','best_val_mse','last_val_mse','complete','competent','status')},sort_keys=True),flush=True)
