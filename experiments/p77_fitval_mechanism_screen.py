"""P77 frozen FIT/VAL-only author synthetic DEV comparison, no OOD calls.

No module has an OOD-authorized selector. Data is read exclusively from
p77_factual_source.make_batch('fit'/'val'). Save complete negative outcomes.
"""
import argparse
import copy
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import torch
from p77_factual_source import make_batch, audit
from p77_write_policy import WritePolicy, MODEL_SEEDS, ETA_GRID

MODES = ('full5','cosine4','non_directional3','p70_four_param')
UPDATES=240
BATCH=32
CHECK=40
SAMPLE_SEED=977299


def mse_per_world(pred, target):
    return (pred-target).square().mean(-1)


def scores(model,val, chunk=32):
    model.eval()
    out=[]
    with torch.no_grad():
        for s in range(0,len(val['action']),chunk):
            t=min(s+chunk,len(val['action']))
            facts={k:val[k][s:t] for k in ('action','effect')}
            pred=model(facts,val['query_action'][s:t])
            out.append(mse_per_world(pred,val['observed_query_effect'][s:t]))
    v=torch.cat(out)
    mask=val['query_changed_audit']
    return {'total':float(v.mean()),'changed':float(v[mask].mean()),'stable':float(v[~mask].mean())}


def validate_seed(seed, fit, val,indices,start):
    rows={}
    for mode in MODES:
        model=WritePolicy(mode,seed)
        initial_sha256=hashlib.sha256(model.theta.detach().numpy().tobytes()).hexdigest()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
        best=float('inf');best_state=None; checkpoints=[]
        for step in range(UPDATES):
            ix=indices[step]
            facts={k:fit[k][ix] for k in ('action','effect')}
            q=fit['query_action'][ix]
            truth=fit['observed_query_effect'][ix]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(facts,q)
            loss=mse_per_world(prediction,truth).mean()
            if not bool(torch.isfinite(loss)):
                return {'status':'DEV_FAIL_NONFINITE','mode':mode,'step':step,'ood_generated':False}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if (step+1)%CHECK==0:
                validation=scores(model,val)
                checkpoints.append({'update':step+1,'mse':validation})
                if validation['total']<best:
                    best=validation['total'];best_state=copy.deepcopy(model.state_dict())
                print('PROGRESS',seed,mode,step+1,round(validation['total'],7),flush=True)
            if time.perf_counter()-start>3600:
                return {'status':'DEV_FAIL_CPU_BUDGET','mode':mode,'step':step,'ood_generated':False}
        model.load_state_dict(best_state)
        val_final=scores(model,val)
        with torch.no_grad():
            sample={k:val[k][:8] for k in ('action','effect')}
            full=model.encode(sample)
            first=model.encode({k:v[:,:48] for k,v in sample.items()})
            last=model.encode({k:v[:,48:] for k,v in sample.items()},carry=first)
            cold=bool(torch.equal(full,last))
        rows[mode]={'seed':seed,'mode':mode,'initial_sha256':initial_sha256,'val':val_final,
                    'checkpoints':checkpoints,'selected_theta':model.theta.detach().tolist(),
                    'cold_restore_bitwise':cold,'train_steps':UPDATES,'params':model.theta.numel()}
    return rows


def run(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if seed not in MODEL_SEEDS:
        raise ValueError('P77_UNREGISTERED_SEED')
    started=time.perf_counter()
    fit=make_batch('fit',512)
    val=make_batch('val',192)
    audits={n:audit(v) for n,v in [('fit',fit),('val',val)]}
    allowed=('target_quartile_pass','index_target_pass','changed_role_index_pass','query_duplicate_pass')
    if not all(all(item[k] for k in allowed) for item in audits.values()):
        return {'status':'DEV_FAIL_SOURCE_AUDIT','audits':audits,'ood_generated':False}
    zero=float(mse_per_world(torch.zeros_like(val['observed_query_effect']),val['observed_query_effect']).mean())
    fixed={}
    for eta in ETA_GRID:
        fixed[str(eta)]=scores(WritePolicy('fixed_eta',eta=eta),val)
    fixed['fixed_contradiction']=scores(WritePolicy('fixed_contradiction'),val)
    best_fixed=min(fixed,key=lambda name:fixed[name]['total'])
    generator=torch.Generator(device='cpu').manual_seed(SAMPLE_SEED)
    indices=torch.randint(len(fit['action']),(UPDATES,BATCH),generator=generator)
    sample_digest=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    trained=validate_seed(seed,fit,val,indices,started)
    if trained.get('status',''):
        return {'status':trained['status'],'detail':trained,'ood_generated':False}
    full=trained['full5']['val'];cos=trained['cosine4']['val'];nodir=trained['non_directional3']['val'];p70=trained['p70_four_param']['val'];b=fixed[best_fixed]
    decisions={
      'vs_calibrated_fixed': full['total']<=.85*b['total'] and full['changed']<=.80*b['changed'] and full['stable']<=1.05*b['stable'],
      'directional_needed': cos['changed']<=.90*nodir['changed'] and cos['stable']<=1.05*nodir['stable'],
      'contradiction_needed': full['changed']<=.95*cos['changed'] and full['stable']<=1.05*cos['stable'],
      'vs_p70_rule': full['total']<=.90*p70['total']
    }
    return {'schema':'ttt-hybrid.p77-frozen-dev-fitval.v1','seed':seed,
            'status':'DEV_VAL_PASS_ALL_MECHANISMS' if all(decisions.values()) else 'DEV_VAL_FAIL_AT_LEAST_ONE_MECHANISM',
            'science_pass':False,'ood_generated':False,'ood_scored':False,
            'source_audits':audits,'zero_val':zero,'fixed_validation':fixed,'best_fixed':best_fixed,
            'sample_indices_sha256':sample_digest,'trained':trained,'decisions':decisions,
            'wall_elapsed_seconds':time.perf_counter()-started,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'environment':{'python':platform.python_version(),'torch':torch.__version__},
            'original_ci':'UNVERIFIED','owner_science_first_use':'NOT_RUN','deployment':'NONE'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True,choices=MODEL_SEEDS)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('DONE',args.seed,result['status'],result.get('decisions'),flush=True)
