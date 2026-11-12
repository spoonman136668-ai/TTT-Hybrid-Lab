"""P78-MR tri-split FIT/CAL then single author-DEV VAL, never OOD.

Must be run only for engineering-author-controlled DEV; no independent science,
production, Forge dispatch, runner management or backend side effects.
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

from p78mr_adaptive_controls import (
    MODEL_SEEDS, configurations, fixed_outputs, new_learned,
    scored_errors, select_calibrated,
)
from p78mr_factual_source import make_batch, audit

STEPS=240
BATCH_SIZE=32
CHECK_EVERY=40
SAMPLER_SEED=978299


def _digest(data):
    h=hashlib.sha256()
    for k in ('action','effect','query_action','observed_query_effect'):
        h.update(k.encode()+b'\0')
        h.update(data[k].contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def _validate_source(data):
    report=audit(data)
    for key in ('target_quartile_pass','index_target_pass',
                'changed_role_index_pass','query_duplicate_pass'):
        if not report[key]:
            raise ValueError('P78MR_SOURCE_AUDIT_FAIL_'+key)
    return report


def _summary(per_world, changed):
    return {'total':float(per_world.mean()),
            'changed':float(per_world[changed].mean()),
            'stable':float(per_world[~changed].mean())}


def _model_scores(model,data):
    with torch.no_grad():
        rows=[]
        for i in range(0,len(data['action']),32):
            j=min(i+32,len(data['action']))
            facts={k:data[k][i:j] for k in ('action','effect')}
            pred=model(facts,data['query_action'][i:j])
            rows.append((pred-data['observed_query_effect'][i:j]).square().mean(-1))
    return _summary(torch.cat(rows),data['query_changed_audit'])


def _one_model(mode,seed,fit,cal,sampler):
    model=new_learned(mode,seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    initial=hashlib.sha256(model.theta.detach().numpy().tobytes()).hexdigest()
    records=[]
    best_error=float('inf');best_state=None
    for step in range(STEPS):
        ix=sampler[step]
        facts={k:fit[k][ix] for k in ('action','effect')}
        pred=model(facts,fit['query_action'][ix])
        loss=(pred-fit['observed_query_effect'][ix]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P78MR_NONFINITE_TRAIN')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%CHECK_EVERY==0:
            cal_row=_model_scores(model,cal)
            records.append({'step':step+1,'cal':cal_row})
            if cal_row['total'] < best_error:
                best_error=cal_row['total']
                best_state=copy.deepcopy(model.state_dict())
    if best_state is None or len(records)!=6:
        raise ValueError('P78MR_TRAIN_INCOMPLETE')
    model.load_state_dict(best_state)
    sample={k:cal[k][:4] for k in ('action','effect')}
    with torch.no_grad():
        full=model.encode(sample)
        first=model.encode({k:v[:,:48] for k,v in sample.items()})
        carry=first.clone()
        resumed=model.encode({k:v[:,48:] for k,v in sample.items()},carry=carry)
        cold=bool(torch.equal(full,resumed) and torch.equal(first,carry))
    if not cold:
        raise ValueError('P78MR_COLD_RESTORE_FAIL')
    return model, {'mode':mode,'seed':seed,'steps':STEPS,'static_params':model.theta.numel(),
                   'initial_theta_sha256':initial,'selected_theta':model.theta.detach().tolist(),
                   'cal_checkpoints':records,'selected_cal':_model_scores(model,cal),
                   'cold_restoration':cold,'persistent_state_floats':256}


def _fixed_score(data,configs):
    with torch.no_grad():
        p,state=fixed_outputs({k:data[k] for k in ('action','effect')},
                              data['query_action'],configs=configs)
        result=scored_errors(p,data['observed_query_effect'],data['query_changed_audit'])
        if state.shape[-2:]!=(16,16):
            raise ValueError('P78MR_STATE_BUDGET')
    return result


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    clock=time.perf_counter()
    # Only FIT and CAL exist in model selection below. VAL first scored after
    # every single model has finished FIT optimization and CAL selection.
    fit=make_batch('fit',512)
    cal=make_batch('cal',192)
    source_audits={'fit':_validate_source(fit),'cal':_validate_source(cal)}
    source_digests={'fit':_digest(fit),'cal':_digest(cal)}
    configs=configurations()
    fixed_cal=_fixed_score(cal,configs)
    chosen_idx,chosen=select_calibrated(configs,fixed_cal)
    sample=torch.Generator(device='cpu').manual_seed(SAMPLER_SEED)
    indices=torch.randint(len(fit['action']),(STEPS,BATCH_SIZE),generator=sample)
    sample_hash=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    trained={};models={}
    for seed in MODEL_SEEDS:
        trained[str(seed)]={}
        models[str(seed)]={}
        for mode in ('non_directional3','cosine4'):
            model,row=_one_model(mode,seed,fit,cal,indices)
            trained[str(seed)][mode]=row
            models[str(seed)][mode]=model
            print('FIT_CAL_COMPLETE',seed,mode,row['selected_cal']['total'],flush=True)
    # All selection is now irreversibly finished; one author-controlled VAL.
    val=make_batch('val',192)
    val_audit=_validate_source(val)
    source_audits['val']=val_audit
    source_digests['val']=_digest(val)
    fixed_val=_fixed_score(val,(chosen,))
    fixed={k:v[0] for k,v in fixed_val.items()}
    decisions={}
    for seed in MODEL_SEEDS:
        row=trained[str(seed)]
        for mode in ('non_directional3','cosine4'):
            row[mode]['val_once']=_model_scores(models[str(seed)][mode],val)
        a=row['non_directional3']['val_once']
        b=row['cosine4']['val_once']
        decisions[str(seed)]={
            'simple_beats_adaptive_fixed':a['total'] <= .9*fixed['total'] and
                a['changed'] <= .9*fixed['changed'] and a['stable']<=1.05*fixed['stable'],
            'direction_adds_10pct_gain':b['changed'] <= .9*a['changed'] and
                b['stable']<=1.05*a['stable']
        }
    primary=all(d['simple_beats_adaptive_fixed'] for d in decisions.values())
    directional=all(d['direction_adds_10pct_gain'] for d in decisions.values())
    return {'schema':'ttt-hybrid.p78mr-fitcal-one-val-dev.v1',
       'status':'DEV_VAL_PASS_SIMPLE_VS_STRONG_FIXED' if primary else 'DEV_VAL_FAIL_SIMPLE_VS_STRONG_FIXED',
       'direction_necessity':'DEV_VAL_PASS' if directional else 'DEV_VAL_FAIL',
       'audited_source_sha256':source_digests,'source_audits':source_audits,
       'sample_index_sha256':sample_hash,
       'frozen_config_count':len(configs),
       'cal_selected_index':chosen_idx,'cal_selected_config':chosen,
       'cal_selected_fixed_total':fixed_cal['total'][chosen_idx],
       'cal_selected_fixed_changed':fixed_cal['changed'][chosen_idx],
       'cal_selected_fixed_stable':fixed_cal['stable'][chosen_idx],
       'val_selected_fixed':fixed,
       'training':trained,'decisions':decisions,
       'whole_process_wall_seconds':time.perf_counter()-clock,
       'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
       'toolchain':{'python':platform.python_version(),'torch':torch.__version__},
       'p78_ood_generated':False,'p78_ood_scored':False,
       'independent_science':'NOT_RUN',
       'original_windows_linux_full_regression':'UNVERIFIED',
       'deployment':'NONE',
       'science_pass':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args()
    results=run()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(results,sort_keys=True,indent=2)+'\n')
    print('COMPLETE',results['status'],results['direction_necessity'],flush=True)
