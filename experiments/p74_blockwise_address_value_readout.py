"""P74 prospectively frozen blockwise original-hybrid representational probe.

Author-generated FIT/VAL only, no physical source/science, no original parameter
training or changes, no OOD. Contract frozen at 51b88ceb7e50bf5b8951035cdb9a54517ea90418.

This deliberately tests frozen *randomly initialized* model representations;
it is NOT a diagnosis of a retained trained P73 checkpoint or layer causality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from model import HybridPilot, Configuration

D = 16
SOURCE_SEEDS = {'fit':740111, 'val':740113, 'negative':740119}
COUNTS = {'fit':192, 'val':128, 'negative':64}
CORE_SEEDS = (740201,740203,740207)
RIDGE_ALPHA = 1e-2


def make_world(split: str, n: int | None = None):
    if split not in SOURCE_SEEDS:
        raise ValueError('P74_UNKNOWN_SPLIT')
    n = COUNTS[split] if n is None else n
    if not isinstance(n,int) or n < 1 or n > COUNTS[split]:
        raise ValueError('P74_BAD_WORLD_COUNT')
    g = torch.Generator().manual_seed(SOURCE_SEEDS[split])
    basis = torch.linalg.qr(torch.randn(n,D,D,generator=g)).Q
    target_keys = basis[:,:,:8].transpose(1,2).contiguous()
    target_effects = torch.tanh(torch.randn(n,8,D,generator=g))
    actions = target_keys.repeat(1,2,1)
    effects = target_effects.repeat(1,2,1)
    nuisance_actions = F.normalize(torch.randn(n,8,D,generator=g),dim=-1)
    nuisance_effects = torch.tanh(torch.randn(n,8,D,generator=g))
    actions = torch.cat((actions,nuisance_actions),dim=1)
    effects = torch.cat((effects,nuisance_effects),dim=1)
    roles = torch.cat((torch.ones(n,16,dtype=torch.bool),torch.zeros(n,8,dtype=torch.bool)),1)
    order = torch.rand(n,24,generator=g).argsort(1)
    actions = actions.gather(1,order[:,:,None].expand_as(actions))
    effects = effects.gather(1,order[:,:,None].expand_as(effects))
    roles = roles.gather(1,order)
    chosen = torch.randint(8,(n,),generator=g)
    query_key = target_keys[torch.arange(n),chosen]
    true_effect = target_effects[torch.arange(n),chosen]
    return {'facts': torch.cat((actions,effects),dim=-1),
            'query':torch.cat((query_key,torch.zeros_like(query_key)),-1)[:,None,:],
            'label':true_effect, 'role_audit':roles}


def source_sha(w):
    return hashlib.sha256(b''.join(w[k].detach().contiguous().cpu().numpy().tobytes()
           for k in ('facts','query','label','role_audit'))).hexdigest()


def position_audit(w):
    role=w['role_audit'];n,t=role.shape
    if t != 24 or n<4: raise ValueError('P74_BAD_AUDIT_WORLD_SHAPE')
    quarters=role.float().reshape(n,4,6).mean((0,2))
    depart=float((quarters-role.float().mean()).abs().max())
    train=role[:n//2];hold=role[n//2:]
    guess=(train.float().mean(0)>=.5)[None].expand_as(hold)
    tpr=float(((guess & hold).sum())/hold.sum().clamp_min(1))
    tnr=float((((~guess)&(~hold)).sum())/(~hold).sum().clamp_min(1))
    ba=.5*(tpr+tnr)
    return {'max_quartile_departure':depart,'position_balanced_accuracy':ba,
            'pass':bool(depart<=.05 and ba<=.55)}


def factual(w):
    return w['facts']


def raw_address(w):
    a=F.normalize(w['facts'][:,:,:D],dim=-1)
    q=F.normalize(w['query'][:,0,:D],dim=-1)
    cosine=(a*q[:,None,:]).sum(-1)
    indices=cosine.argmax(-1)
    predicted=w['facts'][torch.arange(len(indices)),indices,D:]
    positives=(cosine>.99999)
    return {'pred':predicted,'indices':indices,'scores':cosine,'positives':positives,
            'hit':positives[torch.arange(len(indices)),indices]}


def mse(pred,target):
    return float((pred-target).square().mean())


def wrong_credit(w,seed=740399):
    g=torch.Generator().manual_seed(seed)
    out={k:v.clone() for k,v in w.items()}
    perm=torch.rand(len(out['facts']),24,generator=g).argsort(1)
    out['facts'][:,:,D:]=out['facts'][:,:,D:].gather(1,perm[:,:,None].expand(len(perm),24,D))
    return out


def core(seed):
    torch.manual_seed(seed)
    model=HybridPilot(32,16,Configuration(width=16,heads=2,ssm=True,
                    attention=True,ttt=True,depth=True)).eval()
    return model


def core_sha(model):
    buf=b''.join(x.detach().contiguous().cpu().numpy().tobytes()
                  for x in model.state_dict().values())
    return hashlib.sha256(buf).hexdigest()


def native_features(model,w):
    """Exactly original causal stem, SSM, attention, TTT and depth blocks.

    Observed historical values are stored before final query; the original
    TTT forward is read-before-write, so the query's hypothetical update
    cannot change its own output nor any earlier historical features.
    """
    if set(w) != {'facts','query','label','role_audit'}:
        raise ValueError('P74_BAD_WORLD_FIELDS')
    if w['facts'].ndim!=3 or w['facts'].shape[1:]!=(24,32):
        raise ValueError('P74_BAD_FACTS')
    if w['query'].shape!=(len(w['facts']),1,32) or not bool(torch.equal(w['query'][:,:,D:],torch.zeros_like(w['query'][:,:,D:]))):
        raise ValueError('P74_BAD_QUERY')
    with torch.no_grad():
        sequence=torch.cat((w['facts'],w['query']),1)
        z=model.attn(model.ssm(model.stem(sequence)))
        tn=model.ttt.norm(z)
        keys=F.normalize(model.ttt.key(tn[:,:-1]),dim=-1)
        query=F.normalize(model.ttt.query(tn[:,-1]),dim=-1)
        values=torch.tanh(model.ttt.value(tn[:,:-1]))
        projected=(keys*query[:,None,:]).sum(-1)
        idx=projected.argmax(-1)
        raw=raw_address(w)
        addresses=raw['positives'][torch.arange(len(idx)),idx]
        read_effect=w['facts'][torch.arange(len(idx)),idx,D:]
        after_ttt=model.ttt(z,update=True)
        final_feature=model.depth(after_ttt)[:,-1,:]
        reference=model(sequence)[:,-1,:]
        roundtrip=model.head(final_feature)
        torch.testing.assert_close(roundtrip,reference,atol=1e-6,rtol=1e-6)
    return {'keys':keys,'query_key':query,'values':values,
            'addresses':addresses,'selected_index':idx,'projected_cosine':projected,
            'effect_by_projected_address':read_effect,
            'final_feature':final_feature,'native_pred':reference}


def solve_ridge(x,y,alpha=RIDGE_ALPHA):
    """FIT-only float64 closed-form linear ridge; intercept excluded from L2."""
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0]:
        raise ValueError('P74_RIDGE_BAD_DATA')
    X=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),1)
    Y=y.double()
    reg=torch.eye(X.shape[1],dtype=torch.float64)*alpha
    reg[-1,-1]=0.
    return torch.linalg.solve(X.T@X+reg,X.T@Y)


def linear_readout(x,weights):
    X=torch.cat((x.double(),torch.ones((len(x),1),dtype=torch.float64)),1)
    return (X@weights).float()


def gradient_contribution(model,w):
    seq=torch.cat((w['facts'],w['query']),1)
    model.zero_grad(set_to_none=True)
    output=model(seq)[:,-1,:]
    loss=(output-w['label']).square().mean()
    loss.backward()
    mapping={
      'stem':model.stem.weight,
      'ssm_delta':model.ssm.delta.weight,
      'attention_qkv':model.attn.qkv.weight,
      'TTT_key':model.ttt.key.weight,
      'TTT_query':model.ttt.query.weight,
      'TTT_value':model.ttt.value.weight,
      'TTT_proj':model.ttt.proj.weight,
      'depth_router':model.depth.router.weight,
      'head':model.head.weight}
    result={k:(None if p.grad is None else float(p.grad.norm())) for k,p in mapping.items()}
    mandatory=('stem','ssm_delta','attention_qkv','TTT_key','TTT_query','TTT_value','TTT_proj','head')
    ok=all(result[k] is not None and torch.isfinite(torch.tensor(result[k])).item() and result[k]>0 for k in mandatory)
    return {'norms':result,'mandatory_nonzero_finite':bool(ok),'readout_loss':float(loss.detach())}


def study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t0=time.perf_counter()
    fit,val,negative=(make_world(k) for k in ('fit','val','negative'))
    audits={k:position_audit(w) for k,w in [('fit',fit),('val',val)]}
    rec={
      'schema':'ttt-hybrid.p74-native-address-value-head-frozen-dev-result.v1',
      'classification':'AUTHORED_DEV_RANDOM_CORE_FROZEN_REPRESENTATIONS_NOT_FULL_TRAINING_OR_SCIENCE',
      'preregister_commit':'51b88ceb7e50bf5b8951035cdb9a54517ea90418',
      'source_seeds':SOURCE_SEEDS, 'source_hashes':{k:source_sha(w) for k,w in [('fit',fit),('val',val),('negative',negative)]},
      'source_audits':audits,'scientific_pass':False,'original_model_unchanged':True,
      'OOD_generated':False,'OOD_scored':False,'cross_platform_full_regression':'NOT_RUN',
      'native_GPU':'NOT_RUN','FORGE':'UNCHANGED','status':'NOT_RUN'}
    if not all(a['pass'] for a in audits.values()):
        rec['status']='DEV_FAIL_SOURCE_POSITION_LEAKAGE'
        rec['seconds']=time.perf_counter()-t0
        return rec
    fit_raw,val_raw=raw_address(fit),raw_address(val)
    raw_mse=mse(val_raw['pred'],val['label'])
    neg_clean=raw_address(negative)
    neg_wrong=raw_address(wrong_credit(negative))
    credit_mse=mse(neg_wrong['pred'],negative['label'])
    clean_mse=mse(neg_clean['pred'],negative['label'])
    zero_mse=mse(torch.zeros_like(val['label']),val['label'])
    rec['raw_control']={
      'val_mse':raw_mse, 'zero_val_mse':zero_mse, 'val_hit_at_1':float(val_raw['hit'].float().mean()),
      'negative_clean_mse':clean_mse, 'negative_wrong_credit_mse':credit_mse,
      'wrong_credit_10x':bool(credit_mse >= 10*max(clean_mse,1e-5)),
      'raw_competence':bool(raw_mse<=.001 and val_raw['hit'].float().mean()>=.99)}
    if not (rec['raw_control']['raw_competence'] and rec['raw_control']['wrong_credit_10x']):
        rec['status']='DEV_FAIL_RAW_SOURCE_OR_NEGATIVE_CONTROL'
        rec['seconds']=time.perf_counter()-t0
        return rec
    fitted_oracle=solve_ridge(fit_raw['pred'],fit['label'])
    oracle_head_mse=mse(linear_readout(val_raw['pred'],fitted_oracle),val['label'])
    rec['oracle_linear_head_mse']=oracle_head_mse
    if oracle_head_mse>.01:
        rec['status']='DEV_FAIL_ORACLE_LINEAR_READOUT'
        rec['seconds']=time.perf_counter()-t0
        return rec
    rows=[]
    for seed in CORE_SEEDS:
        model=core(seed)
        original_weights=core_sha(model)
        f=native_features(model,fit)
        v=native_features(model,val)
        factual_fit=fit['facts'][:,:,D:]
        factual_val=val['facts'][:,:,D:]
        val_native_address_mse=mse(v['effect_by_projected_address'],val['label'])
        val_native_hit=float(v['addresses'].float().mean())
        decoder=solve_ridge(f['values'].reshape(-1,D),factual_fit.reshape(-1,D))
        decoded_values=linear_readout(v['values'].reshape(-1,D),decoder).reshape(-1,24,D)
        val_value_mse=mse(decoded_values,factual_val)
        raw_selected=val_raw['indices']
        reconstructed=decoded_values[torch.arange(len(raw_selected)),raw_selected]
        val_reconstruction_via_raw_mse=mse(reconstructed,val['label'])
        final_decoder=solve_ridge(f['final_feature'],fit['label'])
        val_final_ridge_mse=mse(linear_readout(v['final_feature'],final_decoder),val['label'])
        grad=gradient_contribution(model,{k:v[:8] for k,v in fit.items()})
        rows.append({
          'seed':seed,'original_weights_sha256':original_weights,
          'val_projected_key_query_hit_at_1':val_native_hit,
          'val_native_address_true_effect_mse':val_native_address_mse,
          'val_native_value_ridge_factual_effect_mse':val_value_mse,
          'val_raw_address_native_decoded_value_mse':val_reconstruction_via_raw_mse,
          'val_original_native_prediction_mse':mse(v['native_pred'],val['label']),
          'val_head_only_fit_ridge_mse':val_final_ridge_mse,
          'addresses_competent':bool(val_native_hit>=.70 and val_native_address_mse<=.60*zero_mse),
          'value_linearly_decodable':bool(val_value_mse<=.60*zero_mse),
          'native_final_linearly_decodable':bool(val_final_ridge_mse<=.85*zero_mse),
          'gradient':grad,
          'model_weights_preserved':original_weights==core_sha(model)})
    rec['model_rows']=rows
    rec['distinct_model_weight_sha256']=len({r['original_weights_sha256'] for r in rows})==3
    rec['all_gradient_controls']=all(r['gradient']['mandatory_nonzero_finite'] for r in rows)
    rec['all_original_weights_unchanged']=all(r['model_weights_preserved'] for r in rows)
    if not rec['distinct_model_weight_sha256'] or not rec['all_gradient_controls'] or not rec['all_original_weights_unchanged']:
        rec['status']='DEV_FAIL_MODEL_PROVENANCE_OR_GRADIENT'
    else:
        rec['status']='DEV_PROBE_COMPLETE_WITH_COMPARTMENT_METRICS'
    rec['seconds']=time.perf_counter()-t0
    rec['whole_process_compute_matched']=False
    rec['native_weights_trained_for_this_task']=False
    return rec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args()
    result=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'science_pass':False,'seconds':result.get('seconds'),
                      'address_competence':[(r['seed'],r['addresses_competent']) for r in result.get('model_rows',[])]}),flush=True)

if __name__=='__main__': main()
