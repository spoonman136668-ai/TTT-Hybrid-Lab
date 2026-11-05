"""Frozen development screen: official TTT-Linear versus no-update, attention, GRU.

Training/evaluation data and budgets are locked in g6_protocol.py.
A pass means *execution only*, not improved intelligence or scientific PASS.
"""
from __future__ import annotations
import argparse, hashlib, inspect, json, os, platform, random, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.g6_protocol import (PROTOCOL, SEEDS, TRAIN_STEPS, ARMS, WIDTH,
                                 VAL_BATCHES, make_batch, protocol_sha256)


def source_admission(ttt_root: Path):
    from authentic_preflight import source_check, module_location_check
    pins=json.loads((ROOT/'source_pins.json').read_text(encoding='utf-8'))
    pin=pins['sources']['ttt_linear']
    checks={'source':source_check(ttt_root,pin),
            'module':module_location_check('ttt',ttt_root)}
    if not checks['source']['pass'] or not checks['module']['pass']:
        raise RuntimeError('EXACT_TTT_SOURCE_ADMISSION_FAILED:'+json.dumps(checks))
    from ttt import TTTLinear, TTTConfig
    expected=(ttt_root/'ttt.py').resolve()
    if any(Path(inspect.getfile(c)).resolve()!=expected for c in (TTTLinear,TTTConfig)):
        raise RuntimeError('OFFICIAL_TTT_CLASS_ORIGIN_MISMATCH')
    return TTTLinear, TTTConfig, checks


def model_class(torch, TTTLinear, TTTConfig):
    nn=torch.nn
    class ExperimentModel(nn.Module):
        def __init__(self,kind):
            super().__init__()
            self.kind=kind
            self.encoder=nn.Linear(3,WIDTH)
            self.norm_in=nn.LayerNorm(WIDTH)
            if kind in ('ttt_live','ttt_off'):
                cfg=TTTConfig(vocab_size=128,hidden_size=WIDTH,intermediate_size=128,
                              num_hidden_layers=1,num_attention_heads=4,mini_batch_size=8,
                              ttt_layer_type='linear',ttt_base_lr=1.0 if kind=='ttt_live' else 0.0,
                              use_cache=False,pre_conv=False,max_position_embeddings=256)
                self.core=TTTLinear(cfg,layer_idx=0)
                self.head=nn.Linear(WIDTH,1)
            elif kind=='causal_attention':
                self.core=nn.MultiheadAttention(WIDTH,4,batch_first=True,dropout=0.0)
                self.head=nn.Linear(WIDTH,1)
            elif kind=='gru':
                self.core=nn.GRU(WIDTH,48,batch_first=True)
                self.head=nn.Linear(48,1)
            else:
                raise ValueError('UNREGISTERED_ARM')
        def forward(self,x):
            h=self.encoder(x)
            h0=self.norm_in(h)
            if self.kind.startswith('ttt_'):
                t=x.shape[1]
                pos=torch.arange(t,device=x.device).unsqueeze(0).expand(x.shape[0],-1)
                h=h+self.core(h0,position_ids=pos,cache_params=None)
            elif self.kind=='causal_attention':
                t=x.shape[1]
                mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(diagonal=1)
                y,_=self.core(h0,h0,h0,attn_mask=mask,need_weights=False)
                h=h+y
            else:
                h,_=self.core(h0)
            return self.head(h[:,-1,:]).squeeze(-1)
    return ExperimentModel


def digest_weights(model):
    h=hashlib.sha256()
    for name,p in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_arm(torch, Model, kind: str, seed: int):
    torch.manual_seed(20261105+seed)
    model=Model(kind)
    initial_sha=digest_weights(model)
    params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=0.002,weight_decay=0.0)
    model.train()
    started=time.perf_counter()
    last_loss=None
    for step in range(TRAIN_STEPS):
        x,target=make_batch(seed,step,phase='train')
        opt.zero_grad(set_to_none=True)
        out=model(x)
        loss=torch.nn.functional.mse_loss(out,target)
        if not torch.isfinite(loss).item():
            raise RuntimeError('NONFINITE_TRAINING_LOSS:'+kind+':'+str(seed)+':'+str(step))
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError('NONFINITE_TRAINING_GRADIENT:'+kind)
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        opt.step()
        last_loss=float(loss.detach())
    train_s=time.perf_counter()-started
    model.eval()
    started=time.perf_counter()
    errors, scrambled_errors, ablated_errors=[],[],[]
    with torch.no_grad():
        for idx in range(VAL_BATCHES):
            x,y=make_batch(seed,idx,phase='validation')
            x_scr,y_scr=make_batch(seed,idx,phase='validation',scramble=True)
            if not torch.equal(y,y_scr):
                raise RuntimeError('SCRAMBLE_CHANGED_LABELS')
            pred=model(x)
            pred_s=model(x_scr)
            if not (torch.isfinite(pred).all().item() and torch.isfinite(pred_s).all().item()):
                raise RuntimeError('NONFINITE_VALIDATION_OUTPUT:'+kind)
            errors.extend(((pred-y)**2).tolist())
            scrambled_errors.extend(((pred_s-y)**2).tolist())
            if kind=='ttt_live':
                previous=model.core.config.ttt_base_lr
                try:
                    model.core.config.ttt_base_lr=0.0
                    ablated=model(x)
                finally:
                    model.core.config.ttt_base_lr=previous
                if not torch.isfinite(ablated).all().item():
                    raise RuntimeError('NONFINITE_LR_ABLATION')
                ablated_errors.extend(((ablated-y)**2).tolist())
    eval_s=time.perf_counter()-started
    return {'arm':kind,'seed':seed,'params':params,'initial_weights_sha256':initial_sha,
            'train_steps':TRAIN_STEPS,'train_batch':16,'training_seconds':round(train_s,3),
            'validation_seconds':round(eval_s,3),'last_train_mse':last_loss,
            'mse':sum(errors)/len(errors),
            'scramble_mse':sum(scrambled_errors)/len(scrambled_errors),
            'errors':errors,'scramble_errors':scrambled_errors,
            'trained_live_weights_update_disabled_mse':sum(ablated_errors)/len(ablated_errors)
                if ablated_errors else None}


def decide(records):
    out={}
    thresholds=PROTOCOL['positive_thresholds']
    for arm in ARMS:
        x=[r for r in records if r['arm']==arm]
        out[arm]={'mean_mse':sum(r['mse'] for r in x)/len(x),
                  'mean_train_seconds':sum(r['training_seconds'] for r in x)/len(x),
                  'params':x[0]['params'],
                  'mean_scramble_mse':sum(r['scramble_mse'] for r in x)/len(x)}
    live=out['ttt_live']; off=out['ttt_off']
    conv=min((out[k] for k in ('causal_attention','gru')), key=lambda d:d['mean_mse'])
    best_kind=min(('causal_attention','gru'),key=lambda k:out[k]['mean_mse'])
    ratios={
        'live_vs_no_update_mse':live['mean_mse']/max(1e-12,off['mean_mse']),
        'live_vs_best_conventional_mse':live['mean_mse']/max(1e-12,conv['mean_mse']),
        'live_vs_best_conventional_time':live['mean_train_seconds']/max(1e-12,conv['mean_train_seconds']),
        'live_vs_best_conventional_params':live['params']/max(1,conv['params']),
        'live_scrambled_vs_clean_mse':live['mean_scramble_mse']/max(1e-12,live['mean_mse']),
    }
    per_seed=all(
        next(r for r in records if r['arm']=='ttt_live' and r['seed']==seed)['mse']
        <=thresholds['individual_seed_tt_vs_no_update_mse_ratio_max']*
        next(r for r in records if r['arm']=='ttt_off' and r['seed']==seed)['mse']
        for seed in SEEDS)
    hit=(ratios['live_vs_no_update_mse']<=thresholds['ttt_vs_no_update_mse_ratio_max']
         and ratios['live_vs_best_conventional_mse']<=thresholds['ttt_vs_best_conventional_mse_ratio_max']
         and ratios['live_vs_best_conventional_time']<=thresholds['ttt_vs_best_conventional_train_time_ratio_max']
         and ratios['live_vs_best_conventional_params']<=thresholds['ttt_vs_best_conventional_params_ratio_max']
         and ratios['live_scrambled_vs_clean_mse']>=thresholds['ttt_support_scramble_mse_ratio_min']
         and per_seed)
    # Three-cluster bootstrap is exploratory; it is NOT independent qualification.
    cluster=[]
    for s in SEEDS:
        a=next(x for x in records if x['arm']=='ttt_live' and x['seed']==s)
        b=next(x for x in records if x['arm']==best_kind and x['seed']==s)
        cluster.append((sum(a['errors'])/len(a['errors']),sum(b['errors'])/len(b['errors'])))
    rng=random.Random(6102026)
    samples=[]
    for _ in range(1000):
        selected=[cluster[rng.randrange(len(cluster))] for _ in cluster]
        samples.append((sum(q[0] for q in selected)+1e-12)/(sum(q[1] for q in selected)+1e-12))
    samples.sort()
    return {'arms':out,'ratios':ratios,'best_conventional':best_kind,
            'per_seed_ttt_better_than_no_update':per_seed,
            'paired_seed_cluster_bootstrap_95pct_ci_ratio':[samples[25],samples[975]],
            'bootstrap_warning':'3 seed clusters only; confidence bounds exploratory',
            'development_decision':'HYPOTHESIS_MET_DEVELOPMENT_ONLY' if hit else 'NO_GO_DEVELOPMENT_SCREEN',
            'scientific_qualification':'UNVERIFIED' }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ttt-root',required=True,type=Path)
    p.add_argument('--receipt',type=Path,default=ROOT/'local_runs'/'g6_development.json')
    args=p.parse_args()
    root=args.ttt_root.resolve();receipt=args.receipt.resolve();receipt.parent.mkdir(parents=True,exist_ok=True)
    result={'schema':'ttt-hybrid.g6-development-receipt.v1',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'protocol':PROTOCOL,'protocol_sha256':protocol_sha256(),
            'source_pins_sha256':hashlib.sha256((ROOT/'source_pins.json').read_bytes()).hexdigest(),
            'scientific_qualification':'UNVERIFIED','sealed_test':'NOT_RUN',
            'gpu_integration':'NOT_RUN','status':'FAIL_UNVERIFIED'}
    start=time.perf_counter()
    try:
        rev=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,timeout=15)
        observed=rev.stdout.strip() if rev.returncode==0 else None
        expected=os.getenv('EXPECTED_CANDIDATE_SHA')
        result['revision']=observed;result['expected_revision']=expected
        if not observed or (expected and observed!=expected) or (not expected and os.getenv('GITHUB_ACTIONS')=='true'):
            raise RuntimeError('CANDIDATE_IDENTITY_MISMATCH')
        sys.path.insert(0,str(root))
        TTTLinear,TTTConfig,check=source_admission(root)
        result['source_admission']=check
        import torch
        result['runtime']={'platform':platform.platform(),'python':platform.python_version(),
                           'torch':torch.__version__,'torch_cuda':torch.version.cuda,
                           'cpu_threads':2,'github_run_id':os.getenv('GITHUB_RUN_ID'),
                           'ttt_class_file':str(Path(inspect.getfile(TTTLinear)).resolve())}
        torch.set_num_threads(2)
        Model=model_class(torch,TTTLinear,TTTConfig)
        records=[]
        for seed in SEEDS:
            for arm in ARMS:
                records.append(run_arm(torch,Model,arm,seed))
        # Verify paired native TTT models have identical initial weights.
        for seed in SEEDS:
            a=next(r for r in records if r['arm']=='ttt_live' and r['seed']==seed)
            b=next(r for r in records if r['arm']=='ttt_off' and r['seed']==seed)
            if a['initial_weights_sha256']!=b['initial_weights_sha256']:
                raise RuntimeError('PAIRED_TTT_INITIAL_WEIGHTS_DIFFER')
        result['measurements']=[{k:v for k,v in r.items() if k not in ('errors','scramble_errors')}
                                for r in records]
        result['summary']=decide(records)
        result['status']='PASS_ENGINEERING_DEVELOPMENT_EXECUTION_ONLY'
        rc=0
    except Exception as e:
        result['error_type']=type(e).__name__
        result['error_message']=str(e)[:800]
        result['traceback_tail']=traceback.format_exc().splitlines()[-12:]
        rc=1
    finally:
        result['elapsed_seconds']=round(time.perf_counter()-start,3)
        receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'status':result['status'],'revision':result.get('revision'),
                          'decision':result.get('summary',{}).get('development_decision'),
                          'metrics':result.get('summary',{}).get('ratios'),
                          'error':result.get('error_message'),'receipt':str(receipt)}))
    return rc

if __name__=='__main__':raise SystemExit(main())
