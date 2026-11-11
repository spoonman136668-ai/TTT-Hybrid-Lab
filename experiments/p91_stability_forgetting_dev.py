"""P91 prospectively frozen stability-protected causal forgetting; authored synthetic DEV."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from experiments import p86_portable_causal_preflight as p86
from experiments import p90_contradiction_update_dev as p90

FREEZE='96fe23932f87b04dd689bd147277a4cea8b250f3'
FAMILIES={'A':{'FIT':910111,'VAL':910113},'B':{'FIT':910121,'VAL':910123}}
SEEDS=(910201,910203,910207)
ARMS=('constant','observation','contradiction','protected')
STEPS,BATCH,CHECK=80,16,10
BLOBS={'model.py':'5d9ea57b827aa1796548447868fa5ee14c251b49',
 'experiments/p86_portable_causal_preflight.py':'008492665ee9090db5791963f2e2336ea283ee62',
 'experiments/p90_contradiction_update_dev.py':'c1a3866f02201bbf87106c073a57c07bd695bcc5'}

def source(seed,n):
    expected={v:128 if k=='FIT' else 64 for fam in FAMILIES.values() for k,v in fam.items()}
    if seed not in expected or n!=expected[seed]:
        raise ValueError('P91_SOURCE_FENCE')
    changed=set(sorted(range(n),key=lambda w:p86.digest_int(seed,w,'regime',0,0,2**32))[:n//2])
    A=[];E=[];C=[]
    for w in range(n):
        u0,v0=p86.basis(seed,w,'u0'),p86.basis(seed,w,'v0')
        u1,v1=p86.basis(seed,w,'u1'),p86.basis(seed,w,'v1')
        aa=[];ee=[]
        for t in range(49):
            a=[p86.digest_int(seed,w,'action',t,j,193)-96 for j in range(16)]
            u,v=(u0,v0) if (t<24 or w not in changed) else (u1,v1)
            aa.append([z/128 for z in a]);ee.append(p86.causal_effect(a,u,v))
        A.append(aa);E.append(ee);C.append(w in changed)
    return {'actions':torch.tensor(A,dtype=torch.float32),
            'effects':torch.tensor(E,dtype=torch.float32),
            'changed_world':torch.tensor(C,dtype=torch.bool)}

def signature(w):return p90.world_sha(w)

def init(seed,arm):
    if arm not in ARMS:raise ValueError('P91_UNKNOWN_ARM')
    g=torch.Generator().manual_seed(seed)
    eta=float(torch.randn((),generator=g)*.02)
    forget=-2.9444389791664403+float(torch.randn((),generator=g)*.02)
    return nn.Parameter(torch.tensor([eta,forget]+([] if arm=='constant' else [0.,0.]),dtype=torch.float32))

def predict(w,arm,theta=None,start=0,stop=49,carry=None):
    if arm not in (*ARMS,'fixed'):raise ValueError('P91_UNKNOWN_ARM')
    if arm!='fixed' and (theta is None or theta.numel()!=(2 if arm=='constant' else 4)):
        raise ValueError('P91_BAD_GLOBAL_PARAMS')
    a=F.normalize(w['actions'],dim=-1); effects=w['effects']
    W=torch.zeros(len(a),16,16,dtype=a.dtype) if carry is None else carry
    out=[]
    for t in range(start,stop):
        k=a[:,t];v=torch.bmm(W,k[:,:,None]).squeeze(-1);out.append(v)
        if t==48:continue
        r=effects[:,t]-v
        if arm=='fixed':eta=1.;decay=.99
        elif arm=='constant':
            eta=2*torch.sigmoid(theta[0]);decay=1-.2*torch.sigmoid(theta[1])
        else:
            feature=r if arm in ('contradiction','protected') else effects[:,t]
            f=torch.sqrt(feature.square().mean(-1)+1e-9)
            f=f/(f+.1)
            eta=2*torch.sigmoid(theta[0]+theta[2]*f)
            forget=.2*torch.sigmoid(theta[1]+theta[3]*f)
            if arm=='protected':forget=forget*f*f
            decay=1-forget
            eta=eta[:,None,None];decay=decay[:,None,None]
        W=decay*W+eta*r[:,:,None]*k[:,None,:]
    return torch.stack(out,1),W

def metrics(y,w):
    x=(y-w['effects']).square().mean(-1); c=w['changed_world']
    return {'final':float(x[:,-1].mean()),'stable':float(x[~c,-1].mean()),
            'changed':float(x[c,-1].mean()),'late':float(x[:,32:].mean()),
            'stable_late':float(x[~c,32:].mean()),
            'changed_early':float(x[c,24:32].mean()),
            'changed_late':float(x[c,40:49].mean())}

def train(seed,arm,fit,vals,indices,deadline):
    theta=init(seed,arm);initial=hashlib.sha256(theta.detach().numpy().tobytes()).hexdigest()
    opt=torch.optim.AdamW([theta],lr=.05,weight_decay=0)
    best=float('inf');best_step=0;best_theta=None;curve=[];started=time.perf_counter()
    for step in range(1,STEPS+1):
        if time.perf_counter()>deadline:raise TimeoutError('P91_CPU_BUDGET')
        idx=indices[step-1];batch={k:v[idx] for k,v in fit.items()}
        opt.zero_grad(set_to_none=True)
        y,_=predict(batch,arm,theta)
        loss=(y[:,32:]-batch['effects'][:,32:]).square().mean()
        if not bool(torch.isfinite(loss)):raise ValueError('P91_NONFINITE_LOSS')
        loss.backward();nn.utils.clip_grad_norm_([theta],1.)
        opt.step()
        if step%CHECK==0:
            with torch.no_grad():each={name:metrics(predict(w,arm,theta)[0],w) for name,w in vals.items()}
            aggregate=sum(r['final'] for r in each.values())/len(each)
            curve.append({'step':step,'mean_final':aggregate,'families':each})
            if aggregate<best:best,best_step,best_theta=aggregate,step,theta.detach().clone()
    with torch.no_grad():scores={name:metrics(predict(w,arm,best_theta)[0],w) for name,w in vals.items()}
    return {'seed':seed,'arm':arm,'trainable_global_params':best_theta.numel(),'updates':STEPS,
        'initial_sha256':initial,'selected_sha256':hashlib.sha256(best_theta.numpy().tobytes()).hexdigest(),
        'selected_logits':[float(x) for x in best_theta], 'best_step':best_step,
        'mean_final':best,'families':scores,'trace':curve,
        'train_seconds':time.perf_counter()-started}

def replay_cold(w,arm,theta):
    with torch.no_grad():
        _,whole=predict(w,arm,theta)
        _,half=predict(w,arm,theta,stop=24)
        _,again=predict(w,arm,theta,start=24,carry=half.clone())
    return bool(torch.equal(whole,again) and whole.numel()==len(w['effects'])*256)

def study():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    start=time.perf_counter();deadline=start+165
    root=Path(__file__).resolve().parents[1]
    o={'schema':'ttt-hybrid.p91-selective-stability-dev-result.v1','freeze_commit':FREEZE,
       'status':'NOT_STARTED','science_pass':False,'OOD_generated':False,'NEGATIVE_generated':False,
       'native_full_hybrid_trained':False,'whole_process_compute_matched':False}
    o['custody']={p:{'expected':sha,'actual':p86.blob(root/p)} for p,sha in BLOBS.items()}
    if not all(v['actual']==v['expected'] for v in o['custody'].values()):
        o['status']='DEV_FAIL_SOURCE_CUSTODY';return o
    datasets={name:{split:source(seed,128 if split=='FIT' else 64) for split,seed in family.items()}
              for name,family in FAMILIES.items()}
    o['sources']={name:{split:{'seed':FAMILIES[name][split],
      'n':len(w['effects']),'sha256':signature(w),'audit':p86.audit(p90.p86_world(w))}
      for split,w in data.items()} for name,data in datasets.items()}
    if not all(z['audit']['pass'] for family in o['sources'].values() for z in family.values()):
        o['status']='DEV_FAIL_SOURCE_AUDIT';return o
    vals={name:family['VAL'] for name,family in datasets.items()}
    combined={k:torch.cat([family['FIT'][k] for family in datasets.values()],dim=0)
              for k in ('actions','effects','changed_world')}
    controls={}
    for name,val in vals.items():
        y,W=predict(val,'fixed');fixed=metrics(y,val)
        zero=metrics(torch.zeros_like(y),val)
        reference=p90.p86.online(p90.p86_world(val),1.,.99)[0]
        parity=bool(torch.allclose(y[:,-1],reference,atol=1e-6,rtol=1e-6))
        wrong={k:v.clone() for k,v in val.items()}
        wrong['effects'][:,:48]=val['effects'][:,:48][:,torch.arange(48).roll(1)]
        corrupted,_=predict(wrong,'fixed')
        wrong_score=float((corrupted[:,-1]-val['effects'][:,-1]).square().mean())
        controls[name]={'fixed':fixed,'zero':zero,'parity':parity,
           'wrong_credit_final':wrong_score,'wrong_credit_pass':wrong_score>=1.5*fixed['final'] and wrong_score>=.01,
           'cold_fixed':replay_cold(val,'fixed',None)}
    o['controls']=controls
    if not all(t['parity'] and t['wrong_credit_pass'] and t['cold_fixed'] and
               t['fixed']['stable']<=.8*t['zero']['stable'] and
               t['fixed']['changed']<=.8*t['zero']['changed'] for t in controls.values()):
        o['status']='DEV_FAIL_CONVENTIONAL_CONTROLS';return o
    indices=torch.randint(len(combined['effects']),(STEPS,BATCH),
       generator=torch.Generator().manual_seed(910299))
    o['sampler_sha256']=hashlib.sha256(indices.numpy().tobytes()).hexdigest()
    rows=[]
    for seed in SEEDS:
        rows.append({'seed':seed,'arms':{arm:train(seed,arm,combined,vals,indices,deadline) for arm in ARMS}})
    o['rows']=rows
    for row in rows:
        # Check selected weights cold restore, not retrained weights or a hidden sidecar.
        row['cold_restores']={}
        for arm,s in row['arms'].items():
            theta=torch.tensor(s['selected_logits'])
            row['cold_restores'][arm]=all(replay_cold(v,arm,theta) for v in vals.values())
        c=row['arms']['constant'];prot=row['arms']['protected'];original=row['arms']['contradiction']
        def plateau(z):return all(t['mean_final']<=1.05*z['mean_final'] for t in z['trace'][-2:])
        row['trained_constant_qualified']=bool(all(c['families'][name]['final']<=.90*controls[name]['zero']['final'] for name in vals)
              and c['best_step']<=70 and plateau(c) and row['cold_restores']['constant'])
        row['protected_qualified']=bool(prot['best_step']<=70 and plateau(prot) and row['cold_restores']['protected']
             and all(prot['families'][name]['final']<=.90*controls[name]['fixed']['final']
                  and prot['families'][name]['final']<=.90*c['families'][name]['final']
                  and prot['families'][name]['final']<=.90*original['families'][name]['final']
                  and prot['families'][name]['stable']<=1.05*controls[name]['fixed']['stable']
                  and prot['families'][name]['changed']<=.85*controls[name]['fixed']['changed']
                  and prot['families'][name]['stable_late']<=1.05*controls[name]['fixed']['stable_late']
                  for name in vals))
    o['all_conventional_qualified']=all(r['trained_constant_qualified'] for r in rows)
    o['all_protected_qualified']=all(r['protected_qualified'] for r in rows)
    o['status']=('DEV_FAIL_BASELINE' if not o['all_conventional_qualified'] else
         'DEV_FAIL_STABILITY_OR_ADAPTATION' if not o['all_protected_qualified'] else
         'DEV_FIT_VAL_CANDIDATE_ONLY_NOT_SCIENCE')
    o['elapsed_seconds']=time.perf_counter()-start
    return o

def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();r=study();args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':r['status'],'seconds':r.get('elapsed_seconds'),
         'families':list(r.get('controls',{})),'science_pass':r['science_pass']}))

if __name__=='__main__':main()
