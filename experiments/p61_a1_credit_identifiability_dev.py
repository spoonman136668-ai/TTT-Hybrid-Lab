"""P61-A1 retrospective: compare noninvertible lost-action evidence and strong classical controls.

Replays ONLY previously consumed P61 authored source. A heldout-prior oracle is
reported as an upper bound, NEVER a fair learned-system baseline/scientific pass.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from experiments.p61_hidden_motor_polarity_dev import source

EXPECTED={
 'fit':'85ddaf146e170dee79a00358479ca0df2971c3ee927c8d9a65d21688e40d7c49',
 'val':'831beae2d2e2e92e6f9d6a9196620608d931b8a46ab0dd148290a88dbb72df1e',
 'unfamiliar_dev':'a151ee31fae9814a2c7e5bba8c199a1f065bf5f237365f5b46bae429c857763b'}

def data(split):
    if split not in EXPECTED:raise ValueError('P61_A1_SOURCE_NOT_REGISTERED')
    d,sha=source(split)
    if sha!=EXPECTED[split]:raise ValueError('P61_A1_SOURCE_CUSTODY_MISMATCH')
    return {k:v.detach().cpu().numpy() for k,v in d.items()},sha

def label_after_action_reward(d):
    # Historical P61 +/-1 actual reward algebraically reveals the true final sign.
    # The evaluator latent label is NEVER a fitting input.
    return d['actual_query_action'][:,0]*d['actual_observed_reward'][:,0]

def features(d,erase=False):
    h=d['history']
    if erase:
        x=h[:,:,1].astype(np.float64)   # outcomes ONLY, no action identities
    else:
        u=(h[:,:,0]*h[:,:,1]).astype(np.float64)
        x=np.stack((u[:,:3].sum(1),u[:,3:].sum(1)),axis=1)
    return np.column_stack((x,np.ones(len(x))))

def ridge(fit,eval,erase=False):
    x=features(fit,erase)
    y=label_after_action_reward(fit).astype(np.float64)
    penalty=np.eye(x.shape[1])*1e-3
    penalty[-1,-1]=0
    weights=np.linalg.solve(x.T@x+penalty,x.T@y)
    return np.where(features(eval,erase)@weights>=0,1.,-1.),weights

def prior_bayes(d,switch_p,noise):
    u=(d['history'][:,:,0]*d['history'][:,:,1]).astype(np.float64)
    first=(.18/noise**2)*u[:,:3].sum(1)
    last=(.18/noise**2)*u[:,3:].sum(1)
    plus=last+np.logaddexp(np.log1p(-switch_p)+first,np.log(switch_p)-first)
    minus=-last+np.logaddexp(np.log1p(-switch_p)-first,np.log(switch_p)+first)
    return np.where(plus>=minus,1.,-1.)

def ci_paired(p,ref,y):
    delta=(p==y).astype(float)-(ref==y).astype(float)
    gen=np.random.default_rng(993399)
    means=[]
    for k in range(10):
        idx=gen.integers(0,len(y),(1000,len(y)))
        means.extend(delta[idx].mean(1).tolist())
    return [float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]

def run():
    fit,fs=data('fit');val,vs=data('val');ood,os=data('unfamiliar_dev')
    cred=(ood['history'][:,:,0]*ood['history'][:,:,1]).astype(np.float64)
    recent=np.where(cred[:,-3:].sum(1)>=0,1.,-1.)
    all6=np.where(cred.sum(1)>=0,1.,-1.)
    fitted,w=ridge(fit,ood)
    redacted,r=ridge(fit,ood,erase=True)
    predictions={'recent3':recent,'all6':all6,'factual_ridge':fitted,
                 'noninvertibly_erased_action_identity':redacted,
                 'fit_prior_author_model':prior_bayes(ood,.5,.24),
                 'ood_prior_ORACLE_NOT_FAIR':prior_bayes(ood,.75,.30),
                 'constant_plus':np.ones(len(recent))}
    y=ood['evaluator_hidden_final_mode'][:,0]
    scores={name:{'n':len(y),'correct':int((x==y).sum()),'accuracy':float((x==y).mean()),
                  'paired_gain_vs_recent3':float((x==y).mean()-(recent==y).mean()),
                  'paired_ci95':ci_paired(x,recent,y)} for name,x in predictions.items()}
    return {'schema':'ttt-hybrid.p61-a1-retrospective-credit-identifiability.v1',
            'source':'EXACT_PUBLISHED_CONSUMED_P61_DEV_NOT_NEW_SCIENCE',
            'parent_pr':160,'parent_sha':'0cabb458dc6881761423368a43fc4e4343cd6dc2',
            'source_sha256_verified':{'fit':fs,'val':vs,'unfamiliar_dev':os},
            'factual_train_reward_reveals_final_sign_after_actual_action':bool(np.array_equal(label_after_action_reward(fit),fit['evaluator_hidden_final_mode'][:,0])),
            'factual_ridge_weights':w.tolist(),'erased_identity_ridge_weights':r.tolist(),
            'factual_ridge_validation_accuracy':float((ridge(fit,val)[0]==val['evaluator_hidden_final_mode'][:,0]).mean()),
            'erased_identity_validation_accuracy':float((ridge(fit,val,erase=True)[0]==val['evaluator_hidden_final_mode'][:,0]).mean()),
            'results':scores,
            'redacted_44_to_56_pct_chance_band':.44<=scores['noninvertibly_erased_action_identity']['accuracy']<=.56,
            'original_p61_frozen_gain':'FAIL_0_OF_3_PRESERVED',
            'oracle_prior_not_valid_learned_control':True,
            'scientific_first_use':'NOT_RUN','full_original_final_sha_windows_wsl_regression':'UNVERIFIED',
            'FORGE_owner_model_promotion':'NONE'}

def main():
    args=argparse.ArgumentParser()
    args.add_argument('--receipt',required=True,type=Path)
    a=args.parse_args()
    report=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'scores':report['results'],'source':report['source_sha256_verified']},indent=2))
if __name__=='__main__':main()
