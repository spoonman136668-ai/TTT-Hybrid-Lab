"""P77 frozen independent negative-cohort readout, no retuning or OOD reuse."""
import json, time, hashlib
from pathlib import Path
import torch
from experiments import p77_identity_bypass_dev as p


def run():
    start=time.perf_counter()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    original=json.loads(Path(__file__).with_name('p77_identity_bypass_dev_result.json').read_text())
    assert original['status']=='DEV_PASS_IDENTITY_BYPASS_SIDE_CAR'
    fit,val=p.make_world('fit'),p.make_world('val')
    assert original['source_hashes']['fit']==p.source_sha(fit)
    assert original['source_hashes']['val']==p.source_sha(val)
    idx=torch.randint(p.COUNTS['fit'],(p.UPDATES,p.BATCH),generator=torch.Generator().manual_seed(p.SAMPLE_SEED))
    assert original['sampler_sha256']==p.p76.tensor_hash(idx)
    neg=p.make_world('negative')
    wrong=p.wrong_credit(neg,seed=p.SEEDS['negative'])
    rows=[]
    for o in original['rows']:
        seed=o['seed']
        c=p.p76.original(seed)
        ff=p.p76.features(c,fit);vf=p.p76.features(c,val)
        m,meta=p.train_arm(seed,'bypass',ff,vf,idx,start)
        expected=o['arms']['bypass']['best_model_sha256']
        if meta['best_model_sha256']!=expected:
            raise RuntimeError('P77_CHECKPOINT_PARITY_FAILURE '+str(seed))
        nfeat=p.p76.features(c,neg)
        wfeat=p.p76.features(c,wrong)
        clean=p.evaluate(m,nfeat,'bypass')
        incorrect=p.evaluate(m,wfeat,'bypass')
        rows.append({'seed':seed,'model_sha256':expected,'cold_state_match':meta['cold_bitwise'],
                     'clean_negative_cohort_mse':clean['mse'],'clean_address_hit':clean['hit'],
                     'incorrect_credit_negative_cohort_mse':incorrect['mse'],
                     'wrong_credit_worse_2x':incorrect['mse']>=2*max(clean['mse'],1e-5),
                     'trained_checkpoint_parity':True})
    obj={'schema':'ttt-hybrid.p77-independent-negative-completion-dev.v1',
         'status':'NEGATIVE_PASS_ALL_3' if all(x['wrong_credit_worse_2x'] for x in rows) else 'NEGATIVE_FAIL',
         'classification':'AUTHORED_DEV_FROZEN_INDEPENDENT_NEGATIVE_COHORT_COMPLETED_AFTER_PRIMARY_OOD_WITH_NO_RETUNE',
         'protocol_commit':'d01f9f4b7246ee88d10cb1d16ed8700b82cbc258',
         'original_study_git_blob_expected':'248b64897562d26cd75f75e804182f9e24dabe42',
         'negative_seed':p.SEEDS['negative'],'negative_cohort_source_sha256':p.source_sha(neg),
         'wrong_credit_source_sha256':p.source_sha(wrong),
         'model_rows':rows,'first_study_ood_already_consumed':True,
         'none_of_p77_parameter_settings_changed':True,
         'no_new_ood_generated_or_rescored':True,
         'elapsed_seconds':time.perf_counter()-start,
         'scientific_pass':False,'full_cross_platform':'NOT_RUN','FORGE':'UNCHANGED','merge':'NONE'}
    Path(__file__).with_name('p77_independent_negative_completion_result.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    return obj

if __name__=='__main__':
    j=run();print(json.dumps({k:j[k] for k in ('status','negative_seed','elapsed_seconds')}));print(j['model_rows'])
