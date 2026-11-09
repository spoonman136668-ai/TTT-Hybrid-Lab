"""P38-R1 fresh authored DEV cohort; no external physical scientific first use.

The original P38 diagnostic data are consumed. New seeded cohort frozen in
experiments/native_p38r1_new_source_cohort_frozen.json before this study.
Uses exact corrected P38 GitHub source, independent sham RNG, new splits.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import torch
import native_p38_conditional_action_choice_dev as p38

ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'experiments/native_p38r1_new_source_cohort_frozen.json'
SCHEMA='ttt-hybrid.native-p38r1.new-authored-source-dev-result.v1'
FRESH=(6101,6113,6127)
TRAIN_SEED=242000
VAL_SEED=242500
HELDOUT_SEED=243000
ORIGINAL_BLOB='16d4b1c691659365251d97d6a5635021b41d080e'


def file_sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def original_gitblob(path):
    src=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest()


def freeze():
    d=json.loads(FROZEN.read_text(encoding='utf-8'))
    if (d.get('schema')!='ttt-hybrid.native-p38r1.fresh-cohort-state-reversing-conditional-choice-frozen.v1'
        or d['phase']!='FROZEN_BEFORE_NEW_AUTHOR_SOURCE_TRAINING_OR_EVALUATION'
        or d['parent']['corrected_source_blob']!=ORIGINAL_BLOB
        or d['parent']['old_original_source_first_use_consumed'] is not True
        or d['cohort']['fresh_model_seeds']!=list(FRESH)
        or d['cohort']['source_train_seed']!=TRAIN_SEED
        or d['cohort']['source_validation_seed']!=VAL_SEED
        or d['cohort']['source_heldout_seed']!=HELDOUT_SEED
        or d['cohort']['real_training_actions']!=1024
        or d['cohort']['real_validation_actions']!=128
        or d['cohort']['real_heldout_actions']!=1024
        or d['comparison']['actual_minibatches_per_neural_arm_per_seed']!=256
        or d['comparison']['identical_minibatch_data_order_all_neural_arms'] is not True
        or d['pass']['all_three_model_seeds'] is not True
        or d['authority']['no_dispatch_or_owner_runner_changes'] is not True):
        raise RuntimeError('P38R1_PRECOMMITTED_SOURCE_AND_PARAMETER_DRIFT')
    if original_gitblob(ROOT/'native_p38_conditional_action_choice_dev.py')!=ORIGINAL_BLOB:
        raise RuntimeError('P38R1_CORRECTED_P38_SOURCE_BLOB_DRIFT')
    if os.environ.get('NATIVE_P38R1_AUTHORED_DEV_ONLY')!='1':
        raise RuntimeError('P38R1_ORIGINAL_SCIENCE_FIRST_USE_NOT_AUTHORIZED')
    return d


def fit_neural(arm,seed,sensors,actions,observed_next):
    original=p38.SEEDS
    try:
        # Strictly scoped to this isolated single-process DEV experiment.
        # The original frozen P38 source does NOT have these seeds registered.
        p38.SEEDS=FRESH
        return p38.train_one(arm,seed,sensors,actions,observed_next)
    finally:
        p38.SEEDS=original


def metrics_gate(scores,frozen):
    cutoff=frozen['pass']
    result={}
    for goal in ('positive_goal','negative_goal'):
        actual=scores[p38.ARMS[0]+':'+goal]
        sham=scores[p38.ARMS[1]+':'+goal]
        mlp=scores[p38.ARMS[2]+':'+goal]
        ridge=scores['fully_trained_conventional_ridge:'+goal]
        fixed=max(scores['fixed_action_minus_one:'+goal]['fraction_optimal'],
                  scores['fixed_action_plus_one:'+goal]['fraction_optimal'])
        result[goal]={
          'optimal_correct_on_both_cues':min(actual['positive_cue_optimal'],actual['negative_cue_optimal'])>=cutoff['native_optimal_min_every_goal_every_cue'],
          'higher_than_wrong_action_credit':actual['fraction_optimal']-sham['fraction_optimal']>=cutoff['improvement_over_sham_min'],
          'higher_than_best_constant':actual['fraction_optimal']-fixed>=cutoff['improvement_over_best_fixed_min'],
          'not_worse_than_fully_trained_MLP':actual['fraction_optimal']>=mlp['fraction_optimal']-cutoff['within_fully_trained_MLP'],
          'not_worse_than_fully_fitted_ridge':actual['fraction_optimal']>=ridge['fraction_optimal']-cutoff['within_fully_fitted_ridge']}
    return result


def run(receipt):
    frozen=freeze()
    if receipt.exists() or (receipt.parent/'native_p38r1_frozen_weights').exists():
        raise RuntimeError('P38R1_NEW_SOURCE_FIRST_USE_ALREADY_CONSUMED_NO_REPLAY')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    sensors,actions,actual_next,train_sha=p38.observed_transitions(TRAIN_SEED,1024)
    val_s,val_a,val_next,val_sha=p38.observed_transitions(VAL_SEED,128)
    heldout_states=p38.source_states(HELDOUT_SEED,512,True)
    weights=receipt.parent/'native_p38r1_frozen_weights'
    weights.mkdir(parents=True)
    started=time.perf_counter();records=[]
    for seed in FRESH:
        models={};trained={}
        for arm in p38.ARMS:
            model,evidence=fit_neural(arm,seed,sensors,actions,actual_next)
            path=weights/(str(seed)+'_'+arm+'.pt')
            torch.save(model.state_dict(),path)
            evidence.update(weights_file=path.name,weights_sha256=file_sha256(path))
            models[arm]=model;trained[arm]=evidence
        fitted_ridge=p38.ridge_fit(sensors,actions,actual_next)
        models['fully_trained_conventional_ridge']=fitted_ridge
        torch.save(fitted_ridge,weights/(str(seed)+'_ridge.pt'))
        scores,real_sha=p38.heldout(models,heldout_states)
        records.append({'seed':seed,'trained':trained,'source_outcome_trace_sha256':real_sha,
                        'real_source_heldout_metrics':scores,'frozen_criteria':metrics_gate(scores,frozen)})
    success=all(all(all(v.values()) for v in rec['frozen_criteria'].values()) for rec in records)
    result={'schema':SCHEMA,'engineering':'LOCAL_AUTHORED_SOURCE_DEV_ONLY',
      'authored_source_protocol_satisfied':success,
      'not_independent_external_scientific_pass':True,
      'new_external_scientific_source_first_use':'NOT_RUN',
      'original_P38_local_prototype_consumed_previously':True,
      'original_corrected_P38_source_gitblob':ORIGINAL_BLOB,
      'real_authored_train_source_steps':1024,'real_authored_validation_source_steps':128,
      'real_authored_heldout_source_steps':1024,'heldout_goal_decisions_per_seed':1024,
      'real_train_source_sha256':train_sha,'real_validation_source_sha256':val_sha,
      'actual_wall_seconds':time.perf_counter()-started,
      'weights_dir':str(weights.resolve()),'seeds':records}
    receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result


def cold(receipt):
    freeze()
    result=json.loads(receipt.read_text(encoding='utf-8'))
    if result['schema']!=SCHEMA or len(result['seeds'])!=3 or result['real_authored_train_source_steps']!=1024:
        raise RuntimeError('P38R1_ORIGINAL_REAL_SOURCE_CUSTODY_INCOMPLETE')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    weights=Path(result['weights_dir'])
    states=p38.source_states(HELDOUT_SEED,512,True)
    for record in result['seeds']:
        models={}
        for arm,info in record['trained'].items():
            path=weights/info['weights_file']
            if file_sha256(path)!=info['weights_sha256']:
                raise RuntimeError('P38R1_COLD_TRAINED_CHECKPOINT_SHA_MISMATCH')
            m=p38.ConventionalMLP() if arm==p38.ARMS[2] else p38.native()
            m.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
            m.eval();models[arm]=m
        models['fully_trained_conventional_ridge']=torch.load(weights/(str(record['seed'])+'_ridge.pt'),weights_only=True)
        scores,sha=p38.heldout(models,states)
        if scores!=record['real_source_heldout_metrics'] or sha!=record['source_outcome_trace_sha256']:
            raise RuntimeError('P38R1_INDEPENDENT_SOURCE_COLD_ACTION_CONSEQUENCE_DRIFT')
    return {'status':'PASS_BIT_EXACT_NATIVE_FRESH_PROCESS_DECISIONS_AND_SOURCE_TRACE',
      'seeds':list(FRESH),'heldout_real_source_actions':1024,
      'replayed_source_goal_decisions_per_seed':1024,'additional_optimizer_steps':0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('train','cold'),required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    if not args.receipt.is_absolute():raise ValueError('P38R1_ABSOLUTE_OUTPUT_PATH_REQUIRED')
    report=run(args.receipt) if args.mode=='train' else cold(args.receipt)
    print('P38R1_NEW_COHORT_SOURCE_RESULT='+json.dumps(report if args.mode=='cold'
          else {'engineering':report['engineering'],'positive_criteria':report['authored_source_protocol_satisfied'],
          'wall_seconds':report['actual_wall_seconds'],
          'seed_metrics':[{'seed':x['seed'],'metrics':x['real_source_heldout_metrics'],'criteria':x['frozen_criteria']}
                          for x in report['seeds']]},sort_keys=True))

if __name__=='__main__':main()
