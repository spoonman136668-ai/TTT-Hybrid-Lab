"""P38 authored token-free conditional intervention DEV trial.

This is a prospective interface implementation only. Original frozen P38
seed cohort was already exercised in a separate local diagnostic prototype;
therefore do not interpret any later replay as independent first-use science.
The supplied goal is not learned. No external physical first-use.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time
from pathlib import Path
import torch
from torch import nn
from native_continuous_state import NativeContinuousHybrid
from native_intervention_choice import select_goal_action, preview_interventions

ROOT=Path(__file__).resolve().parent
SPEC=ROOT/'experiments/native_p38_conditional_action_choice_frozen.json'
SCHEMA='ttt-hybrid.native-p38.authored-conditional-intervention-dev.v1'
SEEDS=(6007,6011,6029)
ARMS=('factual_native_hybrid','shuffled_action_native_hybrid','fully_trained_conventional_mlp')
TRAIN_SEED,VALIDATION_SEED,HELDOUT_SEED=238000,238500,239000
N_TRAIN,N_VALID,N_HELDOUT=1024,128,512
EPOCHS,BATCH,LR=32,128,.003


def frozen():
    s=json.loads(SPEC.read_text(encoding='utf-8'))
    if (s.get('schema')!='ttt-hybrid.native-p38.authored-state-conditional-intervention-choice-frozen.v1'
        or s['state']!='PROSPECTIVELY_FROZEN_BEFORE_NEW_AUTHORED_SOURCE_ACTIONS'
        or s['parent']['sha']!='edcd6936fd7893138f73b94e046d32af79f75ad1'
        or s['authored_source']['source_dataset_seed']!=TRAIN_SEED
        or s['authored_source']['validation_dataset_seed']!=VALIDATION_SEED
        or s['authored_source']['heldout_source_seed']!=HELDOUT_SEED
        or s['learning']['model_seeds']!=list(SEEDS)
        or s['learning']['epochs']!=EPOCHS
        or s['learning']['batch_size']!=BATCH
        or s['learning']['optimizer_steps_per_seed_per_neural_arm']!=256
        or s['science_dev_success']['three_of_three_seeds'] is not True
        or s['authority']['no_new_runner_controller_scheduler_or_dispatch'] is not True):
        raise RuntimeError('P38_FROZEN_ORIGINAL_PRECOMMITTED_PROTOCOL_DRIFT')
    return s


def guard():
    if os.getenv('NATIVE_P38_AUTHORED_DEV_ONLY')!='1':
        raise RuntimeError('P38_AUTHORED_ONLY_PROTECTED_FIRST_USE_DENIED')
    return frozen()


class AuthoredConditionalWorld:
    def __init__(self,x:float,cue:float):
        if not (math.isfinite(x) and math.isfinite(cue) and .45<=abs(cue)<=.99):
            raise ValueError('P38_INVALID_SOURCE_INITIAL_SENSOR')
        self.x=float(x);self.cue=float(cue)
    def observe(self):return (self.x,self.cue)
    def step(self,action):
        if float(action) not in (-1.,1.):
            raise ValueError('P38_SOURCE_ACTION_NOT_PRECOMMITTED')
        before=self.observe()
        self.x=self.x+.3*action*self.cue+.04*action*math.sin(3*self.x)+.02*math.sin(5*self.x)
        return before,float(action),self.observe()


def source_states(seed,n,heldout=False):
    rng=random.Random(seed)
    return [(rng.uniform(-1. if heldout else -.8,1. if heldout else .8),
             (1 if i%2==0 else -1)*rng.uniform(.45 if heldout else .55,.99 if heldout else .95))
            for i in range(n)]


def observed_transitions(seed,n,heldout=False):
    rows=source_states(seed,n,heldout)
    rng=random.Random(seed+83)
    obs=[];actions=[];truth=[];h=hashlib.sha256()
    for row in rows:
        action=1. if rng.randrange(2) else -1.
        previous,taken,next_obs=AuthoredConditionalWorld(*row).step(action)
        obs.append(previous);actions.append((taken,));truth.append(next_obs)
        h.update(json.dumps((previous,taken,next_obs),separators=(',',':')).encode())
    return (torch.tensor(obs,dtype=torch.float32),
            torch.tensor(actions,dtype=torch.float32),
            torch.tensor(truth,dtype=torch.float32),h.hexdigest())


class ConventionalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(3,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh(),nn.Linear(24,2))
    def forward(self,sensors,actions):
        return sensors+self.layers(torch.cat((sensors,actions),1))


def native():return NativeContinuousHybrid(2,1,width=12,history_limit=3,fast_rate=.25)


def train_one(arm,seed,sensors,actions,next_observations,epochs=EPOCHS):
    if arm not in ARMS or seed not in SEEDS:raise ValueError('P38_UNREGISTERED_TRAINING_ARM_OR_SEED')
    torch.manual_seed(seed)
    model=ConventionalMLP() if arm==ARMS[2] else native()
    expected=746 if arm==ARMS[2] else 734
    if sum(p.numel() for p in model.parameters())!=expected:
        raise RuntimeError('P38_ORIGINAL_SOURCE_MODEL_PARAMETER_DRIFT')
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=LR)
    # CRITICAL: independent RNG for SHAM label permutation preserves exact
    # minibatch order for factual/sham/native and trained conventional MLP.
    order_rng=torch.Generator().manual_seed(seed+881)
    sham_rng=torch.Generator().manual_seed(seed+1881)
    incorrect=0;samples=0;steps=0;start=time.perf_counter()
    for epoch in range(epochs):
        indices=torch.randperm(len(sensors),generator=order_rng)
        for subset in indices.split(BATCH):
            selected=actions[subset]
            if arm==ARMS[1]:
                mixed=selected[torch.randperm(len(selected),generator=sham_rng)]
                incorrect+=int((mixed!=selected).sum());selected=mixed
                samples+=len(selected)
            predicted=(model.step(sensors[subset],selected,torch.ones(len(subset),1))[0]
                       if arm!=ARMS[2] else model(sensors[subset],selected))
            loss=(predicted-next_observations[subset]).square().mean()
            if not bool(torch.isfinite(loss)):raise RuntimeError('P38_NONFINITE_SOURCE_LEARNING_LOSS')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
    model.eval()
    if epochs==EPOCHS and (steps!=256 or arm==ARMS[1] and incorrect<samples//3):
        raise RuntimeError('P38_ORIGINAL_SOURCE_TRAINING_OR_SHAM_CUSTODY_FAILED')
    return model,{'real_source_train_examples':len(sensors),'optimizer_minibatches':steps,
                  'trainable_parameters':expected,'wrong_action_aux_samples':samples,
                  'wrong_action_credit_fraction':incorrect/samples if samples else None,
                  'train_seconds':time.perf_counter()-start}


def ridge_features(sensors,actions):
    x,c,a=sensors[:,0],sensors[:,1],actions[:,0]
    return torch.stack((torch.ones_like(x),x,c,a,a*c,a*torch.sin(3*x),torch.sin(5*x)),dim=1)


def ridge_fit(sensors,actions,next_obs):
    features=ridge_features(sensors.double(),actions.double())
    targets=(next_obs-sensors).double()
    regularizer=torch.eye(features.shape[1],dtype=torch.float64)*.0001
    return torch.linalg.solve(features.T@features+regularizer,features.T@targets)


def predict(arm,model,sensor,candidates):
    b,k,_=candidates.shape
    if arm in ARMS[:2]:
        return preview_interventions(model,sensor,candidates,torch.ones((b,1)))
    if arm==ARMS[2]:
        return model(sensor.repeat_interleave(k,0),candidates.reshape(b*k,1)).reshape(b,k,2)
    if arm=='fully_trained_conventional_ridge':
        data=sensor.repeat_interleave(k,0).double()
        delta=ridge_features(data,candidates.reshape(b*k,1).double())@model
        return (data+delta).float().reshape(b,k,2)
    raise ValueError('P38_UNREGISTERED_EVALUATION_ARM')


def heldout(models,states):
    sensor=torch.tensor(states,dtype=torch.float32)
    candidates=torch.tensor([-1.,1.],dtype=torch.float32).reshape(1,2,1).expand(len(states),2,1)
    actual=[];h=hashlib.sha256()
    for x,cue in states:
        outcomes=[]
        for action in (-1.,1.):
            before,taken,result=AuthoredConditionalWorld(x,cue).step(action)
            h.update(json.dumps((before,taken,result),separators=(',',':')).encode())
            outcomes.append(result)
        actual.append(outcomes)
    actual=torch.tensor(actual,dtype=torch.float32)
    scores={}
    for direction in (-1,1):
        target=sensor.clone();target[:,0]+=direction*.32
        true_errors=(actual-target.unsqueeze(1)).square().sum(-1)
        best=true_errors.argmin(1)
        goal='positive_goal' if direction==1 else 'negative_goal'
        positive=(sensor[:,1]>0)
        for arm,model in models.items():
            with torch.no_grad():
                if arm in ARMS[:2]:
                    choice=select_goal_action(model,sensor,candidates,torch.ones((len(states),1)),target)
                    predicted,index=choice.predictions,choice.chosen_indices
                else:
                    predicted=predict(arm,model,sensor,candidates)
                    index=(predicted-target.unsqueeze(1)).square().sum(-1).argmin(-1)
            ok=(index==best)
            reward=-true_errors[torch.arange(len(states)),index].sqrt()
            scores[arm+':'+goal]={
                'fraction_optimal':float(ok.float().mean()),
                'positive_cue_optimal':float(ok[positive].float().mean()),
                'negative_cue_optimal':float(ok[~positive].float().mean()),
                'actual_true_reward_mean':float(reward.mean()),
                'source_transition_MSE':float((predicted-actual).square().mean()),
                'actual_physical_actions_per_state':2}
        for arm,idx in (('fixed_action_minus_one',0),('fixed_action_plus_one',1)):
            scores[arm+':'+goal]={'fraction_optimal':float((best==idx).float().mean())}
    return scores,h.hexdigest()


def gates(scores):
    t=frozen()['science_dev_success']
    verdict={}
    for goal in ('positive_goal','negative_goal'):
        actual=scores[ARMS[0]+':'+goal]
        sham=scores[ARMS[1]+':'+goal]
        mlp=scores[ARMS[2]+':'+goal]
        ridge=scores['fully_trained_conventional_ridge:'+goal]
        fixed=max(scores['fixed_action_minus_one:'+goal]['fraction_optimal'],
                  scores['fixed_action_plus_one:'+goal]['fraction_optimal'])
        verdict[goal]={
            'native_optimal':actual['fraction_optimal']>=t['every_native_seed_goal_action_optimal_fraction_at_least'],
            'beats_sham':actual['fraction_optimal']>=sham['fraction_optimal']+t['native_minus_sham_goal_action_optimal_fraction_at_least'],
            'beats_best_constant':actual['fraction_optimal']>=fixed+t['native_minus_best_fixed_action_optimal_fraction_at_least'],
            'within_fully_trained_MLP':actual['fraction_optimal']>=mlp['fraction_optimal']-t['native_no_more_than_below_mlp_optimal_fraction'],
            'within_fully_fitted_ridge':actual['fraction_optimal']>=ridge['fraction_optimal']-t['native_no_more_than_below_ridge_optimal_fraction'],
            'both_cues':min(actual['positive_cue_optimal'],actual['negative_cue_optimal'])>=t['every_native_seed_goal_action_optimal_fraction_at_least']}
    return verdict


def run(receipt):
    guard()
    if receipt.exists():raise RuntimeError('P38_PREVIOUSLY_CONSUMED_AUTHORED_SOURCE_NO_REPLAY')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train_s,train_a,train_next,train_sha=observed_transitions(TRAIN_SEED,N_TRAIN)
    val_s,val_a,val_next,val_sha=observed_transitions(VALIDATION_SEED,N_VALID)
    source_heldout=source_states(HELDOUT_SEED,N_HELDOUT,heldout=True)
    checkpoint_dir=receipt.parent/'native_p38_frozen_weights'
    if checkpoint_dir.exists():raise RuntimeError('P38_EXISTING_CHECKPOINTS_NO_OVERWRITE')
    checkpoint_dir.mkdir(parents=True)
    records=[]
    for seed in SEEDS:
        models={};per_arm={}
        for arm in ARMS:
            m,training=train_one(arm,seed,train_s,train_a,train_next)
            path=checkpoint_dir/(str(seed)+'_'+arm+'.pt')
            torch.save(m.state_dict(),path)
            training.update(checkpoint_name=path.name,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            per_arm[arm]=training;models[arm]=m
        ridge=ridge_fit(train_s,train_a,train_next)
        rp=checkpoint_dir/(str(seed)+'_fully_trained_conventional_ridge.pt')
        torch.save(ridge,rp)
        models['fully_trained_conventional_ridge']=ridge
        scored,heldout_sha=heldout(models,source_heldout)
        records.append({'seed':seed,'real_training':per_arm,'real_heldout_metrics':scored,
                        'actual_heldout_source_sha256':heldout_sha,'precommitted_conditions':gates(scored)})
    success=all(all(all(v.values()) for v in rec['precommitted_conditions'].values()) for rec in records)
    data={'schema':SCHEMA,'engineering':'AUTHORED_DEV_LOCAL_ONLY',
          'SCIENCE_DEV_PROTOCOL_CRITERIA_PASS':success,
          'independent_external_science':'UNVERIFIED',
          'prior_local_prototype_consumed_same_frozen_cohort':True,
          'protected_first_use':'NOT_RUN',
          'train_actual_source_steps':N_TRAIN,'validation_actual_source_steps':N_VALID,
          'heldout_actual_source_steps':N_HELDOUT*2,'goal_decisions_per_seed':N_HELDOUT*2,
          'train_source_sha256':train_sha,'validation_source_sha256':val_sha,
          'checkpoint_dir':str(checkpoint_dir.resolve()),'records':records}
    receipt.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    return data


def cold(receipt):
    guard()
    data=json.loads(receipt.read_text())
    if data['schema']!=SCHEMA or len(data['records'])!=3:raise RuntimeError('P38_COLD_AUTHORED_RECEIPT_INVALID')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    states=source_states(HELDOUT_SEED,N_HELDOUT,True)
    root=Path(data['checkpoint_dir'])
    for rec in data['records']:
        models={}
        for arm,item in rec['real_training'].items():
            path=root/item['checkpoint_name']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['checkpoint_sha256']:
                raise RuntimeError('P38_COLD_CHECKPOINT_FILE_SHA_CHANGED')
            model=ConventionalMLP() if arm==ARMS[2] else native()
            model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
            model.eval();models[arm]=model
        models['fully_trained_conventional_ridge']=torch.load(
            root/(str(rec['seed'])+'_fully_trained_conventional_ridge.pt'),weights_only=True)
        metrics,trace_sha=heldout(models,states)
        if metrics!=rec['real_heldout_metrics'] or trace_sha!=rec['actual_heldout_source_sha256']:
            raise RuntimeError('P38_FRESH_PROCESS_ACTION_OR_SOURCE_OUTCOMES_NOT_EQUAL')
    return {'status':'PASS_BIT_EXACT_COLD_FROZEN_NATIVE_SOURCE_DECISIONS',
            'model_seeds':list(SEEDS),'actual_source_episodes':N_HELDOUT*2,
            'optimizer_updates':0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('train','cold'),required=True)
    ap.add_argument('--receipt',required=True,type=Path)
    args=ap.parse_args()
    if not args.receipt.is_absolute():raise ValueError('P38_ABSOLUTE_RECEIPT_REQUIRED')
    print(json.dumps(run(args.receipt) if args.mode=='train' else cold(args.receipt),sort_keys=True))


if __name__=='__main__':main()
