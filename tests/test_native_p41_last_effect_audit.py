"""P41 DEV-only P40 falsification: no new scientific source, no training.

The explicit 16-step script is a negative engineering control, NOT a
prospective held-out trial. The zero-parameter last-effect comparator sees
exactly the same factual actions/outcomes as P40; neither sees hidden modes.
"""
import io
import torch
from native_hidden_mode_belief import FactualModeInference, HiddenModeBelief


def scripted_trial():
    model=FactualModeInference(1,capacity=2,novelty=.1,noise=.05)
    sensor=torch.tensor([[.2]],dtype=torch.float32)
    actions=torch.tensor([[[-1.],[1.]]],dtype=torch.float32)
    dt=torch.ones((1,1),dtype=torch.float32)
    belief=model.initial_state(sensor)
    last_effect=None
    outcomes={'p40':0,'last_factual_effect':0,'fixed_action_minus':0,'n':0}
    records=[]
    modes=(1,)*4+(-1,)*4+(1,)*4+(-1,)*4
    goals=(1,-1)*8
    executed=(1,-1,-1,1)*4
    for step,(mode,goal_direction,taken) in enumerate(zip(modes,goals,executed)):
        goal=sensor+goal_direction*.25
        optimal=1 if mode*goal_direction>0 else 0
        if last_effect is not None:
            p40=int(model.choose(sensor,actions,dt,goal,belief)[0])
            baseline_prediction=sensor[:,None,:]+dt[:,None,:]*actions*last_effect[:,None,:]
            baseline=int((baseline_prediction-goal[:,None,:]).square().sum(-1).argmin(-1)[0])
            outcomes['p40']+=int(p40==optimal)
            outcomes['last_factual_effect']+=int(baseline==optimal)
            outcomes['fixed_action_minus']+=int(optimal==0)
            outcomes['n']+=1
            records.append((step,p40,baseline,optimal))
        action=torch.tensor([[float(taken)]])
        observed_next=sensor+(.30*mode)*action
        belief=model.observe(sensor,action,dt,observed_next,belief)
        last_effect=(observed_next-sensor)/(dt*action)
    return outcomes,records,belief


def test_last_effect_ties_p40_on_declared_hidden_mode_reversals():
    outcome,records,_=scripted_trial()
    assert outcome=={'p40':12,'last_factual_effect':12,'fixed_action_minus':8,'n':15}
    assert all(x[1]==x[2] for x in records)
    assert sum(x[1]!=x[3] for x in records)==3


def test_hidden_reversal_not_predictable_before_new_factual_feedback():
    _,records,_=scripted_trial()
    assert [step for step,pred,baseline,true in records if pred!=true]==[4,8,12]
    assert all(pred==baseline for _,pred,baseline,_ in records)


def test_cold_belief_receipt_is_bit_exact_without_optimizer():
    _,_,state=scripted_trial()
    payload=io.BytesIO()
    torch.save({'effects':state.effects,'observations':state.observations,
                'probabilities':state.probabilities,'factual_steps':state.factual_steps},payload)
    payload.seek(0)
    restored=HiddenModeBelief(**torch.load(payload,weights_only=True))
    for name in ('effects','observations','probabilities','factual_steps'):
        assert torch.equal(getattr(restored,name),getattr(state,name))
