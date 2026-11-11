"""P75 conventional-memory auxiliary training, source-free engineering seam.

P75 freeze: experiments/p75_prequential_baseline_freeze.json.
For every executed action, predict the effect using ONLY facts with index < t.
The consequence at t becomes a loss label only after prediction, then a fact.
No role, phase, change-point, future query or P69/P75 OOD data is used.
This module has no FIT/VAL source accessor, trainer main or scientific runner.
"""
import torch
import torch.nn.functional as F
from p69_trainable_competitors import (
    TrainedGRU, DiagonalContinuousSSM, _facts, _events, _query, STATE, D,
)

MATCH_THRESHOLD = 0.85
AUX_WEIGHT = 0.50


def historical_match_mask(facts):
    """Only executed actions a[:t] determine if a[t] has a prior signed match.

    abs cosine handles opposite action signs; no role/effect labels are read.
    """
    actions, _ = _facts(facts)
    normal = F.normalize(actions, dim=-1)
    sims = torch.bmm(normal, normal.transpose(1, 2)).abs()
    prior = torch.ones_like(sims, dtype=torch.bool).tril(-1)
    return ((sims >= MATCH_THRESHOLD) & prior).any(-1).detach()


def before_write_predictions(model, facts):
    """Predict y[t] with state after t-1; final state follows all t facts.

    Passing an observed y[t] to the same step's prediction is forbidden.
    The only persistent memory is the original 256-float model state.
    """
    actions, effects = _facts(facts)
    b, steps, _ = actions.shape
    if steps < 1 or not isinstance(model, (TrainedGRU, DiagonalContinuousSSM)):
        raise ValueError('P75_BAD_MODEL_OR_EMPTY_HISTORY')
    events = _events(actions, effects)
    if isinstance(model, TrainedGRU):
        encoded, final = model.core(events)
        prior = torch.cat((encoded.new_zeros(b, 1, STATE), encoded[:, :-1]), dim=1)
        tail = final.squeeze(0)
    else:
        state = actions.new_zeros(b, STATE)
        decay = torch.sigmoid(model.logit_decay)
        series=[]
        for t in range(steps):
            series.append(state)
            state = decay * state + (1-decay) * torch.tanh(model.input(events[:,t]))
        prior = torch.stack(series,dim=1)
        tail = state
    pred = model.head(torch.cat((prior, actions),dim=-1))
    if pred.shape != (b,steps,D) or tail.shape != (b,STATE):
        raise ValueError('P75_STATE_PARITY_FAIL')
    return pred, tail


def losses(model, facts, final_action, actual_final_effect, mode):
    """Compute predeclared FIT-only loss; must not select on held-out data.

    The final supervised target and past effects are legitimate labels but
    MUST NOT enter the historical memory before final readout.
    """
    if mode not in ('final_only', 'matched_prequential'):
        raise ValueError('P75_UNKNOWN_MODE')
    actions, effects = _facts(facts)
    query = _query(final_action,len(actions),actions.dtype,actions.device)
    if actual_final_effect.shape != query.shape or not bool(torch.isfinite(actual_final_effect).all()):
        raise ValueError('P75_BAD_FINAL_EFFECT')
    pre, tail = before_write_predictions(model,facts)
    final=model.read(tail,query)
    final_loss=(final-actual_final_effect).square().mean()
    mask=historical_match_mask(facts)
    per_event=(pre-effects).square().mean(-1)
    auxiliary=(per_event*mask).sum()/mask.sum().clamp_min(1)
    if mode=='final_only':
        combined=final_loss
    else:
        combined=final_loss + AUX_WEIGHT*auxiliary
    return {'loss':combined,'final_loss':final_loss,
            'auxiliary_loss':auxiliary,'matched_events':int(mask.sum()),
            'world_state_floats':STATE}
