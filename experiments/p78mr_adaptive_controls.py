"""P78-MR 256-float factual fast-memory controls; no data loader or OOD access.

Read each executed action before its observed effect is incorporated. Fixed
rules have no trainable weights; learned controls use the exact P77 equations,
but receive fresh P78-MR seeded parameters rather than consumed P77 weights.
"""
import itertools
import torch
import torch.nn.functional as F
from p76_memory_repair import _facts, _query
from p77_write_policy import WritePolicy

DIM = 16
STATE_FLOATS = 256
MODEL_SEEDS = (978201, 978203, 978207)
LOW = (0.1, 0.3, 0.5)
HIGH = (0.5, 0.75, 1.0)
COS_THRESH = (-0.25, 0.0, 0.25, 0.5)
RES_THRESH = (0.02, 0.1, 0.25, 0.5)
CONSTANT = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def new_learned(mode, seed):
    """Same P77 mechanism, disjoint P78 initial weights and no global RNG drift."""
    if seed not in MODEL_SEEDS or mode not in ('non_directional3', 'cosine4'):
        raise ValueError('P78MR_BAD_LEARNER')
    # P77's pinned module requires an old seed for construction only; overwrite
    # immediately with fresh P78-MR initialization before any forward call.
    model = WritePolicy(mode, seed=977201)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        model.theta.copy_((torch.rand(model.theta.shape, generator=gen)-0.5)*0.1)
    return model


def configurations():
    configs = [('constant', v, 0.0, 0.0) for v in CONSTANT]
    for low, high, threshold in itertools.product(LOW, HIGH, COS_THRESH):
        configs.append(('direction', low, high, threshold))
    for low, high, threshold in itertools.product(LOW, HIGH, RES_THRESH):
        configs.append(('residual', low, high, threshold))
    configs.append(('fixed_contradiction', 0.0, 0.0, 0.0))
    assert len(configs) == 83
    return tuple(configs)


def fixed_outputs(facts, query, configs=None, *, carry=None):
    """Simulate multiple separate equal-state controls in one vectorized batch.

    Returns (predictions[C,B,16], state[C,B,16,16]); each arm's persistence
    is EXACTLY 256 floats per world (never share the state across arms).
    """
    actions, effects = _facts(facts)
    b, steps, d = actions.shape
    configs = configurations() if configs is None else tuple(configs)
    if not configs or any(c not in configurations() for c in configs):
        raise ValueError('P78MR_BAD_FIXED_CONFIG')
    c = len(configs)
    query = _query(query, b, actions.dtype, actions.device)
    if carry is None:
        state = actions.new_zeros((c,b,d,d))
    else:
        if (not isinstance(carry, torch.Tensor) or carry.shape!=(c,b,d,d)
            or carry.dtype!=actions.dtype or carry.device!=actions.device
            or not bool(torch.isfinite(carry).all())):
            raise ValueError('P78MR_BAD_FIXED_CARRY')
        state=carry.clone()
    lo = actions.new_tensor([v[1] for v in configs])[:,None]
    hi = actions.new_tensor([v[2] for v in configs])[:,None]
    thresholds = actions.new_tensor([v[3] for v in configs])[:,None]
    kind = [v[0] for v in configs]
    direction = actions.new_tensor([1 if x=='direction' else 0 for x in kind])[:,None].bool()
    residual = actions.new_tensor([1 if x=='residual' else 0 for x in kind])[:,None].bool()
    contradiction = actions.new_tensor([1 if x=='fixed_contradiction' else 0 for x in kind])[:,None].bool()
    for t in range(steps):
        a=actions[:,t]
        y=effects[:,t]
        pred=torch.matmul(state,a[None,:,:,None]).squeeze(-1)
        error=y[None]-pred
        cosine=F.cosine_similarity(pred,y[None],dim=-1,eps=1e-6)
        confidence=pred.norm(dim=-1)/(y[None].norm(dim=-1)+1e-6)
        mse=error.square().mean(-1)
        direction_hi=(cosine < thresholds)
        residual_hi=(mse > thresholds)
        gain=torch.where(direction & direction_hi,hi,lo)
        gain=torch.where(residual & residual_hi,hi,gain)
        gain=torch.where(contradiction, 0.5+0.5*((cosine<0)&(confidence>.30)).to(actions.dtype),gain)
        state=state+gain[:,:,None,None]*error[:,:,:,None]*a[None,:,None,:]
    pred=torch.matmul(state,query[None,:,:,None]).squeeze(-1)
    return pred,state


def scored_errors(pred, target, changed):
    if pred.ndim!=3 or target.shape!=pred.shape[1:] or changed.shape!=(len(target),):
        raise ValueError('P78MR_BAD_SCORE')
    perworld=(pred-target[None]).square().mean(-1)
    return {
        'total':perworld.mean(-1).tolist(),
        'changed':perworld[:,changed].mean(-1).tolist(),
        'stable':perworld[:,~changed].mean(-1).tolist()
    }


def select_calibrated(configs, scores):
    """Frozen deterministic CAL-only min-total rule, never future VAL."""
    if len(configs)!=len(scores['total']):
        raise ValueError('P78MR_BAD_CAL_SCORES')
    # Input list is deterministic lexicographic grid in configurations().
    index=min(range(len(configs)),key=lambda i:(scores['total'][i],i))
    return index, configs[index]
