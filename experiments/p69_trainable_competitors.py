"""P69 FIT/VAL-only trainable memory competitors; no study entrypoint.

Frozen pre-training recipe: experiments/p69_competitor_training_recipe.json.
Accept factual (executed_action, observed_effect) streams only. The query is
used at readout, never during writes. There are no OOD paths or source seeds.
Engineering unit tests use tiny toy streams, not qualified science data.
"""
from __future__ import annotations

import copy
import hashlib
import time

import torch
from torch import nn
from torch.nn import functional as F

D = 16
STATE = 256
FIT_SEED = 981299
MODEL_SEEDS = (981201, 981203, 981207)
UPDATES = 600
CHECK_EVERY = 50
BATCH = 32


def _facts(facts):
    if not isinstance(facts, dict) or set(facts) != {'action', 'effect'}:
        raise ValueError('P69_HISTORICAL_FACTS_ONLY')
    a, y = facts['action'], facts['effect']
    if (not isinstance(a, torch.Tensor) or not isinstance(y, torch.Tensor) or
            a.ndim != 3 or a.shape != y.shape or a.shape[-1] != D or
            a.dtype != y.dtype or a.device != y.device or not a.is_floating_point()):
        raise ValueError('P69_BAD_FACTS')
    if not bool(torch.isfinite(a).all() and torch.isfinite(y).all()):
        raise ValueError('P69_NONFINITE_FACTS')
    if not bool((a.norm(dim=-1) > 1e-8).all()):
        raise ValueError('P69_ZERO_ACTION')
    return a, y


def _query(q, batch, dtype, device):
    if (not isinstance(q, torch.Tensor) or q.shape != (batch, D) or
            q.dtype != dtype or q.device != device or
            not bool(torch.isfinite(q).all())):
        raise ValueError('P69_BAD_QUERY')
    if not bool((q.norm(dim=-1)>1e-8).all()):
        raise ValueError('P69_ZERO_QUERY')
    return q


def _events(a,y):
    return torch.cat((a, y, torch.ones_like(a[:, :, :1])), dim=-1)


class ResidualTTTGate(nn.Module):
    """Frozen P69 four-parameter residual-gated 16x16 factual fast state."""
    def __init__(self, seed):
        super().__init__()
        if seed not in MODEL_SEEDS:
            raise ValueError('P69_UNREGISTERED_MODEL_SEED')
        rng = torch.Generator(device='cpu').manual_seed(seed)
        self.theta = nn.Parameter((torch.rand(4, generator=rng)-0.5)*0.1)

    def encode(self, facts, carry=None):
        a, y = _facts(facts)
        b = a.shape[0]
        w = a.new_zeros((b,D,D)) if carry is None else carry
        if (w.shape != (b,D,D) or w.dtype != a.dtype or w.device != a.device or
                not bool(torch.isfinite(w).all())):
            raise ValueError('P69_BAD_CARRY')
        for i in range(a.shape[1]):
            x = a[:,i]
            pred = (w @ x.unsqueeze(-1)).squeeze(-1)
            residual = y[:,i]-pred
            features = torch.stack((
                torch.ones(b, dtype=a.dtype, device=a.device),
                torch.log(residual.square().mean(-1)+1e-6),
                torch.log(pred.square().mean(-1)+1e-6),
                F.cosine_similarity(pred,y[:,i],dim=-1,eps=1e-6)
            ),dim=-1)
            weight = torch.sigmoid(features @ self.theta)
            w = w + 0.5*weight[:,None,None]*residual[:,:,None]*x[:,None,:]
        return w

    def read(self, state, query):
        _query(query, len(state), state.dtype, state.device)
        return (state @ query.unsqueeze(-1)).squeeze(-1)

    def forward(self, facts, query, carry=None):
        return self.read(self.encode(facts,carry),query)


class TrainedGRU(nn.Module):
    """Conventional 256-float recurrent-state GRU and query-conditioned head."""
    def __init__(self):
        super().__init__()
        self.core = nn.GRU(33,STATE,batch_first=True)
        self.head = nn.Linear(STATE+D,D)

    def encode(self, facts, carry=None):
        a,y = _facts(facts)
        if carry is not None and (carry.shape != (len(a),STATE) or
                carry.dtype != a.dtype or carry.device != a.device):
            raise ValueError('P69_BAD_CARRY')
        hidden = None if carry is None else carry.unsqueeze(0)
        _, final = self.core(_events(a,y),hidden)
        return final.squeeze(0)

    def read(self, state, query):
        q = _query(query,len(state),state.dtype,state.device)
        return self.head(torch.cat((state,q),-1))

    def forward(self,facts,query,carry=None):
        return self.read(self.encode(facts,carry),query)


class DiagonalContinuousSSM(nn.Module):
    """Trainable diagonal stable continuous recurrent state, 256 floats/world."""
    def __init__(self):
        super().__init__()
        self.logit_decay = nn.Parameter(torch.full((STATE,),2.0))
        self.input = nn.Linear(33,STATE)
        self.head = nn.Linear(STATE+D,D)

    def encode(self, facts, carry=None):
        a,y = _facts(facts)
        b = len(a)
        if carry is not None and (carry.shape != (b,STATE) or
                carry.dtype != a.dtype or carry.device != a.device or
                not bool(torch.isfinite(carry).all())):
            raise ValueError('P69_BAD_CARRY')
        state = a.new_zeros(b,STATE) if carry is None else carry
        decay = torch.sigmoid(self.logit_decay)
        events = _events(a,y)
        for t in range(events.shape[1]):
            state = decay*state + (1-decay)*torch.tanh(self.input(events[:,t]))
        return state

    def read(self,state,query):
        q = _query(query,len(state),state.dtype,state.device)
        return self.head(torch.cat((state,q),-1))

    def forward(self,facts,query,carry=None):
        return self.read(self.encode(facts,carry),query)


def model_from_recipe(kind,seed):
    if seed not in MODEL_SEEDS:
        raise ValueError('P69_UNREGISTERED_MODEL_SEED')
    if kind == 'learned_gate':
        return ResidualTTTGate(seed)
    if kind not in ('gru','ssm'):
        raise ValueError('P69_BAD_MODEL_KIND')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TrainedGRU() if kind == 'gru' else DiagonalContinuousSSM()


def parameter_count(model):
    return sum(x.numel() for x in model.parameters())


def sample_indices(n_fit,steps=UPDATES,batch=BATCH):
    if n_fit <= 0 or steps < 1 or steps > UPDATES or batch != BATCH:
        raise ValueError('P69_BAD_TRAIN_SPEC')
    g = torch.Generator(device='cpu').manual_seed(FIT_SEED)
    return torch.randint(n_fit,(steps,batch),generator=g)


def _public_batch(data,rows):
    # No phase/role/change labels propagate to the learner: only historical facts
    # at encode and final action at read. Evaluation target never goes to encode.
    facts={k:data[k][rows] for k in ('action','effect')}
    return facts,data['query_action'][rows],data['observed_query_effect'][rows]


def validation_mse(model,val,chunk=32):
    model.eval()
    scores=[]
    with torch.no_grad():
        for start in range(0,len(val['action']),chunk):
            rows=slice(start,start+chunk)
            facts,query,target=_public_batch(val,rows)
            prediction=model(facts,query)
            scores.append((prediction-target).square().mean(-1))
    return float(torch.cat(scores).mean())


def zero_mse(val):
    return float(val['observed_query_effect'].square().mean())


def check_competency(history,zero,kind):
    values=[float(h['mse']) for h in history]
    best=min(values) if values else float('inf')
    spread=max(values[-3:])-min(values[-3:]) if len(values)>=3 else float('inf')
    threshold=(.85 if kind=='learned_gate' else .75)*zero
    return {'best_val_mse':best,'threshold_mse':threshold,'recent_three_spread':spread,
            'competent':bool(best<=threshold and (kind=='learned_gate' or spread<=.01)),
            'converged':bool(spread<=.01),'checkpoints':len(values)}


def fit_val_only(model,fit,val,kind,steps=UPDATES):
    """Project-owner admission required for full FIT training; never called at import.

    No data split selector exists; caller must supply authorized FIT/VAL objects.
    Does not generate or score any source, OOD or counterfactual outcomes.
    """
    if kind not in ('learned_gate','gru','ssm') or steps < 1 or steps>UPDATES:
        raise ValueError('P69_BAD_TRAIN_SPEC')
    ids=sample_indices(len(fit['action']),steps)
    opt=torch.optim.AdamW(model.parameters(),lr=.005 if kind=='learned_gate' else .001,
                          weight_decay=.01)
    best=float('inf')
    best_state=None
    checkpoints=[]
    bad=0
    started=time.perf_counter()
    for step,idx in enumerate(ids):
        model.train()
        facts,query,target=_public_batch(fit,idx)
        opt.zero_grad(set_to_none=True)
        pred=model(facts,query)
        loss=(pred-target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError('P69_NONFINITE_TRAIN')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if (step+1)%CHECK_EVERY==0:
            mse=validation_mse(model,val)
            checkpoints.append({'update':step+1,'mse':mse})
            if mse<best:
                best=mse
                best_state=copy.deepcopy(model.state_dict())
                bad=0
            else:
                bad+=1
            if step+1>=200 and bad>=4 and check_competency(checkpoints,zero_mse(val),kind)['converged']:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {'kind':kind,'updates':step+1,'checkpoints':checkpoints,
            'sample_index_sha256':hashlib.sha256(ids[:step+1].numpy().tobytes()).hexdigest(),
            'params':parameter_count(model),'memory_state_floats':STATE,
            'wall_seconds':time.perf_counter()-started,
            'eligibility':check_competency(checkpoints,zero_mse(val),kind),
            'science_pass':False,'ood_accessed':False}
