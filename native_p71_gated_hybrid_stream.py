"""P71 opt-in, exact-source-provenanced factual TTT write-gate engineering pilot.

Embeds the three four-float P70 *authored development* gate checkpoints. Applies
one to the original HybridPilot's TTT write seam via P64-K streaming. This does
NOT change the original model, attest factual sources, prove novel learning, or
claim transfer from P70's synthetic data. Read-only hypothetical previews keep
all original P64K bounds / revision / state guards.
"""
from __future__ import annotations
import hashlib

import torch
from torch.nn import functional as F
from model import HybridPilot
from native_p62_streaming_ttt import StreamingTTT, TTTFactualState
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot

# Validation-exact recovery receipt from P70 PR #178, not independently trained
# original-Hybrid weights. Key order: bias / log(error) / log(pred power) / cosine.
# SHA256 is over little-endian float32 tensor bytes in the above order.
GATE_CHECKPOINTS = {
    1000201: ((-0.9188888072967529, -0.7191678881645203,
               0.003978138789534569, 0.9039514660835266),
              'd703f692883fae50f578ef311dc7ceaa6af5a5ba5f51bd508dcbb4ada7cdaf59'),
    1000203: ((-1.1371526718139648, -0.7640940546989441,
               -0.031886711716651917, 0.9210875034332275),
              'd95402c824e932d74435edd634d5685c407fd8b7df4741169835ea1357bd700a'),
    1000207: ((-1.2033586502075195, -0.7765178084373474,
               -0.03707001730799675, 0.966083288192749),
              '1a460ab42e1c63dd3cc1f98a9d9944392113e9aa3043ca2391b445979ecdcb79'),
}


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_gate_checkpoint(seed: int, reference: torch.Tensor) -> torch.Tensor:
    """Fail closed on missing/tampered source identity, dtype or device."""
    if type(seed) is not int or seed not in GATE_CHECKPOINTS:
        raise ValueError('P71_UNKNOWN_GATE_CHECKPOINT')
    if (not isinstance(reference, torch.Tensor) or
            reference.dtype not in (torch.float32, torch.float64)):
        raise ValueError('P71_INVALID_REFERENCE')
    coeffs, checksum = GATE_CHECKPOINTS[seed]
    frozen = torch.tensor(coeffs, dtype=torch.float32)
    if _sha(frozen) != checksum:
        raise ValueError('P71_GATE_CHECKPOINT_SHA_MISMATCH')
    return frozen.to(dtype=reference.dtype, device=reference.device)


def factual_write_gate(pred: torch.Tensor, observed: torch.Tensor,
                       theta: torch.Tensor) -> torch.Tensor:
    """P70-equivalent gate features, with no future query, time or role input."""
    if (not isinstance(pred,torch.Tensor) or not isinstance(observed,torch.Tensor)
            or not isinstance(theta,torch.Tensor) or pred.ndim!=2
            or pred.shape!=observed.shape or pred.shape[1]<1
            or theta.shape!=(4,) or theta.dtype!=pred.dtype
            or observed.dtype!=pred.dtype or pred.device!=observed.device
            or theta.device!=pred.device or
            not bool(torch.isfinite(pred).all()) or
            not bool(torch.isfinite(observed).all()) or
            not bool(torch.isfinite(theta).all())):
        raise ValueError('P71_INVALID_FACTUAL_GATE_INPUT')
    residual = observed - pred
    error = residual.square().mean(dim=-1)
    energy = pred.square().mean(dim=-1)
    features = torch.stack((torch.ones_like(error), torch.log(error+1e-6),
                            torch.log(energy+1e-6),
                            F.cosine_similarity(pred,observed,dim=-1,eps=1e-6)), dim=-1)
    return torch.sigmoid(features @ theta)


class GatedStreamingTTT(StreamingTTT):
    """Identical P62 read-before-write state with an optional gated factual write."""
    def __init__(self, core, theta:torch.Tensor, max_confirmed_events:int=128,
                 *, mode:str='learned'):
        super().__init__(core, max_confirmed_events)
        if mode not in ('learned','all_write'):
            raise ValueError('P71_INVALID_GATE_MODE')
        if (not isinstance(theta,torch.Tensor) or theta.shape!=(4,) or
                theta.dtype!=core.key.weight.dtype or theta.device!=core.key.weight.device or
                not bool(torch.isfinite(theta).all())):
            raise ValueError('P71_INVALID_GATE_WEIGHTS')
        self.theta=theta.detach().clone()
        self.theta_sha256=_sha(self.theta)
        self.mode=mode

    def require_original_gate(self):
        if self.theta.shape!=(4,) or _sha(self.theta)!=self.theta_sha256:
            raise ValueError('P71_GATE_WEIGHTS_CHANGED_REBUILD_STATE')

    def ingest_confirmed(self, factual:torch.Tensor, state:TTTFactualState,
                         confirmed:torch.Tensor):
        self.require_original_gate()
        self._validate(factual,state)
        if (not isinstance(confirmed,torch.Tensor) or
                confirmed.shape!=factual.shape[:2] or
                confirmed.device!=factual.device or confirmed.dtype!=torch.bool or
                not bool(confirmed.all())):
            raise ValueError('P62_UNCONFIRMED_FACT_MUST_NOT_COMMIT')
        if not bool((state.confirmed_count+factual.shape[1]<=self.max_confirmed_events).all()):
            raise ValueError('P62_BOUNDED_FACTUAL_MEMORY_FULL')
        k,q,v=self._features(factual)
        fast=state.fast_weights
        reads=[]
        for t in range(factual.shape[1]):
            read=torch.bmm(fast,q[:,t,:,None]).squeeze(-1)
            reads.append(read)
            pred=torch.bmm(fast,k[:,t,:,None]).squeeze(-1)
            residual=v[:,t]-pred
            if self.mode=='all_write':
                gated=residual
            else:
                admission=factual_write_gate(pred,v[:,t],self.theta)
                gated=residual*admission[:,None]
            fast=fast+self.core.eta*(gated[:,:,None]*k[:,t,None,:])
        output=factual+self.core.proj(torch.stack(reads,dim=1))
        return output,TTTFactualState(fast,state.confirmed_count+factual.shape[1])

    def preview(self, candidates:torch.Tensor, state:TTTFactualState):
        self.require_original_gate()
        return super().preview(candidates,state)


class GatedStreamingHybridPilot(StreamingHybridPilot):
    """Existing four-block incremental adapter with P70-gated *factual* TTT writes.

    Does not change HybridPilot parameters. The gated policy and source mode are
    included in model_fingerprint. A different gate/mode cannot reuse the same
    SSM/KV/TTT factual carry, including after a cold restore.
    """
    def __init__(self, core:HybridPilot, *, seed:int=1000203,
                 mode:str='learned', max_confirmed_events:int=128):
        if not isinstance(core,HybridPilot) or core.ttt is None:
            raise TypeError('P71_REQUIRES_ORIGINAL_HYBRID')
        if mode not in ('learned','all_write'):
            raise ValueError('P71_INVALID_GATE_MODE')
        self._p71_seed=seed
        self._p71_mode=mode
        self._p71_theta=load_gate_checkpoint(seed,core.ttt.key.weight)
        self._p71_gate_signature=_sha(self._p71_theta)
        super().__init__(core,max_confirmed_events=max_confirmed_events)
        self.ttt=GatedStreamingTTT(core.ttt,self._p71_theta,max_confirmed_events,mode=mode)

    def _compute_model_fingerprint(self):
        base=super()._compute_model_fingerprint()
        message=('P71_GATED_TTT_V1|'+self._p71_mode+'|'+str(self._p71_seed)+'|'+
                 self._p71_gate_signature+'|'+base).encode('ascii')
        return hashlib.sha256(message).hexdigest()

    def _require_same_model_revision(self):
        super()._require_same_model_revision()
        if _sha(self._p71_theta)!=self._p71_gate_signature:
            raise ValueError('P71_GATE_WEIGHTS_CHANGED_REBUILD_STATE')
        if isinstance(self.ttt,GatedStreamingTTT):
            self.ttt.require_original_gate()
