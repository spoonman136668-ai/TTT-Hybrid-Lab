"""G7-S27 target-origin ablations: synthetic engineering only; not scientific first use.

Original raw-target KRR remains untouched. These are NEW independently registered
control candidates. No scientific seeds, simulator, optimizers or heldout data.
"""
from __future__ import annotations
from typing import Literal
import torch
from torch import Tensor

Mode = Literal["train_mean_centered","current_sensor_residual"]
GAMMAS=(0.5,2.0,6.0)
RIDGES=(0.01,0.08)

class DiagnosticInvalid(ValueError):
    pass

def check(ok,code):
    if not bool(ok):
        raise DiagnosticInvalid(code)

def numeric_vectors(h:Tensor,q:Tensor)->Tensor:
    check(isinstance(h,Tensor) and isinstance(q,Tensor),"G7_S27_NOT_NUMERIC_TENSOR")
    check(h.ndim==3 and tuple(h.shape[1:])==(32,4),"G7_S27_INVALID_CAUSAL_HISTORY")
    check(q.ndim==3 and q.shape==(h.shape[0],2,3),"G7_S27_INVALID_PAIRED_QUERY")
    check(h.device.type=="cpu" and q.device.type=="cpu" and
          bool(torch.isfinite(h).all()) and bool(torch.isfinite(q).all()),
          "G7_S27_NONFINITE_OR_NONCPU_CONTEXT")
    check(bool(torch.equal(q[:,0,0],q[:,1,0]) and
               torch.equal(q[:,0,2],q[:,1,2]) and
               torch.all(q[:,0,1]==-0.8) and torch.all(q[:,1,1]==0.8)),
          "G7_S27_ACTION_PROBE_OR_SAME_STATE_DRIFT")
    return torch.cat((h.flatten(1)[:,None].expand(-1,2,-1),q),dim=-1).reshape(-1,131).double()

def target_values(q:Tensor,y:Tensor)->Tensor:
    check(isinstance(y,Tensor) and y.shape==q.shape[:2] and
          y.device.type=="cpu" and bool(torch.isfinite(y).all()),
          "G7_S27_INVALID_TRAIN_OR_VALIDATION_TARGET")
    return y.reshape(-1,1).double()

def rbf(x:Tensor,z:Tensor,gamma:float)->Tensor:
    return torch.exp(-gamma*torch.cdist(x,z).square()/x.shape[-1])

class TargetAdjustedKRR:
    """Six-point original gamma/ridge grid, new train-only target-origin ablation."""
    def __init__(self,mode:Mode):
        check(mode in ("train_mean_centered","current_sensor_residual"),
              "G7_S27_UNREGISTERED_KRR_TARGET_MODE")
        self.mode=mode
        self.mean=None
        self.scale=None
        self.train_vectors=None
        self.weights=None
        self.target_mean=None
        self.gamma=None
        self.ridge=None
        self.validation_mse=None

    def origin(self,q:Tensor)->Tensor:
        if self.mode=="current_sensor_residual":
            return q[:,:,0].reshape(-1,1).double()
        check(self.target_mean is not None,"G7_S27_MEAN_ORIGIN_MISSING_FROM_FIT")
        return self.target_mean.expand(q.shape[0]*q.shape[1],1)

    def fit_calibrated(self,train:tuple[Tensor,Tensor,Tensor],
                       validation:tuple[Tensor,Tensor,Tensor])->dict:
        h,q,y=train; vh,vq,vy=validation
        x=numeric_vectors(h,q); vx=numeric_vectors(vh,vq)
        labels=target_values(q,y); valid=target_values(vq,vy)
        check(16<=x.shape[0]<=512 and vx.shape[0]>=16,
              "G7_S27_INVALID_TRAIN_VALIDATION_OBSERVATIONS")
        self.mean=x.mean(0)
        self.scale=x.std(0,unbiased=False).clamp_min(1e-4)
        self.target_mean=labels.mean().reshape(1,1)
        residual=labels-self.origin(q)
        nx=(x-self.mean)/self.scale
        nv=(vx-self.mean)/self.scale
        eye=torch.eye(x.shape[0],dtype=torch.double)
        best=None
        for gamma in GAMMAS:
            gram=rbf(nx,nx,gamma)
            probe=rbf(nv,nx,gamma)
            for ridge in RIDGES:
                weights=torch.linalg.solve(gram+ridge*eye,residual)
                mse=float((probe@weights+self.origin(vq)-valid).square().mean())
                if best is None or mse<best[0]:
                    best=(mse,gamma,ridge,weights.detach().clone())
        assert best is not None
        self.validation_mse,self.gamma,self.ridge,self.weights=best
        self.train_vectors=nx.detach().clone()
        return {"validation_mse":self.validation_mse,"gamma":self.gamma,
                "ridge":self.ridge,"train_query_samples":int(x.shape[0]),
                "mode":self.mode}

    @torch.no_grad()
    def predict(self,h:Tensor,q:Tensor)->Tensor:
        check(self.train_vectors is not None and self.weights is not None and
              self.mean is not None and self.scale is not None and
              self.gamma is not None and self.ridge is not None,
              "G7_S27_UNTRAINED_CANDIDATE_CONTROL")
        nx=(numeric_vectors(h,q)-self.mean)/self.scale
        prediction=rbf(nx,self.train_vectors,self.gamma)@self.weights
        return (prediction+self.origin(q)).reshape(q.shape[:2]).float()

def five_check_spread(values:tuple[float,...])->tuple[float,...]:
    """Original S26 strict criterion as diagnostic only; no requalification."""
    check(len(values)>=5 and all(isinstance(v,float) and
          v>0 and bool(torch.isfinite(torch.tensor(v))) for v in values),
          "G7_S27_INVALID_GRU_PLATEAU_RECEIPT")
    return tuple((max(values[i:i+5])-min(values[i:i+5]))/
                 max(min(values[i:i+5]),1e-12)
                 for i in range(len(values)-4))

def synthetic_paired_world(n:int,*,shift:float=0.)->tuple[Tensor,Tensor,Tensor]:
    """Invented numeric engineering fixture. NO real/science seed."""
    check(type(n) is int and 8<=n<=48,"G7_S27_SYNTHETIC_SIZE")
    sensor=.05*torch.arange(n,dtype=torch.float64)+shift
    h=torch.zeros((n,32,4),dtype=torch.float64)
    h[:,0,0]=.2*torch.arange(n,dtype=torch.float64)/n
    h[:,0,3]=sensor
    q=torch.zeros((n,2,3),dtype=torch.float64)
    q[:,:,0]=sensor[:,None]
    q[:,:,1]=torch.tensor([-.8,.8],dtype=torch.float64)[None,:]
    q[:,:,2]=.1
    y=q[:,:,0]+.10*q[:,:,1]+.025
    return h,q,y
