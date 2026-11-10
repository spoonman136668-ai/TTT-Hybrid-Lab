"""P50 DEV: raw continuous factual state -> SSM/attention/TTT/depth -> action prediction.

A bounded tensor-only history contains ONLY completed physical sensor/action/
next-sensor/time observations. An unobserved candidate is appended as a tagged
query with ZERO future outcome and run through a fresh, per-branch causal pilot.
It never updates persistent history. The reference SSM, TTT and depth modules
are NOT official fused Mamba2/TTT-E2E/fixed-budget MoD. No actual scientific
causal/generalization/retention benefit is demonstrated by model composition.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from model import Configuration, HybridPilot


@dataclass(frozen=True)
class FactualStreamState:
    sensors: torch.Tensor       # [batch, capacity, sensor_dim] pre-action observation
    deltas: torch.Tensor        # [batch, capacity, sensor_dim] realized next - pre
    actions: torch.Tensor       # [batch, capacity, 1] actual executed action
    elapsed_steps: torch.Tensor # [batch, capacity, 1] actual elapsed dt > 0
    observations: torch.Tensor  # [batch] total completed actions
    cursor: torch.Tensor        # [batch] next ring slot
    elapsed_total: torch.Tensor # [batch, 1] total completed physical time


@dataclass(frozen=True)
class FactualStreamPreview:
    predictions: torch.Tensor   # [batch,candidates,sensor_dim] hypothetical next sensor
    identified: torch.Tensor    # [batch] factual *linear design rank*, not true causality
    facts_used: torch.Tensor    # [batch] bounded history facts used


class NativeFactualHybridStream(nn.Module):
    """No tokenizer, token objectives or source oracle in native cognitive path.

    Recomputes the small reference SSM/TTT causal history from actual ring
    tensors on each candidate query. This is slow O(B*K*window) *network
    passes*, not an efficient persistent fast-weight implementation. The depth
    block may update nonpersistent utilization telemetry during forward only.
    """
    def __init__(self, sensor_dim: int, capacity: int = 8, *,
                 cfg: Configuration | None = None, min_facts: int = 3):
        super().__init__()
        if type(sensor_dim) is not int or sensor_dim < 1:
            raise ValueError('P50_SENSOR_DIM_POSITIVE_INT_REQUIRED')
        if type(capacity) is not int or not 3 <= capacity <= 64:
            raise ValueError('P50_CAPACITY_OUT_OF_RANGE')
        if type(min_facts) is not int or not 3 <= min_facts <= capacity:
            raise ValueError('P50_MIN_FACTS_OUT_OF_RANGE')
        if cfg is None:
            cfg = Configuration(width=16, heads=2)
        if not isinstance(cfg,Configuration) or cfg.width < 2 or cfg.width % cfg.heads:
            raise ValueError('P50_BAD_CONFIG')
        self.sensor_dim = sensor_dim
        self.capacity = capacity
        self.min_facts = min_facts
        self.pilot = HybridPilot(2*sensor_dim+3,sensor_dim,cfg)

    def _analog(self, x: torch.Tensor, expected: tuple[int | None,...], role: str):
        if (not isinstance(x,torch.Tensor) or x.ndim != len(expected)
            or any(v is not None and x.shape[i]!=v for i,v in enumerate(expected))
            or x.shape[0] < 1 or x.dtype not in (torch.float32,torch.float64)
            or not bool(torch.isfinite(x).all())):
            raise ValueError('P50_INVALID_'+role)

    def _model(self,*xs):
        p=next(self.pilot.parameters())
        if any(x.device!=p.device or x.dtype!=p.dtype for x in xs):
            raise ValueError('P50_MODEL_DEVICE_DTYPE_DRIFT')

    def initial_state(self,sensor: torch.Tensor)->FactualStreamState:
        self._analog(sensor,(None,self.sensor_dim),'SENSOR')
        self._model(sensor)
        b=sensor.shape[0]
        zero=sensor.new_zeros
        ints=lambda:torch.zeros((b,),dtype=torch.int64,device=sensor.device)
        return FactualStreamState(
            zero((b,self.capacity,self.sensor_dim)),
            zero((b,self.capacity,self.sensor_dim)),
            zero((b,self.capacity,1)),zero((b,self.capacity,1)),
            ints(),ints(),zero((b,1)))

    def _check(self,sensor:torch.Tensor,state:FactualStreamState):
        self._analog(sensor,(None,self.sensor_dim),'SENSOR')
        self._model(sensor)
        if not isinstance(state,FactualStreamState):
            raise ValueError('P50_FACTUAL_STATE_REQUIRED')
        b=sensor.shape[0]; c=self.capacity; d=self.sensor_dim
        shapes=((b,c,d),(b,c,d),(b,c,1),(b,c,1),(b,),(b,),(b,1))
        vals=(state.sensors,state.deltas,state.actions,state.elapsed_steps,
              state.observations,state.cursor,state.elapsed_total)
        if any(not isinstance(x,torch.Tensor) or x.shape!=sz or x.device!=sensor.device
               for x,sz in zip(vals,shapes)):
            raise ValueError('P50_STATE_SHAPE_OR_DEVICE_DRIFT')
        if (any(x.dtype!=sensor.dtype for x in vals[:4]+(state.elapsed_total,))
            or state.observations.dtype!=torch.int64 or state.cursor.dtype!=torch.int64):
            raise ValueError('P50_STATE_DTYPE_DRIFT')
        if (not all(bool(torch.isfinite(x).all()) for x in vals[:4]+(state.elapsed_total,))
            or not bool((state.observations>=0).all())
            or not bool(((state.cursor>=0)&(state.cursor<c)).all())
            or not bool((state.cursor==state.observations%c).all())
            or not bool((state.elapsed_total>=0).all())):
            raise ValueError('P50_STATE_VALUES_DRIFT')
        for row in range(b):
            count=min(int(state.observations[row]),c)
            for i in range(count):
                j=(int(state.cursor[row])-count+i)%c
                if not bool((state.elapsed_steps[row,j,0]>0)):
                    raise ValueError('P50_STATE_INVALID_FACTUAL_EVENT')

    def _time(self,sensor,dt):
        self._analog(dt,(sensor.shape[0],1),'ELAPSED_TIME')
        self._model(dt)
        if not bool((dt>0).all()):
            raise ValueError('P50_NONPOSITIVE_ELAPSED_TIME')

    def _actions(self,sensor,actions,*,candidate=False):
        if candidate:
            self._analog(actions,(sensor.shape[0],None,1),'CANDIDATES')
            if not 1<=actions.shape[1]<=16:
                raise ValueError('P50_CANDIDATE_LIMIT')
        else:
            self._analog(actions,(sensor.shape[0],1),'EXECUTED_ACTION')
        self._model(actions)
        # A completed neutral motor command is valid physical evidence.
        # Empty ring slots are distinguished by factual count and elapsed dt;
        # zero-only or confounded histories remain unidentifiable by rank.

    def _chronology(self,state,row):
        n=min(int(state.observations[row]),self.capacity)
        indices=[(int(state.cursor[row])-n+i)%self.capacity for i in range(n)]
        return n,indices

    def _rank_ready(self,state,row,indices):
        if len(indices)<self.min_facts:
            return False
        actions=state.actions[row,indices,0].double()
        times=state.elapsed_steps[row,indices,0].double().cumsum(0)
        design=torch.stack((torch.ones_like(times),times,actions),dim=-1)
        # Algebraic check is *necessary*, not sufficient for causal validity.
        singular=torch.linalg.svdvals(design)
        return bool(singular[-1] > torch.maximum(singular[0]*1e-6,
                                                  singular.new_tensor(1e-8)))

    def observe(self,sensor:torch.Tensor,executed_action:torch.Tensor,
                dt:torch.Tensor,actual_next:torch.Tensor,
                state:FactualStreamState)->FactualStreamState:
        """Only caller-witnessed completed physical actions enter durable memory."""
        self._check(sensor,state)
        self._actions(sensor,executed_action)
        self._time(sensor,dt)
        self._analog(actual_next,tuple(sensor.shape),'FACTUAL_NEXT')
        self._model(actual_next)
        total=state.elapsed_total+dt
        if not bool(torch.isfinite(total).all()):
            raise ValueError('P50_NONFINITE_FACTUAL_CLOCK')
        sensors=state.sensors.clone(); deltas=state.deltas.clone()
        actions=state.actions.clone(); steps=state.elapsed_steps.clone()
        row=torch.arange(sensor.shape[0],device=sensor.device)
        slot=state.cursor
        sensors[row,slot]=sensor.detach()
        deltas[row,slot]=(actual_next-sensor).detach()
        actions[row,slot]=executed_action.detach()
        steps[row,slot]=dt.detach()
        return FactualStreamState(sensors,deltas,actions,steps,
                                  state.observations+1,(slot+1)%self.capacity,total.detach())

    def preview(self,sensor:torch.Tensor,candidates:torch.Tensor,dt:torch.Tensor,
                state:FactualStreamState)->FactualStreamPreview:
        """Branch-local prediction; candidate has no actual observed outcome.

        Each candidate gets an independent deterministic read-before-write TTT
        episode; hypothetical fast weights are discarded, not checkpointed.
        """
        self._check(sensor,state)
        self._actions(sensor,candidates,candidate=True)
        self._time(sensor,dt)
        outputs=[];ready=[];counts=[]
        for b in range(sensor.shape[0]):
            n,ids=self._chronology(state,b)
            ready.append(self._rank_ready(state,b,ids))
            counts.append(n)
            observed=[]
            for j in ids:
                observed.append(torch.cat((state.sensors[b,j],state.deltas[b,j],
                     state.actions[b,j],state.elapsed_steps[b,j],
                     sensor.new_zeros((1,))),dim=0))
            by_action=[]
            for k in range(candidates.shape[1]):
                probe=torch.cat((sensor[b],sensor.new_zeros((self.sensor_dim,)),
                      candidates[b,k],dt[b],sensor.new_ones((1,))),dim=0)
                series=torch.stack(observed+[probe],dim=0).unsqueeze(0)
                raw=self.pilot(series,ttt_update=True)[:, -1, :]
                by_action.append((sensor[b:b+1]+raw).squeeze(0))
            outputs.append(torch.stack(by_action,dim=0))
        pred=torch.stack(outputs,dim=0)
        if not bool(torch.isfinite(pred).all()):
            raise ValueError('P50_NONFINITE_PREDICTION')
        return FactualStreamPreview(pred,
            torch.tensor(ready,dtype=torch.bool,device=sensor.device),
            torch.tensor(counts,dtype=torch.int64,device=sensor.device))

    def select(self,sensor:torch.Tensor,candidates:torch.Tensor,dt:torch.Tensor,
               external_goal:torch.Tensor,state:FactualStreamState)->torch.Tensor:
        preview=self.preview(sensor,candidates,dt,state)
        self._analog(external_goal,tuple(sensor.shape),'EXTERNAL_GOAL')
        self._model(external_goal)
        if not bool(preview.identified.all()):
            raise ValueError('P50_FACTUAL_ACTION_EFFECT_NOT_IDENTIFIABLE')
        return ((preview.predictions-external_goal[:,None,:])**2).sum(-1).argmin(-1)
