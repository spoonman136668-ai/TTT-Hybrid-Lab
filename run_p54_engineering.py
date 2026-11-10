"""Bounded authored engineering demonstration using published reference mechanics.

NOT an authentic installed Gymnasium execution or owner scientific first use.
"""
from __future__ import annotations
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import torch
from model import Configuration
from native_factual_hybrid_stream import NativeFactualHybridStream
from native_p54_mountaincar_factual_bridge import NativeMountainCarFactualBridge,published_step

def run():
    torch.set_num_threads(1)
    torch.manual_seed(54054)
    native=NativeFactualHybridStream(2,capacity=8,cfg=Configuration(width=16,heads=2)).eval()
    bridge=NativeMountainCarFactualBridge(native,max_steps=8)
    seeds=tuple(range(5405400,5405464))
    state=bridge.start(seeds)
    sampler=np.random.default_rng(54054)
    actions=sampler.choice(np.array([-1.,1.],dtype=np.float32),size=(8,len(seeds)))
    started=time.perf_counter()
    for a in actions:
        state=bridge.actual_step(state,torch.tensor(a[:,None]))
    candidates=torch.tensor([[[-1.],[1.]]]).expand(len(seeds),-1,-1)
    with torch.no_grad():
        before={k:v.detach().clone() for k,v in native.state_dict().items()}
        p=bridge.preview(state,candidates)
        p_again=bridge.preview(state,candidates)
        assert torch.equal(p.predictions,p_again.predictions)
        assert all(torch.equal(v,before[k]) for k,v in native.state_dict().items())
    outcome=torch.tensor(np.stack([
        np.stack([published_step(state.physical[b].numpy(),a).next_sensor for a in (-1.,1.)])
        for b in range(len(seeds))]),dtype=p.predictions.dtype)
    mse=float((p.predictions-outcome).square().mean())
    rank=sum(bool(x) for x in p.identified)
    result={
        'schema':'ttt-hybrid.p54-pinned-mountaincar-factual-eng-demo.v1',
        'source':'published equation transcription; NOT official upstream runtime',
        'seed_prefix':5405400,'source_seed':54054,'episodes':len(seeds),
        'completed_physical_interventions':int(state.steps.sum()),
        'rank_certified_streams':rank,
        'untrained_native_two_action_next_sensor_mse':mse,
        'prediction_is_reproducible':bool(torch.equal(p.predictions,p_again.predictions)),
        'all_history_commits_completed_only':bool((state.memory.observations==state.steps).all()),
        'no_model_training_or_adaptation':True,
        'original_analytical_mechanics_oracle_MSE':0.,
        'candidate_model_parameters':sum(q.numel() for q in native.parameters()),
        'torch':torch.__version__,'python':platform.python_version(),
        'time_seconds':time.perf_counter()-started,
        'action_schedule_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),
        'official_gymnasium_exact_runtime_parity':'NOT_VERIFIED_PACKAGE_UNAVAILABLE',
        'scientific_result':'NOT_TESTED',
    }
    Path('experiments/p54_physics_reference_eng_demo.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':run()
