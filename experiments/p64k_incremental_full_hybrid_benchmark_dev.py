"""P64-K frozen matched CPU efficiency check: incremental state vs full-prefix replay.

NO physical source, scientific first use, model improvement or GPU claim.
Source models are the original P64-S reference and its distinct incremental cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import torch
from model import HybridPilot, Configuration
from native_p64s_full_hybrid_stream import FullHybridFactualStream
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot

SOURCE_WEIGHTS_SEED=711
SOURCE_DATA_SEEDS=(713,719)
LENGTHS=(12,24,40,60)
BATCH=2
CANDIDATES=2
REPEATS=8
WARMUPS=2
MAX_ERROR=1e-6


def execute(mode,core,factual,queries):
    n=len(factual[0])
    if mode=='incremental':
        adapter=StreamingHybridPilot(core,max_confirmed_events=64)
    elif mode=='prefix_recompute':
        adapter=FullHybridFactualStream(core,max_confirmed=64)
    else:raise ValueError('P64K_UNKNOWN_ARM')
    state=adapter.initial_state(factual[:,0,:])
    output=[]
    start=time.perf_counter()
    with torch.no_grad():
        for j in range(n):
            z,state=adapter.ingest_confirmed(factual[:,j:j+1,:],state,
                torch.ones((BATCH,1),dtype=torch.bool))
            output.append(z)
        preview=adapter.preview(queries,state)
    wall=time.perf_counter()-start
    return torch.cat(output,dim=1),preview,wall


def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SOURCE_WEIGHTS_SEED)
    core=HybridPilot(5,2,Configuration(width=8,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    results=[]
    for data_seed in SOURCE_DATA_SEEDS:
        for n in LENGTHS:
            torch.manual_seed(data_seed*100+n)
            factual=torch.randn(BATCH,n,5)
            queries=torch.randn(BATCH,CANDIDATES,5)
            source_sha=hashlib.sha256(factual.numpy().tobytes()+queries.numpy().tobytes()).hexdigest()
            measurements={'incremental':[],'prefix_recompute':[]}
            maximum=0.
            for rep in range(REPEATS):
                # Swap call order to reduce one-sided thermal/cache/order effects.
                order=('incremental','prefix_recompute') if rep%2==0 else ('prefix_recompute','incremental')
                obs={}
                for name in order:
                    a,b,t=execute(name,core,factual,queries)
                    measurements[name].append(t)
                    obs[name]=(a,b)
                maximum=max(maximum,float((obs['incremental'][0]-obs['prefix_recompute'][0]).abs().max()),
                            float((obs['incremental'][1]-obs['prefix_recompute'][1]).abs().max()))
            inc=statistics.median(measurements['incremental'][WARMUPS:])
            replay=statistics.median(measurements['prefix_recompute'][WARMUPS:])
            ratio=replay/inc
            results.append({'data_seed':data_seed,'n_facts':n,'batches':BATCH,'candidates':CANDIDATES,
                'source_sha256':source_sha,'incremental_median_seconds':inc,'prefix_recompute_median_seconds':replay,
                'speedup_x':ratio,'max_abs_numeric_difference':maximum,
                'numeric_parity':maximum<=MAX_ERROR,'speed_40_pass':n!=40 or ratio>=1.5,
                'speed_60_pass':n!=60 or ratio>=2.0})
    return {'schema':'ttt-hybrid.p64k-original-full-incremental-cache-benchmark.v1',
        'scope':'FROZEN_LOCAL_CPU_ENGINEERING_ONLY_NOT_SCIENTIFIC',
        'original_model_sha':'5d9ea57b827aa1796548447868fa5ee14c251b49',
        'original_prefix_recompute_blob':'56f4d566db9375e4b27e7d9eabcaf67498c25d94',
        'incremental_source_blob':'440aa8717e1aef15d7547be669dfa8d0f4da666a',
        'torch':torch.__version__,'all_pass':all(r['numeric_parity'] and r['speed_40_pass'] and r['speed_60_pass'] for r in results),
        'parameters_original_unchanged':sum(p.numel() for p in core.parameters()),
        'source_seeds':{'weights':SOURCE_WEIGHTS_SEED,'data':SOURCE_DATA_SEEDS},
        'repeats_each':REPEATS,'warmups_excluded':WARMUPS,'single_thread_cpu':True,
        'rows':results,'scientific_first_use':'NOT_RUN','whole_system_windows_linux_acceptance':'UNVERIFIED'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args();report=run()
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'all_pass':report['all_pass'],'rows':report['rows']},indent=2))

if __name__=='__main__':main()
