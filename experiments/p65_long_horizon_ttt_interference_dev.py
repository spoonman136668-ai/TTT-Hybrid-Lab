"""P65 frozen mechanistic DEV: interference stress test of reference TTT online matrix.

Idealized keys=queries (no learned projection; deliberately favorable to TTT).
Does not claim a fully trained Hybrid or physically authenticated observations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F

SEEDS=(8001,8003,8007)
EPISODES=64
DIM=16
N_TARGET=8
PRESENTATIONS=4
DISTRACTORS=64
ETA=0.5

def update(fast,key,value):
    """Exact source formula of model.TTTLinearMemory with ideal key/value projections."""
    predicted=torch.bmm(fast,key.unsqueeze(-1)).squeeze(-1)
    residual=value-predicted
    return fast+ETA*residual.unsqueeze(-1)*key.unsqueeze(1)

def read(fast,key):
    return torch.einsum('bij,bkj->bki',fast,key)

def mse(a,b):
    return float(((a-b)**2).mean().item())

def cosine(a,b):
    return float(F.cosine_similarity(a,b,dim=-1).mean().item())

def run_world(seed,world):
    g=torch.Generator(device='cpu').manual_seed(seed)
    # Each episode begins with an independent orthonormal coordinate system.
    bases=torch.linalg.qr(torch.randn(EPISODES,DIM,DIM,generator=g)).Q
    keys=bases[:,:,:N_TARGET].transpose(1,2).contiguous()
    values=torch.tanh(torch.randn(EPISODES,N_TARGET,DIM,generator=g))
    noise_keys=torch.randn(EPISODES,DISTRACTORS,DIM,generator=g)
    if world=='orthogonal':
        complement=bases[:,:,N_TARGET:]
        noise_keys=torch.einsum('bdj,btj->btd',complement,noise_keys[:,:,:DIM-N_TARGET])
    else:
        assert world=='overlap'
    noise_keys=F.normalize(noise_keys,dim=-1)
    noise_values=torch.tanh(torch.randn(EPISODES,DISTRACTORS,DIM,generator=g))
    # Pin immutable data digest: all arms see exactly the same events and queries.
    data_sha=hashlib.sha256(torch.cat([keys.reshape(-1),values.reshape(-1),
        noise_keys.reshape(-1),noise_values.reshape(-1)]).contiguous().numpy().tobytes()).hexdigest()
    fast=torch.zeros(EPISODES,DIM,DIM)
    with torch.no_grad():
        for t in range(PRESENTATIONS):
            for i in range(N_TARGET):
                fast=update(fast,keys[:,i,:],values[:,i,:])
        before=read(fast,keys)
        gated=fast.clone()
        checkpoint=fast.detach().clone()
        for i in range(DISTRACTORS):
            fast=update(fast,noise_keys[:,i,:],noise_values[:,i,:])
        after=read(fast,keys)
        # Same 256-scalar budget: 8x key+value pairs; oracle prefix gate sees phase.
        exact_cache=values
        # Last-eight FIFO overwrites the target memory with distractors.
        fifo_k=noise_keys[:,-N_TARGET:,:]
        fifo_v=noise_values[:,-N_TARGET:,:]
        similarity=torch.einsum('bid,bjd->bij',keys,fifo_k)
        matched=(similarity>1-1e-5)
        fifo_response=torch.where(matched.any(dim=-1,keepdim=True),
            torch.einsum('bij,bjd->bid',matched.float(),fifo_v),torch.zeros_like(values))
        cold=checkpoint.clone().contiguous()
        assert torch.equal(read(cold,keys),before)
        assert EPISODES*DIM*DIM==gated.numel()
        signal=float((values**2).mean().item())
        pre=mse(before,values)
        post=mse(after,values)
        control=mse(read(gated,keys),values)
        result={'seed':seed,'world':world,'data_sha256':data_sha,
            'pre_mse':pre,'post_mse':post,'oracle_gated_mse':control,
            'equal_budget_indexed_cache_mse':mse(exact_cache,values),
            'equal_budget_fifo_mse':mse(fifo_response,values),
            'zero_state_mse':signal,'post_cosine':cosine(after,values),
            'pre_cosine':cosine(before,values),
            'post_vs_gated_mse_ratio':post/control if control>0 else None,
            'signal_normalized_post_mse':post/signal,
            'cold_restore_bitwise':True,
            'negative_control_preserved':world!='orthogonal' or abs(post-pre)<=1e-5,
            'candidate_success':post<=control*1.1 and post<=mse(exact_cache,values)*0.9}
    return result

def run():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[run_world(seed,world) for seed in SEEDS for world in ('overlap','orthogonal')]
    return {'schema':'ttt-hybrid.p65-fast-matrix-interference-development-result.v1',
        'source_original_model_blob':'5d9ea57b827aa1796548447868fa5ee14c251b49',
        'scope':'IDEAL_KEY_AND_VALUE_MECHANISM_ONLY_NOT_FULL_HYBRID_SCIENCE',
        'seeds':list(SEEDS),'dtype':'torch.float32','torch':torch.__version__,
        'fast_state_floats':DIM*DIM,'indexed_classical_cache_floats':N_TARGET*2*DIM,
        'all_orthogonal_controls_preserved':all(row['negative_control_preserved'] for row in rows),
        'strict_candidate_success_all_seeds':all(row['candidate_success'] for row in rows if row['world']=='overlap'),
        'rows':rows,'scientific_first_use':'NOT_RUN','user_gpu':'NOT_RUN',
        'trained_gru_and_mlp':'NOT_RUN','real_causal_world':'NOT_RUN'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'strict_candidate_success_all_seeds':result['strict_candidate_success_all_seeds'],
        'all_orthogonal_controls_preserved':result['all_orthogonal_controls_preserved'],
        'rows':[{'seed':r['seed'],'world':r['world'],'pre_mse':r['pre_mse'],
            'post_mse':r['post_mse'],'oracle_gated_mse':r['oracle_gated_mse'],
            'cache_mse':r['equal_budget_indexed_cache_mse'],
            'negative_control_preserved':r['negative_control_preserved']} for r in result['rows']]},indent=2))

if __name__=='__main__':main()
