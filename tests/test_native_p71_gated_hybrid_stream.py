"""P71 original-reference four-block integration: engineering tests only."""
from __future__ import annotations
import copy
import hashlib
import io
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from model import HybridPilot, Configuration
from native_p64k_incremental_hybrid_stream import StreamingHybridPilot, HybridFactualState
from native_p62_streaming_ttt import TTTFactualState
from native_p71_gated_hybrid_stream import (
    GATE_CHECKPOINTS, GatedStreamingHybridPilot, factual_write_gate,
    load_gate_checkpoint,
)

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def make(seed=71, batch=2, width=16):
    torch.manual_seed(seed)
    core=HybridPilot(5,2,Configuration(width=width,heads=2,ssm=True,attention=True,ttt=True,depth=True)).eval()
    facts=torch.randn(batch,9,5)
    candidates=torch.randn(batch,3,5)
    return core, facts, candidates


def ingest(adapter, facts, split=(2,3,4)):
    state=adapter.initial_state(facts[:,0,:])
    ys=[]
    first=0
    for n in split:
        out,state=adapter.ingest_confirmed(facts[:,first:first+n,:],state,
                         torch.ones((len(facts),n),dtype=torch.bool))
        ys.append(out); first+=n
    return torch.cat(ys,1),state


def sha(path):
    b=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()


def test_original_reference_blob_custody_and_parameter_identity():
    root=Path(__file__).resolve().parents[1]
    assert sha(root/'model.py') == '5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert sha(root/'native_p62_streaming_ttt.py') == '776e63a0ff9474a860e87088718acb36637a7bec'
    assert sha(root/'native_p64k_incremental_hybrid_stream.py') == '89c3ef5819ac594ad687c67c579a559754109d4c'
    core,_,_=make()
    gated=GatedStreamingHybridPilot(core,seed=1000203,max_confirmed_events=12)
    assert gated.core is core and gated.ttt.core is core.ttt
    assert sum(p.numel() for p in core.parameters()) == sum(p.numel() for p in gated.core.parameters())


def test_all_write_mode_matches_exact_reference_four_block_stream():
    for seed,batch in ((71,1),(73,2),(79,3)):
        core,facts,candidates=make(seed,batch)
        original=StreamingHybridPilot(core,max_confirmed_events=12)
        identical=GatedStreamingHybridPilot(core,seed=1000203,mode='all_write',max_confirmed_events=12)
        with torch.no_grad():
            actual,state_o=ingest(original,facts)
            candidate_o=original.preview(candidates,state_o)
            result,state_i=ingest(identical,facts)
            candidate_i=identical.preview(candidates,state_i)
            full=core(facts)
        assert torch.equal(state_o.ttt.fast_weights,state_i.ttt.fast_weights)
        torch.testing.assert_close(actual,result,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(candidate_o,candidate_i,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(result,full,atol=1e-6,rtol=1e-6)
        assert torch.equal(state_o.attention_k,state_i.attention_k)
        assert torch.equal(state_o.attention_v,state_i.attention_v)
        assert torch.equal(state_o.ssm_hidden,state_i.ssm_hidden)
        assert identical.ttt.theta.shape==(4,)


def test_learned_gate_changes_confirmed_fast_write_only():
    core,facts,candidates=make()
    original=StreamingHybridPilot(core,max_confirmed_events=12)
    gated=GatedStreamingHybridPilot(core,max_confirmed_events=12)
    with torch.no_grad():
        outputs_o,state_o=ingest(original,facts)
        outputs_g,state_g=ingest(gated,facts)
    assert not torch.equal(state_o.ttt.fast_weights,state_g.ttt.fast_weights)
    assert torch.equal(state_o.ssm_hidden,state_g.ssm_hidden)
    assert torch.equal(state_o.attention_k,state_g.attention_k)
    assert torch.equal(state_o.attention_v,state_g.attention_v)
    assert torch.equal(outputs_o[:,0],outputs_g[:,0])
    assert (outputs_o-outputs_g).abs().max()>0
    assert state_g.ttt.confirmed_count.tolist()==[9,9]
    assert not torch.equal(gated.preview(candidates,state_g),original.preview(candidates,state_o))


def test_p70_checkpoint_sha_and_exact_gate_equation():
    ref=torch.zeros(4)
    for seed,(coeffs,sha256) in GATE_CHECKPOINTS.items():
        loaded=load_gate_checkpoint(seed,ref)
        assert hashlib.sha256(loaded.numpy().tobytes()).hexdigest()==sha256
        assert torch.equal(loaded,torch.tensor(coeffs,dtype=torch.float32))
    predicted=torch.tensor([[0.,0.,0.],[.2,.1,.3]])
    observed=torch.tensor([[1.,-.5,.2],[.2,.3,.4]])
    theta=load_gate_checkpoint(1000201,ref)
    err=(observed-predicted).square().mean(-1)
    features=torch.stack((torch.ones_like(err),torch.log(err+1e-6),
            torch.log(predicted.square().mean(-1)+1e-6),
            F.cosine_similarity(predicted,observed,dim=-1,eps=1e-6)),-1)
    torch.testing.assert_close(factual_write_gate(predicted,observed,theta),torch.sigmoid(features@theta),atol=0,rtol=0)


def test_cold_resume_restores_exact_gated_preview():
    core,facts,candidates=make(79,3)
    gated=GatedStreamingHybridPilot(core,seed=1000203,max_confirmed_events=12)
    with torch.no_grad():
        _,state=ingest(gated,facts)
        before=gated.preview(candidates,state)
    buf=io.BytesIO()
    # Serialize only tensors and primitive provenance, never Python dataclasses.
    safe_state={'ssm':state.ssm_hidden, 'k':state.attention_k, 'v':state.attention_v,
                'tw':state.ttt.fast_weights, 'count':state.ttt.confirmed_count,
                'fingerprint':state.model_fingerprint}
    torch.save({'model':core.state_dict(), 'state':safe_state},buf)
    buf.seek(0)
    saved=torch.load(buf,weights_only=True)
    new_model=HybridPilot(5,2,Configuration(width=16,heads=2)).eval()
    new_model.load_state_dict(saved['model'])
    restored=GatedStreamingHybridPilot(new_model,seed=1000203,max_confirmed_events=12)
    st=saved['state']
    recovered=HybridFactualState(st['ssm'],st['k'],st['v'],
                 TTTFactualState(st['tw'],st['count']),st['fingerprint'])
    with torch.no_grad():
        after=restored.preview(candidates,recovered)
    assert torch.equal(before,after)
    assert state.model_fingerprint==restored._model_fingerprint
    assert torch.equal(state.ttt.fast_weights,recovered.ttt.fast_weights)


def test_cross_seed_or_mode_state_refused():
    core,facts,candidates=make()
    source=GatedStreamingHybridPilot(core,seed=1000203,max_confirmed_events=12)
    with torch.no_grad(): _,state=ingest(source,facts)
    for seed,mode in ((1000201,'learned'),(1000203,'all_write')):
        different=GatedStreamingHybridPilot(core,seed=seed,mode=mode,max_confirmed_events=12)
        with pytest.raises(ValueError,match='P64K_STALE_OR_UNTAGGED_FACTUAL_STATE'):
            different.preview(candidates,state)


def test_atomic_unconfirmed_and_bounds_refused_without_state_mutation():
    core,facts,candidates=make()
    stream=GatedStreamingHybridPilot(core,max_confirmed_events=9)
    state=stream.initial_state(facts[:,0,:])
    before=state.ttt.fast_weights.clone()
    with pytest.raises(ValueError,match='P64S_UNCONFIRMED_FACT_REFUSED'):
        stream.ingest_confirmed(facts,state,torch.zeros(2,9,dtype=torch.bool))
    assert torch.equal(state.ttt.fast_weights,before)
    with torch.no_grad(): _,end=ingest(stream,facts)
    with pytest.raises(ValueError,match='P64S_BOUNDED_HISTORY_FULL'):
        stream.ingest_confirmed(facts[:,:1],end,torch.ones(2,1,dtype=torch.bool))


def test_read_only_preview_repeat_permutation_and_no_state_write():
    core,facts,candidates=make()
    stream=GatedStreamingHybridPilot(core,max_confirmed_events=12)
    with torch.no_grad(): _,state=ingest(stream,facts)
    saved=(state.ssm_hidden.clone(),state.attention_k.clone(),state.attention_v.clone(),state.ttt.fast_weights.clone(),state.ttt.confirmed_count.clone())
    first=stream.preview(candidates,state)
    second=stream.preview(candidates.flip(1),state).flip(1)
    third=stream.preview(candidates,state)
    assert torch.equal(first,second) and torch.equal(first,third)
    for a,b in zip(saved,(state.ssm_hidden,state.attention_k,state.attention_v,state.ttt.fast_weights,state.ttt.confirmed_count)):
        assert torch.equal(a,b)


def test_gate_weight_tampering_and_core_weight_revision_fail_closed():
    core,facts,candidates=make()
    stream=GatedStreamingHybridPilot(core,max_confirmed_events=12)
    with torch.no_grad(): _,state=ingest(stream,facts)
    with torch.no_grad(): stream.ttt.theta[0].add_(0.1)
    with pytest.raises(ValueError,match='P71_GATE_WEIGHTS_CHANGED_REBUILD_STATE'):
        stream.preview(candidates,state)
    fresh=GatedStreamingHybridPilot(core,max_confirmed_events=12)
    with torch.no_grad(): _,new_state=ingest(fresh,facts)
    with torch.no_grad(): core.ttt.key.weight[0,0].add_(0.1)
    with pytest.raises(ValueError,match='P64K_MODEL_CHANGED_REBUILD_FACTUAL_CACHE'):
        fresh.preview(candidates,new_state)


def test_invalid_checkpoint_or_gate_input_refused():
    with pytest.raises(ValueError,match='P71_UNKNOWN_GATE_CHECKPOINT'):
        load_gate_checkpoint(123456,torch.zeros(4))
    with pytest.raises(ValueError,match='P71_INVALID_FACTUAL_GATE_INPUT'):
        factual_write_gate(torch.zeros(2,16),torch.ones(2,16),torch.zeros(5))
    with pytest.raises(ValueError,match='P71_INVALID_FACTUAL_GATE_INPUT'):
        factual_write_gate(torch.full((2,16),float('nan')),torch.ones(2,16),torch.zeros(4))
