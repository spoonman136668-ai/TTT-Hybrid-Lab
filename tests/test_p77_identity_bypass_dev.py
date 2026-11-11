"""P77 source, frozen-core and identity-sidecar tests. No first-use science."""
from pathlib import Path
import importlib.util
import hashlib
import pytest
import torch

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p77',root/'experiments'/'p77_identity_bypass_dev.py')
p77=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p77)
torch.set_num_threads(1)


def blob(path):
    data=path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def test_exact_original_and_p76_blobs():
    assert blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert blob(root/'experiments/p76_native_alignment_dev.py')=='d4a9313a760d36a2929a823061ad61aa7098a5d4'


def test_fresh_repeatable_source_and_factual_no_future():
    for split in ('fit','val','negative'):
        w=p77.make_world(split,8)
        assert p77.source_sha(w)==p77.source_sha(p77.make_world(split,8))
        assert w['facts'].shape==(8,24,32)
        assert w['query'].shape==(8,1,32)
        assert torch.equal(w['query'][:,:,16:],torch.zeros(8,1,16))
        assert torch.equal(w['role_audit'].sum(-1),torch.full((8,),16))
        assert p77.p76.labels_mask(w).sum(-1).tolist()==[2]*8
    assert p77.make_world('ood_dev',8)['facts'].shape==(8,48,32)


def test_input_guards_and_index_baseline():
    with pytest.raises(ValueError,match='P77_UNKNOWN_SPLIT'):
        p77.make_world('open_ended')
    with pytest.raises(ValueError,match='P77_BAD_COUNT'):
        p77.make_world('fit',0)
    w=p77.make_world('val',8)
    facts=w['facts'];q=w['query'][:,0,:16]
    sim=(torch.nn.functional.normalize(facts[:,:,:16],dim=-1)*q[:,None,:]).sum(-1)
    at=sim.argmax(-1)
    torch.testing.assert_close(facts[torch.arange(8),at,16:],w['label'],atol=0,rtol=0)


def test_raw_skip_extreme_identity_controls_address():
    w=p77.make_world('fit',8)
    core=p77.p76.original(770201)
    f=p77.p76.features(core,w)
    model=p77.BypassAlignment(770201)
    with torch.no_grad():model.mix_logit.fill_(15.)
    assert p77.evaluate(model,f,'bypass')['hit']==1.
    with pytest.raises(ValueError,match='P77_BAD_ALIGNED_OBSERVATION'):
        model.logits(f['native_keys'],f['native_query'],f['raw_keys'][:,:-1],f['raw_query'])


def test_native_and_bypass_initial_weights_identical():
    seed=770203
    native=p77.p76.Alignment(seed)
    bypass=p77.BypassAlignment(seed)
    assert torch.equal(native.key_map,bypass.key_map)
    assert torch.equal(native.query_map,bypass.query_map)
    assert sum(p.numel() for p in native.parameters())==512
    assert sum(p.numel() for p in bypass.parameters())==513


def test_fit_only_ridge_decoder_and_native_value_shapes():
    w=p77.make_world('fit',8)
    model=p77.p76.original(770201)
    sha=p77.p76.model_hash(model)
    val=p77.value_features(model,w)
    assert val.shape==(8,24,16)
    decoder=p77.ridge_fit(val.reshape(-1,16),w['facts'][:,:,16:].reshape(-1,16))
    assert decoder.shape==(17,16)
    assert p77.ridge_apply(val,decoder).shape==val.shape
    assert sha==p77.p76.model_hash(model)


def test_no_credit_negative_changes_historical_effects_not_query():
    w=p77.make_world('negative',8)
    other=p77.wrong_credit(w)
    assert torch.equal(w['facts'][:,:,:16],other['facts'][:,:,:16])
    assert torch.equal(w['query'],other['query'])
    assert torch.equal(w['label'],other['label'])
    assert not torch.equal(w['facts'][:,:,16:],other['facts'][:,:,16:])


def test_bypass_gradient_and_cold_state():
    w=p77.make_world('fit',8)
    f=p77.p76.features(p77.p76.original(770207),w)
    model=p77.BypassAlignment(770207)
    loss=p77.p76.contrastive(p77.logits(model,f,'bypass'),f['positive'])
    loss.backward()
    assert model.mix_logit.grad is not None and torch.isfinite(model.mix_logit.grad)
    assert model.key_map.grad is not None and torch.isfinite(model.key_map.grad).all()
    clone=p77.BypassAlignment(770207)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(p77.logits(model,f,'bypass'),p77.logits(clone,f,'bypass'))


def test_paired_bootstrap_positive_for_well_calibrated_errors():
    a=torch.tensor([.1,.15,.08,.2])
    b=torch.tensor([.4,.3,.5,.6])
    assert p77.paired_lower(a,b,reps=500)>0
    assert p77.paired_lower(a,b,reps=500)==p77.paired_lower(a,b,reps=500)
