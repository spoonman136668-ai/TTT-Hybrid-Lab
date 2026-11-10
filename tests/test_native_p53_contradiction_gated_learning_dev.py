"""P53 source-free manual fixtures only; no registered heldout DEV consumption."""
from __future__ import annotations
import ast, io, hashlib
from pathlib import Path
import pytest
import torch
from scripts import native_p53_contradiction_gated_learning_dev as p53
from scripts import native_p52_action_factorized_learning_dev as p52
from scripts import native_p51_factual_hybrid_learning_dev as p51


def manual(actions=(1,-1,1,-1,1,-1,1,-1),beta=.3):
    actions=torch.tensor(actions,dtype=torch.float32)[None,:]
    sensor=torch.zeros((1,1));t=torch.arange(8,dtype=torch.float32)[None,:]
    y=.1+.05*t+actions*beta
    h=torch.stack((torch.zeros_like(y),y,actions,torch.full_like(y,.125),torch.zeros_like(y)),dim=-1)
    return {'history':h,'sensor':sensor,'query_a':torch.ones((1,1)),
            'factual_y':torch.full((1,1),.3)}


def blob(path):
    payload=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()


def test_original_exact_source_custody_p50_p51_p52():
    root=Path(__file__).resolve().parents[1]
    assert blob(root/'model.py')=='5d9ea57b827aa1796548447868fa5ee14c251b49'
    assert blob(root/'native_factual_hybrid_stream.py')=='f78803f9f1d864382f7c4e27f72492dad7175bfd'
    assert blob(root/'scripts/native_p51_factual_hybrid_learning_dev.py')=='d7dbf31f5ee46167775eef7b0c273cb18ba0d606'
    assert blob(root/'scripts/native_p52_action_factorized_learning_dev.py')=='1840b31be57d26c8c3de9d5b8ae6fa74b4bc8b2d'


def test_effect_only_from_latest_opposite_completed_action_pair():
    h=manual()['history'];out,age,valid=p53.last_opposite_effect(h)
    assert valid.tolist()==[True]
    assert out.item()==pytest.approx(.275,abs=1e-6)
    assert age.item()==pytest.approx(1/8)
    changed=h.clone();changed[:,0,1]+=1000.
    other=p53.last_opposite_effect(changed)
    torch.testing.assert_close(out,other[0],atol=0,rtol=0)


def test_no_opposite_action_is_explicitly_not_identified():
    h=manual((1,)*8)['history']
    effect,age,valid=p53.last_opposite_effect(h)
    assert not valid.item()
    assert effect.item()==0
    torch.manual_seed(1);m=p53.ContradictionGatedP53().eval()
    d=manual((1,)*8)
    x=m(d,d['query_a'])
    base=m.base(d,d['query_a'])
    torch.testing.assert_close(x,base,atol=0,rtol=0)


def test_action_credit_mutation_changes_causal_pair_without_future_oracle():
    data=manual((1,-1,-1,1,1,1,-1,1))
    direct=p53.last_opposite_effect(data['history'])[0]
    corrupted=p53.last_opposite_effect(p51.history_with_credit(data,True))[0]
    assert direct.item()!=pytest.approx(corrupted.item())
    model=p53.ContradictionGatedP53().eval()
    out=model(data,data['query_a'])
    wrong=model(data,data['query_a'],wrong_credit=True)
    assert not torch.equal(out,wrong)


def test_query_order_invariance_and_no_future_supervision_leakage():
    torch.manual_seed(2)
    model=p53.ContradictionGatedP53().eval()
    d=manual();actions=torch.tensor([[[-1.],[1.]]])
    weights={k:v.clone() for k,v in model.state_dict().items()}
    y=model(d,actions)
    q=model(d,actions.flip(1))
    torch.testing.assert_close(y.flip(1),q,atol=1e-6,rtol=0)
    d['factual_y'].fill_(99999.)
    fresh=model(d,actions)
    torch.testing.assert_close(fresh,y,atol=0,rtol=0)
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())


def test_finite_gradient_reaches_fast_gate_and_reference_modules():
    torch.manual_seed(4);m=p53.ContradictionGatedP53()
    d=manual();out=m(d,d['query_a']);loss=(out-d['factual_y']).square().mean()
    loss.backward()
    assert m.contradiction_gate.grad is not None
    assert m.contradiction_gate.grad.abs().sum().item()>0
    for name in ('stem','ssm','attn','ttt','head'):
        sub=getattr(m.base.native.pilot,name)
        assert any(w.grad is not None and w.grad.abs().sum().item()>0 for w in sub.parameters())
    assert all(torch.isfinite(w.grad).all() for w in m.parameters() if w.grad is not None)


def test_cold_trainable_gate_restore_exact_on_manual_factual_data():
    torch.manual_seed(8);m=p53.ContradictionGatedP53().eval()
    d=manual();a=torch.tensor([[[-1.],[1.]]]);first=m(d,a)
    b=io.BytesIO();torch.save({'model':m.state_dict(),'history':d['history']},b);b.seek(0)
    recovered=torch.load(b,weights_only=True)
    cold=p53.ContradictionGatedP53().eval();cold.load_state_dict(recovered['model'])
    torch.testing.assert_close(cold(d,a),first,atol=0,rtol=0)
    assert torch.equal(recovered['history'],d['history'])


def test_no_consumed_seed_reuse_and_token_free_native_path():
    assert set(p53.SOURCE_SEEDS.values())=={871101,871103,871107}
    assert p53.MODEL_SEEDS==(881101,881103,881107)
    for filename in ('scripts/native_p53_contradiction_gated_learning_dev.py',
                     'native_factual_hybrid_stream.py'):
        root=Path(__file__).resolve().parents[1]
        attrs={v.attr for v in ast.walk(ast.parse((root/filename).read_text()))
               if isinstance(v,ast.Attribute)}
        assert not attrs.intersection({'Embedding','CrossEntropyLoss','tokenizer','vocabulary','token_ids'})


def test_strong_last_opposite_control_is_nontrivial_and_source_free():
    data=manual()
    p=data['history'];eff,age,valid=p53.last_opposite_effect(p)
    naive=p[:,-1,1]/p[:,-1,2]
    assert valid.item() and (eff-naive).abs().item()>0.01
    assert torch.isfinite(eff).all()
