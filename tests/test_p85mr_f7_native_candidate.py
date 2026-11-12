"""Source-neutral F7 native bypass, registered scope and negative controls."""
from __future__ import annotations
import pytest
import torch

from experiments import p85mr_f7_protocol as p7
from experiments import p85mr_f7_native_candidate as f7
from experiments import p85mr_f5_native_vs_controls as f5

torch.set_num_threads(1)


def synthetic_world(seed):
    a=torch.zeros(5,16)
    a[:,0]=.7071
    a[:,1]=.5
    a[:,5]=torch.tensor([1.,0.,1.,0.,1.])
    a[:,6]=1.-a[:,5]
    a=a/a.norm(dim=-1,keepdim=True)  # exact P80 unit-norm state/action key
    e=torch.zeros(5,16)
    e[:,0]=torch.arange(5)*.004
    e[:,2]=torch.tensor([.002,.003,.004,.005,.006])
    e[:,4]=.05
    e[:,5]=torch.tensor([0.,0.,0.,0.,1.])
    return {"action":a.clone(),"effect":e.clone()}


def test_f7_research_scope_is_new_and_frozen_before_source():
    x=p7.frozen()
    assert x["scope"]["prior_F3_F4_F5_spent"]
    assert not x["authority"]["approve_F7_CAL"]
    assert not x["authority"]["approve_F6_A5"]
    assert p7.seed(0)==1821000000
    assert p7.seed(127)==1821000127
    assert p7.factual_policy(0,1,[0.,0.,.1,0.]) in (0,1)
    assert all(p7.factual_policy(i,step,[0.,0.,.1,0.]) in (0,1)
               for i in range(128) for step in range(1,97))
    with pytest.raises(ValueError,match="F7_OUT_OF_SCOPE_WORLD"):
        p7.seed(128)
    with pytest.raises(ValueError,match="F7_INVALID_PUBLIC_STATE"):
        p7.factual_policy(0,1,[0.,float("nan"),0.,0.])
    with pytest.raises(PermissionError,match="F7_SOURCE_MAY_RUN_ONLY"):
        p7.no_external_from_CI()
    assert p7.proposal_receipt()["source_firstuse_executed"] is False


def test_f7_bypass_starts_exactly_as_original_native_then_can_use_current_action():
    worlds=synthetic_world(0)
    data={"action":worlds["action"].unsqueeze(0),
          "effect":worlds["effect"].unsqueeze(0)}
    model=f7.NativeRawKeyBypass(985201).eval()
    ref=f5.original_native(985201).eval()
    with torch.no_grad():
        x=f5.factual_inputs(data)
        raw=ref(x)
        actual=model(x)
        assert torch.equal(raw,actual)
        model.raw_key.weight[0,5]=3.
        altered=model(x)
        assert not torch.equal(raw,altered)
        changed={k:v.clone() for k,v in data.items()}
        changed["effect"][:,3,:7]+=0.123
        # No current-effect leakage: predictions for steps <=3 unchanged.
        assert torch.equal(f7.prequential(f7.K_BYPASS,model,data)[:,:4],
                           f7.prequential(f7.K_BYPASS,model,changed)[:,:4])


def test_f7_native_and_strong_ssm_have_genuine_toy_optimizer_gradients():
    worlds=[synthetic_world(0),synthetic_world(1)]
    results=[f7.single_gradient_step(arm,985201,worlds) for arm in p7.ARMS]
    assert len(results)==3
    assert all(r["toy_updated_weights"]!=r["initial_weights"] for r in results)
    assert all(r["real_source_events"]==0 and not r["real_f7_training"] for r in results)


def test_f7_denies_wrong_worlds_model_seed_and_forged_current_effect():
    with pytest.raises(ValueError,match="F7_UNREGISTERED_ARM_OR_SEED"):
        f7.model_from_arm("fake",985201)
    with pytest.raises(ValueError,match="F7_UNREGISTERED_ARM_OR_SEED"):
        f7.model_from_arm(f7.K_BYPASS,1)
    with pytest.raises(ValueError,match="F7_TOY_EPISODES_REQUIRED"):
        f7.single_gradient_step(f7.K_BYPASS,985201,[])
    m=f7.NativeRawKeyBypass(985201).eval()
    with pytest.raises(ValueError,match="F7_INVALID_CAUSAL_INPUT"):
        m(torch.full((1,3,32),float("nan")))
    source=open(f7.__file__,encoding="utf-8").read()
    for prohibited in ("from causal_gym","import causal_gym","env.do(","env.reset(",
                       "dispatch_workflow(","subprocess.run(","requests.get(",
                       "torch.load("):
        assert prohibited not in source


def test_f7_real_gradient_admission_allows_only_hard_inactive_native_depth():
    from experiments import p85mr_f7_train_fit as trainer
    batch,valid=f7.f4._padded([synthetic_world(0),synthetic_world(1)])
    for arm in (f7.K_NATIVE,f7.K_BYPASS):
        model=f7.model_from_arm(arm,985201)
        depth=model.depth if arm==f7.K_NATIVE else model.original.depth
        with torch.no_grad():
            depth.router.weight.zero_()
            depth.router.bias.fill_(-100.)
        model.train()
        model.zero_grad(set_to_none=True)
        f7.loss_for_arm(arm,model,batch,valid).backward()
        assert depth.last_utilization==0.
        receipt=trainer.qualified_gradients(arm,model)
        assert receipt["dormant_depth_count"]>0
        prefix="depth." if arm==f7.K_NATIVE else "original.depth."
        assert all(n.startswith(prefix) for n,p in model.named_parameters()
                   if p.grad is None)
        # Failing an essential model connection must not be mistaken for sparsity.
        stem=model.stem if arm==f7.K_NATIVE else model.original.stem
        stem.weight.grad=None
        with pytest.raises(ValueError,match="F7_UNQUALIFIED_NONDEPTH_MODEL_GRADIENT"):
            trainer.qualified_gradients(arm,model)
        stem.weight.grad=torch.full_like(stem.weight,float("inf"))
        with pytest.raises(ValueError,match="F7_NONFINITE_OR_WRONG_SHAPE_GRADIENT"):
            trainer.qualified_gradients(arm,model)


def test_f7_conventional_ssm_and_missing_native_core_cannot_fake_gradient():
    from experiments import p85mr_f7_train_fit as trainer
    batch,valid=f7.f4._padded([synthetic_world(0),synthetic_world(1)])
    ssm=f7.model_from_arm(f7.K_SSM,985201).train()
    ssm.zero_grad(set_to_none=True)
    f7.loss_for_arm(f7.K_SSM,ssm,batch,valid).backward()
    assert trainer.qualified_gradients(f7.K_SSM,ssm)["dormant_depth_count"]==0
    first_param=next(iter(ssm.parameters()))
    first_param.grad=None
    with pytest.raises(ValueError,match="F7_UNEXPECTED_SSM_DISCONNECTED_GRADIENT"):
        trainer.qualified_gradients(f7.K_SSM,ssm)
