import torch
from model import Configuration, HybridPilot, TTTLinearMemory, CausalDepthRouter

torch.set_num_threads(1)

def make(cfg):
    torch.manual_seed(17)
    return HybridPilot(12, 7, cfg).eval()

def test_dimensions_finite_gradients():
    for cfg in (Configuration(),Configuration(ttt=False,depth=False), Configuration(ssm=False,ttt=False,depth=False), Configuration(attention=False,ttt=False,depth=False)):
        net=make(cfg)
        x=torch.randn(2,7,12,requires_grad=True)
        y=net(x)
        assert y.shape==(2,7,7)
        assert torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

def test_strict_prefix_invariance():
    """Future data cannot change an earlier output, including routed depth."""
    configs=(Configuration(), Configuration(ttt=False), Configuration(ssm=False))
    for cfg in configs:
        net=make(cfg)
        x=torch.randn(1,9,12)
        with torch.no_grad():
            original=net(x)
            changed=x.clone()
            changed[:,5:,:]=100*torch.randn_like(changed[:,5:,:])
            revised=net(changed)
            cropped=net(x[:,:5,:])
        torch.testing.assert_close(original[:,:5],revised[:,:5],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(original[:,:5],cropped,rtol=1e-5,atol=1e-5)

def test_episode_reset_and_no_cross_batch_leakage():
    net=make(Configuration())
    x=torch.randn(2,7,12)
    with torch.no_grad():
        a=net(x)
        _=net(10*torch.randn_like(x))
        b=net(x)
        separate=net(x[:1])
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(a[:1],separate,rtol=1e-5,atol=1e-5)

def test_ttt_read_before_update_and_update_effect():
    torch.manual_seed(4)
    mem=TTTLinearMemory(8,learning_rate=0.8).eval()
    x=torch.randn(1,6,8)
    with torch.no_grad():
        updated=mem(x,update=True)
        frozen=mem(x,update=False)
    torch.testing.assert_close(updated[:,:1],frozen[:,:1])
    assert not torch.allclose(updated[:,1:],frozen[:,1:])

def test_route_sparse_computation_and_causal_decision():
    torch.manual_seed(7)
    route=CausalDepthRouter(8)
    x=torch.randn(2,5,8)
    with torch.no_grad():
        route.router.weight.zero_()
        route.router.bias.fill_(-20)
        y=route(x)
        assert route.last_utilization==0.0
        torch.testing.assert_close(x,y)
        route.router.bias.fill_(20)
        y=route(x)
        assert route.last_utilization==1.0
        assert not torch.allclose(x,y)
