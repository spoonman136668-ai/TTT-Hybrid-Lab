"""Adapters are tested with explicit doubles; no counterfeit official PASS."""
import pytest
import torch
from torch import nn
from authentic_hybrid import AuthenticHybrid, integration_checks

class MambaDouble(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.lin=nn.Linear(kwargs['d_model'], kwargs['d_model'])
    def forward(self,x):
        return self.lin(x)  # no temporal mixing, but exercises adapter contract

class ConfigDouble:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TTTDouble(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.lin=nn.Linear(config.hidden_size, config.hidden_size)
    def forward(self,x,*,position_ids,cache_params):
        assert position_ids.shape==x.shape[:2]
        assert cache_params is None
        return self.lin(x)

CLASSES=(MambaDouble,ConfigDouble,TTTDouble)

def test_real_backend_required():
    with pytest.raises(ValueError,match='Official classes'):
        AuthenticHybrid()

def test_invalid_shapes():
    with pytest.raises(ValueError):
        AuthenticHybrid(width=60,official_classes=CLASSES)
    m=AuthenticHybrid(official_classes=CLASSES)
    with pytest.raises(ValueError):
        m(torch.randn(2,8,32))

def test_injected_backend_cpu_contract_only():
    m=AuthenticHybrid(official_classes=CLASSES)
    out=integration_checks(m,device=torch.device('cpu'))
    assert out['prefix_invariance'] and out['suffix_negative_control']
    assert all(v is True for v in out['backward_finite_nonzero'].values())

def test_ablation_exercises_remaining_components():
    m=AuthenticHybrid(use_ttt=False,use_depth=False,official_classes=CLASSES)
    out=integration_checks(m,device=torch.device('cpu'))
    assert out['backward_finite_nonzero']['ttt']=='OMITTED_ABLATION'
    assert out['backward_finite_nonzero']['depth']=='OMITTED_ABLATION'

def test_source_class_origin_guard(tmp_path,monkeypatch):
    import authentic_hybrid as hybrid
    import sys,types
    mamba=types.ModuleType('mamba_ssm')
    modules=types.ModuleType('mamba_ssm.modules')
    child=types.ModuleType('mamba_ssm.modules.mamba2')
    child.Mamba2=MambaDouble
    tt=types.ModuleType('ttt')
    tt.TTTConfig=ConfigDouble
    tt.TTTLinear=TTTDouble
    monkeypatch.setitem(sys.modules,'mamba_ssm',mamba)
    monkeypatch.setitem(sys.modules,'mamba_ssm.modules',modules)
    monkeypatch.setitem(sys.modules,'mamba_ssm.modules.mamba2',child)
    monkeypatch.setitem(sys.modules,'ttt',tt)
    with pytest.raises(RuntimeError,match='UPSTREAM_CLASS_ORIGIN_MISMATCH'):
        hybrid.load_official_classes(tmp_path,tmp_path)

def test_preflight_blocks_before_upstream_load(tmp_path,monkeypatch):
    import run_gate3
    def wrong(*args,**kwargs):
        raise AssertionError('must not attempt load')
    monkeypatch.setattr(run_gate3,'load_official_classes',wrong)
    r,c=run_gate3.run(tmp_path/'missing_mamba',tmp_path/'missing_ttt',receipt=tmp_path/'receipt.json')
    assert c==2 and r['status']=='BLOCKED_SOURCE_RUNTIME_ADMISSION'
    assert r['authentic_forward']=='NOT_RUN'
    assert (tmp_path/'receipt.json').is_file()

class LeakyMambaDouble(MambaDouble):
    def forward(self,x):
        # Deliberate future leak: every prefix token sees suffix.
        return super().forward(x) + x.mean(dim=1,keepdim=True)

class PersistentTTTDouble(TTTDouble):
    def __init__(self,config,layer_idx):
        super().__init__(config,layer_idx)
        self.calls=0
    def forward(self,x,*,position_ids,cache_params):
        self.calls+=1
        return super().forward(x,position_ids=position_ids,cache_params=cache_params)+0.01*self.calls


def test_suffix_leak_negative_control_rejects_future_mixing():
    m=AuthenticHybrid(official_classes=(LeakyMambaDouble,ConfigDouble,TTTDouble))
    with pytest.raises(AssertionError,match='SUFFIX_LEAKAGE|PREFIX_INVARIANCE_FAILED'):
        integration_checks(m,device=torch.device('cpu'))


def test_episode_reset_negative_control_rejects_carryover():
    m=AuthenticHybrid(official_classes=(MambaDouble,ConfigDouble,PersistentTTTDouble))
    with pytest.raises(AssertionError,match='EPISODE_RESET_NOT_DETERMINISTIC'):
        integration_checks(m,device=torch.device('cpu'))


class GradForbiddenMambaDouble(MambaDouble):
    def forward(self, x):
        assert not torch.is_grad_enabled(), "GRAD_TRACKING_FORBIDDEN"
        assert not self.training, "TRAIN_MODE_FORBIDDEN"
        return super().forward(x)


def test_r6_forward_only_no_grad_deterministic_causal_cpu():
    """Engineering contract with test doubles; cannot qualify authentic CUDA."""
    from authentic_hybrid import forward_only_checks
    model = AuthenticHybrid(
        official_classes=(GradForbiddenMambaDouble, ConfigDouble, TTTDouble),
    )
    report = forward_only_checks(model, device=torch.device("cpu"))
    assert report["forward_finite"] is True
    assert report["shape"] == [2, 16, 64]
    assert report["episode_reset_deterministic"] is True
    assert report["prefix_invariance"] is True
    assert report["suffix_negative_control"] is True
    assert report["gradient_execution"] == "NOT_RUN"
    assert report["optimizer_steps"] == 0
    assert report["training"] == "NOT_RUN"
    assert all(param.grad is None for param in model.parameters())


def test_r6_forward_only_rejects_future_suffix_leak_cpu():
    from authentic_hybrid import forward_only_checks
    model = AuthenticHybrid(
        official_classes=(LeakyMambaDouble, ConfigDouble, TTTDouble),
    )
    with pytest.raises(AssertionError, match="SUFFIX_LEAKAGE|PREFIX_INVARIANCE_FAILED"):
        forward_only_checks(model, device=torch.device("cpu"))


def test_r6_forward_only_rejects_persistent_ttt_state_cpu():
    from authentic_hybrid import forward_only_checks
    model = AuthenticHybrid(
        official_classes=(MambaDouble, ConfigDouble, PersistentTTTDouble),
    )
    with pytest.raises(AssertionError, match="EPISODE_RESET_NOT_DETERMINISTIC"):
        forward_only_checks(model, device=torch.device("cpu"))
