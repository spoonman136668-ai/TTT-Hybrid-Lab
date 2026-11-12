"""F7 bounded native bypass ablation: causal same-step action, prior effect only.

The original four-block HybridPilot is reused unchanged. A 16x16 direct key
readout may repair loss of public action identity; no simulator APIs, source
access or checkpoints appear here. FIT-only heldout is DEVELOPMENT, never
sealed CAL generalization.
"""
from __future__ import annotations
import math
import copy
import hashlib
import time
import torch
from torch import nn

from experiments import p85mr_f7_protocol as p7
from experiments import p85mr_f5_native_vs_controls as f5
from experiments import p85mr_f4_train_controls as f4
from p85mr_dynamics_baselines import SEEDS, new_baseline

K_NATIVE=p7.ARMS[0]
K_BYPASS=p7.ARMS[1]
K_SSM=p7.ARMS[2]


class NativeRawKeyBypass(nn.Module):
    def __init__(self,seed):
        super().__init__()
        if seed not in p7.MODEL_SEEDS:
            raise ValueError("F7_BAD_FROZEN_MODEL_SEED")
        self.original=f5.original_native(seed)
        self.raw_key=nn.Linear(16,16)
        with torch.no_grad():
            self.raw_key.weight.zero_()
            self.raw_key.bias.zero_()

    def forward(self,history,ttt_update=True):
        if (not isinstance(history,torch.Tensor) or history.ndim!=3
            or history.shape[-1]!=32 or not history.is_floating_point()
            or not bool(torch.isfinite(history).all())):
            raise ValueError("F7_INVALID_CAUSAL_INPUT")
        output=self.original(history,ttt_update=ttt_update)
        return output+self.raw_key(history[...,:16])


def model_from_arm(arm,seed):
    if arm not in p7.ARMS or seed not in p7.MODEL_SEEDS:
        raise ValueError("F7_UNREGISTERED_ARM_OR_SEED")
    if arm==K_NATIVE:
        return f5.original_native(seed)
    if arm==K_BYPASS:
        return NativeRawKeyBypass(seed)
    return new_baseline("ssm256",seed)


def prequential(arm,model,facts):
    if arm==K_SSM:
        return f4.prequential(model,facts)
    return f5.native_predict(model,facts)


def loss_for_arm(arm,model,facts,valid):
    if arm==K_SSM:
        return f4._physical_loss(model,facts,valid)[0]
    if arm in (K_NATIVE,K_BYPASS):
        return f5.training_loss(model,facts,valid)
    raise ValueError("F7_UNKNOWN_ARM")


def score_worlds(arm,model,worlds):
    if not isinstance(worlds,list) or not worlds:
        raise ValueError("F7_MISSING_FIT_ONLY_VALIDATION")
    if arm==K_SSM:
        return float(f4.calibration_mse(model,worlds)["cal_delta_mse"])
    return float(f5.cal_mse(model,worlds))


def _weight_hash(model):
    return f5.hash_weights(model)


def single_gradient_step(arm,seed,worlds):
    """Source-neutral real torch optimizer step on fabricated FIT tensors."""
    if not isinstance(worlds,list) or len(worlds)<2:
        raise ValueError("F7_TOY_EPISODES_REQUIRED")
    p7.frozen()
    m=model_from_arm(arm,seed)
    data,valid=f4._padded(worlds)
    m.train()
    initial=_weight_hash(m)
    opt=torch.optim.AdamW(m.parameters(),
        lr=.001 if arm==K_SSM else .002,weight_decay=.01)
    opt.zero_grad(set_to_none=True)
    v=loss_for_arm(arm,m,data,valid)
    if not bool(torch.isfinite(v)):
        raise ValueError("F7_NONFINITE_TRAINING_OBJECTIVE")
    v.backward()
    grads=[p.grad for p in m.parameters()]
    if not any(g is not None and bool(torch.isfinite(g).all()) and bool((g.abs()>0).any()) for g in grads):
        raise ValueError("F7_NO_NATIVE_LEARNING_GRADIENT")
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    opt.step()
    trained=_weight_hash(m)
    if initial==trained:
        raise ValueError("F7_NATIVE_WEIGHTS_UNCHANGED")
    m.eval()
    return {"arm":arm,"seed":seed,"initial_weights":initial,
        "toy_updated_weights":trained,"finite_loss":float(v.item()),
        "real_source_events":0,"real_f7_training":False,
        "scientific_pass":False,"deployment":False}
