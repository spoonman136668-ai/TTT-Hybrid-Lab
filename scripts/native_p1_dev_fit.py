"""Bounded analytic DEV fit of continuous-state controls; NOT scientific first-use.

Uses only public algebraic arrays; no S20/S26/S27 generators, protected seeds,
GPU, world admission, heldout access or persistent research training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_p1_conventional_controls import ConventionalAnalogGRU, ConventionalAnalogSSM
from native_continuous_state import NativeContinuousHybrid

DEV_INITIALIZATION = 46109
MAX_UPDATES = 1600
EVAL_EVERY = 64
BATCH = 48
SIGN_FLOOR = .05
ARMS = ("conventional_analog_gru", "conventional_analog_ssm", "native_state_no_online_feedback")

def synthetic_pairs(*, fit: bool):
    """Two interventions with identical observed sensor/prior-like history/time.

    No new world, latent physical simulator, scientific seed, or sealed split.
    """
    u = torch.linspace(-.9, .9, 9) if fit else torch.linspace(-.85, .85, 8)
    v = torch.linspace(-.7, .7, 7) if fit else torch.linspace(-.65, .65, 6)
    x, y = torch.meshgrid(u, v, indexing="ij")
    sensor = torch.stack((x.flatten(), y.flatten()), 1).repeat_interleave(2, 0)
    action = torch.tensor([[-.8], [.8]], dtype=torch.float32).repeat(x.numel(), 1)
    dt = .14 + .03 * sensor[:, :1].square()
    history = sensor.new_zeros((sensor.shape[0], 4, 4))
    for lag in range(4):
        history[:, lag, :2] = sensor + torch.tensor([-.02*(4-lag), .01*(4-lag)])
        history[:, lag, 2:3] = (.25 * history[:, lag, :1]).tanh()
        history[:, lag, 3:4] = .11 + lag*.02
    nonlinear = .035 * torch.tanh(1.4*sensor[:, :1] - .8*sensor[:, 1:2])
    target = (sensor + torch.cat((nonlinear, -nonlinear), 1)
              + action*torch.tensor([[.18, -.11]])
              + dt*torch.tensor([[.02, -.015]]))
    return history, sensor, action, dt, target

def forward(model, batch):
    if isinstance(model, NativeContinuousHybrid):
        return model.step(batch[1], batch[2], batch[3])[0]
    return model(batch[0], batch[1], batch[2], batch[3])

def evaluate(model, data):
    with torch.no_grad():
        yhat = forward(model, data)
        y = data[4]
        effect = yhat[1::2, 0] - yhat[::2, 0]
        truth = y[1::2, 0] - y[::2, 0]
        significant = truth.abs() >= SIGN_FLOOR
        eligible = significant & (effect.abs() >= SIGN_FLOOR)
        return {
            "validation_mse": float(F.mse_loss(yhat, y)),
            "static_validation_mse": float(F.mse_loss(data[1], y)),
            "effect_sign_accuracy": float(((effect*truth) > 0)[eligible].float().mean()) if eligible.any() else 0.0,
            "effect_eligible_coverage": float(eligible.sum()/significant.sum()) if significant.any() else 0.0,
            "effect_mae": float((effect-truth).abs().mean()),
        }

def train_arm(arm, train, valid):
    if arm not in ARMS:
        raise ValueError("P1_DEV_UNREGISTERED_ARM")
    torch.manual_seed(DEV_INITIALIZATION + ARMS.index(arm))
    if arm == ARMS[0]:
        model = ConventionalAnalogGRU(2, 1, width=12).cpu()
    elif arm == ARMS[1]:
        model = ConventionalAnalogSSM(2, 1, width=12).cpu()
    else:
        model = NativeContinuousHybrid(2, 1, width=12).cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=.009, weight_decay=.001)
    t0 = time.monotonic()
    c0 = time.process_time()
    curve = []
    for step in range(1, MAX_UPDATES+1):
        start = ((step-1)*BATCH) % train[0].shape[0]
        indices = torch.arange(start, start+BATCH) % train[0].shape[0]
        sample = tuple(t[indices] for t in train)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(forward(model, sample), sample[4])
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            curve.append({"step": step, "validation_mse": evaluate(model, valid)["validation_mse"]})
    m = evaluate(model, valid)
    recent = [row["validation_mse"] for row in curve[-5:]]
    spread = (max(recent)-min(recent))/min(recent)
    passed = spread <= .01 and m["validation_mse"] < m["static_validation_mse"] and m["effect_sign_accuracy"] >= .95 and m["effect_eligible_coverage"] >= .95
    return {
        "arm":arm,
        "status": "DEV_NUMERIC_ADMISSIBLE" if passed else "DEV_BASELINE_NOT_QUALIFIED",
        "updates":MAX_UPDATES,
        "unique_train_examples":len(train[0]),
        "validation_examples":len(valid[0]),
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "curve":curve,
        "five_check_relative_spread":spread,
        "metrics":m,
        "cpu_seconds":time.process_time()-c0,
        "wall_seconds":time.monotonic()-t0,
        "science_first_use":"NOT_RUN",
        "native_online_fast_memory":"NOT_USED" if arm==ARMS[2] else "NOT_APPLICABLE",
    }

def run():
    if os.getenv("NATIVE_P1_DEV_FORBID_SCIENCE", "1") != "1":
        raise RuntimeError("P1_DEV_SCIENCE_FENCE_INVALID")
    torch.set_num_threads(1)
    train, val = synthetic_pairs(fit=True), synthetic_pairs(fit=False)
    if train[0].shape != (126, 4, 4) or val[0].shape != (96, 4, 4):
        raise AssertionError("P1_DEV_FIXTURE_DRIFT")
    if not (torch.equal(train[1][::2],train[1][1::2]) and torch.equal(train[0][::2],train[0][1::2]) and torch.equal(train[2][::2],-train[2][1::2])):
        raise AssertionError("P1_DEV_CAUSAL_PAIR_DRIFT")
    digest = hashlib.sha256()
    for role, tensors in (("development_fit",train),("development_val",val)):
        digest.update(role.encode())
        for t in tensors:
            digest.update(str(tuple(t.shape)).encode())
            digest.update(t.contiguous().numpy().tobytes())
    return {
        "schema":"ttt-hybrid.native-p1.analytic-development-fit.v1",
        "status":"DEV_RUN_FINISHED_NOT_SCIENCE",
        "qualification":"DEVELOPMENT_ONLY",
        "source":"PUBLIC_ANALYTIC_FIXTURE_NOT_SCIENCE",
        "torch":torch.__version__,
        "python":platform.python_version(),
        "fixture_initialization":DEV_INITIALIZATION,
        "analytic_data_sha256":digest.hexdigest(),
        "science_seeds_consumed":[],
        "science_worlds_created":False,
        "heldout_transfer":"NOT_ACCESSED",
        "arms":[train_arm(arm,train,val) for arm in ARMS],
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    report=run()
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__":
    main()
