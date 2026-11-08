"""Synthetic engineering only: no original scientific data or model-training run."""
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import shutil
import torch
import pytest
from native_training_checkpoint import (CheckpointRejected, load_checkpoint,
                                        save_checkpoint, validate_chain)


def identity():
    return {"source_sha": "a" * 40, **{k: hashlib.sha256(k.encode()).hexdigest() for k in
        ("architecture_sha256", "interface_sha256", "protocol_sha256", "data_split_sha256", "harness_sha256", "environment_sha256")}}


def fixture():
    m = torch.nn.Linear(2, 2)
    o = torch.optim.AdamW(m.parameters(), lr=0.01)
    s = {"recurrent": torch.ones(1, 3), "fast_weights": torch.ones(1, 2, 3), "history": torch.ones(1, 1, 3)}
    return m, o, s


def synthetic_step(m, o):
    x = torch.ones(1, 2)
    o.zero_grad(); m(x).square().sum().backward(); o.step()


def chain(tmp_path):
    m, o, s = fixture()
    root = save_checkpoint(tmp_path, model=m, optimizer=o, identity=identity(), native_state=s,
                           sampler_state={"position": 0}, steps=0, consumed_examples=0,
                           parent_manifest_sha256=None)
    synthetic_step(m, o)
    s["fast_weights"] += 0.5
    tip = save_checkpoint(tmp_path, model=m, optimizer=o, identity=identity(), native_state=s,
                          sampler_state={"position": 5}, steps=1, consumed_examples=5,
                          parent_manifest_sha256=root)
    return (m, o, s), root, tip


def test_resume_weights_optimizer_rng_memory_and_cursor_exact(tmp_path):
    (m, o, s), root, tip = chain(tmp_path)
    expected_model = deepcopy(m.state_dict()); expected_optim = deepcopy(o.state_dict())
    next_random = (random.random(), torch.rand(2))
    cold_m, cold_o, _ = fixture()
    restore_state, sampler = load_checkpoint(tmp_path, tip, model=cold_m, optimizer=cold_o, identity=identity())
    for name, tensor in expected_model.items():
        torch.testing.assert_close(cold_m.state_dict()[name], tensor, atol=0, rtol=0)
    assert cold_o.state_dict()["state"].keys() == expected_optim["state"].keys()
    for state_id, tensors in expected_optim["state"].items():
        for name, tensor in tensors.items():
            torch.testing.assert_close(cold_o.state_dict()["state"][state_id][name], tensor, atol=0, rtol=0)
    torch.testing.assert_close(restore_state["fast_weights"], s["fast_weights"])
    assert sampler["position"] == 5
    assert random.random() == next_random[0]
    torch.testing.assert_close(torch.rand(2), next_random[1], atol=0, rtol=0)
    assert validate_chain(tmp_path, tip, identity())["steps"] == 1


def test_reset_from_scratch_after_training_rejected(tmp_path):
    m, o, s = fixture()
    with pytest.raises(CheckpointRejected, match="UNDECLARED_RESTART"):
        save_checkpoint(tmp_path, model=m, optimizer=o, identity=identity(), native_state=s,
                        sampler_state={"position": 1}, steps=1, consumed_examples=1,
                        parent_manifest_sha256=None)


def test_no_replaying_parent_or_training_cursor(tmp_path):
    (_, _, _), root, tip = chain(tmp_path)
    m, o, s = fixture()
    with pytest.raises(CheckpointRejected, match="NONADVANCING_RESTART"):
        save_checkpoint(tmp_path, model=m, optimizer=o, identity=identity(), native_state=s,
                        sampler_state={"position": 1}, steps=1, consumed_examples=5,
                        parent_manifest_sha256=tip)


def test_tampered_blob_and_manifest_rejected(tmp_path):
    _, root, tip = chain(tmp_path)
    blob = next(tmp_path.glob("*.pt"))
    blob.write_bytes(blob.read_bytes() + b"tamper")
    with pytest.raises(CheckpointRejected, match="PAYLOAD_TAMPERED"):
        validate_chain(tmp_path, tip, identity())


def test_source_and_architecture_drift_rejected(tmp_path):
    _, root, tip = chain(tmp_path)
    wrong = identity(); wrong["source_sha"] = "b" * 40
    with pytest.raises(CheckpointRejected, match="IDENTITY_OR_CURSOR_DRIFT"):
        validate_chain(tmp_path, tip, wrong)
    wrong = identity(); wrong["architecture_sha256"] = "f" * 64
    with pytest.raises(CheckpointRejected, match="IDENTITY_OR_CURSOR_DRIFT"):
        validate_chain(tmp_path, tip, wrong)


def test_missing_parent_and_truncated_memory_rejected(tmp_path):
    (_, _, _), root, tip = chain(tmp_path)
    (tmp_path / (root + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        validate_chain(tmp_path, tip, identity())
    m, o, s = fixture()
    with pytest.raises(CheckpointRejected, match="NATIVE_STATE_INCOMPLETE"):
        save_checkpoint(tmp_path, model=m, optimizer=o, identity=identity(),
                        native_state={"fast_weights": s["fast_weights"]},
                        sampler_state={"position": 0}, steps=0, consumed_examples=0,
                        parent_manifest_sha256=None)


def test_candidate_save_does_not_advance_accepted_pointer(tmp_path):
    _, root, tip = chain(tmp_path)
    assert not (tmp_path / "latest").exists()
    assert not (tmp_path / "accepted").exists()
    assert root != tip


def test_interrupted_training_equals_uninterrupted_with_optimizer_scheduler_and_dropout(tmp_path):
    """Actual synthetic optimizer continuation, not merely weight reloading."""
    def setup():
        net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Dropout(.2),
                                  torch.nn.Linear(8, 2))
        opt = torch.optim.AdamW(net.parameters(), lr=.008)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=.7)
        return net, opt, sch

    def train(net, opt, sch, start, end):
        x = torch.tensor([[.1, .3], [.6, -.2], [-.4, .9], [.8, .2]])
        y = x * 1.7 - .3
        for _ in range(start, end):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(net(x), y).backward()
            opt.step()
            sch.step()

    torch.manual_seed(7342)
    net, opt, sch = setup()
    state = {"recurrent": torch.zeros(1, 3), "fast_weights": torch.zeros(1, 2, 3),
             "history": torch.zeros(1, 0, 3)}
    root = save_checkpoint(tmp_path, model=net, optimizer=opt, scheduler=sch,
                           native_state=state, identity=identity(),
                           sampler_state={"cursor": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    train(net, opt, sch, 0, 20)
    full = deepcopy(net.state_dict())
    full_optimizer = deepcopy(opt.state_dict())
    full_scheduler = deepcopy(sch.state_dict())

    net, opt, sch = setup()
    _, sampler = load_checkpoint(tmp_path, root, model=net, optimizer=opt,
                                 scheduler=sch, identity=identity())
    assert sampler["cursor"] == 0
    train(net, opt, sch, 0, 10)
    checkpoint = save_checkpoint(tmp_path, model=net, optimizer=opt, scheduler=sch,
                                 native_state=state, identity=identity(),
                                 sampler_state={"cursor": 10}, steps=10,
                                 consumed_examples=40, parent_manifest_sha256=root)
    net, opt, sch = setup()
    _, sampler = load_checkpoint(tmp_path, checkpoint, model=net,
                                 optimizer=opt, scheduler=sch, identity=identity())
    assert sampler["cursor"] == 10
    train(net, opt, sch, 10, 20)
    for k in full:
        torch.testing.assert_close(net.state_dict()[k], full[k], rtol=0, atol=0)
    assert opt.state_dict()["param_groups"] == full_optimizer["param_groups"]
    for slot, vals in full_optimizer["state"].items():
        for k, v in vals.items():
            torch.testing.assert_close(opt.state_dict()["state"][slot][k], v,
                                       rtol=0, atol=0)
    assert sch.state_dict() == full_scheduler
