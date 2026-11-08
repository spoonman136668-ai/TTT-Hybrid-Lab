"""C2 DEV-only candidate isolation; deliberately no scientific model promotion."""
from copy import deepcopy
import hashlib
import random

import pytest
import torch

import native_transactional_resume as tx
from native_training_checkpoint import (
    CheckpointRejected, save_checkpoint, validate_chain
)


def identity():
    fields = ("architecture_sha256", "interface_sha256", "protocol_sha256",
              "data_split_sha256", "harness_sha256", "environment_sha256")
    return {"source_sha": "c" * 40,
            **{k: hashlib.sha256(("c2:"+k).encode()).hexdigest()
               for k in fields}}


def make_pair():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=.004)
    return model, opt


def fixture(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    model, opt = make_pair()
    memory = {"recurrent": torch.zeros(1, 2),
              "fast_weights": torch.zeros(1, 2, 2),
              "history": torch.zeros(1, 0, 2)}
    root = save_checkpoint(directory, model=model, optimizer=opt,
                           identity=identity(), native_state=memory,
                           sampler_state={"position": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    x = torch.tensor([[.3, .8]])
    opt.zero_grad(set_to_none=True)
    model(x).square().sum().backward()
    opt.step()
    memory["fast_weights"].add_(.25)
    tip = save_checkpoint(directory, model=model, optimizer=opt,
                          identity=identity(), native_state=memory,
                          sampler_state={"position": 1}, steps=1,
                          consumed_examples=1, parent_manifest_sha256=root)
    return directory, model, opt, root, tip


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(
            equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(
            equal(x, y) for x, y in zip(a, b))
    return a == b


def prepare(directory, tip, **kwargs):
    return tx.prepare_isolated_candidate(
        directory, tip_sha256=tip, identity=kwargs.pop("identity", identity()),
        make_fresh_pair=kwargs.pop("make_fresh_pair", make_pair),
        expected_position=kwargs.pop("expected_position", 1), **kwargs)


def test_protected_champion_and_global_rng_unchanged_on_success(tmp_path):
    directory, champion, old_optimizer, root, tip = fixture(tmp_path)
    champion_before = deepcopy(champion.state_dict())
    optimizer_before = deepcopy(old_optimizer.state_dict())
    torch_before = torch.get_rng_state().clone()
    python_before = deepcopy(random.getstate())
    prepared = prepare(directory, tip, protected_model=champion,
                       protected_optimizer=old_optimizer,
                       expected_parent_sha256=root)
    assert prepared.status == "ISOLATED_CANDIDATE_NOT_ACCEPTED"
    assert prepared.steps == prepared.consumed_examples == 1
    assert prepared.sampler == {"position": 1}
    assert prepared.checkpoint_sha256 == tip
    assert prepared.model is not champion
    assert prepared.optimizer is not old_optimizer
    assert equal(prepared.model.state_dict(), champion_before)
    assert equal(prepared.optimizer.state_dict(), optimizer_before)
    assert equal(champion.state_dict(), champion_before)
    assert equal(old_optimizer.state_dict(), optimizer_before)
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    assert not (directory / "accepted").exists()
    assert not (directory / "latest").exists()


def test_trained_cursor_cannot_fall_back_to_root_or_wrong_parent(tmp_path):
    directory, champion, opt, root, tip = fixture(tmp_path)
    with pytest.raises(tx.CandidateRestoreRejected,
                       match="RESETS_TRAINING_PROGRESS"):
        prepare(directory, root)
    with pytest.raises(tx.CandidateRestoreRejected, match="PARENT_MISMATCH"):
        prepare(directory, tip, expected_parent_sha256="f" * 64)
    with pytest.raises(tx.CandidateRestoreRejected, match="CURSOR_MISMATCH"):
        prepare(directory, tip, expected_position=2)
    with pytest.raises(tx.CandidateRestoreRejected, match="NOT_IMMUTABLE"):
        prepare(directory, "latest")


def test_factory_must_not_return_protected_objects(tmp_path):
    directory, champion, opt, root, tip = fixture(tmp_path)
    champion_before = deepcopy(champion.state_dict())
    with pytest.raises(tx.CandidateRestoreRejected,
                       match="IS_PROTECTED_INSTANCE"):
        prepare(directory, tip, make_fresh_pair=lambda: (champion, opt),
                protected_model=champion, protected_optimizer=opt)
    assert equal(champion_before, champion.state_dict())


def test_factory_rejects_optimizer_bound_to_another_model(tmp_path):
    directory, champion, opt, root, tip = fixture(tmp_path)
    def wrong():
        model, _ = make_pair()
        return model, opt
    with pytest.raises(tx.CandidateRestoreRejected,
                       match="OPTIMIZER_REFERENCES_OTHER_MODEL"):
        prepare(directory, tip, make_fresh_pair=wrong)


def test_mid_restore_failure_does_not_touch_champion_or_rng(tmp_path, monkeypatch):
    directory, champion, opt, root, tip = fixture(tmp_path)
    champion_before = deepcopy(champion.state_dict())
    optimizer_before = deepcopy(opt.state_dict())
    torch_before = torch.get_rng_state().clone()
    python_before = deepcopy(random.getstate())
    def partial_then_fail(directory, tip, *, model, optimizer, identity):
        for param in model.parameters():
            with torch.no_grad():
                param.add_(15.0)
        torch.manual_seed(11411)
        random.seed(11411)
        raise RuntimeError("INJECTED_AFTER_PARTIAL_CANDIDATE_MUTATION")
    monkeypatch.setattr(tx, "load_checkpoint", partial_then_fail)
    with pytest.raises(RuntimeError, match="INJECTED_AFTER_PARTIAL"):
        prepare(directory, tip, protected_model=champion,
                protected_optimizer=opt)
    assert equal(champion.state_dict(), champion_before)
    assert equal(opt.state_dict(), optimizer_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert python_before == random.getstate()


def test_tampered_checkpoint_rejected_before_candidate_factory(tmp_path):
    directory, champion, opt, root, tip = fixture(tmp_path)
    (directory / (tip + ".json")).write_bytes(b"tampered")
    called = []
    def fake():
        called.append(True)
        return make_pair()
    with pytest.raises(CheckpointRejected, match="MANIFEST_TAMPERED"):
        prepare(directory, tip, make_fresh_pair=fake)
    assert called == []


def test_rng_restored_even_on_bad_factory_shape(tmp_path):
    directory, champion, opt, root, tip = fixture(tmp_path)
    before = torch.get_rng_state().clone()
    with pytest.raises(tx.CandidateRestoreRejected,
                       match="FACTORY_CONTRACT"):
        prepare(directory, tip, make_fresh_pair=lambda: (make_pair()[0],))
    assert torch.equal(before, torch.get_rng_state())
