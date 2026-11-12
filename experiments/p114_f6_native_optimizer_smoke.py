"""P114 source-neutral *synthetic-only* native F4 SSM optimizer smoke.

Eight real optimizer updates on fabricated 24+8 factual trajectories; no
external source, no F6 policy, no trained external checkpoint or science.
Not a callable owner trainer; only a bounded deterministic engineering test.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from experiments import p106_f6_source_neutral_policies as p106
from experiments import p112_f6_holdout_preflight as p112
from experiments import p113_f6_normalized_ssm_preflight as p113

ROOT = Path(__file__).resolve().parents[1]
FROZEN_BLOB = "deaa70f86ea96c4849f03301f98548f92e4d89a7"
PARENT_SHA = "8827d31c77110daa17530626dc9281bc2c600a91"
TOY_UPDATES = 8
SEEDS = (985201, 985203, 985207)

# The original P85/P69 modules use pre-existing top-level legacy imports.
legacy = str(ROOT / "experiments")
if legacy not in sys.path:
    sys.path.insert(0, legacy)
from p85mr_dynamics_baselines import (  # noqa: E402
    new_baseline, factual_objective, prequential, model_metadata,
)


def locked():
    data = (ROOT / "experiments/p114_f6_native_optimizer_smoke_freeze.json").read_bytes()
    if p106._git_blob(data) != FROZEN_BLOB:
        raise ValueError("P114_FROZEN_BLOB_DRIFT")
    f = json.loads(data)
    if (f.get("schema") != "ttt-hybrid.p114-f6-source-neutral-native-optimizer-smoke.v1"
        or f["parent"]["exact_sha"] != PARENT_SHA
        or f["toy"]["training_worlds"] != 24
        or f["toy"]["validation_worlds"] != 8
        or f["toy"]["seeds"] != list(SEEDS)
        or f["toy"]["optimizer"] != "AdamW"
        or f["toy"]["updates_per_seed_smoke"] != TOY_UPDATES
        or f["toy"]["trained_on_validation_data"] is not False
        or f["authority"]["external_F6_first_use_authorized"] is not False
        or f["authority"]["owner_source_sessions"] != 0
        or f["authority"]["source_effects"] != 0
        or f["authority"]["scientific_pass"] is not False
        or f["authority"]["protocol_P106A_B_resolved"] is not False):
        raise ValueError("P114_FROZEN_SCOPE_DRIFT")
    p113.frozen()
    return f


def fictional_events(block, index):
    if type(block) is not int or block not in (0, 1) or type(index) is not int or not 0 <= index <= 15:
        raise ValueError("P114_NOT_A_TOY_WORLD")
    state = [.012 * (index - 8), .015 + .002 * block,
             .006 * ((index % 5) - 2), -.009 + .002 * block]
    events = []
    for step in range(3):
        action = (index + step + block) % 2
        sign = 1.0 if action else -1.0
        after = [state[0] + .013 * sign,
                 state[1] + .009 * sign,
                 state[2] + .003 * sign,
                 state[3] + .007 * sign]
        events.append(p113.fixture_event(step + 1, action, state, after,
                                          reward=1.0 + .1 * step,
                                          terminated=step == 2))
        state = after
    return events


def fictional_corpus():
    """Only internally generated fabricated outcomes; accepts no source path."""
    locked()
    training_meta = [p112.fixture_train_row(block, index, 3)
                     for block, index in p112.TRAIN_IDS]
    heldout = [p113.fixture_world(index, fictional_events(1, index))
               for index in range(8, 16)]
    p113.validate_complete_fixture(training_meta, heldout)
    training = [(block, index, fictional_events(block, index))
                for block, index in p112.TRAIN_IDS]
    return training, heldout


def _facts(episodes):
    a, y = [], []
    for events in episodes:
        a.append([p106.action_key(e["state_before"], e["executed_action"])
                  for e in events])
        y.append([p113.label_from_fact(
            e["state_before"], e["state_after"], e["reward"],
            e["terminated"], e["truncated"]) for e in events])
    return {"action": torch.tensor(a, dtype=torch.float32, device="cpu"),
            "effect": torch.tensor(y, dtype=torch.float32, device="cpu")}


def _fingerprint(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _cold_npz(model, seed, validation):
    arrays = {key: tensor.detach().cpu().numpy()
              for key, tensor in model.state_dict().items()}
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise ValueError("P114_COLD_CHECKPOINT_KEYS")
        restored = {key: torch.from_numpy(archive[key].copy())
                    for key in arrays}
    cold = new_baseline("ssm256", seed).cpu().eval()
    cold.load_state_dict(restored, strict=True)
    with torch.no_grad():
        hot_predictions = prequential(model.eval(), validation)
        cold_predictions = prequential(cold, validation)
    if not torch.equal(hot_predictions, cold_predictions) or _fingerprint(model) != _fingerprint(cold):
        raise ValueError("P114_COLD_CHECKPOINT_PREDICTION_DRIFT")
    return hashlib.sha256(data).hexdigest(), _fingerprint(cold)


def smoke_seed(seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("P114_UNREGISTERED_SEED")
    train, validation = fictional_corpus()
    train_facts = _facts([events for _, _, events in train])
    valid_facts = _facts([w["events"] for w in validation])
    if train_facts["action"].shape != (24, 3, 16) or valid_facts["action"].shape != (8, 3, 16):
        raise ValueError("P114_TOY_SHAPE")
    model = new_baseline("ssm256", seed).cpu()
    info = model_metadata(model)
    if info["kind"] != "ssm256" or info["persistent_floats_per_world"] != 256:
        raise ValueError("P114_NOT_ORIGINAL_SSM")
    origin = _fingerprint(model)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    model.eval()
    with torch.no_grad():
        initial_train = float(factual_objective(model, train_facts)["loss"])
        initial_val = float(factual_objective(model, valid_facts)["loss"])
    for _ in range(TOY_UPDATES):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = factual_objective(model, train_facts)["loss"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("P114_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if not grads or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError("P114_INVALID_GRADIENT")
        if not any(bool((g.abs() > 0).any()) for g in grads):
            raise ValueError("P114_ZERO_ALL_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        final_train = float(factual_objective(model, train_facts)["loss"])
        final_val = float(factual_objective(model, valid_facts)["loss"])
    if (origin == _fingerprint(model) or
        not all(math.isfinite(x) for x in (initial_train, final_train, initial_val, final_val))):
        raise ValueError("P114_NO_FINITE_PARAMETER_UPDATE")
    checkpoint_sha, restored_sha = _cold_npz(model, seed, valid_facts)
    return {
        "schema": "ttt-hybrid.p114-native-f4-synthetic-optimizer-smoke.v1",
        "seed": seed, "original_model": info["kind"],
        "train_worlds_fake": 24, "validation_worlds_fake": 8,
        "adamw_updates_fake": TOY_UPDATES,
        "initial_train_f4_loss_fake": initial_train,
        "postupdate_train_f4_loss_fake": final_train,
        "initial_validation_f4_loss_fake": initial_val,
        "postupdate_validation_f4_loss_fake": final_val,
        "initial_weight_sha256": origin,
        "trained_toy_weight_sha256": restored_sha,
        "checkpoint_npz_sha256": checkpoint_sha,
        "checkpoint_cold_prediction_exact": True,
        "real_F6_source_sessions": 0, "real_F6_effects": 0,
        "real_F6_training": False,
        "trained_source_checkpoint_authenticated": False,
        "scientific_pass": False, "deployment": False,
    }


def refuse_source(*_args, **_kwargs):
    raise PermissionError("P114_F6_EXTERNAL_FIRST_USE_NOT_AUTHORIZED")
