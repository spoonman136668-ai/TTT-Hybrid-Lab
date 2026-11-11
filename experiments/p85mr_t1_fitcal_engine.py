"""P85-MR-T1 exact-sampler variable-length baseline FIT/CAL core.

NO external Causal-Gymnasium import, data loader, env reset/do, OOD selector,
runner, CLI, file writes, or scientific first-use. The only exported
training call is a bounded 2..6-update *toy engineering* diagnostic.
The 800-update owner FIT/CAL first use is explicitly denied here.

Frozen before implementation: p85mr_t1_fitcal_engine_freeze.json.
"""
import copy
import hashlib

import torch

from p85mr_dynamics_baselines import (
    KINDS, SEEDS, _validate_factual, component_errors, model_metadata,
    new_baseline, prequential,
)

MAX_UPDATES = 800
BATCH = 16
SAMPLE_SEED = 985299
CHECK_EVERY = 50
LR = {"linear7": .005, "mlp7": .005, "gru256": .001, "ssm256": .001}


def _episodes(items):
    """Snapshot pure factual tensor input; never accept oracle or query fields."""
    if not isinstance(items, (list, tuple)) or len(items) < 2:
        raise ValueError("P85T1_BAD_COHORT")
    frozen = []
    for episode in items:
        if not isinstance(episode, dict) or set(episode) != {"action", "effect"}:
            raise ValueError("P85T1_NONFACTUAL_EPISODE")
        a, y = episode["action"], episode["effect"]
        if (not isinstance(a, torch.Tensor) or
                not isinstance(y, torch.Tensor) or
                a.ndim != 2 or a.shape != y.shape or
                a.shape[1] != 16 or
                not 2 <= a.shape[0] <= 96 or
                a.dtype != torch.float32 or y.dtype != torch.float32 or
                a.device.type != "cpu" or y.device.type != "cpu"):
            raise ValueError("P85T1_BAD_EPISODE_SHAPE")
        aa, yy = a.detach().clone(), y.detach().clone()
        _validate_factual({"action": aa.unsqueeze(0),
                           "effect": yy.unsqueeze(0)})
        frozen.append({"action": aa, "effect": yy})
    return tuple(frozen)


def _fingerprint(episode):
    h = hashlib.sha256()
    for key in ("action", "effect"):
        t = episode[key].contiguous()
        h.update(key.encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _check_fit_cal(fit, cal):
    train, validation = _episodes(fit), _episodes(cal)
    hf, hc = [_fingerprint(x) for x in train], [_fingerprint(x) for x in validation]
    if len(set(hf)) != len(hf) or len(set(hc)) != len(hc):
        raise ValueError("P85T1_DUPLICATE_EPISODE")
    if set(hf) & set(hc):
        raise ValueError("P85T1_FIT_CAL_CONTENT_LEAKAGE")
    return train, validation


def frozen_indices(n_fit):
    """Identical 800x16 CPU index stream for every architecture and seed."""
    if type(n_fit) is not int or not 2 <= n_fit <= 128:
        raise ValueError("P85T1_BAD_FIT_COUNT")
    generator = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED)
    return torch.randint(n_fit, (MAX_UPDATES, BATCH), generator=generator)


def _padded(rows):
    """Valid steps must NEVER be trained as if future observations were known."""
    n = len(rows)
    if n == 0 or n > BATCH:
        raise ValueError("P85T1_BAD_BATCH")
    length = max(len(row["action"]) for row in rows)
    actions = torch.zeros(n, length, 16, dtype=torch.float32)
    actions[:, :, 0] = 1.0   # unit sentinel: never in objective
    effects = torch.zeros_like(actions)
    mask = torch.zeros(n, length, dtype=torch.bool)
    for k, row in enumerate(rows):
        t = len(row["action"])
        actions[k, :t] = row["action"]
        effects[k, :t] = row["effect"]
        mask[k, 1:t] = True
    return {"action": actions, "effect": effects}, mask


def _physical_loss(model, batch, valid):
    """Masked P85 prequential loss; no first event, no padded event/channel."""
    a, y = _validate_factual(batch)
    if valid.shape != a.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("P85T1_BAD_MASK")
    if bool(valid[:, 0].any()) or not bool(valid.any()):
        raise ValueError("P85T1_MASK_LEAKAGE")
    pred = prequential(model, batch)
    errors = component_errors(pred, y)
    v = valid.to(dtype=pred.dtype)
    denom = v.sum()
    delta = (errors["deltas"] * v).sum() / denom
    reward = (errors["reward"] * v).sum() / denom
    terminated = (errors["terminated"] * v).sum() / denom
    truncated = (errors["truncated"] * v).sum() / denom
    loss = delta + .2 * reward + .1 * terminated + .1 * truncated
    if not bool(torch.isfinite(loss)):
        raise ValueError("P85T1_NONFINITE_LOSS")
    return loss, delta


def calibration_mse(model, cal):
    """All worlds weigh equally regardless of length, CAL only."""
    validated = _episodes(cal)
    score = []
    zero = []
    with torch.no_grad():
        for first in range(0, len(validated), BATCH):
            batch, valid = _padded(validated[first:first + BATCH])
            observed = batch["effect"]
            preds = prequential(model, batch)
            mse = component_errors(preds, observed)["deltas"]
            base = observed[..., :4].square().mean(-1)
            denom = valid.sum(1).clamp_min(1)
            mask = valid.to(mse.dtype)
            score.extend(((mse * mask).sum(1) / denom).tolist())
            zero.extend(((base * mask).sum(1) / denom).tolist())
    return {"cal_delta_mse": sum(score) / len(score),
            "zero_cal_delta_mse": sum(zero) / len(zero),
            "worlds": len(validated),
            "science_pass": False}


def _best_checkpoint(current, best, selected_step, current_step):
    """Strict improvement only, so an exact tie retains earliest checkpoint."""
    if (not isinstance(current, float) or current < 0 or
            not isinstance(best, float) or current != current):
        raise ValueError("P85T1_INVALID_CAL")
    if current < best:
        return current, current_step
    return best, selected_step


def _toy_engineering_core(kind, seed, fit, cal, *, updates, every):
    if (kind not in KINDS or seed not in SEEDS or type(updates) is not int
            or not 2 <= updates <= 6 or type(every) is not int
            or not 1 <= every <= 3 or updates % every != 0):
        raise ValueError("P85T1_ENGINEERING_BOUNDS")
    train, validation = _check_fit_cal(fit, cal)
    sampler = frozen_indices(len(train))
    model = new_baseline(kind, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[kind],
                            weight_decay=.01)
    best = float("inf")
    checkpoint = None
    chosen_step = None
    log = []
    for step in range(1, updates + 1):
        selected = [train[i] for i in sampler[step - 1].tolist()]
        data, valid = _padded(selected)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, delta = _physical_loss(model, data, valid)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if (not all(g is not None and bool(torch.isfinite(g).all())
                    for g in grads)):
            raise ValueError("P85T1_NONFINITE_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0:
            model.eval()
            metrics = calibration_mse(model, validation)
            new, winner = _best_checkpoint(
                float(metrics["cal_delta_mse"]), best, chosen_step, step)
            if winner == step:
                best = new
                chosen_step = step
                checkpoint = copy.deepcopy(model.state_dict())
            log.append({"update": step, "cal_delta_mse": metrics["cal_delta_mse"]})
    if checkpoint is None or chosen_step is None:
        raise ValueError("P85T1_NO_CHECKPOINT")
    model.load_state_dict(checkpoint)
    model.eval()
    selected_cal = calibration_mse(model, validation)
    h = hashlib.sha256(sampler[:updates].contiguous().numpy().tobytes()).hexdigest()
    return model, {
        "schema": "ttt-hybrid.p85mr-t1-toy-fitcal-engineering-receipt.v1",
        "classification": "TOY_ONLY_NOT_AUTHORIZED_SCIENCE",
        "kind": kind, "model_seed": seed, "steps": updates,
        "selected_update": chosen_step, "checkpoint_count": len(log),
        "checkpoints": log, "best_cal_delta_mse": best,
        "restored_cal_delta_mse": selected_cal["cal_delta_mse"],
        "zero_cal_delta_mse": selected_cal["zero_cal_delta_mse"],
        "state_floats": model_metadata(model)["persistent_floats_per_world"],
        "fit_worlds": len(train), "cal_worlds": len(validation),
        "sample_index_sha256": h,
        "full_800_update_protocol_completed": False,
        "fit_cal_external_source_accessed": False,
        "ood_accessed": False, "science_pass": False,
    }


def toy_diagnostic(kind, seed, fit, cal, *, updates=2, every=2):
    """Engineering-only mini-run. Never counts as conventional competence."""
    return _toy_engineering_core(kind, seed, fit, cal, updates=updates, every=every)


def full_external_fitcal(*_args, **_kwargs):
    """An exact project-owner witness/permission path is not implemented here."""
    raise PermissionError("P85T1_OWNER_EXTERNAL_FIRST_USE_REQUIRED")
