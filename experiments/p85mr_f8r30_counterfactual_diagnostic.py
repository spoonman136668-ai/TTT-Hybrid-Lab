"""F8-R30: separate factual neural prediction from evaluator-only counterfactual scoring.

Source-neutral DEV diagnostics only. R29B's model, data, optimizer and update
budget are unchanged. This module never trains on an unselected effect.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from experiments import p85mr_f8r29b_neural_information as b

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r30_counterfactual_diagnostic_freeze.json"
BLOB = "c13f32fc0a0671808d1e5d1ab0ae1b05f43badb8"
SCHEMA = "ttt-hybrid.f8r30-counterfactual-diagnostic.v1"
ARMS = tuple((arm, seed) for arm in b.ARMS
             for seed in (b.SEEDS if arm == "native_dynamics" else b.SEEDS[:1]))


def locked():
    raw = FREEZE.read_bytes()
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if actual != BLOB:
        raise ValueError("F8R30_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_AND_RESULTS" or
        f["parent"]["sha"] != "dfb2176f04e661372aa8d250b1fafc154f6f552f" or
        f["parent"]["r29b_freeze_blob"] != b.FREEZE_BLOB or
        f["design"]["score_worlds_per_panel"] != 48 or
        f["design"]["updates_per_model"] != b.UPDATES or
        f["design"]["eval_panels"] != list(b.PANELS) or
        f["design"]["eval_policies"] != list(b.POLICIES) or
        f["design"]["arms"] != list(b.ARMS) or
        f["design"]["seeds_native"] != list(b.SEEDS) or
        f["design"]["seeds_other"] != [b.SEEDS[0]] or
        f["design"]["training_only_selected_factual_return"] is not True or
        f["design"]["unselected_action_effect_never_enters_training"] is not True or
        f["design"]["conventional_convergence_qualified"] is not False or
        any(v is not False for v in f["authority"].values())):
        raise ValueError("F8R30_PROTOCOL_OR_AUTHORITY_DRIFT")
    b.locked()
    return f


def _sample_indices(size):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20469999)
    return torch.randint(0, size, (b.UPDATES, b.BATCH),
                         generator=generator).tolist()


def _train(arm, seed, fit, samples):
    """Reproduce R29B's 128 factual selected-return AdamW updates."""
    model = b.model_for(arm, seed)
    digest = b.r25.r22.r18.r5.f5.hash_weights
    before = digest(model)
    optim = torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("native") else .001, weight_decay=.01)
    started = time.perf_counter()
    losses = []
    for ix in samples:
        batch = b._batch(fit, ix)
        model.train()
        optim.zero_grad(set_to_none=True)
        pred = b._predict(arm, model, {"action": batch["action"],
                                      "effect": batch["effect"]})
        if pred.shape != batch["effect"].shape:
            raise ValueError("F8R30_BAD_TRAIN_SHAPE")
        target = (batch["direct_target"] if arm == "native_direct_value"
                  else batch["physical_target"])
        loss = (pred[:, -1, 0] - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R30_NONFINITE_LOSS")
        losses.append(float(loss.detach()))
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g != 0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R30_INVALID_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
    after = digest(model)
    if before == after or len(losses) != b.UPDATES:
        raise ValueError("F8R30_NO_MODEL_UPDATE")
    return model, {"arm": arm, "seed": seed, "optimizer_updates": len(losses),
                   "initial_weight_hash": before, "trained_weight_hash": after,
                   "first_batch_loss": losses[0], "last_batch_loss": losses[-1],
                   "train_seconds": time.perf_counter() - started,
                   "parameters": sum(p.numel() for p in model.parameters())}


def _features(actions, effects):
    """Factual keys and selected fifth measurement; no evaluator labels."""
    if (actions.ndim != 3 or effects.shape != actions.shape or
        actions.shape[1] < 14 or actions.shape[2] != 16):
        raise ValueError("F8R30_BAD_FACTUAL_FEATURE_SHAPE")
    ret = actions[:, -1, :7].double()
    fifth = actions[:, 4, :7].double()
    value = effects[:, 4, :1].double()
    return torch.cat((torch.ones_like(value), ret, fifth, value,
                      value * ret, value * fifth, value * ret * fifth), dim=1)


def fit_only_ridge(fit):
    a = torch.stack([row["action"] for row in fit])
    e = torch.stack([row["effect"] for row in fit])
    y = torch.stack([row["physical_target"] for row in fit]).double()
    x = _features(a, e)
    regularization = .001 * torch.eye(x.shape[1], dtype=torch.float64)
    regularization[0, 0] = 0.
    w = torch.linalg.solve(x.T @ x + regularization, x.T @ y)
    if not bool(torch.isfinite(w).all()):
        raise ValueError("F8R30_INVALID_FIT_ONLY_RIDGE")
    return w


def pair_raw(arm, model, panel, policy, erase=False):
    if panel not in b.PANELS or policy not in b.POLICIES:
        raise ValueError("F8R30_INVALID_PANEL_OR_POLICY")
    inputs = [b._candidate_inputs(w, policy, erase)
              for w in b.r29.worlds(panel)]
    actions = torch.cat([c[0] for c in inputs])
    effects = torch.cat([c[1] for c in inputs])
    with torch.no_grad():
        if arm == "fit_only_ridge":
            out = _features(actions, effects) @ model
        else:
            model.eval()
            predicted = b._predict(arm, model, {"action": actions,
                                               "effect": effects})
            if predicted.shape != effects.shape:
                raise ValueError("F8R30_INVALID_PREACTION_SHAPE")
            out = predicted[:, -1, 0].double()
    if out.numel() != 96 or not bool(torch.isfinite(out).all()):
        raise ValueError("F8R30_INVALID_PAIR_OUTPUT")
    raw = out.cpu().tolist()
    return [(raw[2*i], raw[2*i + 1]) for i in range(48)]


def _physical(value, state):
    return state + 2.4 * math.atanh(max(-.98, min(.98, value)))


def _loss_pair(arm, pair, world):
    if arm == "native_direct_value":
        return (4 * pair[0], 4 * pair[1])
    return tuple((_physical(p, world["return_state"]) - world["goal"]) ** 2
                 for p in pair)


def _corr(xs, ys):
    a, c = statistics.fmean(xs), statistics.fmean(ys)
    den = math.fsum((v-a)**2 for v in xs) * math.fsum((v-c)**2 for v in ys)
    return (math.fsum((x-a)*(y-c) for x,y in zip(xs,ys)) / math.sqrt(den)
            if den > 1e-18 else None)


def score_evaluator_only(arm, worlds, raw, erased):
    """Unchosen return outcomes are calculated ONLY here, after inference."""
    if len(worlds) != 48 or len(raw) != 48 or len(erased) != 48:
        raise ValueError("F8R30_UNBALANCED_SCORER")
    physical_errors, value_errors = [], []
    true_gaps, predicted_gaps, response = [], [], []
    choices, oracle, erased_choices, regrets = [], [], [], []
    for w, prediction, no_measurement in zip(worlds, raw, erased):
        physical = [b.r29.after(w["return_state"], action,
                                w["public_context"], w["hidden_q_evaluator_only"])
                    for action in (0, 1)]
        actual = [(v - w["goal"])**2 for v in physical]
        forecast = _loss_pair(arm, prediction, w)
        erased_forecast = _loss_pair(arm, no_measurement, w)
        chosen = min((0, 1), key=lambda a: (forecast[a], a))
        erased_chosen = min((0, 1), key=lambda a: (erased_forecast[a], a))
        optimal = min((0, 1), key=lambda a: (actual[a], a))
        choices.append(chosen)
        erased_choices.append(erased_chosen)
        oracle.append(optimal)
        regrets.append(max(0., actual[chosen] - actual[optimal]))
        true_gaps.append(actual[1] - actual[0])
        predicted_gaps.append(forecast[1] - forecast[0])
        for a in (0, 1):
            value_errors.append((forecast[a] - actual[a])**2)
            response.append(abs(prediction[a] - no_measurement[a]))
            if arm != "native_direct_value":
                physical_errors.append(
                    (_physical(prediction[a], w["return_state"]) - physical[a])**2)
    return {
        "worlds": len(worlds),
        "physical_effect_mse": (statistics.fmean(physical_errors)
                                if physical_errors else None),
        "action_value_mse": statistics.fmean(value_errors),
        "choice_accuracy_vs_oracle": sum(a==o for a,o in zip(choices,oracle))/48,
        "action_gap_correlation": _corr(true_gaps, predicted_gaps),
        "action_gap_std": statistics.pstdev(predicted_gaps),
        "mean_raw_effect_change_with_erasure": statistics.fmean(response),
        "decision_changes_on_erasure": sum(a!=c for a,c in zip(
            choices, erased_choices)),
        "constant_action_collapse": len(set(choices)) == 1,
        "chosen_action1_count": sum(choices),
        "mean_regret": statistics.fmean(regrets),
        "choices": choices,
    }


def diagnose(arm, model):
    if arm not in b.ARMS and arm != "fit_only_ridge":
        raise ValueError("F8R30_INVALID_ARM")
    report = {}
    for panel in b.PANELS:
        worlds = b.r29.worlds(panel)
        erased = pair_raw(arm, model, panel, "active_information", True)
        result = {}
        for policy in b.POLICIES:
            raw = pair_raw(arm, model, panel, policy)
            result[policy] = score_evaluator_only(
                arm, worlds, raw, erased if policy == "active_information"
                else raw)
        report[panel] = result
    return report


def run():
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit = b.fit_cohort()
    if len(fit) != 3072:
        raise ValueError("F8R30_FIT_COHORT_DRIFT")
    samples = _sample_indices(len(fit))
    ridge = fit_only_ridge(fit)
    models = []
    for arm, seed in ARMS:
        model, receipt = _train(arm, seed, fit, samples)
        models.append({**receipt, "panels": diagnose(arm, model)})
    return {
        "schema": SCHEMA, "freeze_blob": BLOB,
        "classification": "SOURCE_NEUTRAL_DIAGNOSTIC_DEV_ONLY",
        "frozen_parent_sha": "dfb2176f04e661372aa8d250b1fafc154f6f552f",
        "factual_fit_episodes": len(fit),
        "factual_fit_effects": len(fit) * 14,
        "neural_models": models,
        "real_optimizer_updates": b.UPDATES * len(models),
        "fit_only_ridge": {"factual_fit_episodes": len(fit),
                           "feature_count": ridge.numel(),
                           "panels": diagnose("fit_only_ridge", ridge)},
        "conventional_independent_convergence_qualified": False,
        "matched_capacity_and_compute": False,
        "independent_scientific_pass": False,
        "original_external_effects": 0,
        "spent_F7_replay": False, "deployment": "NONE",
    }


def retain_ci_receipt(data):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r30_counterfactual_diagnostic_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R30_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(data, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def refuse_external_firstuse(*args, **kwargs):
    raise PermissionError("F8R30_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
