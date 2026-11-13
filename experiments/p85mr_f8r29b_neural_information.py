"""F8-R29B: neural use of one actually selected fifth-action consequence.

This is source-neutral DEVELOPMENT research, not external scientific first use.
Six genuinely updated models receive only actual selected factual effects. The
analytic R29 controller is a stronger known-class comparison, not neural evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from experiments import p85mr_f8r29_information_calibration as r29
from experiments import p85mr_f8r25_multimechanism as r25

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r29b_neural_information_freeze.json"
FREEZE_BLOB = "50629c2aee9aa0744cd2524c265b1dca30d107d5"
SCHEMA = "ttt-hybrid.f8r29b-neural-information-uptake.v1"
ARMS = ("native_dynamics", "native_direct_value", "gru_dynamics", "ssm_dynamics")
SEEDS = (985201, 985203, 985207)
POLICIES = r29.POLICIES
PANELS = r29.PANELS
FIT_GROUPS = 64
FIT_Q = (.75, 1.0, 1.15)
UPDATES = 128
BATCH = 8
DRAWS = 10000
BOOT_SEED = 20479999
GOAL = r29.GOAL


def _blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if _blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R29B_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    d, tr, sc, auth = f["design"], f["train"], f["score"], f["authority"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "FROZEN_BEFORE_LINKED_IMPLEMENTATION_TRAINING_AND_RESULTS"
        or f["classification"] != "SOURCE_NEUTRAL_NEURAL_INFORMATION_DEV_ONLY"
        or f["parent"]["head"] != "bc220a22495c2b4c718e3737c616f783e4f1f28a"
        or _blob((ROOT / "experiments/p85mr_f8r29_information_calibration.py").read_bytes())
           != f["parent"]["r29_source_blob"]
        or _blob((ROOT / "experiments/p85mr_f8r29_information_calibration_freeze.json").read_bytes())
           != f["parent"]["r29_calibration_freeze_blob"]
        or d["fit_groups"] != FIT_GROUPS
        or d["fit_q_abs"] != list(FIT_Q)
        or d["fit_contexts"] != [0, 1]
        or d["fit_q_signs"] != [-1, 1]
        or d["fit_return_actions"] != [0, 1]
        or d["fit_b_steps"] != 8
        or d["evaluation_panels"] != list(PANELS)
        or d["evaluation_groups"] != r29.GROUPS
        or d["B_steps"] != r29.B_STEPS
        or tr["arms"] != list(ARMS)
        or tr["seeds_native_dynamics"] != list(SEEDS)
        or tr["model_instances"] != 6
        or tr["updates_each"] != UPDATES
        or tr["total_updates"] != 6 * UPDATES
        or tr["batch_size"] != BATCH
        or sc["bootstrap_draws"] != DRAWS
        or sc["bootstrap_seed"] != BOOT_SEED
        or sc["independent_scientific_pass"] is not False
        or any(v is not False for v in auth.values())):
        raise ValueError("F8R29B_PROTOCOL_OR_AUTHORITY_DRIFT")
    r29.locked()
    return f


def _record(state: float, action: int, following: float,
            context: int) -> tuple[torch.Tensor, torch.Tensor]:
    if action not in (0, 1) or context not in (0, 1):
        raise ValueError("F8R29B_INVALID_FACTUAL_ACTION")
    if not math.isfinite(following):
        raise ValueError("F8R29B_NONFINITE_EFFECT")
    # The action key is the only model-visible context: no q, sign or group.
    key = r25.r22.r18.r5._action_key(
        (float(state), float(context), float(GOAL), 0.), action)
    effect = r25.r22.r18._observed(float(state), float(following))
    return (torch.tensor(key, dtype=torch.float32),
            torch.tensor(effect, dtype=torch.float32))


def _fit_B(group: int) -> tuple[tuple[int, float], ...]:
    if not 0 <= group < FIT_GROUPS:
        raise ValueError("F8R29B_FIT_GROUP_INVALID")
    # New FIT stream, independent of q, context and all R29 DEV/OOD nuisance histories.
    return tuple((t % 2, round(.009 * (group - 31) + .019 * (t % 5)
                                + .001 * math.sin((group + 64) * .17 + t), 8))
                 for t in range(8))


def _episode(group: int, context: int, sign: int, q_abs: float,
             fifth_action: int, return_action: int) -> dict:
    if (type(group) is not int or not 0 <= group < FIT_GROUPS
        or context not in (0, 1) or sign not in (-1, 1)
        or q_abs not in FIT_Q or fifth_action not in (0, 1)
        or return_action not in (0, 1)):
        raise ValueError("F8R29B_UNREGISTERED_FACTUAL_FIT")
    q = sign * q_abs  # Evaluator-owned; NEVER a model input.
    events = [_record(s, a, r29.after(s, a, context, q), context)
              for s, a in r29.EARLY]
    events.append(_record(r29.PROBE_STATE, fifth_action,
                          r29.after(r29.PROBE_STATE, fifth_action, context, q),
                          context))
    events.extend(_record(0., a, y, context) for a, y in _fit_B(group))
    last = r29.after(r29.RETURN_STATES["dev"], return_action, context, q)
    events.append(_record(r29.RETURN_STATES["dev"], return_action,
                          last, context))
    if len(events) != 14:
        raise ValueError("F8R29B_FACTUAL_FIT_LENGTH")
    actions = torch.stack([x for x, _ in events])
    effects = torch.stack([y for _, y in events])
    # Only actually selected return action's realized error is observed.
    direct_target = .25 * (last - GOAL) ** 2
    if (actions.shape != (14, 16) or effects.shape != (14, 16)
        or not bool(torch.isfinite(actions).all())
        or not bool(torch.isfinite(effects).all())
        or not 0. <= direct_target < 1.):
        raise ValueError("F8R29B_BAD_FACTUAL_TENSOR_OR_TARGET")
    return {"action": actions, "effect": effects,
            "physical_target": effects[-1, 0].clone(),
            "direct_target": torch.tensor(direct_target, dtype=torch.float32)}


def fit_cohort() -> list[dict]:
    locked()
    return [_episode(g, c, sign, mag, fifth, ret)
            for g in range(FIT_GROUPS)
            for c in (0, 1) for sign in (-1, 1) for mag in FIT_Q
            for fifth in (0, 1) for ret in (0, 1)]


def model_for(arm: str, seed: int):
    if arm not in ARMS or seed not in SEEDS or (
        arm != "native_dynamics" and seed != SEEDS[0]):
        raise ValueError("F8R29B_UNREGISTERED_MODEL")
    kind = ("native_focus20" if arm.startswith("native")
            else "gru_focus20" if arm.startswith("gru") else "ssm_focus20")
    return r25.r22.model_for(kind, seed).cpu()


def _kind(arm: str) -> str:
    return ("native_focus20" if arm.startswith("native")
            else "gru_focus20" if arm.startswith("gru") else "ssm_focus20")


def _predict(arm: str, model, data: dict) -> torch.Tensor:
    return r25.r22.predict(_kind(arm), model, data)


def _batch(fit: list[dict], indices: list[int]) -> dict:
    rows = [fit[i] for i in indices]
    return {"action": torch.stack([r["action"] for r in rows]),
            "effect": torch.stack([r["effect"] for r in rows]),
            "physical_target": torch.stack([r["physical_target"] for r in rows]),
            "direct_target": torch.stack([r["direct_target"] for r in rows])}


def _candidate_inputs(w: dict, policy: str,
                      erase: bool) -> tuple[torch.Tensor, torch.Tensor]:
    context = w["public_context"]
    fifth = r29.probe_action(policy, w["panel"], w["group"], context)
    # Only the actually chosen fifth action's physical result is computed here.
    observed = r29.after(r29.PROBE_STATE, fifth, context,
                         w["hidden_q_evaluator_only"])
    events = [_record(s, a, y, context) for s, a, y in w["early_actual"]]
    events.append(_record(r29.PROBE_STATE, fifth, observed, context))
    events.extend(_record(0., a, y, context) for a, y in w["b_history"])
    actions = torch.stack([x for x, _ in events])
    effects = torch.stack([y for _, y in events])
    if erase:
        effects[4].zero_()  # Same selected action and B; no fifth measurement.
    aa, ee = [], []
    for a in (0, 1):
        key, _ = _record(w["return_state"], a, w["return_state"], context)
        aa.append(torch.cat((actions, key.unsqueeze(0)), dim=0))
        ee.append(torch.cat((effects, torch.zeros((1, 16),
                                                dtype=torch.float32)), dim=0))
    return torch.stack(aa), torch.stack(ee)


def _choices(arm: str, model, panel: str, policy: str,
             erase: bool = False) -> list[int]:
    if panel not in PANELS or policy not in POLICIES:
        raise ValueError("F8R29B_INVALID_SCORE_PANEL_OR_POLICY")
    worlds = r29.worlds(panel)
    inputs = [_candidate_inputs(w, policy, erase) for w in worlds]
    acts = torch.cat([z[0] for z in inputs])
    obs = torch.cat([z[1] for z in inputs])
    model.eval()
    with torch.no_grad():
        out = _predict(arm, model, {"action": acts, "effect": obs})
    if out.shape != (2 * len(worlds), len(r29.EARLY) + 1 +
                     r29.B_STEPS[panel] + 1, 16) or not bool(torch.isfinite(out).all()):
        raise ValueError("F8R29B_INVALID_PREACTION_PREDICTION")
    raw = out[:, -1, 0].detach().cpu().tolist()
    answers = []
    for i, w in enumerate(worlds):
        if arm == "native_direct_value":
            losses = [raw[2*i], raw[2*i+1]]
        else:
            est = [w["return_state"] + 2.4 * math.atanh(
                max(-.98, min(.98, raw[2*i + a]))) for a in (0, 1)]
            losses = [(s - GOAL) ** 2 for s in est]
        answers.append(min((0, 1), key=lambda a: (losses[a], a)))
    return answers


def _group_regrets(worlds: list[dict], actions: list[int]) -> tuple[float, list[float]]:
    if len(worlds) != 4 * r29.GROUPS or len(actions) != len(worlds):
        raise ValueError("F8R29B_UNBALANCED_SCORE")
    regrets = [r29.actual_regret(w, a) for w, a in zip(worlds, actions)]
    grouped = [math.fsum(regrets[g*4:(g+1)*4]) / 4 for g in range(r29.GROUPS)]
    return math.fsum(grouped) / r29.GROUPS, grouped


def score_panel(arm: str, model, panel: str) -> dict:
    worlds = r29.worlds(panel)
    scores = {}
    for policy in POLICIES:
        choices = _choices(arm, model, panel, policy)
        mean, by_group = _group_regrets(worlds, choices)
        scores[policy] = {"mean_regret": mean, "pergroup_regret": by_group,
                          "choices": choices, "selected_fifth_effects": len(worlds)}
    erased = _choices(arm, model, panel, "active_information", erase=True)
    erased_mean, erased_group = _group_regrets(worlds, erased)
    return {"policies": scores, "erased_active_mean_regret": erased_mean,
            "erased_active_pergroup_regret": erased_group,
            "fifth_effect_erasure_penalty": erased_mean -
                scores["active_information"]["mean_regret"]}


def _gain_lower(scores: dict) -> float:
    rng = random.Random(BOOT_SEED)
    active = scores["active_information"]["pergroup_regret"]
    f0 = scores["fixed_probe0"]["pergroup_regret"]
    f1 = scores["fixed_probe1"]["pergroup_regret"]
    gains = []
    for _ in range(DRAWS):
        ix = [rng.randrange(r29.GROUPS) for __ in range(r29.GROUPS)]
        gains.append(min(math.fsum(f0[i] for i in ix),
                         math.fsum(f1[i] for i in ix)) / r29.GROUPS
                     - math.fsum(active[i] for i in ix) / r29.GROUPS)
    return sorted(gains)[int(.025 * DRAWS)]


def _judge(row: dict) -> dict:
    protocol = locked()["score"]
    panels = {}
    for panel in PANELS:
        scored = row["panels"][panel]
        pol = scored["policies"]
        active = pol["active_information"]["mean_regret"]
        best_fixed = min(pol["fixed_probe0"]["mean_regret"],
                         pol["fixed_probe1"]["mean_regret"])
        gain = best_fixed - active
        lower = _gain_lower(pol)
        erasure = scored["fifth_effect_erasure_penalty"]
        passed = bool(
            gain >= protocol["native_" + panel + "_active_gain_vs_best_fixed_min"]
            and erasure >= protocol["native_" + panel +
                                    "_information_erasure_penalty_min"]
            and lower > protocol["native_" + panel +
                                 "_pair_bootstrap_2p5_lower_gt"])
        panels[panel] = {"gain_over_best_fixed": gain,
                         "fifth_effect_erasure_penalty": erasure,
                         "paired_bootstrap_lower": lower,
                         "development_win": passed}
    return {"arm": row["arm"], "seed": row["seed"],
            "panels": panels,
            "both_panel_win": all(z["development_win"] for z in panels.values())}


def train(arm: str, seed: int, fit: list[dict],
          samples: list[list[int]]) -> dict:
    model = model_for(arm, seed)
    digest = r25.r22.r18.r5.f5.hash_weights
    initial = digest(model)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("native") else .001, weight_decay=.01)
    start = time.perf_counter()
    for ix in samples:
        batch = _batch(fit, ix)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # The P69/P85 predictor admits exactly factual action/effect keys.
        # Realized return targets are loss labels, never historical model input.
        pred = _predict(arm, model, {"action": batch["action"],
                                      "effect": batch["effect"]})
        if pred.shape != batch["effect"].shape:
            raise ValueError("F8R29B_TRAIN_PREDICTION_SHAPE")
        target = (batch["direct_target"] if arm == "native_direct_value"
                  else batch["physical_target"])
        loss = (pred[:, -1, 0] - target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R29B_NONFINITE_LOSS")
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if (not any(x is not None and bool(torch.any(x != 0)) for x in grads)
            or any(x is not None and not bool(torch.isfinite(x).all())
                   for x in grads)):
            raise ValueError("F8R29B_NONFINITE_OR_ZERO_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    trained = digest(model)
    if initial == trained:
        raise ValueError("F8R29B_MODEL_NOT_ACTUALLY_UPDATED")
    seconds = time.perf_counter() - start
    outcomes = {panel: score_panel(arm, model, panel) for panel in PANELS}
    cold = model_for(arm, seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    if trained != digest(cold) or any(
        _choices(arm, cold, panel, "active_information")
        != outcomes[panel]["policies"]["active_information"]["choices"]
        for panel in PANELS
    ):
        raise ValueError("F8R29B_COLD_RESTORATION_DRIFT")
    return {"arm": arm, "seed": seed, "optimizer_updates": UPDATES,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "initial_weight_hash": initial, "trained_weight_hash": trained,
            "train_wall_seconds": seconds, "cold_actions_equal": True,
            "panels": outcomes}


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit = fit_cohort()
    if len(fit) != FIT_GROUPS * 2 * 2 * len(FIT_Q) * 2 * 2:
        raise ValueError("F8R29B_FACTUAL_FIT_FACTORIAL_GAP")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20469999)
    samples = torch.randint(0, len(fit), (UPDATES, BATCH),
                            generator=generator).tolist()
    start = time.perf_counter()
    arm_seeds = [(arm, seed) for arm in ARMS for seed in (
        SEEDS if arm == "native_dynamics" else SEEDS[:1])]
    outcomes = [train(arm, seed, fit, samples) for arm, seed in arm_seeds]
    judgments = [_judge(x) for x in outcomes]
    n = sum(x["both_panel_win"] for x in judgments
            if x["arm"] == "native_dynamics")
    return {
        "schema": SCHEMA,
        "classification": "SOURCE_NEUTRAL_NEURAL_INFORMATION_DEV_ONLY",
        "freeze_blob": FREEZE_BLOB,
        "frozen_parent_head": "bc220a22495c2b4c718e3737c616f783e4f1f28a",
        "factual_fit_episodes": len(fit),
        "factual_fit_effects": len(fit) * 14,
        "neural_model_instances": len(outcomes),
        "real_optimizer_updates": len(outcomes) * UPDATES,
        "trained_models": outcomes,
        "judgments": judgments,
        "native_development_pass_2_of_3": n >= 2,
        "native_development_winning_seeds": n,
        "analytic_scripted_ceiling": {
            panel: r29.panel_score(panel)["policy_mean_regret"]
            for panel in PANELS},
        "conventional_independent_convergence_qualified": False,
        "matched_capacity_and_compute": False,
        "independent_scientific_pass": False,
        "original_external_effects": 0,
        "spent_F7_replay": False,
        "deployment": "NONE",
        "seconds": time.perf_counter() - start,
    }


def retain_ci_receipt(value: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r29b_neural_information_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R29B_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def refuse_original_external_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R29B_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
