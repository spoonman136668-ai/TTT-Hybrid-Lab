"""F8-R43: pre-frozen eight-cell length × salient-position × physics transfer DEV.

The R42 source and scientific NO-GO remain intact. This new experiment is
source-neutral synthetic development only, not independent scientific first-use.
No hidden evaluator variable or unselected consequence enters learner inputs.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r42_history_binding_interaction as r42
from experiments import p85mr_f8r41_history_conditional as r41
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r43_factorial_transfer_freeze.json"
FREEZE_BLOB = "65ea25ecf1112b44f897f47641f0b9671043d2c2"
SCHEMA = "ttt-hybrid.f8r43-factorial-transfer-prereg.v1"
PARENT = "d33fa9aa274179d13a55de83597a1a6cbc3dddc8"
SEEDS = (985201, 985207)
ARMS = tuple([(arm, seed) for arm in
    ("native_recurrent", "history_only_recurrent",
     "native_salience", "history_only_salience")
    for seed in SEEDS] + [("gru_recurrent", SEEDS[0]),
                         ("ssm_recurrent", SEEDS[0])])
PANELS = tuple(f"e{n}_{pos}_{law}"
    for n in (8, 16) for pos in ("seen", "novel")
    for law in ("quartic", "cubic"))
HARD = "e16_novel_cubic"
FAMILIAR = "e8_seen_quartic"
STEPS = 256
BATCH = 16
TRAIN_PAIRS, HELDOUT_PAIRS = 256, 128
GROUPS = 16
SALT = {"fit": 21710000, "eval": 21720000}
STATES = (.61, .73)
SAMPLE_SEED = 21749999
BOOT_SEED = 21769999
BOOT_DRAWS = 10000


def _git_blob(raw):
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if _git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R43_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    data = json.loads(raw)
    if (data["schema"] != SCHEMA or
        data["stage"] != "FROZEN_BEFORE_R43_IMPLEMENTATION_OR_NEW_SOURCE_GENERATION" or
        data["lineage"]["parent_sha"] != PARENT or
        data["source"]["new_salts"] != SALT or
        data["source"]["factorial_eval"]["panels"] != len(PANELS) or
        data["source"]["fit"]["selected_factual_return_episodes"] != 768 or
        data["methods"]["budget"]["new_updates_per_arm"] != STEPS or
        data["methods"]["budget"]["paired_factual_worlds_per_update"] != BATCH or
        data["methods"]["budget"]["sample_seed"] != SAMPLE_SEED or
        data["prospective_decision"]["scientific_classification"].find(
            "development only") == -1 or
        data["prospective_decision"]["independent_scientific_pass"] is not False or
        any(data["authority"][k] for k in (
            "external_original_simulator_first_use", "spent_F7_replay",
            "R42_scored_eval_replay", "new_runner", "new_workflow",
            "new_controller", "paid_api", "GPU_first_use",
            "merge", "promotion", "deployment"))):
        raise ValueError("F8R43_SCOPE_OR_METHOD_DRIFT")
    if [(a["name"], tuple(a["seeds"])) for a in data["methods"]["arms"]] != [
        (name, tuple(seed for n, seed in ARMS if n == name))
        for name in dict.fromkeys(n for n, _ in ARMS)
    ]:
        raise ValueError("F8R43_ARM_DRIFT")
    r42.locked()
    return data


def _panel(panel):
    if panel == "fit":
        return 8, "seen", "quartic"
    if panel not in PANELS:
        raise ValueError("F8R43_UNREGISTERED_PANEL")
    a, pos, law = panel.split("_")
    return int(a[1:]), pos, law


def marker(group, family):
    if family == "seen":
        return 1 + (group % 4)
    if family == "novel":
        return 5 + (group % 3)
    raise ValueError("F8R43_INVALID_MARKER_FAMILY")


def b_history(panel, group, gate):
    length, family, _ = _panel(panel)
    max_group = 48 if panel == "fit" else GROUPS
    if (type(group) is not int or not 0 <= group < max_group
        or gate not in (-1, 1)):
        raise ValueError("F8R43_INVALID_B_HISTORY")
    spot = marker(group, family)
    salt = SALT["fit" if panel == "fit" else "eval"]
    rows = []
    for t in range(length):
        nuisance = .006 * math.sin((salt + group) * .137 + t * .271)
        nuisance += .004 * (((t + group) % 4) - 1.5)
        rows.append((t % 2, round(nuisance + (.24 * gate if t == spot else 0.), 8)))
    return tuple(rows)


def world(panel, group, context, sign, gate):
    steps, family, law = _panel(panel)
    limit = 48 if panel == "fit" else GROUPS
    if (type(group) is not int or not 0 <= group < limit
        or context not in (0, 1) or sign not in (-1, 1)
        or gate not in (-1, 1)):
        raise ValueError("F8R43_UNREGISTERED_WORLD")
    return {"panel": panel, "group": group, "public_context": context,
            "q_evaluator_only": sign,
            "gate_evaluator_only": gate,
            "return_state": STATES[group % 2], "goal": 1.,
            "b_history": b_history(panel, group, gate)}


def worlds(panel):
    _panel(panel)
    count = 48 if panel == "fit" else GROUPS
    return [world(panel, group, c, q, h)
            for group in range(count)
            for c in (0, 1) for q in (-1, 1) for h in (-1, 1)]


def _return_law(w, action):
    # Used only by factual selected-return source birth and evaluator.
    # This function must not be imported into a model or training loss.
    q = w["q_evaluator_only"] * w["gate_evaluator_only"]
    _, _, law = _panel(w["panel"])
    s, c = w["return_state"], w["public_context"]
    return (r41._law(s, action, c, q, "ood") if law == "cubic"
            else r33.r32.after(s, action, c, q))


def _selected_history(w, *, erase_b=False, swap_b=False, erase_fifth=False):
    c, q = w["public_context"], w["q_evaluator_only"]
    # Early and selected fifth are identical across all factorial cells.
    rows = [r33.b._record(s, action, r33.r32.after(s, action, c, q), c)
            for s, action in r33.r32.EARLY]
    rows.append(r33.b._record(
        r33.PROBE, c, r33.r32.after(r33.PROBE, c, c, q), c))
    if swap_b:
        b = b_history(w["panel"], w["group"], -w["gate_evaluator_only"])
    else:
        b = w["b_history"]
    rows += [r33.b._record(0., action, effect, c) for action, effect in b]
    actions = torch.stack([row[0] for row in rows])
    effects = torch.stack([row[1] for row in rows])
    if erase_b:
        effects[5 + marker(w["group"], _panel(w["panel"])[1])].zero_()
    if erase_fifth:
        effects[4].zero_()
    return actions, effects


def candidates(w, **controls):
    a, e = _selected_history(w, **controls)
    result = []
    for chosen in (0, 1):
        candidate = r33.b._record(
            w["return_state"], chosen, w["return_state"],
            w["public_context"])[0]
        result.append((torch.cat((a, candidate[None])),
                       torch.cat((e, torch.zeros((1, 16), dtype=e.dtype)))))
    return (torch.stack([x[0] for x in result]),
            torch.stack([x[1] for x in result]))


def factual_fit():
    locked()
    facts, labels = [], []
    for w in worlds("fit"):
        factual_costs = []
        for action in (0, 1):
            a, e = _selected_history(w)
            following = _return_law(w, action)
            actual = r33.b._record(
                w["return_state"], action, following, w["public_context"])
            aa = torch.cat((a, actual[0][None]))
            ee = torch.cat((e, actual[1][None]))
            final_state = w["return_state"] + r31._physical_delta(ee[-1, 0])
            factual_costs.append((final_state - 1.) ** 2)
            facts.append({"action": aa, "effect": ee})
        labels.append(min((0, 1), key=lambda x: (factual_costs[x], x)))
    if len(facts) != 768 or len(labels) != 384:
        raise ValueError("F8R43_WRONG_FIT_FACTUAL_EPISODES")
    return facts, labels


def _regret(w, action):
    if action not in (0, 1):
        raise ValueError("F8R43_INVALID_ACTION")
    costs = [(_return_law(w, a) - w["goal"])**2 for a in (0, 1)]
    return max(0., costs[action] - min(costs))


class SalienceGate(nn.Module):
    """Trainable content reader over observed B rows; no marker/index oracle."""

    def __init__(self, arm, seed, binder):
        super().__init__()
        if (arm, seed) not in ARMS or "salience" not in arm:
            raise ValueError("F8R43_UNREGISTERED_SALIENCE_MODEL")
        self.arm, self.seed = arm, seed
        self.binder = copy.deepcopy(binder)
        for param in self.binder.parameters():
            param.requires_grad_(False)
        self.binder.eval()
        self.native = (r34.FactualMemory("native_factual_memory", seed)
                       if arm == "native_salience" else None)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 21780000 + ARMS.index((arm, seed)))
            self.attention = nn.Sequential(
                nn.Linear(32, 24), nn.GELU(), nn.Linear(24, 1))
            self.reader = nn.Sequential(
                nn.Linear(32, 24), nn.GELU(), nn.Linear(24, 1))
            self.fusion = nn.Sequential(
                nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))

    def train(self, mode=True):
        super().train(mode)
        self.binder.eval()
        return self

    def forward(self, a, e, *, zero_backbone=False, zero_gate=False,
                zero_binder=False, zero_salience=False):
        if (a.ndim != 3 or e.shape != a.shape or a.shape[-1] != 16
            or a.shape[1] not in (14, 22) or a.shape[0] % 2):
            raise ValueError("F8R43_INVALID_PAIRED_PREACTION_FACTS")
        obs = e.clone()
        obs[:, -1, :].zero_()
        baseline = r38.predict(self.binder, "binder", a, obs).reshape(-1, 2)
        selected_rows = torch.cat((a[:, 5:-1, :], obs[:, 5:-1, :]), dim=-1)
        logits = self.attention(selected_rows).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        summary = (weights.unsqueeze(-1) * selected_rows).sum(dim=1)
        signal = self.reader(summary).reshape(-1, 2).mean(dim=-1)
        if zero_salience:
            signal = torch.zeros_like(signal)
        if self.native is not None:
            native_prediction = self.native(
                a, obs, ablate_backbone=zero_backbone).reshape(-1, 2)
            native_signal = native_prediction[:, 1] - native_prediction[:, 0]
        else:
            native_signal = torch.zeros_like(signal)
        gate = torch.tanh(self.fusion(
            torch.stack((signal, native_signal), dim=-1)).squeeze(-1))
        contrast = (baseline[:, 1] - baseline[:, 0]) * gate
        if zero_gate or zero_binder:
            contrast = torch.zeros_like(contrast)
        return torch.stack((-.5 * contrast, .5 * contrast), dim=-1)


class RecurrentGate(nn.Module):
    """Use the exact R42 trained recurrent mechanism as an independently fitted control."""

    def __init__(self, arm, seed, binder):
        super().__init__()
        mapping = {
            "native_recurrent": "native_pair_gate",
            "history_only_recurrent": "history_only_pair_gate",
            "gru_recurrent": "gru_pair_gate",
            "ssm_recurrent": "ssm_pair_gate",
        }
        if (arm, seed) not in ARMS or arm not in mapping:
            raise ValueError("F8R43_UNREGISTERED_RECURRENT_MODEL")
        self.inner = r42.PairGate(mapping[arm], seed, binder)

    def train(self, mode=True):
        super().train(mode)
        self.inner.binder.eval()
        return self

    def forward(self, a, e, *, zero_backbone=False, zero_gate=False,
                zero_binder=False, zero_salience=False):
        return self.inner(a, e, zero_backbone=zero_backbone,
                          zero_gate=zero_gate, zero_binder=zero_binder)


def model_for(arm, seed, binder):
    return (SalienceGate(arm, seed, binder) if "salience" in arm
            else RecurrentGate(arm, seed, binder))


def _batch(facts, labels, ids):
    if (len(ids) != BATCH or any(type(i) is not int or
         not 0 <= i < TRAIN_PAIRS for i in ids)):
        raise ValueError("F8R43_INVALID_TRAIN_IDS")
    sample = [facts[2 * i + a] for i in ids for a in (0, 1)]
    return (torch.stack([w["action"] for w in sample]),
            torch.stack([w["effect"] for w in sample]),
            torch.tensor([labels[i] for i in ids], dtype=torch.long))


@lru_cache(maxsize=32)
def _scoring_inputs(panel, erase_b=False, erase_fifth=False, swap_b=False):
    """Only immutable source-neutral selected factual observations; no labels."""
    if panel not in PANELS:
        raise ValueError("F8R43_UNREGISTERED_SCORE_PANEL")
    ww = worlds(panel)
    paired = [candidates(w, erase_b=erase_b,
                         erase_fifth=erase_fifth, swap_b=swap_b) for w in ww]
    return (ww, torch.cat([p[0] for p in paired]),
            torch.cat([p[1] for p in paired]))


def score(model, panel, *, erase_b=False, erase_fifth=False,
          swap_b=False, zero_gate=False, zero_backbone=False,
          zero_binder=False, zero_salience=False):
    ww, a, e = _scoring_inputs(panel, erase_b, erase_fifth, swap_b)
    model.eval()
    with torch.no_grad():
        logits = model(a, e, zero_gate=zero_gate,
                       zero_backbone=zero_backbone,
                       zero_binder=zero_binder,
                       zero_salience=zero_salience)
    if logits.shape != (len(ww), 2) or not bool(torch.isfinite(logits).all()):
        raise ValueError("F8R43_INVALID_CAUSAL_ACTION_LOGITS")
    choices = logits.argmin(dim=-1).tolist()
    regret = [_regret(w, action) for w, action in zip(ww, choices)]
    return {"choices": choices, "mean_regret": statistics.fmean(regret),
            "pergroup_regret": [statistics.fmean(regret[8*g:8*g+8])
                for g in range(GROUPS)],
            "gate_pair_flips": sum(
                choices[i] != choices[i+1] for i in range(0, len(choices), 2))}


def _heldout(model, facts, labels):
    model.eval()
    count = 0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS, TRAIN_PAIRS + HELDOUT_PAIRS, BATCH):
            rows = [facts[2*i+a] for i in range(start, start+BATCH)
                    for a in (0, 1)]
            a = torch.stack([w["action"] for w in rows])
            e = torch.stack([w["effect"] for w in rows])
            y = torch.tensor(labels[start:start+BATCH], dtype=torch.long)
            count += int((model(a, e).argmin(dim=-1) == y).sum())
    return count / HELDOUT_PAIRS


def train_arm(arm, seed, binder, facts, labels, draws):
    model = model_for(arm, seed, binder)
    binder_hash = r40._hash(model.inner.binder if isinstance(
        model, RecurrentGate) else model.binder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=.001, weight_decay=.01)
    t0 = time.perf_counter()
    checkpoints = []
    for step, ids in enumerate(draws, 1):
        a, e, labels_batch = _batch(facts, labels, ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(-model(a, e), labels_batch)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R43_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g != 0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R43_MISSING_OR_INVALID_GRADIENTS")
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step in (128, 256):
            checkpoints.append({"step": step, "train_ce": float(loss),
                                "heldout_choice_accuracy": _heldout(
                                    model, facts, labels)})
    after = r40._hash(model.inner.binder if isinstance(
        model, RecurrentGate) else model.binder)
    if after != binder_hash:
        raise ValueError("F8R43_FROZEN_BINDER_CHANGED")
    panel_results = {panel: {"active": score(model, panel)}
                     for panel in PANELS}
    # All eight cells receive active causal scores; mechanistic ablations
    # are evaluated on the preregistered hardest cell to avoid repeated
    # expensive scoring unrelated to the frozen transfer falsifier.
    for panel in (HARD,):
        for name, kwargs in (
            ("erase_B", {"erase_b": True}),
            ("swap_B", {"swap_b": True}),
            ("erase_fifth", {"erase_fifth": True}),
            ("zero_gate", {"zero_gate": True}),
            ("zero_binder", {"zero_binder": True}),
            ("zero_backbone", {"zero_backbone": True}),
            ("zero_salience", {"zero_salience": True})):
            panel_results[panel][name] = score(model, panel, **kwargs)
    cold = model_for(arm, seed, binder)
    cold.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    if (r40._hash(cold) != r40._hash(model) or
        any(score(cold, panel)["choices"] != panel_results[panel]["active"]["choices"]
            for panel in PANELS)):
        raise ValueError("F8R43_COLD_RESTORE_DECISION_DRIFT")
    return {"arm": arm, "seed": seed,
            "optimizer_updates": STEPS,
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
            "frozen_binder_hash": binder_hash,
            "wall_seconds": time.perf_counter() - t0,
            "checkpoints": checkpoints,
            "panels": panel_results, "cold_restored": True}


def _control_feature_from_observations(a, e, state, context):
    """Operate on public state/context and selected factual measurements only."""
    if (a.shape != e.shape or a.ndim != 3 or a.shape[0] != 2 or
        a.shape[-1] != 16 or a.shape[1] not in (14, 22)):
        raise ValueError("F8R43_BAD_CONTROL_OBSERVATIONS")
    fifth = r31._physical_delta(e[0, 4, 0])
    b_peak = r42._observed_b_peak(e[0])
    interaction = fifth * b_peak
    return [1., float(state), float(context), fifth, b_peak,
            interaction, state * interaction, state*state * interaction]


def _control_feature(w):
    a, e = candidates(w)
    return _control_feature_from_observations(
        a, e, w["return_state"], w["public_context"])


def observational_control(facts, labels):
    """FIT-only ridge-calibrated conventional diagnostic, no generator marker."""
    fit_worlds = worlds("fit")
    if len(fit_worlds) != len(labels) or len(labels) != 384:
        raise ValueError("F8R43_BAD_OBSERVATIONAL_CALIBRATION")
    features = torch.tensor([_control_feature(w)
        for w in fit_worlds[:TRAIN_PAIRS]], dtype=torch.float64)
    targets = torch.tensor([1. if y else -1.
        for y in labels[:TRAIN_PAIRS]], dtype=torch.float64)
    eye = torch.eye(features.shape[1], dtype=torch.float64)
    params = torch.linalg.solve(
        features.T @ features + .001 * eye, features.T @ targets)
    result = {}
    for panel in PANELS:
        ww = worlds(panel)
        x = torch.tensor([_control_feature(w) for w in ww],
                         dtype=torch.float64)
        choices = (x @ params > 0).long().tolist()
        regret = [_regret(w, choice) for w, choice in zip(ww, choices)]
        result[panel] = {
            "choices": choices, "mean_regret": statistics.fmean(regret),
            "gate_pair_flips": sum(choices[i] != choices[i+1]
                                   for i in range(0, len(choices), 2)),
            "pergroup_regret": [statistics.fmean(regret[8*g:8*g+8])
                                for g in range(GROUPS)]}
    return {"fit_only_ridge_coefficients": params.tolist(),
            "panels": result,
            "oracle_metadata_used": False,
            "calibration_fit_worlds": TRAIN_PAIRS}


def _lower_paired_gain(salience, recurrent):
    rng = random.Random(BOOT_SEED)
    n = len(salience)
    diffs = []
    for _ in range(BOOT_DRAWS):
        sample = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.fmean(
            recurrent[i] - salience[i] for i in sample))
    return sorted(diffs)[int(.025*BOOT_DRAWS)]


def _axes_effects(row):
    p = row["panels"]
    contrasts = {}
    for changed, choices in (
        ("length", (("e16", "e8"),)),
        ("marker", (("novel", "seen"),)),
        ("physics", (("cubic", "quartic"),))):
        vals = []
        for panel in PANELS:
            chunks = panel.split("_")
            if changed == "length":
                if chunks[0] != "e8": continue
                other = "e16_" + "_".join(chunks[1:])
            elif changed == "marker":
                if chunks[1] != "seen": continue
                other = "_".join([chunks[0], "novel", chunks[2]])
            else:
                if chunks[2] != "quartic": continue
                other = "_".join([chunks[0], chunks[1], "cubic"])
            vals.append(p[other]["active"]["mean_regret"] -
                        p[panel]["active"]["mean_regret"])
        contrasts[changed] = {
            "paired_cell_differences": vals,
            "mean_regret_change": statistics.fmean(vals)}
    return contrasts


def judge(rows, freeze):
    by_key = {(x["arm"], x["seed"]): x for x in rows}
    result = []
    threshold = freeze["prospective_decision"]["salience_transfer"]
    for seed in SEEDS:
        sal = by_key["native_salience", seed]
        recur = by_key["native_recurrent", seed]
        hard_s = sal["panels"][HARD]["active"]
        hard_r = recur["panels"][HARD]["active"]
        familiar_s = sal["panels"][FAMILIAR]["active"]
        familiar_r = recur["panels"][FAMILIAR]["active"]
        penalty = (sal["panels"][HARD]["erase_B"]["mean_regret"] -
                   hard_s["mean_regret"])
        zero_salience = (sal["panels"][HARD]["zero_salience"]["mean_regret"] -
                         hard_s["mean_regret"])
        gain = hard_r["mean_regret"] - hard_s["mean_regret"]
        lower = _lower_paired_gain(hard_s["pergroup_regret"],
                                   hard_r["pergroup_regret"])
        passed = bool(
            hard_s["mean_regret"] <= threshold["max_salience_regret_each_seed"]
            and hard_s["gate_pair_flips"] >= threshold["min_salience_gate_flips"]
            and gain >= threshold["min_regret_gain_over_same_seed_native_recurrent"]
            and familiar_s["mean_regret"] - familiar_r["mean_regret"]
                <= threshold["max_loss_vs_same_seed_native_recurrent_fit_like"]
            and penalty >= threshold["min_regret_penalty_erase_B"]
            and zero_salience >= threshold["min_regret_penalty_zero_salience"]
            and lower > threshold["paired_bootstrap_lower_gain_gt"])
        conventional_rows = [
            row for row in rows
            if row["arm"] in ("history_only_recurrent",
                              "history_only_salience",
                              "gru_recurrent", "ssm_recurrent")]
        strongest_conv = min(
            conventional_rows,
            key=lambda row: row["panels"][HARD]["active"]["mean_regret"])
        conventional_regret = strongest_conv["panels"][HARD]["active"]["mean_regret"]
        native_backbone_penalty = (
            sal["panels"][HARD]["zero_backbone"]["mean_regret"]
            - hard_s["mean_regret"])
        specific = freeze["prospective_decision"]["native_incremental"]
        native_specific_development_conditions = bool(
            conventional_regret - hard_s["mean_regret"] >=
                specific["gain_vs_strongest_fully_trained_history_only_gru_ssm"]
            and native_backbone_penalty >=
                specific["backbone_ablation_regret_penalty"])
        result.append({"seed": seed, "frozen_salience_transfer_pass": passed,
                       "native_specific_development_conditions":
                           native_specific_development_conditions,
                       "strict_native_specific_science": "UNVERIFIED_COMPUTE_ORIGINAL_SOURCE",
                       "strongest_trained_conventional_arm": strongest_conv["arm"],
                       "strongest_trained_conventional_regret": conventional_regret,
                       "native_backbone_regret_penalty": native_backbone_penalty,
                       "hard_native_salience_regret": hard_s["mean_regret"],
                       "hard_native_recurrent_regret": hard_r["mean_regret"],
                       "hard_gain": gain,
                       "B_erase_penalty": penalty,
                       "zero_salience_penalty": zero_salience,
                       "paired_bootstrap_lower_gain": lower,
                       "factorial_native_salience_effects": _axes_effects(sal),
                       "factorial_native_recurrent_effects": _axes_effects(recur)})
    return {"salience_both_seeds_pass": all(
                x["frozen_salience_transfer_pass"] for x in result),
            "seeds": result,
            "native_specific_science": "UNVERIFIED_WITHOUT_MATCHED_WHOLE_PROCESS_COMPUTE_AND_INDEPENDENT_SOURCE",
            "independent_scientific_pass": False}


def run():
    freeze = locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    facts, labels = factual_fit()
    original = r33.fit_cohort()
    winners = r36.fit_pairs(original)
    pools = r37.selection_pools(original, winners)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(21249999)
    original_draws = torch.randint(
        0, len(pools["informative"]), (512, 16), generator=rng).tolist()
    binders = {}
    for seed in SEEDS:
        binders[seed], receipt = r39.replay_original_seed(
            seed, original, winners, pools, original_draws)
        if not receipt["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R43_R38_REFERENCE_REPLAY_DRIFT")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SAMPLE_SEED)
    draws = torch.randint(
        0, TRAIN_PAIRS, (STEPS, BATCH), generator=generator).tolist()
    models = [train_arm(arm, seed, binders[seed], facts, labels, draws)
              for arm, seed in ARMS]
    observed_control = observational_control(facts, labels)
    judgment = judge(models, freeze)
    return {"schema": SCHEMA, "freeze_blob": FREEZE_BLOB,
            "parent": PARENT, "factual_fit_episodes": len(facts),
            "train_pairs": TRAIN_PAIRS, "heldout_pairs": HELDOUT_PAIRS,
            "factored_panels": list(PANELS),
            "original_binder_replay_updates": len(SEEDS) * 512,
            "new_neural_updates": len(ARMS) * STEPS,
            "models": models, "observational_control": observed_control,
            "judgment": judgment,
            "synthetic_development_only": True,
            "independent_scientific_pass": False,
            "external_scientific_first_use": False,
            "GPU": False, "deployment": "NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r43_factorial_transfer_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R43_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(result, indent=2, sort_keys=True,
                               allow_nan=False) + "\n")


def refuse_external_scientific_first_use(*args, **kwargs):
    raise PermissionError("F8R43_EXTERNAL_FIRST_USE_NOT_AUTHORIZED")
