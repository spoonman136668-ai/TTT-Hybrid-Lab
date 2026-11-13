"""F8-R42: prospective learned history × factual binder interaction (synthetic DEV).

The causal fifth-action binder is replayed/frozen. The history branch must
learn a gate from actually observed B effects under selected factual returns.
No hidden generator label or unselected alternative action effect is available
to any optimizer; positive diagnostic is separately hand-engineered.
"""
from __future__ import annotations

import copy
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

from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r41_history_conditional as r41

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r42_history_binding_interaction_freeze.json"
FREEZE_BLOB = "6bc8b93765acfeba7d4ef6c27f9ab1540010cf52"
PARENT = "88cb3c5296223d5e212b66d5126f368646e1e47c"
SCHEMA = "ttt-hybrid.f8r42-learned-history-binder-interaction-prereg.v1"
SEEDS = (985201, 985207)
ARMS = tuple([(a, s) for a in ("native_pair_gate", "history_only_pair_gate")
              for s in SEEDS] + [("gru_pair_gate", SEEDS[0]),
                                  ("ssm_pair_gate", SEEDS[0])])
PANELS = ("dev", "ood")
STEPS, BATCH, TRAIN_PAIRS, VAL_PAIRS = 384, 16, 256, 128
SALT = {"fit": 21610000, "dev": 21620000, "ood": 21630000}
STATES = {"fit": (.61, .73), "dev": (.59, .74), "ood": (.57, .76)}
MAG = {"fit": 1., "dev": .94, "ood": 1.07}
B_STEPS = {"fit": 8, "dev": 8, "ood": 16}
SAMPLE_SEED, BOOT_SEED, BOOT_DRAWS = 21649999, 21669999, 10000
GROUPS = 16


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R42_FROZEN_PROTOCOL_BLOB_DRIFT")
    p = json.loads(raw)
    if (p["schema"] != SCHEMA or
        p["stage"] != "FROZEN_BEFORE_R42_IMPLEMENTATION_OR_ANY_R42_FIT_RESULTS" or
        p["parent"]["sha"] != PARENT or
        p["parent"]["r41_freeze_blob"] != r41.FREEZE_BLOB or
        p["classification"] != "SOURCE_NEUTRAL_SYNTHETIC_NEURAL_HISTORY_INTERACTION_DEVELOPMENT_ONLY" or
        p["source"]["new_cohort_salt"] != SALT or
        p["source"]["counts"]["fit_separately_selected_factual_episodes"] != 768 or
        p["learner"]["budget"]["new_updates_each"] != STEPS or
        p["learner"]["budget"]["fit_pairs_per_update"] != BATCH or
        p["learner"]["budget"]["sample_seed"] != SAMPLE_SEED or
        p["prospective_score"]["fixed_bootstrap_seed"] != BOOT_SEED or
        p["prospective_score"]["fixed_bootstrap_resamples"] != BOOT_DRAWS or
        p["prospective_score"]["independent_scientific_pass"] is not False or
        any(p["scope"][k] for k in ("spent_F7_source_replay",
            "external_simulator_first_use", "GPU_first_use", "paid_API",
            "new_runner", "new_workflow", "new_controller", "merge",
            "promotion", "deployment"))):
        raise ValueError("F8R42_FROZEN_SCOPE_OR_THRESHOLD_DRIFT")
    r41.locked()
    return p


def marker(panel, group):
    if panel == "fit":
        return 1 + (group % 4)
    if panel == "dev":
        return 1 + ((3 * group + 1) % 5)
    if panel == "ood":
        return 2 + ((5 * group + 1) % 11)
    raise ValueError("F8R42_UNKNOWN_PANEL")


def history(panel, group, gate):
    if panel not in SALT or gate not in (-1, 1):
        raise ValueError("F8R42_INVALID_HISTORY")
    m = marker(panel, group)
    result = []
    for t in range(B_STEPS[panel]):
        noise = .006 * math.sin((SALT[panel] + group) * .137 + t * .271)
        noise += .004 * (((t + group) % 4) - 1.5)
        result.append((t % 2, round(noise + (.24 * gate if t == m else 0.), 8)))
    return tuple(result)


def world(panel, group, context, sign, gate):
    lim = 48 if panel == "fit" else GROUPS
    if (panel not in SALT or type(group) is not int or not 0 <= group < lim
        or context not in (0, 1) or sign not in (-1, 1)
        or gate not in (-1, 1)):
        raise ValueError("F8R42_UNREGISTERED_WORLD")
    return {"panel": panel, "group": group, "public_context": context,
            "q_evaluator_only": sign * MAG[panel],
            "gate_evaluator_only": gate,
            "return_state": STATES[panel][group % 2], "goal": 1.,
            "b_history": history(panel, group, gate)}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R42_INVALID_PANEL")
    return [world(panel, g, c, q, h)
            for g in range(48 if panel == "fit" else GROUPS)
            for c in (0, 1) for q in (-1, 1) for h in (-1, 1)]


def factual_history(w, *, erase_b=False, erase_fifth=False, swap_b=False):
    panel, c, q = w["panel"], w["public_context"], w["q_evaluator_only"]
    rows = [r33.b._record(s, a, r41._law(s, a, c, q, panel), c)
            for s, a in r33.r32.EARLY]
    rows.append(r33.b._record(r33.PROBE, c,
                              r41._law(r33.PROBE, c, c, q, panel), c))
    source = (history(panel, w["group"], -w["gate_evaluator_only"])
              if swap_b else w["b_history"])
    for act, observed in source:
        rows.append(r33.b._record(0., act, observed, c))
    a, e = (torch.stack([x[j] for x in rows]) for j in (0, 1))
    if erase_b:
        e[5 + marker(panel, w["group"])].zero_()
    if erase_fifth:
        e[4].zero_()
    return a, e


def candidates(w, *, erase_b=False, erase_fifth=False, swap_b=False):
    a, e = factual_history(w, erase_b=erase_b, erase_fifth=erase_fifth,
                           swap_b=swap_b)
    rows = []
    for choice in (0, 1):
        key = r33.b._record(w["return_state"], choice, w["return_state"],
                             w["public_context"])[0]
        rows.append((torch.cat((a, key[None])), torch.cat(
            (e, torch.zeros((1, 16), dtype=e.dtype)))))
    return (torch.stack([x[0] for x in rows]),
            torch.stack([x[1] for x in rows]))


def factual_fit():
    locked()
    facts, labels = [], []
    for w in worlds("fit"):
        pairs = []
        for action in (0, 1):
            a, e = factual_history(w)
            physical = r41._law(w["return_state"], action,
                    w["public_context"],
                    w["q_evaluator_only"] * w["gate_evaluator_only"], "fit")
            chosen = r33.b._record(w["return_state"], action, physical,
                                   w["public_context"])
            aa = torch.cat((a, chosen[0][None]))
            ee = torch.cat((e, chosen[1][None]))
            measured = w["return_state"] + r31._physical_delta(ee[-1, 0])
            pairs.append((aa, ee, (measured - 1.) ** 2))
        facts.extend({"action": a, "effect": e} for a, e, _ in pairs)
        labels.append(min((0, 1), key=lambda i: (pairs[i][2], i)))
    if len(facts) != 768 or len(labels) != 384:
        raise ValueError("F8R42_FIT_COHORT_SIZE")
    return facts, labels


class PairGate(nn.Module):
    """Same trainable gate interface for Native and conventional recurrent arms."""

    def __init__(self, arm, seed, binder):
        super().__init__()
        if (arm, seed) not in ARMS:
            raise ValueError("F8R42_INVALID_MODEL")
        self.arm, self.seed = arm, seed
        self.binder = copy.deepcopy(binder)
        for v in self.binder.parameters():
            v.requires_grad_(False)
        self.binder.eval()
        name = {"native_pair_gate": "native_factual_memory",
                "history_only_pair_gate": "native_factual_memory",
                "gru_pair_gate": "gru_factual_memory",
                "ssm_pair_gate": "ssm_factual_memory"}[arm]
        self.history = r34.FactualMemory(name, seed)
        if arm == "history_only_pair_gate":
            self.history.backbone = None
            self.history.base_arm = None
            self.history.arm = "history_only"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 21680000 + ARMS.index((arm, seed)))
            self.gate = nn.Sequential(nn.Linear(4, 24), nn.GELU(),
                                      nn.Linear(24, 1))

    def train(self, mode=True):
        super().train(mode)
        self.binder.eval()
        return self

    def forward(self, a, e, *, zero_backbone=False,
                zero_gate=False, zero_binder=False):
        if (a.ndim != 3 or e.shape != a.shape or a.shape[-1] != 16 or
            a.shape[1] not in (14, 22) or a.shape[0] % 2):
            raise ValueError("F8R42_EXPECT_PAIRED_PREACTION_FACTS")
        observed = e.clone()
        observed[:, -1, :].zero_()
        base = r38.predict(self.binder, "binder", a, observed).reshape(-1, 2)
        memory = self.history(a, observed,
                              ablate_backbone=zero_backbone).reshape(-1, 2)
        features = torch.stack((memory[:, 0], memory[:, 1],
                   memory[:, 1]-memory[:, 0],
                   .5*(memory[:, 0]+memory[:, 1])), dim=-1)
        gate = torch.tanh(self.gate(features).squeeze(-1))
        contrast = (base[:, 1]-base[:, 0]) * gate
        if zero_gate or zero_binder:
            contrast = torch.zeros_like(contrast)
        return torch.stack((-.5*contrast, .5*contrast), dim=-1)


def factual_batch(fit, labels, ids):
    if len(ids) != BATCH or any(type(i) is not int or
         not 0 <= i < TRAIN_PAIRS for i in ids):
        raise ValueError("F8R42_INVALID_FACTUAL_PAIR_IDS")
    chosen = [fit[2*i+j] for i in ids for j in (0, 1)]
    return (torch.stack([x["action"] for x in chosen]),
            torch.stack([x["effect"] for x in chosen]),
            torch.tensor([labels[i] for i in ids], dtype=torch.long))


def _regret(w, action):
    """Evaluate the same final-state goal cost used by factual FIT labels.

    Earlier R42 receipts scored only the physical *change* against the goal;
    they remain immutable historical evidence, not valid factual-goal regret.
    """
    if action not in (0, 1):
        raise ValueError("F8R42_INVALID_RETURN_ACTION")
    state = w["return_state"]
    costs = [(state + r41._law(state, c, w["public_context"],
              w["q_evaluator_only"]*w["gate_evaluator_only"],
              w["panel"]) - w["goal"])**2 for c in (0, 1)]
    return max(0., costs[action] - min(costs))


def score(model, panel, *, erase_b=False, erase_fifth=False, swap_b=False,
          zero_backbone=False, zero_gate=False, zero_binder=False):
    ww = worlds(panel)
    batch = [candidates(w, erase_b=erase_b, erase_fifth=erase_fifth,
                        swap_b=swap_b) for w in ww]
    a = torch.cat([pair[0] for pair in batch])
    e = torch.cat([pair[1] for pair in batch])
    model.eval()
    with torch.no_grad():
        cost = model(a, e, zero_backbone=zero_backbone,
                     zero_gate=zero_gate, zero_binder=zero_binder)
    choices = cost.argmin(-1).tolist()
    regret = [_regret(w, int(c)) for w, c in zip(ww, choices)]
    return {"choices": choices, "mean_regret": statistics.fmean(regret),
            "pergroup_regret": [statistics.fmean(regret[8*g:8*g+8])
                                 for g in range(GROUPS)],
            "history_gate_pair_flips": sum(choices[i] != choices[i+1]
                                   for i in range(0, len(choices), 2))}


def _validation(model, fit, labels):
    correct = 0
    with torch.no_grad():
        for first in range(TRAIN_PAIRS, TRAIN_PAIRS+VAL_PAIRS, 16):
            ids = range(first, first+16)
            rows = [fit[2*i+j] for i in ids for j in (0,1)]
            a = torch.stack([x["action"] for x in rows])
            e = torch.stack([x["effect"] for x in rows])
            y = torch.tensor([labels[i] for i in ids], dtype=torch.long)
            pred = model(a, e)
            correct += int((pred.argmin(-1) == y).sum())
    return correct / VAL_PAIRS


def train_arm(arm, seed, binder, fit, labels, draws):
    model = PairGate(arm, seed, binder)
    binder_hash = r40._hash(model.binder)
    before = r40._hash(model.history)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=.001, weight_decay=.01)
    checkpoints = []
    start = time.perf_counter()
    for k, ids in enumerate(draws, 1):
        a, e, y = factual_batch(fit, labels, ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(-model(a, e), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R42_NONFINITE_LOSS")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g != 0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R42_INVALID_REAL_GRADIENTS")
        nn.utils.clip_grad_norm_([p for p in model.parameters()
                                 if p.requires_grad], 1.)
        opt.step()
        if k in (192, 384):
            model.eval()
            checkpoints.append({"updates": k, "factual_fit_train_CE": float(loss),
                                "factual_fit_heldout_choice_accuracy":
                                    _validation(model, fit, labels)})
    if (len(checkpoints) != 2 or r40._hash(model.binder) != binder_hash
        or r40._hash(model.history) == before):
        raise ValueError("F8R42_TRAINING_OR_BINDER_CUSTODY_FAILURE")
    results = {}
    for panel in PANELS:
        results[panel] = {
            "active": score(model, panel),
            "erase_B": score(model, panel, erase_b=True),
            "swap_B": score(model, panel, swap_b=True),
            "erase_fifth": score(model, panel, erase_fifth=True),
            "zero_backbone": score(model, panel, zero_backbone=True),
            "zero_gate": score(model, panel, zero_gate=True),
            "zero_binder": score(model, panel, zero_binder=True)}
    cold = PairGate(arm, seed, binder)
    cold.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    if (r40._hash(cold) != r40._hash(model) or
        any(score(cold, p)["choices"] != results[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R42_COLD_ACTION_RESTORE_FAILURE")
    return {"arm": arm, "seed": seed, "optimizer_updates": STEPS,
            "frozen_binder_hash": binder_hash,
            "history_initial_hash": before,
            "history_trained_hash": r40._hash(model.history),
            "trainable_parameters": sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
            "wall_seconds": time.perf_counter()-start,
            "checkpoints": checkpoints, "panels": results,
            "cold_restore": True}


def _observed_b_peak(effect_history):
    """Read the strongest actually selected B measurement, never generator metadata."""
    if (effect_history.ndim != 2 or effect_history.shape[-1] != 16 or
        effect_history.shape[0] not in (14, 22)):
        raise ValueError("F8R42_INVALID_OBSERVED_B_HISTORY")
    observed_b = effect_history[5:-1, 0]
    if not bool(torch.isfinite(observed_b).all()):
        raise ValueError("F8R42_NONFINITE_OBSERVED_B_HISTORY")
    peak = int(torch.argmax(torch.abs(observed_b)))
    return r31._physical_delta(observed_b[peak])


def positive_control(fit, panel):
    fifth_offsets = [r31._physical_delta(fit[2*i]["effect"][4, 0])
                     for i in range(TRAIN_PAIRS)]
    offset = statistics.fmean(fifth_offsets)
    ww = worlds(panel)
    choices = []
    for w in ww:
        a, e = candidates(w)
        fifth = r31._physical_delta(e[0, 4, 0]) - offset
        h = _observed_b_peak(e[0])
        c = w["public_context"]
        choices.append(c if fifth*h > 0 else 1-c)
    regrets = [_regret(w, c) for w,c in zip(ww,choices)]
    return {"factual_fit_fifth_offset": offset, "choices": choices,
            "mean_regret": statistics.fmean(regrets),
            "history_gate_pair_flips": sum(choices[i] != choices[i+1]
                                          for i in range(0,len(choices),2))}


def _bootstrap_lower(native, control):
    rng = random.Random(BOOT_SEED)
    values = []
    for _ in range(BOOT_DRAWS):
        ix = [rng.randrange(GROUPS) for __ in range(GROUPS)]
        values.append(statistics.fmean(control[k]-native[k] for k in ix))
    return sorted(values)[int(.025*BOOT_DRAWS)]


def judge(models, positive, freeze):
    t = freeze["prospective_score"]
    by_key = {(m["arm"], m["seed"]): m for m in models}
    reports = []
    for seed in SEEDS:
        native = by_key["native_pair_gate",seed]
        control = by_key["history_only_pair_gate",seed]
        comparison = {}
        for panel in PANELS:
            vals = native["panels"][panel]
            base = vals["active"]
            baseline = control["panels"][panel]["active"]
            contenders = [m["panels"][panel]["active"]
                          for m in models if m["arm"] != "native_pair_gate"]
            strongest = min(contenders, key=lambda x: x["mean_regret"])
            benefit = strongest["mean_regret"]-base["mean_regret"]
            def penalty(name):
                return vals[name]["mean_regret"]-base["mean_regret"]
            lower = _bootstrap_lower(base["pergroup_regret"],
                                       strongest["pergroup_regret"])
            success = bool(
                base["mean_regret"] <= t["maximum_native_regret_each"] and
                base["history_gate_pair_flips"] >= t["minimum_native_gate_pair_flips_each"] and
                penalty("erase_B") >= t["minimum_native_regret_penalty_erase_B_each"] and
                penalty("zero_gate") >= t["minimum_native_regret_penalty_zero_gate_each"] and
                penalty("zero_backbone") >= t["minimum_native_regret_penalty_zero_backbone_each"] and
                baseline["mean_regret"]-base["mean_regret"] >=
                    t["minimum_native_regret_gain_over_same_seed_history_only_each"] and
                benefit >= t["minimum_native_regret_gain_over_strongest_fully_trained_conventional_each"] and
                penalty("zero_binder") >= t["minimum_native_regret_gain_over_binder_only_each"] and
                penalty("erase_fifth") >= t["minimum_native_regret_penalty_erase_fifth_each"] and
                lower > t["paired_group_bootstrap_lower_gain_vs_strongest_conventional_gt"])
            comparison[panel] = {"native_regret": base["mean_regret"],
               "native_gate_flips": base["history_gate_pair_flips"],
               "strongest_control_regret": strongest["mean_regret"],
               "B_erasure_penalty": penalty("erase_B"),
               "native_backbone_penalty": penalty("zero_backbone"),
               "gain_over_history_only": baseline["mean_regret"]-base["mean_regret"],
               "gain_over_strongest_conventional": benefit,
               "bootstrap_lower": lower, "frozen_pass": success}
        reports.append({"seed": seed, "panels": comparison,
                        "both_panel_pass": all(x["frozen_pass"]
                                               for x in comparison.values())})
    positive_ok = all(positive[p]["mean_regret"] == 0 and
                      positive[p]["history_gate_pair_flips"] >=
                      t["minimum_positive_control_gate_pair_flips"]
                      for p in PANELS)
    passed = sum(x["both_panel_pass"] for x in reports)
    return {"factual_positive_diagnostic_pass":positive_ok,
            "native_both_panels_pass_seeds":passed,
            "frozen_native_incremental_development_pass":
                  passed >= t["minimum_native_both_panel_pass_seeds"] and positive_ok,
            "independent_scientific_pass":False,"seeds":reports}


def run():
    freeze = locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit, labels = factual_fit()
    # Pinned original R39 factual binder replay; never access original external simulator.
    original_fit = r33.fit_cohort()
    winners = r36.fit_pairs(original_fit)
    pools = r37.selection_pools(original_fit, winners)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(21249999)
    indices = torch.randint(0, len(pools["informative"]), (512, 16),
                            generator=generator).tolist()
    saved = {}
    for seed in SEEDS:
        saved[seed], receipt = r39.replay_original_seed(
            seed, original_fit, winners, pools, indices)
        if not receipt["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R42_BASE_BINDER_REPLAY_DRIFT")
    g = torch.Generator(device="cpu")
    g.manual_seed(SAMPLE_SEED)
    draws = torch.randint(0, TRAIN_PAIRS, (STEPS, BATCH),
                          generator=g).tolist()
    models = [train_arm(arm, seed, saved[seed], fit, labels, draws)
              for arm, seed in ARMS]
    positive = {panel: positive_control(fit, panel) for panel in PANELS}
    verdict = judge(models,positive,freeze)
    return {"schema": SCHEMA, "freeze_blob": FREEZE_BLOB,
            "parent": PARENT, "factual_fit_episodes":len(fit),
            "factual_train_pairs": TRAIN_PAIRS,
            "factual_heldout_pairs": VAL_PAIRS,
            "original_binder_replay_updates":len(SEEDS)*512,
            "new_neural_updates":len(ARMS)*STEPS,
            "models":models,"positive_diagnostic":positive,
            "judgment":verdict,"synthetic_development_only":True,
            "independent_scientific_pass":False,
            "spent_external_firstuse":False,"GPU":False,"deployment":"NONE"}


def retain_ci_receipt(x):
    if os.getenv("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r42_history_binding_interaction_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R42_RECEIPT_EXISTS")
    path.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_firstuse(*args, **kwargs):
    raise PermissionError("F8R42_EXTERNAL_SCIENTIFIC_FIRSTUSE_DENIED")
