"""F8-R46: fresh observed-probe calibration and tagged causal selection.

This is a frozen-source synthetic DEV falsifier, not independent acceptance.
Only actually selected observations can enter decisions or training. The
original R45 scientific NO-GO and scores remain immutable.
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
from experiments import p85mr_f8r44_gradient_gate_attribution as r44

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r46_probe_centered_learning_freeze.json"
FREEZE_BLOB = "266703eefe3d90111c432f9fcc79f4ac54fdf619"
SCHEMA = "ttt-hybrid.f8r46-probe-centered-conditional-learning-freeze.v1"
PARENT = "f42313838efa6c93726e028c5a004e01f90edb33"
SALT = {"fit": 22010000, "cal": 22020000, "ood": 22030000}
SEEDS = (985201, 985207)
ARMS = tuple((arm, seed) for arm in (
    "calibrated_content_native", "calibrated_content_history_only",
    "unconditioned_content_native", "trained_GRU") for seed in SEEDS)
FIT_TRAIN_GROUPS, FIT_GROUPS, EVAL_GROUPS = 24, 32, 12
STEPS, BATCH = 192, 16
TRAIN_PAIRS, HELDOUT_PAIRS = 192, 64
DRAW_SEED, BOOT_SEED, BOOT_DRAWS = 22049999, 22069999, 10000
PROBE, CUE, DECOY = .42, .42, .18
PANELS = ("cal", "ood")


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() +
                        b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R46_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_ANY_R46_IMPLEMENTATION_SOURCE_BIRTH_OR_SCORING" or
        f["lineage"]["parent_sha"] != PARENT or
        f["source"]["salts"] != SALT or
        f["source"]["train_pairs"] != TRAIN_PAIRS or
        f["source"]["heldout_pairs"] != HELDOUT_PAIRS or
        f["source"]["fit_selected_returns"] != 512 or
        f["methods"]["train_updates_each"] != STEPS or
        f["methods"]["sample_seed"] != DRAW_SEED or
        f["preregistered_score"]["bootstrap"]["seed"] != BOOT_SEED or
        f["preregistered_score"]["independent_scientific_pass"] is not False or
        any(f["authority"][k] for k in (
            "new_workflow", "new_runner", "new_controller", "paid_api",
            "spent_F7_or_prior_scored_source_replay",
            "original_external_first_use", "GPU_first_use",
            "merge", "promotion", "deployment"))):
        raise ValueError("F8R46_FROZEN_METHOD_OR_AUTHORITY_DRIFT")
    if tuple(f["methods"]["two_seeds"]) != SEEDS or tuple(
        f["methods"]["mechanism_arms"]) != tuple(dict.fromkeys(
            arm for arm, _ in ARMS)):
        raise ValueError("F8R46_MODEL_PROTOCOL_DRIFT")
    r44.locked()
    return f


def positions(panel, group):
    if (panel not in SALT or type(group) is not int or
        not 0 <= group < (FIT_GROUPS if panel == "fit" else EVAL_GROUPS)):
        raise ValueError("F8R46_UNREGISTERED_POSITIONS")
    length = 16 if panel == "ood" else 8
    digest = hashlib.sha256(
        f"f8r46/{SALT[panel]}/{group}".encode()).digest()
    cue = int.from_bytes(digest[:4], "big") % length
    decoy = int.from_bytes(digest[4:8], "big") % (length-1)
    if decoy >= cue:
        decoy += 1
    return cue, decoy


def world(panel, group, context, sign, gate):
    positions(panel, group)
    if context not in (0, 1) or sign not in (-1, 1) or gate not in (-1, 1):
        raise ValueError("F8R46_UNREGISTERED_WORLD")
    return {"panel": panel, "group": group, "public_context": context,
            "q_evaluator_only": sign, "gate_evaluator_only": gate,
            "return_state": (.61, .73)[group % 2], "goal": 1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R46_UNKNOWN_PANEL")
    length = FIT_GROUPS if panel == "fit" else EVAL_GROUPS
    return [world(panel, g, c, q, h)
            for g in range(length)
            for c in (0, 1) for q in (-1, 1) for h in (-1, 1)]


def b_history(w):
    panel, group, gate = (w["panel"], w["group"],
                          w["gate_evaluator_only"])
    cue, decoy = positions(panel, group)
    length = 16 if panel == "ood" else 8
    rows = []
    for t in range(length):
        if t == cue:
            state, delta = CUE, .24*gate
        elif t == decoy:
            state, delta = DECOY, -.24*gate
        else:
            state = 0.
            delta = round(
                .006 * math.sin((SALT[panel]+group)*.137+t*.271)
                + .004 * (((t+group)%4)-1.5), 8)
        rows.append((state, t % 2, delta))
    return rows


def _following(w, action):
    s, c = w["return_state"], w["public_context"]
    q = w["q_evaluator_only"] * w["gate_evaluator_only"]
    return (r41._law(s, action, c, q, "ood") if w["panel"] == "ood"
            else r33.r32.after(s, action, c, q))


def selected_history(w, *, erase_B=False, erase_tag=False,
                     erase_fifth=False, swap_B=False):
    c, q = w["public_context"], w["q_evaluator_only"]
    rows = [r33.b._record(s, act, r33.r32.after(s, act, c, q), c)
            for s, act in r33.r32.EARLY]
    rows.append(r33.b._record(PROBE, c,
            r33.r32.after(PROBE, c, c, q), c))
    for tag, act, delta in b_history(w):
        if swap_B and tag in (CUE, DECOY):
            delta = -delta
        observed_tag = 0. if erase_tag and tag == CUE else tag
        action, effect = r33.b._record(
            observed_tag, act, observed_tag + delta, c)
        effect = effect.clone()
        effect[4] = 0.
        rows.append((action, effect))
    a = torch.stack([z[0] for z in rows])
    e = torch.stack([z[1] for z in rows])
    if erase_B:
        e[5:, :].zero_()
    if erase_fifth:
        e[4, :].zero_()
    return a, e


def candidates(w, **kwargs):
    a, e = selected_history(w, **kwargs)
    return (torch.stack([
                torch.cat((a, r33.b._record(
                    w["return_state"], choice, w["return_state"],
                    w["public_context"])[0][None]))
                for choice in (0, 1)]),
            torch.stack([
                torch.cat((e, torch.zeros((1, 16), dtype=e.dtype)))
                for _ in (0, 1)]))


def balanced_mean_check(panel):
    n = FIT_GROUPS if panel == "fit" else EVAL_GROUPS
    worst = 0.
    for g in range(n):
        x1, e1 = candidates(world(panel, g, 0, 1, 1))
        x2, e2 = candidates(world(panel, g, 0, 1, -1))
        if not torch.equal(x1, x2):
            raise ValueError("F8R46_PUBLIC_KEY_OR_RETURN_CHOICE_LEAK")
        mismatch = (e1[:, 5:-1].mean(1) -
                    e2[:, 5:-1].mean(1)).abs().max().item()
        worst = max(worst, mismatch)
        if mismatch > 1e-7:
            raise ValueError("F8R46_B_MEAN_GATE_LEAK")
        if torch.equal(e1[:, 5:-1], e2[:, 5:-1]):
            raise ValueError("F8R46_CONSEQUENCE_NOT_OBSERVABLE")
    return {"groups": n, "max_mean_delta": worst}


def factual_fit():
    locked()
    records, labels = [], []
    for w in worlds("fit"):
        costs = []
        for chosen in (0, 1):
            a, e = selected_history(w)
            following = _following(w, chosen)
            outcome_key, outcome_effect = r33.b._record(
                w["return_state"], chosen, following,
                w["public_context"])
            aa = torch.cat((a, outcome_key[None]))
            ee = torch.cat((e, outcome_effect[None]))
            selected_following = (w["return_state"]
                                  + r31._physical_delta(ee[-1, 0]))
            costs.append((selected_following - 1.)**2)
            records.append({"action": aa, "effect": ee})
        labels.append(min((0, 1), key=lambda a: (costs[a], a)))
    if (len(records) != 512 or len(labels) != 256 or
        labels.count(0) != 128):
        raise ValueError("F8R46_INVALID_FACTUAL_FIT")
    return records, labels


def cal_offset(facts):
    """FIT-only midpoint of actually observed selected fifth outcomes."""
    if len(facts) != 512:
        raise ValueError("F8R46_INVALID_CALIBRATION_SOURCE")
    observed = [PROBE + r31._physical_delta(
                facts[2*i]["effect"][4, 0]) for i in range(TRAIN_PAIRS)]
    estimate = statistics.fmean(observed)
    if not math.isfinite(estimate):
        raise ValueError("F8R46_INVALID_FIT_CALIBRATION")
    return estimate


def _public_sensor(action, effect, offset):
    """Only observed selected action keys + effects, never simulator metadata."""
    state, context, ret = r31._decode_key(action[-1])
    fifth_after = PROBE + r31._physical_delta(effect[4, 0])
    q_signal = fifth_after - offset
    b = []
    for j in range(5, action.shape[0] - 1):
        tag, _, _ = r31._decode_key(action[j])
        delta = r31._physical_delta(effect[j, 0])
        b.append((tag, delta))
    cue_delta = next((delta for tag, delta in b
                      if abs(tag-CUE)<1e-5), None)
    if cue_delta is None:
        raise ValueError("F8R46_NO_OBSERVED_CUE")
    optimal = context if q_signal * cue_delta > 0 else 1 - context
    return {"choice": optimal, "q_centered": q_signal,
            "cue_selected_delta": cue_delta}


def _regret(w, choice):
    if choice not in (0, 1):
        raise ValueError("F8R46_BAD_RETURN_ACTION")
    costs = [(_following(w, act) - w["goal"])**2 for act in (0, 1)]
    return max(0., costs[choice] - min(costs))


def observation_ceiling(offset):
    proof = {}
    for panel in ("fit",) + PANELS:
        ww = worlds(panel)
        decisions = []
        for w in ww:
            a, e = candidates(w)
            decisions.append(_public_sensor(a[0], e[0], offset)["choice"])
        mistakes = sum(_regret(w, action) > 1e-10 for w, action
                       in zip(ww, decisions))
        proof[panel] = {"worlds": len(ww), "mistakes": mistakes,
                        "mean_regret": statistics.fmean(
                            _regret(w, a) for w, a in zip(ww, decisions))}
        if mistakes:
            raise ValueError("F8R46_SOURCE_NOT_OBSERVATION_IDENTIFIABLE")
    return proof


class ConditionalReader(nn.Module):
    """Learn which observed B role matters; calibrate q only with FIT data."""
    def __init__(self, arm, seed, binder, offset):
        super().__init__()
        if (arm, seed) not in ARMS or arm == "trained_GRU":
            raise ValueError("F8R46_UNREGISTERED_CONDITIONAL")
        self.arm = arm
        self.binder = copy.deepcopy(binder)
        for p in self.binder.parameters():
            p.requires_grad_(False)
        self.binder.eval()
        self.register_buffer("fit_offset", torch.tensor(offset, dtype=torch.float32))
        self.native = (r34.FactualMemory("native_factual_memory", seed)
            if arm != "calibrated_content_history_only" else None)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(22080000 + seed + ARMS.index((arm, seed)))
            self.attention = nn.Sequential(
                nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
            self.gain = nn.Parameter(torch.tensor(3., dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.binder.eval()
        return self

    def forward(self, a, e, *, zero_native=False, zero_reader=False,
                uniform=False, zero_binder=False):
        if (a.ndim != 3 or a.shape != e.shape or a.shape[-1] != 16 or
            a.shape[1] not in (14, 22) or a.shape[0] % 2):
            raise ValueError("F8R46_BAD_PAIRED_FACTUAL_INPUT")
        observed = e.clone()
        observed[:, -1, :].zero_()
        b = torch.cat((a[:, 5:-1, :], observed[:, 5:-1, :]), dim=-1)
        attn = self.attention(b).squeeze(-1)
        use_uniform = uniform or self.arm == "unconditioned_content_native"
        weights = (torch.full_like(attn, 1. / attn.shape[-1])
                   if use_uniform else attn.softmax(dim=-1))
        b_value = (weights * observed[:, 5:-1, 0]).sum(dim=-1)
        if zero_reader:
            b_value = torch.zeros_like(b_value)
        # Normalized selected probe differences must be decoded as a delta
        # and ADDED ONCE to the actual public probe state, not the goal.
        q_signal = (PROBE + 2.4*torch.atanh(
                    observed[:, 4, 0].clamp(-.98, .98))
                    - self.fit_offset)
        # Pair candidates see identical past; don't duplicate q/B training labels.
        q_signal = q_signal.reshape(-1, 2).mean(dim=-1)
        b_value = b_value.reshape(-1, 2).mean(dim=-1)
        bind = r38.predict(self.binder, "binder", a, observed).reshape(-1, 2)
        bind_strength = .5 + torch.sigmoid(
            (bind[:, 1] - bind[:, 0]).abs())
        if zero_binder:
            bind_strength = torch.zeros_like(bind_strength)
        if self.native is None:
            native_factor = torch.ones_like(q_signal)
        else:
            native_cost = self.native(
                a, observed, ablate_backbone=zero_native).reshape(-1, 2)
            native_factor = 1. + .3 * torch.tanh(
                native_cost[:, 1] - native_cost[:, 0])
        scale = F.softplus(self.gain) * 40.
        pref = (scale * q_signal * b_value
                * native_factor * bind_strength).tanh()
        # Decode public action-key context only; no q/gate/evaluator oracle.
        contexts = torch.tensor([
            r31._decode_key(a[i, -1])[1]
            for i in range(0, len(a), 2)],
            device=pref.device, dtype=torch.long)
        cost0 = torch.where(contexts == 0, -pref, pref)
        return torch.stack((cost0, -cost0), dim=-1)


class FactualPolicy(nn.Module):
    def __init__(self, arm, seed, binder, offset):
        super().__init__()
        if (arm, seed) not in ARMS:
            raise ValueError("F8R46_UNREGISTERED_POLICY")
        self.arm = arm
        self.inner = (r44.ConventionalPairGate("gru_conventional", seed, binder)
            if arm == "trained_GRU" else ConditionalReader(arm, seed, binder, offset))

    def train(self, mode=True):
        super().train(mode)
        self.inner.binder.eval()
        return self

    def forward(self, a, e, *, zero_native=False, zero_reader=False,
                uniform=False, zero_binder=False):
        if self.arm == "trained_GRU":
            return self.inner(a, e, zero_backbone=zero_native,
                zero_gate=zero_reader, zero_binder=zero_binder)
        return self.inner(a, e, zero_native=zero_native,
            zero_reader=zero_reader, uniform=uniform, zero_binder=zero_binder)


def batch(facts, labels, indices):
    if (len(indices) != BATCH or any(type(i) is not int or
         not 0 <= i < TRAIN_PAIRS for i in indices)):
        raise ValueError("F8R46_BAD_TRAIN_PAIR_INDEX")
    rows = [facts[2*i+j] for i in indices for j in (0, 1)]
    return (torch.stack([r["action"] for r in rows]),
            torch.stack([r["effect"] for r in rows]),
            torch.tensor([labels[i] for i in indices], dtype=torch.long))


def heldout_accuracy(model, facts, labels):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS, TRAIN_PAIRS + HELDOUT_PAIRS, BATCH):
            indices = list(range(start, start+BATCH))
            rows = [facts[2*i+j] for i in indices for j in (0, 1)]
            a = torch.stack([r["action"] for r in rows])
            e = torch.stack([r["effect"] for r in rows])
            target = torch.tensor([labels[i] for i in indices], dtype=torch.long)
            correct += int((model(a,e).argmin(-1) == target).sum())
    return correct/HELDOUT_PAIRS


def score(model, panel, *, erase_B=False, erase_tag=False,
          erase_fifth=False, swap_B=False, zero_native=False,
          zero_reader=False, uniform=False, zero_binder=False):
    if panel not in PANELS:
        raise ValueError("F8R46_UNREGISTERED_SCORE_PANEL")
    ww = worlds(panel)
    src = [candidates(w, erase_B=erase_B, erase_tag=erase_tag,
                      erase_fifth=erase_fifth, swap_B=swap_B) for w in ww]
    a = torch.cat([x[0] for x in src])
    e = torch.cat([x[1] for x in src])
    model.eval()
    with torch.no_grad():
        costs = model(a, e, zero_native=zero_native,
            zero_reader=zero_reader, uniform=uniform,
            zero_binder=zero_binder)
    if costs.shape != (len(ww), 2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R46_INVALID_CAUSAL_ACTION_SCORE")
    choices = costs.argmin(-1).tolist()
    regrets = [_regret(w, a) for w, a in zip(ww, choices)]
    return {"choices": choices,
            "mean_regret": statistics.fmean(regrets),
            "pergroup_regret": [statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips": sum(choices[i] != choices[i+1]
                                   for i in range(0, len(choices), 2))}


def train_arm(arm, seed, binder, offset, facts, labels, draws):
    model = FactualPolicy(arm, seed, binder, offset)
    frozen_hash = r40._hash(model.inner.binder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=.001, weight_decay=.01)
    checkpoints = []
    started = time.perf_counter()
    for step, indices in enumerate(draws, 1):
        a, e, y = batch(facts, labels, indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(-model(a,e), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R46_NONFINITE_FIT_LOSS")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g != 0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R46_FIT_GRADIENT_FAILURE")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if step in (64,128,192):
            checkpoints.append({"updates":step,
                "loss":float(loss.detach()),
                "heldout_accuracy":heldout_accuracy(model, facts, labels)})
    if frozen_hash != r40._hash(model.inner.binder):
        raise ValueError("F8R46_BINDER_CHANGED")
    scores = {p:{"active":score(model,p)} for p in PANELS}
    for name, kwargs in (
        ("erase_B", {"erase_B":True}),
        ("erase_tag", {"erase_tag":True}),
        ("erase_fifth", {"erase_fifth":True}),
        ("swap_B", {"swap_B":True}),
        ("uniform", {"uniform":True}),
        ("zero_native", {"zero_native":True}),
        ("zero_reader", {"zero_reader":True}),
        ("zero_binder", {"zero_binder":True})
    ):
        scores["ood"][name] = score(model, "ood", **kwargs)
    cold = FactualPolicy(arm, seed, binder, offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    if (r40._hash(cold) != r40._hash(model) or
        any(score(cold,p)["choices"] != scores[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R46_COLD_RESTORE_FAILURE")
    return {"arm":arm, "seed":seed, "updates":STEPS,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                       if p.requires_grad),
            "fit_seconds":time.perf_counter()-started,
            "checkpoints":checkpoints, "panels":scores, "cold_restore":True}


def bootstrap_lower(a,b):
    rng = random.Random(BOOT_SEED)
    vals = []
    for _ in range(BOOT_DRAWS):
        indices = [rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        vals.append(statistics.fmean(b[i]-a[i] for i in indices))
    return sorted(vals)[int(.025*BOOT_DRAWS)]


def judge(models, frozen):
    t = frozen["preregistered_score"]
    by = {(r["arm"],r["seed"]):r for r in models}
    verdicts = []
    for seed in SEEDS:
        row=by["calibrated_content_native",seed]
        uniform=by["unconditioned_content_native",seed]
        others=[r for r in models if r["arm"] in (
            "calibrated_content_history_only","trained_GRU")]
        strongest=min(others,key=lambda r:r["panels"]["ood"]["active"]["mean_regret"])
        cur=row["panels"]["ood"]["active"]
        cal=row["panels"]["cal"]["active"]
        ablation=lambda name:row["panels"]["ood"][name]["mean_regret"]-cur["mean_regret"]
        best=strongest["panels"]["ood"]["active"]
        lower=bootstrap_lower(cur["pergroup_regret"],best["pergroup_regret"])
        accept=bool(
            row["checkpoints"][-1]["heldout_accuracy"] >=
                t["FIT_heldout_accuracy_each_seed_at_least"]
            and cal["mean_regret"] <= t["cal_mean_regret_each_seed_max"]
            and cur["mean_regret"] <= t["ood_mean_regret_each_seed_max"]
            and cal["gate_pair_flips"] >= t["cal_gate_reversals_each_seed_at_least"]
            and cur["gate_pair_flips"] >= t["ood_gate_reversals_each_seed_at_least"]
            and ablation("erase_B") >= t["B_erasure_regret_penalty_at_least"]
            and ablation("erase_tag") >= t["cue_tag_erasure_regret_penalty_at_least"]
            and ablation("uniform") >= t["uniform_pool_regret_penalty_at_least"]
            and ablation("erase_fifth") >= t["fifth_erasure_regret_penalty_at_least"]
            and ablation("zero_native") >= t["native_backbone_penalty_at_least"]
            and uniform["panels"]["ood"]["active"]["mean_regret"] -
                cur["mean_regret"] >=
                t["learned_reader_gain_vs_own_unconditioned_at_least"]
            and lower > 0)
        verdicts.append({"seed":seed,"development_conditions_pass":accept,
            "heldout_FIT":row["checkpoints"][-1]["heldout_accuracy"],
            "cal_regret":cal["mean_regret"],"ood_regret":cur["mean_regret"],
            "gate_flips":cur["gate_pair_flips"],
            "control":strongest["arm"],
            "control_ood_regret":best["mean_regret"],
            "ablation_regret_penalties":{x:ablation(x) for x in (
                "erase_B","erase_tag","uniform","erase_fifth",
                "zero_native","zero_reader","zero_binder")},
            "bootstrap_lower_gain_vs_strongest":lower})
    return {"two_seed_dev_pass":all(z["development_conditions_pass"]
                                  for z in verdicts),
            "seeds":verdicts, "whole_process_compute_matched":False,
            "independent_scientific_pass":False}


def run():
    frozen = locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity={p:balanced_mean_check(p) for p in ("fit",)+PANELS}
    facts,labels=factual_fit()
    offset=cal_offset(facts)
    ceiling=observation_ceiling(offset)
    original=r33.fit_cohort()
    winners=r36.fit_pairs(original)
    pools=r37.selection_pools(original, winners)
    generator=torch.Generator(device="cpu")
    generator.manual_seed(21249999)
    old_draws=torch.randint(0,len(pools["informative"]), (512,16),
                            generator=generator).tolist()
    binders={}
    for seed in SEEDS:
        binders[seed], receipt=r39.replay_original_seed(
            seed,original,winners,pools,old_draws)
        if not receipt["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R46_REFERENCE_BINDER_DRIFT")
    sampler=torch.Generator(device="cpu")
    sampler.manual_seed(DRAW_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(STEPS,BATCH),generator=sampler).tolist()
    rows=[train_arm(arm,seed,binders[seed],offset,facts,labels,draws)
          for arm,seed in ARMS]
    verdict=judge(rows,frozen)
    return {"schema":SCHEMA, "freeze_blob":FREEZE_BLOB,
            "parent":PARENT,
            "factual_selected_returns":len(facts),
            "fit_train_pairs":TRAIN_PAIRS,
            "fit_heldout_pairs":HELDOUT_PAIRS,
            "probe_calibration_from_FIT_only":offset,
            "B_paired_mean_parity":parity,
            "observational_reference":ceiling,
            "binder_replay_updates":len(SEEDS)*512,
            "new_neural_updates":len(ARMS)*STEPS,
            "models":rows, "judgment":verdict,
            "source_neutral_dev_only":True,
            "independent_scientific_pass":False,
            "external_first_use":False,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS") != "true":
        return
    path=ROOT/"local_runs/p85mr_f8r46_probe_centered_learning_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R46_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_first_use(*args, **kwargs):
    raise PermissionError("F8R46_EXTERNAL_FIRST_USE_UNAUTHORIZED")
