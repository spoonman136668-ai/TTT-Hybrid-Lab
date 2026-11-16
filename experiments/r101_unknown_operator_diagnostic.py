"""R101 CPU TEST-only source and learned mechanism development (NOT scored science).

Operator functions are evaluator-only physical rules. Learners see no operator
identifier, hardcoded algebra, private factor signs, unchosen reward or teacher
label. All R101 scored panels are DISABLED until exact-SHA original first-use
custody is independently qualified; only independent UNIT_ONLY fixtures run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r98_native_program_diagnostic as r98

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r101_unknown_operator_freeze.json"
FREEZE_BLOB = "38146a196b59bc78ca66d43d2179e8e465dd54fa"
SCHEMA = "ttt-hybrid.r101-unknown-operator-induction.cpu-testonly.v1"
PARENT = "f56274e1d6ee67d8f6bf70d3e1b39cb07de12b82"
SEEDS = (1002511, 1002523, 1002537)
UNIT_SALT = 25429991
KNOWN = ("MUX3", "PARITY3", "AND3", "OR3", "NAND3")
UNSEEN = ("MAJORITY3", "EXACTLY_ONE3")
ARMS = (
    "NATIVE32_LEARNED_SUPPORT_MEMORY",
    "NATIVE32_SAME_INIT_SHUFFLED_SUPPORT_OUTCOMES",
    "NATIVE32_SAME_INIT_NO_SUPPORT_OUTCOMES",
    "NATIVE32_SAME_INIT_STATE_RESET_BEFORE_QUERY",
    "EXTERNAL_LEARNED_SUPPORT_STATE_SAME_CAPACITY",
    "ROLE_EQUIVARIANT_SUPPORT_SET",
    "GRU64_FACTUAL_SUPPORT",
    "MLP64_FLATTENED_FACTUAL_SUPPORT",
    "STATIC_PRIOR_NO_FEEDBACK",
)
NATIVE, SHUFFLED, NO_OUTCOME, RESET, EXTERNAL, ROLE_SET, GRU, MLP, STATIC = ARMS
STEPS, BATCH, SUPPORT = 3072, 16, 6


def frozen():
    """Pin the unmodified freeze-only Git blob and mandatory negatives."""
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R101_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (
        f["schema"] != SCHEMA
        or f["stage"] != "PROSPECTIVE_FREEZE_ONLY_BEFORE_R101_IMPLEMENTATION_TESTS_AND_FIRST_USE"
        or f["parent"]["r100_sealed_parent_SHA"] != PARENT
        or f["parent"]["prior_scored_source_or_weights_reused"]
        or tuple(f["source"]["seeds"]) != SEEDS
        or tuple(f["source"]["known_function_families"]) != KNOWN
        or tuple(f["source"]["unseen_function_families"]) != UNSEEN
        or f["source"]["salts"]["UNIT_ONLY"] != UNIT_SALT
        or len(set(f["source"]["salts"].values())) != len(f["source"]["salts"])
        or tuple(f["arms"]) != ARMS
        or f["source"]["support_factual_interventions"] != SUPPORT
        or f["training"]["steps_each_arm"] != STEPS
        or f["training"]["batch"] != BATCH
        or f["evaluation"]["independent_seeds"] != 3
        or f["evaluation"]["independent_scientific_pass"]
        or not f["authority"]["prospective_protocol_freeze_and_isolated_CPU_engineering_proposal"]
        or any(bool(v) for k, v in f["authority"].items()
               if k != "prospective_protocol_freeze_and_isolated_CPU_engineering_proposal")
    ):
        raise PermissionError("R101_FROZEN_PROTOCOL_OR_AUTHORITY_MISMATCH")
    return f


def _admit(*_unused, **_kw):
    """No scored first-use admission on engineering-only candidate."""
    frozen()
    raise PermissionError("R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED")


def _world(panel, seed, index, sha=None, *, event=0, family=None):
    """Prospective original source constructor; ALWAYS sealed on this branch.

    Once a separately qualified, original first-use route exists, the
    immutable source identity, run and owner scope must be admitted before
    ANY original physical world is constructed. UNIT_ONLY is elsewhere.
    """
    freeze = frozen()
    if panel not in freeze["source"]["counts"]:
        raise PermissionError("R101_OLD_SCORED_OR_PROTECTED_SOURCE_DENIED")
    if seed not in SEEDS or not isinstance(index, int) or not 0 <= index < freeze["source"]["counts"][panel]:
        raise PermissionError("R101_UNFROZEN_SOURCE_INDEX_OR_SEED")
    if not isinstance(event, int) or not 0 <= event <= SUPPORT:
        raise PermissionError("R101_UNREGISTERED_PHYSICAL_EVENT")
    # Keep historical and future original source shut before evaluating
    # family-specific selectors. This preserves admission-negative behavior.
    if not freeze["authority"]["protected_scored_first_use"]:
        raise PermissionError("R101_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED")
    if family not in KNOWN + UNSEEN:
        raise PermissionError("R101_UNREGISTERED_PHYSICAL_FAMILY")
    if panel in ("FIT", "VALID", "DEV_FAMILIAR") and family not in KNOWN:
        raise PermissionError("R101_UNSEEN_OPERATOR_NOT_TRAINING")
    if panel in ("DEV_UNKNOWN", "OOD_SHIFT", "DELAYED_COLD", "REVERSAL") and family not in UNSEEN:
        raise PermissionError("R101_FAMILIAR_OPERATOR_NOT_HELDOUT_UNKNOWN")
    _admit(seed=seed, candidate_sha=sha)
    rng = np.random.default_rng(np.random.SeedSequence(
        [freeze["source"]["salts"][panel], seed, index, event]))
    pivot_selector = 1 if panel == "DELAYED_COLD" else index % 2
    physics_index = (2 * (index % (r85.N // 2)) + pivot_selector) % r85.N
    world = r85._make_synthetic_world(physics_index, rng)
    world["factors_a"] = rng.choice([-1, 1], size=4).astype(np.int64)
    world["r101_roles"] = tuple(int(k) for k in rng.permutation(4)[:3])
    world["law_a"] = _operator_truth(world["factors_a"], world["r101_roles"], family)
    if panel == "OOD_SHIFT":
        world = r86._shift_public_world(world, index)
    return world


def _operator_truth(factors, roles, family):
    """EVALUATOR ONLY. Never import/call this function from any model."""
    if family not in KNOWN + UNSEEN:
        raise ValueError("R101_UNKNOWN_PHYSICAL_OPERATOR")
    factors = np.asarray(factors)
    if factors.shape != (4,) or not np.all(np.isin(factors, [-1, 1])):
        raise ValueError("R101_INVALID_PRIVATE_FACTORS")
    if len(roles) != 3 or len(set(roles)) != 3 or any(not 0 <= k < 4 for k in roles):
        raise ValueError("R101_INVALID_PHYSICAL_OPERAND_ROLES")
    a, b, c = (int(factors[k]) > 0 for k in roles)
    if family == "MUX3":
        out = b if a else c
    elif family == "PARITY3":
        out = (int(a) + int(b) + int(c)) % 2 == 1
    elif family == "AND3":
        out = a and b and c
    elif family == "OR3":
        out = a or b or c
    elif family == "NAND3":
        out = not (a and b and c)
    elif family == "MAJORITY3":
        out = int(a) + int(b) + int(c) >= 2
    else:
        out = int(a) + int(b) + int(c) == 1
    return 1 if out else -1


def _public_view(world, chosen):
    """Original lawful R85 224-tick physics; only four A cues and role tags."""
    prefix, reward = r86._view(world, chosen)
    slots = r88._public_slots(prefix).copy()
    tags = np.zeros(4, np.float32)
    for k, role in enumerate(world["r101_roles"], start=1):
        tags[role] = k / 4.0
    # Channel 8 is a public PREACTION role marker, not an operator opcode.
    slots[:4, 8] = tags
    if slots.shape != (5, 13) or not np.all(np.isfinite(slots)):
        raise RuntimeError("R101_PUBLIC_PHYSICAL_INPUT_INVALID")
    return slots, float(reward)


def _unit_world(index, event, family):
    if not isinstance(index, int) or not 0 <= index < 14 or not 0 <= event <= 6:
        raise PermissionError("R101_UNIT_ONLY_INDEX_BOUNDS")
    if family not in KNOWN + UNSEEN:
        raise PermissionError("R101_UNIT_ONLY_UNREGISTERED_FAMILY")
    rng = np.random.default_rng(np.random.SeedSequence([UNIT_SALT, index, event]))
    # All six supports have the SAME public pivot stratum, enabling 3+3
    # chosen-action support shuffling without single-member strata.
    parity = index % 2
    physics_index = (2 * (index % (r85.N // 2)) + parity) % r85.N
    world = r85._make_synthetic_world(physics_index, rng)
    roles = tuple(int(k) for k in rng.permutation(4)[:3])
    factors = rng.choice([-1, 1], size=4).astype(np.int64)
    world["factors_a"] = factors
    world["r101_roles"] = roles
    world["law_a"] = _operator_truth(factors, roles, family)
    return world


def unit_fixture(index):
    """Separate non-scored, non-trainable fixture; keep truth away from models."""
    frozen()
    if not isinstance(index, int) or not 0 <= index < 14:
        raise PermissionError("R101_UNIT_ONLY_FIXTURE_RANGE")
    family = (KNOWN + UNSEEN)[index % (len(KNOWN) + len(UNSEEN))]
    observed, rewards, actions = [], [], []
    for j in range(SUPPORT):
        w = _unit_world(index, j, family)
        a = r85.CALIBRATE if j % 2 == 0 else r85.EXPLOIT_POS
        slots, reward = _public_view(w, a)
        observed.append(slots)
        rewards.append(reward)
        actions.append(a)
    qworld = _unit_world(index, SUPPORT, family)
    q0, r0 = _public_view(qworld, r85.CALIBRATE)
    q2, r2 = _public_view(qworld, r85.EXPLOIT_POS)
    if not np.array_equal(q0[:4], q2[:4]) or not np.array_equal(q0[-1, 4:], q2[-1, 4:]):
        raise RuntimeError("R101_PHYSICAL_QUERY_CHOICE_CAUSAL_LEAK")
    return {
        "public": {
            "support_slots": np.stack(observed),
            "support_rewards": np.asarray(rewards, dtype=np.float32),
            "support_actions": np.asarray(actions, dtype=np.int8),
            "query0": q0, "query2": q2,
        },
        "evaluator_only": {"family": family, "private_law": qworld["law_a"],
                           "return0": r0, "return2": r2},
    }


def support_shuffle_indices(actions, slots, index):
    """Public-stratum derangement fixed without inspecting rewards/law."""
    if actions.shape != (SUPPORT,) or slots.shape != (SUPPORT, 5, 13):
        raise ValueError("R101_SHUFFLE_PUBLIC_SOURCE_SHAPE")
    if not isinstance(index, int) or not 0 <= index < 14:
        raise PermissionError("R101_UNIT_ONLY_SHUFFLE")
    perm = np.arange(SUPPORT)
    rng = np.random.default_rng(np.random.SeedSequence([UNIT_SALT, index, 901]))
    keys = [(int(actions[i]), float(slots[i, -1, 12])) for i in range(SUPPORT)]
    for group in sorted(set(keys)):
        ix = [i for i, key in enumerate(keys) if key == group]
        if len(ix) < 2:
            raise RuntimeError("R101_NOT_ENOUGH_MATCHED_SUPPORTS_FOR_DERANGEMENT")
        ix = np.asarray(ix, dtype=np.int64)
        shift = int(rng.integers(1, len(ix)))
        perm[ix] = np.roll(ix, shift)
    if any(perm[i] == i or keys[i] != keys[perm[i]] for i in range(SUPPORT)):
        raise RuntimeError("R101_SUPPORT_SHUFFLE_FIXED_POINT_OR_STRATUM_DRIFT")
    return perm


class PublicEncoder(nn.Module):
    """Shared observed-cue/role encoder: NO Boolean or opcode shortcuts."""
    def __init__(self):
        super().__init__()
        self.event = nn.Sequential(nn.Linear(2, 32), nn.GELU(),
                                   nn.Linear(32, 32), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(35, 32), nn.GELU(),
                                 nn.Linear(32, 32))

    def forward(self, slots):
        if slots.ndim != 3 or slots.shape[1:] != (5, 13):
            raise RuntimeError("R101_LEARNER_INPUT_MUST_BE_PUBLIC_5x13")
        z = self.event(torch.stack((slots[:, :4, 10], slots[:, :4, 8]), -1)).mean(1)
        action_time = slots[:, -1, :][:, [0, 2, 12]]
        return self.mix(torch.cat((z, action_time), -1))


class LearnedSupport(nn.Module):
    def __init__(self, width=32, pooled=False):
        super().__init__()
        self.public = PublicEncoder()
        self.pooled = pooled
        if pooled:
            self.update = nn.Sequential(nn.Linear(33, width), nn.GELU(),
                                        nn.Linear(width, width))
        else:
            self.update = nn.GRU(33, width, batch_first=True)

    def forward(self, slots, rewards):
        if slots.ndim != 4 or slots.shape[1:] != (SUPPORT, 5, 13):
            raise RuntimeError("R101_SUPPORT_SHAPE_INVALID")
        if rewards.shape != slots.shape[:2]:
            raise RuntimeError("R101_FACTUAL_SUPPORT_REWARDS_SHAPE_INVALID")
        batch = len(slots)
        vectors = self.public(slots.reshape(-1, 5, 13)).reshape(batch, SUPPORT, 32)
        combined = torch.cat((vectors, rewards.unsqueeze(-1)), dim=-1)
        if self.pooled:
            return self.update(combined).mean(dim=1)
        return self.update(combined)[1][-1]


class NativeSupportValue(nn.Module):
    def __init__(self, core, mode="real"):
        super().__init__()
        self.core = core
        self.support = LearnedSupport(32)
        self.projection = nn.Linear(32, 32)
        self.mode = mode

    def forward(self, support_slots, support_rewards, query):
        if self.mode == "none":
            support_rewards = torch.zeros_like(support_rewards)
        memory = self.support(support_slots, support_rewards)
        if self.mode == "reset":
            memory = torch.zeros_like(memory)
        stem = self.core.stem(query)
        context = self.projection(memory)
        stem = torch.cat((stem[:, :-1], stem[:, -1:] + context[:, None]), 1)
        return r98._native_trunk(self.core, stem)


class ExternalSupportValue(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.support = LearnedSupport(32)
        # Nearly identical adapter parameter budget to native-internal
        # projection; both arms share native core and lawful query access.
        self.value = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                   nn.Linear(32, 1))

    def forward(self, support_slots, support_rewards, query):
        latent = self.support(support_slots, support_rewards)
        return self.core(query)[:, -1, 0] + self.value(latent)[:, 0]


class ConventionalValue(nn.Module):
    """One real, end-to-end trainable graph per conventional comparator.

    Instantiate only layers invoked by this comparator. The previous common
    constructor registered unused support/flat/head weights in other modes,
    inflating naive capacity counts without providing trainable capacity.
    """
    def __init__(self, mode):
        super().__init__()
        if mode not in ("static", "flat", "gru", "set"):
            raise ValueError("R101_CONVENTIONAL_MODE_NOT_REGISTERED")
        self.mode = mode
        self.query = PublicEncoder()
        if mode == "flat":
            self.support_public = PublicEncoder()
            self.flat = nn.Sequential(nn.Linear(230, 64), nn.GELU(),
                                      nn.Linear(64, 64), nn.GELU(),
                                      nn.Linear(64, 1))
        else:
            if mode in ("gru", "set"):
                self.support = LearnedSupport(64 if mode == "gru" else 32,
                                              pooled=(mode == "set"))
            width = 64 if mode == "gru" else 32
            self.head = nn.Sequential(nn.Linear(width + 32, 64),
                                      nn.GELU(), nn.Linear(64, 1))

    def forward(self, support_slots, support_rewards, query):
        q = self.query(query)
        if self.mode == "static":
            return self.head(torch.cat((torch.zeros_like(q), q), -1))[:, 0]
        if self.mode == "flat":
            batch = len(query)
            v = self.support_public(support_slots.reshape(-1, 5, 13)).reshape(batch, -1)
            return self.flat(torch.cat((v, support_rewards, q), -1))[:, 0]
        z = self.support(support_slots, support_rewards)
        return self.head(torch.cat((z, q), -1))[:, 0]


def make_models(seed):
    """Original R74/R75 core unchanged, same-init native negative variants."""
    if seed not in SEEDS:
        raise PermissionError("R101_MODELS_ONLY_FROZEN_SEEDS")
    result = {}
    for name, mode in ((NATIVE, "real"), (SHUFFLED, "real"),
                       (NO_OUTCOME, "none"), (RESET, "reset")):
        core = r95._native(seed)
        torch.manual_seed(seed + 101)
        result[name] = NativeSupportValue(core, mode)
    core = r95._native(seed)
    torch.manual_seed(seed + 101)
    result[EXTERNAL] = ExternalSupportValue(core)
    # Compare ACTUAL trainable adapters, not the >16M-param native core
    # shared by both arms. Counting the huge common core could conceal a
    # material mismatch in the learned support mechanism being tested.
    def adapter_count(model):
        return sum(p.numel() for name, p in model.named_parameters()
                   if not name.startswith("core.") and p.requires_grad)
    native_params = adapter_count(result[NATIVE])
    external_params = adapter_count(result[EXTERNAL])
    if (native_params == 0 or external_params == 0
        or abs(native_params - external_params) >
            .10 * max(native_params, external_params)):
        raise RuntimeError("R101_EXTERNAL_LEARNED_STATE_ADAPTER_NOT_CAPACITY_MATCHED")
    ref_core = dict(result[NATIVE].core.named_parameters())
    external_core = dict(result[EXTERNAL].core.named_parameters())
    if (ref_core.keys() != external_core.keys()
        or any(not torch.equal(ref_core[name],external_core[name])
               for name in ref_core)):
        raise RuntimeError("R101_EXTERNAL_ORIGINAL_NATIVE_CORE_INIT_DIFFERS")
    for i, (name, mode) in enumerate(((ROLE_SET, "set"), (GRU, "gru"),
                                      (MLP, "flat"), (STATIC, "static"))):
        torch.manual_seed(seed + 150 + i)
        result[name] = ConventionalValue(mode)
    ref = dict(result[NATIVE].named_parameters())
    for name in (SHUFFLED, NO_OUTCOME, RESET):
        params = dict(result[name].named_parameters())
        if params.keys() != ref.keys() or any(
            not torch.equal(ref[k], params[k]) for k in ref):
            raise RuntimeError("R101_SAME_INIT_NEGATIVES_NOT_SAME")
    return result


def unit_value(model, fixture, action=r85.CALIBRATE, shuffled=False):
    """Unit-only forward; scientific scoring and training remain disabled."""
    public = fixture["public"]
    perm = support_shuffle_indices(public["support_actions"], public["support_slots"], 0)
    rewards = public["support_rewards"][perm] if shuffled else public["support_rewards"]
    return model(
        torch.as_tensor(public["support_slots"])[None],
        torch.as_tensor(rewards)[None],
        torch.as_tensor(public["query0" if action == r85.CALIBRATE else "query2"])[None]
    )


def original_ci(*_unused, **_kw):
    _admit()
