"""P69 prospectively frozen author-controlled *synthetic* causal-shift source.

Engineering source integrity ONLY. No training, holdout scoring, scientific first use,
or changes to the original Hybrid. Frozen contract:
experiments/p69_causal_shift_prospective_dev_freeze.json at 02c88c4f58579fa0fa3bdeda3236e4589b04bc2c.

World creation is independently seeded by (split, world index), using Python's
Mersenne Twister and deterministic, scalar modified Gram-Schmidt. Model writes
receive exactly the historical executed action and measured effect; the future
query and hidden role/change indicators are never provided by factual_history.
"""
import hashlib
import math
import random

import torch

DIM = 16
ROLES = 8
LENGTH = 96
CHANGED = 4
SEEDS = {"fit": 981101, "val": 981103, "ood_0": 981107,
         "ood_1": 981109, "ood_2": 981111, "negative": 981113}
COUNTS = {"fit": 512, "val": 192, "ood_0": 192,
          "ood_1": 192, "ood_2": 192, "negative": 96}


def _unit(v):
    norm = math.sqrt(math.fsum(x*x for x in v))
    if norm < 1e-10:
        raise ValueError("P69_DEGENERATE_ACTION")
    return [x/norm for x in v]


def _basis(rng):
    """Stable orthonormal 16D action basis, no platform-dependent QR library."""
    result = []
    for _ in range(DIM):
        v = [rng.gauss(0, 1) for _ in range(DIM)]
        for axis in result:
            proj = math.fsum(x*y for x, y in zip(v, axis))
            v = [x-proj*y for x, y in zip(v, axis)]
        result.append(_unit(v))
    return result


def _effect(rng):
    return [math.tanh(rng.gauss(0, 1)) for _ in range(DIM)]


def _jitter(rng, vector, sigma):
    return _unit([x+sigma*rng.gauss(0,1) for x in vector])


def _seed(split, index):
    payload = f"ttt-hybrid/p69/frozen-v1/{SEEDS[split]}/{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _segment_roles(rng, repeats):
    """Each of 8 roles appears exactly repeats times; balanced signs in pairs."""
    pairs = []
    for role in range(ROLES):
        signs = [1, -1] * (repeats//2)
        if repeats % 2:
            signs.append(rng.choice((-1, 1)))
        rng.shuffle(signs)
        pairs.extend((role, sign) for sign in signs)
    return pairs


def make_world(split, index):
    """Return one independent deterministic world; diagnostics are private to evaluator."""
    if split not in SEEDS or not isinstance(index, int) or not 0 <= index < COUNTS[split]:
        raise ValueError("P69_BAD_SOURCE_SELECTOR")
    rng = random.Random(_seed(split, index))
    is_ood = split.startswith("ood_")
    rep = 2 if is_ood else 3
    action_noise = 0.12 if is_ood else 0.04
    effect_noise = 0.06 if is_ood else 0.03
    basis = _basis(rng)
    original = [_effect(rng) for _ in range(ROLES)]
    changed = set(rng.sample(range(ROLES), CHANGED))
    new_effects = [_effect(rng) if r in changed else original[r] for r in range(ROLES)]
    boundary = rng.randrange(40, 57)  # position of first post-change event
    pre = [("target", r, s) for r, s in _segment_roles(rng, rep)]
    post = [("target", r, s) for r, s in _segment_roles(rng, rep)]
    pre += [("distractor", -1, 1)] * (boundary-len(pre))
    post += [("distractor", -1, 1)] * (LENGTH-boundary-len(post))
    rng.shuffle(pre)
    rng.shuffle(post)
    actions, outcomes, target_mask, role_ids, changed_mask = [], [], [], [], []
    for phase, segment in enumerate((pre, post)):
        for kind, role, sign in segment:
            if kind == "target":
                base = basis[role]
                action = _jitter(rng, [sign*x for x in base], action_noise)
                eff = original[role] if phase == 0 else new_effects[role]
                observed = [sign*x + effect_noise*rng.gauss(0,1) for x in eff]
                target_mask.append(True)
                role_ids.append(role)
                changed_mask.append(role in changed)
            else:
                action = _unit([rng.gauss(0,1) for _ in range(DIM)])
                observed = [x+effect_noise*rng.gauss(0,1) for x in _effect(rng)]
                target_mask.append(False)
                role_ids.append(-1)
                changed_mask.append(False)
            actions.append(action)
            outcomes.append(observed)
    # Evaluation-only final query; independent fresh observation after all writes.
    query_is_changed = bool(rng.randrange(2))
    eligible = sorted(changed) if query_is_changed else sorted(set(range(ROLES))-changed)
    query_role = rng.choice(eligible)
    query_sign = rng.choice((-1, 1))
    query_action = _jitter(rng, [query_sign*x for x in basis[query_role]], action_noise)
    query_effect = [query_sign*x + effect_noise*rng.gauss(0,1) for x in new_effects[query_role]]
    return {
        "action": torch.tensor(actions, dtype=torch.float32),
        "effect": torch.tensor(outcomes, dtype=torch.float32),
        "query_action": torch.tensor(query_action, dtype=torch.float32),
        "observed_query_effect": torch.tensor(query_effect, dtype=torch.float32),
        "role_audit": torch.tensor(role_ids, dtype=torch.int16),
        "target_audit": torch.tensor(target_mask, dtype=torch.bool),
        "role_changed_audit": torch.tensor(changed_mask, dtype=torch.bool),
        "changed_roles_audit": tuple(sorted(changed)),
        "boundary_audit": boundary,
        "query_role_audit": query_role,
        "query_changed_audit": query_is_changed,
        "query_sign_audit": query_sign,
    }


def make_batch(split, count, start=0):
    if split not in SEEDS or not isinstance(count, int) or not isinstance(start, int) or count <= 0 or start < 0 or start+count > COUNTS[split]:
        raise ValueError("P69_BAD_BATCH")
    worlds = [make_world(split, i) for i in range(start, start+count)]
    return {"action": torch.stack([w["action"] for w in worlds]),
            "effect": torch.stack([w["effect"] for w in worlds]),
            "query_action": torch.stack([w["query_action"] for w in worlds]),
            "observed_query_effect": torch.stack([w["observed_query_effect"] for w in worlds]),
            "target_audit": torch.stack([w["target_audit"] for w in worlds]),
            "role_audit": torch.stack([w["role_audit"] for w in worlds]),
            "role_changed_audit": torch.stack([w["role_changed_audit"] for w in worlds]),
            "boundary_audit": torch.tensor([w["boundary_audit"] for w in worlds]),
            "query_changed_audit": torch.tensor([w["query_changed_audit"] for w in worlds])}


def factual_history(batch):
    """The only permitted online inputs; query and hidden labels cannot leak."""
    if not isinstance(batch, dict) or not all(k in batch for k in ("action","effect")):
        raise ValueError("P69_BAD_HISTORY")
    return {"action": batch["action"], "effect": batch["effect"]}


def digest_world(world):
    fields=("action","effect","query_action","observed_query_effect")
    h=hashlib.sha256()
    for k in fields:
        h.update(k.encode()+b"\0")
        h.update(world[k].contiguous().numpy().tobytes())
    return h.hexdigest()


def _index_balanced_accuracy(labels):
    """Train on first half of worlds, predict second from event index only."""
    labels = labels.to(torch.bool)
    if labels.ndim != 2 or labels.shape[0] < 8:
        raise ValueError("P69_AUDIT_SHAPE")
    train, held = labels[:len(labels)//2], labels[len(labels)//2:]
    # Class imbalance cannot create a fake classifier win: use BA.
    pred=(train.float().mean(0) > .5)[None].expand_as(held)
    tpr=((pred & held).sum().float()/held.sum().clamp_min(1)).item()
    tnr=(((~pred)&(~held)).sum().float()/(~held).sum().clamp_min(1)).item()
    return (tpr+tnr)/2


def source_integrity(batch):
    """Structural audit on an ENGINEERING sample, never an OOD outcome score."""
    target=batch["target_audit"]
    num_worlds, num_events=target.shape
    if num_events != LENGTH or num_worlds < 8:
        raise ValueError("P69_AUDIT_SHAPE")
    frac=target.float().reshape(num_worlds,4,LENGTH//4).mean((0,2))
    deviation=float((frac-target.float().mean()).abs().max())
    index_ba=_index_balanced_accuracy(target)
    changed=batch["role_changed_audit"] & target
    role_change_ba=_index_balanced_accuracy(changed)
    query_label=batch["query_changed_audit"].bool()
    query_bias=abs(query_label.float().mean().item()-.5)
    same=(batch["action"] == batch["query_action"][:,None,:]).all(-1).any(-1)
    leakage=bool(same.any())
    return {"target_quartile_max_deviation":deviation,
            "index_only_target_balanced_accuracy":index_ba,
            "index_only_changed_role_balanced_accuracy":role_change_ba,
            "query_changed_fraction":float(query_label.float().mean()),
            "query_changed_imbalance":query_bias,
            "query_exact_duplicate":leakage,
            "target_quartile_pass":deviation <= .05,
            "index_target_pass":index_ba<=.55,
            "changed_role_index_pass":role_change_ba<=.55,
            "query_duplicate_pass":not leakage}
