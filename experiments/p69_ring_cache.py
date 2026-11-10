"""P69 bounded FIFO/round-robin factual key/value memory (DEV engineering only).

Each of eight slots stores a 16-float key and 16-float last-observed effect.
The one replacement cursor is represented by a key norm of 2 (other occupied
keys have norm 1; empty keys are zero). There is NO extra persistent counter.
The replacement cursor is the slot of the most recent MISS insertion; hits
update only the already-known value and never advance the cursor. An exact
cosine match requires >=0.98. Historical executed observations only.

This is an engineering comparator for a future prospectively frozen P69 source,
not an amendment of consumed P68 outcomes or independent scientific evidence.
"""
import torch
import torch.nn.functional as F

DIM = 16
SLOTS = 8
MATCH = .98


def _check_events(events):
    if not isinstance(events, dict) or set(events) != {'action', 'effect'}:
        raise ValueError('P69_FACTUAL_ONLY_REQUIRED')
    action, effect = events['action'], events['effect']
    if action.ndim != 3 or action.shape != effect.shape or action.shape[-1] != DIM:
        raise ValueError('P69_BAD_FACTS')
    if not bool(torch.isfinite(action).all() and torch.isfinite(effect).all()):
        raise ValueError('P69_NONFINITE_FACTS')
    if not bool((action.norm(dim=-1) > 1e-8).all()):
        raise ValueError('P69_ZERO_ACTION')
    return action, effect


def _check_carry(carry, batch, dtype, device):
    if not isinstance(carry, (tuple, list)) or len(carry) != 2:
        raise ValueError('P69_BAD_CARRY')
    keys, vals = carry
    if (keys.shape != (batch, SLOTS, DIM) or vals.shape != keys.shape or
        keys.dtype != dtype or vals.dtype != dtype or keys.device != device or vals.device != device):
        raise ValueError('P69_BAD_CARRY')
    if not bool(torch.isfinite(keys).all() and torch.isfinite(vals).all()):
        raise ValueError('P69_BAD_CARRY')
    norms = keys.norm(dim=-1)
    occupied = norms > .25
    markers = norms > 1.5
    valid_norms = ((norms < 1e-8) | ((norms - 1).abs() < .001) |
                   ((norms - 2).abs() < .001)).all(-1)
    valid = (valid_norms &
             (markers.sum(-1) == occupied.any(-1).long())).all()
    if not bool(valid):
        raise ValueError('P69_CORRUPT_CARRY')
    return keys.clone(), vals.clone()


def stream(events, carry=None):
    """Return only persistent `(keys, effects)`, shape `(B,8,16)` each.

    No step index, phase, ring pointer, role label or query is passed between
    calls; chunked calls reconstruct the same cursor solely from key norms.
    """
    actions, effects = _check_events(events)
    batch, steps, _ = actions.shape
    if carry is None:
        keys = actions.new_zeros(batch, SLOTS, DIM)
        vals = actions.new_zeros(batch, SLOTS, DIM)
    else:
        keys, vals = _check_carry(carry, batch, actions.dtype, actions.device)
    row = torch.arange(batch, device=actions.device)
    for t in range(steps):
        key = F.normalize(actions[:, t], dim=-1)
        effect = effects[:, t]
        norms = keys.norm(dim=-1)
        similarity = (F.normalize(keys, dim=-1) * key[:, None, :]).sum(-1)
        best, ix = similarity.max(dim=-1)
        hits = best >= MATCH
        # A key with zero norm is an empty slot. Fill lowest empty slot first.
        empty = norms < .25
        has_empty = empty.any(-1)
        empty_ix = empty.to(dtype=torch.int64).argmax(-1)
        # Occupied cache: last insertion's norm=2, all other keys norm=1.
        last_insert = norms.argmax(-1)
        next_ix = torch.where(has_empty, empty_ix, (last_insert+1) % SLOTS)
        # Hits update an observed effect, without advancing replacement cursor.
        if bool(hits.any()):
            rr, cc = row[hits], ix[hits]
            vals[rr, cc] = effect[hits]
        misses = ~hits
        if bool(misses.any()):
            rr, cc = row[misses], next_ix[misses]
            # Remove previous norm-2 marker (if present) from precisely one slot.
            old = norms[misses].argmax(-1)
            old_norm = norms[rr, old]
            prev = old_norm > 1.5
            if bool(prev.any()):
                rprev, cprev = rr[prev], old[prev]
                keys[rprev,cprev] /= 2
            keys[rr, cc] = key[misses] * 2
            vals[rr, cc] = effect[misses]
    return keys, vals


def predict(carry, query):
    """Read without modification; query is a factual executed action only."""
    if query.ndim != 2 or query.shape[-1] != DIM or not bool(torch.isfinite(query).all()):
        raise ValueError('P69_BAD_QUERY')
    keys, vals = _check_carry(carry, len(query), query.dtype, query.device)
    if not bool((query.norm(dim=-1)>1e-8).all()):
        raise ValueError('P69_ZERO_QUERY')
    score, ix = (F.normalize(keys, dim=-1) * F.normalize(query,dim=-1)[:,None,:]).sum(-1).max(-1)
    return vals[torch.arange(len(query),device=query.device), ix] * (score >= MATCH)[:,None]
