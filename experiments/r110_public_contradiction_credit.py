"""R110 bounded observational contradiction credit from *actual chosen* public facts.

The score is the evidence of two public-phase rules vs one shared rule under
a 256-Boolean-rule reference model. It is neither ground-truth change labels
nor a claim of causal identification. Never admit UNIT_UNSEEN or original
protected science, private truth tables, future query rewards or weights.
"""
from __future__ import annotations

import math

import numpy as np

from experiments import r104_operator_identifiability_diagnostic as r104
from experiments import r110_public_custody as custody

PUBLIC_FIT_KEYS = frozenset((
    "support_slots", "support_rewards", "support_actions",
    "support_context", "support_rewards_shuffled", "query", "target"))
LOG_MATCH = math.log(.85)
LOG_MISMATCH = math.log(.15)


def _logmeanexp(values):
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (256,) or not bool(np.isfinite(v).all()):
        raise RuntimeError("R110_PUBLIC_BAYES_CREDIT_LIKELIHOOD_INVALID")
    maximum = float(v.max())
    return maximum + math.log(float(np.exp(v - maximum).mean()))


def factual_credit(slots, returns, actions, contexts):
    """P(two independent rules | observed support), never evaluator truth."""
    custody.audit_freeze()
    slots = np.asarray(slots, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    contexts = np.asarray(contexts, dtype=np.int64)
    if (slots.shape != (16,5,13) or returns.shape != (16,)
        or actions.shape != (16,) or contexts.shape != (16,)
        or not bool(np.isfinite(slots).all())
        or not bool(np.isfinite(returns).all())
        or not bool(np.isin(actions, (0, 2)).all())
        or not bool(np.isin(contexts, (0,1)).all())
        or not bool(np.array_equal(contexts, np.sort(contexts)))
        or int((contexts==0).sum()) not in (6,8,10)
        or float(np.max(np.abs(returns))) > 10):
        raise ValueError("R110_NONPUBLIC_OR_INVALID_FACTUAL_SUPPORT_DENIED")
    patterns=np.asarray(r104._patterns_from_public(slots),dtype=np.int64)
    if patterns.shape!=(16,) or not bool(np.all((0<=patterns)&(patterns<8))):
        raise ValueError("R110_PHYSICAL_ROLE_PATTERN_INVALID")
    rules=np.arange(256,dtype=np.uint16)
    guess=((rules[:,None]>>patterns[None,:])&1)!=0
    match=guess==(returns[None,:]>0)
    likelihood=np.where(match,LOG_MATCH,LOG_MISMATCH)
    a=likelihood[:,contexts==0].sum(axis=1)
    b=likelihood[:,contexts==1].sum(axis=1)
    same=_logmeanexp(a+b)
    independent=_logmeanexp(a)+_logmeanexp(b)
    # Stable logistic; a 0.5 prior is specified in the immutable freeze.
    log_ratio=np.clip(independent-same,-30.,30.)
    probability=float(1./(1.+math.exp(-float(log_ratio))))
    return float(np.clip(probability,.05,.95))


def fit_credit(data):
    """Only exact FIT panel's public support fields, no private evaluator data."""
    custody.audit_freeze()
    if type(data) is not dict or frozenset(data)!=PUBLIC_FIT_KEYS:
        raise PermissionError("R110_NONFIT_OR_PRIVILEGED_CREDIT_SOURCE_DENIED")
    n=len(data["support_slots"])
    if n!=176 or data["target"].shape!=(n,2):
        raise PermissionError("R110_FIT_SOURCE_OR_PRIVATE_TARGET_DRIFT")
    values=np.asarray([factual_credit(
        data["support_slots"][i],data["support_rewards"][i],
        data["support_actions"][i],data["support_context"][i])
        for i in range(n)],dtype=np.float32)
    if values.shape!=(176,) or not bool(np.isfinite(values).all()):
        raise RuntimeError("R110_PUBLIC_CREDIT_NONFINITE")
    return values
