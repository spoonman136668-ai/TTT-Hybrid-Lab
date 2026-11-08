"""Pure science source-admission screening; must not create or consume worlds."""
from __future__ import annotations

from collections.abc import Mapping

ROLES = ("fit", "validation", "heldout_transfer", "interference",
         "signflip_negative", "bootstrap")
PREDECESSOR_USED = frozenset((7207,7219,7229,7307,7319,7327))
PREDECESSOR_SEALED = frozenset((7403,7411,7429,7507,7517,7529,
                               7603,7613,7627,7807))
S27_RESERVED = frozenset((92017,92029,92041,93017,93029,93041,
                          94017,94029,94041,95017,95029,95041,
                          96017,96029,96041,98017))


def inspect_candidate_seed_plan(roles: Mapping | None, *,
                                exclusive_owner_verified: bool = False,
                                first_use_authorized: bool = False,
                                exact_science_source_locked: bool = False) -> dict:
    """Fail closed on duplicate and earlier-source seeds, then authority/custody.

    No seed is generated or consumed here. Passing schema checks never
    dispatches scientific first use; ownership must be independently witnessed.
    """
    if not isinstance(roles, Mapping) or set(roles) != set(ROLES):
        return {"admitted": False, "reason": "P1_MISSING_OR_EXTRA_ROLE"}
    seen = set()
    for role in ROLES:
        values = roles[role]
        if not isinstance(values, (list, tuple)) or len(values) < 1:
            return {"admitted": False, "reason": "P1_UNASSIGNED_ROLE"}
        for n in values:
            if type(n) is not int or n < 0:
                return {"admitted": False, "reason": "P1_INVALID_SEED_TYPE"}
            if n in seen:
                return {"admitted": False, "reason": "P1_OVERLAPPING_SEED_ROLES"}
            seen.add(n)
    if seen & PREDECESSOR_USED:
        return {"admitted": False, "reason": "P1_S26_CONSUMED_SEED_COLLISION"}
    if seen & PREDECESSOR_SEALED:
        return {"admitted": False, "reason": "P1_S26_SEALED_SEED_COLLISION"}
    if seen & S27_RESERVED:
        return {"admitted": False, "reason": "P1_S27_RESERVED_SEED_COLLISION"}
    if not exclusive_owner_verified:
        return {"admitted": False, "reason": "P1_EXTERNAL_SOURCE_OWNERSHIP_UNVERIFIED"}
    if not exact_science_source_locked:
        return {"admitted": False, "reason": "P1_SCIENCE_SOURCE_LOCK_ABSENT"}
    if not first_use_authorized:
        return {"admitted": False, "reason": "P1_FIRST_USE_NOT_AUTHORIZED"}
    return {"admitted": True, "reason": "SOURCE_ADMISSION_INPUTS_ASSERTED_ONLY",
            "warning": "Must separately validate source, owner, runner and signed receipts."}
