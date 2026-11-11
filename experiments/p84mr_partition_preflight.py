"""P84-MR independent external-science seed custody — OFFLINE PREFLIGHT ONLY.

The original P81 freeze has 127 cross-split episode-seed collisions. This
module detects them and requires the P84 prospectively frozen replacement.
Neither an external source nor a model is imported, run, reset, trained,
scored, or authorized by any function in this file.

The P84 freeze is SHA-pinned; changing thresholds/source identity after
preregistration must fail closed rather than adapt to observed results.
"""
import hashlib
import json
import math
from pathlib import Path

FREEZE_PATH = "p84mr_external_science_preflight_freeze.json"
FREEZE_GIT_BLOB = "9c259f7e03ecbd32f8d3e5f9bfdaef2a8860f724"
EXPECTED_SCHEMA = "ttt-hybrid.p84mr-disjoint-external-causal-science-preflight-freeze.v1"
INT32_MAX = 2147483647
SPLITS = (
    "fit", "cal", "dev_external",
    "science_ood_0", "science_ood_1", "science_ood_2",
)
NAMESPACE = "ttt-hybrid/p84mr/independent-cartpolewind-source-v1"
NO_REAL_SOURCE_FIRST_USE = "P84MR_SCIENTIFIC_FIRST_USE_NOT_AUTHORIZED"


def _locked_freeze():
    path = Path(__file__).with_name(FREEZE_PATH)
    raw = path.read_bytes()
    git_blob = hashlib.sha1(
        f"blob {len(raw)}\0".encode("ascii") + raw
    ).hexdigest()
    if git_blob != FREEZE_GIT_BLOB:
        raise ValueError("P84MR_FROZEN_PROTOCOL_DRIFT")
    cfg = json.loads(raw)
    if (cfg["schema"] != EXPECTED_SCHEMA or
            cfg["execution_authority"]["first_use_status"] != "NOT_AUTHORIZED"):
        raise ValueError("P84MR_PROTOCOL_AUTHORITY_DRIFT")
    return cfg


def _ranges(bases, counts, *, stride=1):
    if set(bases) != set(SPLITS) or set(counts) != set(SPLITS):
        raise ValueError("P84MR_SPLIT_SCHEMA_DRIFT")
    if (type(stride) is not int or stride <= 0 or
            any(type(bases[k]) is not int or type(counts[k]) is not int
                or counts[k] <= 0 for k in SPLITS)):
        raise ValueError("P84MR_BAD_SEED_SPEC")
    out = {}
    for split in SPLITS:
        values = tuple(bases[split] + stride * index for index in range(counts[split]))
        if any(v < 0 or v > INT32_MAX for v in values):
            raise ValueError("P84MR_SEED_OUT_OF_RANGE")
        out[split] = values
    return out


def _collisions(scheduled):
    """Cross-split collisions only, in deterministic lexicographic order."""
    collision_pairs = []
    for index, a in enumerate(SPLITS):
        for b in SPLITS[index + 1:]:
            intersection = sorted(set(scheduled[a]) & set(scheduled[b]))
            if intersection:
                collision_pairs.append({
                    "a": a,
                    "b": b,
                    "overlap": len(intersection),
                    "first": intersection[0],
                })
    return collision_pairs


def preflight_partition():
    """Verify source *schedule*, not source outcomes or third-party provenance."""
    cfg = _locked_freeze()
    old = cfg["preexisting_protocol_negative_control"]
    previous = _ranges(
        old["previous_split_bases"], old["previous_counts"], stride=10
    )
    prior_pairs = _collisions(previous)
    expected = old["observed_collision_pairs"]
    if (len(prior_pairs) != len(expected) or
            any(p["a"] != e["a"] or p["b"] != e["b"] or
                p["overlap"] != e["overlap"] or p["first"] != e["example"]
                for p, e in zip(prior_pairs, expected))):
        raise ValueError("P84MR_LEGACY_SEED_COLLISION_AUDIT_DRIFT")
    if (sum(p["overlap"] for p in prior_pairs) != 127 or
            sum(map(len, previous.values())) != 448 or
            len(set().union(*(set(x) for x in previous.values()))) != 321):
        raise ValueError("P84MR_EXPECTED_PREVIOUS_FAILURE_NOT_REPRODUCED")

    spec = cfg["frozen_successor"]
    if spec["namespace"] != NAMESPACE:
        raise ValueError("P84MR_NAMESPACE_DRIFT")
    scheduled = _ranges(spec["seed_bases"], spec["episode_counts"])
    if len({v for vals in scheduled.values() for v in vals}) != spec["expected_unique"]:
        raise ValueError("P84MR_CROSS_SPLIT_SEED_COLLISION")
    if _collisions(scheduled):
        raise ValueError("P84MR_CROSS_SPLIT_SEED_COLLISION")
    if sum(len(scheduled[k]) for k in SPLITS) != spec["total"] or spec["total"] != 448:
        raise ValueError("P84MR_EPISODE_TOTAL_DRIFT")

    regimes = cfg["upstream_regimes"]
    if set(regimes) != set(SPLITS):
        raise ValueError("P84MR_REGIME_SCHEMA_DRIFT")
    for k in SPLITS:
        reg = regimes[k]
        if (set(reg) != {"wind_mean", "wind_std", "init_theta_mean",
                         "init_theta_std"} or
                any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in reg.values()) or reg["wind_std"] < 0 or
                reg["init_theta_std"] < 0):
            raise ValueError("P84MR_BAD_REGIME_SPEC")

    checksum = hashlib.sha256()
    for k in SPLITS:
        for i, seed in enumerate(scheduled[k]):
            checksum.update(f"{NAMESPACE}|{k}|{i}|{seed}\n".encode("ascii"))
    return {
        "schema": "ttt-hybrid.p84mr-partition-offline-preflight-receipt.v1",
        "status": "OFFLINE_SEED_PREFLIGHT_PASS_NO_SOURCE_BIRTH",
        "legacy_total_cross_split_collisions": 127,
        "legacy_pairs": prior_pairs,
        "new_episode_count": spec["total"],
        "new_unique_seeds": spec["expected_unique"],
        "cross_split_collisions": 0,
        "proposed_schedule_sha256": checksum.hexdigest(),
        "frozen_git_blob": FREEZE_GIT_BLOB,
        "external_source_executed": False,
        "external_data_generated": False,
        "external_model_scored": False,
        "owner_first_use": "NOT_AUTHORIZED",
        "scientific_pass": False,
    }


def proposed_seed(split, episode_index):
    """Offline-only planned seed; does not fetch, initialize or run a source."""
    cfg = _locked_freeze()
    spec = cfg["frozen_successor"]
    if (split not in SPLITS or type(episode_index) is not int or
            episode_index < 0 or episode_index >= spec["episode_counts"][split]):
        raise ValueError("P84MR_UNREGISTERED_EPISODE")
    return spec["seed_bases"][split] + episode_index


def planned_discrete_action(split, episode_index, step):
    """No external action dispatch or outcome access."""
    cfg = _locked_freeze()
    if type(step) is not int or not 1 <= step <= 96:
        raise ValueError("P84MR_BAD_ACTION_STEP")
    seed = proposed_seed(split, episode_index)
    message = f"{NAMESPACE}/{split}/{episode_index}/{seed}/{step}".encode("ascii")
    return hashlib.sha256(message).digest()[0] & 1


def forbid_first_use(*_args, **_kwargs):
    """Fail closed regardless of caller-supplied source labels or approvals."""
    raise PermissionError(NO_REAL_SOURCE_FIRST_USE)
