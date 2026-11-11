"""P81-MR inert source-collection seam: NO environment calls, NO external first-use.

Consumes caller-supplied fictional reset/step observations to test contract.
The actual CartPoleWindPCH.reset()/do() would be called by a separately
authorized, project-owned collector that is intentionally NOT implemented.
Never imports Causal-Gymnasium, opens files/network, trains or scores models.
"""
import hashlib
import math
import re

from p80mr_causalgym_trace_adapter import (
    CAPTURE_SCHEMA, UPSTREAM_REPO, UPSTREAM_SHA, UPSTREAM_PATH,
    UPSTREAM_BLOB, adapt_capture,
)

NAMESPACE = "ttt-hybrid/p81mr/collection-prospective-v1"
SPLIT_SEEDS = {
    "fit": 811101, "cal": 811103, "dev_external": 811107,
    "science_ood_0": 811109, "science_ood_1": 811111,
    "science_ood_2": 811113,
}
SPLIT_COUNTS = {
    "fit": 128, "cal": 64, "dev_external": 64,
    "science_ood_0": 64, "science_ood_1": 64,
    "science_ood_2": 64,
}
ALLOWED_TOY_SPLITS = frozenset(("fit", "cal"))
MAX_STEPS = 96


def _id(text):
    if (not isinstance(text, str) or not 1 <= len(text) <= 128 or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*", text)):
        raise ValueError("P81MR_BAD_CUSTODY_ID")
    return text


def _obs(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("P81MR_BAD_OBSERVATION")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in raw):
        raise ValueError("P81MR_BAD_OBSERVATION")
    return [float(x) for x in raw]


def _real(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("P81MR_BAD_REALIZED_REWARD")
    return float(value)


def frozen_episode_seed(split, index):
    if (split not in SPLIT_SEEDS or type(index) is not int or
            not 0 <= index < SPLIT_COUNTS[split]):
        raise ValueError("P81MR_UNREGISTERED_EPISODE")
    return SPLIT_SEEDS[split] + 10 * index


def planned_action(split, index, step):
    """Factual policy intent only: this does NOT execute the action."""
    if split not in ALLOWED_TOY_SPLITS:
        raise ValueError("P81MR_SPLIT_FIRST_USE_REQUIRED")
    seed = frozen_episode_seed(split, index)
    if type(step) is not int or not 1 <= step <= MAX_STEPS:
        raise ValueError("P81MR_BAD_STEP")
    data = f"{NAMESPACE}/{split}/{index}/{seed}/{step}".encode()
    return hashlib.sha256(data).digest()[0] & 1


class ToyCaptureRecorder:
    """One fictional episode, data supplied by tests, never an env object.

    The recorder has no reset/do/step execution method and cannot touch an
    external simulator. An uncertain/mismatched reported action permanently
    aborts this episode; no retry of that potentially effected action.
    """

    def __init__(self, *, source_kind, split, episode_index, initial_observation,
                 source_id, producer_id, custody_ref, world_id):
        # Refuse before inspecting any source object or external data.
        if source_kind != "toy_engineering":
            raise PermissionError("P81MR_EXTERNAL_FIRST_USE_NOT_AUTHORIZED")
        if split not in ALLOWED_TOY_SPLITS:
            raise PermissionError("P81MR_SPLIT_FIRST_USE_REQUIRED")
        self.episode_seed = frozen_episode_seed(split, episode_index)
        self.split = split
        self.episode_index = episode_index
        self.source_id = _id(source_id)
        self.producer_id = _id(producer_id)
        self.custody_ref = _id(custody_ref)
        self.world_id = _id(world_id)
        self._state = _obs(initial_observation)
        self._events = []
        self._pending = None
        self._ended = False
        self._aborted = False
        self._finished = False

    @property
    def event_count(self):
        return len(self._events)

    def prepare_action(self):
        if self._aborted or self._ended or self._finished or self._pending is not None:
            raise RuntimeError("P81MR_EPISODE_NOT_PREPARABLE")
        if len(self._events) == MAX_STEPS:
            raise RuntimeError("P81MR_STEP_BUDGET")
        event_id = len(self._events)+1
        action = planned_action(self.split, self.episode_index, event_id)
        self._pending = (event_id, action)
        return {"event_id": event_id, "intended_action": action}

    def record_observed_step(self, *, info, next_observation, reward,
                             terminated, truncated, receipt_ref):
        """Only consume actual supplied observations; never call the source."""
        if self._aborted or self._ended or self._finished or self._pending is None:
            raise RuntimeError("P81MR_NO_PENDING_EXECUTED_ACTION")
        try:
            if not isinstance(info, dict) or type(info.get("action")) is not int:
                raise ValueError("P81MR_APPLIED_ACTION_UNVERIFIED")
            step_id, intended = self._pending
            if info["action"] not in (0, 1) or info["action"] != intended:
                raise ValueError("P81MR_APPLIED_ACTION_MISMATCH")
            state_after = _obs(next_observation)
            result = _real(reward)
            if type(terminated) is not bool or type(truncated) is not bool:
                raise ValueError("P81MR_BAD_TERMINAL_FLAG")
            witness = _id(receipt_ref)
            row = {
                "event_id": step_id,
                "state_before": list(self._state),
                "executed_action": intended,
                "state_after": state_after,
                "reward": result,
                "terminated": terminated,
                "truncated": truncated,
                "mode": "do",
                "receipt_ref": witness,
            }
            # Capture only actual factual fields. Never copy info extras,
            # including hypothetical hidden wind, debug labels or oracle data.
            self._events.append(row)
            self._state = state_after
            self._ended = terminated or truncated
            self._pending = None
        except (ValueError, TypeError):
            self._aborted = True
            self._pending = None
            raise

    def finalize(self):
        """Return unchanged toy capture after >=2 observed factual events."""
        if self._aborted or self._finished or self._pending is not None:
            raise RuntimeError("P81MR_UNFINALIZABLE")
        if not self._ended and len(self._events) < MAX_STEPS:
            raise RuntimeError("P81MR_INCOMPLETE_EPISODE")
        if len(self._events) < 2:
            raise RuntimeError("P81MR_TOO_SHORT_NO_PADDING")
        world = {
            "world_id": self.world_id,
            "episode_seed": self.episode_seed,
            "transitions": [dict(e) for e in self._events],
            "probe_event_ids": list(range(2, len(self._events)+1)),
        }
        record = {
            "schema": CAPTURE_SCHEMA,
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": UPSTREAM_SHA,
            "upstream_path": UPSTREAM_PATH,
            "upstream_blob": UPSTREAM_BLOB,
            "source_id": self.source_id,
            "producer_id": self.producer_id,
            "custody_ref": self.custody_ref,
            "worlds": [world],
        }
        # Schema-only, no external prediction. No status elevates to science.
        adapted = adapt_capture(record)
        if adapted["custody"]["scientific_pass"] is not False:
            raise RuntimeError("P81MR_ADMISSION_BYPASS")
        self._finished = True
        return {
            "capture": record,
            "schema_only_sha256": adapted["custody"]["capture_sha256"],
            "classification": "TOY_ENGINEERING_ONLY",
            "external_environment_executed": False,
            "science_pass": False,
        }
