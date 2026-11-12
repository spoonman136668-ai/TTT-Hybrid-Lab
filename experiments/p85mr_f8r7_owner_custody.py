"""F8-R7 source-neutral durable custody prototype: NO real scientific first-use.

Provides an exclusive, fsynced, non-replayable local DEV journal for proposed
F8-R6 sessions. There is deliberately no source adapter, runner entrypoint,
workflow, real owner approval or API to authorize real effects. Filesystem
crash safety is a *candidate*, not a qualified native owner execution route.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from experiments import p85mr_f8r6_full_action_budget as r6

HERE = Path(__file__).resolve().parent
FREEZE = HERE / "p85mr_f8r7_owner_custody_freeze.json"
FREEZE_BLOB = "2928e7ae034d198c74513d2d7d30fadc1e7f08c4"
BUDGET_BLOB = "59475472a4c316fa04a3510e4138a872abc88a91"
R8_FREEZE = HERE / "p85mr_f8r8_owner_epoch_freeze.json"
R8_FREEZE_BLOB = "37c285ff776b71a1455f0ebe0379cd7ac945c1ff"
ATTEMPT_FILE = "DEV_ATTEMPT.json"
SCHEMA = "ttt-hybrid.f8r7-owner-native-custody-candidate.v1"
PROTOTYPE = "FABRICATED_DEV_ONLY"
PHASES = ("calibration", "fit", "selector", "test")
COUNTS = (48, 64, 16, 352)
HORIZON = 96
TOTAL_CLAIMS = 480
TOTAL_ACTION_RESERVE = 46080
EVENT_TYPES = ("constructor_intent", "reset_intent", "model_query_intent",
               "action_intent", "action_observed", "session_terminal")


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _canonical(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if _git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R7_FROZEN_CONTRACT_BLOB_DRIFT")
    data = json.loads(raw)
    r6_plan = r6.locked()
    budget = r6.FREEZE.read_bytes()
    if (_git_blob(budget) != BUDGET_BLOB
            or data["schema"] != SCHEMA
            or data["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_OR_TEST_RESULTS"
            or data["scope"]["budget_git_blob"] != BUDGET_BLOB
            or data["proposed_limits"]["sessions"] != TOTAL_CLAIMS
            or data["proposed_limits"]["real_source_max_effects"] !=
                TOTAL_ACTION_RESERVE
            or r6_plan["global"]["max_actual_selected_action_effects"] !=
                TOTAL_ACTION_RESERVE
            or data["proposed_limits"]["phase_counts"] !=
                dict(zip(PHASES, COUNTS))
            or data["authority"]["real_source_firstuse"] is not False
            or data["authority"]["owner_runner_execution"] is not False
            or data["authority"]["science_pass"] is not False):
        raise ValueError("F8R7_SCOPE_OR_AUTHORITY_DRIFT")
    return data


def r8_locked() -> dict:
    """Only the bounded DEV ledger can acquire a root marker."""
    locked()
    raw = R8_FREEZE.read_bytes()
    if _git_blob(raw) != R8_FREEZE_BLOB:
        raise ValueError("F8R8_FROZEN_OWNER_SCOPE_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.f8r8-global-dev-owner-attempt-exclusivity.v1"
            or f["stage"] !=
                "FROZEN_BEFORE_GLOBAL_OWNER_ATTEMPT_IMPLEMENTATION_AND_TEST"
            or f["scope"]["previous_R7_source_blob"] !=
                "e11868c2570c1c79ad6f23c149dd17d0dca85970"
            or f["scope"]["R6_budget_blob"] != BUDGET_BLOB
            or f["design"]["root_marker"] != ATTEMPT_FILE
            or f["limits"]["real_source_effects"] != 0
            or f["limits"]["owner_execution"] is not False
            or f["limits"]["independent_scientific_pass"] is not False):
        raise ValueError("F8R8_SCOPE_OR_AUTHORITY_DRIFT")
    return f


def _directory(path: Path) -> None:
    if path.exists():
        item = path.lstat()
        if not stat.S_ISDIR(item.st_mode) or path.is_symlink():
            raise ValueError("F8R7_UNTRUSTED_JOURNAL_DIRECTORY")
    else:
        path.mkdir(mode=0o700)
    # Project owner scientific execution uses WSL/ext4, where parent fsync
    # exists. Native Windows development tests have file fsync + exclusive
    # creation but cannot qualify parent-directory durability with os.open.
    # Never represent that fallback as native owner custody acceptance.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _exclusive(path: Path, record: dict) -> None:
    """Create once. If writing/fsync fails, NEVER erase/retry the claim."""
    raw = _canonical(record)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    _directory(path.parent)


def _read_exact(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("F8R7_INVALID_JOURNAL_RECORD")
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or len(raw) > 8192:
        raise ValueError("F8R7_TORN_OR_OVERSIZE_RECORD")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("F8R7_TORN_OR_INVALID_JSON") from exc
    if type(data) is not dict or _canonical(data) != raw:
        raise ValueError("F8R7_NONCANONICAL_RECORD")
    return data, _hash(raw)


def _key(identity: str) -> str:
    if type(identity) is not str or not identity or len(identity) > 512:
        raise ValueError("F8R7_INVALID_SESSION_IDENTITY")
    return _hash(identity.encode("utf-8"))


class DevCustodyJournal:
    """Pure fabricated test journal. Never accepts owner/real source receipts."""

    def __init__(self, root: Path, *, run_id: str = "DEV-RUN-1",
                 attempt: int = 1, classification: str = PROTOTYPE):
        r8_locked()
        if (classification != PROTOTYPE or type(attempt) is not int or
                attempt != 1 or type(run_id) is not str or
                re.fullmatch(r"DEV-RUN-[1-9][0-9]{0,9}", run_id) is None):
            raise PermissionError("F8R7_REAL_OWNER_FIRSTUSE_NOT_AUTHORIZED")
        self.root = Path(root)
        if not self.root.exists() or not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("F8R7_CALLER_MUST_PROVIDE_EXISTING_DEV_DIRECTORY")
        _directory(self.root)
        self.claims = self.root / "claims"
        self.events = self.root / "events"
        _directory(self.claims)
        _directory(self.events)
        self.plan = {s["identity"]: s for s in r6.planned_sessions()}
        if len(self.plan) != TOTAL_CLAIMS:
            raise ValueError("F8R7_GLOBAL_PLAN_DRIFT")
        self.run_id = run_id
        self.attempt = attempt
        self._live: set[str] = set()  # never serialized; crash forbids resumption
        # Global single-writer DEV attempt is irrevocable even after crash.
        proposal = {
            "schema": "ttt-hybrid.f8r8-global-dev-attempt.v1",
            "classification": PROTOTYPE, "run_id": run_id, "attempt": 1,
            "source_sha": r6.locked()["original_source"]["sha"],
            "r6_budget_blob": BUDGET_BLOB,
            "status": "GLOBAL_DEV_ATTEMPTED_NO_REPLAY",
        }
        try:
            _exclusive(self.root / ATTEMPT_FILE, proposal)
            self._writer = True
            self._global_attempt = proposal
        except FileExistsError:
            prior, _ = _read_exact(self.root / ATTEMPT_FILE)
            if (prior != dict(proposal, run_id=prior.get("run_id"))
                    or type(prior.get("run_id")) is not str
                    or re.fullmatch(r"DEV-RUN-[1-9][0-9]{0,9}",
                                    prior["run_id"]) is None):
                raise ValueError("F8R8_INVALID_EXISTING_GLOBAL_ATTEMPT")
            self._writer = False
            self._global_attempt = prior

    def _verify_global_attempt(self) -> None:
        record, _ = _read_exact(self.root / ATTEMPT_FILE)
        if record != self._global_attempt:
            raise ValueError("F8R8_GLOBAL_ATTEMPT_DRIFT")

    def _claim_path(self, identity: str) -> Path:
        return self.claims / (_key(identity) + ".json")

    def _events_path(self, identity: str) -> Path:
        return self.events / _key(identity)

    def _read_claim(self, identity: str) -> dict:
        row, _ = _read_exact(self._claim_path(identity))
        expected = self.plan.get(identity)
        if (expected is None or row != {
            "schema": SCHEMA, "classification": PROTOTYPE,
            "session_identity": identity, "source_seed": expected["source_seed"],
            "source_sha": r6.locked()["original_source"]["sha"],
            "phase": expected["phase"],
            "run_id": row.get("run_id"), "attempt": 1,
            "reserved_effects": HORIZON,
            "status": "ATTEMPTED_NO_REPLAY",
        } or not isinstance(row.get("run_id"), str) or
                not row["run_id"].startswith("DEV-RUN-")):
            raise ValueError("F8R7_INVALID_OR_TAMPERED_CLAIM")
        return row

    def _records(self, identity: str) -> list[dict]:
        directory = self._events_path(identity)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("F8R7_MISSING_OR_BAD_EVENT_DIRECTORY")
        paths = sorted(directory.iterdir())
        rows = []
        prior = "GENESIS"
        for i, path in enumerate(paths, 1):
            if path.name != f"{i:06d}.json":
                raise ValueError("F8R7_GAP_OR_UNEXPECTED_EVENT")
            item, digest = _read_exact(path)
            if (item.get("index") != i or item.get("previous_sha256") != prior
                    or item.get("kind") not in EVENT_TYPES
                    or item.get("session_identity") != identity
                    or item.get("classification") != PROTOTYPE):
                raise ValueError("F8R7_EVENT_CHAIN_DRIFT")
            rows.append(item)
            prior = digest
        self._validate_sequence(rows, phase=self._read_claim(identity)["phase"])
        return rows

    @staticmethod
    def _validate_sequence(rows: list[dict], *, phase: str) -> None:
        constructor = reset = 0
        action_step = 0
        pending: int | None = None
        terminal = False
        queried: set[int] = set()
        for row in rows:
            kind = row["kind"]
            step = row.get("step")
            action = row.get("action")
            digest = row.get("effect_sha256")
            fields = {"classification", "session_identity", "index",
                      "previous_sha256", "kind", "step", "action",
                      "effect_sha256"}
            if kind == "session_terminal":
                fields.add("outcome")
            if set(row) != fields or terminal:
                raise ValueError("F8R7_UNKNOWN_FIELDS_OR_EVENT_AFTER_TERMINAL")
            if kind == "constructor_intent":
                if (constructor or reset or action_step or step is not None
                        or action is not None or digest is not None):
                    raise ValueError("F8R7_CONSTRUCTOR_ORDER")
                constructor = 1
            elif kind == "reset_intent":
                if (not constructor or reset or action_step or step is not None
                        or action is not None or digest is not None):
                    raise ValueError("F8R7_RESET_ORDER")
                reset = 1
            elif kind == "model_query_intent":
                if (phase != "test" or not reset or pending is not None
                        or type(step) is not int or step != action_step + 1
                        or type(action) is not int or action not in (0, 1)
                        or action in queried or digest is not None):
                    raise ValueError("F8R7_MODEL_QUERY_ORDER_OR_CAP")
                queried.add(action)
            elif kind == "action_intent":
                if (not reset or pending is not None or
                        type(step) is not int or step != action_step + 1
                        or step > HORIZON or type(action) is not int
                        or action not in (0, 1) or digest is not None):
                    raise ValueError("F8R7_ACTION_REPLAY_OR_ORDER")
                action_step = step
                pending = action
                queried.clear()
            elif kind == "action_observed":
                if (pending is None or type(step) is not int
                        or step != action_step or type(action) is not int
                        or action != pending or type(digest) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
                    raise ValueError("F8R7_UNWITNESSED_OR_WRONG_ACTION_RECEIPT")
                pending = None
            elif kind == "session_terminal":
                if (step is not None or action is not None or digest is not None
                        or row.get("outcome") not in ("completed", "aborted", "unknown")):
                    raise ValueError("F8R7_INVALID_TERMINAL")
                if row["outcome"] == "completed" and (
                        not reset or pending is not None or action_step == 0):
                    raise ValueError("F8R7_UNQUALIFIED_SESSION_COMPLETION")
                terminal = True
            else:
                raise ValueError("F8R7_UNREGISTERED_EVENT")

    def _can_claim_phase(self, phase: str) -> None:
        index = PHASES.index(phase)
        for prior in PHASES[:index]:
            expected = {x["identity"] for x in self.plan.values()
                        if x["phase"] == prior}
            actual = set()
            for p in self.claims.iterdir():
                if not p.name.endswith(".json"):
                    raise ValueError("F8R7_UNEXPECTED_CLAIM_FILE")
                claim, _ = _read_exact(p)
                if claim["phase"] == prior:
                    actual.add(claim["session_identity"])
            if actual != expected:
                raise PermissionError("F8R7_PREVIOUS_PHASE_NOT_QUALIFIED")
            for ident in actual:
                events = self._records(ident)
                if not events or events[-1]["kind"] != "session_terminal" or (
                        events[-1]["outcome"] != "completed"):
                    raise PermissionError("F8R7_PRIOR_PHASE_UNKNOWN_OR_ABORTED")

    def claim(self, identity: str) -> dict:
        row = self.plan.get(identity)
        if row is None:
            raise ValueError("F8R7_UNREGISTERED_SESSION")
        self._can_claim_phase(row["phase"])
        count = list(self.claims.glob("*.json"))
        if len(count) >= TOTAL_CLAIMS or (
                len(count) + 1) * HORIZON > TOTAL_ACTION_RESERVE:
            raise ValueError("F8R7_GLOBAL_BUDGET_EXHAUSTED")
        record = {
            "schema": SCHEMA, "classification": PROTOTYPE,
            "session_identity": identity, "source_seed": row["source_seed"],
            "source_sha": r6.locked()["original_source"]["sha"],
            "phase": row["phase"], "run_id": self.run_id, "attempt": 1,
            "reserved_effects": HORIZON, "status": "ATTEMPTED_NO_REPLAY",
        }
        _exclusive(self._claim_path(identity), record)
        # If event-directory setup fails, the claim stays consumed.
        _directory(self._events_path(identity))
        self._live.add(identity)
        return record

    def append(self, identity: str, kind: str, *, step: int | None = None,
               action: int | None = None, effect_sha256: str | None = None,
               outcome: str | None = None) -> dict:
        if identity not in self._live:
            raise PermissionError("F8R7_RESTART_OR_UNCLAIMED_SESSION_NO_REPLAY")
        if kind not in EVENT_TYPES:
            raise ValueError("F8R7_UNREGISTERED_EVENT")
        claim = self._read_claim(identity)
        if claim["run_id"] != self.run_id:
            raise PermissionError("F8R7_RUN_ID_MISMATCH")
        current = self._records(identity)
        prior = "GENESIS"
        if current:
            _, prior = _read_exact(self._events_path(identity) /
                                   f"{len(current):06d}.json")
        record = {
            "classification": PROTOTYPE, "session_identity": identity,
            "index": len(current)+1, "previous_sha256": prior,
            "kind": kind, "step": step, "action": action,
            "effect_sha256": effect_sha256,
        }
        if kind == "session_terminal":
            record["outcome"] = outcome
        elif outcome is not None:
            raise ValueError("F8R7_OUTCOME_ONLY_FOR_TERMINAL")
        self._validate_sequence(current + [record], phase=claim["phase"])
        # No effectful calls here; file contents only record fictional intents.
        _exclusive(self._events_path(identity) /
                   f"{len(current)+1:06d}.json", record)
        return record

    def recover_readonly(self, identity: str) -> dict:
        """Never grants a resumption token, even for an apparently clean file."""
        row = self._read_claim(identity)
        events = self._records(identity)
        count = sum(v["kind"] == "action_intent" for v in events)
        query_count = sum(v["kind"] == "model_query_intent" for v in events)
        return {
            "classification": PROTOTYPE,
            "session_identity": identity,
            "reserved_factual_effects": row["reserved_effects"],
            "recorded_attempted_actions": count,
            "recorded_hypothetical_queries": query_count,
            "unknown_or_aborted_consumes_reservation": True,
            "may_resume": False, "source_effects_executed": 0,
            "owner_admitted": False, "scientific_pass": False,
            "parent_directory_fsync_qualified": os.name != "nt",
        }


def refuse_real_source_firstuse(*_args, **_kwargs):
    raise PermissionError("F8R7_REAL_OWNER_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
