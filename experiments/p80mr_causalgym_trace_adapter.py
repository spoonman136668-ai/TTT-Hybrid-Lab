"""Pure P80-MR adapter for previously captured *factual* CartPoleWind do() steps.

NO upstream import, environment step, model inference, training or source scoring.
This is a loss-visible deterministic conversion, not proof of independent source
provenance or of matching the original 16D action-only Hybrid research task.

Parent: P79-MR trace schema. Freeze: p80mr_causalgym_bridge_freeze.json.
"""
import math

from p79mr_trace_admission import (
    SCHEMA as TRACE_SCHEMA, canonical_sha256, validate_trace,
)

CAPTURE_SCHEMA = "ttt-hybrid.p80mr.cartpole-do-capture.v1"
UPSTREAM_REPO = "CausalAILab/Causal-Gymnasium"
UPSTREAM_SHA = "c94e9a4cd3641588b6b8507e71ccee5c27032bdf"
UPSTREAM_PATH = "causal_gym/envs/cartpole_wind.py"
UPSTREAM_BLOB = "d65a6a31941709dc9201259d80dd4ebbcf48b4fa"
STATE_SCALES = (2.4, 3.0, 0.2095, 3.5)
WORLD_KEYS = frozenset(("world_id", "episode_seed", "transitions", "probe_event_ids"))
STEP_KEYS = frozenset(("event_id", "state_before", "executed_action",
                       "state_after", "reward", "terminated", "truncated",
                       "mode", "receipt_ref"))
CAPTURE_KEYS = frozenset(("schema", "upstream_repo", "upstream_commit",
                          "upstream_path", "upstream_blob", "source_id",
                          "producer_id", "custody_ref", "worlds"))


def _exact_fields(value, fields, code):
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(code)


def _finite_real(value, code):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(code)
    return float(value)


def _state(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("P80MR_BAD_STATE")
    return tuple(_finite_real(x, "P80MR_BAD_STATE") for x in value)


def _name(s):
    if not isinstance(s, str) or not (1 <= len(s) <= 128):
        raise ValueError("P80MR_BAD_RECEIPT_OR_ID")
    # P79 also checks its stricter identifier grammar.
    return s


def _action_key(state, executed_action):
    # State-conditioned factual action context, NOT action-only features.
    raw = [1.0] + [math.tanh(x/scale) for x, scale in zip(state, STATE_SCALES)]
    raw += [float(executed_action == 0), float(executed_action == 1)]
    raw += [0.0] * 9
    denom = math.sqrt(math.fsum(x*x for x in raw))
    return [v/denom for v in raw]


def _observed_effect(before, after, reward, terminated, truncated):
    differences = [y-x for x, y in zip(before, after)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("P80MR_BAD_EFFECT")
    delta = [math.tanh(d/scale) for d, scale in zip(differences, STATE_SCALES)]
    delta += [math.tanh(reward), float(terminated), float(truncated)]
    delta += [0.0] * 9
    if len(delta) != 16 or not all(math.isfinite(v) for v in delta):
        raise ValueError("P80MR_BAD_EFFECT")
    return delta


def adapt_capture(capture):
    """Convert received OFFLINE records into P79 trace + UNVERIFIED provenance.

    No public causal software is run, installed or imported. The returned
    externally labeled manifest is deliberately rejected by P79 toy_replay.
    External forward inference requires a separate project-owned admission.
    """
    _exact_fields(capture, CAPTURE_KEYS, "P80MR_BAD_CAPTURE")
    expected = (UPSTREAM_REPO, UPSTREAM_SHA, UPSTREAM_PATH, UPSTREAM_BLOB)
    actual = tuple(capture[x] for x in ("upstream_repo", "upstream_commit",
                                       "upstream_path", "upstream_blob"))
    if actual != expected or capture["schema"] != CAPTURE_SCHEMA:
        raise ValueError("P80MR_UNPINNED_UPSTREAM")
    for key in ("source_id", "producer_id", "custody_ref"):
        _name(capture[key])
    raw_worlds = capture["worlds"]
    if not isinstance(raw_worlds, list) or not (1 <= len(raw_worlds) <= 256):
        raise ValueError("P80MR_BAD_WORLDS")

    worlds = []
    for raw in raw_worlds:
        _exact_fields(raw, WORLD_KEYS, "P80MR_BAD_WORLD")
        world_id = _name(raw["world_id"])
        seed = raw["episode_seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("P80MR_BAD_SEED")
        steps = raw["transitions"]
        if not isinstance(steps, list) or not (2 <= len(steps) <= 4096):
            raise ValueError("P80MR_BAD_TRANSITIONS")
        probe_ids = raw["probe_event_ids"]
        if (not isinstance(probe_ids, list) or not (1 <= len(probe_ids) <= 256)
                or any(type(x) is not int or x < 2 or x > len(steps)
                       for x in probe_ids) or len(set(probe_ids)) != len(probe_ids)):
            raise ValueError("P80MR_BAD_PROBE_IDS")
        result_events = []
        previous_after = None
        for index, step in enumerate(steps, 1):
            _exact_fields(step, STEP_KEYS, "P80MR_BAD_TRANSITION")
            if type(step["event_id"]) is not int or step["event_id"] != index:
                raise ValueError("P80MR_EVENT_SEQUENCE")
            if step["mode"] != "do":
                raise ValueError("P80MR_INTERVENTION_NOT_EXECUTED")
            act = step["executed_action"]
            if type(act) is not int or act not in (0, 1):
                raise ValueError("P80MR_BAD_EXECUTED_ACTION")
            pre, post = _state(step["state_before"]), _state(step["state_after"])
            if (previous_after is not None and
                    any(abs(x-y) > 1e-5 for x,y in zip(previous_after, pre))):
                raise ValueError("P80MR_BROKEN_TRAJECTORY")
            previous_after = post
            reward = _finite_real(step["reward"], "P80MR_BAD_REWARD")
            if type(step["terminated"]) is not bool or type(step["truncated"]) is not bool:
                raise ValueError("P80MR_BAD_TERMINATION")
            if (step["terminated"] or step["truncated"]) and index < len(steps):
                raise ValueError("P80MR_EVENTS_AFTER_TERMINATION")
            receipt_ref = _name(step["receipt_ref"])
            result_events.append({
                "event_id": index,
                "action": _action_key(pre, act),
                "effect": _observed_effect(pre, post, reward,
                                           step["terminated"], step["truncated"]),
                "receipt_ref": receipt_ref,
            })
        probes = []
        for event_id in probe_ids:
            source_event = result_events[event_id-1]
            # Read historical events [0:event_id-1] only. Target is the actual
            # realized event at index event_id-1, withheld until readout.
            probes.append({
                "probe_id": f"actual-event-{event_id}",
                "after_events": event_id-1,
                "action": list(source_event["action"]),
                "realized_effect": list(source_event["effect"]),
                "group": "unknown",
                "receipt_ref": source_event["receipt_ref"],
            })
        worlds.append({"world_id": world_id, "events": result_events,
                       "probes": probes})
    manifest = {
        "schema": TRACE_SCHEMA,
        "source_kind": "external_simulator",
        "source_id": capture["source_id"],
        "producer_id": capture["producer_id"],
        "custody_ref": capture["custody_ref"],
        "expected_sha256": canonical_sha256(worlds),
    }
    validation = validate_trace(manifest, worlds)
    return {
        "manifest": manifest,
        "worlds": worlds,
        "custody": {
            "upstream_repo": UPSTREAM_REPO,
            "upstream_commit": UPSTREAM_SHA,
            "upstream_source_blob": UPSTREAM_BLOB,
            "capture_sha256": canonical_sha256(capture),
            "adapted_worlds_sha256": validation["worlds_sha256"],
            "factual_event_count": sum(len(w["events"]) for w in worlds),
            "read_only_probes": sum(len(w["probes"]) for w in worlds),
            "upstream_execution": "NOT_PERFORMED_BY_ADAPTER",
            "independent_origin": "UNVERIFIED",
            "project_owner_first_use": "NOT_AUTHORIZED",
            "scientific_pass": False,
        },
    }
