"""P82-MR isolated, single-episode toy reset/do adapter.

No Causal-Gymnasium imports, real environment creation, scientific first-use,
model evaluation, training, CLI, scheduler, disk or network operations.
This module only drives caller-injected fake session tests. A real-source
owner adapter requires a distinct project-authorized admission route and SHA.

Frozen before implementation: p82mr_owner_session_freeze.json.
"""
from p81mr_collection_contract import (
    ALLOWED_TOY_SPLITS, MAX_STEPS, ToyCaptureRecorder,
    _id, _obs, frozen_episode_seed,
)


def _obs_as_list(observation):
    """Require 4 finite realized channels; allow numpy-array-like tolist()."""
    raw = observation
    if not isinstance(raw, (list, tuple)) and callable(getattr(raw, "tolist", None)):
        raw = raw.tolist()
    return _obs(raw)


def _five(result):
    if not isinstance(result, (tuple, list)) or len(result) != 5:
        raise ValueError("P82MR_BAD_DO_RESULT")
    return result


def _start_result(result):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("P82MR_BAD_RESET_RESULT")
    observation, extra_info = result
    if not isinstance(extra_info, dict):
        raise ValueError("P82MR_BAD_RESET_INFO")
    return _obs_as_list(observation)


def drive_toy_session(*, source_kind, split, episode_index, source_id,
                      producer_id, custody_ref, world_id, session):
    """Inject a fake toy session, with real reset/do ordering but no real source.

    The source_kind string is NOT external authority or provenance proof.
    This unregistered method is toy-only and must not be used as a backdoor
    by passing an actual simulator under a fabricated toy label.

    It issues at most one do(policy) per step, checks the reported actual
    action and protects against ambiguous results without any replay.
    """
    # Every registration check must happen before any session attribute access.
    if source_kind != "toy_engineering":
        raise PermissionError("P82MR_OWNER_EXTERNAL_FIRST_USE_REQUIRED")
    if split not in ALLOWED_TOY_SPLITS:
        raise PermissionError("P82MR_SPLIT_FIRST_USE_REQUIRED")
    seed = frozen_episode_seed(split, episode_index)
    for name in (source_id, producer_id, custody_ref, world_id):
        _id(name)

    if (session is None or not callable(getattr(session, "reset", None))
            or not callable(getattr(session, "do", None))):
        raise ValueError("P82MR_BAD_FAKE_SESSION")

    started = False
    try:
        # Reset is potentially effectful. Never rerun it following an error.
        started = True
        initial_state = _start_result(session.reset(seed=seed))
        ledger = ToyCaptureRecorder(
            source_kind="toy_engineering", split=split,
            episode_index=episode_index, initial_observation=initial_state,
            source_id=source_id, producer_id=producer_id,
            custody_ref=custody_ref, world_id=world_id)
        before = list(initial_state)
        for _ in range(MAX_STEPS):
            intent = ledger.prepare_action()
            intended_action = intent["intended_action"]
            callbacks = [0]

            def policy(current_observation):
                callbacks[0] += 1
                if callbacks[0] != 1:
                    raise RuntimeError("P82MR_POLICY_INVOKED_MORE_THAN_ONCE")
                visible = _obs_as_list(current_observation)
                if any(abs(x-y) > 1e-5 for x,y in zip(visible, before)):
                    raise ValueError("P82MR_PRE_ACTION_OBSERVATION_DRIFT")
                return intended_action

            # Outcome could have happened even if an exception is raised.
            # Never replay an attempted do on this session.
            try:
                next_obs, reward, terminated, truncated, info = _five(
                    session.do(policy))
            except Exception as exc:
                raise RuntimeError("P82MR_DO_OUTCOME_UNCERTAIN_NO_REPLAY") from exc
            if callbacks[0] != 1:
                raise RuntimeError("P82MR_POLICY_CALLBACK_NOT_ONCE")

            observed_next = _obs_as_list(next_obs)
            ledger.record_observed_step(
                info=info, next_observation=observed_next, reward=reward,
                terminated=terminated, truncated=truncated,
                receipt_ref=f"toy:step-{intent['event_id']}")
            before = list(observed_next)
            if terminated or truncated:
                break

        outcome = ledger.finalize()
        assert outcome["classification"] == "TOY_ENGINEERING_ONLY"
        assert outcome["science_pass"] is False
        return {
            "status": "TOY_SESSION_ENGINEERING_ONLY",
            "episode_seed": seed,
            "observed_events": ledger.event_count,
            "capture": outcome["capture"],
            "captured_digest": outcome["schema_only_sha256"],
            "independent_source_attested": False,
            "external_first_use": False,
            "scientific_pass": False,
        }
    finally:
        if started:
            close = getattr(session, "close", None)
            if callable(close):
                # Attempt cleanup once. No retries on ambiguous effects.
                close()
