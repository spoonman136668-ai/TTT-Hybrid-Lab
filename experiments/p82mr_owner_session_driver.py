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


class _ToyArrayLike:
    def __init__(self,values):
        self.values=list(values)
    def tolist(self):
        return list(self.values)


class InMemoryToySession:
    """Simple deterministic fake, never a real causal simulator."""
    __slots__ = ("behavior", "limit", "arraylike", "reset_calls", "do_calls",
                 "close_calls", "reset_seed", "state", "actions", "policy_calls",
                 "closed")

    def __init__(self, behavior="normal", *, limit=3, arraylike=False):
        self.behavior=behavior
        self.limit=limit
        self.arraylike=arraylike
        self.reset_calls=0
        self.do_calls=0
        self.close_calls=0
        self.reset_seed=None
        self.state=[0.0,0.0,0.0,0.0]
        self.actions=[]
        self.policy_calls=0
        self.closed=False

    def reset(self,*,seed):
        self.reset_calls+=1
        self.reset_seed=seed
        if self.behavior=="reset_exception":
            raise RuntimeError("injected reset unavailable")
        if self.behavior=="bad_reset":
            return self.state
        return (_ToyArrayLike(self.state) if self.arraylike else list(self.state),
                {"hidden_wind":999.0})

    def do(self, policy):
        self.do_calls+=1
        if self.behavior=="uncertain_do":
            raise RuntimeError("injected uncertain effect")
        if self.behavior=="never_calls_policy":
            action=0
        else:
            visible=list(self.state)
            if self.behavior=="wrong_visible":
                visible[0]+=0.02
            action=policy(_ToyArrayLike(visible) if self.arraylike else visible)
            self.policy_calls+=1
            if self.behavior=="callback_twice":
                policy(visible)
                self.policy_calls+=1
        self.actions.append(action)
        if self.behavior=="wrong_info":
            recorded=1-action
        elif self.behavior=="no_info":
            recorded=None
        else:
            recorded=action
        next_state=[0.1*self.do_calls,
                    0.02*self.do_calls,
                    0.003*self.do_calls,
                    0.005*self.do_calls]
        self.state=list(next_state)
        ended=(self.do_calls==self.limit)
        info={"action":recorded,"secret_phase":"oracle","wind":555}
        if self.behavior=="bad_return":
            return (next_state,1.0,ended,False)
        if self.behavior=="bad_reward":
            reward=float("nan")
        else:
            reward=1.0
        return (_ToyArrayLike(next_state) if self.arraylike else next_state,
                reward,ended,False,info)

    def close(self):
        self.close_calls+=1
        self.closed=True


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

    # Type identity is checked without touching the user-supplied instance.
    # A caller-provided source_kind label cannot authorize arbitrary reset/do.
    if type(session) is not InMemoryToySession:
        raise PermissionError("P82MR_UNTRUSTED_SESSION_CLASS")
    if (not callable(getattr(session, "reset", None))
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
