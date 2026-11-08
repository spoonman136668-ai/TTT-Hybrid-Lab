"""S29 source-locked MIT reference Gymnasium step adapter negative tests.

No real Gymnasium package import or environment first-use occurs in pytest.
"""
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from scripts import native_s29_mit_gymnasium_smoke as s29
from scripts.native_s28_causal_wind_offline import S28Blocked


class FakePublicGym:
    """Engineering fixture matching only required public CartPole API."""

    class Space:
        n = 2

    class Obs:
        shape = (4,)

    action_space = Space()
    observation_space = Obs()

    def __init__(self):
        self.i = 0
        self.closed = False

    def reset(self, *, seed):
        self.i = 0
        return [0., 0., .01, 0.], {}

    def step(self, action):
        self.i += 1
        obs = [float(self.i)*.01, .0, .01, float(action)*.005]
        return obs, 1., False, self.i == 3, {}

    def close(self):
        self.closed = True


class FakeGymModule:
    def __init__(self, factory):
        self.factory=factory
        self.created=[]

    def make(self, name, render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        env=self.factory()
        self.created.append(env)
        return env


def test_fixed_mit_source_and_non_science_firstuse():
    d=s29.frozen()
    assert d["upstream"]["exact_commit"]=="17eff00220d210beda933f78b0e52850022b0690"
    assert d["upstream"]["license"]=="MIT"
    assert d["upstream"]["git_blobs"]==s29.PINS
    assert d["smoke"]["public_engineering_smoke_seeds"]==[42,43]
    assert d["smoke"]["repeat_per_seed"]==2
    assert d["smoke"]["causal_gym_source_import_allowed"] is False
    assert d["qualification"]["no_environment_install_or_global_mutation"] is True


def test_true_factual_step_api_equivalence_with_fixture():
    module=FakeGymModule(FakePublicGym)
    first=s29._one_episode(module,42,32)
    second=s29._one_episode(module,42,32)
    assert first==second
    assert first["events"]==3
    assert first["records"][-1]["truncated"] is True
    assert first["records"][0]["executed_action"]==0
    assert first["records"][1]["executed_action"]==1
    assert len(module.created)==2
    assert all(env.closed for env in module.created)


def test_privileged_info_is_rejected_and_external_env_closed():
    class Leaker(FakePublicGym):
        def step(self,action):
            a=list(super().step(action))
            a[4]={"u_wind":0.2}
            return tuple(a)
    gym=FakeGymModule(Leaker)
    with pytest.raises(S28Blocked,match="ENV_INFO_CONTAINS_PRIVILEGED_FIELD"):
        s29._one_episode(gym,43,5)
    assert gym.created[0].closed


def test_wrong_source_and_science_admission_denied(tmp_path,monkeypatch):
    d=s29.frozen()
    with pytest.raises(s29.S29Blocked,match="GIT_FAILURE"):
        s29.source_identity(tmp_path,d)
    with pytest.raises(s29.S29Blocked,match="PROTOCOL_OR_AUTHORITY_DRIFT|FROZEN"):
        bad=deepcopy(d)
        bad["upstream"]["exact_commit"]="0"*40
        s29.frozen() if False else s29.require(
            bad["upstream"]["exact_commit"]==d["upstream"]["exact_commit"],
            "FROZEN_SOURCE")
    for flag in ("NATIVE_S29_ENGINEERING_ONLY","NATIVE_S28_ENGINEERING_ONLY"):
        monkeypatch.setenv(flag,"0")
        with pytest.raises(s29.S29Blocked,match="SCIENCE_FIRST_USE_DENIED"):
            s29.run(tmp_path)
        monkeypatch.delenv(flag)


def test_implementation_has_no_installer_or_external_wind_import():
    src=Path(s29.__file__).read_text("utf-8")
    assert "pip install" not in src
    assert "from causal_gym" not in src
    assert 'import_module("gymnasium")' in src
    assert list(inspect.signature(s29._one_episode).parameters)==[
        "gym","seed","max_steps"]


def test_r2_source_only_dependency_is_registered_and_no_installer():
    r2=s29.frozen_r2()
    assert r2["r1_run"]==42523092142
    assert r2["parent_r1_sha"]=="b9428a5824874996c11094a56c0d852df0b9115b"
    assert r2["provider"]=="cloudpipe/cloudpickle"
    assert r2["exact_commit"]=="ff54d93221082d4ce22630ecdb8fbbccc60f3b62"
    assert r2["git_blobs"]==s29.CLOUD_PINS
    assert r2["cloudpickle_or_gymnasium_installer_allowed"] is False
    assert r2["protected_scientific_first_use_authorized"] is False
    assert r2["real_causal_wind_environment_allowed"] is False
    assert r2["qualification"]["public_smoke_seeds_and_repeats"]==[42,43,2]


def test_r2_missing_cloud_source_rejected_before_any_runtime_import(tmp_path):
    r2=s29.frozen_r2()
    with pytest.raises(s29.S29Blocked,match="GIT_FAILURE"):
        s29.cloudpickle_source_identity(tmp_path,r2)
    monkey_env = s29.frozen()
    assert monkey_env["qualification"]["missing_dependency_means_runtime_blocked_not_auto_installed"] is True
