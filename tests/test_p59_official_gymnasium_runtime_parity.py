"""P59 CPU source custody, fail-closed and optional installed runtime checks."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts import p59_official_gymnasium_runtime_parity as p59


def test_source_missing_and_mismatch_fail_without_fallback(tmp_path):
    with pytest.raises(p59.SourceQualificationFailure, match='UNREADABLE'):
        p59.require_source_file(tmp_path/'no-source.py')
    wrong=tmp_path/'continuous_mountain_car.py'
    wrong.write_text('class Continuous_MountainCarEnv: pass\n')
    with pytest.raises(p59.SourceQualificationFailure, match='MISMATCH'):
        p59.require_source_file(wrong)


def test_wrong_installed_class_and_env_id_fail_before_reset():
    class Counterfeit:
        pass
    class Env:
        unwrapped=Counterfeit()
        spec=SimpleNamespace(id=p59.ENV_ID,max_episode_steps=p59.REGISTERED_TIME_LIMIT)
        def __init__(self): self.closed=False
        def close(self): self.closed=True
        def reset(self,*args,**kwargs): raise AssertionError('SHOULD_NOT_RESET')
    env=Env()
    with pytest.raises(p59.SourceQualificationFailure,match='CLASS'):
        p59.make_verified_env(SimpleNamespace(make=lambda *a,**k:env))
    assert env.closed


def test_registered_environment_and_frozen_reused_seed_contract():
    assert p59.ENV_ID=='MountainCarContinuous-v0'
    assert p59.UPSTREAM_GIT_BLOB=='96dbb0d31a17d144468562f4bc68fce902e2a69f'
    assert p59.UPSTREAM_COMMIT=='e6be96abfb0a9df36da5e7884522c4f1f2432e53'
    assert p59.P58_REUSED_SOURCE_SEEDS==tuple(range(985100,985116))
    assert p59.POLICIES==('neutral','momentum_bangbang')
    assert p59.REGISTERED_TIME_LIMIT==999
    assert p59.ENGINEERING_ATOL==5e-7


def test_unavailable_package_writes_nonpass_receipt(tmp_path,monkeypatch):
    def absent(): raise p59.OfficialPackageUnavailable('NO_INSTALLED_GYM')
    monkeypatch.setattr(p59,'import_installed_gymnasium',absent)
    path=tmp_path/'result.json'
    assert p59.main(['--receipt',str(path)])==2
    data=json.loads(path.read_text())
    assert data['status']=='UNAVAILABLE' and data['reason']=='NO_INSTALLED_GYM'
    assert data['scientific_learning_benefit']=='NOT_TESTED'


def test_wrong_pinned_source_writes_failure_not_pass(tmp_path,monkeypatch):
    def drift():raise p59.SourceQualificationFailure('WRONG_GIT_BLOB')
    monkeypatch.setattr(p59,'qualify',drift)
    path=tmp_path/'drift.json'
    assert p59.main(['--receipt',str(path)])==3
    assert json.loads(path.read_text())['status']=='FAIL'


def test_reused_source_actions_from_completed_observations_only():
    # Source-free policy control: no caller-provided next observation or hidden mode.
    import numpy as np
    for velocity in (-.05,0.,0.01):
        sensor=np.array([-.5,velocity],np.float32)
        assert p59.action_for('neutral',sensor,None)==0.0
        assert p59.action_for('momentum_bangbang',sensor,None) in (-1.0,1.0)


def test_official_installed_original_package_parity_if_source_qualified():
    try:
        p59.import_installed_gymnasium()
    except p59.OfficialPackageUnavailable:
        pytest.skip('Pinned official Gymnasium package not installed; no runtime parity claim')
    # A present-but-wrong package is an actual qualification FAILURE, not skip.
    result=p59.qualify()
    assert result['status']=='PASS'
    assert result['pinned_file_git_blob']==p59.UPSTREAM_GIT_BLOB
    assert result['official_wrapper']['final_truncated']
    assert result['official_wrapper']['actual_steps']==999
    assert result['total_completed_steps']>0
