"""P57: frozen two-probe factual physics, identity and negative result controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import native_p57_two_probe_dev as p57
ROOT=Path(__file__).resolve().parents[1]
SHA={
 870101:'4fbeba79427f2cb0a24d53e8f2a007927d9f218f80ed0d163d018b372e2b7130',
 870103:'12126f89021985e901a00231674e837eced19197bf5f11074d0cd1181275c9a2',
 870201:'f677f056c4454ffd4ef80bf22e94bea7d0feaf9c920efe72b8d206a6f00a19fb',
 870203:'a78267749fd627e72feab4965ca78536b712101a0db50082a34b916f04af06e1',
 870301:'72a18b29dfb27fb9ab85d34cff41ee4231a1872ac39857c693252e7e6c48f338',
 870303:'bffa749a6272631c1601c6b3731529feac78c2c6634b66b7f437fc1f8eeed8dd',
 870401:'b74dea6673738624b3e8b50d6a090d6f85d7891d5795ccae0aaf7609289fc8ee',
}

def test_parent_original_source_unchanged_and_frozen_seeds():
    src=(ROOT/'scripts/native_p56_family_selection_dev.py').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(src)).encode()+b'\0'+src).hexdigest()=='851ddf45fbd6e9330786bcc0be34c92b898b29cb'
    assert set(p57.FIRST_USE_SET)==set(SHA)
    with pytest.raises(ValueError,match='P57_UNREGISTERED_SOURCE'):p57.source(860003)


def test_exact_source_hashes_and_no_unexecuted_measured_facts():
    for seed,expected in SHA.items():
        w=p57.source(seed)
        assert w['sha256']==expected
        log=w['log'];p57.check_log(log)
        assert set(log)=={'context','first','second','mask_A','mask_B','measurement_A','measurement_B','actual_final_action','observed_reward'}
        assert not any(x in log for x in ('theta','noise_A','noise_B','reward_noise'))
        assert np.all(log['measurement_A'][log['mask_A']==0]==0)
        assert np.all(log['measurement_B'][log['mask_B']==0]==0)
        assert np.all((log['first']==0)|(log['first']!=log['second']))
        c=log['context'];actual=log['actual_final_action']
        expected_reward=w['env']['theta']*actual+w['env']['reward_noise']-c[:,5]*log['mask_A']-c[:,8]*log['mask_B']
        np.testing.assert_allclose(log['observed_reward'],expected_reward,atol=2e-6,rtol=0)


def test_wrong_or_impossible_factual_log_rejected():
    log=p57.source(870101)['log']
    with pytest.raises(ValueError,match='P57_BAD_FACTUAL_LOG_SCHEMA'):
        p57.moments({**log,'theta':np.ones(len(log['first']))})
    bad={k:v.copy() for k,v in log.items()}
    ix=np.flatnonzero(bad['mask_A']==0)[0];bad['measurement_A'][ix]=123.
    with pytest.raises(ValueError,match='P57_UNEXECUTED_MEASUREMENT_LEAK'):p57.moments(bad)
    bad={k:v.copy() for k,v in log.items()}
    ix=np.flatnonzero(bad['first']==1)[0];bad['second'][ix]=1
    with pytest.raises(ValueError,match='P57_DUPLICATE_ACTION'):p57.moments(bad)


def test_disjoint_factual_training_and_family_selection():
    for fit_seed,ood_seed in p57.PAIRS:
        w=p57.source(fit_seed);cut=p57.FIT_N-p57.SELECT_N
        tr={k:v[:cut] for k,v in w['log'].items()}
        held={k:v[cut:] for k,v in w['log'].items()}
        model=p57.select_family(p57.moments(tr),held)
        assert model['family']==dict(zip(('cue','A','B','reward'),w['laws']))
        assert model['stats']['training_worlds']==24576
        assert model['selection_cases']==4096
        assert model['stats']['executed_training_A']>10000
        assert model['stats']['executed_training_B']>10000
        assert model['selection_acts_A']>2000
        assert model['selection_acts_B']>2000


def test_only_selected_physical_probes_and_no_reused_action_slots():
    model=p57.run()['rows']['870103']['fitted']
    w=p57.source(870103);calls=[]
    def actuator(label,mask):
        calls.append((label,mask.copy()))
        return (w['env']['theta'][mask]*(w['env']['context'][mask,3] if label=='A' else w['env']['context'][mask,6])+
                (w['env']['context'][mask,4] if label=='A' else w['env']['context'][mask,7])*
                (w['env']['noise_A'][mask] if label=='A' else w['env']['noise_B'][mask]))
    observed=p57.action_policy(w['env']['context'],model,'selected',actuator)
    assert len(calls)<=4
    assert np.all((observed['first']==0)|(observed['first']!=observed['second']))
    assert ((observed['second']!=0).mean()>.01)
    assert all(label in ('A','B') and mask.any() for label,mask in calls)
    assert np.all(observed['measurement_A'][~observed['mask_A']]==0)
    assert np.all(observed['measurement_B'][~observed['mask_B']]==0)


def test_outcome_contingent_second_decision_and_zero_information_sham():
    w=p57.source(870303)
    fit=p57.source(870301);cut=p57.FIT_N-p57.SELECT_N
    mdl=p57.select_family(p57.moments({k:v[:cut] for k,v in fit['log'].items()}),
                          {k:v[cut:] for k,v in fit['log'].items()})
    context=w['env']['context']
    def actual(label,mask):
        col=3 if label=='A' else 6
        return w['env']['theta'][mask]*context[mask,col]+context[mask,col+1]*w['env']['noise_'+label][mask]
    orig=p57.action_policy(context,mdl,'selected',actual)
    def altered(label,mask):
        # Engineering sensitivity control only. NOT a scientific alternative-outcome observation.
        x=actual(label,mask)
        return x+2.*np.where(x>=0,1.,-1.)
    changed=p57.action_policy(context,mdl,'selected',altered)
    assert np.array_equal(orig['first'],changed['first'])
    assert np.any(orig['second']!=changed['second']) or np.any(orig['action']!=changed['action'])
    sham=p57.source(870401)
    sh=p57.executed(sham['env'],mdl,'selected')
    assert sh['probes_A']==sh['probes_B']==0


def test_frozen_result_honestly_fails_and_is_cold_exact():
    out=p57.run()
    assert not out['all_frozen_pass']
    assert out['source_hashes']=={str(k):v for k,v in SHA.items()}
    assert out['sham']['probe_count_A']==out['sham']['probe_count_B']==0
    assert [k for k,v in out['rows'].items() if v['gates']['better_strong_fixed_0point005']]==['870103']
    for row in out['rows'].values():
        assert row['gates']['correct_all_source_families']
        assert row['gates']['cold_exact']
        assert row['gates']['better_never_0point03']
        assert row['gates']['paired_lower_positive']
        assert row['gates']['outcome_contingent_second_probe_at_least_1pct']


def test_persisted_receipt_preserves_prior_frozen_failure():
    p=ROOT/'experiments/native_p57_two_probe_physical_dev_receipt.json'
    if not p.exists():pytest.skip('receipt not committed during authoring')
    r=json.loads(p.read_text())
    assert r['frozen_all_pass'] is False
    assert r['external_scientific_first_use']=='NOT_RUN'
    assert r['original_full_regression']=='NOT_RUN'
