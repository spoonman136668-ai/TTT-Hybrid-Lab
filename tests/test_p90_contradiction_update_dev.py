"""P90 frozen causal fast-state development and negative qualification."""
from pathlib import Path
import pytest
import torch
from experiments import p90_contradiction_update_dev as p90

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_original_source_custody():
    assert all(p90.p86.blob(ROOT/p)==sha for p,sha in p90.SOURCE_BLOBS.items())


def test_pre_data_freeze_and_new_seeds():
    assert p90.FREEZE=='a04c67295d43a886b63c0113ece137cdf7d65aa6'
    assert p90.SEEDS=={'FIT':900111,'VAL':900113}
    assert p90.MODEL_SEEDS==(900201,900203,900207)
    assert (p90.STEPS,p90.BATCH,p90.CHECK)==(80,16,10)


def test_protected_ood_and_negative_source_refusal():
    for key in ('OOD','NEGATIVE','P89_VAL'):
        with pytest.raises(ValueError,match='P90_PROTECTED_OR_UNKNOWN_SOURCE'):
            p90.source(key)
    for n in (0,7,9,129,257):
        with pytest.raises(ValueError,match='P90_BAD_WORLD_COUNT'):
            p90.source('VAL',n)


def test_exact_cross_platform_fit_val_source_sha():
    assert p90.world_sha(p90.source('FIT'))=='d96cabb61a25a9e835a267babb1615de3da744e7fe204513389ddc516296feaa'
    assert p90.world_sha(p90.source('VAL'))=='3e86ba93ef101ca151f3ef57ac80dd5e38267c67fc030ea01a7bf2e839e786cf'


def test_factual_source_never_leaks_future_effect_and_has_equal_regimes():
    for split in ('FIT','VAL'):
        a=p90.source(split,8)
        assert a['actions'].shape==(8,49,16)
        assert a['effects'].shape==(8,49,16)
        assert p90.p86.audit(p90.p86_world(a))['pass']
        assert int(a['changed_world'].sum())==4
        assert p90.world_sha(a)==p90.world_sha(p90.source(split,8))
        assert not torch.count_nonzero(p90.p86_world(a)['query'][:,16:])


def test_256_state_exact_original_p86_online_recurrence():
    a=p90.source('VAL',8)
    y,W=p90.rollout(a,'fixed')
    ref=p90.p86.online(p90.p86_world(a),1.,.99)[0]
    assert W.shape==(8,16,16)
    torch.testing.assert_close(y[:,-1],ref,atol=1e-6,rtol=1e-6)


def test_initial_conditioned_gates_equal_trained_constant_rule():
    a=p90.source('VAL',8)
    for seed in p90.MODEL_SEEDS:
        c=p90.initial(seed,'constant')
        for arm in ('observation','contradiction'):
            cond=p90.initial(seed,arm)
            torch.testing.assert_close(c,cond[:2],atol=0,rtol=0)
            assert torch.equal(cond[2:],torch.zeros(2))
            with torch.no_grad():
                m,_=p90.rollout(a,'constant',c)
                n,_=p90.rollout(a,arm,cond)
            torch.testing.assert_close(m,n,rtol=0,atol=0)


def test_actual_factual_state_cold_restart_bitwise():
    a=p90.source('VAL',8)
    for arm in ('fixed','constant','observation','contradiction'):
        params=None if arm=='fixed' else p90.initial(900201,arm)
        with torch.no_grad():
            _,whole=p90.rollout(a,arm,params,0,49)
            _,half=p90.rollout(a,arm,params,0,24)
            _,resumed=p90.rollout(a,arm,params,24,49,state=half.clone())
        assert torch.equal(whole,resumed)
        assert resumed.numel()==8*256


def test_future_unseen_final_outcome_never_changes_prediction():
    a=p90.source('VAL',8)
    b={k:v.clone() for k,v in a.items()}
    b['effects'][:,-1]*=-1
    for arm in ('fixed','contradiction'):
        theta=None if arm=='fixed' else p90.initial(900201,arm)
        with torch.no_grad():
            y,_=p90.rollout(a,arm,theta)
            z,_=p90.rollout(b,arm,theta)
        assert torch.equal(y,z)


def test_all_models_finite_qualified_baseline_and_frozen_negative():
    r=p90.run()
    assert r['status']=='DEV_FAIL_CONTRADICTION_VS_STRONG_CONTROLS'
    assert not r['scientific_pass'] and not r['OOD_generated'] and not r['negative_generated']
    assert not r['original_hybrid_trained'] and not r['whole_process_compute_matched']
    assert r['W256_matches_P86'] and r['cold_restore_original_P78']
    assert r['controls']['wrong_credit_pass']
    assert r['frozen_gates']['all_conventional_qualified']
    assert not r['frozen_gates']['all_contradiction_qualified']
    assert len(r['rows'])==3
    for row in r['rows']:
        assert row['conventional_qualified'] and not row['contradiction_qualified']
        assert row['models']['contradiction']['validation']['final'] < row['models']['constant']['validation']['final']
        assert all(v['num_updates']==80 and len(v['trace'])==8 for v in row['models'].values())
        assert row['models']['constant']['parameter_count']==2
        assert row['models']['contradiction']['parameter_count']==row['models']['observation']['parameter_count']==4
