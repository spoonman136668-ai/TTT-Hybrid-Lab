"""P61-A1 historical-source negative controls (local shim, original GitHub uses parent P61)."""
import numpy as np
from experiments.p61_a1_credit_identifiability_dev import data,run,ridge,prior_bayes,label_after_action_reward

def test_source_exact_sha_for_all_historical_splits():
    for cohort in ('fit','val','unfamiliar_dev'):
        d,sha=data(cohort)
        assert len(sha)==64 and len(d['history'])>0

def test_fit_targets_are_only_realized_query_action_reward():
    fit,_=data('fit')
    y=label_after_action_reward(fit)
    original=y.copy()
    fit['evaluator_hidden_final_mode'][:]=99
    assert np.array_equal(original,label_after_action_reward(fit))

def test_noninvertible_erasure_does_not_use_historical_actions():
    fit,_=data('fit');ood,_=data('unfamiliar_dev')
    out,_=ridge(fit,ood,erase=True)
    ood['history'][:,:,0]*=-1
    assert np.array_equal(out,ridge(fit,ood,erase=True)[0])

def test_consumed_development_baselines_and_oracle_warning():
    d=run()
    x=d['results']
    assert x['recent3']['correct']==874
    assert x['all6']['correct']==617
    assert x['factual_ridge']['correct']==876
    assert x['noninvertibly_erased_action_identity']['correct']==500
    assert x['ood_prior_ORACLE_NOT_FAIR']['correct']==897
    assert d['oracle_prior_not_valid_learned_control']
    assert d['original_p61_frozen_gain']=='FAIL_0_OF_3_PRESERVED'

def test_determinism_and_no_new_science():
    first=run();second=run()
    assert first==second
    assert first['scientific_first_use']=='NOT_RUN'
    assert first['redacted_44_to_56_pct_chance_band']
