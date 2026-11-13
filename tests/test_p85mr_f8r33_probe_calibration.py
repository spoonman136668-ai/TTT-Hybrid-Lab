"""R33 frozen FIT-only indicator, centering, and uncertainty falsifier."""
import inspect
import json
import math

import pytest
import torch

from experiments import p85mr_f8r33_probe_calibration as x

torch.set_num_threads(1)


def test_frozen_protocol_and_authority_negative():
    f=x.locked()
    assert f["parent"]["head"]=="8f0b42075784509fe0616e8c3a3577ac38e5947d"
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()
    with pytest.raises(ValueError,match="UNREGISTERED_FEATURE_METHOD"):
        x._features(torch.zeros(14,16),torch.zeros(14,16),
                    method="oracle",probe_offset=0.)


def test_factorial_FIT_disjoint_public_state_range_and_selected_only():
    fit=x.fit_cohort()
    assert len(fit)==3072
    assert all(set(row)=={"action","effect","physical_target","direct_target"}
               for row in fit[:4])
    assert fit[0]["action"].shape==(14,16)
    for i,state in enumerate(x.FIT_STATES):
        seen=x.r31._decode_key(fit[i*x.FIT_GROUP_SIZE]["action"][-1])[0]
        assert abs(seen-state)<.0001
    lo=x._episode(0,0,-1,1.,0,0)
    hi=x._episode(0,0,+1,1.,0,0)
    assert torch.equal(lo["effect"][:4],hi["effect"][:4])
    assert not torch.equal(lo["effect"][4],hi["effect"][4])
    assert torch.equal(lo["effect"][5:13],hi["effect"][5:13])
    assert x._B("fit",0)!=x._B("dev",0)
    assert x._B("ood",0)[:8]!=x._B("dev",0)
    with pytest.raises(ValueError,match="INVALID_B_GROUP"):
        x._B("fit",64)


def test_information_erasure_and_score_oracle_isolation():
    for panel in x.PANELS:
        ww=x.worlds(panel)
        assert len(ww)==48
        for g in range(x.GROUPS):
            for ctx in (0,1):
                neg,pos=ww[g*4+ctx*2:g*4+ctx*2+2]
                assert neg["b_history"]==pos["b_history"]
                assert neg["q_evaluator_only"]==-pos["q_evaluator_only"]
                assert x.evaluate_regret_only([neg]*48,[ctx]*48)[0] != (
                    x.evaluate_regret_only([pos]*48,[ctx]*48)[0])
        a,e=x.candidate_inputs(ww[0],"active_information")
        aa,ee=x.candidate_inputs(ww[0],"active_information",erase=True)
        assert a.shape[0]==2 and a.shape[-1]==16
        assert torch.equal(a,aa)
        assert torch.count_nonzero(ee[:,4])==0
        assert torch.count_nonzero(ee[:,-1])==0
        assert torch.equal(e[:,:4],ee[:,:4])
        assert torch.equal(e[:,5:],ee[:,5:])


def test_closed_form_fit_only_and_validation():
    fit=x.fit_cohort()
    m=x._fit_model(fit,48,"centered")
    indicator=x._fit_model(fit,48,"indicator")
    legacy=x._legacy(fit)
    assert m["factual_examples"]==2304
    assert len(m["weights"])==9
    assert len(indicator["weights"])==9 and len(legacy["weights"])==7
    assert abs(m["fit_only_probe_offset"])>.01
    assert m["normal_gradient_abs_max"]<1e-5
    assert math.isfinite(m["fit_validation_mse"])
    assert m["variance"]["uncertain"]>m["variance"]["informed"]
    with pytest.raises(ValueError,match="UNREGISTERED_FIT_CONFIGURATION"):
        x._fit_model(fit,64,"centered")


def test_all_frozen_dev_metrics_and_cold_restore():
    out=x.run()
    assert out["schema"]==x.SCHEMA
    assert out["factual_FIT_episodes"]==3072
    assert out["factual_train_episodes"]==2304
    assert out["factual_validation_episodes"]==768
    assert out["neural_optimizer_updates"]==0
    assert not out["independent_neural_convergence_qualified"]
    assert not out["capacity_FLOPs_matched"]
    assert not out["independent_scientific_pass"]
    assert not out["spent_F7_replay"] and out["original_external_effects"]==0
    assert out["deployment"]=="NONE" and out["cold_restore_actions_equal"]
    assert isinstance(out["judgment"]["source_neutral_development_pass"],bool)
    assert set(out["scores"])==set(x.MODELS)
    for model in x.MODELS:
        for panel in x.PANELS:
            p=out["scores"][model][panel]
            assert set(p["policies"])==set(x.POLICIES)
            for policy in x.POLICIES:
                q=p["policies"][policy]
                assert len(q["choices"])==48
                assert len(q["pergroup_regret"])==12
                assert math.isfinite(q["mean_regret"])
    x.retain_ci_receipt(out)


def test_fit_and_choice_sources_have_no_evaluator_hidden_sign():
    for fn in (x._episode,x._features,x._fit_safe_probe_offset,
               x._fit_model,x._legacy,x._estimate,x._choose):
        code=inspect.getsource(fn)
        assert "q_evaluator_only" not in code
        assert "evaluate_regret_only(" not in code
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym","workflow_dispatch(",
                      "subprocess.run(","requests.get(","urlopen("):
        assert forbidden not in src
