"""R46 prospective FIT-only calibration, observed B cue and model negative controls."""
import pytest
import torch

from experiments import p85mr_f8r46_probe_centered_learning as r46


def test_frozen_independent_source_and_external_admission():
    f = r46.locked()
    assert f["lineage"]["parent_sha"] == r46.PARENT
    assert f["source"]["salts"] == r46.SALT
    assert f["source"]["train_pairs"] == r46.TRAIN_PAIRS
    assert f["source"]["heldout_pairs"] == r46.HELDOUT_PAIRS
    assert f["preregistered_score"]["independent_scientific_pass"] is False
    assert f["authority"]["original_external_first_use"] is False
    assert len(r46.ARMS) == 8
    with pytest.raises(PermissionError, match="FIRST_USE_UNAUTHORIZED"):
        r46.refuse_external_first_use("original")
    with pytest.raises(ValueError, match="UNKNOWN_PANEL"):
        r46.worlds("external_original")
    with pytest.raises(ValueError, match="UNREGISTERED_POSITIONS"):
        r46.world("cal", 12, 0, 1, 1)


def test_full_paired_mean_parity_and_tagged_causal_observations():
    for panel in ("fit", "cal", "ood"):
        p = r46.balanced_mean_check(panel)
        assert p["groups"] == (r46.FIT_GROUPS if panel == "fit"
                               else r46.EVAL_GROUPS)
        assert p["max_mean_delta"] <= 1e-7
        for group in range(p["groups"]):
            w1=r46.world(panel,group,0,1,1)
            w2=r46.world(panel,group,0,1,-1)
            a1,e1=r46.candidates(w1)
            a2,e2=r46.candidates(w2)
            assert a1.shape==e1.shape==((2,22,16) if panel=="ood"
                                        else (2,14,16))
            assert torch.equal(a1,a2)
            assert torch.allclose(e1[:,5:-1].mean(dim=1),
                e2[:,5:-1].mean(dim=1),atol=1e-7,rtol=0)
            assert not torch.equal(e1[:,5:-1],e2[:,5:-1])
            assert torch.count_nonzero(e1[:,5:-1,4])==0
            assert torch.count_nonzero(e1[:,-1])==0
            assert torch.equal(e1[:,:5],e2[:,:5])
            erased=r46.candidates(w1,erase_B=True)[1]
            assert torch.count_nonzero(erased[:,5:-1])==0
            assert torch.equal(erased[:,:5],e1[:,:5])
            tagless=r46.candidates(w1,erase_tag=True)[0]
            assert not torch.equal(tagless,a1)


def test_selected_effect_reconstruction_matches_final_state_goal():
    facts,labels=r46.factual_fit()
    assert len(facts)==512
    assert len(labels)==256
    assert labels.count(0)==labels.count(1)==128
    for panel in ("fit","cal","ood"):
        for g in (0,1):
            for ctx in (0,1):
                for q in (-1,1):
                    for gate in (-1,1):
                        w=r46.world(panel,g,ctx,q,gate)
                        costs=[]
                        for action in (0,1):
                            following=r46._following(w,action)
                            k,e=r46.r33.b._record(
                                w["return_state"],action,following,ctx)
                            observed=w["return_state"]+r46.r31._physical_delta(e[0])
                            assert observed==pytest.approx(following,abs=1e-6)
                            costs.append((observed-w["goal"])**2)
                        for action in (0,1):
                            assert r46._regret(w,action)==pytest.approx(
                                max(0.,costs[action]-min(costs)),abs=1e-6)
    with pytest.raises(ValueError,match="BAD_RETURN_ACTION"):
        r46._regret(r46.world("cal",0,0,1,1),-1)


def test_FIT_only_calibration_recovers_observed_conditional_signal():
    facts,labels=r46.factual_fit()
    offset=r46.cal_offset(facts)
    assert offset==pytest.approx(
        r46.r33.r32.after(r46.PROBE,1,0,0.),abs=1e-6)
    # Raw selected fifth physical delta is negative for both q signs.
    # It must never be interpreted as the sign of q.
    for q in (-1,1):
        w=r46.world("cal",0,0,q,1)
        a,e=r46.candidates(w)
        raw=r46.r31._physical_delta(e[0,4,0])
        assert raw < 0.
        calibrated=r46._public_sensor(a[0],e[0],offset)
        assert (calibrated["q_centered"] > 0)==(q>0)
        assert (calibrated["cue_selected_delta"] > 0)
    certified=r46.observation_ceiling(offset)
    assert all(v["mistakes"]==0 and v["mean_regret"]==0
               for v in certified.values())
    assert certified["fit"]["worlds"]==256
    assert certified["cal"]["worlds"]==96
    assert certified["ood"]["worlds"]==96
    with pytest.raises(ValueError,match="INVALID_CALIBRATION_SOURCE"):
        r46.cal_offset(facts[:5])


def test_trained_two_seed_mechanisms_full_frozen_result_and_no_false_promotion():
    scored=r46.run()
    assert scored["schema"]==r46.SCHEMA
    assert scored["freeze_blob"]==r46.FREEZE_BLOB
    assert scored["parent"]==r46.PARENT
    assert scored["factual_selected_returns"]==512
    assert scored["fit_train_pairs"]==192
    assert scored["fit_heldout_pairs"]==64
    assert len(scored["models"])==8
    assert scored["binder_replay_updates"]==1024
    assert scored["new_neural_updates"]==1536
    assert scored["source_neutral_dev_only"]
    assert not scored["independent_scientific_pass"]
    assert not scored["external_first_use"]
    assert scored["deployment"]=="NONE"
    assert all(v["mistakes"]==0 for v in
               scored["observational_reference"].values())
    assert all(v["max_mean_delta"]<=1e-7 for v in
               scored["B_paired_mean_parity"].values())
    assert len(scored["judgment"]["seeds"])==2
    assert not scored["judgment"]["whole_process_compute_matched"]
    assert not scored["judgment"]["independent_scientific_pass"]
    for row in scored["models"]:
        assert row["updates"]==r46.STEPS
        assert row["trainable_parameters"]>0
        assert row["cold_restore"]
        assert len(row["checkpoints"])==3
        assert all(0.<=c["heldout_accuracy"]<=1.
                   for c in row["checkpoints"])
        assert set(row["panels"])=={"cal","ood"}
        assert len(row["panels"]["cal"]["active"]["choices"])==96
        assert len(row["panels"]["ood"]["active"]["choices"])==96
        for ablation in ("erase_B","erase_tag","erase_fifth",
                         "swap_B","uniform","zero_native",
                         "zero_reader","zero_binder"):
            assert len(row["panels"]["ood"][ablation]["choices"])==96
    r46.retain_ci_receipt(scored)
