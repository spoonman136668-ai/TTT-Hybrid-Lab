"""F8-R47 source-neutral learned public cue-key selector falsifier."""
import pytest
import torch

from experiments import p85mr_f8r47_cue_key_contrast as r47


def test_freeze_and_external_firstuse_denial():
    f = r47.locked()
    assert f["parent"]["sha"] == r47.PARENT
    assert f["source"]["salts"] == r47.SALT
    assert f["source"]["fit_physically_selected_returns"] == 384
    assert f["frozen_dev_score"]["independent_scientific_pass"] is False
    assert len(r47.ARMS) == 8
    with pytest.raises(PermissionError, match="ORIGINAL_FIRSTUSE_DENIED"):
        r47.deny_original_first_use("external")
    with pytest.raises(ValueError, match="UNREGISTERED_PANEL"):
        r47.worlds("external")
    with pytest.raises(ValueError, match="UNREGISTERED_SOURCE"):
        r47.world("cal", r47.EVAL_GROUPS, 0, 1, 1)


def test_selected_observations_are_distinguishable_but_mean_uninformative():
    for panel in ("fit","cal","ood"):
        cert=r47.parity(panel)
        assert cert["groups"]==(r47.FIT_GROUPS if panel=="fit" else r47.EVAL_GROUPS)
        assert cert["max_full_B_mean_diff"]<=1e-7
        for group in range(cert["groups"]):
            cue, decoy = r47.positions(panel,group)
            assert cue != decoy
            w1=r47.world(panel,group,0,1,1)
            w2=r47.world(panel,group,0,1,-1)
            a1,e1=r47.candidates(w1)
            a2,e2=r47.candidates(w2)
            assert a1.shape==e1.shape==((2,22,16) if panel=="ood" else (2,14,16))
            assert torch.equal(a1,a2)
            assert torch.allclose(e1[:,5:-1].mean(1),
                                  e2[:,5:-1].mean(1),atol=1e-7,rtol=0)
            assert not torch.equal(e1[:,5:-1],e2[:,5:-1])
            assert torch.count_nonzero(e1[:,5:-1,4])==0
            assert torch.count_nonzero(e1[:,-1])==0
            erased=r47.candidates(w1,erase_B=True)[1]
            assert torch.count_nonzero(erased[:,5:-1])==0
            assert torch.equal(erased[:,:5],e1[:,:5])
            tagless=r47.candidates(w1,erase_tag=True)[0]
            assert not torch.equal(tagless,a1)


def test_factual_physics_and_fit_only_cue_observability():
    records,labels=r47.factual_fit()
    assert len(records)==384
    assert len(labels)==192
    assert labels.count(0)==labels.count(1)==96
    offset=r47.fit_offset(records)
    assert offset==pytest.approx(
        r47.r33.r32.after(r47.PROBE,0,0,0.),abs=1e-6)
    reference=r47.reference_ceiling(offset)
    assert {k: v["worlds"] for k,v in reference.items()} == {
        "fit":192,"cal":64,"ood":64}
    assert all(v["mistakes"]==0 for v in reference.values())
    for panel in ("fit","cal","ood"):
        for group in (0,1):
            for context in (0,1):
                for sign in (-1,1):
                    for gate in (-1,1):
                        w=r47.world(panel,group,context,sign,gate)
                        cost=[]
                        for choice in (0,1):
                            following=r47._following(w,choice)
                            key,measured=r47.r33.b._record(
                                w["return_state"],choice,following,context)
                            actual=w["return_state"]+r47.r31._physical_delta(
                                measured[0])
                            assert actual==pytest.approx(following,abs=1e-6)
                            cost.append((actual-1.)**2)
                        for choice in (0,1):
                            assert r47.regret(w,choice)==pytest.approx(
                                max(0.,cost[choice]-min(cost)),abs=1e-6)
    with pytest.raises(ValueError, match="CALIBRATION_MUST_USE_FULL_FIT"):
        r47.fit_offset(records[:-1])


def test_genuine_two_seed_learned_selector_conventional_native_and_cold_memory():
    result=r47.run()
    assert result["schema"]==r47.SCHEMA
    assert result["freeze_blob"]==r47.FREEZE_BLOB
    assert result["parent"]==r47.PARENT
    assert result["factual_FIT_returns"]==384
    assert result["FIT_training_pairs"]==128
    assert result["FIT_heldout_pairs"]==64
    assert result["neural_optimizer_updates"]==2048
    assert len(result["models"])==8
    assert set(result["source_parity"])=={"fit","cal","ood"}
    assert all(z["max_full_B_mean_diff"]<=1e-7
               for z in result["source_parity"].values())
    assert all(z["mistakes"]==0 for z in
               result["observational_ceiling"].values())
    assert len(result["judgment"]["per_seed"])==2
    assert not result["judgment"]["independent_scientific_pass"]
    assert not result["judgment"]["Native_specific_gain_verified"]
    assert not result["independent_scientific_pass"]
    assert not result["external_first_use"]
    assert result["deployment"]=="NONE"
    for model in result["models"]:
        assert model["updates"]==256
        assert model["cold_restore"]
        assert model["trainable_parameters"]>0
        assert [c["step"] for c in model["traces"]]==list(r47.CHECKPOINTS)
        assert all(0<=c["heldout_FIT_accuracy"]<=1 for c in model["traces"])
        assert len(model["scores"]["cal"]["active"]["choices"])==64
        assert len(model["scores"]["ood"]["active"]["choices"])==64
        for intervention in ("erase_B","erase_tag","erase_fifth","swap_B",
                             "uniform","zero_native","zero_key"):
            assert len(model["scores"]["ood"][intervention]["choices"])==64
    r47.retain_ci_receipt(result)
