"""R48 source-neutral learned context-dependent causal cue-role qualification."""
import pytest
import torch

from experiments import p85mr_f8r48_context_role_switch as r48


def test_frozen_ancestry_and_external_firstuse_denied():
    f=r48.locked()
    assert f["parent"]["head"]==r48.PARENT
    assert f["source"]["salts"]==r48.SALT
    assert f["frozen_dev_score"]["independent_scientific_pass"] is False
    assert len(r48.ARMS)==10
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r48.deny_external_original_first_use("original")
    with pytest.raises(ValueError,match="UNREGISTERED_PANEL"):
        r48.worlds("original_F7")
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r48.world("ood",r48.EVAL_GROUPS,0,1,1)


def test_role_switch_source_mean_parity_and_public_context_access():
    for panel in ("fit","cal","ood"):
        cert=r48.parity(panel)
        assert cert["groups"]==(r48.FIT_GROUPS if panel=="fit"
                                else r48.EVAL_GROUPS)
        assert cert["contexts"]==2
        assert cert["max_paired_full_B_mean_difference"]<=1e-7
        for g in range(cert["groups"]):
            cue,decoy=r48.positions(panel,g)
            assert cue!=decoy
            for ctx in (0,1):
                wp=r48.world(panel,g,ctx,1,1)
                wm=r48.world(panel,g,ctx,1,-1)
                assert r48.B_selected(wp)[cue][0]==(
                       r48.CUE_TAG if ctx==0 else r48.DECOY_TAG)
                assert r48.B_selected(wp)[decoy][0]==(
                       r48.DECOY_TAG if ctx==0 else r48.CUE_TAG)
                a1,e1=r48.candidates(wp)
                a2,e2=r48.candidates(wm)
                assert a1.shape==e1.shape==((2,22,16) if panel=="ood"
                                             else (2,14,16))
                assert torch.equal(a1,a2)
                assert torch.allclose(e1[:,5:-1].mean(1),
                                      e2[:,5:-1].mean(1),atol=1e-7,rtol=0)
                assert not torch.equal(e1[:,5:-1],e2[:,5:-1])
                assert torch.count_nonzero(e1[:,5:-1,4])==0
                assert torch.count_nonzero(e1[:,-1])==0
                assert torch.equal(e1[:,:5],e2[:,:5])
                e_erased=r48.candidates(wp,erase_B=True)[1]
                assert torch.count_nonzero(e_erased[:,5:-1])==0
                assert torch.equal(e_erased[:,:5],e1[:,:5])
                a_tagless=r48.candidates(wp,erase_tag=True)[0]
                assert not torch.equal(a_tagless,a1)


def test_physical_selected_outcome_factual_fit_and_observational_reference():
    facts,labels=r48.factual_fit()
    assert len(facts)==384
    assert len(labels)==192
    assert labels.count(0)==labels.count(1)==96
    offset=r48.calibration(facts)
    assert offset==pytest.approx(
        r48.r33.r32.after(r48.PROBE,0,0,0.),abs=1e-6)
    reference=r48.positive_observation_reference(offset)
    assert {p: reference[p]["worlds"] for p in reference}=={
        "fit":192,"cal":64,"ood":64}
    assert all(row["mistakes"]==0 for row in reference.values())
    for panel in ("fit","cal","ood"):
        for group in (0,1):
            for ctx in (0,1):
                for q in (-1,1):
                    for gate in (-1,1):
                        w=r48.world(panel,group,ctx,q,gate)
                        costs=[]
                        for action in (0,1):
                            f=r48.following(w,action)
                            k,e=r48.r33.b._record(
                                w["return_state"],action,f,ctx)
                            measured=w["return_state"]+r48.r31._physical_delta(e[0])
                            assert measured==pytest.approx(f,abs=1e-6)
                            costs.append((measured-w["goal"])**2)
                        for action in (0,1):
                            assert r48.regret(w,action)==pytest.approx(
                                max(0.,costs[action]-min(costs)),abs=1e-6)
    with pytest.raises(ValueError,match="CALIBRATION_UNQUALIFIED"):
        r48.calibration(facts[:-1])


def test_context_conditioning_is_observable_only_and_gradients_are_real():
    facts,labels=r48.factual_fit()
    offset=r48.calibration(facts)
    model=r48.Policy("context_key_selector",r48.SEEDS[0],offset)
    assert model.slopes.detach().tolist()==[0.,0.]
    a,e,y=r48.batch(facts,labels,list(range(r48.BATCH)))
    costs=model(a,e)
    assert costs.shape==(r48.BATCH,2)
    loss=torch.nn.functional.cross_entropy(-costs,y)
    loss.backward()
    assert model.slopes.grad is not None
    assert bool(torch.isfinite(model.slopes.grad).all())
    # Interventions must not change hidden labels, only publicly selected
    # measurement/key information.
    w=r48.world("cal",0,1,1,1)
    original=r48.candidates(w)
    erased=r48.candidates(w,erase_fifth=True)
    assert torch.equal(original[0],erased[0])
    assert not torch.equal(original[1][:,4],erased[1][:,4])


def test_actual_two_seed_frozen_training_and_full_cold_regression():
    result=r48.run()
    assert result["schema"]==r48.SCHEMA
    assert result["freeze_blob"]==r48.FREEZE_BLOB
    assert result["parent"]==r48.PARENT
    assert result["factual_FIT_selected_returns"]==384
    assert result["FIT_training_pairs"]==128
    assert result["FIT_heldout_pairs"]==64
    assert result["neural_optimizer_updates"]==2560
    assert len(result["models"])==10
    assert len(result["judgment"]["seeds"])==2
    assert not result["judgment"]["native_specific_gain_verified"]
    assert not result["judgment"]["independent_scientific_pass"]
    assert not result["independent_scientific_pass"]
    assert not result["external_original_first_use"]
    assert result["deployment"]=="NONE"
    assert result["source_neutral_DEV_only"]
    assert all(row["mistakes"]==0 for row in result["observable_reference"].values())
    assert all(row["max_paired_full_B_mean_difference"]<=1e-7
               for row in result["B_parity"].values())
    for model in result["models"]:
        assert model["actual_updates"]==256
        assert model["cold_restore"]
        assert model["trainable_parameters"]>0
        assert [t["step"] for t in model["traces"]]==list(r48.CHECKPOINTS)
        assert all(0<=t["heldout_accuracy"]<=1 for t in model["traces"])
        assert len(model["scores"]["cal"]["active"]["choices"])==64
        assert len(model["scores"]["ood"]["active"]["choices"])==64
        for ablation in ("erase_B","erase_tag","erase_fifth","swap_B",
                         "force_uniform","force_context0_slope",
                         "zero_native","zero_key"):
            assert len(model["scores"]["ood"][ablation]["choices"])==64
    r48.retain_ci_receipt(result)
