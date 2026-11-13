"""F8-R49 prospective public-key event retrieval versus unfiltered history.

All original/previous science remains outside this fresh source-neutral DEV.
"""
import pytest
import torch

from experiments import p85mr_f8r49_public_event_selection as r49


def test_frozen_prior_source_and_first_use_authority():
    frozen=r49.locked()
    assert frozen["stage"]=="FROZEN_BEFORE_R49_GENERATOR_AND_MODEL_IMPLEMENTATION"
    assert frozen["parent"]["head"]==r49.PARENT
    assert frozen["source"]["salts"]==r49.SALT
    assert frozen["source"]["length"]=={"fit":8,"cal":12,"ood":32}
    assert frozen["source"]["FIT_selected_factual_returns"]==384
    assert frozen["preregistered_score"]["independent_scientific_pass"] is False
    assert len(r49.ARMS)==12
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r49.deny_external_original_first_use("F7")
    with pytest.raises(ValueError,match="UNREGISTERED_PANEL"):
        r49.worlds("external_original")
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r49.world("ood",r49.EVAL_GROUPS,0,1,1)


def test_full_source_parity_and_public_top2_events():
    for panel, length in (("fit",8),("cal",12),("ood",32)):
        checks=r49.parity(panel)
        assert checks["groups"]==(r49.FIT_GROUPS if panel=="fit" else r49.EVAL_GROUPS)
        assert checks["contexts"]==2
        assert checks["max_paired_full_B_mean_difference"]<=1e-7
        for group in range(checks["groups"]):
            cue,decoy=r49.positions(panel,group)
            assert cue!=decoy
            for context in (0,1):
                plus=r49.world(panel,group,context,1,1)
                minus=r49.world(panel,group,context,1,-1)
                aa,ee=r49.candidates(plus)
                ab,eb=r49.candidates(minus)
                assert aa.shape==ee.shape==(2,6+length,16)
                assert torch.equal(aa,ab)
                assert torch.allclose(ee[:,5:-1].mean(1),
                                      eb[:,5:-1].mean(1),atol=1e-7,rtol=0)
                assert not torch.equal(ee[:,5:-1],eb[:,5:-1])
                assert torch.count_nonzero(ee[:,5:-1,4])==0
                assert torch.count_nonzero(ee[:,-1])==0
                tag,delta,fifth,ctx,obs=r49.components(
                    aa,ee,torch.tensor(0.))
                highest=torch.topk(tag,2,dim=-1).values[0].tolist()
                assert highest==pytest.approx([.42,.18],abs=1e-5)
                assert ctx[0].item()==context
                assert tag.shape==(1,length)
                assert delta.shape==(1,length)
                assert bool(torch.isfinite(fifth).all())
                erased=r49.candidates(plus,erase_B=True)[1]
                assert torch.count_nonzero(erased[:,5:-1])==0
                assert torch.equal(erased[:,:5],ee[:,:5])
                tagless=r49.candidates(plus,erase_tag=True)[0]
                assert not torch.equal(tagless,aa)


def test_true_physical_consequence_factual_fit_and_positive_reference():
    facts,labels=r49.factual_fit()
    assert len(facts)==384
    assert len(labels)==192
    assert labels.count(0)==labels.count(1)==96
    offset=r49.calibration(facts)
    assert offset==pytest.approx(
        r49.r33.r32.after(r49.PROBE,0,0,0.),abs=1e-6)
    observed=r49.positive_observation_reference(offset)
    assert {k:z["worlds"] for k,z in observed.items()}=={
        "fit":192,"cal":64,"ood":64}
    assert all(z["mistakes"]==0 for z in observed.values())
    for panel in ("fit","cal","ood"):
        for group in (0,1):
            for context in (0,1):
                for q in (-1,1):
                    for gate in (-1,1):
                        w=r49.world(panel,group,context,q,gate)
                        costs=[]
                        for choice in (0,1):
                            after=r49.following(w,choice)
                            key,e=r49.r33.b._record(
                                w["return_state"],choice,after,context)
                            decoded=w["return_state"]+r49.r31._physical_delta(e[0])
                            assert decoded==pytest.approx(after,abs=1e-6)
                            costs.append((decoded-1.)**2)
                        for choice in (0,1):
                            assert r49.regret(w,choice)==pytest.approx(
                                max(0.,costs[choice]-min(costs)),abs=1e-6)
    with pytest.raises(ValueError,match="CALIBRATION_UNQUALIFIED"):
        r49.calibration(facts[:4])


def test_new_event_selector_is_trainable_from_only_public_inputs():
    facts,labels=r49.factual_fit()
    offset=r49.calibration(facts)
    model=r49.Policy("event_context_selector",r49.SEEDS[0],offset)
    assert model.slopes.detach().tolist()==[0.,0.]
    a,e,y=r49.batch(facts,labels,list(range(r49.BATCH)))
    loss=torch.nn.functional.cross_entropy(-model(a,e),y)
    loss.backward()
    assert model.slopes.grad is not None
    assert bool(torch.isfinite(model.slopes.grad).all())
    # Content and readout are determined from the *same* observed selected
    # action/effect tensors in both event and full-history comparisons.
    raw=r49.Policy("raw_context_selector",r49.SEEDS[0],offset)
    assert raw(a,e).shape==model(a,e).shape==(r49.BATCH,2)


def test_full_two_seed_training_ablations_and_cold_restore():
    result=r49.run()
    assert result["schema"]==r49.SCHEMA
    assert result["freeze_blob"]==r49.FREEZE_BLOB
    assert result["parent"]==r49.PARENT
    assert result["factual_FIT_selected_returns"]==384
    assert result["FIT_training_pairs"]==128
    assert result["FIT_heldout_pairs"]==64
    assert result["neural_optimizer_updates"]==3072
    assert len(result["models"])==12
    assert len(result["judgment"]["seeds"])==2
    assert result["judgment"]["handcoded_public_top2_event_selection"]
    assert not result["judgment"]["native_specific_gain_verified"]
    assert not result["judgment"]["independent_scientific_pass"]
    assert not result["independent_scientific_pass"]
    assert not result["external_original_first_use"]
    assert result["source_neutral_DEV_only"]
    assert result["deployment"]=="NONE"
    assert all(z["mistakes"]==0 for z in result["observable_reference"].values())
    assert all(z["max_paired_full_B_mean_difference"]<=1e-7
               for z in result["B_parity"].values())
    for m in result["models"]:
        assert m["actual_updates"]==256
        assert m["cold_restore"]
        assert m["trainable_parameters"]>0
        assert [t["step"] for t in m["traces"]]==list(r49.CHECKPOINTS)
        assert all(0<=t["heldout_accuracy"]<=1 for t in m["traces"])
        for panel in ("cal","ood"):
            assert len(m["scores"][panel]["active"]["choices"])==64
        for intervention in ("erase_B","erase_tag","erase_fifth",
                 "swap_B","force_uniform","force_raw_history",
                 "force_context0_slope","zero_native","zero_key"):
            assert len(m["scores"]["ood"][intervention]["choices"])==64
    r49.retain_ci_receipt(result)
