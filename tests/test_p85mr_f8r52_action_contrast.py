"""R52 actual paired selected-action contrast versus tag/raw magnitude controls."""
import pytest
import torch
from experiments import p85mr_f8r52_action_contrast as r52


def test_prereg_frozen_before_source_and_external_authority():
    f=r52.locked()
    assert f["stage"]=="FROZEN_BEFORE_GENERATOR_MODEL_OR_SCORED_RESULTS"
    assert r52.git_blob(r52.FREEZE.read_bytes())==r52.FREEZE_BLOB
    assert f["parent"]["qualified_R51_HEAD"]==r52.PARENT
    assert f["source"]["new_salts"]==r52.SALT
    assert f["source"]["selected_B_record_count"]=={
        "fit":24,"cal":32,"ood":40}
    assert len(r52.ARMS)==14
    assert not any(f["authority"].values())
    assert not f["frozen_score"]["independent_scientific_pass"]
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r52.reject_external_original_firstuse("external")
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r52.world("ood",r52.EVAL_GROUPS,0,1,1)
    with pytest.raises(ValueError,match="INVALID_FACTOR"):
        r52.world("ood",0,2,1,1)


def test_two_genuinely_selected_action_effects_per_public_source():
    for panel,npairs,events in (("fit",12,30),("cal",16,38),("ood",20,46)):
        par=r52.source_parity(panel)
        assert par["groups"]==(24 if panel=="fit" else 8)
        assert par["max_paired_full_B_mean_difference"]<=1e-7
        for group in range(par["groups"]):
            cue,decoy,sp1,sp2=r52.positions(panel,group)
            assert len({cue,decoy,sp1,sp2})==4
            for ctx in (0,1):
                for gate in (-1,1):
                    w=r52.world(panel,group,ctx,1,gate)
                    selected=r52.selected_B_interventions(w)
                    assert len(selected)==2*npairs
                    for pair in range(npairs):
                        assert selected[2*pair][0]==selected[2*pair+1][0]
                        assert selected[2*pair][1]==0
                        assert selected[2*pair+1][1]==1
                    assert selected[2*sp1][0]==.88
                    assert selected[2*sp2][0]==.77
                    assert selected[2*sp1][2]==selected[2*sp1+1][2]
                    assert selected[2*sp2][2]==selected[2*sp2+1][2]
                    assert selected[2*cue][2]==pytest.approx(
                        r52.AMP[panel]*gate)
                    assert selected[2*cue+1][2]==pytest.approx(
                        -r52.AMP[panel]*gate)
                    assert selected[2*decoy][2]==pytest.approx(
                        -r52.AMP[panel]*gate)
                    assert selected[2*decoy+1][2]==pytest.approx(
                        r52.AMP[panel]*gate)
                    a,e=r52.candidates(w)
                    assert a.shape==e.shape==(2,events,16)
                    assert torch.count_nonzero(e[:,-1])==0
                    assert torch.count_nonzero(e[:,5:-1,4])==0
                    tags,contrast,magnitude,d0,d1,fifth,contexts,obs=(
                        r52.public_pair_features(a,e,torch.tensor(0.)))
                    assert tags.shape==contrast.shape==magnitude.shape==(
                        1,npairs)
                    assert contexts.tolist()==[ctx]
                    assert bool(torch.isfinite(contrast).all())
                    assert bool(torch.isfinite(fifth).all())
                    assert contrast[0,cue].item()==pytest.approx(
                        r52.AMP[panel]*gate,abs=1e-6)
                    assert contrast[0,decoy].item()==pytest.approx(
                        -r52.AMP[panel]*gate,abs=1e-6)
                    assert abs(float(contrast[0,sp1]))<=1e-7
                    assert abs(float(contrast[0,sp2]))<=1e-7
                    erase=r52.candidates(w,erase_B=True)[1]
                    assert not bool(torch.any(erase[:,5:-1,:]))


def test_public_intervention_reference_uses_selected_consequences_only():
    fit,labels=r52.factual_fit()
    assert len(fit)==384 and len(labels)==192
    assert labels.count(0)==labels.count(1)==96
    offset=r52.calibration(fit)
    assert offset==pytest.approx(
        r52.r51.r33.r32.after(r52.PROBE,0,0,0.),abs=1e-6)
    q=r52.positive_observation_reference(offset)
    assert {p:d["worlds"] for p,d in q.items()}=={
        "fit":192,"cal":64,"ood":64}
    assert all(d["mistakes"]==0 for d in q.values())


def test_learning_salience_zero_initialization_and_true_grads():
    fit,labels=r52.factual_fit()
    offset=r52.calibration(fit)
    model=r52.Policy("learned_action_contrast",r52.SEEDS[0],offset)
    raw=r52.Policy("raw_allpair_contrast",r52.SEEDS[0],offset)
    assert float(model.salience.detach())==0.
    assert model.slopes.detach().tolist()==[0.,0.]
    a,e,y=r52.batch(fit,labels,list(range(r52.BATCH)))
    assert torch.equal(model(a,e),raw(a,e))
    loss=torch.nn.functional.cross_entropy(-model(a,e),y)
    loss.backward()
    assert model.slopes.grad is not None
    assert bool(torch.isfinite(model.slopes.grad).all())
    assert model.salience.grad is not None
    assert bool(torch.isfinite(model.salience.grad).all())


def test_Native_true_full_46_event_history_adapter():
    m=r52.R52FullHistoryFactualMemory(
        "native_factual_memory",r52.SEEDS[0])
    for panel,records in (("fit",30),("cal",38),("ood",46)):
        a,e=r52.candidates(r52.world(panel,0,1,-1,1))
        assert a.shape==(2,records,16)
        for ablated in (False,True):
            val=m(a,e,ablate_backbone=ablated)
            assert val.shape==(2,)
            assert bool(torch.isfinite(val).all())
    with pytest.raises(ValueError,match="UNREGISTERED_NATIVE_HISTORY"):
        m(torch.zeros((2,18,16)),torch.zeros((2,18,16)))


def test_full_two_seed_train_score_and_preserve_negative_outcomes():
    d=r52.run()
    assert d["schema"]==r52.SCHEMA
    assert d["freeze_blob"]==r52.FREEZE_BLOB
    assert d["parent"]==r52.PARENT
    assert d["source_neutral_synthetic_DEV_only"]
    assert d["selected_FIT_factual_return_effects"]==384
    assert d["FIT_training_pairs"]==128
    assert d["FIT_heldout_pairs"]==64
    assert d["neural_optimizer_updates"]==3584
    assert len(d["models"])==14
    assert len(d["judgment"]["seeds"])==2
    assert not d["independent_scientific_pass"]
    assert not d["external_original_first_use"]
    assert d["deployment"]=="NONE"
    assert all(v["mistakes"]==0 for v in
               d["observable_positive_reference"].values())
    assert all(v["max_paired_full_B_mean_difference"]<=1e-7 for v in
               d["source_parity"].values())
    for arm in d["models"]:
        assert arm["actual_updates"]==256
        assert arm["cold_restore"]
        assert 0<=arm["productive_gradient_steps"]<=256
        assert arm["trainable_parameters"]>0
        assert [x["step"] for x in arm["traces"]]==list(r52.CHECKPOINTS)
        assert len(arm["scores"]["cal"]["active"]["choices"])==64
        assert len(arm["scores"]["ood"]["active"]["choices"])==64
        for key in ("erase_B","erase_tag","erase_fifth","zero_salience",
                    "force_tag_top2","force_raw_magnitude_top2",
                    "force_uniform","zero_native","zero_contrast"):
            assert len(arm["scores"]["ood"][key]["choices"])==64
    assert all(isinstance(x["frozen_DEV_pass"],bool)
               for x in d["judgment"]["seeds"])
    r52.retain_ci_receipt(d)
