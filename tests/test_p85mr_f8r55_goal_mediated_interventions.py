"""F8-R55 frozen goal-mediated, selected downstream intervention science."""
import pytest
import torch
from experiments import p85mr_f8r55_goal_mediated_interventions as r55


def test_prospective_freeze_and_original_external_firstuse_admission():
    f=r55.locked()
    assert f["schema"]==r55.SCHEMA
    assert f["stage"]=="FROZEN_BEFORE_NEW_GENERATOR_MODEL_OR_SCORE"
    assert r55.git_blob(r55.FREEZE.read_bytes())==r55.FREEZE_BLOB
    assert f["parent"]["r54_head"]==r55.PARENT
    assert f["source"]["salts"]==r55.SALT
    assert f["source"]["full_candidate_events"]=={
        "fit":54,"cal":66,"ood":102}
    assert f["training"]["optimizer_attempts_total"]==4096
    assert f["source"]["no_old_OOD_source_reuse"]
    assert len(r55.ARMS)==16
    assert not any(f["authority"].values())
    assert f["frozen_score"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r55.reject_external_original_firstuse()
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r55.world("ood",8,0,1,1)
    with pytest.raises(ValueError,match="INVALID_WORLD_FACTOR"):
        r55.world("fit",0,2,1,1)


def test_each_selected_upstream_action_has_actual_immediate_and_delayed_target():
    for panel,nsource,length in (
        ("fit",8,54),("cal",10,66),("ood",16,102)):
        audit=r55.source_parity(panel)
        assert audit["groups"]==(24 if panel=="fit" else 8)
        assert audit["max_paired_full_B_mean_difference"]<=1e-7
        for group in range(audit["groups"]):
            cue,decoy,sp1,sp2=r55.positions(panel,group)
            assert len({cue,decoy,sp1,sp2})==4
            for ctx in (0,1):
                for gate in (-1,1):
                    w=r55.world(panel,group,ctx,1,gate)
                    observations=r55.selected_upstream_downstream(w)
                    assert len(observations)==6*nsource
                    for src in range(nsource):
                        selected=observations[6*src:6*src+6]
                        assert [x[1] for x in selected]==[0,0,0,1,1,1]
                        assert selected[0][0]==selected[3][0]
                        assert selected[1][0]==selected[4][0]==r55.NOW_STATE
                        assert selected[2][0]==selected[5][0]==r55.DELAY_STATE
                    assert observations[6*sp1][0]==.88
                    assert observations[6*sp2][0]==.77
                    assert observations[6*cue][0]==(.42 if ctx==0 else .18)
                    assert observations[6*decoy][0]==(.18 if ctx==0 else .42)
                    a,e=r55.candidates(w)
                    assert a.shape==e.shape==(2,length,16)
                    assert not torch.any(e[:,-1])
                    assert not torch.any(e[:,5:-1,4])
                    tag,up,now,delay,fifth,c,obs=r55.observed_goal_features(
                        a,e,torch.tensor(0.))
                    assert tag.shape==up.shape==now.shape==delay.shape==(
                        1,nsource)
                    assert c.tolist()==[ctx]
                    assert float(delay[0,cue])==pytest.approx(
                        r55.AMP[panel]*gate,abs=1e-6)
                    assert float(delay[0,decoy])==pytest.approx(
                        -r55.AMP[panel]*gate,abs=1e-6)
                    assert abs(float(delay[0,sp1]))<=1e-7
                    assert abs(float(delay[0,sp2]))<=1e-7
                    assert abs(float(up[0,sp1]))>=r55.UP_SPUR[panel]-1e-6
                    assert abs(float(now[0,sp2]))>=r55.NOW_SPUR[panel]-1e-6
                    assert float(delay[0,sp1])!=pytest.approx(
                        float(now[0,sp1]),abs=.1)
                    erased=r55.candidates(w,erase_delayed=True)[1]
                    assert not torch.any(erased[:,7:-1:6])
                    assert not torch.any(erased[:,10:-1:6])
                    assert torch.equal(erased[:,6,:],e[:,6,:])


def test_handcoded_observed_only_goal_sensor_reference_and_factual_fit():
    facts,labels=r55.factual_fit()
    assert len(facts)==384 and len(labels)==192
    assert labels.count(0)==96 and labels.count(1)==96
    off=r55.calibration(facts)
    assert off==pytest.approx(
        r55.r53.r52.r51.r33.r32.after(r55.PROBE,0,0,0.),abs=1e-6)
    ref=r55.positive_reference(off)
    assert {k:x["worlds"] for k,x in ref.items()}=={
        "fit":192,"cal":64,"ood":64}
    assert all(x["mistakes"]==0 for x in ref.values())


def test_goal_beta_starts_exactly_zero_and_gradients_from_actual_fit():
    facts,labels=r55.factual_fit()
    off=r55.calibration(facts)
    main=r55.Policy("learned_goal_mediation",985201,off)
    raw=r55.Policy("raw_delayed_context",985201,off)
    assert float(main.beta.detach())==0.
    assert main.slopes.detach().tolist()==[0.,0.]
    a,e,y=r55.batch(facts,labels,list(range(16)))
    assert torch.equal(main(a,e),raw(a,e))
    loss=torch.nn.functional.cross_entropy(-main(a,e),y)
    loss.backward()
    assert main.beta.grad is not None
    assert bool(torch.isfinite(main.beta.grad).all())
    assert main.slopes.grad is not None
    assert bool(torch.isfinite(main.slopes.grad).all())


def test_native_uses_complete_target_sensor_interventions_not_padding():
    native=r55.r53.LongNative("native_factual_memory",985201)
    for panel,length in (("fit",54),("cal",66),("ood",102)):
        a,e=r55.candidates(r55.world(panel,0,1,1,1))
        assert a.shape==(2,length,16)
        for ablation in (True,False):
            costs=native(a,e,ablate_backbone=ablation)
            assert costs.shape==(2,)
            assert torch.isfinite(costs).all()


def test_frozen_4096_training_attempts_and_all_16_models_real_ablations():
    z=r55.run()
    assert z["schema"]==r55.SCHEMA
    assert z["freeze_blob"]==r55.FREEZE_BLOB
    assert z["parent"]==r55.PARENT
    assert z["source_neutral_synthetic_DEV_only"]
    assert z["FIT_selected_factual_returns"]==384
    assert z["FIT_train_pairs"]==128 and z["FIT_heldout_pairs"]==64
    assert z["optimizer_attempts"]==4096
    assert len(z["models"])==16
    assert len(z["judgment"]["seeds"])==2
    assert z["independent_scientific_pass"] is False
    assert z["judgment"]["independent_scientific_pass"] is False
    assert z["external_original_firstuse"] is False
    assert z["deployment"]=="NONE"
    assert all(x["mistakes"]==0 for x in z["observable_reference"].values())
    assert all(x["max_paired_full_B_mean_difference"]<=1e-7
               for x in z["source_mean_parity"].values())
    for m in z["models"]:
        assert m["optimizer_steps"]==256
        assert 0<=m["productive_steps"]<=256
        assert m["trainable_parameters"]>0
        assert m["cold_restore"]
        assert [x["step"] for x in m["trace"]]==list(r55.CHECKPOINTS)
        assert len(m["scores"]["cal"]["active"]["choices"])==64
        assert len(m["scores"]["ood"]["active"]["choices"])==64
        for name in (
            "erase_B","erase_tag","erase_fifth","erase_immediate",
            "erase_delayed","zero_beta","force_tag_top2",
            "force_upstream_top2","force_immediate_top2",
            "zero_native","zero_signal"):
            assert len(m["scores"]["ood"][name]["choices"])==64
    for seed in z["judgment"]["seeds"]:
        assert seed["seed"] in r55.SEEDS
        assert isinstance(seed["frozen_DEV_pass"],bool)
    r55.retain_ci_receipt(z)
