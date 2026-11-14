"""F8-R53 frozen repeated actual selected action-intervention study."""
import pytest
import torch

from experiments import p85mr_f8r53_repeated_intervention_stability as r53


def test_preregistered_freeze_original_firstuse_and_independent_salts():
    frozen=r53.locked()
    assert frozen["stage"]=="FROZEN_BEFORE_GENERATOR_MODEL_OR_SCORE"
    assert r53.git_blob(r53.FREEZE.read_bytes())==r53.FREEZE_BLOB
    assert frozen["parent"]["qualified_r52_head"]==r53.PARENT
    assert frozen["source"]["fresh_salts"]==r53.SALT
    assert frozen["source"]["full_candidate_record_length"]=={
        "fit":54,"cal":66,"ood":102}
    assert frozen["source"]["no_old_OOD_source_reuse"]
    assert len(r53.ARMS)==14
    assert frozen["methods"]["total_optimizer_attempts"]==3584
    assert not any(frozen["authority"].values())
    assert not frozen["frozen_score"]["independent_scientific_pass"]
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r53.forbid_original_external_firstuse("original")
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r53.world("ood",r53.EVAL_GROUPS,0,-1,1)
    with pytest.raises(ValueError,match="INVALID_WORLD_FACTOR"):
        r53.world("fit",0,3,1,1)


def test_three_independent_selected_action_pairs_and_full_mean_parity():
    for panel,nsource,length in (("fit",8,54),("cal",10,66),("ood",16,102)):
        parity=r53.source_parity(panel)
        assert parity["groups"]==(24 if panel=="fit" else 8)
        assert parity["max_gate_full_B_mean_difference"]<=1e-7
        for g in range(parity["groups"]):
            cue,decoy,high1,high2=r53.positions(panel,g)
            assert len({cue,decoy,high1,high2})==4
            for c in (0,1):
                for gate in (-1,1):
                    w=r53.world(panel,g,c,1,gate)
                    events=r53.selected_repeated_B(w)
                    assert len(events)==6*nsource
                    for source in range(nsource):
                        a=events[source*6:(source+1)*6]
                        assert len(a)==6
                        assert {x[0] for x in a}=={a[0][0]}
                        assert [x[1] for x in a]==[0,1,0,1,0,1]
                        for rep in range(3):
                            assert a[2*rep][1]!=a[2*rep+1][1]
                    assert events[cue*6][0]==(.42 if c==0 else .18)
                    assert events[decoy*6][0]==(.18 if c==0 else .42)
                    assert events[high1*6][0]==.88
                    assert events[high2*6][0]==.77
                    a,e=r53.candidates(w)
                    assert a.shape==e.shape==(2,length,16)
                    assert torch.count_nonzero(e[:,-1])==0
                    assert torch.count_nonzero(e[:,5:-1,4])==0
                    tag,mean,var,reliability,mag,ratio,fifth,ctx,obs=(
                        r53.public_repeated_features(a,e,torch.tensor(0.)))
                    assert tag.shape==mean.shape==var.shape==(
                        1,nsource)
                    assert ctx.tolist()==[c]
                    assert mean[0,cue].item()==pytest.approx(
                        r53.AMP[panel]*gate,abs=1e-6)
                    assert mean[0,decoy].item()==pytest.approx(
                        -r53.AMP[panel]*gate,abs=1e-6)
                    assert var[0,cue].item()==pytest.approx(0.,abs=1e-10)
                    assert var[0,decoy].item()==pytest.approx(0.,abs=1e-10)
                    assert var[0,high1]>0 and var[0,high2]>0
                    assert mag[0,high1]>mag[0,cue]
                    assert mag[0,high2]>mag[0,cue]
                    assert ratio[0,cue]>ratio[0,high1]
                    assert ratio[0,decoy]>ratio[0,high2]
                    erased=r53.candidates(w,erase_B=True)[1]
                    assert not torch.any(erased[:,5:-1])


def test_source_only_selected_intervention_reference_without_hidden_oracle():
    fit,labels=r53.factual_fit()
    assert len(fit)==384 and len(labels)==192
    assert labels.count(0)==96 and labels.count(1)==96
    offset=r53.calibration(fit)
    assert offset==pytest.approx(
        r53.r52.r51.r33.r32.after(r53.PROBE,0,0,0.),abs=1e-6)
    reference=r53.positive_reference(offset)
    assert {p:v["worlds"] for p,v in reference.items()}=={
        "fit":192,"cal":64,"ood":64}
    assert all(v["mistakes"]==0 for v in reference.values())


def test_beta_starts_zero_and_factual_gradient_is_real_and_finite():
    fit,labels=r53.factual_fit()
    center=r53.calibration(fit)
    main=r53.Policy("learned_repeatability",985201,center)
    raw=r53.Policy("raw_mean_context",985201,center)
    assert float(main.beta.detach())==0.
    assert main.slopes.detach().tolist()==[0.,0.]
    a,e,y=r53.batch(fit,labels,list(range(r53.BATCH)))
    assert torch.equal(main(a,e),raw(a,e))
    loss=torch.nn.functional.cross_entropy(-main(a,e),y)
    loss.backward()
    assert main.beta.grad is not None
    assert bool(torch.isfinite(main.beta.grad).all())
    assert main.slopes.grad is not None
    assert bool(torch.isfinite(main.slopes.grad).all())


def test_local_native_accepts_full_repeated_selected_history():
    model=r53.LongNative("native_factual_memory",985201)
    for panel,length in (("fit",54),("cal",66),("ood",102)):
        a,e=r53.candidates(r53.world(panel,0,1,-1,1))
        assert a.shape==(2,length,16)
        for ablated in (False,True):
            p=model(a,e,ablate_backbone=ablated)
            assert p.shape==(2,)
            assert bool(torch.isfinite(p).all())
    with pytest.raises(ValueError,match="UNREGISTERED_LENGTH"):
        model(torch.zeros((2,100,16)),torch.zeros((2,100,16)))


def test_full_real_two_seed_training_and_preserve_negative_science():
    result=r53.run()
    assert result["schema"]==r53.SCHEMA
    assert result["freeze_blob"]==r53.FREEZE_BLOB
    assert result["parent"]==r53.PARENT
    assert result["source_neutral_synthetic_DEV_only"]
    assert result["FIT_factual_selected_returns"]==384
    assert result["FIT_train_pairs"]==128
    assert result["FIT_heldout_pairs"]==64
    assert result["optimizer_attempts"]==3584
    assert len(result["models"])==14
    assert len(result["judgment"]["seeds"])==2
    assert result["independent_scientific_pass"] is False
    assert result["judgment"]["independent_scientific_pass"] is False
    assert result["external_original_first_use"] is False
    assert result["deployment"]=="NONE"
    assert all(d["mistakes"]==0 for d in result["observable_reference"].values())
    assert all(d["max_gate_full_B_mean_difference"]<=1e-7
               for d in result["source_parity"].values())
    for m in result["models"]:
        assert m["optimizer_attempts"]==256
        assert 0<=m["productive_gradient_steps"]<=256
        assert m["parameter_count"]>0
        assert m["cold_restore"]
        assert [v["step"] for v in m["traces"]]==list(r53.CHECKPOINTS)
        assert len(m["scores"]["cal"]["active"]["choices"])==64
        assert len(m["scores"]["ood"]["active"]["choices"])==64
        for k in ("erase_B","erase_tag","erase_fifth","zero_beta",
                  "force_tag_top2","force_mean_top2","force_uniform",
                  "zero_native","zero_signal"):
            assert len(m["scores"]["ood"][k]["choices"])==64
    assert all(isinstance(s["frozen_DEV_pass"],bool)
               for s in result["judgment"]["seeds"])
    r53.retain_ci_receipt(result)
