"""F8-R51 prospective learned event salience versus pre-frozen controls."""
import pytest
import torch

from experiments import p85mr_f8r51_learned_eligibility as r51


def test_before_training_prereg_source_and_authority():
    f=r51.locked()
    assert f["stage"]=="FROZEN_BEFORE_R51_MODEL_GENERATOR_OR_RESULTS"
    assert r51.git_blob(r51.FREEZE.read_bytes())==r51.FREEZE_BLOB
    assert f["parent"]["qualified_R49_head"]==r51.PARENT
    assert f["source"]["panel_salts"]==r51.SALT
    assert f["source"]["selected_B_history_length"]=={
        "fit":12,"cal":20,"ood":40}
    assert f["source"]["no_reused_R50_OOD"] is True
    assert len(r51.ARMS)==12
    assert all(f["authority"].values())
    assert f["frozen_score"]["scientific_pass"] is False
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        r51.deny_external_original_first_use()
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r51.world("ood",r51.EVAL_GROUPS,0,1,1)


def test_36_plus_history_full_public_observation_and_context_causality():
    for p,length in (("fit",12),("cal",20),("ood",40)):
        check=r51.parity(p)
        assert check["max_paired_full_B_mean_difference"]<=1e-7
        assert check["groups"]==(24 if p=="fit" else 8)
        for g in range(check["groups"]):
            cue,decoy,hi1,hi2=r51.positions(p,g)
            assert len({cue,decoy,hi1,hi2})==4
            confounded=p!="fit" or g%2==0
            for c in (0,1):
                for gate in (-1,1):
                    w=r51.world(p,g,c,1,gate)
                    facts=r51.B_selected(w)
                    assert len(facts)==length
                    assert facts[cue][0]==(.42 if c==0 else .18)
                    assert facts[decoy][0]==(.18 if c==0 else .42)
                    assert facts[cue][2]==pytest.approx(r51.AMP[p]*gate)
                    assert facts[decoy][2]==pytest.approx(-r51.AMP[p]*gate)
                    if confounded:
                        assert facts[hi1][0]==.88
                        assert facts[hi2][0]==.77
                        assert facts[hi1][2]==.015
                        assert facts[hi2][2]==-.015
                    a,e=r51.candidates(w)
                    assert a.shape==e.shape==(2,length+6,16)
                    assert torch.count_nonzero(e[:,-1])==0
                    assert torch.count_nonzero(e[:,5:-1,4])==0
                    tag,delta,phys,fifth,ctx,obs=r51.components(
                        a,e,torch.tensor(0.))
                    assert tag.shape==delta.shape==phys.shape==(1,length)
                    assert ctx.tolist()==[c]
                    assert torch.isfinite(fifth).all()
                    assert torch.isfinite(phys).all()
                    assert torch.count_nonzero(obs[:,-1])==0
                    erase=r51.candidates(w,erase_B=True)[1]
                    assert torch.count_nonzero(erase[:,5:-1])==0
                    assert torch.equal(erase[:,:5],e[:,:5])


def test_source_only_positive_reference_fits_before_model_training():
    facts,labels=r51.factual_fit()
    assert len(facts)==384
    assert len(labels)==192
    assert labels.count(0)==96
    offset=r51.calibration(facts)
    assert offset==pytest.approx(
        r51.r33.r32.after(r51.PROBE,0,0,0.),abs=1e-6)
    source=r51.positive_observation_reference(offset)
    assert {p:z["worlds"] for p,z in source.items()}=={
        "fit":192,"cal":64,"ood":64}
    assert all(z["mistakes"]==0 for z in source.values())


def test_learned_magnitude_initialized_zero_and_ablation_is_raw_view():
    facts,labels=r51.factual_fit()
    offset=r51.calibration(facts)
    arm=r51.Policy("learned_magnitude_context",r51.SEEDS[0],offset)
    raw=r51.Policy("raw_context_selector",r51.SEEDS[0],offset)
    assert arm.salience.detach().item()==0.0
    assert arm.slopes.detach().tolist()==[0.0,0.0]
    a,e,y=r51.batch(facts,labels,list(range(r51.BATCH)))
    assert torch.equal(arm(a,e,zero_salience=True),raw(a,e))
    loss=torch.nn.functional.cross_entropy(-arm(a,e),y)
    loss.backward()
    assert arm.slopes.grad is not None
    assert bool(torch.isfinite(arm.slopes.grad).all())
    assert arm.salience.grad is not None
    assert bool(torch.isfinite(arm.salience.grad).all())


def test_native_full_history_adapter_no_downsampling():
    model=r51.R51FullHistoryMemory("native_factual_memory",r51.SEEDS[0])
    for p,length in (("fit",18),("cal",26),("ood",46)):
        a,e=r51.candidates(r51.world(p,0,1,1,1))
        assert a.shape==(2,length,16)
        for ablate in (False,True):
            vals=model(a,e,ablate_backbone=ablate)
            assert vals.shape==(2,)
            assert bool(torch.isfinite(vals).all())
    with pytest.raises(ValueError,match="UNREGISTERED_HISTORY"):
        model(torch.zeros((2,35,16)),torch.zeros((2,35,16)))


def test_actual_frozen_two_seed_training_ablations_and_original_cold_restore():
    out=r51.run()
    assert out["schema"]==r51.SCHEMA
    assert out["freeze_blob"]==r51.FREEZE_BLOB
    assert out["parent"]==r51.PARENT
    assert out["factual_FIT_selected_returns"]==384
    assert out["FIT_training_pairs"]==128
    assert out["FIT_heldout_pairs"]==64
    assert len(out["models"])==12
    assert out["neural_optimizer_updates"]==3072
    assert len(out["judgment"]["seeds"])==2
    assert not out["judgment"]["independent_scientific_pass"]
    assert not out["independent_scientific_pass"]
    assert out["source_neutral_DEV_only"]
    assert out["deployment"]=="NONE"
    assert all(z["mistakes"]==0 for z in out["observable_reference"].values())
    assert all(z["max_paired_full_B_mean_difference"]<=1e-7
               for z in out["B_parity"].values())
    for model in out["models"]:
        assert model["actual_updates"]==256
        assert model["cold_restore"]
        assert 0<=model["productive_gradient_steps"]<=256
        assert model["trainable_parameters"]>0
        assert [t["step"] for t in model["traces"]]==list(r51.CHECKPOINTS)
        assert len(model["scores"]["cal"]["active"]["choices"])==64
        assert len(model["scores"]["ood"]["active"]["choices"])==64
        for key in ("erase_B","erase_tag","erase_fifth","zero_salience",
                    "force_uniform","force_top_two_tag","zero_native",
                    "zero_causal"):
            assert len(model["scores"]["ood"][key]["choices"])==64
    for s in out["judgment"]["seeds"]:
        assert s["seed"] in r51.SEEDS
        assert isinstance(s["frozen_DEV_pass"],bool)
    r51.retain_ci_receipt(out)
