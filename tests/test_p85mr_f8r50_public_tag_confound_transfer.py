"""R50 prospectively frozen public-tag OOD confound; R49 fit-only replay."""
import pytest
import torch

from experiments import p85mr_f8r49_public_event_selection as r49
from experiments import p85mr_f8r50_public_tag_confound_transfer as r50


def test_original_R49_parent_and_new_freeze_not_movable():
    f=r50.locked()
    assert r50.FREEZE_BLOB==r50.git_blob(r50.FREEZE.read_bytes())
    assert f["stage"]=="FROZEN_BEFORE_GENERATOR_CODE_OR_ANY_R50_SCORE"
    assert f["source"]["new_ood_salt"]==22490011
    assert f["source"]["selected_B_length"]==32
    assert f["parent"]["source_head"]==r50.PARENT
    assert f["frozen_decision"]["independently_qualified_scientific_gain"] is False
    assert len(r50.worlds())==64
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE_UNAUTHORIZED"):
        r50.refuse_external_firstuse("original-source")
    with pytest.raises(ValueError,match="UNREGISTERED_GROUP"):
        r50.world(8,0,1,1)
    with pytest.raises(ValueError,match="UNREGISTERED_FACTOR"):
        r50.world(0,2,1,1)


def test_independent_tag_confounds_are_public_and_gate_balanced():
    source=r50.source_parity()
    assert source["groups"]==8
    assert source["contexts"]==2
    assert source["all_keys_equal_opposite_gate"]
    assert source["maximum_opposite_gate_B_mean_difference"]<=1e-7
    for group in range(r50.GROUPS):
        ids=r50.positions(group)
        assert len(ids)==len(set(ids))==4
        for ctx in (0,1):
            for gate in (-1,1):
                w=r50.world(group,ctx,1,gate)
                bs=r50.B_selected(w)
                assert len(bs)==32
                cue,decoy,hi1,hi2=ids
                assert bs[hi1][0]==.88
                assert bs[hi2][0]==.77
                assert bs[hi1][2]==.015
                assert bs[hi2][2]==-.015
                assert bs[cue][0]==(.42 if ctx==0 else .18)
                assert bs[decoy][0]==(.18 if ctx==0 else .42)
                assert bs[cue][2]==pytest.approx(.12*gate)
                assert bs[decoy][2]==pytest.approx(-.12*gate)
                a,e=r50.candidates(w)
                assert a.shape==e.shape==(2,38,16)
                assert not bool(torch.any(e[:,-1,:]))
                assert not bool(torch.any(e[:,5:-1,4]))
                # All tied public tag keys erase to zero but selected physical
                # consequences remain, independent of hidden generator data.
                erased_a,erased_e=r50.candidates(w,erase_tag=True)
                tag,delta,_,context,_=r49.components(
                    erased_a,erased_e,torch.tensor(0.))
                assert torch.count_nonzero(tag)==0
                assert context.item()==ctx
                assert delta.shape==(1,32)


def test_public_factual_reference_solves_all_new_shifted_worlds_without_secret():
    records,labels=r49.factual_fit()
    assert len(records)==384 and len(labels)==192
    offset=r49.calibration(records)
    ref=r50.positive_reference(offset)
    assert ref["worlds"]==64
    assert ref["mistakes"]==0
    assert not ref["uses_hidden_cue_indices"]
    assert ref["handcoded_physical_effect_magnitude_threshold"]==.06
    # This positive control is deliberately not a neural model.


def test_original_R49_seed_weights_reproduced_without_OOD_training():
    freeze=r50.locked()
    records,labels=r49.factual_fit()
    offset=r49.calibration(records)
    draws=r50.original_draws()
    assert len(draws)==256
    assert all(len(x)==16 for x in draws)
    model,receipt=r50.reproduce_R49_frozen(
        "event_context_selector",985201,offset,records,labels,draws,freeze)
    assert receipt["R49_original_heldout_FIT"]==1.
    assert receipt["productive_gradient_steps"]==256
    assert receipt["trained_parameters"]==3
    assert receipt["reproduced_R49_FIT_updates"]==256
    expected=freeze["source"]["original_R49_seed_slope_repro_target"]["seed985201"]
    assert receipt["context_slopes"]==pytest.approx(expected,abs=.002)
    assert len(receipt["trained_weights_sha256"])==64
    eval_snapshot=r50.evaluate(model)
    assert len(eval_snapshot["choices"])==64
    assert eval_snapshot["gate_pair_flips"]<=32
    assert eval_snapshot["mean_regret"]>=0.


def test_full_source_neutral_blind_shift_qualifies_and_records_negative_result():
    result=r50.run()
    assert result["schema"]==r50.SCHEMA
    assert result["freeze_blob"]==r50.FREEZE_BLOB
    assert result["parent"]==r50.PARENT
    assert result["source_neutral_new_panel_only"]
    assert result["original_R49_FIT_only_training"]
    assert result["original_R49_FIT_factual_selected_returns"]==384
    assert result["new_R50_synthetic_OOD_worlds"]==64
    assert result["original_R49_reproduction_optimizer_updates"]==2048
    assert result["observational_source_parity"]["maximum_opposite_gate_B_mean_difference"]<=1e-7
    assert result["observable_positive_reference"]["mistakes"]==0
    assert len(result["models"])==8
    assert len(result["judgment"]["seeds"])==2
    assert not result["judgment"]["independent_scientific_pass"]
    assert not result["independent_scientific_pass"]
    assert not result["external_original_firstuse"]
    assert result["deployment"]=="NONE"
    for model in result["models"]:
        assert model["cold_restore"]
        assert model["reproduced_R49_FIT_updates"]==256
        assert model["trained_parameters"]>0
        assert set(model["transfer"])=={
            "active","erase_B","erase_tag","erase_fifth","zero_native"}
        assert len(model["transfer"]["active"]["choices"])==64
        assert all(len(x["choices"])==64 for x in model["transfer"].values())
    for seed in result["judgment"]["seeds"]:
        assert seed["seed"] in r50.SEEDS
        assert seed["R49_original_FIT_heldout"]==1.
        assert isinstance(seed["transfer_conditions_pass"],bool)
        assert 0<=seed["OOD_gate_pair_flips"]<=32
    r50.retain_ci_receipt(result)
