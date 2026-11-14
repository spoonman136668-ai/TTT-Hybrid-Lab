"""Frozen R54 independent cold-transfer falsifier of R53 learned reliability."""
import pytest
import torch

from experiments import p85mr_f8r54_cold_repeatability_transfer as r54


def test_freeze_preexists_new_generator_and_original_firstuse_denied():
    f=r54.locked()
    assert r54.git_blob(r54.FREEZE.read_bytes())==r54.FREEZE_BLOB
    assert f["stage"]=="FROZEN_BEFORE_NEW_GENERATOR_OR_ANY_R54_SCORE"
    assert f["parent"]["qualified_R53_SHA"]==r54.PARENT
    assert f["source"]["new_salts"]==r54.SALT
    assert f["source"]["no_training_on_new_OOD"]
    assert f["source"]["new_data_separate_from_spent_R53_R52_R51_R50_R49"]
    assert len(r54.ARMS)==14
    assert f["original_replay"]["total_replayed_fit_steps"]==3584
    assert not any(f["authority"].values())
    assert not f["frozen_decision"]["independent_scientific_pass"]
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r54.reject_external_original_firstuse()
    with pytest.raises(ValueError,match="UNREGISTERED_SHIFT_GROUP"):
        r54.world("oscillating",8,0,1,1)
    with pytest.raises(ValueError,match="INVALID_FACTOR"):
        r54.world("oscillating",0,2,1,1)


def test_new_independent_source_salts_pairing_and_no_gate_leak():
    for panel in r54.PANELS:
        parity=r54.source_parity(panel)
        assert parity["groups"]==8
        assert parity["maximum_paired_full_B_mean_difference"]<=1e-7
        assert len(r54.worlds(panel))==64
        for group in range(8):
            cue,decoy,conf1,conf2=r54.positions(panel,group)
            assert len({cue,decoy,conf1,conf2})==4
            for ctx in (0,1):
                for gate in (-1,1):
                    w=r54.world(panel,group,ctx,1,gate)
                    rows=r54.selected_history(w)
                    assert len(rows)==96
                    assert rows[cue*6][0]==(.42 if ctx==0 else .18)
                    assert rows[decoy*6][0]==(.18 if ctx==0 else .42)
                    assert rows[conf1*6][0]==.88
                    assert rows[conf2*6][0]==.77
                    for src in range(16):
                        six=rows[src*6:(src+1)*6]
                        assert [x[1] for x in six]==[0,1,0,1,0,1]
                        assert len({x[0] for x in six})==1
                    a,e=r54.candidates(w)
                    assert a.shape==e.shape==(2,102,16)
                    assert not torch.any(e[:,-1])
                    assert not torch.any(e[:,5:-1,4])
                    tag,mean,variance,rel,magnitude,ratio,fifth,c,obs=(
                        r54.r53.public_repeated_features(
                            a,e,torch.tensor(0.)))
                    assert tag.shape==mean.shape==variance.shape==(1,16)
                    assert c.tolist()==[ctx]
                    assert float(mean[0,cue])==pytest.approx(.12*gate,abs=1e-6)
                    assert float(mean[0,decoy])==pytest.approx(-.12*gate,abs=1e-6)
                    if panel=="stationary_confounder":
                        assert float(variance[0,conf1])==pytest.approx(0.,abs=1e-10)
                        assert float(variance[0,conf2])==pytest.approx(0.,abs=1e-10)
                        assert abs(float(mean[0,conf1]))>abs(float(mean[0,cue]))
                    else:
                        assert float(variance[0,conf1])>0
                        assert float(variance[0,conf2])>0


def test_public_only_feasibility_on_both_new_ood_shifts_without_hidden_labels():
    fit,labels=r54.r53.factual_fit()
    assert len(fit)==384 and len(labels)==192
    offset=r54.r53.calibration(fit)
    result=r54.public_tag_reference(offset)
    assert result.keys()==set(r54.PANELS)
    assert all(v["worlds"]==64 and v["mistakes"]==0
               and v["handcoded_public_tag_identity"] for v in result.values())


def test_unmodified_original_R53_FIT_policy_replayed_with_cold_restore():
    f=r54.locked()
    fit,labels=r54.r53.factual_fit()
    offset=r54.r53.calibration(fit)
    draws=r54.original_draws()
    assert len(draws)==256 and all(len(x)==16 for x in draws)
    main,receipt=r54.reproduce_original_FIT(
        "learned_repeatability",985201,offset,fit,labels,draws,f)
    assert receipt["original_FIT_heldout"]==1.0
    assert receipt["original_R53_OOD_regret"]==0.
    assert receipt["productive_gradient_steps"]==256
    assert receipt["trainable_parameters"]==4
    assert len(receipt["replayed_weight_hash"])==64
    assert receipt["original_FIT_replayed_optimizer_updates"]==256
    assert receipt["replayed_beta"]>0
    for panel in r54.PANELS:
        test=r54.score(main,panel)
        assert len(test["choices"])==64
        assert test["gate_pair_flips"]<=32
        assert test["mean_regret"]>=0.


def test_full_original_FIT_replay_new_transfer_preserves_negative_result():
    result=r54.run()
    assert result["schema"]==r54.SCHEMA
    assert result["freeze_blob"]==r54.FREEZE_BLOB
    assert result["parent"]==r54.PARENT
    assert result["source_neutral_synthetic_cold_transfer_only"]
    assert result["original_R53_FIT_selected_factual_returns"]==384
    assert result["original_R53_FIT_train_pairs"]==128
    assert result["original_R53_FIT_heldout_pairs"]==64
    assert result["original_R53_FIT_replayed_optimizer_steps"]==3584
    assert len(result["original_R53_models"])==14
    assert len(result["judgment"]["seeds"])==2
    assert result["independent_scientific_pass"] is False
    assert result["judgment"]["independent_scientific_pass"] is False
    assert result["external_original_first_use"] is False
    assert result["deployment"]=="NONE"
    for panel in r54.PANELS:
        assert result["new_panel_source_parity"][panel][
            "maximum_paired_full_B_mean_difference"]<=1e-7
        assert result["positive_observation_reference"][panel]["mistakes"]==0
    for m in result["original_R53_models"]:
        assert m["cold_restore"]
        assert m["original_FIT_replayed_optimizer_updates"]==256
        assert 0<=m["productive_gradient_steps"]<=256
        assert m["trainable_parameters"]>0
        assert len(m["original_R53_OOD_choices"])==64
        for panel in r54.PANELS:
            assert set(m["new_panels"][panel])=={
                "active","zero_beta","zero_native","erase_tag"}
            for scored in m["new_panels"][panel].values():
                assert len(scored["choices"])==64
    for seed in result["judgment"]["seeds"]:
        assert seed["seed"] in r54.SEEDS
        assert isinstance(seed["full_cold_transfer_pass"],bool)
        assert set(seed["panels_pass"])==set(r54.PANELS)
    r54.retain_ci_receipt(result)
