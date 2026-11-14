"""F8-R56 prospective cold delayed-goal-proxy shift without any new OOD adaptation."""
import pytest
import torch
from experiments import p85mr_f8r56_cold_delayed_proxy_shift as r56


def test_prospective_frozen_protocol_and_original_first_use_denial():
    f=r56.locked()
    assert f["stage"]=="FROZEN_BEFORE_R56_GENERATOR_OR_SCORED_RESULTS"
    assert r56.git_blob(r56.FREEZE.read_bytes())==r56.FREEZE_BLOB
    assert f["parent"]["qualified_R55_SHA"]==r56.PARENT
    assert f["source"]["salts"]==r56.SALT
    assert f["source"]["no_new_OOD_training"]
    assert f["source"]["no_old_R49_R55_scored_episodes"]
    assert len(r56.ARMS)==16
    assert f["replay"]["total_replayed_optimizer_attempts"]==4096
    assert not any(f["authority"].values())
    assert f["frozen_decision"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r56.reject_external_original_firstuse("original")
    with pytest.raises(ValueError,match="UNREGISTERED_SHIFT_GROUP"):
        r56.world("control",8,0,1,1)
    with pytest.raises(ValueError,match="INVALID_FACTOR"):
        r56.world("control",0,2,1,1)


def test_all_six_selected_upstream_and_goal_sensor_observations_each_source():
    for panel in r56.PANELS:
        parity=r56.source_parity(panel)
        assert parity["groups"]==8
        assert parity["maximum_paired_full_B_mean_difference"]<=1e-7
        assert len(r56.worlds(panel))==64
        for group in range(8):
            cue,decoy,sp1,sp2=r56.positions(panel,group)
            assert len({cue,decoy,sp1,sp2})==4
            for ctx in (0,1):
                for gate in (-1,1):
                    w=r56.world(panel,group,ctx,1,gate)
                    rows=r56.selected_history(w)
                    assert len(rows)==96
                    for source in range(16):
                        six=rows[6*source:6*(source+1)]
                        assert [x[1] for x in six]==[0,0,0,1,1,1]
                        assert six[0][0]==six[3][0]
                        assert six[1][0]==six[4][0]==r56.r55.NOW_STATE
                        assert six[2][0]==six[5][0]==r56.r55.DELAY_STATE
                    assert rows[6*cue][0]==(.42 if ctx==0 else .18)
                    assert rows[6*decoy][0]==(.18 if ctx==0 else .42)
                    assert rows[6*sp1][0]==.88
                    assert rows[6*sp2][0]==.77
                    a,e=r56.candidates(w)
                    assert a.shape==e.shape==(2,102,16)
                    assert not torch.any(e[:,-1])
                    assert not torch.any(e[:,5:-1,4])
                    tag,up,immediate,delay,fifth,c,obs=(
                        r56.r55.observed_goal_features(a,e,torch.tensor(0.)))
                    assert tag.shape==delay.shape==(1,16)
                    assert c.tolist()==[ctx]
                    assert float(delay[0,cue])==pytest.approx(.12*gate,abs=1e-6)
                    assert float(delay[0,decoy])==pytest.approx(-.12*gate,abs=1e-6)
                    if panel=="control":
                        assert abs(float(delay[0,sp1]))<1e-7
                        assert abs(float(delay[0,sp2]))<1e-7
                    else:
                        assert float(delay[0,sp1])==pytest.approx(.50,abs=1e-6)
                        assert float(delay[0,sp2])==pytest.approx(-.48,abs=1e-6)
                    assert float(up[0,sp1])==pytest.approx(.45,abs=1e-6)
                    assert float(immediate[0,sp2])==pytest.approx(-.52,abs=1e-6)


def test_source_only_public_feasibility_both_fresh_panels():
    fit,labels=r56.r55.factual_fit()
    assert len(fit)==384 and len(labels)==192
    offset=r56.r55.calibration(fit)
    ref=r56.public_tag_reference(offset)
    assert set(ref)==set(r56.PANELS)
    assert all(x["worlds"]==64 and x["mistakes"]==0 and
               x["handcoded_FIT_known_tag_windows"] for x in ref.values())


def test_replay_original_4096_FIT_only_steps_and_score_no_new_update():
    result=r56.run()
    assert result["schema"]==r56.SCHEMA
    assert result["freeze_blob"]==r56.FREEZE_BLOB
    assert result["parent"]==r56.PARENT
    assert result["source_neutral_synthetic_cold_transfer_only"]
    assert result["original_R55_FIT_selected_factual_returns"]==384
    assert result["original_R55_FIT_train_pairs"]==128
    assert result["original_R55_FIT_heldout_pairs"]==64
    assert result["original_R55_FIT_replayed_optimizer_steps"]==4096
    assert len(result["original_R55_models"])==16
    assert len(result["judgment"]["seeds"])==2
    assert result["independent_scientific_pass"] is False
    assert result["judgment"]["independent_scientific_pass"] is False
    assert result["external_original_first_use"] is False
    assert result["deployment"]=="NONE"
    for panel in r56.PANELS:
        assert result["new_panel_source_parity"][panel][
            "maximum_paired_full_B_mean_difference"]<=1e-7
        assert result["positive_observation_reference"][panel]["mistakes"]==0
    for row in result["original_R55_models"]:
        assert row["cold_restore"]
        assert row["original_FIT_replayed_optimizer_updates"]==256
        assert row["trainable_parameters"]>0
        assert 0<=row["productive_gradient_steps"]<=256
        assert len(row["original_R55_OOD_choices"])==64
        for panel in r56.PANELS:
            assert set(row["new_panels"][panel])=={
                "active","zero_beta","zero_native","erase_tag"}
            for scored in row["new_panels"][panel].values():
                assert len(scored["choices"])==64
    for seed in result["judgment"]["seeds"]:
        assert seed["seed"] in r56.SEEDS
        assert isinstance(seed["full_cold_transfer_pass"],bool)
        assert set(seed["panels_pass"])==set(r56.PANELS)
    r56.retain_ci_receipt(result)
