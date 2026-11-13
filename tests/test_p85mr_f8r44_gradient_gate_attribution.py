"""F8-R44 prospective source-neutral, causal salience training and negatives."""
import pytest
import torch

from experiments import p85mr_f8r44_gradient_gate_attribution as r44


def test_frozen_r44_custody_and_external_source_denial():
    f=r44.locked()
    assert f["lineage"]["parent_sha"]==r44.PARENT
    assert f["source"]["salts"]==r44.SALT
    assert len(r44.ARMS)==10
    assert r44.TRACE_STEPS==(1,32,64,128,192,256)
    assert f["frozen_score"]["independent_scientific_pass"] is False
    assert f["authority"]["spent_F7_replay"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_DENIED"):
        r44.refuse_external_original_firstuse("no-source")
    with pytest.raises(ValueError,match="UNREGISTERED_PANEL"):
        r44.worlds("external")
    with pytest.raises(ValueError,match="UNREGISTERED_WORLD"):
        r44.world("ood",16,0,1,1)
    with pytest.raises(ValueError,match="UNREGISTERED_MODEL"):
        r44.model_for("unregistered",985201,None)


def test_fresh_selected_B_and_fifth_isolation_and_ood_length():
    for g in range(16):
        for c in (0,1):
            for sign in (-1,1):
                a=r44.world("fit",g,c,sign,1)
                b=r44.world("fit",g,c,sign,-1)
                assert len(a["b_history"])==8
                assert len(r44.world("ood",g,c,sign,1)["b_history"])==16
                x,y=r44.candidates(a)
                xx,yy=r44.candidates(b)
                assert x.shape==y.shape==(2,14,16)
                assert torch.equal(x[:,:5,:],xx[:,:5,:])
                assert torch.equal(y[:,:5,:],yy[:,:5,:])
                assert not torch.equal(y[:,5:-1,:],yy[:,5:-1,:])
                assert torch.equal(y[:,-1,:],torch.zeros_like(y[:,-1,:]))
                erased=r44.candidates(a,erase_B=True)[1]
                assert torch.any(y[:,5:-1,:]!=erased[:,5:-1,:])
                swapped=r44.candidates(a,swap_B=True)[1]
                assert torch.equal(swapped[:,5:-1,:],yy[:,5:-1,:])
                ood=r44.candidates(r44.world("ood",g,c,sign,1))
                assert ood[0].shape==ood[1].shape==(2,22,16)


def test_r44_true_goal_is_factual_selected_post_action_state():
    fit,labels=r44.factual_fit()
    assert len(fit)==768 and len(labels)==384
    assert labels.count(0)==labels.count(1)==192
    for panel in ("fit","cal","ood"):
        for g in (0,1):
            for c in (0,1):
                for sign in (-1,1):
                    for gate in (-1,1):
                        w=r44.world(panel,g,c,sign,gate)
                        costs=[]
                        for action in (0,1):
                            following=r44._return_following(w,action)
                            selected=r44.r33.b._record(
                                w["return_state"],action,following,c)
                            observed=w["return_state"]+r44.r31._physical_delta(
                                selected[1][0])
                            assert observed==pytest.approx(following,abs=1e-6)
                            costs.append((observed-w["goal"])**2)
                        for action in (0,1):
                            assert r44.regret(w,action)==pytest.approx(
                                max(0.,costs[action]-min(costs)),abs=1e-6)
    with pytest.raises(ValueError,match="INVALID_RETURN_ACTION"):
        r44.regret(r44.world("cal",0,0,1,1),4)


def test_full_two_seed_mechanism_real_training_and_fail_closed_science():
    out=r44.run()
    assert out["schema"]==r44.SCHEMA
    assert out["freeze_blob"]==r44.FREEZE_BLOB
    assert out["parent"]==r44.PARENT
    assert out["factual_fit_episodes"]==768
    assert out["factual_train_pairs"]==256
    assert out["factual_heldout_pairs"]==128
    assert out["original_binder_replay_updates"]==1024
    assert out["new_neural_updates"]==2560
    assert len(out["models"])==10
    assert out["source_neutral_development_only"]
    assert not out["spent_external_firstuse"]
    assert not out["independent_scientific_pass"]
    assert out["deployment"]=="NONE"
    assert set(out["observational_reference"]["panels"])=={"cal","ood"}
    assert not out["observational_reference"]["evaluator_marker_used"]
    assert not out["judgment"]["whole_process_compute_matched"]
    assert len(out["judgment"]["per_seed"])==2
    for row in out["models"]:
        assert row["optimizer_updates"]==256
        assert row["cold_restore"]
        assert row["trainable_parameters"]>0
        assert len(row["trace"])==6
        assert [x["update"] for x in row["trace"]]==list(r44.TRACE_STEPS)
        assert all(x["all_finite"] for x in row["trace"])
        assert all(0 <= x["heldout_fit_accuracy"]<=1 for x in row["trace"])
        assert set(row["panels"])=={"cal","ood"}
        assert len(row["panels"]["cal"]["active"]["choices"])==128
        assert len(row["panels"]["ood"]["active"]["choices"])==128
        for control in ("erase_B","swap_B","erase_fifth","zero_backbone",
                        "zero_gate","zero_binder","zero_content"):
            assert len(row["panels"]["ood"][control]["choices"])==128
        if "multiplicative" in row["arm"]:
            assert row["panels"]["ood"]["uniform_pool"] is not None
        else:
            assert row["panels"]["ood"]["uniform_pool"] is None
    r44.retain_ci_receipt(out)
