"""F8-R45 prospective counterbalanced signal-selection source and learned control tests."""
import pytest
import torch

from experiments import p85mr_f8r45_balanced_decoy_selection as r45


def test_freeze_before_source_birth_and_deny_external_firstuse():
    f = r45.locked()
    assert f["lineage"]["parent_sha"] == r45.PARENT
    assert f["source"]["new_salts"] == r45.SALT
    assert f["prospective_score"]["independent_scientific_pass"] is False
    assert f["source"]["identification"][
        "uniform_raw_B_effect_mean_equal_for_opposite_gates"] is True
    assert "Mask its auxiliary reward" in f["source"]["identification"]["physical_observation"]
    assert len(r45.ARMS) == 8
    with pytest.raises(PermissionError, match="FIRSTUSE_DENIED"):
        r45.refuse_external_original_firstuse("external-original")
    with pytest.raises(ValueError, match="UNREGISTERED_PANEL"):
        r45.worlds("external")
    with pytest.raises(ValueError, match="UNREGISTERED_WORLD"):
        r45.world("cal", 16, 0, 1, 1)


def test_full_normalized_action_and_B_mean_parity_across_opposite_gates():
    for panel in ("fit", "cal", "ood"):
        cert = r45.parity_certificate(panel)
        assert cert["matched_groups"] == (48 if panel == "fit" else 16)
        assert cert["max_paired_mean_difference"] <= 1e-7
        for g in range(48 if panel == "fit" else 16):
            wplus = r45.world(panel, g, 0, 1, 1)
            wminus = r45.world(panel, g, 0, 1, -1)
            cue, decoy = r45.positions(panel, g)
            assert cue != decoy
            assert cue in range(16 if panel == "ood" else 8)
            assert decoy in range(16 if panel == "ood" else 8)
            a1, e1 = r45.candidates(wplus)
            a2, e2 = r45.candidates(wminus)
            assert torch.equal(a1, a2)
            assert torch.allclose(e1[:, 5:-1].mean(dim=1),
                                  e2[:, 5:-1].mean(dim=1),
                                  atol=1e-7, rtol=0)
            assert torch.equal(e1[:, 5:-1, 4], torch.zeros_like(e1[:, 5:-1, 4]))
            assert not torch.equal(e1[:, 5:-1], e2[:, 5:-1])
            assert torch.equal(e1[:, :5], e2[:, :5])
            assert torch.equal(e1[:, -1], torch.zeros_like(e1[:, -1]))
            if panel == "ood":
                assert a1.shape == e1.shape == (2,22,16)
            else:
                assert a1.shape == e1.shape == (2,14,16)


def test_true_postaction_cost_consistent_with_factual_selected_measurements():
    facts, labels = r45.factual_fit()
    assert len(facts) == 768
    assert len(labels) == 384
    assert labels.count(0) == labels.count(1) == 192
    for panel in ("fit","cal","ood"):
        for g in (0, 1):
            for c in (0,1):
                for q in (-1,1):
                    for gate in (-1,1):
                        w=r45.world(panel,g,c,q,gate)
                        costs=[]
                        for action in (0,1):
                            following=r45._following(w,action)
                            key, eff=r45.r33.b._record(
                                w["return_state"],action,following,c)
                            measured=(w["return_state"] +
                                      r45.r31._physical_delta(eff[0]))
                            assert measured == pytest.approx(following,abs=1e-6)
                            costs.append((measured-w["goal"])**2)
                        for action in (0,1):
                            assert r45.regret(w,action) == pytest.approx(
                                max(0.,costs[action]-min(costs)),abs=1e-6)


def test_attentive_content_model_and_programmatic_observable_key_boundaries():
    # The public cue tag carries the role but not the hidden sign. Erasing
    # the cue key is a controlled intervention on a selected observation.
    w = r45.world("ood", 4, 1, -1, 1)
    a,e = r45.candidates(w)
    z,ez = r45.candidates(w, erase_cue_tag=True)
    assert not torch.equal(a,z)
    assert torch.equal(e,ez)
    erase = r45.candidates(w, erase_b=True)[1]
    assert torch.count_nonzero(erase[:,5:-1,:]) == 0
    assert torch.equal(e[:,:5],erase[:,:5])
    swap = r45.candidates(w, swap_b=True)[1]
    opp = r45.candidates(r45.world("ood",4,1,-1,-1))[1]
    assert torch.equal(swap,opp)
    assert not torch.equal(swap[:,5:-1],e[:,5:-1])


def test_original_sources_and_training_frozen_preregs_not_scientific_pass():
    out = r45.run()
    assert out["schema"] == r45.SCHEMA
    assert out["freeze_blob"] == r45.FREEZE_BLOB
    assert out["parent"] == r45.PARENT
    assert out["factual_fit_episodes"] == 768
    assert out["train_pairs"] == 256
    assert out["heldout_pairs"] == 128
    assert out["reference_binder_replay_updates"] == 1024
    assert out["new_optimizer_updates"] == 2048
    assert len(out["models"]) == 8
    assert out["source_neutral_dev_only"]
    assert not out["independent_scientific_pass"]
    assert not out["external_original_firstuse"]
    assert out["deployment"] == "NONE"
    assert not out["judgment"]["independent_scientific_pass"]
    assert not out["judgment"]["whole_process_compute_matched"]
    assert len(out["judgment"]["seeds"]) == 2
    assert set(out["source_parity"]) == {"fit","cal","ood"}
    assert out["observable_tag_reference"]["selected_observable_tag_only"]
    for row in out["models"]:
        assert row["optimizer_updates"] == 256
        assert len(row["checkpoints"]) == 2
        assert row["cold_restore"]
        assert row["trainable_parameters"] > 0
        assert set(row["panels"]) == {"cal","ood"}
        assert len(row["panels"]["cal"]["active"]["choices"]) == 128
        assert len(row["panels"]["ood"]["active"]["choices"]) == 128
        for name in ("erase_B","erase_cue_tag","swap_B",
                     "erase_fifth","uniform_pool","zero_native",
                     "zero_selector","zero_binder"):
            assert len(row["panels"]["ood"][name]["choices"]) == 128
    r45.retain_ci_receipt(out)
