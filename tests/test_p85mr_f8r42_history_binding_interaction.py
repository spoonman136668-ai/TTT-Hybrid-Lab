"""F8-R42 prospective source-neutral falsifier: controls and genuine training."""
import pytest
import torch

from experiments import p85mr_f8r42_history_binding_interaction as r42


def test_preregistration_is_exact_source_bound_and_no_external_firstuse():
    frozen = r42.locked()
    assert frozen["parent"]["sha"] == r42.PARENT
    assert frozen["prospective_score"]["independent_scientific_pass"] is False
    assert len(r42.ARMS) == 6
    assert set(r42.PANELS) == {"dev", "ood"}
    with pytest.raises(PermissionError, match="FIRSTUSE_DENIED"):
        r42.refuse_external_firstuse("synthetic-does-not-authorize-external")


def test_new_generator_hides_gate_from_early_fifth_and_fails_closed():
    left = r42.world("fit", 0, 1, 1, 1)
    right = r42.world("fit", 0, 1, 1, -1)
    a, e = r42.factual_history(left)
    aa, ee = r42.factual_history(right)
    assert torch.equal(a, aa)
    assert torch.equal(e[:5], ee[:5])
    m = 5+r42.marker("fit",0)
    assert not torch.equal(e[m],ee[m])
    assert torch.equal(torch.cat((e[:m],e[m+1:])),
                       torch.cat((ee[:m],ee[m+1:])))
    with pytest.raises(ValueError, match="UNREGISTERED_WORLD"):
        r42.world("test",0,0,1,1)
    with pytest.raises(ValueError, match="INVALID_PANEL"):
        r42.worlds("external")
    with pytest.raises(ValueError, match="INVALID_MODEL"):
        r42.PairGate("unregistered",985201,None)


def test_factual_only_labels_and_masked_candidate_return():
    fit, labels = r42.factual_fit()
    assert len(fit) == 768 and len(labels) == 384
    assert set(labels) == {0,1}
    assert labels.count(0) == labels.count(1) == 192
    aa,ee = r42.candidates(r42.world("dev",0,0,1,1))
    assert aa.shape == ee.shape == (2,14,16)
    assert torch.equal(ee[:,-1,:],torch.zeros_like(ee[:,-1,:]))
    swapped = r42.candidates(r42.world("dev",0,0,1,-1))
    assert torch.equal(ee[:,:5,:],swapped[1][:,:5,:])


def test_frozen_factual_native_and_strong_controls_no_science_promotion():
    receipt = r42.run()
    assert receipt["schema"] == r42.SCHEMA
    assert receipt["freeze_blob"] == r42.FREEZE_BLOB
    assert receipt["parent"] == r42.PARENT
    assert receipt["original_binder_replay_updates"] == 1024
    assert receipt["new_neural_updates"] == 2304
    assert receipt["factual_fit_episodes"] == 768
    assert receipt["factual_train_pairs"] == 256
    assert receipt["factual_heldout_pairs"] == 128
    assert len(receipt["models"]) == len(r42.ARMS)
    assert receipt["independent_scientific_pass"] is False
    assert receipt["synthetic_development_only"] is True
    assert receipt["spent_external_firstuse"] is False
    assert receipt["GPU"] is False
    assert receipt["deployment"] == "NONE"
    assert receipt["judgment"]["independent_scientific_pass"] is False
    assert len(receipt["judgment"]["seeds"]) == 2
    for row in receipt["models"]:
        assert row["optimizer_updates"] == r42.STEPS
        assert row["cold_restore"] is True
        assert row["history_initial_hash"] != row["history_trained_hash"]
        assert len(row["checkpoints"]) == 2
        assert set(row["panels"]) == {"dev","ood"}
        for p in r42.PANELS:
            assert set(row["panels"][p]) == {
                "active","erase_B","swap_B","erase_fifth",
                "zero_backbone","zero_gate","zero_binder"}
            assert len(row["panels"][p]["active"]["choices"]) == 128
    for p in r42.PANELS:
        assert len(receipt["positive_diagnostic"][p]["choices"]) == 128
    r42.retain_ci_receipt(receipt)
