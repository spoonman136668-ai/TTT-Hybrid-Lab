"""R118 FIT/VALID-only exact initial-state and optimizer negative controls."""
import hashlib
import numpy as np
import pytest
import torch
from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r116_pair_source as original
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_source_custody as custody
from experiments import r118_shared_init_optimizer_diagnostic as r118


def test_r118_frozen_source_boundaries_and_immutable_protocol(monkeypatch):
    freeze=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="567f553bc260183508627714971eee90f55f0176"
    assert freeze["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_DIAGNOSTIC_NO_SCIENTIFIC_FIRST_USE"
    assert freeze["lineage"]["parent_R117_exact_SHA"]=="8a0da2dc2f907ce342bcfcd21bf1970e8c44a1c3"
    assert tuple(freeze["source"]["seeds"])==custody.SEEDS
    assert tuple(freeze["diagnostic"]["all_native_raw_snapshot_update_indices"])==custody.CHECKPOINTS
    assert not any(freeze["authority"].values())
    assert freeze["evaluation"]["scientific_acceptance"] is False
    for panel in ["UNIT_UNSEEN","R116_original_spent","protected","original_scientific"]:
        with pytest.raises(PermissionError,match="FIRST_USE_DENIED"):
            custody.allow(panel)
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    assert custody.allow("UNIT_VALID")=="UNIT_VALID"
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r118_canonical_bit_grid_does_not_depend_on_torch_rng():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    torch.manual_seed(seed+1)
    a=r116.new_models(seed)
    torch.manual_seed(seed+2)
    b=r116.new_models(seed)
    for arm in r116.TRAINABLE:
        first,count=r118.seed_shared_initial_bytes(a[arm],seed,arm)
        second,count2=r118.seed_shared_initial_bytes(b[arm],seed,arm)
        assert count==count2 and first==second
        assert r118._parameter_bytes(a[arm])==r118._parameter_bytes(b[arm])
        assert count>100
    with pytest.raises(PermissionError,match="INVALID_CANONICAL_INITIAL_MODEL"):
        r118.seed_shared_initial_bytes(a[r116.NATIVE],1150011,r116.NATIVE)


def test_r118_exact_F32_canonical_generator_changes_with_public_seed_arm_name_index():
    seed=custody.SEEDS[0];arm=r116.NATIVE
    original_val=r118._pure_canonical(seed,arm,"scorer.0.weight",0)
    assert -0.125<=original_val<0.125
    assert original_val*262144==int(original_val*262144)
    # Reference explicitly implements the FROZEN SHA256 key; no numpy/torch RNG.
    b=hashlib.sha256(
        f"R118|V1|{seed}|{arm}|scorer.0.weight|0".encode("utf8")).digest()
    assert original_val==(int.from_bytes(b[:2],"little")-32768)/262144
    assert original_val!=r118._pure_canonical(seed+12,arm,"scorer.0.weight",0)
    assert original_val!=r118._pure_canonical(seed,arm,"scorer.0.bias",0)


def test_r118_native_initial_snapshot_matches_fresh_reference_and_original_shapes():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    initial,n=r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    assert n==3409
    fresh=r118._canonical_native_check_against_initial(seed)
    assert r118._parameter_bytes(fresh)==(initial,3409)
    assert r117.raw_fp32_sha256(fresh)==r117.raw_fp32_sha256(model)
    q384,sha=r116.r115.quantize_native(model)
    assert len(sha)==64 and len(q384)==3
    assert r116.r115.integer_cold_restore(q384)==0


def test_r118_actual_public_FIT_only_512_original_AdamW_updates_source_preserved():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,_=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=rng.integers(0,256,size=(512,16))
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    receipt,snaps=r117.original_native_fit_train(model,fit,batches)
    assert receipt["updates"]==512 and receipt["parameters"]==3409
    assert set(snaps)=={str(i) for i in custody.CHECKPOINTS}
    assert snaps["0"]!=snaps["512"]
    with pytest.raises(PermissionError,match="MINIBATCH"):
        r117.original_native_fit_train(model,fit,batches[:100])


def test_r118_public_VALID_only_and_physical_action_budget_against_unseen_guard():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    for scale in custody.SCALES:
        layers,_,_,_=r117.quantized_layers(model,scale)
        receipt=r117.public_VALID_trial(seed,scale,layers)
        assert 0<=receipt["correct"]<=96
        assert len(receipt["next_actions"])==96
        assert set(receipt["next_actions"])<={12,14}
        assert all(len(receipt[key])==64 for key in (
            "all_chosen_actions_sha256","all_physical_outcome_signs_sha256",
            "all_correctness_sha256"))
    with pytest.raises(PermissionError,match="SEALED_UNSEEN"):
        original.PhysicalPairWorld("UNIT_UNSEEN",seed,0)


def test_r118_independent_process_original_integer_scores_and_model_integrity():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    layers={q:r117.quantized_layers(model,q)[0] for q in custody.SCALES}
    assert r117.cold_original_integer_score_check(layers)==0
    assert "UNIT_UNSEEN" not in r118.run_seed.__code__.co_consts
    assert r118.RECEIPT.name=="r118_shared_canonical_init_optimizer_fit_valid.json"
