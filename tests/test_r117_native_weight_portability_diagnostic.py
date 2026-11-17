"""R117 FIT/VALID ONLY optimizer-rounding portability tests.

No original R116 UNIT_UNSEEN world may be created even to test a model.
"""
import numpy as np
import pytest
import torch

from experiments import r115_query_targeted_diagnostic as r115
from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r116_pair_source as pair
from experiments import r117_source_custody as custody
from experiments import r117_native_weight_portability_diagnostic as r117


def test_r117_prospective_freeze_and_no_reused_original_heldout(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="9e342ddcbec2122c8fc6486ab1b83bd21de7e827"
    assert f["stage"]=="FROZEN_PREIMPLEMENTATION_FIT_VALID_ONLY_NO_ORIGINAL_HELDOUT_FIRSTUSE"
    assert f["source"]["R116_original_UNIT_UNSEEN_spent"] is True
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert tuple(f["diagnostics"]["candidate_integer_scales_descending"])==custody.SCALES
    assert tuple(f["diagnostics"]["snapshot_native_raw_FP32_at_updates"])==custody.CHECKPOINTS
    assert f["evaluation"]["scientific_pass_from_this_study"] is False
    assert not any(f["authority"].values())
    assert custody.permit_public_panel("UNIT_FIT")=="UNIT_FIT"
    assert custody.permit_public_panel("UNIT_VALID")=="UNIT_VALID"
    for panel in ("UNIT_UNSEEN","R116_consumed","protected","original_scientific"):
        with pytest.raises(PermissionError,match="FIRST_USE_DENIED"):
            custody.permit_public_panel(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r117_Q384_generic_quantizer_rounds_same_original_native_3409_values():
    torch.set_num_threads(1)
    native=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    original,orig_sha=r115.quantize_native(native)
    test,sha,values,margin=r117.quantized_layers(native,384)
    assert original==test
    assert len(orig_sha)==len(sha)==64
    assert len(values)==3409
    assert all(isinstance(x,int) for x in values)
    assert 0<=margin["nearest_half_integer_margin"]<=.5
    assert sha==r117.quantized_layers(native,384)[1]
    with pytest.raises(PermissionError,match="UNFROZEN_INTEGER_SCALE"):
        r117.quantized_layers(native,123)


def test_r117_requantized_integer_scores_are_original_Q384_native_scores_on_public_feedback():
    torch.set_num_threads(1)
    native=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    original,_=r115.quantize_native(native)
    new,_,_,_=r117.quantized_layers(native,384)
    for facts in ([(8,1),(9,-1),(7,1)],
                  [(8,1),(9,-1),(7,-1)],
                  [(8,1),(9,-1),(7,1),(12,-1)]):
        assert r117.scaled_native_scores(new,384,facts)==r115.native_scores(
            original,facts,(12,14),15,0)
    for scale in custody.SCALES:
        layers,_,_,_=r117.quantized_layers(native,scale)
        result=r117.scaled_native_scores(layers,scale,[(8,1),(9,-1),(7,1)])
        assert len(result)==2 and all(type(v) is int for v in result)
    with pytest.raises(PermissionError):
        r117.scaled_native_scores(new,384,[(8,1),(9,-1)],offers=(1,7))
    with pytest.raises(RuntimeError,match="LAYER_SHAPE"):
        r117.scaled_native_scores(new[:2]+[{"weight":[[0]],"bias":[0]}],
                                  384,[(8,1),(9,-1),(7,1)])


def test_r117_model_snapshot_fp32_does_not_mutate_weights():
    torch.set_num_threads(1)
    model=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    original=r117.raw_fp32_sha256(model)
    assert original==r117.raw_fp32_sha256(model)
    assert len(original)==64
    with torch.no_grad():
        model.scorer[0].bias[0]+=1.
    assert r117.raw_fp32_sha256(model)!=original


def test_r117_fit_only_original_512_update_training_and_all_checkpoints():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,_=r116.make_fit(seed)
    gen=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=gen.integers(0,len(fit[-1]),size=(512,16))
    models=r116.new_models(seed)
    receipt,states=r117.original_native_fit_train(models[r116.NATIVE],fit,batches)
    assert receipt["updates"]==512 and receipt["parameters"]==3409
    assert set(states)==set(str(x) for x in custody.CHECKPOINTS)
    assert states["0"]!=states["1"]
    assert states["1"]!=states["512"]
    with pytest.raises(PermissionError):
        r117.original_native_fit_train(models[r116.NATIVE],fit,batches[:100])


def test_r117_public_VALID_only_intervention_budget_and_reproducible_96_queries():
    torch.set_num_threads(1)
    native=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    layers,_=r115.quantize_native(native)
    a=r117.public_VALID_trial(custody.SEEDS[0],384,layers)
    b=r117.public_VALID_trial(custody.SEEDS[0],384,layers)
    assert a==b
    assert len(a["next_actions"])==96
    assert set(a["next_actions"])<={12,14}
    assert a["correct"] in range(97)
    assert 0<=a["accuracy"]<=1
    assert len(a["all_chosen_actions_sha256"])==64
    assert len(a["all_physical_outcome_signs_sha256"])==64
    assert len(a["all_correctness_sha256"])==64
    with pytest.raises(PermissionError,match="SEALED_UNSEEN"):
        pair.PhysicalPairWorld("UNIT_UNSEEN",custody.SEEDS[0],0)


def test_r117_independent_process_cold_integer_scores_across_frozen_candidate_scales():
    torch.set_num_threads(1)
    native=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    models={q:r117.quantized_layers(native,q)[0] for q in custody.SCALES}
    assert r117.cold_original_integer_score_check(models)==0


def test_r117_scoring_function_and_test_source_never_request_original_unseen():
    assert "UNIT_UNSEEN" not in r117.public_VALID_trial.__code__.co_consts
    assert "UNIT_UNSEEN" not in r117.run_seed.__code__.co_consts
    assert r117.SCALES==custody.SCALES
    assert r117.RECEIPT.name=="r117_fit_valid_native_weight_portability.json"
