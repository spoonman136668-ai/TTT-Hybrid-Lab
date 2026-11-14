"""F8-R58 frozen task admission: five selected bridges, exact weak shortcuts,
two fully trained conventional baselines, negative information controls.
No post-score source or acceptance tuning and no new unconventional model.
"""
import itertools

import pytest
import torch

from experiments import p85mr_f8r58_five_bridge_task_admission as r58


def test_source_freeze_custody_and_original_first_use_boundary():
    f=r58.locked()
    assert f["schema"]==r58.SCHEMA
    assert r58.git_blob(r58.FREEZE.read_bytes())==r58.FREEZE_BLOB
    assert f["stage"]=="FROZEN_BEFORE_F58_SOURCE_MODEL_OR_SCORED_RESULTS"
    assert f["parent"]["r57_sha"]==r58.PARENT
    assert f["source"]["heldout_OOD_law_indices"]==list(r58.OOD_LAWS)
    assert f["trained_controls"]["total_optimizer_attempts"]==16384
    assert len(r58.ARMS)==4
    assert not any(f["authority"].values())
    assert f["admission"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r58.deny_external_original_firstuse("external")
    with pytest.raises(ValueError,match="INVALID_LAW"):
        r58.target_sign(32)
    with pytest.raises(ValueError,match="LAW_SPLIT_CONTAMINATION"):
        r58.selected_case("OOD",0,0)
    with pytest.raises(ValueError,match="INVALID_PANEL_OR_CASE"):
        r58.selected_case("OOD",r58.OOD_LAWS[0],16)


def test_full_exhaustive_proper_parity_and_majority_shortcut_audit():
    audit=r58.source_only_heuristics(r58.locked())
    assert audit["all_30_proper_parities_enumerated"]==30
    assert len(audit["all_shortcuts"])==32
    assert audit["OOD_law_indices"]==[4,5,6,8,13,20,25,30]
    assert audit["balanced_four_each"]
    assert audit["weak_source_shortcuts_admitted"]
    assert audit["max_shortcut_accuracy"]==.75
    assert audit["majority_accuracy"]==.625
    for signs in r58.LAWS:
        assert r58.target_sign(r58.LAWS.index(signs))==(
            signs[0]*signs[1]*signs[2]*signs[3]*signs[4])
    assert len(r58.LAWS)==32
    assert set(r58.FIT_LAWS).isdisjoint(r58.OOD_LAWS)


def test_original_selected_actions_proxy_nuisance_and_hidden_future():
    for panel in r58.PANELS:
        ids=r58.OOD_LAWS if panel=="OOD" else r58.FIT_LAWS
        for law in ids:
            for index in (0,r58.PER_LAW[panel]-1):
                case=r58.selected_case(panel,law,index)
                records=case["public_selected_records"]
                assert len(records)==20
                assert len(case["observed_model_vector"])==10
                assert set(case)=={
                    "law_index_evaluator_only","case_index_evaluator_only",
                    "panel_evaluator_only","public_selected_records",
                    "public_pairs","observed_model_vector",
                    "selected_future_return_choice_label_evaluator_only"}
                for bridge in range(5):
                    j=bridge*4
                    selected=records[j:j+4]
                    assert [z["selected_action"] for z in selected]==[0,0,1,1]
                    assert [z["public_sensor"] for z in selected]==[
                        "terminal","proxy","terminal","proxy"]
                    assert all(z["bridge"]==bridge for z in selected)
                    c=case["public_pairs"][bridge][0]
                    assert c==pytest.approx(r58.AMP[panel]*r58.LAWS[law][bridge])
                    assert abs(case["public_pairs"][bridge][1])==pytest.approx(
                        r58.PROXY[panel])
                assert case["selected_future_return_choice_label_evaluator_only"]==(
                    0 if r58.target_sign(law)==1 else 1)
        x,y=r58.cohort(panel)
        assert x.shape==(len(ids)*r58.PER_LAW[panel],10)
        assert y.shape==(len(ids)*r58.PER_LAW[panel],)
        assert int((y==0).sum())>0 and int((y==1).sum())>0


def test_model_selected_feature_interface_and_deterministic_cold_weights():
    for arm,seed in r58.ARMS:
        model=r58.Policy(arm,seed)
        x=torch.zeros((3,10),dtype=torch.float32)
        with torch.no_grad():
            y=model(x)
        assert y.shape==(3,2)
        assert bool(torch.isfinite(y).all())
        assert sum(p.numel() for p in model.parameters())>0
        with pytest.raises(ValueError,match="INVALID_SELECTED_FEATURES"):
            model(torch.zeros((3,8)))


def test_original_16k_steps_conventional_task_admission_preserves_negative():
    result=r58.run()
    assert result["schema"]==r58.SCHEMA
    assert result["freeze_blob"]==r58.FREEZE_BLOB
    assert result["parent"]==r58.PARENT
    assert result["source_neutral_task_admission_only"]
    assert result["selected_factual_observations_per_case"]==20
    assert result["cohort_sizes"]=={
        "FIT_train":576,"FIT_heldout":192,"CAL":192,"OOD":128}
    assert result["actual_optimizer_attempts"]==16384
    assert result["public_analytic_parity_oracle_accuracy"]==1.
    assert len(result["models"])==4
    assert len(result["judgment"]["original_seeds"])==2
    assert result["judgment"]["independent_scientific_pass"] is False
    assert result["judgment"]["no_novel_learning_mechanism_tested"]
    assert result["independent_scientific_pass"] is False
    assert result["new_unconventional_model_trained"] is False
    assert result["external_original_firstuse"] is False
    assert result["deployment"]=="NONE"
    for m in result["models"]:
        assert m["cold_restore"]
        assert m["training_updates"]==4096
        assert 0<=m["productive_gradient_steps"]<=4096
        assert len(m["trace"])==len(r58.CHECKPOINTS)
        assert m["trainable_parameters"]>0
        for panel,count in (
            ("FIT_train",576),("FIT_heldout",192),("CAL",192),("OOD",128)):
            assert len(m["scores"][panel]["active"]["choices"])==count
            assert 0<=m["scores"][panel]["active"]["accuracy"]<=1
        for name in ("erase_selected_terminal","erase_selected_proxy"):
            assert len(m["scores"]["OOD"][name]["choices"])==128
    for seed in result["judgment"]["original_seeds"]:
        assert seed["seed"] in r58.SEEDS
        assert isinstance(seed["task_admission_seed_pass"],bool)
    r58.retain_ci_receipt(result)
