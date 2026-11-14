"""R57 source-only falsification: reject easy OOD without model training."""
from experiments import p85mr_f8r57_source_admission_audit as audit


def test_frozen_R57_OOD_has_disqualifying_single_cue_shortcut():
    result=audit.source_only_audit()
    assert result["freeze_blob"]==audit.FREEZE_BLOB
    assert result["source_only_not_model_scoring"]
    assert result["single_contrast_OOD_accuracy"]==1.0
    assert result["frozen_max_single_contrast_accuracy"]==0.749
    assert not result["frozen_task_admissible"]
    assert not result["training_new_learning_mechanism_authorized"]
    assert {r["law_index"] for r in result["law_results"]}=={3,4}
    assert all(r["single_largest_correct"] for r in result["law_results"])
    assert sum(r["new_ood_cases"] for r in result["law_results"])==128
