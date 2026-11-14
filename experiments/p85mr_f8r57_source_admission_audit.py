"""Frozen F8-R57 task-admission static falsifier, without running learners.

All three selected terminal action-contrast magnitudes are equal by protocol.
A 'single largest observed action contrast' rule therefore has a three-way tie,
resolved in public setting order (bridge0 first). The prespecified held-out
law codes happen to make bridge0 equal the hidden bridge3 return sign for
every OOD law, so this weak baseline is perfect. Do not modify a scored/frozen
task's codebook to rescue it, and do not dispatch novel learner training.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r57_intervention_task_admission_freeze.json"
FREEZE_BLOB = "caa43052c4cf74a6cae16b0e2b9033d4fa828eb1"
SCHEMA = "ttt-hybrid.f8r57-intervention-task-admission-freeze.v1"


def source_only_audit():
    raw = FREEZE.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() +
                        b"\\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F8R57_FROZEN_PROTOCOL_DRIFT")
    freeze = json.loads(raw)
    source = freeze["source"]
    criteria = freeze["admission"]
    if (freeze["schema"] != SCHEMA or
        freeze["stage"] != "FROZEN_BEFORE_NEW_GENERATOR_OR_BASELINE_TRAINING" or
        source["OOD_law_indices"] != [3, 4] or
        source["source_physical_action_contrast"]["OOD"] != .08 or
        criteria["negative_single_contrast_OOD_accuracy_max"] != .749 or
        not criteria["task_admissible_only_if_every_criterion_passes"]):
        raise ValueError("F8R57_STATIC_AUDIT_PROTOCOL_DRIFT")
    triples = list(itertools.product((-1, 1), repeat=3))
    evaluated = []
    for law_idx in source["OOD_law_indices"]:
        observed = triples[law_idx]
        future = observed[0] * observed[1] * observed[2]
        # All three absolute selected intervention contrasts equal
        # OOD amplitude 0.08, so a stable first-index argmax picks bridge0.
        single_largest = observed[0]
        evaluated.append({
            "law_index":law_idx,
            "observed_bridge_signs":list(observed),
            "future_unseen_bridge_sign":future,
            "single_largest_bridge_sign":single_largest,
            "single_largest_correct":single_largest == future,
            "new_ood_cases":source["OOD_per_law"],
        })
    accuracy = sum(row["new_ood_cases"] for row in evaluated
                   if row["single_largest_correct"]) / sum(
                       row["new_ood_cases"] for row in evaluated)
    passes = accuracy <= criteria["negative_single_contrast_OOD_accuracy_max"]
    return {
        "schema":"ttt-hybrid.f8r57-static-task-admission-audit.v1",
        "freeze_blob":FREEZE_BLOB,
        "source_only_not_model_scoring":True,
        "law_results":evaluated,
        "single_contrast_OOD_accuracy":accuracy,
        "frozen_max_single_contrast_accuracy":
            criteria["negative_single_contrast_OOD_accuracy_max"],
        "frozen_task_admissible":False if not passes else None,
        "training_new_learning_mechanism_authorized":False,
        "explanation":"Equal intervention magnitudes create a first-index tie; "
          "both frozen OOD law codes [3,4] have future sign equal to bridge0. "
          "The single-cue baseline therefore scores 128/128 without learning. "
          "This breaches the frozen maximum 0.749 for task admission. "
          "Do not retune or run new model experiments on this source."
    }
