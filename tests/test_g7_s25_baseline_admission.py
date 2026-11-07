"""S25 *synthetic receipt* schema negatives; DO NOT fit any model or consume seeds."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s25_baseline_admission as adm


def digest(thing: str) -> str:
    return hashlib.sha256(thing.encode("ascii")).hexdigest()


def synthetic_claimed_records(candidate_sha: str) -> list[dict]:
    """Deliberately invented numbers test validation only, never evidence."""
    records = []
    for i, (fit, valid) in enumerate(zip(adm.TRAIN_FIT_SEEDS, adm.VALIDATION_SEEDS)):
        shared = {
            "fit_seed": fit,
            "validation_seed": valid,
            "candidate_sha": candidate_sha,
            "frozen_protocol_git_blob": adm.FROZEN_PROTOCOL_GIT_BLOB,
            "s20_world_git_blob": adm.S20_SOURCE_GIT_BLOB,
            "training_stream_sha256": digest("synthetic-fit-" + str(i)),
            "validation_stream_sha256": digest("synthetic-valid-" + str(i)),
            "training_cpu_seconds": 12.0,
            "validation_cpu_seconds": 2.0,
            "checkpoint_bytes": 4096,
            "peak_rss_bytes": 100_000_000,
            "static_no_dynamics_mse": 0.4,
            "training_source_only": True,
            "validation_only_for_selection": True,
        }
        gru = dict(shared, arm="GRU_fully_trained",
                   checkpoint_sha256=digest("synthetic-GRU" + str(i)),
                   validation_mse=0.3, updates=896,
                   batch_worlds=32, train_world_exposures=896*32,
                   converged=True, trainer_status="BASELINE_VALIDATION_PLATEAU",
                   width=32, optimizer="AdamW_lr0.001_wd0.01",
                   validation_history=[
                       {"step": 640+64*j, "mse": 0.3}
                       for j in range(5)])
        krr = dict(shared, arm="RBF_KRR",
                   checkpoint_sha256=digest("synthetic-KRR" + str(i)),
                   validation_mse=0.2, gamma=0.5, ridge=0.01,
                   train_query_samples=256,
                   normalization_fitted_on_train_only=True,
                   hyperparameter_selected_on_validation_only=True,
                   history_width=32, numeric_query_width=3)
        records += [gru, krr]
    return records


class BaselineAdmissionLogicalContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = adm.validate_frozen_protocol(
            ROOT / "experiments/g7_s25_frozen_contradiction_ttt.json")
        cls.candidate = "a" * 40

    def setUp(self):
        self.records = synthetic_claimed_records(self.candidate)

    def verify(self):
        return adm.validate_receipt_shapes_only(
            self.protocol, self.candidate, self.records)

    def test_frozen_registration_and_disjoint_seed_roles(self):
        self.assertEqual(self.protocol["seeds"]["fit"], [6203,6211,6221])
        self.assertEqual(self.protocol["seeds"]["validation"], [6301,6311,6323])
        self.assertEqual(self.protocol["seeds"]["engineering_only"], [3101,3109,3119])
        self.assertFalse(self.protocol["authority"]["scientific_first_use_authorized"])

    def test_synthetic_claims_only_satisfy_logic_never_authenticate_science(self):
        outcome = self.verify()
        self.assertEqual(outcome["record_count"], 6)
        self.assertEqual(outcome["fit_seed_count"], 3)
        self.assertEqual(outcome["status"],
                         "PASS_RECEIPT_STRUCTURE_ONLY_NEVER_SCIENTIFIC_PASS")
        self.assertFalse(outcome["trained_baseline_execution_verified"])
        self.assertFalse(outcome["receipt_authenticity_verified"])
        self.assertFalse(outcome["information_and_training_cost_matched_scientifically"])

    def test_missing_strong_control_or_seed_fails(self):
        self.records.pop()
        with self.assertRaisesRegex(ValueError, "MISSING_CONVENTIONAL_CONTROL_OR_SEED"):
            self.verify()

    def test_duplicate_or_relabelled_seed_control_fails(self):
        self.records[1] = copy.deepcopy(self.records[0])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CONTROL_OR_SEED"):
            self.verify()

    def test_frozen_fit_and_validation_seed_roles_cannot_swap(self):
        self.records[0]["validation_seed"] = 6407
        with self.assertRaisesRegex(ValueError, "TRAIN_VALIDATION_SPLIT_DRIFT"):
            self.verify()

    def test_true_science_or_future_oracle_receipt_fields_rejected(self):
        self.records[0]["heldout_transfer_target"] = 0.01
        with self.assertRaisesRegex(ValueError, "FUTURE_INFORMATION_FIELD"):
            self.verify()

    def test_provenance_or_candidate_drift_rejected(self):
        self.records[0]["candidate_sha"] = "b"*40
        with self.assertRaisesRegex(ValueError, "CANDIDATE_SHA_DRIFT"):
            self.verify()
        self.records[0]["candidate_sha"] = self.candidate
        self.records[0]["s20_world_git_blob"] = "f"*40
        with self.assertRaisesRegex(ValueError, "SOURCE_PROVENANCE_DRIFT"):
            self.verify()

    def test_train_validation_sample_hash_collisions_rejected(self):
        self.records[0]["validation_stream_sha256"] = (
            self.records[0]["training_stream_sha256"])
        with self.assertRaisesRegex(ValueError, "BASELINE_DATA_LEAKAGE"):
            self.verify()

    def test_trained_gru_minimum_updates_and_plateau_required(self):
        self.records[0]["updates"] = 512
        with self.assertRaisesRegex(ValueError, "GRU_UNDERTRAINED"):
            self.verify()
        self.records[0]["updates"] = 896
        self.records[0]["validation_history"][0]["mse"] = 0.5
        with self.assertRaisesRegex(ValueError, "GRU_VALIDATION_PLATEAU_NOT_ESTABLISHED"):
            self.verify()

    def test_validation_must_have_five_64_spaced_checks(self):
        self.records[0]["validation_history"][1]["step"] = 712
        with self.assertRaisesRegex(ValueError, "GRU_VALIDATION_CHECKPOINT_INTERVAL_DRIFT"):
            self.verify()

    def test_kernel_grid_and_train_only_calibration_are_required(self):
        self.records[1]["gamma"] = 100.0
        with self.assertRaisesRegex(ValueError, "KRR_NOT_FROZEN_CALIBRATION_GRID"):
            self.verify()
        self.records[1]["gamma"] = 0.5
        self.records[1]["normalization_fitted_on_train_only"] = False
        with self.assertRaisesRegex(ValueError, "KRR_ORACLE_OR_HELDOUT_CALIBRATION"):
            self.verify()

    def test_competent_baselines_must_beat_naive_source_predictor(self):
        self.records[0]["validation_mse"] = .5
        with self.assertRaisesRegex(ValueError, "HAS_NOT_BEAT_NAIVE"):
            self.verify()

    def test_nan_cost_and_memory_budget_excess_rejected(self):
        self.records[0]["training_cpu_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "INVALID_TRAINING_CPU_SECONDS"):
            self.verify()
        self.records[0]["training_cpu_seconds"] = 12.0
        self.records[0]["peak_rss_bytes"] = 3_000_000_000
        with self.assertRaisesRegex(ValueError, "RSS_CAP_EXCEEDED"):
            self.verify()

    def test_two_control_exposure_streams_must_match(self):
        self.records[1]["training_stream_sha256"] = digest("different-raw-worlds")
        with self.assertRaisesRegex(ValueError, "CONTROLS_NOT_MATCHED_ON_OBSERVATIONS"):
            self.verify()

    def test_trained_checkpoint_identity_cannot_be_aliased(self):
        self.records[1]["checkpoint_sha256"] = self.records[0]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "CHECKPOINT_ALIASING"):
            self.verify()

    def test_no_executed_training_is_performed_by_contract(self):
        self.assertFalse(hasattr(adm, "fit_conventional_gru"))
        self.assertFalse(hasattr(adm, "CalibratedRBFKernelRidge"))
        self.assertEqual(len(synthetic_claimed_records(self.candidate)), 6)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1))
