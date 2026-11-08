"""S26 claimed-baseline admission negative controls: NO model training.

All examples are fabricated receipts, not empirical model evidence.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import math
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_baseline_direction_admission as admission


def fake_sha(label:str)->str:
    return hashlib.sha256(label.encode()).hexdigest()


def fake_records(candidate:str)->list[dict]:
    records=[]
    for idx,(fit,val) in enumerate(zip(admission.FIT_SEEDS,
                                       admission.VALIDATION_SEEDS)):
        shared=dict(
            fit_seed=fit,validation_seed=val,candidate_sha=candidate,
            s25_ancestor_sha=admission.S25_HEAD,
            frozen_s26_protocol_blob=admission.S26_BLOB,
            frozen_s25_protocol_blob=admission.S25_BLOB,
            s20_physics_source_blob=admission.S20_BLOB,
            fit_stream_sha256=fake_sha(f"fit-{fit}"),
            validation_stream_sha256=fake_sha(f"val-{val}"),
            model_source_sha256=fake_sha("frozen-model-code"),
            training_uses_fit_only=True,
            model_selection_uses_validation_only=True,
            evaluation_uses_future_observation=False,
            actual_checkpoint_restored=True,
            fit_pair_exposures=512,validation_pair_exposures=384,
            train_wall_seconds=35.0,validate_wall_seconds=2.0,
            train_cpu_seconds=30.0,validate_cpu_seconds=1.7,
            peak_rss_bytes=450_000_000,checkpoint_bytes=32768,
            validation_paired_mse=0.2,naive_validation_paired_mse=0.4,
            action_effect_validation=dict(
                paired_world_steps=384,true_effect_significant=320,
                model_effect_significant=316,
                sign_correct=288,sign_wrong=16,
                uninformative_model_effect=16,
                mean_absolute_effect_error=0.04,
                predictions_made_before_observation=True,
                same_state_paired_action_contrast=True,
                validation_only_not_heldout=True,
            ),
        )
        gru=dict(shared,arm="GRU_fully_trained",
            checkpoint_sha256=fake_sha(f"GRU-{fit}"),
            optimizer_updates=896,batch_worlds=32,width=32,
            optimizer="AdamW_lr0.001_wd0.01",converged=True,
            trainer_status="BASELINE_VALIDATION_PLATEAU",
            validation_history=[
                {"step":640+64*t,"paired_mse":0.2} for t in range(5)])
        krr=dict(shared,arm="RBF_KRR",
            checkpoint_sha256=fake_sha(f"KRR-{fit}"),
            gamma=0.5,ridge=0.01,
            normalization_train_only=True,
            calibration_validation_only=True,
            history_steps=32,numeric_query_width=3,
            train_query_samples=1024)
        records.extend([gru,krr])
    return records


class TestS26DirectionPreAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=admission.frozen_protocol(
            ROOT/"experiments/g7_s26_frozen_quarter_reserve.json")
        cls.candidate="c"*40

    def setUp(self):
        self.claims=fake_records(self.candidate)

    def validate(self):
        return admission.validate_claimed_baselines(
            self.protocol,self.candidate,self.claims)

    def test_frozen_s26_seed_namespace_and_no_first_use(self):
        p=self.protocol
        self.assertEqual(p["evaluation"]["science_seed_namespace_reserved_unconsumed"]["fit"],
                         [7207,7219,7229])
        self.assertEqual(p["evaluation"]["science_seed_namespace_reserved_unconsumed"]["validation"],
                         [7307,7319,7327])
        self.assertFalse(p["authority"]["scientific_first_use_authorized"])
        self.assertTrue(p["authority"]["no_scientific_training_or_fit"])

    def test_synthetic_claims_pass_only_schema_not_authentic_science(self):
        result=self.validate()
        self.assertEqual(result["claimed_fit_validation_records"],6)
        self.assertEqual(result["status"],
                         "PASS_CLAIMED_RECEIPT_SHAPES_ONLY_NEVER_SCIENCE")
        self.assertFalse(result["independent_receipt_authenticity_verified"])
        self.assertFalse(result["baseline_training_performed_here"])
        self.assertFalse(result["trained_baseline_science_accepted"])
        self.assertEqual(result["scientific_first_use"],"NOT_RUN")

    def test_wrong_effect_sign_with_good_mse_is_rejected(self):
        m=self.claims[0]["action_effect_validation"]
        m.update(sign_correct=0,sign_wrong=304,
                 uninformative_model_effect=16)
        self.claims[0]["validation_paired_mse"]=0.002
        with self.assertRaisesRegex(ValueError,"WRONG_ACTION_EFFECT_DIRECTION"):
            self.validate()

    def test_nearly_zero_effect_sign_with_low_mse_is_not_readiness(self):
        m=self.claims[0]["action_effect_validation"]
        m.update(model_effect_significant=10,sign_correct=8,
                 sign_wrong=0,uninformative_model_effect=312)
        self.claims[0]["validation_paired_mse"]=0.002
        with self.assertRaisesRegex(ValueError,"SIGN_COVERAGE_UNINFORMATIVE"):
            self.validate()

    def test_malformed_or_inflated_confusion_matrix_fails(self):
        self.claims[0]["action_effect_validation"]["sign_correct"]=400
        with self.assertRaisesRegex(ValueError,"CONFUSION_MATRIX_INCONSISTENT"):
            self.validate()

    def test_no_current_target_or_science_oracle_may_enter_validation(self):
        m=self.claims[0]["action_effect_validation"]
        m["predictions_made_before_observation"]=False
        with self.assertRaisesRegex(ValueError,"NONCAUSAL_ACTION_EFFECT_VALIDATION"):
            self.validate()
        m["predictions_made_before_observation"]=True
        self.claims[0]["heldout_signflip_mse"]=0.01
        with self.assertRaisesRegex(ValueError,"FORBIDDEN_SCIENCE_ORACLE_FIELD"):
            self.validate()

    def test_convergence_requires_five_real_spaced_validation_checks(self):
        self.claims[0]["validation_history"][1]["step"]=650
        with self.assertRaisesRegex(ValueError,"VALIDATION_NOT_SPACED_64"):
            self.validate()
        self.claims[0]["validation_history"][1]["step"]=704
        self.claims[0]["validation_history"][0]["paired_mse"]=0.35
        with self.assertRaisesRegex(ValueError,"PLATEAU_NOT_ESTABLISHED"):
            self.validate()

    def test_train_val_seed_or_source_stream_drift_is_blocked(self):
        self.claims[0]["validation_seed"]=7403
        with self.assertRaisesRegex(ValueError,"FIT_VALIDATION_ROLE_DRIFT"):
            self.validate()
        self.claims[0]["validation_seed"]=7307
        self.claims[0]["fit_stream_sha256"]=self.claims[0]["validation_stream_sha256"]
        with self.assertRaisesRegex(ValueError,"FIT_VALIDATION_STREAM_COLLISION"):
            self.validate()

    def test_unmatched_gru_krr_information_or_checkpoint_alias_is_rejected(self):
        self.claims[1]["fit_stream_sha256"]=fake_sha("fake-KRR-source")
        with self.assertRaisesRegex(ValueError,"NOT_MATCHED_ON_SOURCE_EXPOSURE"):
            self.validate()
        self.claims[1]["fit_stream_sha256"]=self.claims[0]["fit_stream_sha256"]
        self.claims[1]["checkpoint_sha256"]=self.claims[0]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError,"REUSED_BASELINE_CHECKPOINT"):
            self.validate()

    def test_missing_arm_seed_or_hyperparameter_frozen_grid_fails(self):
        self.claims.pop()
        with self.assertRaisesRegex(ValueError,"INCOMPLETE_BASELINE_ARM_SEED_MATRIX"):
            self.validate()
        self.claims=fake_records(self.candidate)
        self.claims[1]["gamma"]=100.0
        with self.assertRaisesRegex(ValueError,"UNREGISTERED_CALIBRATION_GRID"):
            self.validate()

    def test_frozen_resource_budget_and_positive_metrics_enforced(self):
        self.claims[0]["peak_rss_bytes"]=3_000_000_000
        with self.assertRaisesRegex(ValueError,"MEMORY_BUDGET_EXCEEDED"):
            self.validate()
        self.claims[0]["peak_rss_bytes"]=400_000_000
        self.claims[0]["train_wall_seconds"]=901.0
        with self.assertRaisesRegex(ValueError,"TIME_CAP_EXCEEDED"):
            self.validate()
        self.claims[0]["train_wall_seconds"]=35.0
        self.claims[0]["validation_paired_mse"]=0.8
        with self.assertRaisesRegex(ValueError,"NOT_BETTER_THAN_NAIVE_PAIRED_MSE"):
            self.validate()

    def test_no_training_import_or_scientific_source_use(self):
        path=ROOT/"scripts/g7_s26_baseline_direction_admission.py"
        source=path.read_text()
        tree=ast.parse(source)
        self.assertNotIn("import torch",source)
        self.assertNotIn("torch.optim",source)
        self.assertNotIn("fit_conventional_gru",source)
        self.assertNotIn("fit_calibrated",source)
        forbidden={"backward","optimizer_step","train",
                   "fit","load_state_dict"}
        calls={n.func.attr for n in ast.walk(tree)
               if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute)}
        self.assertFalse(forbidden.intersection(calls))


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
