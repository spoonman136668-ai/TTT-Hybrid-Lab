"""Untrained readout evaluator: distinguish effect sign from aggregate MSE.

Only development seeds. No model fit, optimizer, scientific data or threshold
adjustment. Fabricated pairs are deliberately NOT baseline competence evidence.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_contrast_competence as diag


class S26ContrastCompetenceTests(unittest.TestCase):
    def test_correct_significant_action_effect_is_diagnostic_only(self):
        p=torch.tensor([[-0.05,0.05]],dtype=torch.float32)
        d=diag.contrast_score(p,p)
        summary=diag.sum_counts([d])
        self.assertEqual(summary["both_significant"],1)
        self.assertEqual(summary["sign_correct"],1)
        self.assertEqual(summary["sign_opposite_eligible"],0)
        self.assertEqual(summary["effect_sign_accuracy_when_evaluable"],1.0)
        self.assertTrue(summary["readout_measures_only_no_competence_pass"])

    def test_low_mse_can_conceal_wrong_direction(self):
        actual=torch.tensor([[-0.0061,0.0061]])
        predicted=torch.tensor([[0.0061,-0.0061]])
        score=diag.sum_counts([diag.contrast_score(predicted,actual)])
        self.assertLess(score["mean_squared_pair_mse"],0.001)
        self.assertEqual(score["sign_opposite_eligible"],1)
        self.assertEqual(score["sign_correct"],0)
        self.assertEqual(score["effect_sign_accuracy_when_evaluable"],0.0)

    def test_zero_or_tiny_action_contrast_is_not_correct_direction(self):
        target=torch.tensor([[-0.05,0.05]])
        constant=torch.tensor([[0.0,0.0]])
        score=diag.sum_counts([diag.contrast_score(constant,target)])
        self.assertEqual(score["sign_correct"],0)
        self.assertEqual(score["both_significant"],0)
        self.assertEqual(score["unidentifiable_small_prediction"],1)
        self.assertIsNone(score["effect_sign_accuracy_when_evaluable"])
        self.assertEqual(score["effect_sign_coverage_of_real_large_effects"],0.0)

    def test_subthreshold_true_effect_never_counts_as_signed_accuracy(self):
        almost=torch.tensor([[-0.004,0.004]])
        score=diag.sum_counts([diag.contrast_score(almost,almost)])
        self.assertEqual(score["observed_significant"],0)
        self.assertIsNone(score["effect_sign_coverage_of_real_large_effects"])
        self.assertEqual(score["sign_correct"],0)

    def test_nonfinite_shape_and_gpu_fail_closed(self):
        valid=torch.tensor([[-0.05,0.05]])
        with self.assertRaisesRegex(ValueError,"PAIRED_SAMPLE_SHAPE"):
            diag.contrast_score(valid,torch.zeros(1,1))
        with self.assertRaisesRegex(ValueError,"NONFINITE_OR_DEVICE"):
            diag.contrast_score(torch.tensor([[float("nan"),0.05]]),valid)

    def test_science_seeds_and_off_protocol_regimes_rejected(self):
        for kwargs in ({"seeds":(6203,6211,6221)},
                       {"regimes":("heldout_signflip",)},
                       {"batch":32}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError,"DEV_WORLD_ONLY_FIXED_MATRIX"):
                    diag.run_untrained_development_diagnostic(**kwargs)
        with self.assertRaisesRegex(ValueError,"BOUNDED_RUNTIME"):
            diag.run_untrained_development_diagnostic(max_seconds=0)

    def test_source_never_trains_or_adjusts_frozen_thresholds(self):
        text=(ROOT/"scripts/g7_s26_contrast_competence.py").read_text()
        parsed=ast.parse(text)
        forbidden={"fit","fit_calibrated","fit_conventional_gru",
                   "backward","train","step","optimizer_step"}
        calls=[node.func.attr for node in ast.walk(parsed)
               if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute)]
        self.assertFalse(forbidden.intersection(calls))
        self.assertNotIn("torch.optim",text)
        self.assertNotIn("S26_SCIENTIFIC_PASS",text)
        self.assertEqual(diag.SEEDS,(3101,3109,3119))
        self.assertEqual(diag.BLOB_S25,"31633b341403729b7ffbb7d73f3a04098ad44380")
        self.assertEqual(diag.BLOB_S26,"34251b2cde6373ed484dbc21e4617c9165047e93")


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
