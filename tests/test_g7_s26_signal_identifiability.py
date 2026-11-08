"""Positive/negative causal signal controls for the unchanged S25/S26 gates.

Entirely standard-library, no neural model, optimizer, simulated worlds,
training, scientific seeds, heldout evaluation or package installation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_signal_identifiability as signal


class TestS26SignalIdentifiability(unittest.TestCase):
    def test_five_fixed_controls_and_eight_real_updates(self):
        result=signal.synthetic_mechanism_audit()
        self.assertEqual(result["status"],
                         "PASS_HANDCRAFTED_SIGNAL_CONTROLS_ONLY")
        self.assertEqual([row["case"] for row in result["cases"]],
                         list(signal.CASES))
        for row in result["cases"]:
            for arm in ("S25","S26"):
                self.assertEqual(len(row[arm]["updates"]),8)
                self.assertEqual(
                    len(row[arm]["selected_by_signal"])+
                    len(row[arm]["forced_by_budget"]),8)
            self.assertEqual(row["S26"]["quarter_update_counts"],[2]*4)

    def test_sustained_sign_reversal_detected_on_second_paired_observation(self):
        candidate=signal.trace("persistent_reversal_at_20","S26")
        self.assertEqual(candidate["first_signal_selected"],21)
        self.assertIn(21,candidate["selected_by_signal"])
        self.assertGreater(candidate["post20_signals"],0)
        self.assertGreater(candidate["persistent_conflict_frames"],0)

    def test_s25_also_detects_sign_reversal_when_predictions_are_sane(self):
        candidate=signal.trace("persistent_reversal_at_20","S25")
        self.assertGreater(candidate["post20_signals"],0)
        self.assertEqual(candidate["first_signal_selected"],21)

    def test_stable_correct_predictions_never_trigger_contradiction(self):
        for arm in ("S25","S26"):
            candidate=signal.trace("stable",arm)
            self.assertEqual(candidate["selected_by_signal"],[])
            self.assertEqual(candidate["persistent_conflict_frames"],0)
            self.assertEqual(len(candidate["forced_by_budget"]),8)

    def test_single_frame_and_subthreshold_signals_cannot_be_misclassified(self):
        for case in ("one_step_reversal_at_20",
                     "subthreshold_reversal_at_20"):
            for arm in ("S25","S26"):
                with self.subTest(case=case,arm=arm):
                    item=signal.trace(case,arm)
                    self.assertEqual(item["selected_by_signal"],[])
                    self.assertEqual(item["first_signal_selected"],None)

    def test_persistent_early_miscalibration_depletes_s25_not_s26(self):
        old=signal.trace("persistent_initial_miscalibration","S25")
        fresh=signal.trace("persistent_initial_miscalibration","S26")
        self.assertEqual(old["post20_updates"],0)
        self.assertEqual(fresh["post20_updates"],2)
        self.assertEqual(fresh["quarter_update_counts"],[2,2,2,2])

    def test_unknown_scenario_or_mode_or_timestep_fails_closed(self):
        for args in (("oracle",0),("stable",-1),("stable",32)):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError,"NOT_REGISTERED"):
                    signal.observation(*args)
        with self.assertRaisesRegex(ValueError,"UNREGISTERED_PREQUENTIAL_GATE"):
            signal.trace("stable","science")

    def test_no_fitting_or_heldout_or_model_execution_claim(self):
        p=ROOT/"scripts/g7_s26_signal_identifiability.py"
        text=p.read_text()
        syntax=ast.parse(text)
        forbidden={"fit","backward","optimizer_step","train",
                   "load_state_dict","evaluate_heldout"}
        calls=[node.func.attr for node in ast.walk(syntax)
               if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute)]
        self.assertFalse(forbidden.intersection(calls))
        self.assertNotIn("torch",text)
        result=signal.synthetic_mechanism_audit()
        self.assertFalse(result["models_trained"])
        self.assertFalse(result["frozen_scientific_seeds_consumed"])
        self.assertEqual(result["interpretation"][
            "learned_model_competence"],"NOT_ESTABLISHED")


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
