"""G7-S26 dev comparison boundaries, source locks and one-shot execution guards."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_dev_comparison as dev
import g7_s26_reserve_gate as reserve
import g7_s25_gate_diagnostic as s25diag
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as paired


class TestS26ComparisonContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dev_matrix_exact_and_scientific_seed_override_rejected(self):
        for kwargs in (
            {"seeds":(7207,)},{"seeds":(3101,)},
            {"regimes":("science_heldout",)},
            {"modes":("contradiction",)},
            {"batch":32},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError,"FULL_DEV_MATRIX_ONLY"):
                    dev.run_development_comparison(**kwargs)

    def test_wall_budget_is_bounded_without_silent_extension(self):
        for n in (-1,0,301):
            with self.assertRaisesRegex(ValueError,"BOUNDED_DIAGNOSTIC_WALL_LIMIT"):
                dev.run_development_comparison(max_seconds=n)

    def test_frozen_source_and_seed_separation_are_explicit(self):
        p=json.loads((ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())
        self.assertEqual(dev.S25_HEAD,p["rationale"]["s25_baseline_head"])
        self.assertEqual(dev.DEV_SEEDS,(3101,3109,3119))
        self.assertFalse(p["authority"]["scientific_first_use_authorized"])
        self.assertTrue(p["authority"]["no_scientific_training_or_fit"])
        self.assertTrue(p["authority"]["no_heldout_seed_consumption"])
        self.assertEqual(p["design"]["quarter_boundaries_end_inclusive"],
                         [7,15,23,31])
        self.assertEqual(p["design"]["guaranteed_updates_steps_24_to_31_per_world"],2)

    def test_s25_reference_is_actual_unchanged_diagnostic_not_reimplementation(self):
        content=(ROOT/"scripts/g7_s26_dev_comparison.py").read_text()
        tree=ast.parse(content)
        self.assertIn("original.run_diagnostic",content)
        self.assertIn("proposed.inject_quarter_reserve",content)
        self.assertIn("S26_S25_MATCHED_INFORMATION_OR_CHECKPOINT_FAILED",content)
        self.assertIn("S26_QUARTER_RESOURCE_OR_REASON_MISMATCH",content)
        self.assertNotIn("torch.optim",content)
        self.assertNotIn("scientific_holdout",content)
        self.assertFalse(any(isinstance(n,ast.Call) and
            isinstance(n.func,ast.Attribute) and n.func.attr in
            ("fit","backward","train","optimizer_step","fit_conventional_gru")
            for n in ast.walk(tree)))

    def test_single_engineering_seed_trace_can_be_replayed_with_identical_inputs(self):
        torch.manual_seed(3101)
        net=numeric.NumericFastWeightTTT(numeric.ConventionalGRU()).eval()
        seen=[]
        for mode in ("periodic","contradiction"):
            sim=paired.PairedS20World(3101,2,"source_stable")
            ep=numeric.PrequentialTTTEpisode(
                net,mode=mode,seed=3101,batch=2,reference=torch.zeros((2,2)))
            reserve.inject_quarter_reserve(ep,seed=3101)
            assert all(isinstance(g,reserve.QuarterReserveGate) for g in ep.gates)
            traces=[]
            for _ in range(32):
                step=sim.emit()
                traces.append((step.history.clone(),step.query_pair.clone(),
                               step.observed_pair.clone(),
                               step.realized_action_index.clone()))
                ep.step(step)
            ep.close()
            seen.append(traces)
        self.assertTrue(all(torch.equal(x,y)
             for trace_a,trace_b in zip(*seen)
             for x,y in zip(trace_a,trace_b)))

    def test_s26_only_additive_paths_relative_to_pinned_s25_are_required(self):
        p=json.loads((ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())
        self.assertEqual(p["rationale"]["s25_baseline_branch"],
                         "research/g7-s25-contradiction-ttt-frozen-20261107")
        self.assertEqual(p["rationale"]["s20_world_git_blob"],
                         "ac3f9ddd5b879e53e62030707fb6419adaf732ca")
        self.assertEqual(p["rationale"]["s25_protocol_git_blob"],
                         "31633b341403729b7ffbb7d73f3a04098ad44380")


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=1))
