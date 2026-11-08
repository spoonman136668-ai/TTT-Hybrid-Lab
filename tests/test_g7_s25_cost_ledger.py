"""G7-S25 work-ledger qualification on engineering-only S20 twin-action worlds.

No training or science seeds, no optimizer fitting, no benchmark claims.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
import unittest
from unittest import mock
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as physics
import g7_s25_cost_ledger as accounting
import g7_s20_hidden_multicycle as original_s20
from g7_process_peak_memory import PeakMemory, MemoryProbeBlocked


class TestCostLedger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3101)
        self.model = numeric.NumericFastWeightTTT(numeric.ConventionalGRU()).eval()

    def execute(self, mode="periodic", seed=3101, batch=2, regime="source_signflip"):
        world = physics.PairedS20World(seed, batch, regime)
        episode = numeric.PrequentialTTTEpisode(
            self.model, mode=mode, seed=seed, batch=batch,
            reference=world._obs)
        ledger = accounting.EpisodeWorkLedger(episode, source_seed=seed)
        for _ in range(32):
            y = ledger.step(world.emit())
            self.assertEqual(tuple(y.shape), (batch,2))
        self.assertEqual(world.close()["paired_observations"], 32 * batch * 2)
        return ledger.close()

    def test_equal_budget_arms_match_bytewise_information_and_fast_gradient_work(self):
        rows = [self.execute(name) for name in
                ("periodic", "random", "surprise", "contradiction")]
        receipt = accounting.compare_equal_budget_receipts(rows)
        self.assertEqual(receipt["status"],
                         "PASS_EQUAL_INFORMATION_AND_FAST_UPDATE_WORK_ONLY")
        self.assertEqual(receipt["real_fast_updates_each"], 16)
        self.assertEqual(receipt["paired_observations_each"], 128)
        self.assertEqual(len({r["input_stream_sha256"] for r in rows}), 1)
        self.assertEqual(len({r["model_checkpoint_sha256"] for r in rows}), 1)
        self.assertEqual(len({r["real_fast_weight_update_worlds"] for r in rows}), 1)

    def test_full_update_reference_reports_64_not_16_actual_world_updates(self):
        ref = self.execute("full")
        self.assertEqual(ref["real_fast_weight_update_worlds"], 64)
        self.assertEqual(ref["skipped_fast_gradient_worlds"], 0)
        self.assertEqual(ref["updates_per_world"], 32)
        self.assertEqual(ref["paired_intervention_observations"], 128)
        self.assertEqual(ref["fast_state_update_flops_lower_bound"],
                         64 * accounting.FAST_UPDATE_MIN_ARITHMETIC_PER_SELECTED_WORLD)

    def test_sparse_gradient_skips_all_false_masks_even_if_unselected_features_nan(self):
        state = torch.zeros((2, 16))
        features = torch.full((2, 2, 16), float("nan"))
        pred = torch.zeros((2, 2))
        targets = torch.ones((2, 2))
        no_updates = self.model.update_after_observation(
            state, features, pred, targets, torch.tensor([False, False]))
        self.assertTrue(torch.equal(no_updates, state))
        features[0] = 0.5
        one_update = self.model.update_after_observation(
            state, features, pred, targets, torch.tensor([True, False]))
        self.assertTrue(torch.isfinite(one_update).all())
        self.assertTrue(torch.equal(one_update[1], state[1]))
        self.assertGreater(one_update[0].abs().sum().item(), 0)

    def test_accounted_fast_gradient_only_on_true_mask_rows(self):
        result = self.execute("contradiction", batch=3)
        self.assertEqual(result["steps"], 32)
        self.assertEqual(result["real_fast_weight_update_worlds"], 3*8)
        self.assertEqual(result["skipped_fast_gradient_worlds"], 3*(32-8))
        self.assertEqual(result["read_forward_query_predictions"], 3*32*2)
        self.assertEqual(result["realized_transitions"], 3*32)
        self.assertEqual(result["fast_state_bytes"], 3*16*4)

    def test_same_seed_same_stream_different_seed_rejected_by_parity(self):
        same = [self.execute(m) for m in
                ("periodic", "random", "surprise", "contradiction")]
        same[2] = self.execute("surprise", seed=3109)
        with self.assertRaisesRegex(ValueError, "LEDGER_UNMATCHED_INFORMATION"):
            accounting.compare_equal_budget_receipts(same)

    def test_tampered_work_or_missing_control_fails_closed(self):
        records = [self.execute(m) for m in
                   ("periodic", "random", "surprise", "contradiction")]
        with self.assertRaisesRegex(ValueError, "LEDGER_MISSING_EQUAL_BUDGET_CONTROL"):
            accounting.compare_equal_budget_receipts(records[:3])
        tampered = copy.deepcopy(records)
        tampered[0]["real_fast_weight_update_worlds"] += 1
        with self.assertRaisesRegex(ValueError, "LEDGER_UNMATCHED_INFORMATION_OR_UPDATE_WORK"):
            accounting.compare_equal_budget_receipts(tampered)
        tampered = copy.deepcopy(records)
        tampered[0]["trained_baselines_qualified"] = True
        with self.assertRaisesRegex(ValueError, "LEDGER_PREMATURE_BASELINE_ACCEPTANCE"):
            accounting.compare_equal_budget_receipts(tampered)

    def test_model_checkpoint_drift_fails_closed(self):
        world = physics.PairedS20World(3101, 2, "source_stable")
        ep = numeric.PrequentialTTTEpisode(
            self.model, mode="periodic", seed=3101,
            batch=2, reference=world._obs)
        ledger = accounting.EpisodeWorkLedger(ep, source_seed=3101)
        for _ in range(32):
            ledger.step(world.emit())
        with torch.no_grad():
            next(self.model.parameters()).add_(1.0)
        with self.assertRaisesRegex(ValueError, "LEDGER_SLOW_MODEL_CHECKPOINT_DRIFT"):
            ledger.close()

    def test_development_only_seeds_enforced_before_episode(self):
        world = physics.PairedS20World(3101, 2, "source_stable")
        ep = numeric.PrequentialTTTEpisode(
            self.model, mode="periodic", seed=3101,
            batch=2, reference=world._obs)
        with self.assertRaisesRegex(ValueError, "LEDGER_ENGINEERING_SEED_ONLY"):
            accounting.EpisodeWorkLedger(ep, source_seed=6203)
        receipt = self.execute("periodic", seed=3119)
        self.assertEqual(receipt["development_seed"], 3119)

    def test_resource_values_measured_but_not_claimed_full_model_flops(self):
        result = self.execute("periodic")
        self.assertIn(result["process_peak_memory_metric"],
                      ("linux_ru_maxrss_bytes",
                       "darwin_ru_maxrss_bytes",
                       "windows_psapi_peak_working_set_bytes"))
        self.assertFalse(result["peak_rss_is_isolated_per_arm"])
        if sys.platform == "linux":
            self.assertGreater(result["process_peak_rss_bytes"], 0)
            self.assertTrue(result["process_peak_memory_comparable_to_linux_ru_maxrss"])
            self.assertIsNone(result["process_peak_working_set_bytes"])
        elif sys.platform == "win32":
            self.assertIsNone(result["process_peak_rss_bytes"])
            self.assertGreater(result["process_peak_working_set_bytes"], 0)
            self.assertFalse(result["process_peak_memory_comparable_to_linux_ru_maxrss"])
        self.assertGreater(result["slow_checkpoint_bytes"], 0)
        self.assertTrue(math.isfinite(result["episode_process_cpu_seconds"]))
        self.assertTrue(math.isfinite(result["episode_step_wall_seconds"]))
        self.assertFalse(result["estimated_flops_are_total_model_flops"])
        self.assertFalse(result["forward_or_gate_flops_exactly_profiled"])
        self.assertIsNone(result["whole_process_training_and_validation_cpu_seconds"])
        self.assertFalse(result["trained_baselines_qualified"])
        self.assertFalse(result["whole_process_matched_compute_qualified"])

    def test_windows_working_set_not_presented_as_linux_rss(self):
        from unittest import mock
        reading=PeakMemory(12345678, "windows_psapi_peak_working_set_bytes", False)
        with mock.patch.object(accounting, "read_peak_memory", return_value=reading):
            result=self.execute("periodic")
        self.assertIsNone(result["process_peak_rss_bytes"])
        self.assertEqual(result["process_peak_working_set_bytes"], 12345678)
        self.assertEqual(result["process_peak_memory_metric"],
                         "windows_psapi_peak_working_set_bytes")
        self.assertFalse(result["process_peak_memory_comparable_to_linux_ru_maxrss"])
        self.assertFalse(result["trained_baselines_qualified"])

    def test_linux_frozen_memory_measurement_unchanged(self):
        reading=PeakMemory(8192, "linux_ru_maxrss_bytes", True)
        with mock.patch.object(accounting, "read_peak_memory", return_value=reading):
            result=self.execute("periodic")
        self.assertEqual(result["process_peak_rss_bytes"], 8192)
        self.assertIsNone(result["process_peak_working_set_bytes"])
        self.assertTrue(result["process_peak_memory_comparable_to_linux_ru_maxrss"])

    def test_memory_probe_failure_cannot_produce_ledger_receipt(self):
        with mock.patch.object(accounting, "read_peak_memory",
                               side_effect=MemoryProbeBlocked("WINDOWS_PSAPI_QUERY_FAILED")):
            with self.assertRaisesRegex(MemoryProbeBlocked, "WINDOWS_PSAPI_QUERY_FAILED"):
                self.execute("periodic")

    def test_original_s20_science_fails_before_opening_frozen_data_on_windows(self):
        with mock.patch.object(original_s20.sys, "platform", "win32"):
            with self.assertRaisesRegex(original_s20.Blocked,
                                         "S20_FROZEN_RSS_REQUIRES_LINUX"):
                original_s20.run(object())

    def test_reuse_and_invalid_sample_or_nonfinite_observation_rejected(self):
        world = physics.PairedS20World(3101, 2, "source_stable")
        ep = numeric.PrequentialTTTEpisode(
            self.model, mode="periodic", seed=3101,
            batch=2, reference=world._obs)
        ledger = accounting.EpisodeWorkLedger(ep, source_seed=3101)
        bad = world.emit()
        with self.assertRaisesRegex(ValueError, "LEDGER_INVALID_WORLD_STEP"):
            ledger.step(None)
        changed = numeric.WorldStep(
            bad.history, bad.query_pair, torch.full((2,2), float('nan')),
            bad.realized_action_index)
        with self.assertRaisesRegex(ValueError, "LEDGER_NONFINITE_TENSOR"):
            ledger.step(changed)
        ledger.step(bad)
        for _ in range(31):
            ledger.step(world.emit())
        ledger.close()
        with self.assertRaisesRegex(ValueError, "LEDGER_ALREADY_CLOSED"):
            ledger.close()
        with self.assertRaisesRegex(ValueError, "LEDGER_ALREADY_COMPLETE"):
            ledger.step(bad)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1))
