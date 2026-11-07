"""No training. Prove causal history, paired-target isolation and KRR/GRU parity."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s25_causal_baseline_stream as stream
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as worlds


class TestCausalBaselineStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causal_prefix_is_32_slots_with_no_future_transition_rows(self):
        rows = stream.engineering_only_prefixes(seed=3101, batch=3)
        self.assertEqual(len(rows), 32)
        for tick, row in enumerate(rows):
            self.assertEqual(tuple(row.history.shape), (3, 32, 4))
            self.assertTrue(torch.equal(
                row.history[:, tick + (tick == 0):],
                torch.zeros_like(row.history[:, tick + (tick == 0):])))
            self.assertEqual(tuple(row.queries.shape), (3, 2, 3))
            self.assertEqual(tuple(row.targets.shape), (3, 2))

    def test_causal_prefix_matches_exact_original_emitted_past(self):
        world = worlds.PairedS20World(3109, 4, "source_stable")
        rows = stream.engineering_only_prefixes(seed=3109, batch=4)
        for tick in range(32):
            raw = world.emit()
            prefix = rows[tick]
            self.assertTrue(torch.equal(
                prefix.history[:, :raw.history.shape[1]], raw.history))
            self.assertTrue(torch.equal(prefix.queries, raw.query_pair))
            self.assertTrue(torch.equal(prefix.targets, raw.observed_pair))
        self.assertEqual(world.close()["paired_observations"], 256)

    def test_same_numeric_context_admitted_by_gru_and_32_frame_kernel(self):
        rows = stream.engineering_only_prefixes(seed=3119, batch=5)
        for t in (0, 1, 17, 20, 31):
            row = rows[t]
            accepted = stream.admit_cross_architecture_sample(row)
            self.assertEqual(accepted["kernel_flat_features"], 131)
            self.assertEqual(accepted["numeric_history_steps"], 32)
            self.assertFalse(accepted["fitted_baselines"])
            torch.manual_seed(3101)
            gru = numeric.ConventionalGRU().eval()
            with torch.no_grad():
                predicted = gru.predict(*row.model_inputs())
            self.assertEqual(tuple(predicted.shape), (5, 2))
            self.assertTrue(torch.isfinite(predicted).all())

    def test_current_targets_cannot_enter_gru_or_kernel_feature_inputs(self):
        row = stream.engineering_only_prefixes(seed=3101, batch=4)[20]
        labels = row.targets.clone()
        poisoned = dataclasses.replace(row, targets=labels + 500.0)
        self.assertTrue(torch.equal(row.history, poisoned.history))
        self.assertTrue(torch.equal(row.queries, poisoned.queries))
        a = stream.stream_fingerprint((row,), include_targets=False)
        b = stream.stream_fingerprint((poisoned,), include_targets=False)
        self.assertEqual(a, b)
        self.assertNotEqual(
            stream.stream_fingerprint((row,), include_targets=True),
            stream.stream_fingerprint((poisoned,), include_targets=True))
        kernel = numeric.CalibratedRBFKernelRidge()
        self.assertTrue(torch.equal(
            kernel.numeric_vectors(*row.model_inputs()),
            kernel.numeric_vectors(*poisoned.model_inputs())))

    def test_stream_deterministic_across_replay_and_seed_sensitive(self):
        a = stream.engineering_only_prefixes(seed=3101, batch=3)
        b = stream.engineering_only_prefixes(seed=3101, batch=3)
        c = stream.engineering_only_prefixes(seed=3109, batch=3)
        self.assertEqual(stream.stream_fingerprint(a, include_targets=True),
                         stream.stream_fingerprint(b, include_targets=True))
        self.assertNotEqual(stream.stream_fingerprint(a, include_targets=True),
                            stream.stream_fingerprint(c, include_targets=True))
        self.assertNotEqual(stream.stream_fingerprint(a, include_targets=False),
                            stream.stream_fingerprint(c, include_targets=False))

    def test_split_first_use_protected_against_frozen_seeds(self):
        for seed in (6203, 6211, 6221, 6301, 6311, 6323,
                     6407, 6413, 6421, 6503, 6511, 6521):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "ENGINEERING_SEEDS_ONLY"):
                    stream.engineering_only_prefixes(seed=seed, batch=2)

    def test_only_source_training_family_exposed_by_development_adapter(self):
        for regime in ("source_signflip", "sine_signflip", "rational_signflip"):
            with self.assertRaisesRegex(ValueError, "DEV_SOURCE_ONLY"):
                stream.engineering_only_prefixes(seed=3101, batch=2,
                                                 regime=regime)

    def test_model_facing_triplet_contains_no_privileged_fields(self):
        row = stream.engineering_only_prefixes(seed=3101, batch=2)[7]
        self.assertEqual(set(row.__dataclass_fields__),
                         {"history", "queries", "targets"})
        self.assertEqual(len(row.training_triplet()), 3)
        self.assertEqual(len(row.model_inputs()), 2)
        self.assertEqual(row.model_inputs()[0].dtype, torch.float32)
        self.assertEqual(row.model_inputs()[1].dtype, torch.float32)

    def test_invalid_nonfinite_and_unmatched_actions_fail(self):
        world = worlds.PairedS20World(3101, 2, "source_stable")
        original = world.emit()
        bad = original.query_pair.clone()
        bad[:, 1, 1] = .1
        with self.assertRaisesRegex(ValueError, "ACTION_DRIFT"):
            stream.causal_supervision(numeric.WorldStep(
                original.history, bad, original.observed_pair,
                original.realized_action_index))
        bad = original.observed_pair.clone()
        bad[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            stream.causal_supervision(numeric.WorldStep(
                original.history, original.query_pair, bad,
                original.realized_action_index))
        with self.assertRaisesRegex(ValueError, "INVALID_CAUSAL_PREFIX"):
            numeric.causal_history32(torch.zeros((2, 0, 4)))

    def test_offline_baseline_source_inputs_have_no_model_fit_or_future_lookahead(self):
        rows = stream.engineering_only_prefixes(seed=3119, batch=2)
        inputs = stream.source_paired_baseline_inputs(rows)
        self.assertEqual(len(inputs), 32)
        self.assertTrue(all(len(item) == 3 for item in inputs))
        self.assertTrue(all(torch.equal(
            x[0], rows[t].history) and torch.equal(x[1], rows[t].queries)
            and torch.equal(x[2], rows[t].targets)
            for t,x in enumerate(inputs)))
        # The adapter imports no training optimizer, and only the downstream
        # separately admitted runner can invoke actual model fitting.
        self.assertFalse(hasattr(stream, "fit_conventional_gru"))
        self.assertFalse(hasattr(stream, "fit_calibrated"))


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1))
