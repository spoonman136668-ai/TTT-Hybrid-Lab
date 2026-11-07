"""No scientific data or training. Standard-library S25 gate contract tests."""
from pathlib import Path
import inspect
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s25_contradiction_gate as gate


def observations(sign_reverse=False, residual=0.0):
    return dict(predicted_minus=-0.05, predicted_plus=0.05,
                observed_minus=(0.05 if sign_reverse else -0.05),
                observed_plus=(-0.05 if sign_reverse else 0.05),
                primary_residual=residual)


class G7S25GateContract(unittest.TestCase):
    def test_all_equal_budget_policies_complete_exactly_eight(self):
        for mode in ("periodic", "random", "surprise", "contradiction"):
            instance = gate.PrequentialGate(mode, engineering_seed=3101)
            positions = []
            for t in range(32):
                decision = instance.observe(t, **observations(sign_reverse=t >= 20))
                if decision.update:
                    positions.append(t)
            self.assertEqual(len(positions), 8, mode)
            self.assertEqual(instance.close(), 8)
            self.assertEqual(len(set(positions)), 8)

    def test_full_update_reference_is_separately_costed(self):
        g = gate.PrequentialGate("full", engineering_seed=3101)
        self.assertTrue(all(g.observe(t, **observations()).update for t in range(32)))
        self.assertEqual(g.close(), 32)

    def test_contradiction_requires_two_of_three_opposite_sign_pairs(self):
        g = gate.PrequentialGate("contradiction", engineering_seed=3101)
        self.assertFalse(g.observe(0, **observations(True, .1)).update)
        s = g.observe(1, **observations(True, .1))
        self.assertTrue(s.update)
        self.assertTrue(s.trigger)
        self.assertEqual(s.contradiction_evidence_count, 2)

    def test_surprise_alone_does_not_create_causal_contradiction(self):
        contradiction = gate.PrequentialGate("contradiction", engineering_seed=3101)
        surprise = gate.PrequentialGate("surprise", engineering_seed=3101)
        a = contradiction.observe(0, **observations(False, .2))
        b = surprise.observe(0, **observations(False, .2))
        self.assertFalse(a.trigger)
        self.assertTrue(b.trigger)
        self.assertFalse(a.update)
        self.assertTrue(b.update)

    def test_one_contradiction_then_consistent_sign_does_not_persist(self):
        g = gate.PrequentialGate("contradiction", engineering_seed=3101)
        self.assertFalse(g.observe(0, **observations(True)).update)
        self.assertFalse(g.observe(1, **observations(False)).update)
        self.assertFalse(g.observe(2, **observations(False)).update)
        self.assertFalse(g.observe(3, **observations(False)).update)
        self.assertEqual(g.updates_used, 0)

    def test_random_decisions_reproducible_with_no_future_access(self):
        def trial(seed):
            g = gate.PrequentialGate("random", engineering_seed=seed)
            return [g.observe(t, **observations()).update for t in range(32)]
        self.assertEqual(trial(3101), trial(3101))
        self.assertEqual(sum(trial(3101)), 8)
        self.assertNotEqual(trial(3101), trial(3109))

    def test_order_nonfinite_and_bad_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNKNOWN_GATE_MODE"):
            gate.PrequentialGate("oracle", engineering_seed=3101)
        with self.assertRaisesRegex(ValueError, "INVALID_GATE_SEED"):
            gate.PrequentialGate("random", engineering_seed=-1)
        g = gate.PrequentialGate("surprise", engineering_seed=3101)
        with self.assertRaisesRegex(ValueError, "NONCAUSAL_OR_OUT_OF_ORDER"):
            g.observe(1, **observations())
        with self.assertRaisesRegex(ValueError, "NONFINITE_GATE_OBSERVATION"):
            g.observe(0, **observations(residual=float("nan")))
        with self.assertRaisesRegex(ValueError, "UNFINISHED_OR_UNMATCHED"):
            g.close()

    def test_only_current_observed_and_predicted_numeric_inputs(self):
        fields = list(inspect.signature(gate.PrequentialGate.observe).parameters)
        self.assertEqual(fields, ["self", "step", "predicted_minus", "predicted_plus",
                                  "observed_minus", "observed_plus", "primary_residual"])
        self.assertFalse(any(n in fields for n in (
            "family_id", "true_coefficients", "future", "hidden_x", "oracle")))
        self.assertEqual(set(gate.MODES),
                         {"periodic", "random", "surprise", "contradiction", "full"})

    def test_source_protocol_matches_logic_and_is_first_use_closed(self):
        p = json.loads((ROOT / "experiments/g7_s25_frozen_contradiction_ttt.json")
                       .read_text(encoding="utf-8"))
        self.assertEqual(p["schema"], "ttt-hybrid.g7-s25.contradiction-gated-continuous-ttt.v1")
        self.assertEqual(p["seeds"]["engineering_only"], [3101, 3109, 3119])
        self.assertEqual(p["budgets"]["real_updates_equal_budget_arms"], 8)
        self.assertEqual(p["budgets"]["episode_steps"], 32)
        self.assertEqual(p["mechanism"]["pair_actions"], [-0.8, 0.8])
        self.assertEqual(gate.MIN_SIGNED_EFFECT, .012)
        self.assertEqual(gate.MIN_CONTRAST_DISAGREEMENT, .018)
        self.assertEqual(gate.SURPRISE_THRESHOLD, .018)
        self.assertEqual(gate.CONTRADICTION_WINDOW, 3)
        self.assertEqual(gate.CONTRADICTION_COUNT, 2)
        self.assertIs(p["authority"]["scientific_first_use_authorized"], False)
        self.assertIs(p["authority"]["no_training_execution"], True)
        roles=("fit","validation","heldout_signflip","heldout_transfer","negative")
        seeds=[s for role in roles for s in p["seeds"][role]]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertTrue(set(seeds).isdisjoint(p["seeds"]["engineering_only"]))


if __name__ == "__main__":
    unittest.main()
