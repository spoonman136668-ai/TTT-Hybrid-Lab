"""Engineering-only synthetic numerical fixtures; no pretrained science seeds."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import g7_s25_numeric_ttt as m


def input_step(batch=2, step=0):
    history = torch.zeros(batch, step + 1, 4)
    history[:, :, 0] = 0.2
    history[:, :, 2] = 0.1
    queries = torch.tensor([[0.2, -0.8, 0.1], [0.2, 0.8, 0.1]],
                           dtype=torch.float).unsqueeze(0).repeat(batch, 1, 1)
    observations = torch.tensor([[0.12, 0.28]], dtype=torch.float).repeat(batch, 1)
    return m.WorldStep(history, queries, observations, torch.arange(batch, dtype=torch.int64) % 2)


class NumericTTTTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3101)
        self.net = m.NumericFastWeightTTT(m.ConventionalGRU())
        self.net.eval()

    def test_numeric_shape_deterministic_zero_fast_weights(self):
        step = input_step()
        state = self.net.new_state(2, reference=step.query_pair)
        output, phi = self.net.prequential_predict(step.history, step.query_pair, state)
        repeated, phi2 = self.net.prequential_predict(step.history, step.query_pair, state)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(tuple(phi.shape), (2, 2, 16))
        self.assertTrue(torch.equal(output, repeated))
        self.assertTrue(torch.equal(phi, phi2))
        self.assertTrue(torch.isfinite(output).all())

    def test_masked_no_update_keeps_exact_state(self):
        s = input_step()
        w = self.net.new_state(2, reference=s.query_pair)
        predicted, phi = self.net.prequential_predict(s.history, s.query_pair, w)
        untouched = self.net.update_after_observation(w, phi, predicted,
                                                        s.observed_pair, torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.equal(w, untouched))

    def test_update_changes_only_selected_world_and_not_slow_weights(self):
        s = input_step()
        w = self.net.new_state(2, reference=s.query_pair)
        before = [p.detach().clone() for p in self.net.parameters()]
        predicted, phi = self.net.prequential_predict(s.history, s.query_pair, w)
        observed = predicted + torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        changed = self.net.update_after_observation(w, phi, predicted, observed,
                                                     torch.tensor([True, False]))
        self.assertGreater(float(changed[0].abs().sum()), 0)
        self.assertTrue(torch.equal(changed[1], w[1]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, self.net.parameters())))

    def test_prequential_read_before_current_observation(self):
        s = input_step()
        a = m.PrequentialTTTEpisode(self.net, mode='full', seed=3101,
                                     batch=2, reference=s.query_pair)
        b = m.PrequentialTTTEpisode(self.net, mode='full', seed=3101,
                                     batch=2, reference=s.query_pair)
        r1 = a.step(s)
        altered = m.WorldStep(s.history, s.query_pair, s.observed_pair + 100.0, s.realized_action_index)
        r2 = b.step(altered)
        self.assertTrue(torch.equal(r1, r2))
        self.assertFalse(torch.equal(a.state, b.state))

    def test_all_policy_arms_complete_32_steps_with_matched_or_reference_updates(self):
        expected = {'periodic': 8, 'random': 8, 'surprise': 8,
                    'contradiction': 8, 'full': 32}
        for mode, steps in expected.items():
            ep = m.PrequentialTTTEpisode(self.net, mode=mode, seed=3101, batch=2,
                                         reference=input_step().query_pair)
            for t in range(32):
                ep.step(input_step(step=t))
            result = ep.close()
            self.assertEqual(result['updates_per_world'], steps)
            self.assertEqual(result['steps'], 32)
            self.assertEqual(int(torch.stack(ep.update_masks).sum()), steps * 2)

    def test_wrong_action_pair_fails_closed(self):
        ep = m.PrequentialTTTEpisode(self.net, mode='periodic', seed=3101, batch=2,
                                     reference=input_step().query_pair)
        s = input_step()
        q = s.query_pair.clone()
        q[:, 1, 1] = 0.3
        with self.assertRaisesRegex(ValueError, 'UNMATCHED_ACTION_INTERVENTIONS'):
            ep.step(m.WorldStep(s.history, q, s.observed_pair, s.realized_action_index))
        self.assertEqual(ep.steps, 0)

    def test_fastweight_and_baseline_budget_are_frozen(self):
        with self.assertRaisesRegex(ValueError, 'FROZEN_FAST_WEIGHT_CAPACITY_OR_LR_DRIFT'):
            m.NumericFastWeightTTT(m.ConventionalGRU(), learning_rate=0.1)
        with self.assertRaisesRegex(ValueError, 'FROZEN_GRU_WIDTH_DRIFT'):
            m.ConventionalGRU(16)
        with self.assertRaisesRegex(ValueError, 'BASELINE_FROZEN_TRAINING_BUDGET_DRIFT'):
            m.fit_conventional_gru(m.ConventionalGRU(), lambda _: None, lambda: None,
                                   max_updates=50)

    def test_rbf_kernel_control_requires_separate_training(self):
        kernel = m.CalibratedRBFKernelRidge()
        s = input_step(step=31)
        features = kernel.numeric_vectors(s.history, s.query_pair)
        self.assertEqual(tuple(features.shape), (4, 131))
        self.assertEqual(kernel.GAMMAS, (0.5, 2.0, 6.0))
        self.assertEqual(kernel.RIDGES, (0.01, 0.08))
        with self.assertRaisesRegex(ValueError, 'UNTRAINED_KERNEL_BASELINE'):
            kernel.predict(s.history, s.query_pair)
        with self.assertRaisesRegex(ValueError, 'KERNEL_REQUIRES_32_OBSERVED_TRANSITIONS'):
            kernel.numeric_vectors(s.history[:, :-1, :], s.query_pair)

    def test_pretrained_controls_not_implied_by_constructors(self):
        s = input_step()
        self.assertFalse(hasattr(self.net.slow, 'training_receipt'))
        self.assertTrue(any(p.requires_grad for p in self.net.slow.parameters()))
        ep = m.PrequentialTTTEpisode(self.net, mode='periodic', seed=3101,
                                     batch=2, reference=s.query_pair)
        with self.assertRaisesRegex(ValueError, 'EPISODE_INCOMPLETE'):
            ep.close()

    def test_missing_or_nonfinite_inputs_rejected(self):
        s = input_step()
        with self.assertRaisesRegex(ValueError, 'INVALID_NUMERIC_HISTORY'):
            self.net.slow.encode(s.history[:, :0, :])
        q = s.query_pair.clone()
        q[0,0,0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'NONFINITE_QUERY'):
            self.net.prequential_predict(s.history, q,
                                         self.net.new_state(2, reference=q))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1))
