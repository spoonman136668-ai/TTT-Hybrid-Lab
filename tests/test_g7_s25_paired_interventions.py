"""S25 source-equivalent paired physics tests (development seeds, no fitting)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s20_hidden_multicycle as s20
import g7_s25_paired_interventions as paired
import g7_s25_numeric_ttt as ttt


class PairedInterventionsContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_excluded_frozen_science_seeds_and_invalid_regimes(self):
        for seed in (6203, 6301, 6407, 6503, 6607):
            with self.assertRaisesRegex(ValueError, "S25_ENGINEERING_SEEDS_ONLY"):
                paired.PairedS20World(seed, 2, "source_stable")
        with self.assertRaisesRegex(ValueError, "S25_UNREGISTERED_WORLD_REGIME"):
            paired.PairedS20World(3101, 2, "invalid")
        with self.assertRaisesRegex(ValueError, "S25_INVALID_BATCH"):
            paired.PairedS20World(3101, 0, "source_stable")

    def test_source_coefficient_initialization_preserves_s20_original_rng(self):
        world = paired.PairedS20World(3101, 5, "source_stable")
        generator = torch.Generator(device="cpu").manual_seed(3101)
        old = s20.coefficients(generator, 5, False)
        new = s20.coefficients(generator, 5, False)
        x = -1 + 2 * torch.rand((5, 1), generator=generator)
        z = -.8 + 1.6 * torch.rand((5, 1), generator=generator)
        first_y = x + .25*z + .002*torch.randn((5,1),generator=generator)
        self.assertTrue(torch.equal(old, world._old))
        self.assertTrue(torch.equal(new, world._new))
        self.assertTrue(torch.equal(x, world._x))
        self.assertTrue(torch.equal(z, world._z))
        self.assertTrue(torch.equal(first_y, world._obs))
        self.assertTrue(torch.equal(generator.get_state(), world._gen.get_state()))

    def test_counterfactual_probes_are_exact_s20_physical_step_from_clones(self):
        for regime in paired.REGIMES:
            world = paired.PairedS20World(3119, 3, regime)
            for tick in range(21):
                old_x, old_z = world._x.clone(), world._z.clone()
                initial_obs = world._obs.clone()
                saved_rng = world._gen.get_state()
                clone_rng = torch.Generator(device="cpu")
                clone_rng.set_state(saved_rng)
                dt = .05 + .11 * torch.rand((3, 1), generator=clone_rng)
                u = torch.tensor(paired.PAIR_ACTIONS).view(1, 2).expand(3, 2)
                chosen_coef = world._new if tick >= 20 else world._old
                x2 = old_x.expand(-1, 2)
                z2 = old_z.expand(-1, 2)
                dx = s20.rate(x2, u, chosen_coef, world._family) + .35 * z2
                dz = -.85*z2 + .25*u + .15*x2
                xnext = x2 + dt*dx
                znext = z2 + dt*dz
                observed = xnext + .25*znext + .002*torch.randn((3,2), generator=clone_rng)
                primary = torch.randint(0, 2, (3,), generator=clone_rng)[:,None]
                step = world.emit()
                self.assertTrue(torch.equal(step.observed_pair, observed), regime)
                self.assertTrue(torch.equal(step.query_pair[:,:,0],
                                            initial_obs.expand(-1, 2)), regime)
                self.assertTrue(torch.equal(step.query_pair[:,:,1], u), regime)
                self.assertTrue(torch.equal(step.query_pair[:,:,2],
                                            dt.expand(-1,2)), regime)
                self.assertTrue(torch.equal(world._x, xnext.gather(1,primary)), regime)
                self.assertTrue(torch.equal(world._z, znext.gather(1,primary)), regime)
                self.assertTrue(torch.equal(world._obs, observed.gather(1,primary)), regime)
                self.assertTrue(torch.equal(world._gen.get_state(),
                                            clone_rng.get_state()), regime)

    def test_new_negative_b_does_not_change_previous_world_history_or_rng(self):
        stable = paired.PairedS20World(3109, 4, "source_stable")
        flip = paired.PairedS20World(3109, 4, "source_signflip")
        self.assertTrue(torch.equal(stable._old, flip._old))
        self.assertTrue(torch.equal(stable._x, flip._x))
        self.assertTrue(torch.equal(stable._z, flip._z))
        self.assertTrue(torch.equal(stable._gen.get_state(), flip._gen.get_state()))
        self.assertTrue(torch.all(flip._new[:,1] >= -1.10))
        self.assertTrue(torch.all(flip._new[:,1] <= -.45))
        for tick in range(20):
            a,b = stable.emit(), flip.emit()
            for key in a.__dataclass_fields__:
                self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)),
                                f"pre-switch mismatch {tick} {key}")
        self.assertTrue(torch.equal(stable._x, flip._x))
        self.assertTrue(torch.equal(stable._z, flip._z))
        a,b = stable.emit(), flip.emit()
        self.assertTrue(torch.equal(a.query_pair, b.query_pair))
        self.assertFalse(torch.equal(a.observed_pair, b.observed_pair))

    def test_causal_history_excludes_current_probe_and_is_immutable(self):
        world = paired.PairedS20World(3101, 2, "source_signflip")
        self.assertEqual(set(ttt.WorldStep.__dataclass_fields__),
                         {"history", "query_pair", "observed_pair"})
        first = world.emit()
        saved = first.history.clone()
        self.assertEqual(tuple(first.history.shape), (2,1,4))
        self.assertTrue(torch.equal(first.history[:,:,0],first.history[:,:,3]))
        second = world.emit()
        self.assertTrue(torch.equal(first.history, saved))
        self.assertEqual(tuple(second.history.shape), (2,1,4))
        self.assertTrue(torch.equal(second.history, torch.stack(world._frames[:1],1)))
        self.assertTrue(torch.equal(second.query_pair[:,0,0],second.history[:,-1,3]))
        for _ in range(30):
            world.emit()
        self.assertTrue(torch.equal(first.history, saved))
        self.assertEqual(world.close()["paired_observations"], 128)
        self.assertEqual(world.close()["realized_transitions"], 64)
        with self.assertRaisesRegex(ValueError, "S25_EPISODE_ALREADY_FINISHED"):
            world.emit()

    def test_same_arms_get_same_realized_and_counterfactual_exposure(self):
        a,b = paired.identical_arm_streams(3101, 6, "rational_signflip")
        for _ in range(32):
            sa,sb = a.emit(),b.emit()
            for key in sa.__dataclass_fields__:
                self.assertTrue(torch.equal(getattr(sa,key),getattr(sb,key)), key)
        self.assertEqual(a.close(), b.close())

    def test_three_development_seed_integrity_and_numeric_finiteness(self):
        for seed in paired.ENGINEERING_SEEDS:
            for regime in paired.REGIMES:
                world = paired.PairedS20World(seed, 2, regime)
                last = None
                for tick in range(32):
                    sample = world.emit()
                    for key in sample.__dataclass_fields__:
                        self.assertTrue(torch.isfinite(getattr(sample,key)).all())
                    self.assertEqual(tuple(sample.query_pair.shape),(2,2,3))
                    self.assertEqual(tuple(sample.observed_pair.shape),(2,2))
                    if tick:
                        self.assertEqual(sample.history.shape[1],tick)
                    last = sample
                self.assertEqual(last.history.shape[1],31)
                self.assertEqual(world.close()["steps"],32)

    def test_numeric_ttt_consumes_shared_paired_stream_under_exact_gate_budgets(self):
        torch.manual_seed(3101)
        net = ttt.NumericFastWeightTTT(ttt.ConventionalGRU()).eval()
        original_weights = [t.detach().clone() for t in net.parameters()]
        for mode, count in (("periodic",8),("random",8),("surprise",8),
                            ("contradiction",8),("full",32)):
            world = paired.PairedS20World(3101, 2, "source_signflip")
            sample = world.emit()
            ep = ttt.PrequentialTTTEpisode(net, mode=mode, seed=3101,
                                          batch=2, reference=sample.query_pair)
            ep.step(sample)
            for _ in range(31):
                ep.step(world.emit())
            self.assertEqual(ep.close()["updates_per_world"],count)
            self.assertEqual(world.close()["paired_observations"],128)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(
            original_weights, net.parameters())))

    def test_no_synthetic_privileged_field_in_model_facing_output(self):
        sample = paired.PairedS20World(3109, 2, "sine_signflip").emit()
        self.assertEqual(set(sample.__dict__), {"history","query_pair","observed_pair"})
        self.assertFalse(any(name in sample.__dict__ for name in (
            "_x","_z","_old","_new","family","regime","seed","coefficients")))
        self.assertEqual(sample.query_pair.dtype,torch.float32)
        self.assertEqual(sample.history.dtype,torch.float32)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1))
