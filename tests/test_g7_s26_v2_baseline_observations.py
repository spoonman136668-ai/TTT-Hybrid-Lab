"""G7-S26-on-V2 conventional baseline input readiness: NO MODEL TRAINING."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s26_v2_baseline_observations as obs
from g7_s25_numeric_ttt import causal_history32


class TestDevBaselinePrequentialObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.first=obs.collect_development_observations(3101)
        cls.other=obs.collect_development_observations(3109)

    def test_only_dev_seed_subset_no_fit_val_heldout(self):
        self.assertEqual(obs.DEV_SEEDS,(3101,3109,3119))
        for forbidden in (7207,7219,7229,7307,7319,7327,
                          7403,7411,7429,7507,7517,7529,
                          3100,False,True):
            with self.subTest(seed=forbidden):
                with self.assertRaisesRegex(ValueError,"DEV_ONLY_SEED_REQUIRED"):
                    obs.collect_development_observations(forbidden)

    def test_no_world_batch_or_regime_drift(self):
        for kwargs in ({"batch":31},{"batch":33},{"batch":True},
                       {"regime":"source_signflip"},
                       {"regime":"heldout_transfer"},
                       {"world_factory":lambda *a: None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    obs.collect_development_observations(3101,**kwargs)

    def test_real_prequential_shapes_and_exact_exposure_count(self):
        d=self.first
        self.assertEqual(tuple(d.history.shape),(1024,32,4))
        self.assertEqual(tuple(d.query_pair.shape),(1024,2,3))
        self.assertEqual(tuple(d.observed_pair.shape),(1024,2))
        self.assertEqual(tuple(d.realized_action.shape),(1024,))
        self.assertEqual(d.worlds,32)
        self.assertEqual(d.steps,32)
        self.assertEqual(d.regime,"source_stable")
        self.assertTrue(torch.isfinite(d.history).all())
        self.assertTrue(torch.isfinite(d.query_pair).all())
        self.assertTrue(torch.isfinite(d.observed_pair).all())

    def test_action_plus_minus_probes_same_current_sensor_and_dt(self):
        d=self.first
        self.assertTrue(torch.all(d.query_pair[:,0,1]==-0.8))
        self.assertTrue(torch.all(d.query_pair[:,1,1]==0.8))
        self.assertTrue(torch.equal(d.query_pair[:,0,0],d.query_pair[:,1,0]))
        self.assertTrue(torch.equal(d.query_pair[:,0,2],d.query_pair[:,1,2]))
        self.assertTrue(torch.all(d.query_pair[:,:,2]>0))
        self.assertTrue(torch.all(d.query_pair[:,:,2]<0.2))

    def test_prior_observations_not_current_labels(self):
        d=self.first
        # In the first step, the only history is the original sensor
        # sentinel; its synthetic past-action/dt are zero and never
        # current paired prediction labels.
        self.assertTrue(torch.equal(d.history[:32,0,1],torch.zeros(32)))
        self.assertTrue(torch.equal(d.history[:32,0,2],torch.zeros(32)))
        self.assertFalse(torch.allclose(
            d.history[:32,0,3],d.observed_pair[:32,0]))
        # For step t=1, the most recent sensor is the actual executed
        # observed sensor from t=0. The unused probe is never in history.
        first_labels=d.observed_pair[:32]
        selectors=d.realized_action[:32,None]
        realized=first_labels.gather(1,selectors).squeeze(1)
        self.assertTrue(torch.equal(d.history[32:64,0,3],realized))
        self.assertTrue(torch.equal(d.history[32:64,1:],torch.zeros_like(
            d.history[32:64,1:])))
        # At the final prequential step, the last past frame corresponds
        # to step 30, never step 31 labels.
        last_previous_labels=d.observed_pair[30*32:31*32]
        last_previous_action=d.realized_action[30*32:31*32,None]
        previous=last_previous_labels.gather(1,last_previous_action).squeeze(1)
        self.assertTrue(torch.equal(d.history[31*32:32*32,30,3],previous))

    def test_independent_dev_world_streams(self):
        self.assertNotEqual(self.first.stream_sha256,self.other.stream_sha256)
        self.assertEqual(len(self.first.stream_sha256),64)
        self.assertEqual(len(self.other.stream_sha256),64)

    def test_same_seed_replay_exact_bytes(self):
        again=obs.collect_development_observations(3101)
        self.assertEqual(self.first.stream_sha256,again.stream_sha256)
        self.assertTrue(torch.equal(self.first.history,again.history))
        self.assertTrue(torch.equal(self.first.query_pair,again.query_pair))
        self.assertTrue(torch.equal(self.first.observed_pair,again.observed_pair))
        self.assertTrue(torch.equal(self.first.realized_action,again.realized_action))

    def test_causal_padding_nonfinite_is_rejected(self):
        bad=torch.zeros(1,4,4)
        bad[0,2,3]=float("nan")
        with self.assertRaisesRegex(ValueError,"NONFINITE_CAUSAL_PREFIX"):
            causal_history32(bad)
        with self.assertRaisesRegex(ValueError,"INVALID_CAUSAL_PREFIX"):
            causal_history32(torch.zeros(1,0,4))

    def test_three_seeds_audited_with_no_fit_and_no_heldout(self):
        result=obs.audit_development_example_inputs()
        self.assertEqual(result["status"],
                         "PASS_DEVELOPMENT_PREQUENTIAL_INPUTS_NO_MODEL_FIT")
        self.assertEqual(result["paired_world_steps"],3072)
        self.assertEqual(result["actual_observations"],6144)
        self.assertEqual([r["seed"] for r in result["rows"]],[3101,3109,3119])
        self.assertEqual(result["conventional_baseline_fit"],"NOT_RUN")
        self.assertEqual(result["krr_calibration"],"NOT_RUN")
        self.assertEqual(result["scientific_first_use"],"NOT_AUTHORIZED_NOT_RUN")
        self.assertEqual(result["scientific_learning_conclusion"],"UNVERIFIED")

    def test_source_never_calls_fitter_optimizer_or_exposes_private_world_state(self):
        source=(ROOT/"scripts/g7_s26_v2_baseline_observations.py").read_text()
        tree=ast.parse(source)
        attrs={node.attr for node in ast.walk(tree)
               if isinstance(node,ast.Attribute)}
        self.assertFalse({"fit_conventional_gru","fit_calibrated","train",
                          "backward","optimizer_step","step",
                          "_x","_z","_old","_new","_gen"}.intersection(attrs))
        self.assertNotIn("torch.optim",source)
        self.assertNotIn("science_seed_namespace_reserved_unconsumed",source)


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
