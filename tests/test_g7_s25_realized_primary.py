"""G7-S25 actual-primary-residual comparator regression; engineering seeds only.

Pre-first-use bug fix: surprise policy must use the residual for the action
actually executed, not the counterfactual +0.8 probe by convention.
No model fitting, scientific data, scientific conclusions or GPU execution.
"""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_s25_numeric_ttt as model
import g7_s25_paired_interventions as world
import g7_s25_cost_ledger as ledger


class ActualPrimaryResidualContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(3101)
        self.net = model.NumericFastWeightTTT(model.ConventionalGRU()).eval()

    def synthetic(self, primary: list[int], minus=0.0, plus=0.10):
        batch = len(primary)
        hist = torch.zeros((batch, 1, 4))
        hist[:,:,0] = 0.2
        hist[:,:,3] = 0.2
        queries = torch.tensor(
            [[0.2,-0.8,0.1],[0.2,0.8,0.1]],dtype=torch.float32
        )[None].expand(batch,-1,-1).clone()
        observations = torch.tensor(
            [minus,plus],dtype=torch.float32)[None].expand(batch,-1).clone()
        return model.WorldStep(hist,queries,observations,
                               torch.tensor(primary,dtype=torch.int64))

    def test_real_world_selection_matches_actual_observed_transition(self):
        w = world.PairedS20World(3101,8,"source_signflip")
        for _ in range(32):
            sample = w.emit()
            chosen = sample.realized_action_index
            self.assertTrue(torch.equal(chosen,w._last_primary_index))
            self.assertEqual(tuple(chosen.shape),(8,))
            self.assertTrue(bool(((chosen==0)|(chosen==1)).all()))
            indices = chosen[:,None]
            action = sample.query_pair[:,:,1].gather(1,indices)
            observation = sample.observed_pair.gather(1,indices)
            self.assertTrue(torch.equal(w._frames[-1][:,1:2],action))
            self.assertTrue(torch.equal(w._frames[-1][:,3:4],observation))
        self.assertEqual(w.close()["steps"],32)

    def test_surprise_gate_uses_minus_when_minus_action_is_realized(self):
        # Same counterfactual plus target is high error. But when minus action
        # is actually executed, its residual is zero and must NOT trigger.
        self.net.prequential_predict = lambda h,q,w: (
            torch.zeros((q.shape[0],2)),torch.zeros((q.shape[0],2,16)))
        source = self.synthetic([0,1],minus=0.0,plus=0.10)
        ep = model.PrequentialTTTEpisode(
            self.net,mode="surprise",seed=3101,batch=2,
            reference=source.query_pair)
        predictions = ep.step(source)
        self.assertTrue(torch.equal(predictions,torch.zeros(2,2)))
        self.assertTrue(torch.equal(ep.update_masks[0],
                                    torch.tensor([False,True])))
        self.assertEqual(ep.gates[0].updates_used,0)
        self.assertEqual(ep.gates[1].updates_used,1)

    def test_primary_selection_is_known_action_not_oracle_and_no_lookahead(self):
        w = world.PairedS20World(3109,3,"rational_signflip")
        sample = w.emit()
        self.assertEqual(set(sample.__dataclass_fields__),
                         {"history","query_pair","observed_pair",
                          "realized_action_index"})
        self.assertFalse(any(key in sample.__dict__ for key in
                             ("_x","_z","coefficients","family","seed","oracle")))
        before = [p.detach().clone() for p in self.net.parameters()]
        left = model.PrequentialTTTEpisode(
            self.net,mode="surprise",seed=3101,batch=3,
            reference=sample.query_pair)
        other = model.PrequentialTTTEpisode(
            self.net,mode="surprise",seed=3101,batch=3,
            reference=sample.query_pair)
        changed = model.WorldStep(sample.history,sample.query_pair,
                                  sample.observed_pair,
                                  1 - sample.realized_action_index)
        predicted_a=left.step(sample)
        predicted_b=other.step(changed)
        self.assertTrue(torch.equal(predicted_a,predicted_b))
        self.assertTrue(all(torch.equal(a,b) for a,b in
                            zip(before,self.net.parameters())))

    def test_invalid_or_missing_action_selector_fails_before_update(self):
        sample = self.synthetic([0,1])
        ep = model.PrequentialTTTEpisode(
            self.net,mode="surprise",seed=3101,batch=2,
            reference=sample.query_pair)
        for selector in (None,torch.tensor([0,2],dtype=torch.int64),
                         torch.tensor([0,1],dtype=torch.float32),
                         torch.tensor([[0],[1]],dtype=torch.int64)):
            invalid = model.WorldStep(sample.history,sample.query_pair,
                                      sample.observed_pair,selector)
            with self.assertRaisesRegex(ValueError,
                                        "INVALID_OBSERVED_REALIZED_ACTION"):
                ep.step(invalid)
            self.assertEqual(ep.steps,0)
            self.assertEqual(ep.gates[0].updates_used,0)

    def test_realized_action_is_hashed_in_equal_information_receipts(self):
        sample=self.synthetic([0,1])
        other=model.WorldStep(sample.history,sample.query_pair,
                              sample.observed_pair,torch.tensor([1,0]))
        def fingerprint(data):
            instance=model.PrequentialTTTEpisode(
                self.net,mode="periodic",seed=3101,batch=2,
                reference=data.query_pair)
            meter=ledger.EpisodeWorkLedger(instance,source_seed=3101)
            for _ in range(32):
                meter.step(data)
            return meter.close()
        a=fingerprint(sample)
        b=fingerprint(other)
        self.assertNotEqual(a["input_stream_sha256"],
                            b["input_stream_sha256"])
        self.assertEqual(a["real_fast_weight_update_worlds"],16)
        self.assertEqual(b["real_fast_weight_update_worlds"],16)

    def test_no_regression_of_periodic_budget_and_paired_read_before_write(self):
        sample=self.synthetic([0,1])
        ep=model.PrequentialTTTEpisode(
            self.net,mode="periodic",seed=3101,batch=2,
            reference=sample.query_pair)
        for _ in range(32):
            ep.step(sample)
        self.assertEqual(ep.close()["updates_per_world"],8)
        self.assertEqual(int(torch.stack(ep.update_masks).sum()),16)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=1))
