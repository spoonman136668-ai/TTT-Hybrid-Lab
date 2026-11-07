"""G7-S26 quarter-reserve preregistration and negative controls: NO scientific use."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s25_contradiction_gate as s25
import g7_s25_cost_ledger as custody
import g7_s25_numeric_ttt as model
import g7_s25_paired_interventions as paired
import g7_s26_reserve_gate as reserve

def obs(*, signflip:bool=False, primary_residual:float=0.0):
    return dict(predicted_minus=-0.05,predicted_plus=0.05,
        observed_minus=0.05 if signflip else -0.05,
        observed_plus=-0.05 if signflip else 0.05,
        primary_residual=primary_residual)

def run_gate(mode:str, *, flip:bool=False, residual:float=0.0,
             seed:int=3101) -> tuple[list[int],list[str]]:
    g=reserve.QuarterReserveGate(mode,engineering_seed=seed)
    selected=[]; reasons=[]
    for step in range(32):
        d=g.observe(step, **obs(signflip=flip,primary_residual=residual))
        if d.update:
            selected.append(step)
            reasons.append(d.reason)
    assert g.close()==8
    return selected,reasons

class S26QuarterReserveContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol=json.loads(
            (ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())

    def setUp(self):
        torch.manual_seed(3101)
        self.network=model.NumericFastWeightTTT(model.ConventionalGRU()).eval()

    def test_protocol_declares_separate_s26_not_mutated_s25(self):
        p=self.protocol
        self.assertEqual(p["schema"],
                         "ttt-hybrid.g7-s26.reserve-aware-contradiction-timing.v1")
        self.assertEqual(p["rationale"]["s25_baseline_head"],
                         "108808f7b685107a76fae1ce5605b7ff9e2bac27")
        self.assertEqual(p["rationale"]["s25_protocol_git_blob"],
                         "31633b341403729b7ffbb7d73f3a04098ad44380")
        self.assertFalse(p["authority"]["scientific_first_use_authorized"])
        self.assertTrue(p["authority"]["do_not_modify_s25_pr40_or_s24_pr39"])

    def test_exact_two_updates_per_quarter_all_four_arms(self):
        for mode in reserve.RESERVE_MODES:
            with self.subTest(mode=mode):
                slots,_=run_gate(mode,flip=True,residual=.20)
                self.assertEqual(len(slots),8)
                self.assertEqual(
                    [sum(a<=t<=b for t in slots) for a,b in
                     ((0,7),(8,15),(16,23),(24,31))],
                    [2,2,2,2])
                self.assertEqual(sum(t>=24 for t in slots),2)

    def test_periodic_arm_exactly_matches_unchanged_s25(self):
        old=s25.PrequentialGate("periodic",engineering_seed=3101)
        s25_slots=[t for t in range(32)
                   if old.observe(t,**obs()).update]
        self.assertEqual(old.close(),8)
        self.assertEqual(run_gate("periodic")[0],s25_slots)
        self.assertEqual(s25_slots,[3,7,11,15,19,23,27,31])

    def test_constant_contradiction_does_not_exhaust_all_k8_by_step8(self):
        old=s25.PrequentialGate("contradiction",engineering_seed=3101)
        old_slots=[t for t in range(32)
                   if old.observe(t,**obs(signflip=True)).update]
        new_slots,reasons=run_gate("contradiction",flip=True)
        self.assertEqual(old.close(),8)
        self.assertEqual(len(old_slots),8)
        self.assertTrue(all(t<20 for t in old_slots))
        self.assertEqual(new_slots,[1,2,8,9,16,17,24,25])
        self.assertEqual(sum(t>=20 for t in new_slots),2)
        self.assertTrue(all(r=="trigger" for r in reasons))

    def test_no_contradictions_forces_by_each_quarter_end(self):
        slots,reasons=run_gate("contradiction",flip=False)
        self.assertEqual(slots,[6,7,14,15,22,23,30,31])
        self.assertEqual(reasons,["forced_budget"]*8)

    def test_random_schedule_seeded_and_block_matched(self):
        a=run_gate("random",seed=3101)[0]
        b=run_gate("random",seed=3101)[0]
        c=run_gate("random",seed=3109)[0]
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertEqual(sum(t>=24 for t in a),2)

    def test_surprise_policy_triggered_from_primary_residual_not_sign(self):
        g=reserve.QuarterReserveGate("surprise",engineering_seed=3101)
        first=g.observe(0,**obs(signflip=True,primary_residual=0.0))
        second=g.observe(1,**obs(signflip=False,primary_residual=0.2))
        self.assertFalse(first.trigger)
        self.assertTrue(second.trigger)
        self.assertTrue(second.update)

    def test_nonfinite_out_of_order_unknown_arm_rejected(self):
        with self.assertRaisesRegex(ValueError,"UNKNOWN_RESERVE_MODE"):
            reserve.QuarterReserveGate("oracle",engineering_seed=3101)
        with self.assertRaisesRegex(ValueError,"INVALID_GATE_SEED"):
            reserve.QuarterReserveGate("random",engineering_seed=-1)
        g=reserve.QuarterReserveGate("contradiction",engineering_seed=3101)
        with self.assertRaisesRegex(ValueError,"NONCAUSAL_OR_OUT_OF_ORDER"):
            g.observe(4,**obs())
        with self.assertRaisesRegex(ValueError,"NONFINITE_GATE_OBSERVATION"):
            g.observe(0,**obs(primary_residual=float("nan")))
        with self.assertRaisesRegex(ValueError,"UNFINISHED_OR_UNMATCHED"):
            g.close()

    def test_fresh_episode_required_and_science_seeds_forbidden(self):
        q=torch.zeros(2,2)
        e=model.PrequentialTTTEpisode(
            self.network,mode="contradiction",seed=3101,batch=2,reference=q)
        for forbidden in (6203,6301,6407,6503,7207,7307,7403):
            with self.subTest(seed=forbidden):
                with self.assertRaisesRegex(ValueError,"ENGINEERING_SEEDS_ONLY"):
                    reserve.inject_quarter_reserve(e,seed=forbidden)
        reserve.inject_quarter_reserve(e,seed=3101)
        self.assertTrue(all(isinstance(g,reserve.QuarterReserveGate)
                            for g in e.gates))
        example=paired.PairedS20World(3101,2,"source_stable").emit()
        e.step(example)
        with self.assertRaisesRegex(ValueError,"FRESH_S25_EPISODE_REQUIRED"):
            reserve.inject_quarter_reserve(e,seed=3101)

    def test_s26_injected_episode_exact_k8_updates_and_identical_source(self):
        for mode in reserve.RESERVE_MODES:
            environment=paired.PairedS20World(3101,2,"source_signflip")
            episode=model.PrequentialTTTEpisode(
                self.network,mode=mode,seed=3101,batch=2,
                reference=torch.zeros(2,2))
            reserve.inject_quarter_reserve(episode,seed=3101)
            ledger=custody.EpisodeWorkLedger(episode,source_seed=3101)
            for t in range(32):
                ledger.step(environment.emit())
            receipt=ledger.close()
            self.assertEqual(receipt["real_fast_weight_update_worlds"],16)
            self.assertEqual(receipt["paired_intervention_observations"],128)
            self.assertEqual(receipt["updates_per_world"],8)
            times=torch.stack(episode.update_masks).sum(1)
            self.assertEqual([int(times[x:x+8].sum()) for x in (0,8,16,24)],
                             [4,4,4,4])

    def test_gate_observe_signature_does_not_accept_switch_or_oracle(self):
        fields=set(inspect.signature(reserve.QuarterReserveGate.observe).parameters)
        self.assertEqual(fields,set(inspect.signature(
            s25.PrequentialGate.observe).parameters))
        self.assertFalse(any(x in fields for x in
                             ("switch","oracle","coefficients","hidden_state")))

    def test_science_frozen_seed_roles_disjoint_from_s25(self):
        new=self.protocol["evaluation"]["science_seed_namespace_reserved_unconsumed"]
        existing=json.loads((ROOT/"experiments/g7_s25_frozen_contradiction_ttt.json").read_text())["seeds"]
        new_seeds=[x for k,v in new.items() if k!="bootstrap" for x in v]
        old_seeds=[x for k,v in existing.items() if isinstance(v,list)
                   for x in v]
        self.assertEqual(len(new_seeds),len(set(new_seeds)))
        self.assertTrue(set(new_seeds).isdisjoint(old_seeds))
        self.assertTrue(set(new_seeds).isdisjoint((3101,3109,3119)))


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=1))
