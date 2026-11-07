"""S25 measurable gate-mechanism diagnostic admission on development-only worlds.

No model training, optimizer runs, frozen scientific seeds or scientific gate.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s25_gate_diagnostic as diagnostic


def fake_rows(*, selected: int, post_selected: int):
    rows=[]
    for mode in diagnostic.COMPARE_MODES:
        for seed in diagnostic.DEV_SEEDS:
            for regime in diagnostic.DEV_REGIMES:
                rows.append({
                    "mode":mode,"seed":seed,"regime":regime,
                    "selected_trigger_steps":selected if mode=="contradiction" else 0,
                    "post_switch_selected_trigger_steps": (
                        post_selected if mode=="contradiction" else 0),
                    "forced_budget_updates":8,
                    "real_fast_weight_update_worlds":32,
                })
    return rows


class TestMechanismDiagnostic(unittest.TestCase):
    def test_precise_mechanism_null_when_only_forced_budget_updates(self):
        result=diagnostic.summarize(
            fake_rows(selected=0,post_selected=0))
        self.assertEqual(result["status"],
                         "UNTRAINED_DIAGNOSTIC_TRIGGER_NULL")
        self.assertEqual(result["untrained_contradiction_selected_trigger_world_steps"],0)
        self.assertEqual(result["total_equal_budget_contradiction_world_updates"],384)
        self.assertEqual(result["science"],"NOT_SCIENTIFIC_PASS")

    def test_post_switch_trigger_missing_is_recorded_as_mechanism_null(self):
        result=diagnostic.summarize(
            fake_rows(selected=3,post_selected=0))
        self.assertEqual(result["status"],
                         "UNTRAINED_DIAGNOSTIC_TRIGGER_NULL")
        self.assertEqual(
            result["untrained_signflip_post_switch_selected_trigger_world_steps"],0)

    def test_positive_trigger_diagnostic_is_never_a_science_pass(self):
        result=diagnostic.summarize(
            fake_rows(selected=2,post_selected=1))
        self.assertEqual(result["status"],
                         "UNTRAINED_DIAGNOSTIC_TRIGGER_EXERCISED_NOT_SCIENCE")
        self.assertEqual(result["science"],"NOT_SCIENTIFIC_PASS")

    def test_unregistered_science_seed_and_regime_fail_before_world_use(self):
        for seed in (6203,6301,6407,6503,6607):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError,"DEVELOPMENT_SEEDS_ONLY"):
                    diagnostic.run_diagnostic(seeds=(seed,))
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT_REGIMES_ONLY"):
            diagnostic.run_diagnostic(regimes=("heldout_signflip",))
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT_MODES_ONLY"):
            diagnostic.run_diagnostic(modes=("oracle",))

    def test_partial_real_world_replay_reports_real_trigger_and_forced_counts(self):
        result=diagnostic.run_diagnostic(
            seeds=(3101,),regimes=("source_signflip",),
            modes=("periodic","contradiction"),batch=2,max_seconds=45)
        self.assertEqual(result["status"],
                         "PASS_DEVELOPMENT_MECHANISM_DIAGNOSTIC_EXECUTED")
        self.assertEqual(result["summary"]["status"],
                         "PASS_PARTIAL_DEVELOPMENT_MECHANICS_ONLY")
        self.assertEqual(len(result["rows"]),2)
        first,second=result["rows"]
        self.assertEqual(first["stream_sha256"],second["stream_sha256"])
        self.assertEqual(first["slow_checkpoint_sha256"],
                         second["slow_checkpoint_sha256"])
        for row in result["rows"]:
            self.assertEqual(row["real_fast_weight_update_worlds"],16)
            self.assertEqual(row["observed_action_pairs"],128)
            self.assertEqual(sum(row["updates_by_time"]),16)
            self.assertEqual(row["selected_trigger_steps"]+
                             row["forced_budget_updates"],16)
            self.assertEqual(len(row["updates_by_time"]),32)
        self.assertTrue(result["untrained_slow_weights"])
        self.assertEqual(result["scientific_qualification"],"NOT_SCIENTIFIC_PASS")

    def test_no_shuffled_input_or_frozen_science_significance_claim(self):
        content=(ROOT/"scripts/g7_s25_gate_diagnostic.py").read_text()
        tree=ast.parse(content)
        call_names=[x.func.attr for x in ast.walk(tree)
                    if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
        for forbidden in ("fit_calibrated","backward","optimizer",
                          "zero_grad","train","load_state_dict"):
            self.assertNotIn(forbidden,call_names)
        self.assertNotIn("torch.optim",content)
        self.assertNotIn("SCIENTIFIC_PASS",content.replace("NOT_SCIENTIFIC_PASS",""))

    def test_strict_budget_and_duplicate_seed_guards(self):
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT_BATCH_CAP"):
            diagnostic.run_diagnostic(batch=9)
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT_SEEDS_ONLY"):
            diagnostic.run_diagnostic(seeds=(3101,3101))
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT_REGIMES_ONLY"):
            diagnostic.run_diagnostic(regimes=(
                "source_stable","source_stable"))
        with self.assertRaisesRegex(ValueError,"DIAGNOSTIC_TIME_LIMIT"):
            diagnostic.run_diagnostic(max_seconds=0)

    def test_isolated_python_entrypoint_imports_exact_repo_source(self):
        # Executes only argparse --help, not the development world study.
        p=subprocess.run(
            [sys.executable,"-I","-B",
             str(ROOT/"scripts/g7_s25_gate_diagnostic.py"),"--help"],
            cwd=ROOT,capture_output=True,text=True,timeout=25,check=False)
        self.assertEqual(p.returncode,0,p.stderr[-500:])
        self.assertIn("--expected-sha",p.stdout)
        self.assertIn("--receipt",p.stdout)

    def test_missing_modes_block_mechanism_summary(self):
        rows=fake_rows(selected=1,post_selected=1)
        with self.assertRaisesRegex(ValueError,
                                    "DIAGNOSTIC_MISSING_SEED_OR_REGIME"):
            diagnostic.summarize(rows[:-1])


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(
        stream=sys.stdout,verbosity=1))
