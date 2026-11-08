"""G7-S26 V2 composite source: strict, pre-science engineering custody.

Run on an isolated checkout. This uses standard-library only and explicitly
never invokes frozen scientific worlds, model fitting or any optimizer.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"experiments/g7_s26_v2_engineering_overlay_custody.json"


def git(*argv):
    p=subprocess.run(["git","-C",str(ROOT),*argv],
                     text=True,capture_output=True,timeout=16,check=False)
    if p.returncode!=0:
        raise AssertionError(f"G7_S26_V2_GIT_FAIL {argv}: {p.stderr}")
    return p.stdout.strip()


class S26V2EngineeringCustody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=json.loads(REG.read_text(encoding="utf-8"))

    def test_registered_scope_and_one_distinct_scientific_source(self):
        p=self.p
        self.assertEqual(p["schema"],"ttt-hybrid.g7-s26.v2-engineering-overlay-custody.v1")
        self.assertEqual(p["original_s26_head"],"b8f4bc6628e15057b76dc1f085c6067b972cd5f9")
        self.assertEqual(p["qualified_portable_v2_head"],"55f2dd4cd2699e6910a727ab691f3e26ccc3222e")
        self.assertEqual(p["original_s25_head"],"108808f7b685107a76fae1ce5605b7ff9e2bac27")
        self.assertNotEqual(p["original_s20_physics_blob"],
                            p["derivative_s20_physics_blob"])
        self.assertEqual(p["scientific_first_use"],"NOT_AUTHORIZED_NOT_RUN")

    def test_all_original_s26_source_blobs_still_exact(self):
        p=self.p
        blobs=p["original_s26_additive_blobs"]
        self.assertEqual(len(blobs),15)
        one="scripts/g7_s26_dev_comparison.py"
        self.assertEqual(blobs[one],p["composite_source_delta"]["original_blob"])
        for path,digest in blobs.items():
            observed=git("hash-object","--",path)
            expected=(p["composite_source_delta"]["revised_blob"]
                      if path==one else digest)
            with self.subTest(path=path):
                self.assertEqual(observed,expected)

    def test_original_frozen_science_and_source_pins_are_unmodified(self):
        p=self.p
        for path,key in (
            ("experiments/g7_s25_frozen_contradiction_ttt.json",
             "original_s25_frozen_protocol_blob"),
            ("experiments/g7_s26_frozen_quarter_reserve.json",
             "original_s26_frozen_protocol_blob"),
            ("source_pins.json","original_source_pins_blob"),
        ):
            with self.subTest(path=path):
                self.assertEqual(git("hash-object","--",path),p[key])
        self.assertEqual(git("hash-object","--",
                            "scripts/g7_s20_hidden_multicycle.py"),
                         p["derivative_s20_physics_blob"])
        self.assertNotEqual(p["original_s20_physics_blob"],
                            git("hash-object","--","scripts/g7_s20_hidden_multicycle.py"))

    def test_no_drift_in_v2_phyics_models_and_ledger(self):
        previous={
          "scripts/g7_s16_nonlinear_switch.py":
              "868c9ffb91f62279c1a6a21cf63e874dbf501b1e",
          "scripts/g7_s18_causal_feature_replication.py":
              "aa09ac6bc2787691e349889f96c1b791e2df0b64",
          "scripts/g7_s20_hidden_multicycle.py":
              "bc3e9f255a4d1b967252ada8782ac9f8c000f7b7",
          "scripts/g7_s25_cost_ledger.py":
              "e2bea80e39913892c79e9ad715782df1ba0f3507",
        }
        for path,digest in previous.items():
            with self.subTest(path=path):
                self.assertEqual(git("hash-object","--",path),digest)

    def test_qualified_v2_ancestor_and_exact_additions_only(self):
        v2=self.p["qualified_portable_v2_head"]
        git("merge-base","--is-ancestor",v2,"HEAD")
        paths=set(git("diff","--name-only",v2,"HEAD").splitlines())
        expected=set(self.p["original_s26_additive_blobs"]) | {
          "experiments/g7_s26_v2_engineering_overlay_custody.json",
          "tests/test_g7_s26_v2_engineering_custody.py",
          ".github/workflows/g7-s26-v2-engineering-once.yml",
          "scripts/g7_s26_v2_baseline_observations.py",
          "tests/test_g7_s26_v2_baseline_observations.py",
          ".github/workflows/g7-s26-v2-baseline-input-once.yml",
        }
        self.assertEqual(paths,expected)
        self.assertEqual(len(paths),21)
        self.assertEqual(git("status","--porcelain","--untracked-files=all"),"")

    def test_separate_dev_comparator_has_no_unix_resource_import(self):
        script=(ROOT/"scripts/g7_s26_dev_comparison.py").read_text(encoding="utf-8")
        tree=ast.parse(script)
        imports=[v for v in tree.body if isinstance(v,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(
            (isinstance(v,ast.Import) and
             any(a.name=="resource" for a in v.names)) or
            (isinstance(v,ast.ImportFrom) and v.module=="resource")
            for v in imports))
        self.assertIn("from g7_process_peak_memory import read_peak_memory",script)
        self.assertIn("dev_peak_memory_comparable_to_linux_ru_maxrss",script)
        self.assertIn('if memory.comparable_to_linux_ru_maxrss else None',script)
        self.assertIn('"dev_peak_working_set_bytes"',script)

    def test_all_science_seed_namespaces_unconsumed(self):
        p=self.p
        self.assertEqual(p["development_seed_set_only"],[3101,3109,3119])
        self.assertEqual(p["training_or_fit"],"NOT_AUTHORIZED_NOT_RUN")
        self.assertEqual(p["heldout_or_transfer_seed_use"],"NOT_AUTHORIZED_NOT_RUN")
        self.assertEqual(p["accepted_ref_or_merge"],"NOT_AUTHORIZED")
        science=json.loads((ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())
        self.assertFalse(science["authority"]["scientific_first_use_authorized"])
        self.assertTrue(science["authority"]["no_scientific_training_or_fit"])
        self.assertTrue(science["authority"]["no_heldout_seed_consumption"])

    def test_original_s25_and_s26_source_admissions_remain_fail_closed(self):
        original=(ROOT/"scripts/g7_s25_engineering_qualifier.py").read_text()
        original_s26=(ROOT/"scripts/g7_s26_baseline_direction_admission.py").read_text()
        protocol=json.loads((ROOT/"experiments/g7_s26_frozen_quarter_reserve.json").read_text())
        old=self.p["original_s20_physics_blob"]
        self.assertIn(old,original)
        self.assertIn(old,original_s26)
        self.assertEqual(protocol["rationale"]["s20_world_git_blob"],old)
        self.assertNotIn(self.p["derivative_s20_physics_blob"],original_s26)


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=1))
