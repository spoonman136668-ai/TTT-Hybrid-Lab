"""Source-only V2 lineage negative controls; no science first-use or training."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import g7_s25_versioned_lineage_audit as lineage


def original(path):
    p=subprocess.run(["git","-C",str(ROOT),"show",
                      f"{lineage.FROZEN_BASE}:{path}"],
                     capture_output=True,text=True,check=False,timeout=14)
    if p.returncode:
        raise RuntimeError("MISSING_FROZEN_BLOB")
    return p.stdout


class TestS25V2Lineage(unittest.TestCase):

    def test_registered_original_and_derivative_blobs_all_distinct(self):
        self.assertEqual(len(lineage.DERIVATIVE_BLOBS),4)
        self.assertEqual(len(lineage.SCIENTIFIC_MODULES),3)
        for name in lineage.DERIVATIVE_BLOBS:
            self.assertNotEqual(lineage.DERIVATIVE_BLOBS[name],
                                lineage.FROZEN_BLOBS[name])
            self.assertEqual(lineage.blob_digest(original(name)),
                             lineage.FROZEN_BLOBS[name])
            self.assertEqual(lineage.blob_digest((ROOT/name).read_text()),
                             lineage.DERIVATIVE_BLOBS[name])

    def test_exact_whitelist_transforms_reconstruct_four_v2_files(self):
        for name in lineage.DERIVATIVE_BLOBS:
            with self.subTest(source=name):
                before=original(name)
                after=(ROOT/name).read_text()
                result=lineage.verify(name,before,after)
                self.assertTrue(result["physics_and_model_code_exactly_preserved"])
                self.assertFalse(result["original_scientific_source_pin_matches_derivative"])
                self.assertGreaterEqual(result["allowed_replacements"],4)

    def test_mutated_physics_or_model_body_fails_closed(self):
        pairs=[
            ("scripts/g7_s16_nonlinear_switch.py",
             "def basis(torch,x,u,kind=\"nonlinear\"):",
             "def basis(torch,x,u,kind=\"tampered\"):"),
            ("scripts/g7_s18_causal_feature_replication.py",
             "class RawGRU(nn.Module):",
             "class TamperedRawGRU(nn.Module):"),
            ("scripts/g7_s20_hidden_multicycle.py",
             "def coefficients(gen,b,ood):",
             "def coefficients(gen,b,oracle):"),
            ("scripts/g7_s25_cost_ledger.py",
             "FAST_WIDTH - 1", "FAST_WIDTH + 1"),
        ]
        for path,needle,replacement in pairs:
            with self.subTest(path=path):
                original_bytes=original(path)
                edited=(ROOT/path).read_text()
                self.assertIn(needle,edited)
                with self.assertRaisesRegex(lineage.LineageBlocked,
                                           "DERIVATIVE_BLOB_CHANGED"):
                    lineage.verify(path,original_bytes,
                                   edited.replace(needle,replacement,1))

    def test_source_pin_and_frozen_seed_protocol_cannot_be_rebased(self):
        self.assertEqual(lineage.FROZEN_BASE,
                         "108808f7b685107a76fae1ce5605b7ff9e2bac27")
        self.assertEqual(lineage.FROZEN_BLOBS[
                         "experiments/g7_s25_frozen_contradiction_ttt.json"],
                         "31633b341403729b7ffbb7d73f3a04098ad44380")
        self.assertEqual(lineage.FROZEN_BLOBS[
                         "source_pins.json"],
                         "1438b51f8548db6bfe7e04805ba8c156e4b5b630")
        self.assertNotEqual(lineage.FROZEN_BLOBS[
                            "scripts/g7_s20_hidden_multicycle.py"],
                            lineage.DERIVATIVE_BLOBS[
                            "scripts/g7_s20_hidden_multicycle.py"])

    def test_nonlinux_science_guard_first_statement_each_run(self):
        for path in lineage.SCIENTIFIC_MODULES:
            with self.subTest(path=path):
                tree=ast.parse((ROOT/path).read_text())
                fn=next(n for n in tree.body
                        if isinstance(n,ast.FunctionDef) and n.name=="run")
                self.assertIsInstance(fn.body[0],ast.Expr)
                self.assertIsInstance(fn.body[0].value,ast.Call)
                self.assertNotIn("protocol.read_bytes",
                                 ast.unparse(fn.body[0]))
                self.assertIn("FROZEN_RSS_REQUIRES_LINUX",
                              ast.unparse(fn.body[0]))

    def test_deleted_nonlinux_guard_fails_exact_change_contract(self):
        path="scripts/g7_s20_hidden_multicycle.py"
        before=original(path)
        after=(ROOT/path).read_text()
        line='    require(sys.platform == "linux", "S20_FROZEN_RSS_REQUIRES_LINUX")\n'
        self.assertIn(line,after)
        with self.assertRaisesRegex(lineage.LineageBlocked,
                                    "DERIVATIVE_BLOB_CHANGED"):
            lineage.verify(path,before,after.replace(line,"",1))

    def test_unregistered_path_and_different_frozen_original_fails(self):
        with self.assertRaisesRegex(lineage.LineageBlocked,"UNREGISTERED_PATH"):
            lineage.verify("scripts/g7_s25_numeric_ttt.py","","")
        p="scripts/g7_s16_nonlinear_switch.py"
        with self.assertRaisesRegex(lineage.LineageBlocked,
                                    "FROZEN_SOURCE_BLOB_CHANGED"):
            lineage.verify(p,original(p)+"# altered\n",(ROOT/p).read_text())

    def test_source_only_audit_never_claims_science(self):
        import inspect
        tree=ast.parse((ROOT/"scripts/g7_s25_versioned_lineage_audit.py").read_text())
        calls={node.func.attr for node in ast.walk(tree)
               if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Attribute)}
        self.assertFalse(set(("train","backward","fit","optimizer_step",
                              "evaluate_heldout")).intersection(calls))
        self.assertNotIn("torch",inspect.getsource(lineage))
        self.assertNotIn("scientific_first_use_authorized",inspect.getsource(lineage))


if __name__=="__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=1))
