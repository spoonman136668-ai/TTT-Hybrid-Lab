"""Private supplier source, replay and authority falsifiers. No model calls."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from scripts.native_ttt_write_admission import (
    choose_ttt_update, forward_existing_hybrid,
)
from scripts.native_ttt_candidate_supplier import (
    BASE_BLOB, PARENT, PATH, PrivateCandidateDenied, git_blob,
    mechanism_id, propose_private, sha,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / PATH
HISTORY = "e" * 64


def source_git_bytes():
    """Use canonical Git LF blob bytes across Linux/Windows worktree settings.

    The supplier's immutable BASE_BLOB check remains strict. Windows checkout
    may present CRLF working bytes; only this test fixture normalizes them.
    """
    return SOURCE_PATH.read_bytes().replace(b"\r\n", b"\n")


def build(**change):
    before = source_git_bytes()
    args = {
        "parent_commit": PARENT, "source_path": PATH,
        "original_bytes": before,
        "original_blob": BASE_BLOB,
        "original_sha256": hashlib.sha256(before).hexdigest(),
        "original_owner_history_sha256": HISTORY,
        "consumed_mechanisms": set(),
        "consumed_candidate_sha256": set(),
    }
    args.update(change)
    return propose_private(**args)


class FakeModel:
    def __init__(self):
        self.calls = []

    def forward(self, x, ttt_update=True):
        self.calls.append((x, ttt_update))
        return "ok"


class PrivateHybridSupplierTests(unittest.TestCase):
    def test_source_lineage_is_project_owned(self):
        original = source_git_bytes()
        self.assertEqual(git_blob(original), BASE_BLOB)
        self.assertEqual(len(PARENT), 40)

    def test_original_all_write_default_is_unchanged(self):
        self.assertTrue(choose_ttt_update(
            previous_observed_error=0, previous_evidence_confidence=0))
        model = FakeModel()
        self.assertEqual(forward_existing_hybrid(
            model, "fixture", previous_observed_error=0,
            previous_evidence_confidence=0), "ok")
        self.assertEqual(model.calls, [("fixture", True)])

    def test_proposal_is_deterministic(self):
        left, right = build(), build()
        self.assertEqual(left, right)
        self.assertNotEqual(left["private_source"], source_git_bytes())
        self.assertEqual(left["candidate_sha256"], sha(left["private_source"]))

    def test_candidate_changes_only_opt_in_sequence_gate(self):
        ns = {}
        exec(compile(build()["private_source"], PATH, "exec"), ns)
        gate = ns["choose_ttt_update"]
        self.assertTrue(gate(previous_observed_error=.8,
                             previous_evidence_confidence=.9,
                             mode="observed-contradiction"))
        self.assertFalse(gate(previous_observed_error=.1,
                              previous_evidence_confidence=.9,
                              mode="observed-contradiction"))
        self.assertFalse(gate(previous_observed_error=.9,
                              previous_evidence_confidence=.1,
                              mode="observed-contradiction"))
        self.assertTrue(gate(previous_observed_error=0,
                             previous_evidence_confidence=0,
                             mode="all-write"))

    def test_public_identity_never_exposes_private_source(self):
        result = build()
        public = result["public_identity"]
        self.assertFalse(public["private_provider_authenticated"])
        self.assertFalse(public["owner_signed"])
        self.assertFalse(public["original_worker_qualified"])
        self.assertFalse(public["scientific_first_use"])
        self.assertFalse(public["deployment_authorized"])
        self.assertNotIn("private_source", public)
        self.assertNotIn("replacement", public)
        self.assertRegex(result["public_identity_sha256"], "^[a-f0-9]{64}$")

    def test_duplicate_recipe_denied(self):
        with self.assertRaisesRegex(PrivateCandidateDenied, "MECHANISM_ALREADY_CONSUMED"):
            build(consumed_mechanisms={mechanism_id()})

    def test_consumed_candidate_denied(self):
        with self.assertRaisesRegex(PrivateCandidateDenied, "SOURCE_ALREADY_CONSUMED"):
            build(consumed_candidate_sha256={build()["candidate_sha256"]})

    def test_private_target_ancestry_cannot_drift(self):
        for kwargs in ({"parent_commit": "0" * 40},
                       {"source_path": "model.py"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(PrivateCandidateDenied, "LINEAGE_DRIFT"):
                    build(**kwargs)

    def test_source_identity_and_history_cannot_drift(self):
        for kwargs in ({"original_blob": "0" * 40},
                       {"original_sha256": "0" * 64},
                       {"original_owner_history_sha256": "0"},
                       {"consumed_mechanisms": {"malformed"}},
                       {"scientific_first_use": True}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(PrivateCandidateDenied, "ORIGIN_OR_AUTHORITY"):
                    build(**kwargs)

    def test_incomplete_seed_or_source_is_denied(self):
        raw = source_git_bytes()
        for old in (b'    if mode == "all-write":', b"def forward_existing_hybrid("):
            broken = raw.replace(old, b"# deliberately removed")
            with self.subTest(old=old):
                with self.assertRaisesRegex(PrivateCandidateDenied, "ORIGIN_OR_AUTHORITY"):
                    build(original_bytes=broken)

    def test_observation_validation_fail_closed_before_model(self):
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "HYBRID_GATE_OBSERVATION_INVALID"):
            forward_existing_hybrid(
                model, "fixture", previous_observed_error=float("nan"),
                previous_evidence_confidence=.9)
        self.assertFalse(model.calls)

    def test_unknown_mode_denied_without_call(self):
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "HYBRID_GATE_MODE_UNQUALIFIED"):
            forward_existing_hybrid(
                model, "fixture", previous_observed_error=.2,
                previous_evidence_confidence=.9, mode="untrusted")
        self.assertFalse(model.calls)


if __name__ == "__main__":
    unittest.main()
