"""Source-neutral F7 recovery must never create additional source actions."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from experiments import p85mr_f7_training_recovery as recovery
from experiments import p85mr_f7_train_fit as trainer


def test_recovery_protocol_original_source_spent_and_frozen():
    x=recovery.locked()
    assert x["original_collection"]["owner_run"]==43195349254
    assert x["original_collection"]["source_actions_consumed"]==12288
    assert x["original_collection"]["never_retry_source"] is True
    assert x["recovery"]["one_shot_only"] is True
    assert x["recovery"]["no_upstream_checkout"] is True
    assert x["science"]["fit_dev_only"] is True
    assert not x["authority"]["new_external_firstuse"]
    assert x["authority"]["no_CAL_or_OOD"] is True
    assert x["science"]["scientific_PASS_possible"] is False
    with pytest.raises(PermissionError,match="F7_RECOVERY_NOT_UNIQUE_TRAINING_OWNER"):
        recovery.owner({})


def test_recovery_refuses_unauthenticated_source_and_cannot_write_effect(tmp_path):
    fake={"source_worlds":128,"authentic_source_actions":12288,
          "source_capture_sha256":"0"*64,"native_owner_journal_audit":True}
    with pytest.raises(PermissionError,match="F7_TRAIN_UNQUALIFIED_OR_NONAUTHENTIC_FIT_SOURCE"):
        trainer.train_admitted_fit(train=[None]*96,holdout=[None]*32,
             source_meta=fake,owner={},output=tmp_path/"models",
             receipt=tmp_path/"receipt.json")
    with pytest.raises(PermissionError,match="F7_TRAIN_UNQUALIFIED_OR_NONAUTHENTIC_FIT_SOURCE"):
        trainer.train_admitted_fit(train=[],holdout=[],source_meta={},
             owner={},output=tmp_path/"models",receipt=tmp_path/"receipt.json")


def test_original_owner_summary_digest_matches_archived_source():
    f=recovery.locked()
    assert recovery.CAPTURE_SHA==f["original_collection"]["source_capture_sha256"]
    assert recovery.SUMMARY_SHA==f["original_collection"]["source_summary_sha256"]
    assert recovery.ORIGINAL_SHA==f["original_collection"]["sha"]
    assert recovery.SOURCE_OWNER["run_id"]==str(f["original_collection"]["owner_run"])
    assert recovery.RECOVERY_ROOT!=recovery.SOURCE_ROOT


def test_no_effectful_or_external_import_entrypoint_in_training_recovery():
    body=Path(recovery.__file__).read_text(encoding="utf-8")
    for forbidden in ("env.do(","env.reset(","from causal_gym","import causal_gym",
        "dispatch_workflow(","torch.load(","subprocess.run(","import requests",
        "workflow_dispatch(","start_new_source(","_load_scoped_source("):
        assert forbidden not in body
