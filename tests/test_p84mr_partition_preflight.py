"""P84-MR proof of disjoint scientific source seeds; zero external first-use."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(ROOT))

from p84mr_partition_preflight import (
    FREEZE_GIT_BLOB, NAMESPACE, SPLITS, NO_REAL_SOURCE_FIRST_USE,
    _collisions, _locked_freeze, _ranges, forbid_first_use,
    planned_discrete_action, preflight_partition, proposed_seed,
)


def test_exact_frozen_source_born_before_implementation():
    freeze = _locked_freeze()
    assert freeze["schema"].endswith("science-preflight-freeze.v1")
    assert freeze["execution_authority"]["first_use_status"] == "NOT_AUTHORIZED"
    path = ROOT / "p84mr_external_science_preflight_freeze.json"
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assert blob == FREEZE_GIT_BLOB
    assert freeze["frozen_successor"]["namespace"] == NAMESPACE


def test_reproduce_real_historical_127_seed_overlaps():
    freeze = _locked_freeze()
    old = freeze["preexisting_protocol_negative_control"]
    schedules = _ranges(old["previous_split_bases"], old["previous_counts"], stride=10)
    pairs = _collisions(schedules)
    assert pairs == [
        {"a": "fit", "b": "science_ood_1", "overlap": 64, "first": 811111},
        {"a": "cal", "b": "science_ood_2", "overlap": 63, "first": 811113},
    ]
    assert sum(v["overlap"] for v in pairs) == 127
    assert len(set.union(*(set(v) for v in schedules.values()))) == 321


def test_all_448_new_episode_seeds_unique_and_nonoverlapping():
    result = preflight_partition()
    assert result["status"] == "OFFLINE_SEED_PREFLIGHT_PASS_NO_SOURCE_BIRTH"
    assert result["legacy_total_cross_split_collisions"] == 127
    assert result["new_episode_count"] == 448
    assert result["new_unique_seeds"] == 448
    assert result["cross_split_collisions"] == 0
    assert len(result["proposed_schedule_sha256"]) == 64
    assert result["external_data_generated"] is False
    assert result["scientific_pass"] is False


def test_new_seed_windows_cannot_overlap_even_at_boundaries():
    freeze = _locked_freeze()["frozen_successor"]
    schedule = _ranges(freeze["seed_bases"], freeze["episode_counts"])
    assert len(set.union(*(set(v) for v in schedule.values()))) == 448
    assert not _collisions(schedule)
    assert proposed_seed("fit", 0) == 1810000000
    assert proposed_seed("fit", 127) == 1810000127
    assert proposed_seed("cal", 0) == 1810100000
    assert proposed_seed("dev_external", 0) == 1810200000
    assert proposed_seed("science_ood_0", 0) == 1810300000
    assert proposed_seed("science_ood_1", 0) == 1810400000
    assert proposed_seed("science_ood_2", 63) == 1810500063
    assert len(SPLITS) == 6


def test_partition_calibration_action_plan_deterministic():
    assert planned_discrete_action("fit", 0, 1) in (0, 1)
    for split in SPLITS:
        for ix in (0, 1):
            a = [planned_discrete_action(split, ix, s) for s in range(1, 97)]
            b = [planned_discrete_action(split, ix, s) for s in range(1, 97)]
            assert a == b and set(a) == {0, 1}


@pytest.mark.parametrize("bad", [
    ("FIT", 0), ("unknown", 0), ("fit", -1),
    ("fit", 128), ("cal", 64), ("science_ood_2", 64),
    ("fit", 0.0), ("fit", True), ("fit", None),
])
def test_nonregistered_source_seeds_denied(bad):
    with pytest.raises(ValueError, match="P84MR_UNREGISTERED_EPISODE"):
        proposed_seed(*bad)


@pytest.mark.parametrize("bad_step", [0, -1, 97, True, 2.0, None])
def test_action_plan_outside_exact_bounds_denied(bad_step):
    with pytest.raises(ValueError, match="P84MR_BAD_ACTION_STEP"):
        planned_discrete_action("fit", 0, bad_step)


def test_old_protocol_rejected_when_substituted_into_preflight(monkeypatch):
    frozen = _locked_freeze()
    fake = json.loads(json.dumps(frozen))
    previous = fake["preexisting_protocol_negative_control"]
    replacement = fake["frozen_successor"]
    replacement["seed_bases"] = previous["previous_split_bases"]
    # Old stepped 10; this test also checks collisions when revised as a
    # purported contiguous independent source without changing cohort widths.
    replacement["seed_bases"]["science_ood_1"] = replacement["seed_bases"]["fit"]
    import p84mr_partition_preflight as mod
    monkeypatch.setattr(mod, "_locked_freeze", lambda: fake)
    with pytest.raises(ValueError, match="P84MR_CROSS_SPLIT_SEED_COLLISION"):
        mod.preflight_partition()


def test_seed_exhaustion_rejected_even_if_freeze_data_modified(monkeypatch):
    original = _locked_freeze()
    fake = json.loads(json.dumps(original))
    fake["frozen_successor"]["episode_counts"]["fit"] = 129
    fake["frozen_successor"]["expected_unique"] = 449
    import p84mr_partition_preflight as mod
    monkeypatch.setattr(mod, "_locked_freeze", lambda: fake)
    with pytest.raises(ValueError, match="P84MR_EPISODE_TOTAL_DRIFT"):
        mod.preflight_partition()


def test_permission_refuses_science_execution_independently_of_caller_labels():
    for labels in (
        (),
        ("external_simulator",),
        ({"source_kind": "toy_engineering", "signed": True},),
    ):
        with pytest.raises(PermissionError, match=NO_REAL_SOURCE_FIRST_USE):
            forbid_first_use(*labels, source_kind="external_simulator")


def test_code_has_no_external_import_or_source_scoring_entrypoint():
    body = (ROOT / "p84mr_partition_preflight.py").read_text()
    for forbidden in (
        "import gymnasium", "import causal_gym",
        "env.reset(", "env.do(", "env.step(", "torch",
        "requests.", "subprocess", "urllib", "if __name__",
        "openai", "run_science(", "model.predict(",
    ):
        assert forbidden not in body, forbidden


def test_new_regimes_are_frozen_but_not_labels_for_model():
    config = _locked_freeze()
    regimes = config["upstream_regimes"]
    assert regimes["fit"] == regimes["cal"]
    assert regimes["science_ood_0"]["wind_mean"] == 0.08
    assert regimes["science_ood_1"]["wind_mean"] == -0.08
    assert regimes["science_ood_2"]["wind_std"] == 0.05
    assert config["metrics_and_outcomes"]["primary"].startswith("Normalized four-channel")
