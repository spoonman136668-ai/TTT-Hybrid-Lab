"""F8-R2: fabricated baseline fixtures and source/authority negative tests."""
from pathlib import Path
import numpy as np
import pytest
from experiments import p85mr_f8r2_linear_f7_retrospective as lin


def toy():
    rng = np.random.default_rng(8125)
    x = rng.normal(size=(6, 5, 7))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = x @ rng.normal(size=(7, 4))
    return x, y


def test_freeze_precedes_original_source_access():
    f = lin.locked()
    assert f["stage"] == "FROZEN_BEFORE_R2_LINEAR_FIT_OR_HELDOUT_RESCORING"
    assert f["source"]["original_worlds"] == 128
    assert f["source"]["original_actions"] == 12288
    assert f["authority"]["new_source_first_use"] is False
    assert f["authority"]["original_F7_owner_replay"] is False
    assert f["authority"]["closed_form_baseline_on_spent_source"] is True


def test_linear_fit_reconstructs_fictional_action_conditioned_dynamics():
    x, y = toy()
    w = lin.fit(x[:4], y[:4], 0.)
    assert w.shape == (7, 4)
    assert lin.errors(x[4:], y[4:], w).mean() < 1e-17
    assert lin.features(x, "quadratic35").shape == (6, 5, 35)
    assert lin.features(x, "state5").shape[-1] == 5
    assert lin.features(x, "action3").shape[-1] == 3


def test_wrong_information_and_numerical_poison_are_denied():
    x, y = toy()
    with pytest.raises(ValueError, match="UNREGISTERED_ARM"):
        lin.features(x, "future_effect")
    with pytest.raises(ValueError, match="INVALID_FIT"):
        lin.fit(x, y, -1)
    with pytest.raises(ValueError, match="INVALID_FIT"):
        lin.fit(x, y[..., :3], 0)
    poisoned = x.copy()
    poisoned[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="INVALID_FIT"):
        lin.fit(poisoned, y, 0.)


def test_spoofed_original_archives_rejected_before_analysis(tmp_path):
    source = tmp_path / "f7-fake.zip"
    trained = tmp_path / "models-fake.zip"
    source.write_bytes(b"fake")
    trained.write_bytes(b"fake")
    with pytest.raises(ValueError, match="ORIGINAL_ARTIFACT_DRIFT"):
        lin.read_spent(source, trained)


def test_no_simulator_or_owner_source_replay_possible_from_module():
    source = (Path(__file__).resolve().parents[1] /
              "experiments/p85mr_f8r2_linear_f7_retrospective.py").read_text()
    for forbidden in ("causal_gym", "env.reset(", "env.do(", "optimizer.step(",
                      "torch.save(", "subprocess.run(", "requests.get(",
                      "urlopen(", "workflow_dispatch("):
        assert forbidden not in source
