"""R5 engineering checks on fabricated histories only; never real F3 data."""
from pathlib import Path
import math
import pytest
import torch
from torch.nn import functional as F
import experiments.p78mr_r5_trainable_residual as r5


def toy(n=3, steps=7):
    rng = torch.Generator().manual_seed(7)
    actions = F.normalize(torch.randn(n, steps, 16, generator=rng), dim=-1)
    targets = torch.randn(n, steps, 16, generator=rng) * 0.04
    return [(actions[i], targets[i], torch.zeros_like(targets[i]))
            for i in range(n)]


def test_frozen_before_training_and_no_new_source_permission():
    f = r5.locked()
    assert f["stage"] == "PRECOMMIT_BEFORE_ANY_R5_TRAINING_OR_CAL_SCORING"
    assert not f["authority"]["new_science_source"]
    assert not f["authority"]["owner_first_use"]
    assert not f["authority"]["protected_ood"]
    assert f["source"]["cal_scored_once_after_selection"]


def test_zero_memory_is_exact_residual_baseline():
    a, y, mask = r5.pack(toy())
    original = y[:, 1:, :4].square().mean(-1).mean(-1).mean()
    actual = r5.residual_loss(None, (a, y, mask), fixed_gain=0.)
    torch.testing.assert_close(actual, original, atol=1e-9, rtol=0)


def test_three_trainable_scalars_have_finite_nonzero_gradients():
    theta = torch.nn.Parameter(torch.tensor([0.02, -.04, .05]))
    loss = r5.residual_loss(theta, r5.pack(toy(4)))
    loss.backward()
    assert theta.numel() == 3
    assert theta.grad is not None
    assert bool(torch.isfinite(theta.grad).all())
    assert theta.grad.abs().sum() > 0


def test_invalid_facts_and_unregistered_gain_fail_closed():
    a, y, mask = r5.pack(toy())
    with pytest.raises(ValueError, match="INVALID_FIXED_GAIN"):
        r5.residual_loss(None, (a, y, mask), fixed_gain=.33)
    with pytest.raises(ValueError, match="MISSING_OR_INVALID_THETA"):
        r5.residual_loss(torch.zeros(5), (a, y, mask))
    nan = a.clone()
    nan[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="BAD_BATCH"):
        r5.residual_loss(None, (nan, y, mask), fixed_gain=.4)
    with pytest.raises(ValueError, match="BAD_EPISODE"):
        r5.pack([])
    t = toy(1)[0]
    with pytest.raises(ValueError, match="BAD_FACTS_OR_UNITS"):
        r5.pack([(2*t[0], t[1], t[2])])


def test_batched_padding_excludes_nonexistent_factual_events():
    original = toy(2, 6)
    short = tuple(x[:2] for x in original[1])
    theta = torch.tensor([-.01, -.2, .3])
    actual = float(r5.residual_loss(theta, r5.pack([original[0], short])))
    expected = (r5.evaluate([original[0]], theta=theta) +
                r5.evaluate([short], theta=theta)) / 2
    assert actual == pytest.approx(expected, rel=1e-5)


def test_miniature_three_seed_fit_uses_only_fake_FIT(monkeypatch):
    # Reduced testing duration; production lock rejects modified constants.
    monkeypatch.setattr(r5, "STEPS", 4)
    monkeypatch.setattr(r5, "BATCH", 4)
    monkeypatch.setattr(r5, "CHECK", 2)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        made = toy(128, 3)
        result = r5.fit_only(made[:96], made[96:])
    finally:
        torch.set_num_threads(original_threads)
    assert result["selected_eta"] in r5.GRID
    assert len(result["models"]) == 3
    assert {x["seed"] for x in result["models"]} == set(r5.SEEDS)
    assert all(x["selected_step"] in (2, 4) and
               len(x["selected_theta"]) == 3 and
               math.isfinite(x["fitdev_mse"]) and
               len(x["trace"]) == 2 for x in result["models"])


def test_untrusted_original_artifact_denied(tmp_path):
    candidate = tmp_path / "spoof.zip"
    candidate.write_bytes(b"no owner provenance")
    with pytest.raises(ValueError, match="PRIOR_F3_ARCHIVE_DRIFT"):
        r5.read_consumed_fit(candidate)


def test_no_simulator_reset_owner_dispatch_or_automatic_training():
    script = (Path(__file__).resolve().parents[1] /
              "experiments" / "p78mr_r5_trainable_residual.py").read_text()
    for forbidden in ("env.reset(", "env.do(", "make_world(", "make_batch(",
                      "torch.save(", "requests.get(", "urlopen(", "subprocess.run("):
        assert forbidden not in script
