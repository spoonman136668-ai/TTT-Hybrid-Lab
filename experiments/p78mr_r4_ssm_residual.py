"""P78-MR R4: frozen trained SSM plus bounded factual residual memory.

RETROSPECTIVE only on original already-consumed F3 CAL, not independent science.
No simulator, data birth, training, optimizer, protected OOD or owner dispatch.
Reuse the original P69 SSM and P77/P78 update implementation without editing
their source or original checkpoints. No original F3/F4 first-use replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p69_trainable_competitors import DiagonalContinuousSSM  # noqa: E402
from p77_write_policy import WritePolicy  # noqa: E402
from p78mr_adaptive_controls import new_learned  # noqa: E402
from p78mr_gain_posthoc_audit import audit as frozen_theta_audit, read_locked  # noqa: E402
from p78mr_r3_f3_transfer import (  # noqa: E402
    read_prior_cal, read_f4_controls, factual_event, F4_ZIP,
)

FREEZE_PATH = HERE / "p78mr_r4_frozen_ssm_residual_freeze.json"
FREEZE_BLOB = "e29997529ad87560ea8a2caacfb4152ae8c1da50"
SSM_SEED = 985203
SSM_CHECKPOINT = "p85mr_f4_models/ssm256-985203.npz"
SSM_CHECKPOINT_SHA256 = "c6af43defa91f8cf308845d72e9b6acf15008761d88d3ede054029affcce4cc5"
SSM_F4_CAL = 1.2177126073709132e-5
P78_SEEDS = (978201, 978203, 978207)
SSM_STATE = 256
FAST_STATE = 256
ETA = 0.4


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\\0" + raw).hexdigest()


def lock_protocol() -> dict:
    raw = read_locked(FREEZE_PATH, FREEZE_BLOB)
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.p78mr-r4-trained-ssm-residual-memory-retro-v1"
            or f["trained_dynamics"]["ssm_checkpoint"] != "ssm256-985203.npz"
            or f["trained_dynamics"]["checkpoint_sha256"] != SSM_CHECKPOINT_SHA256
            or f["trained_dynamics"]["original_CAL_delta_MSE"] != SSM_F4_CAL
            or f["source"]["cal_worlds"] != 64
            or f["source"]["cal_actual_do_events"] != 1524
            or f["p78_memory"]["frozen_model_seeds"] != list(P78_SEEDS)
            or f["authority"]["first_use"] is not False
            or f["authority"]["model_train"] is not False
            or f["authority"]["OOD"] is not False):
        raise ValueError("P78MR_R4_FROZEN_PROTOCOL_DRIFT")
    return f


def frozen_ssm(controls_archive: Path) -> DiagonalContinuousSSM:
    # Full archive and 12-control source receipt verified by parent R3 reader.
    original_scores = read_f4_controls(controls_archive)
    if original_scores.get(SSM_SEED) != SSM_F4_CAL:
        raise ValueError("P78MR_R4_ORIGINAL_TRAINED_SSM_DRIFT")
    if not controls_archive.is_file() or hashlib.sha256(
            controls_archive.read_bytes()).hexdigest() != F4_ZIP:
        raise ValueError("P78MR_R4_F4_ARCHIVE_DRIFT")
    with zipfile.ZipFile(controls_archive) as z:
        raw = z.read(SSM_CHECKPOINT)
    if hashlib.sha256(raw).hexdigest() != SSM_CHECKPOINT_SHA256:
        raise ValueError("P78MR_R4_TRAINED_CHECKPOINT_DRIFT")
    with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
        if set(npz.files) != {"logit_decay", "input.weight", "input.bias",
                               "head.weight", "head.bias"}:
            raise ValueError("P78MR_R4_TRAINED_CHECKPOINT_KEYS_DRIFT")
        weight_dict = {k: torch.from_numpy(npz[k].copy()) for k in npz.files}
    model = DiagonalContinuousSSM()
    model.load_state_dict(weight_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def frozen_policies() -> dict:
    rows = frozen_theta_audit()["rows"]
    if tuple(v["seed"] for v in rows) != P78_SEEDS:
        raise ValueError("P78MR_R4_FROZEN_THETA_DRIFT")
    controls = {"ssm": None, "fixed0p4": WritePolicy("fixed_eta", eta=ETA)}
    for r in rows:
        policy = new_learned("non_directional3", r["seed"])
        with torch.no_grad():
            policy.theta.copy_(torch.tensor(r["theta"], dtype=torch.float32))
        policy.eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        controls[str(r["seed"])] = policy
    return controls


def start_states(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (not isinstance(action, torch.Tensor) or action.ndim != 2
            or action.shape[-1] != 16 or action.dtype != torch.float32
            or not bool(torch.isfinite(action).all())):
        raise ValueError("P78MR_R4_INVALID_FACTUAL_KEY")
    return (action.new_zeros((len(action), SSM_STATE)),
            action.new_zeros((len(action), 16, 16)))


def step(base: DiagonalContinuousSSM, policy: WritePolicy | None,
         ssm_state: torch.Tensor, fast_state: torch.Tensor,
         action: torch.Tensor, observed_effect: torch.Tensor):
    """Predict BEFORE writing the current observed factual effect."""
    if (not isinstance(action, torch.Tensor) or not isinstance(observed_effect, torch.Tensor)
            or action.ndim != 2 or action.shape != observed_effect.shape
            or action.shape[-1] != 16 or action.dtype != torch.float32
            or action.dtype != observed_effect.dtype
            or not bool(torch.isfinite(action).all())
            or not bool(torch.isfinite(observed_effect).all())
            or not bool((action.norm(dim=-1) - 1).abs().lt(1e-4).all())
            or ssm_state.shape != (len(action), SSM_STATE)
            or fast_state.shape != (len(action), 16, 16)
            or not bool(torch.isfinite(ssm_state).all())
            or not bool(torch.isfinite(fast_state).all())):
        raise ValueError("P78MR_R4_INVALID_FACT_OR_STATE")
    ssmpred = base.read(ssm_state, action)
    memorypred = (fast_state @ action.unsqueeze(-1)).squeeze(-1) if policy is not None else 0
    before = ssmpred + memorypred
    facts = {"action": action[:, None], "effect": observed_effect[:, None]}
    if policy is None:
        next_fast = fast_state.clone()
    else:
        residual = observed_effect - ssmpred
        next_fast = policy.encode({"action": action[:, None],
                                   "effect": residual[:, None]}, carry=fast_state)
    next_ssm = base.encode(facts, carry=ssm_state)
    if (not bool(torch.isfinite(before).all())
            or not bool(torch.isfinite(next_fast).all())
            or not bool(torch.isfinite(next_ssm).all())):
        raise ValueError("P78MR_R4_NONFINITE_UPDATE")
    return before, next_ssm, next_fast


def score_world(base: DiagonalContinuousSSM, policy: WritePolicy | None,
                world: dict) -> float:
    events = world["transitions"]
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("P78MR_R4_NOT_FACTUAL_EPISODE")
    action, _ = factual_event(events[0])
    ssm, fast = start_states(action.unsqueeze(0))
    losses = []
    with torch.no_grad():
        for t, evt in enumerate(events):
            a, y = factual_event(evt)
            pred, ssm, fast = step(base, policy, ssm, fast, a[None], y[None])
            if t:
                losses.append(float((pred[0, :4] - y[:4]).square().mean()))
    return statistics.mean(losses)


def retrospect(source_archive: Path, control_archive: Path) -> dict:
    lock_protocol()
    f3 = read_prior_cal(source_archive)
    if len(f3) != 64 or sum(len(w["transitions"]) for w in f3) != 1524:
        raise ValueError("P78MR_R4_SOURCE_COHORT_DRIFT")
    base = frozen_ssm(control_archive)
    policies = frozen_policies()
    if set(policies) != {"ssm", "fixed0p4", "978201", "978203", "978207"}:
        raise ValueError("P78MR_R4_ARM_DRIFT")
    per_world = {}
    elapsed = {}
    for name, policy in policies.items():
        t0 = time.perf_counter()
        per_world[name] = [score_world(base, policy, world) for world in f3]
        elapsed[name] = time.perf_counter() - t0
    means = {name: statistics.mean(xs) for name, xs in per_world.items()}
    if abs(means["ssm"] - SSM_F4_CAL) > 1e-8:
        raise ValueError("P78MR_R4_ORIGINAL_SSM_REPLAY_PARITY_FAIL")
    frozen_results = {
        str(s): {
            "ratio_vs_trained_ssm": means[str(s)] / means["ssm"],
            "ratio_vs_frozen_fixed_residual": means[str(s)] / means["fixed0p4"],
            "quality_gate_pass": (
                means[str(s)] <= .90 * means["ssm"]
                and means[str(s)] <= .95 * means["fixed0p4"]
            ),
        }
        for s in P78_SEEDS
    }
    return {
        "schema": "ttt-hybrid.p78mr-r4-frozen-ssm-memory-retro-result.v1",
        "classification": "CONSUMED_F3_CAL_RETROSPECTIVE_DEV_ONLY",
        "scientific_pass": False,
        "source_effects_created": 0,
        "models_trained": 0,
        "original_first_use_replayed": False,
        "ood_opened": False,
        "freeze_blob": FREEZE_BLOB,
        "trained_ssm_checkpoint_sha256": SSM_CHECKPOINT_SHA256,
        "worlds": len(f3),
        "actual_effects": 1524,
        "prewrite_scored_events": 1460,
        "persistent_state_floats": {"ssm": 256, "fixed0p4": 512,
                                    **{str(s): 512 for s in P78_SEEDS}},
        "mean_equal_world_four_delta_mse": means,
        "per_world_four_delta_mse": per_world,
        "hypotheses": frozen_results,
        "frozen_primary_all_three_pass": all(v["quality_gate_pass"]
                                            for v in frozen_results.values()),
        "elapsed_per_arm_wall_seconds_diagnostic_only": elapsed,
        "note": "Spent F3 CAL selected the original SSM checkpoint, not independent "
                "heldout; additive 256-state cost is not capacity matched.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-existing-archive", type=Path, required=True)
    parser.add_argument("--f4-existing-controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = retrospect(args.f3_existing_archive, args.f4_existing_controls)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False) + "\\n", encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "mean_equal_world_four_delta_mse": result["mean_equal_world_four_delta_mse"],
        "frozen_primary_all_three_pass": result["frozen_primary_all_three_pass"],
        "scientific_pass": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
