"""P78-MR R3: retrospective, immutable-weight evaluation on spent F3 CAL.

Reads ONLY already-consumed owner F3 FIT/CAL archive and already-trained F4
control receipt. No new simulator action, source birth, fitting, CAL
selection, official workflow dispatch, model promotion or OOD access.
This analysis is descriptive, never an independently fresh SCIENTIFIC_PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from p77_write_policy import WritePolicy  # noqa: E402
from p78mr_adaptive_controls import new_learned  # noqa: E402
from p78mr_gain_posthoc_audit import (  # noqa: E402
    audit as frozen_theta_audit, read_locked, RESULT_BLOB, REFERENCE_BLOB,
)

FREEZE = HERE / "p78mr_r3_f3_transfer_freeze.json"
FREEZE_BLOB = "5cf6d46eb0d99c9835bcaffc9e70425d4b73248e"
F3_ZIP = "a04697bdfba1eb93548244329f4f76fb807489aa7766f4a343de897a56caceb5"
F4_ZIP = "ac93a3616619bf4376023e18cfc2ab32757a1a89715fbc02394164682c0159b5"
CAL_SHA = "15bcbe25b492ad095fca35ee37ec97d0d141c9beb9899549a30558659ded3aef"
FIT_SHA = "b495d6c41fd54c6955fd83ead7569282146af45bc68b3a3dba2a63808e1d6f7b"
F4_RESULT_SHA = "0ce28fcf2976d4a97e922998dbc8696e483db1cfbf1a01bfdd5cb77052ecc56d"
F3_ROOT = "evidence/p85mr_fitcal_f3/ONCE/"
SCALES = (2.4, 3.0, 0.2095, 3.5)
SEEDS = (978201, 978203, 978207)
F4_SEEDS = (985201, 985203, 985207)
FIXED_GAIN = 0.4


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def locked_protocol() -> dict:
    v = json.loads(read_locked(FREEZE, FREEZE_BLOB))
    if (v["schema"] != "ttt-hybrid.p78mr-r3-retrospective-f3-cal-transfer-v1"
            or v["classification"] != "ALREADY_CONSUMED_EXTERNAL_F3_CAL_RETROSPECTIVE_DEV_ONLY_NOT_FIRST_USE"
            or v["source"]["zip_sha256"] != F3_ZIP
            or v["source"]["source_capture_cal_sha256"] != CAL_SHA
            or v["evaluation"]["trained_ssm_cal_delta_mse"] != [
                1.328794543553613e-5, 1.2177126073709132e-5, 1.2313318364931547e-5]
            or v["authority"]["new_effects"] is not False
            or v["authority"]["training"] is not False
            or v["authority"]["OOD"] is not False):
        raise ValueError("P78MR_R3_FROZEN_PROTOCOL_DRIFT")
    return v


def _zip(path: Path, sha: str) -> zipfile.ZipFile:
    if not path.is_file() or sha256(path.read_bytes()) != sha:
        raise ValueError("P78MR_R3_PRIOR_ARCHIVE_DRIFT")
    return zipfile.ZipFile(path)


def read_prior_cal(archive: Path) -> list[dict]:
    with _zip(archive, F3_ZIP) as z:
        expected = {
            "local_runs/p85mr_f3_owner_result.json",
            "local_runs/p85mr_windows_cpu.json",
            "local_runs/p85mr_wsl_cpu.json",
            *(F3_ROOT + item for item in (
                "ATTEMPT.json", "RESULT.json", "fit.json", "cal.json",
                "fit.ndjson", "cal.ndjson", "fit.steps.ndjson", "cal.steps.ndjson")),
        }
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
            raise ValueError("P78MR_R3_UNEXPECTED_ARCHIVE_MEMBERS")
        result = json.loads(z.read(F3_ROOT + "RESULT.json"))
        if (result.get("status") != "SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING"
                or result.get("executed_actions") != 4447
                or result.get("ood_created") is not False
                or result.get("trained_baselines") is not False
                or result.get("science_pass") is not False
                or result.get("owner", {}).get("run_id") != "43073165842"):
            raise ValueError("P78MR_R3_UNQUALIFIED_OR_SPENT_F3")
        raw_cal, raw_fit = z.read(F3_ROOT + "cal.json"), z.read(F3_ROOT + "fit.json")
        if sha256(raw_cal) != CAL_SHA or sha256(raw_fit) != FIT_SHA:
            raise ValueError("P78MR_R3_CAPTURE_CUSTODY_DRIFT")
        cap = json.loads(raw_cal)
    expected_source = (
        "CausalAILab/Causal-Gymnasium",
        "c94e9a4cd3641588b6b8507e71ccee5c27032bdf",
        "causal_gym/envs/cartpole_wind.py",
        "d65a6a31941709dc9201259d80dd4ebbcf48b4fa",
    )
    if (tuple(cap.get(k) for k in ("upstream_repo", "upstream_commit", "upstream_path", "upstream_blob"))
            != expected_source
            or cap.get("schema") != "ttt-hybrid.p80mr.cartpole-do-capture.v1"
            or cap.get("source_id") != "p85mr-f3-cal-43073165842"
            or cap.get("custody_ref") != "ghrun:43073165842"
            or cap.get("producer_id") != "ttt-hybrid-f3-owner"
            or len(cap.get("worlds", ())) != 64):
        raise ValueError("P78MR_R3_CAL_SOURCE_DRIFT")
    seen = set()
    total = 0
    for wi, world in enumerate(cap["worlds"]):
        if (set(world) != {"world_id", "episode_seed", "transitions", "probe_event_ids"}
                or len(world["transitions"]) < 2
                or world["probe_event_ids"] != list(range(2, len(world["transitions"]) + 1))):
            raise ValueError("P78MR_R3_WORLD_OR_PROBE_DRIFT")
        prev = None
        for i, step in enumerate(world["transitions"], 1):
            if (set(step) != {"event_id", "executed_action", "mode",
                              "receipt_ref", "reward", "state_after",
                              "state_before", "terminated", "truncated"}
                    or step["event_id"] != i or step["mode"] != "do"
                    or type(step["executed_action"]) is not int
                    or step["executed_action"] not in (0, 1)
                    or step["receipt_ref"] != f"ghrun:43073165842:cal:{wi}:{i}"
                    or step["receipt_ref"] in seen):
                raise ValueError("P78MR_R3_EVENT_CUSTODY_DRIFT")
            seen.add(step["receipt_ref"])
            a, b = step["state_before"], step["state_after"]
            if (len(a) != 4 or len(b) != 4
                    or not all(type(v) in (int, float) and math.isfinite(v)
                               for v in [*a, *b, step["reward"]])
                    or (prev is not None and any(abs(x-y) > 1e-5
                                                   for x, y in zip(prev, a)))
                    or type(step["terminated"]) is not bool
                    or type(step["truncated"]) is not bool
                    or (step["terminated"] or step["truncated"])
                    and i != len(world["transitions"])):
                raise ValueError("P78MR_R3_PHYSICAL_TRACE_DRIFT")
            prev = b
            total += 1
    if total != 1524 or len(seen) != total:
        raise ValueError("P78MR_R3_F3_EVENT_COUNT_DRIFT")
    return cap["worlds"]


def read_f4_controls(archive: Path) -> dict[int, float]:
    with _zip(archive, F4_ZIP) as z:
        raw = z.read("p85mr_f4_controls_result.json")
        if sha256(raw) != F4_RESULT_SHA:
            raise ValueError("P78MR_R3_F4_RECEIPT_DRIFT")
        d = json.loads(raw)
        if (d["status"] != "FIT_CAL_CONVENTIONAL_CONTROLS_COMPETENT_OOD_SEALED"
                or d["source_artifact_sha256"] != F3_ZIP
                or d["scientific_pass"] is not False
                or len(d["models"]) != 12):
            raise ValueError("P78MR_R3_F4_CONTROL_CUSTODY_DRIFT")
        ssm = {}
        for row in d["models"]:
            if row["kind"] == "ssm256":
                seed = row["model_seed"]
                if seed in ssm or seed not in F4_SEEDS or not row["competent"]:
                    raise ValueError("P78MR_R3_SSM_DUPLICATE_OR_NOT_COMPETENT")
                ssm[seed] = row["cal_delta_mse"]
        if (tuple(sorted(ssm)) != F4_SEEDS
                or any(abs(ssm[k] - expect) > 1e-16 for k, expect in zip(
                    F4_SEEDS, (1.328794543553613e-5,
                               1.2177126073709132e-5,
                               1.2313318364931547e-5)))):
            raise ValueError("P78MR_R3_PRIOR_SSM_DRIFT")
        return ssm


def factual_event(step: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pre, post, act = step["state_before"], step["state_after"], step["executed_action"]
    a = ([1.] + [math.tanh(x/s) for x, s in zip(pre, SCALES)]
         + [float(act == 0), float(act == 1)] + [0.] * 9)
    den = math.sqrt(math.fsum(x*x for x in a))
    effect = ([math.tanh((x-y)/s) for x, y, s in zip(post, pre, SCALES)]
              + [math.tanh(step["reward"]), float(step["terminated"]),
                 float(step["truncated"])] + [0.] * 9)
    return (torch.tensor([v/den for v in a], dtype=torch.float32),
            torch.tensor(effect, dtype=torch.float32))


def score_world(model: WritePolicy | None, world: dict) -> float:
    state = torch.zeros((1, 16, 16), dtype=torch.float32)
    observed_mse = []
    with torch.no_grad():
        for i, step in enumerate(world["transitions"]):
            action, effect = factual_event(step)
            if model is None:
                pred = torch.zeros_like(effect)
            else:
                pred = model.read(state, action.unsqueeze(0))[0]
            if i > 0:
                observed_mse.append(float(((pred[:4] - effect[:4])**2).mean()))
            if model is not None:
                state = model.encode({
                    "action": action.reshape(1, 1, 16),
                    "effect": effect.reshape(1, 1, 16),
                }, carry=state)
                if not bool(torch.isfinite(state).all()):
                    raise ValueError("P78MR_R3_NONFINITE_STATE")
    return sum(observed_mse) / len(observed_mse)


def run(source_archive: Path, controls_archive: Path) -> dict:
    frozen = locked_protocol()
    ssm = read_f4_controls(controls_archive)
    worlds = read_prior_cal(source_archive)
    theta = frozen_theta_audit()["rows"]
    if tuple(x["seed"] for x in theta) != SEEDS:
        raise ValueError("P78MR_R3_ORIGINAL_COEFFICIENTS_DRIFT")
    model_arms = {"fixed_0p4": WritePolicy("fixed_eta", eta=FIXED_GAIN),
                  "zero": None}
    for row in theta:
        model = new_learned("non_directional3", row["seed"])
        with torch.no_grad():
            model.theta.copy_(torch.tensor(row["theta"], dtype=torch.float32))
        model.eval()
        model_arms[str(row["seed"])] = model
    started = time.perf_counter()
    per_world = {
        arm: [score_world(model, world) for world in worlds]
        for arm, model in model_arms.items()
    }
    means = {arm: statistics.mean(x) for arm, x in per_world.items()}
    # Ground exact scoring semantics: the previously trained F4 receipt's
    # all-zero predictor on identical CAL has this value in every control row.
    if abs(means["zero"] - 0.00466479521128349) > 1e-8:
        raise ValueError("P78MR_R3_F4_ZERO_BASELINE_MISMATCH")
    strongest_ssm = min(ssm.values())
    decisions = {
        seed: {
            "vs_p78_fit_selected_fixed_ratio": means[str(seed)] / means["fixed_0p4"],
            "vs_strongest_original_f4_ssm_ratio": means[str(seed)] / strongest_ssm,
            "retrospective_frozen_margin_pass": (
                means[str(seed)] <= .9 * means["fixed_0p4"]
                and means[str(seed)] <= .9 * strongest_ssm),
        }
        for seed in SEEDS
    }
    return {
        "schema": "ttt-hybrid.p78mr-r3-prior-f3-cal-retrospective-result.v1",
        "classification": "RETROSPECTIVE_ALREADY_CONSUMED_EXTERNAL_F3_CAL_DEV",
        "scientific_pass": False, "new_effects": False,
        "new_training": False, "new_holdout_opened": False,
        "freeze_git_blob": FREEZE_BLOB,
        "p78_selected_coefficients_receipt_blob": RESULT_BLOB,
        "p77_equation_git_blob": REFERENCE_BLOB,
        "source_f3_zip_sha256": F3_ZIP, "f4_controls_zip_sha256": F4_ZIP,
        "cal_original_worlds": len(worlds),
        "cal_original_factual_effects": sum(len(w["transitions"]) for w in worlds),
        "cal_scored_prewrite_probes": sum(len(w["transitions"])-1 for w in worlds),
        "per_world_four_delta_mse": per_world,
        "mean_equal_world_four_delta_mse": means,
        "f4_original_pretrained_ssm_by_seed": ssm,
        "f4_strongest_ssm_cal_delta_mse": strongest_ssm,
        "decisions": decisions,
        "margin_all_three_pass": all(x["retrospective_frozen_margin_pass"]
                                   for x in decisions.values()),
        "evaluation_wall_seconds_local": time.perf_counter() - started,
        "note": "F3 CAL previously used for F4 checkpoint selection; not unseen, "
                "not matched compute; no scientific first-use or cross-domain proof.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-existing-archive", type=Path, required=True)
    parser.add_argument("--f4-existing-controls", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.f3_existing_archive, args.f4_existing_controls)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True,
                                   allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in
                      ("classification", "mean_equal_world_four_delta_mse",
                       "decisions", "scientific_pass")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
