"""R86 frozen fresh max-width native/strong-conventional causal composition learning.

Only NEW R86 FIT and FIT_VALIDATION factual one-branch labels are used in
training. Original DEV/OOD paired intervention worlds are born only AFTER all
five model/optimizer training budgets complete, in original first-attempt
GitHub-hosted Ubuntu CPU CI. No R85 calibration or reserved source access.

A 224-tick physical world is presented as 48 deterministic selected public
rows including all A/B probes and original timestamps; this is NOT 224 model
recurrent steps or full-fused CUDA Mamba2/TTT qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r86_native_compositional_freeze.json"
FREEZE_BLOB = "69c939c5603faf306e0560b04802c1e854ae2dce"
SCHEMA = "ttt-hybrid.r86-native-compositional-value-learning-v1"
PARENT = "33e3dbd20049e42bfbca48bb11e18b310fd86384"
BRANCH = "research/r86-fresh-compositional-native-learning-20261115"
SEEDS = (999611, 999623)
SALT = {"FIT": 25280111, "FIT_VALIDATION": 25280211,
        "DEV": 25280311, "OOD_SHIFT": 25280411}
COUNT = {"FIT": 2048, "FIT_VALIDATION": 128, "DEV": 128, "OOD_SHIFT": 128}
ARMS = ("NATIVE_FULL", "NATIVE_OBSERVATION_BLIND", "GRU64_FULL",
        "GRU256_FULL", "MLP256_FULL")
CONVS = ("GRU64_FULL", "GRU256_FULL", "MLP256_FULL")
STEPS = {"NATIVE_FULL": 1024, "NATIVE_OBSERVATION_BLIND": 1024,
         "GRU64_FULL": 8192, "GRU256_FULL": 8192, "MLP256_FULL": 4096}
OBSERVED_STEPS = 48
BATCH = 2
WIDTH = 1024


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R86_FREEZE_BLOB_DRIFT")
    r85.frozen()
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    w, m, e = f["worlds"], f["models"], f["evaluation"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "IMMUTABLE_PROSPECTIVE_FREEZE_ONLY_BEFORE_R86_IMPLEMENTATION_TEST_CI_OR_FIRST_USE"
        or f["parent"]["R85_exact_SHA"] != PARENT
        or f["parent"]["R85_freeze_blob"] != r85.FREEZE_BLOB
        or f["parent"]["R85_original_status"] != "ENGINEERING_R85_COMPOSITIONAL_INFORMATION_CALIBRATION_PASS_ONLY_BOTH_SEEDS"
        or f["parent"]["R85_reserved_salts_still_sealed"] != [25270211, 25270311, 25270411]
        or w["new_salts"] != SALT or w["seeds"] != list(SEEDS)
        or w["counts"] != COUNT or w["episode_length"] != 224
        or w["public_dim"] != 13 or w["model_observed_timesteps"] != OBSERVED_STEPS
        or w["pivots"] != {"DEV_FIT_early": 64, "DEV_FIT_cold": 192,
                            "OOD_early": 68, "OOD_cold": 208}
        or w["four_A_probes"] != list(r85.A_PROBES)
        or w["four_B_probes"] != list(r85.B_PROBES)
        or w["shifted_A_probes"] != [12, 22, 30, 40]
        or w["shifted_B_probes"] != [110, 120, 136, 148]
        or m["native_parameter_count"] != 16818178
        or m["arms"] != list(ARMS)
        or m["training"]["native_updates"] != 1024
        or m["training"]["GRU64_updates"] != 8192
        or m["training"]["GRU256_updates"] != 8192
        or m["training"]["MLP256_updates"] != 4096
        or m["training"]["batch"] != BATCH
        or e["paired_bootstrap"]["resamples"] != 2000
        or e["thresholds"]["DEV_native_minus_best_trained_conventional_min"] != .20
        or e["thresholds"]["OOD_cold_native_minus_DEV_selected_conventional_min"] != .15
        or any(f["authority"].values())):
        raise PermissionError("R86_FROZEN_SOURCE_OR_THRESHOLD_SEMANTICS_DRIFT")
    if set(SALT.values()) & ({25270111, 25270211, 25270311, 25270411}
                               | set(f["parent"]["R84_spent_salts"])):
        raise RuntimeError("R86_PREVIOUS_OR_SEALED_DATA_OVERLAP")
    return f


def _fresh_world(panel: str, seed: int, index: int, *, custody_sha: str | None = None):
    if panel not in SALT:
        raise PermissionError("R86_PRIOR_OR_PROTECTED_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index, int) or not 0 <= index < COUNT[panel]:
        raise PermissionError("R86_SOURCE_SEED_OR_INDEX_DENIED")
    if (not custody_sha or custody_sha != os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_JOB") != "r86-native-compositional-value"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("R86_SEED") != str(seed)):
        raise PermissionError("R86_ORIGINAL_SOURCE_NOT_ADMITTED")
    # R85 pure physics constructor, NOT its once-used CALIBRATION generator.
    rng = np.random.default_rng(np.random.SeedSequence([SALT[panel], seed, index]))
    w = r85._make_synthetic_world(index, rng)
    if panel == "OOD_SHIFT":
        # Change probe schedule and public decision timing prospectively.
        old = w["probe"].copy()
        for t in old:
            w["actions"][t] = r85.EXPLOIT_POS
        w["probe"] = {}
        for times, source in (([12, 22, 30, 40], r85.A_PROBES),
                              ([110, 120, 136, 148], r85.B_PROBES)):
            for new_t, old_t in zip(times, source):
                w["probe"][new_t] = old[old_t]
                w["actions"][new_t] = r85.PROBE
        w["pivot"] = 68 if index % 2 == 0 else 208
        w["innovations"] *= 1.35
    return w, int(rng.choice([r85.CALIBRATE, r85.EXPLOIT_POS]))


def _compressed(x: np.ndarray, probe_times: tuple[int, ...] | list[int]) -> np.ndarray:
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 13:
        raise ValueError("R86_INVALID_OR_PRIVILEGED_PUBLIC_DIMENSIONS")
    pivot = len(x) - 1
    required = sorted({int(t) + 1 for t in probe_times if int(t) + 1 <= pivot})
    if not required or pivot < OBSERVED_STEPS - 1:
        raise ValueError("R86_MISSING_PUBLIC_PROBES_OR_SHORT_PREFIX")
    remaining = sorted(set(range(pivot)) - set(required))
    positions = np.linspace(0, len(remaining) - 1,
                            OBSERVED_STEPS - 1 - len(required), dtype=int)
    rows = sorted(required + [remaining[j] for j in positions]) + [pivot]
    if len(rows) != OBSERVED_STEPS or len(set(rows)) != OBSERVED_STEPS:
        raise RuntimeError("R86_COMPRESSED_PUBLIC_TIMELINE_NOT_UNIQUE")
    if any(t + 1 not in rows for t in probe_times if t + 1 <= pivot):
        raise RuntimeError("R86_OBSERVED_CAUSAL_FACTOR_DROPPED")
    out = x[rows].copy()
    if out[-1, 12] != x[-1, 12]:
        raise RuntimeError("R86_PUBLIC_TIME_DISCARDED")
    return out


def _view(w: dict, action: int):
    x, ret = r85._rollout(w, action)
    probes = sorted(int(t) for t in w["probe"] if t < w["pivot"])
    return _compressed(x, probes), float(ret)


def _test_only_world(index: int):
    # Independent non-source synthetic fixture; never use any R86 salts/seeds.
    rng = np.random.default_rng(np.random.SeedSequence([25289991, index]))
    return r85._make_synthetic_world(index, rng)


def _factual(panel: str, seed: int, *, custody_sha: str):
    if panel not in ("FIT", "FIT_VALIDATION"):
        raise PermissionError("R86_ONLY_FIT_AND_VALIDATION_ALLOWED_FOR_LEARNING")
    x = np.empty((COUNT[panel], OBSERVED_STEPS, 13), np.float32)
    y = np.empty(COUNT[panel], np.float32)
    choices = np.empty(COUNT[panel], np.int8)
    for i in range(COUNT[panel]):
        w, chosen = _fresh_world(panel, seed, i, custody_sha=custody_sha)
        prefix, ret = _view(w, chosen)
        x[i], y[i], choices[i] = prefix, ret, chosen
    return {"input": x, "target": y, "logged_action": choices,
            "panel": panel, "single_factual_branch_only": True}


def _paired_scoring(panel: str, seed: int, *, custody_sha: str):
    if panel not in ("DEV", "OOD_SHIFT"):
        raise PermissionError("R86_ONLY_FRESH_PROTECTED_EVALUATION_ALLOWED")
    worlds = []
    for i in range(COUNT[panel]):
        w, _unused = _fresh_world(panel, seed, i, custody_sha=custody_sha)
        x0, r0 = _view(w, r85.CALIBRATE)
        x2, r2 = _view(w, r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1], x2[:-1])
            or not np.array_equal(x0[-1, 4:], x2[-1, 4:])
            or x0[-1, 0] != 1 or x2[-1, 2] != 1):
            raise RuntimeError("R86_CAUSAL_FUTURE_OR_CHOICE_LEAK_IN_PUBLIC_PREFIX")
        worlds.append({"x0": x0, "x2": x2, "r0": r0, "r2": r2,
                       "cold": bool(i % 2), "pivot": w["pivot"]})
    return worlds


def _blind(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[..., 4:12] = 0
    return out


def _models(seed: int):
    torch.manual_seed(seed)
    native = r74.make_model("HybridPilot_FULL", WIDTH)
    old = native.ttt
    ttt = r75.TriangularFastWeightMemory(WIDTH, learning_rate=old.eta)
    ttt.load_state_dict(old.state_dict(), strict=True)
    native.ttt = ttt
    blind = copy.deepcopy(native)
    p1, p2 = dict(native.named_parameters()), dict(blind.named_parameters())
    if p1.keys() != p2.keys() or any(
        not torch.equal(p1[k], p2[k]) for k in p1):
        raise RuntimeError("R86_NATIVE_AND_BLINDED_NOT_INITIALIZED_IDENTICALLY")
    if sum(p.numel() for p in native.parameters()) != 16818178:
        raise RuntimeError("R86_NATIVE_WIDTH_OR_ARCHITECTURE_DRIFT")
    torch.manual_seed(seed + 64)
    gru64 = r81.GRU(64)
    torch.manual_seed(seed + 256)
    gru256 = r81.GRU(256)
    torch.manual_seed(seed + 1024)
    mlp = r81.MLP256()
    return dict(zip(ARMS, (native, blind, gru64, gru256, mlp)))


def _validation(model, val: dict, blind: bool) -> float:
    model.eval()
    x = _blind(val["input"]) if blind else val["input"]
    out = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            batch = torch.from_numpy(x[i:i + BATCH])
            preds = model(batch)[:, -1, 0]
            target = torch.from_numpy(val["target"][i:i + BATCH])
            out.extend((preds - target).square().tolist())
    model.train()
    return float(np.mean(out))


def _train(arm: str, model, fit: dict, val: dict, indices: np.ndarray):
    blind = arm == "NATIVE_OBSERVATION_BLIND"
    data = torch.from_numpy(_blind(fit["input"]) if blind else fit["input"])
    labels = torch.from_numpy(fit["target"])
    updates = STEPS[arm]
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    checkpoints = {"0": _validation(model, val, blind)}
    started = time.perf_counter()
    for step in range(1, updates + 1):
        index = indices[step - 1]
        pred = model(data[index])[:, -1, 0]
        loss = F.mse_loss(pred, labels[index])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in (512, 1024, 2048, 4096, 8192):
            checkpoints[str(step)] = _validation(model, val, blind)
    elapsed = time.perf_counter() - started
    if not all(math.isfinite(v) for v in checkpoints.values()):
        raise RuntimeError("R86_INVALID_FIT_VALIDATION")
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("R86_NONFINITE_MODEL_WEIGHTS")
    return {"updates_completed": updates,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "factual_selected_exposures": updates * BATCH,
            "distinct_logged_FIT_worlds": COUNT["FIT"],
            "validation_MSE_by_update": checkpoints,
            "train_wall_seconds": elapsed,
            "hard_native_router": arm.startswith("NATIVE"),
            "observation_history_erased": blind}


def _choices(model, worlds: list[dict], *, blind: bool) -> np.ndarray:
    # Batch both lawful current action candidates, no alternative reward input.
    paired = np.asarray(
        [x for w in worlds for x in (w["x0"], w["x2"])], np.float32)
    if blind:
        paired = _blind(paired)
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, len(paired), BATCH):
            vals.extend(model(torch.from_numpy(paired[i:i + BATCH]))[:, -1, 0].tolist())
    model.train()
    v = np.asarray(vals, np.float64).reshape(len(worlds), 2)
    if not np.isfinite(v).all():
        raise RuntimeError("R86_INVALID_ACTION_VALUE_AT_EVALUATION")
    return np.where(v[:, 0] > v[:, 1], r85.CALIBRATE, r85.EXPLOIT_POS)


def _reward(choices: np.ndarray, worlds: list[dict]):
    a = np.asarray([w["r0"] for w in worlds])
    b = np.asarray([w["r2"] for w in worlds])
    return np.where(choices == r85.CALIBRATE, a, b)


def _evaluate(models, worlds: list[dict]):
    results = {}
    chosen = {}
    cold = np.asarray([w["cold"] for w in worlds], bool)
    for arm, model in models.items():
        c = _choices(model, worlds, blind=arm == "NATIVE_OBSERVATION_BLIND")
        chosen[arm] = c
        results[arm] = _reward(c, worlds)
    controls = {
        "PUBLIC_A_PARITY_MEMORY_RULE":
            np.asarray([r85._policy(w["x0"]) for w in worlds], np.int64),
        "LAST_A_CUE":
            np.asarray([r85._policy(w["x0"], "LAST_A") for w in worlds], np.int64),
        "RECENT_B_CUE":
            np.asarray([r85._policy(w["x0"], "RECENT_B") for w in worlds], np.int64),
        "ALWAYS_CALIBRATE":
            np.full(len(worlds), r85.CALIBRATE, np.int64),
        "ALWAYS_EXPLOIT_POS":
            np.full(len(worlds), r85.EXPLOIT_POS, np.int64)}
    for label, c in controls.items():
        results[label] = _reward(c, worlds)
    r0 = results["ALWAYS_CALIBRATE"]
    r2 = results["ALWAYS_EXPLOIT_POS"]
    results["EVALUATOR_ONLY_ORACLE"] = np.maximum(r0, r2)
    scores = {}
    for arm, rewards in results.items():
        scores[arm] = {
            "reward": float(rewards.mean()),
            "cold": float(rewards[cold].mean()),
            "early": float(rewards[~cold].mean()),
            "regret_to_oracle": float((results["EVALUATOR_ONLY_ORACLE"] - rewards).mean())}
    return results, chosen, scores


def _bootstrap_lower(conventional: np.ndarray, native: np.ndarray, seed: int) -> float:
    if conventional.shape != (COUNT["DEV"],) or native.shape != conventional.shape:
        raise RuntimeError("R86_INVALID_PAIRED_BOOTSTRAP")
    rng = np.random.default_rng(np.random.SeedSequence([25280555, seed]))
    sample = rng.integers(0, len(native), (2000, len(native)))
    return float(np.quantile((native - conventional)[sample].mean(axis=1), .025))


def _admit(seed: int) -> str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_EVENT_NAME") != "pull_request"
        or os.getenv("GITHUB_REPOSITORY") != "spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("GITHUB_JOB") != "r86-native-compositional-value"
        or os.getenv("RUNNER_OS") != "Linux"
        or os.getenv("RUNNER_ENVIRONMENT") != "github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT") != "1"
        or "/merge" not in os.getenv("GITHUB_REF", "")
        or os.getenv("R86_SEED") != str(seed)):
        raise PermissionError("R86_ORIGINAL_PR_HOSTED_CPU_FIRST_ATTEMPT_ONLY")
    sha = os.getenv("EXPECTED_CANDIDATE_SHA", "")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=False)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT,
        capture_output=True, timeout=10, check=False)
    if (len(sha) != 40 or actual.returncode != 0
        or actual.stdout.strip() != sha or dirty.returncode != 0):
        raise PermissionError("R86_ORIGINAL_EXACT_SHA_OR_CLEAN_SOURCE_CUSTODY_FAIL")
    return sha


def original_ci(seed: int):
    sha = _admit(seed)  # No official R86 data access above this fence.
    torch.set_num_threads(1)
    path = ROOT / f"local_runs/r86_native_composition_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R86_REPLAY_OR_UNCERTAIN_FIRST_USE_REFUSED")
    started = time.perf_counter()
    result = {
        "schema": SCHEMA, "candidate_SHA": sha, "seed": seed,
        "freeze_blob": FREEZE_BLOB, "parent_R85_SHA": PARENT,
        "source_born_after_admission": True, "completed_arms": {},
        "all_model_weights_frozen_before_DEV_OOD": False,
        "DEV_OOD_FIRST_USE": False, "prior_scored_or_R85_reserved_used": False,
        "scientific_pass": False, "status": "INCOMPLETE_ENGINEERING_OR_RESOURCE",
        "platform": platform.platform(), "torch": str(torch.__version__)}
    try:
        fit = _factual("FIT", seed, custody_sha=sha)
        valid = _factual("FIT_VALIDATION", seed, custody_sha=sha)
        if not fit["single_factual_branch_only"]:
            raise RuntimeError("R86_ILLEGAL_UNCHOSEN_FIT_LABELS")
        models = _models(seed)
        rng = np.random.default_rng(np.random.SeedSequence([25280611, seed]))
        indices = rng.integers(0, COUNT["FIT"], size=(max(STEPS.values()), BATCH))
        for arm in ARMS:
            result["completed_arms"][arm] = _train(arm, models[arm], fit, valid, indices)
        if set(result["completed_arms"]) != set(ARMS):
            raise RuntimeError("R86_INCOMPLETE_TRAINED_STRONG_CONTROLS")
        result["all_model_weights_frozen_before_DEV_OOD"] = True
        # First-use boundary: no training, checkpoint selection or updates after here.
        result["DEV_OOD_FIRST_USE"] = True
        dev = _paired_scoring("DEV", seed, custody_sha=sha)
        ood = _paired_scoring("OOD_SHIFT", seed, custody_sha=sha)
        d_raw, d_chosen, d_scores = _evaluate(models, dev)
        o_raw, _, o_scores = _evaluate(models, ood)
        best_conv = max(CONVS, key=lambda x: d_scores[x]["reward"])
        strongest_fixed = max(
            d_scores["ALWAYS_CALIBRATE"]["reward"],
            d_scores["ALWAYS_EXPLOIT_POS"]["reward"])
        d_native = d_scores["NATIVE_FULL"]
        d_blind = d_scores["NATIVE_OBSERVATION_BLIND"]
        d_control = d_scores[best_conv]
        d_rule = d_scores["PUBLIC_A_PARITY_MEMORY_RULE"]
        o_native, o_control = o_scores["NATIVE_FULL"], o_scores[best_conv]
        frac_denom = d_rule["reward"] - strongest_fixed
        if frac_denom <= 0:
            raise RuntimeError("R86_FRESH_TASK_PUBLIC_RULE_NOT_INFORMATIVE")
        agree = float(np.mean(d_chosen["NATIVE_FULL"] ==
            np.asarray([r85._policy(w["x0"]) for w in dev])))
        boot = _bootstrap_lower(d_raw[best_conv], d_raw["NATIVE_FULL"], seed)
        comparison = {
            "best_DEV_conventional_arm": best_conv,
            "native_minus_blinded_DEV": d_native["reward"] - d_blind["reward"],
            "native_minus_best_conventional_DEV": d_native["reward"] - d_control["reward"],
            "native_minus_best_fixed_DEV": d_native["reward"] - strongest_fixed,
            "native_minus_last_A_cue_DEV": d_native["reward"] - d_scores["LAST_A_CUE"]["reward"],
            "native_choice_agreement_with_public_rule": agree,
            "native_fraction_of_legal_rule_gain": (d_native["reward"] - strongest_fixed) / frac_denom,
            "native_cold_minus_conventional_DEV": d_native["cold"] - d_control["cold"],
            "paired_bootstrap_lower95_native_minus_conventional": boot,
            "native_minus_conventional_OOD_SHIFT": o_native["reward"] - o_control["reward"],
            "native_cold_minus_conventional_OOD_SHIFT": o_native["cold"] - o_control["cold"],
            "native_DEV_oracle_regret": d_native["regret_to_oracle"]}
        t = frozen()["evaluation"]["thresholds"]
        checks = {
            "DEV_memory_history_gain": comparison["native_minus_blinded_DEV"] >= t["DEV_native_minus_blinded_min"],
            "DEV_beats_best_trained_conventional": comparison["native_minus_best_conventional_DEV"] >= t["DEV_native_minus_best_trained_conventional_min"],
            "DEV_beats_fixed": comparison["native_minus_best_fixed_DEV"] >= t["DEV_native_minus_strongest_fixed_min"],
            "DEV_beats_shortcut": comparison["native_minus_last_A_cue_DEV"] >= t["DEV_native_minus_last_A_cue_min"],
            "DEV_legal_rule_agreement": agree >= t["DEV_native_legal_A_parity_choice_agreement_min"],
            "DEV_recovers_rule_gain": comparison["native_fraction_of_legal_rule_gain"] >= t["DEV_native_fraction_of_legal_rule_gain_over_fixed_min"],
            "DEV_cold_beats_conventional": comparison["native_cold_minus_conventional_DEV"] >= t["DEV_cold_native_minus_best_trained_conventional_min"],
            "DEV_paired_bootstrap_positive": boot > 0,
            "OOD_beats_conventional": comparison["native_minus_conventional_OOD_SHIFT"] >= t["OOD_native_minus_DEV_selected_conventional_min"],
            "OOD_cold_beats_conventional": comparison["native_cold_minus_conventional_OOD_SHIFT"] >= t["OOD_cold_native_minus_DEV_selected_conventional_min"],
            "DEV_oracle_regret": d_native["regret_to_oracle"] <= t["DEV_oracle_regret_max"]}
        result["comparison"] = comparison
        result["DEV"] = d_scores
        result["OOD_SHIFT"] = o_scores
        result["frozen_checks"] = {k: bool(v) for k, v in checks.items()}
        result["status"] = t["on_positive"] if all(checks.values()) else t["on_negative"]
    except BaseException as e:
        result["error_class"] = type(e).__name__
        result["status"] = "INCOMPLETE_ENGINEERING_OR_RESOURCE"
        raise
    finally:
        result["total_process_wall_seconds"] = time.perf_counter() - started
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as h:
            json.dump(result, h, sort_keys=True, indent=2, allow_nan=False)
            h.write("\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()
    r = original_ci(args.seed)
    print(json.dumps({"seed": args.seed, "status": r["status"],
        "arms_completed": len(r["completed_arms"]),
        "DEV_OOD_scored": r["DEV_OOD_FIRST_USE"],
        "scientific_pass": r["scientific_pass"]}, sort_keys=True))
