"""R87: frozen public causal-composition feature learning, not native promotion.

A new two-seed experiment that diagnoses whether a four-factor nonlinear
public feature helps factual action-value learning. This is NOT a native
Hybrid training run; nor is the composition feature learned autonomously.
Official fresh source only generated inside exact-SHA first-attempt hosted
GitHub CI AFTER both-platform full regression. No old R84-86 data access.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r81_paired_history as r81

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r87_public_composition_freeze.json"
FREEZE_BLOB = "794719491f4c07cb36a68c405f6deb9f1c0621c8"
SCHEMA = "ttt-hybrid.r87-legal-composition-representation-bottleneck-v1"
PARENT = "6db67a26b9066057880b6124d27f3d215663e7e3"
BRANCH = "research/r87-legal-composition-representation-diagnostic-20261115"
SEEDS = (999811, 999823)
SALT = {"FIT": 25290111, "FIT_VALIDATION": 25290211,
        "DEV": 25290311, "OOD_SHIFT": 25290411}
COUNT = {"FIT": 2048, "FIT_VALIDATION": 256, "DEV": 256, "OOD_SHIFT": 256}
ARMS = ("MLP64_RAW", "MLP64_COMPOSED", "MLP64_COMPOSED_SHUFFLED",
        "GRU64_PUBLIC")
RAW_CONTROLS = ("MLP64_RAW", "GRU64_PUBLIC")
BATCH = 16
STEPS = 4096
FEAT_DIM = 8
COMP_SLOT = 7
PUBLIC_LENGTH = 48


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R87_PROSPECTIVE_FREEZE_BLOB_DRIFT")
    # Check original lineage immutability; never execute prior science.
    r86.frozen()
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    p, s, m, e = f["parent"], f["sources"], f["training"], f["evaluation"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "IMMUTABLE_PROSPECTIVE_FREEZE_BEFORE_R87_IMPLEMENTATION_TEST_CI_OR_FIRST_USE"
        or p["R86_exact_SHA"] != PARENT
        or p["R86_freeze_blob"] != r86.FREEZE_BLOB
        or p["R86_two_seed_result"] != "DEV_NATIVE_COMPOSITIONAL_VALUE_NO_GO_BOTH_SEEDS_ALL_11_FROZEN_CHECKS_FAILED"
        or s["salts"] != SALT or s["seeds"] != list(SEEDS)
        or s["world_counts"] != COUNT or s["model_public_rows"] != PUBLIC_LENGTH
        or f["arms"] != list(ARMS)
        or m["batch"] != BATCH or m["MLP64_RAW_updates"] != STEPS
        or m["MLP64_COMPOSED_updates"] != STEPS
        or m["MLP64_COMPOSED_SHUFFLED_updates"] != STEPS
        or m["GRU64_PUBLIC_updates"] != STEPS
        or e["bootstrap"]["resamples"] != 2000
        or e["thresholds"]["DEV_composed_minus_strongest_raw_control_min"] != .30
        or e["thresholds"]["OOD_cold_composed_minus_DEV_selected_raw_min"] != .20
        or any(f["authority"].values())):
        raise PermissionError("R87_FROZEN_SPEC_OR_THRESHOLDS_DRIFT")
    if set(SALT.values()) & (
        set(p["R86_FIT_DEV_OOD_salts_spent"])
        | {p["R85_CALIBRATION_spent"]}
        | set(p["R85_QUAL_DEV_OOD_reserved_untouched"])):
        raise RuntimeError("R87_SALT_REUSE")
    return f


def _admit(seed: int) -> str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_EVENT_NAME") != "pull_request"
        or os.getenv("GITHUB_REPOSITORY") != "spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("GITHUB_JOB") != "r87-public-composition-diagnostic"
        or os.getenv("RUNNER_OS") != "Linux"
        or os.getenv("RUNNER_ENVIRONMENT") != "github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT") != "1"
        or "/merge" not in os.getenv("GITHUB_REF", "")
        or os.getenv("R87_SEED") != str(seed)):
        raise PermissionError("R87_ORIGINAL_FIRST_ATTEMPT_HOSTED_PR_ADMISSION_DENIED")
    expected = os.getenv("EXPECTED_CANDIDATE_SHA", "")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False, timeout=10)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT,
        capture_output=True, check=False, timeout=10)
    if len(expected) != 40 or head.returncode or head.stdout.strip() != expected or clean.returncode:
        raise PermissionError("R87_EXACT_CLEAN_CANDIDATE_CUSTODY_MISMATCH")
    return expected


def _world(panel: str, seed: int, index: int, *, custody_sha: str | None = None):
    if panel not in SALT:
        raise PermissionError("R87_PRIOR_OR_SEALED_SCIENCE_SOURCE_DENIED")
    if seed not in SEEDS or not isinstance(index, int) or not 0 <= index < COUNT[panel]:
        raise PermissionError("R87_NONFROZEN_SEED_OR_INDEX")
    if (not custody_sha or custody_sha != os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_JOB") != "r87-public-composition-diagnostic"
        or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("R87_SEED") != str(seed)):
        raise PermissionError("R87_ORIGINAL_SOURCE_ACCESS_NOT_ADMITTED")
    # Reuse only pure prior physics, never R85/R86 scored data.
    rng = np.random.default_rng(np.random.SeedSequence([SALT[panel], seed, index]))
    w = r85._make_synthetic_world(index % r85.N, rng)
    if panel == "OOD_SHIFT":
        w = r86._shift_public_world(w, index)
    logged = int(rng.choice([r85.CALIBRATE, r85.EXPLOIT_POS]))
    return w, logged


def _view(w, action):
    return r86._view(w, action)


def _legal_features(prefix: np.ndarray) -> tuple[np.ndarray, float]:
    """Only lawful selected prior public A probes and public current action."""
    if prefix.shape != (PUBLIC_LENGTH, 13) or prefix.dtype != np.float32:
        raise ValueError("R87_ILLEGAL_PUBLIC_PREFIX")
    probes = r85._public_probes(prefix, 1)
    cues = np.zeros(4, np.float32)
    for role, cue in probes:
        cues[role] = float(cue)
    action0 = float(prefix[-1, r85.CALIBRATE])
    action2 = float(prefix[-1, r85.EXPLOIT_POS])
    if (action0, action2) not in ((1., 0.), (0., 1.)):
        raise RuntimeError("R87_ILLEGAL_SELECTED_CURRENT_ACTION")
    parity = float(np.prod(np.where(cues > 0, 1, -1)))
    feature = np.zeros(FEAT_DIM, np.float32)
    feature[:4] = cues
    feature[4:6] = [action0, action2]
    feature[6] = prefix[-1, 12]
    return feature, parity


def _shuffle_parity(raw: np.ndarray, parity: np.ndarray, seed: int, panel: str) -> np.ndarray:
    if panel not in SALT or len(raw) != COUNT[panel] or len(parity) != len(raw):
        raise PermissionError("R87_SHUFFLED_NEGATIVE_ONLY_ON_REGISTERED_PANEL")
    times = raw[:, 6]
    # Time/pivot stratum preserved exactly. No privileged world info in shuffle.
    u = np.unique(times)
    if len(u) != 2:
        raise RuntimeError("R87_NO_EARLY_COLD_STRATUM")
    rng = np.random.default_rng(np.random.SeedSequence([25290555, seed, SALT[panel]]))
    ix = np.arange(len(raw))
    for t in u:
        where = np.flatnonzero(times == t)
        ix[where] = rng.permutation(where)
    return parity[ix]


def _assemble(panel: str, seed: int, *, custody_sha: str) -> dict:
    if panel not in ("FIT", "FIT_VALIDATION"):
        raise PermissionError("R87_TRAINING_ON_SCORED_PANEL_DENIED")
    n = COUNT[panel]
    x = np.zeros((n, PUBLIC_LENGTH, 13), np.float32)
    feats = np.zeros((n, FEAT_DIM), np.float32)
    parity = np.zeros(n, np.float32)
    targets = np.zeros(n, np.float32)
    for i in range(n):
        w, act = _world(panel, seed, i, custody_sha=custody_sha)
        prefix, ret = _view(w, act)
        f, p = _legal_features(prefix)
        x[i], feats[i], parity[i], targets[i] = prefix, f, p, ret
    shifted = _shuffle_parity(feats, parity, seed, panel)
    return {"sequence": x, "features": feats,
            "parity": parity, "shuffled_parity": shifted,
            "target": targets, "only_one_factual_return": True}


def _evaluation(panel: str, seed: int, *, custody_sha: str):
    if panel not in ("DEV", "OOD_SHIFT"):
        raise PermissionError("R87_SCORING_ONLY_FRESH_PROTECTED_PANELS")
    n = COUNT[panel]
    raw = np.zeros((n, FEAT_DIM), np.float32)
    parity = np.zeros(n, np.float32)
    worlds = []
    lawful_actions = []
    true_factor_parity = []
    for i in range(n):
        w, _ = _world(panel, seed, i, custody_sha=custody_sha)
        x0, r0 = _view(w, r85.CALIBRATE)
        x2, r2 = _view(w, r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1], x2[:-1])
            or not np.array_equal(x0[-1, 4:], x2[-1, 4:])
            or x0[-1, 0] != 1 or x2[-1, 2] != 1):
            raise RuntimeError("R87_CAUSAL_PREFIX_OR_ACTION_LEAK")
        f, p = _legal_features(x0)
        raw[i], parity[i] = f, p
        lawful_actions.append(r85._policy(x0))
        true_factor_parity.append(w["law_a"])  # evaluator-only, never model input
        worlds.append({"x0": x0, "x2": x2, "r0": r0, "r2": r2,
                       "cold": bool(i % 2)})
    return {"worlds": worlds, "raw": raw, "parity": parity,
            "shuffled_parity": _shuffle_parity(raw, parity, seed, panel),
            "legal_actions": np.asarray(lawful_actions, np.int64),
            "private_eval_truth": np.asarray(true_factor_parity, np.int64)}


class PublicValueMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEAT_DIM, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)[:, 0]


def _models(seed: int):
    models = {}
    for offset, arm in enumerate(ARMS):
        torch.manual_seed(seed + 80 + offset)
        models[arm] = r81.GRU(64) if arm == "GRU64_PUBLIC" else PublicValueMLP()
    if (len({sum(p.numel() for p in models[a].parameters()) for a in ARMS[:3]}) != 1):
        raise RuntimeError("R87_MLP_PARAMETER_MISMATCH")
    return models


def _arm_inputs(source: dict, arm: str):
    if arm == "GRU64_PUBLIC":
        return source["sequence"]
    x = source["features"].copy()
    if arm == "MLP64_COMPOSED":
        x[:, COMP_SLOT] = source["parity"]
    elif arm == "MLP64_COMPOSED_SHUFFLED":
        x[:, COMP_SLOT] = source["shuffled_parity"]
    elif arm != "MLP64_RAW":
        raise ValueError("R87_NONFROZEN_ARM")
    return x


def _val(model, arm: str, valid: dict):
    model.eval()
    x = torch.from_numpy(_arm_inputs(valid, arm))
    targets = torch.from_numpy(valid["target"])
    losses = []
    with torch.no_grad():
        for i in range(0, len(x), BATCH):
            pred = model(x[i:i+BATCH])
            if arm == "GRU64_PUBLIC":
                pred = pred[:, -1, 0]
            losses.extend((pred-targets[i:i+BATCH]).square().tolist())
    model.train()
    return float(np.mean(losses))


def _train(model, arm: str, fit: dict, valid: dict, indices: np.ndarray):
    x = torch.from_numpy(_arm_inputs(fit, arm))
    y = torch.from_numpy(fit["target"])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    hist = {"0": _val(model, arm, valid)}
    began = time.perf_counter()
    for i in range(1, STEPS+1):
        ix = indices[i-1]
        est = model(x[ix])
        if arm == "GRU64_PUBLIC":
            est = est[:, -1, 0]
        loss = F.mse_loss(est, y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i in (1024, 2048, 4096):
            hist[str(i)] = _val(model, arm, valid)
    if any(not math.isfinite(k) for k in hist.values()):
        raise RuntimeError("R87_NONFINITE_FIT_VALIDATION")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("R87_NONFINITE_TRAINED_WEIGHTS")
    return {"updates": STEPS, "params": sum(p.numel() for p in model.parameters()),
            "selected_event_exposures": STEPS*BATCH, "FIT_worlds": COUNT["FIT"],
            "FIT_validation_MSE": hist, "train_wall_seconds": time.perf_counter()-began}


def _score_actions(models: dict, src: dict, panel: str):
    worlds = src["worlds"]
    raw = src["raw"]
    pars = src["parity"]
    wrong = src["shuffled_parity"]
    action0 = raw.copy()
    action2 = raw.copy()
    action0[:, 4:6] = [1, 0]
    action2[:, 4:6] = [0, 1]
    results = {}
    actions = {}
    for arm, model in models.items():
        model.eval()
        if arm == "GRU64_PUBLIC":
            pairs = np.asarray([x for w in worlds for x in (w["x0"], w["x2"])], np.float32)
            with torch.no_grad():
                outs = []
                for i in range(0, len(pairs), BATCH):
                    outs.extend(model(torch.from_numpy(pairs[i:i+BATCH]))[:, -1, 0].tolist())
            values = np.array(outs, np.float64).reshape(-1, 2)
        else:
            xa, xb = action0.copy(), action2.copy()
            if arm == "MLP64_COMPOSED":
                xa[:, COMP_SLOT], xb[:, COMP_SLOT] = pars, pars
            elif arm == "MLP64_COMPOSED_SHUFFLED":
                xa[:, COMP_SLOT], xb[:, COMP_SLOT] = wrong, wrong
            with torch.no_grad():
                pa = model(torch.from_numpy(xa)).cpu().numpy()
                pb = model(torch.from_numpy(xb)).cpu().numpy()
            values = np.stack((pa, pb), axis=1)
        if not np.isfinite(values).all():
            raise RuntimeError("R87_SCORING_INVALID_LEARNED_VALUE")
        actions[arm] = np.where(values[:, 0] > values[:, 1], r85.CALIBRATE, r85.EXPLOIT_POS)
        model.train()
    actions["PUBLIC_A_PARITY_RULE"] = src["legal_actions"]
    actions["LAST_A_CUE"] = np.asarray(
        [r85._policy(w["x0"], "LAST_A") for w in worlds], np.int64)
    actions["RECENT_B_PARITY"] = np.asarray(
        [r85._policy(w["x0"], "RECENT_B") for w in worlds], np.int64)
    actions["ALWAYS_CALIBRATE"] = np.full(len(worlds), r85.CALIBRATE, np.int64)
    actions["ALWAYS_EXPLOIT_POS"] = np.full(len(worlds), r85.EXPLOIT_POS, np.int64)
    r0 = np.asarray([w["r0"] for w in worlds], np.float64)
    r2 = np.asarray([w["r2"] for w in worlds], np.float64)
    cold = np.asarray([w["cold"] for w in worlds], bool)
    for arm, a in actions.items():
        results[arm] = np.where(a == r85.CALIBRATE, r0, r2)
    results["EVALUATOR_ONLY_ORACLE"] = np.maximum(r0, r2)
    metrics = {}
    for arm, reward in results.items():
        metrics[arm] = {"mean_reward": float(reward.mean()),
            "cold_reward": float(reward[cold].mean()),
            "early_reward": float(reward[~cold].mean()),
            "oracle_regret": float((results["EVALUATOR_ONLY_ORACLE"]-reward).mean())}
    # This compares public probe-derived parity with withheld private physics
    # exclusively in evaluator after training, NEVER used as a training label.
    evaluator_cue_accuracy = float(np.mean(np.where(pars > 0, 1, -1) == src["private_eval_truth"]))
    return {"metrics": metrics, "raw_rewards": results, "actions": actions,
            "public_A_probe_accuracy_evaluator_only": evaluator_cue_accuracy}


def _bootstrap_lower(a: np.ndarray, b: np.ndarray, seed: int):
    if a.shape != (COUNT["DEV"],) or b.shape != a.shape:
        raise RuntimeError("R87_PAIRED_BOOTSTRAP_WRONG_WORLD_COUNT")
    rng = np.random.default_rng(np.random.SeedSequence([25290555, seed]))
    indices = rng.integers(0, len(a), (2000, len(a)))
    return float(np.quantile((b-a)[indices].mean(axis=1), .025))


def original_ci(seed: int):
    sha = _admit(seed)
    torch.set_num_threads(1)
    receipt = ROOT / f"local_runs/r87_public_composition_seed{seed}.json"
    if receipt.exists():
        raise FileExistsError("R87_DUPLICATE_OR_UNCERTAIN_FIRST_USE_DENIED")
    began = time.perf_counter()
    result = {"schema": SCHEMA, "freeze_blob": FREEZE_BLOB,
        "parent_R86_SHA": PARENT, "candidate_SHA": sha, "seed": seed,
        "completed_arms": {}, "all_weights_frozen_before_DEV_OOD": False,
        "DEV_OOD_FIRST_USE": False, "original_fit_private_law_labels": False,
        "R85_R86_prior_scored_or_reserved_used": False,
        "status": "R87_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NEGATIVE",
        "scientific_pass": False, "platform": platform.platform()}
    try:
        fit = _assemble("FIT", seed, custody_sha=sha)
        valid = _assemble("FIT_VALIDATION", seed, custody_sha=sha)
        models = _models(seed)
        rng = np.random.default_rng(np.random.SeedSequence([25290611, seed]))
        indices = rng.integers(0, COUNT["FIT"], size=(STEPS, BATCH))
        for arm in ARMS:
            result["completed_arms"][arm] = _train(models[arm], arm, fit, valid, indices)
        if set(result["completed_arms"]) != set(ARMS):
            raise RuntimeError("R87_INCOMPLETE_FROZEN_TRAINED_ARMS")
        result["all_weights_frozen_before_DEV_OOD"] = True
        result["DEV_OOD_FIRST_USE"] = True
        dev = _evaluation("DEV", seed, custody_sha=sha)
        ood = _evaluation("OOD_SHIFT", seed, custody_sha=sha)
        d = _score_actions(models, dev, "DEV")
        o = _score_actions(models, ood, "OOD_SHIFT")
        baseline = max(RAW_CONTROLS, key=lambda arm: d["metrics"][arm]["mean_reward"])
        dm, om = d["metrics"], o["metrics"]
        plus, strong, shuff = dm["MLP64_COMPOSED"], dm[baseline], dm["MLP64_COMPOSED_SHUFFLED"]
        fixed = max(dm["ALWAYS_CALIBRATE"]["mean_reward"], dm["ALWAYS_EXPLOIT_POS"]["mean_reward"])
        agree = float(np.mean(d["actions"]["MLP64_COMPOSED"] == dev["legal_actions"]))
        lower = _bootstrap_lower(d["raw_rewards"][baseline], d["raw_rewards"]["MLP64_COMPOSED"], seed)
        comparison = {
            "best_raw_control_DEV": baseline,
            "composed_minus_best_raw_DEV": plus["mean_reward"]-strong["mean_reward"],
            "composed_minus_shuffled_DEV": plus["mean_reward"]-shuff["mean_reward"],
            "composed_minus_best_fixed_DEV": plus["mean_reward"]-fixed,
            "composed_cold_minus_best_raw_DEV": plus["cold_reward"]-strong["cold_reward"],
            "composed_choice_agreement_with_public_A_rule_DEV": agree,
            "composed_oracle_regret_DEV": plus["oracle_regret"],
            "paired_bootstrap_lower95_composed_minus_raw_DEV": lower,
            "composed_minus_selected_raw_OOD": om["MLP64_COMPOSED"]["mean_reward"]-om[baseline]["mean_reward"],
            "composed_cold_minus_selected_raw_OOD": om["MLP64_COMPOSED"]["cold_reward"]-om[baseline]["cold_reward"],
            "legal_A_probe_parity_accuracy_DEV_evaluator_only": d["public_A_probe_accuracy_evaluator_only"]}
        t = frozen()["evaluation"]["thresholds"]
        checks = {
            "DEV_over_strongest_raw": comparison["composed_minus_best_raw_DEV"] >= t["DEV_composed_minus_strongest_raw_control_min"],
            "DEV_over_corrupted_composition": comparison["composed_minus_shuffled_DEV"] >= t["DEV_composed_minus_shuffled_min"],
            "DEV_over_best_fixed": comparison["composed_minus_best_fixed_DEV"] >= t["DEV_composed_minus_best_fixed_min"],
            "DEV_cold_over_strongest_raw": comparison["composed_cold_minus_best_raw_DEV"] >= t["DEV_composed_cold_minus_strongest_raw_min"],
            "DEV_rule_agreement": agree >= t["DEV_composed_legal_rule_action_agreement_min"],
            "DEV_oracle_regret": plus["oracle_regret"] <= t["DEV_composed_oracle_regret_max"],
            "DEV_paired_bootstrap": lower > 0,
            "OOD_over_DEV_selected_raw": comparison["composed_minus_selected_raw_OOD"] >= t["OOD_composed_minus_DEV_selected_raw_min"],
            "OOD_cold_over_DEV_selected_raw": comparison["composed_cold_minus_selected_raw_OOD"] >= t["OOD_cold_composed_minus_DEV_selected_raw_min"],
            "DEV_public_probe_signal_identifiable": comparison["legal_A_probe_parity_accuracy_DEV_evaluator_only"] >= t["DEV_legal_A_parity_probe_agreement_min"]}
        result["comparison"] = comparison
        result["DEV"] = dm
        result["OOD_SHIFT"] = om
        result["frozen_checks"] = {k: bool(v) for k,v in checks.items()}
        result["status"] = t["on_positive"] if all(checks.values()) else t["on_negative"]
    except BaseException as e:
        result["status"] = "R87_ENGINEERING_OR_RESOURCE_INCOMPLETE_NOT_SCIENTIFIC_NEGATIVE"
        result["error_class"] = type(e).__name__
        raise
    finally:
        result["total_wall_seconds"] = time.perf_counter()-began
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
            handle.write("\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    result = original_ci(args.seed)
    print(json.dumps({"seed":args.seed, "completed_arms":len(result["completed_arms"]),
        "status":result["status"], "DEV_OOD_FIRST_USE":result["DEV_OOD_FIRST_USE"],
        "scientific_pass":False},sort_keys=True))
