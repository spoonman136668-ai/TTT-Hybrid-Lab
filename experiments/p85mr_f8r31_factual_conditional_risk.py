"""F8-R31: fit-only conditional physical-effects learning with risk calibration.

All fitted labels are actually selected, factual return effects from R29B FIT.
The return action is scored before its effect is exposed. Counterfactual
outcomes are evaluator-only, never passed into training or candidate features.
The polynomial/causal-gate representation is an explicit conventional prior,
not evidence that a neural Hybrid learned this representation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics

import torch

from experiments import p85mr_f8r29b_neural_information as b
from experiments import p85mr_f8r30_counterfactual_diagnostic as r30

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r31_factual_conditional_risk_freeze.json"
BLOB = "d14dcad1a36bb472dc3b3b70b94f9e1b1b4ad1e0"
SCHEMA = "ttt-hybrid.f8r31-factual-conditional-risk.v1"
FIT_PER_GROUP = 48
TRAIN_GROUPS = (12, 24, 48)
RIDGE = .00001
FEATURES = ("constant", "public_return_state", "matching_return",
            "informative_matching_return", "informative_times_fifth_delta")
BUCKETS = ("safe", "informed", "uncertain")


def _blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if _blob(raw) != BLOB:
        raise ValueError("F8R31_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    d, t, auth = f["design"], f["precommitted_development_thresholds"], f["authority"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_FIT_AND_RESULTS"
        or f["classification"] != "SOURCE_NEUTRAL_FACTUAL_ONLY_CAUSAL_ESTIMATOR_DEV"
        or f["parent"]["head"] != "3c06a637df37f4d9bdb4192a2f4a4fb81fbd8f3f"
        or f["parent"]["r29b_source_blob"] != "56bfbc8749f4ad41881eea89a215bc6d3d386894"
        or f["parent"]["r30_source_blob"] != "0c12d49272fec4666e9b9b7e5bfd7b897a9eb08f"
        or f["parent"]["r30_freeze_blob"] != r30.BLOB
        or d["fit_groups_total"] != 64 or
        d["learning_curve_train_groups"] != list(TRAIN_GROUPS)
        or d["fit_groups_training"] != "group 0..47"
        or d["fit_groups_heldout_validation"] != "group 48..63"
        or d["features_order"] != ["constant", "public_return_state",
             "selected_return_matches_public_context",
             "probe_matches_context_and_measured_and_selected_return_matches_context",
             "previous_feature_times_actual_fifth_delta"]
        or d["worlds_per_panel"] != 48
        or d["evaluation_panels"] != list(b.PANELS)
        or d["only_evaluator_can_compute_unselected_return_effect"] is not True
        or t["score_variant_train_groups"] != 48
        or t["bootstrap_seed"] != 20489999
        or t["bootstrap_draws"] != 10000
        or t["comparison_rules"]["risk_aware_gain_over_mean_only_min"] !=
           "mean of all four prospective policy mean-regrets per panel; frozen before any implementation or result"
        or t["independent_scientific_pass"] is not False
        or any(v is not False for v in auth.values())):
        raise ValueError("F8R31_PROTOCOL_OR_AUTHORITY_DRIFT")
    b.locked()
    r30.locked()
    return f


def _decode_key(key):
    """Invert factual P80 action-key normalization, not evaluator physics."""
    if tuple(key.shape) != (16,):
        raise ValueError("F8R31_ACTION_KEY_SHAPE")
    denom = float(key[0])
    if not .1 < denom < 1:
        raise ValueError("F8R31_BAD_KEY_NORMALIZER")
    state = 2.4 * math.atanh(max(-.98, min(.98, float(key[1])/denom)))
    context_raw = 3.0 * math.atanh(max(-.98, min(.98, float(key[2])/denom)))
    context = int(round(context_raw))
    if context not in (0, 1) or abs(context_raw-context) > .0001:
        raise ValueError("F8R31_INVALID_PUBLIC_CONTEXT")
    action = int(float(key[6]) > float(key[5]))
    if (abs(float(key[5])/denom - float(action == 0)) > .0001
        or abs(float(key[6])/denom - float(action == 1)) > .0001):
        raise ValueError("F8R31_INVALID_FACTUAL_ACTION_KEY")
    return state, context, action


def _physical_delta(normalized):
    y = float(normalized)
    if not math.isfinite(y) or not -.98 < y < .98:
        raise ValueError("F8R31_INVALID_PHYSICAL_EFFECT")
    return 2.4 * math.atanh(y)


def _feature_from_factual(actions, effects):
    """Only public keys and actual selected fifth effect; no unseen returns."""
    if (actions.ndim != 2 or effects.shape != actions.shape
        or actions.shape[0] < 14 or actions.shape[1] != 16):
        raise ValueError("F8R31_INVALID_FACTUAL_HISTORY")
    state, ctx, ret = _decode_key(actions[-1])
    probe_state, pctx, fifth = _decode_key(actions[4])
    if pctx != ctx or abs(probe_state - .4) > .0001:
        raise ValueError("F8R31_INVALID_PUBLIC_PROBE")
    # Zero measurement is a deliberately erased observation, not physical zero.
    measured = bool(torch.count_nonzero(effects[4]).item())
    matched = ret == ctx
    informed = matched and fifth == ctx and measured
    delta = _physical_delta(effects[4,0]) if measured else 0.
    info = float(informed)
    return ([1., state, float(matched), info, info * delta],
            "safe" if not matched else ("informed" if informed else "uncertain"))


def _fit_row(row):
    # Single actually selected return effect; no hindsight/oracle label.
    features, bucket = _feature_from_factual(row["action"], row["effect"])
    target = _physical_delta(row["physical_target"])
    return features, bucket, target


def fit_factual_only(fit, group_count):
    if group_count not in TRAIN_GROUPS or len(fit) != 3072:
        raise ValueError("F8R31_UNREGISTERED_FIT")
    rows = [_fit_row(r) for r in fit[:group_count*FIT_PER_GROUP]]
    X = torch.tensor([r[0] for r in rows], dtype=torch.float64)
    y = torch.tensor([r[2] for r in rows], dtype=torch.float64)
    L = RIDGE * torch.eye(len(FEATURES), dtype=torch.float64)
    L[0,0] = 0.
    weights = torch.linalg.solve(X.T @ X + L, X.T @ y)
    error = X @ weights - y
    residual = X.T @ error + L @ weights
    variance = {bucket: statistics.fmean(
        float(error[i]**2) for i,r in enumerate(rows) if r[1] == bucket)
        for bucket in BUCKETS}
    if not (bool(torch.isfinite(weights).all())
            and all(math.isfinite(v) and v >= 0 for v in variance.values())):
        raise ValueError("F8R31_INVALID_RIDGE_FIT")
    return {"groups":group_count, "examples":len(rows),
            "feature_count":len(FEATURES), "weights":weights.tolist(),
            "conditional_variance":variance,
            "fit_mse":float(torch.mean(error**2)),
            "normal_equation_gradient_max":float(residual.abs().max()),
            "l2_regularization":RIDGE,
            "coefficient_sha256":hashlib.sha256(
                json.dumps(weights.tolist(),allow_nan=False).encode()).hexdigest()}


def _estimate(model, actions, effects, risk_aware):
    x, bucket = _feature_from_factual(actions, effects)
    delta = math.fsum(a*w for a,w in zip(x,model["weights"]))
    s = x[1]
    variance = model["conditional_variance"][bucket] if risk_aware else 0.
    return {"predicted_return":s+delta, "variance":variance,
            "bucket":bucket, "physical_delta":delta}


def _choose(w, policy, model, *, risk_aware, erase=False):
    actions, effects = b._candidate_inputs(w, policy, erase)
    if actions.shape[0] != 2 or effects.shape != actions.shape:
        raise ValueError("F8R31_BAD_PREACTION_CANDIDATES")
    options = [_estimate(model, actions[a], effects[a], risk_aware)
               for a in (0,1)]
    risks = [(v["predicted_return"]-w["goal"])**2 + v["variance"]
             for v in options]
    return min((0,1), key=lambda a:(risks[a],a)), options


def _score(w, choices):
    """Evaluator only; no call to this function on a FIT or model-training path."""
    if len(w) != len(choices) or len(w) != 48:
        raise ValueError("F8R31_UNBALANCED_EVALUATION")
    return [b.r29.actual_regret(world, action)
            for world,action in zip(w,choices)]


def _policy(model, panel, policy, *, risk_aware, erase=False):
    worlds = b.r29.worlds(panel)
    decisions = [_choose(w,policy,model,risk_aware=risk_aware,erase=erase)
                 for w in worlds]
    choices = [x[0] for x in decisions]
    regrets = _score(worlds,choices)
    group = [statistics.fmean(regrets[i*4:(i+1)*4]) for i in range(12)]
    return {"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":group,
            "choices":choices,
            "action1_count":sum(choices),
            "uncertain_candidate_count":sum(
                x["bucket"] == "uncertain" for decision in decisions
                for x in decision[1]),
            "mean_candidate_variance":statistics.fmean([
                x["variance"] for dec in decisions for x in dec[1]])}


def _validation(fit, model):
    subset = fit[48*FIT_PER_GROUP:]
    residuals = []
    for row in subset:
        feature,_,y = _fit_row(row)
        residuals.append((math.fsum(x*w for x,w in
                 zip(feature,model["weights"]))-y)**2)
    if len(subset) != 16*FIT_PER_GROUP:
        raise ValueError("F8R31_FIT_VALIDATION_LEAKAGE")
    return statistics.fmean(residuals)


def _bootstrap_lower(previous, current, seed, draws):
    rng = random.Random(seed)
    gains = []
    for _ in range(draws):
        idx = [rng.randrange(12) for __ in range(12)]
        gains.append(statistics.fmean(previous[i]-current[i] for i in idx))
    return sorted(gains)[int(.025*draws)]


def _judge(f, learned, panels, r30_reference, curve):
    th = f["precommitted_development_thresholds"]
    gradient_ok = all(row["normal_equation_gradient_max"] <=
                      th["fit_normal_equation_gradient_max"] for row in curve)
    drift = math.sqrt(math.fsum((a-bb)**2 for a,bb in zip(
        curve[1]["weights"],curve[2]["weights"])))
    convergence_ok = gradient_ok and drift <= th["coefficient_l2_diff_24_vs_48_max"]
    decisions = {}
    for panel in b.PANELS:
        p=panels[panel]
        risk=p["risk_aware"]
        average_gain = statistics.fmean(p["mean_only"][policy]["mean_regret"] -
                               risk[policy]["mean_regret"] for policy in b.POLICIES)
        before=r30_reference[panel]["active_information"]["choices"]
        reference=_score(b.r29.worlds(panel), before)
        old_groups=[statistics.fmean(reference[g*4:(g+1)*4]) for g in range(12)]
        new=risk["active_information"]
        bootstrap=_bootstrap_lower(old_groups,new["pergroup_regret"],
                       th["bootstrap_seed"],th["bootstrap_draws"])
        old_mean=statistics.fmean(reference)
        erasure=p["erased_risk_aware"]["mean_regret"]-new["mean_regret"]
        gain=old_mean-new["mean_regret"]
        passed=bool(
            new["mean_regret"] <= th["mean_regret_risk_aware_max_both_panels"]
            and average_gain >= th["risk_aware_gain_over_mean_only_min_both_panels"]
            and gain >= th["risk_aware_gain_over_r30_ridge_min_both_panels"]
            and erasure >= th["fifth_erasure_penalty_min_both_panels"]
            and bootstrap > th["paired_group_bootstrap_lower_gt"])
        decisions[panel]={"active_risk_regret":new["mean_regret"],
                    "average_gain_over_mean_only_all_four_policies":average_gain,
                    "gain_over_R30_ridge_active":gain,
                    "R30_ridge_active_regret":old_mean,
                    "active_fifth_erasure_penalty":erasure,
                    "paired_bootstrap_lower":bootstrap,
                    "panel_development_pass":passed}
    return {"fit_convergence_pass":convergence_ok,
            "fit_coefficient_drift_24_to_48":drift,
            "panels":decisions,
            "source_neutral_development_pass":convergence_ok and
                all(v["panel_development_pass"] for v in decisions.values()),
            "independent_scientific_pass":False}


def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=b.fit_cohort()
    curve=[]
    for groups in TRAIN_GROUPS:
        model=fit_factual_only(fit,groups)
        model["fit_only_validation_mse"]=_validation(fit,model)
        curve.append(model)
    learned=curve[-1]
    old_ridge=r30.fit_only_ridge(fit)
    previous=r30.diagnose("fit_only_ridge",old_ridge)
    panels={}
    for panel in b.PANELS:
        risk={policy:_policy(learned,panel,policy,risk_aware=True)
              for policy in b.POLICIES}
        mean={policy:_policy(learned,panel,policy,risk_aware=False)
              for policy in b.POLICIES}
        erased=_policy(learned,panel,"active_information",
                       risk_aware=True,erase=True)
        panels[panel]={"risk_aware":risk, "mean_only":mean,
                       "erased_risk_aware":erased}
    # JSON/cold restoration verifies fixed coefficients and decisions.
    restored=json.loads(json.dumps(learned,sort_keys=True,allow_nan=False))
    if any(_policy(learned,p,pol,risk_aware=True)["choices"] !=
           _policy(restored,p,pol,risk_aware=True)["choices"]
           for p in b.PANELS for pol in b.POLICIES):
        raise ValueError("F8R31_COLD_RESTORE_POLICY_DRIFT")
    judgment=_judge(frozen,learned,panels,previous,curve)
    return {"schema":SCHEMA,
            "classification":"SOURCE_NEUTRAL_FACTUAL_ONLY_CAUSAL_ESTIMATOR_DEV",
            "freeze_blob":BLOB,
            "parent_head":"3c06a637df37f4d9bdb4192a2f4a4fb81fbd8f3f",
            "same_factual_R29B_episodes":len(fit),
            "training_source_selected_return_factual_only":True,
            "factual_fit_curve":curve,
            "R30_reference":{p:{
              "active_regret":previous[p]["active_information"]["mean_regret"],
              "active_choices":previous[p]["active_information"]["choices"]}
              for p in b.PANELS},
            "panels":panels,
            "judgment":judgment,
            "neural_optimizer_updates":0,
            "independent_neural_baseline_convergence_qualified":False,
            "matched_neural_FLOPs":False,
            "independent_scientific_pass":False,
            "original_external_effects":0,
            "spent_F7_replay":False,
            "cold_restore_actions_equal":True,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path=ROOT/"local_runs/p85mr_f8r31_factual_conditional_risk_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R31_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R31_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
