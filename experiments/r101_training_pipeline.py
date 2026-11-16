"""R101 prospective nine-arm training and physical scoring pipeline.

ENGINEERING-ONLY until R101._admit and immutable freeze independently authorize
one original first use. Do not call from unit qualification, retries or a second
run. All source creation is delegated to R101._world, currently DENIED.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PANELS = ("FIT", "VALID")
SCORING_PANELS = ("DEV_FAMILIAR", "DEV_UNKNOWN", "OOD_SHIFT", "DELAYED_COLD", "REVERSAL")
ALL_PANELS = TRAIN_PANELS + SCORING_PANELS
STEPS, BATCH, SUPPORT = r101.STEPS, r101.BATCH, r101.SUPPORT
SEEDS, ARMS = r101.SEEDS, r101.ARMS


def protocol():
    f = r101.frozen()
    if (tuple(f["source"]["counts"]) != ALL_PANELS
        or f["training"]["steps_each_arm"] != STEPS
        or f["training"]["batch"] != BATCH
        or f["training"]["all_nine_arms_fully_trained_before_any_DEV_first_use"] is not True
        or f["source"]["support_factual_interventions"] != SUPPORT
        or f["source"]["one_original_first_use_per_seed"] is not True
        or f["evaluation"]["paired_bootstrap_draws"] != 2000
        or f["source"]["no_prior_cohort_replay"] is not True
        or f["evaluation"]["independent_scientific_pass"] is not False):
        raise PermissionError("R101_TRAINING_SCORING_FREEZE_DRIFT")
    return f



def non_scoring_design_audit():
    """Exact combinatorial source-design audit; NEVER generate scored worlds.

    Six randomly and independently sampled three-operand Boolean inputs
    cannot always distinguish candidate operators. This is a mathematical
    information limit even with hypothetical noiseless full-law feedback.
    Real R85 factual returns and noisy public cues may add uncertainty.
    These diagnostics do NOT change precommitted scientific thresholds.
    """
    f = protocol()
    candidates = tuple(r101.KNOWN + r101.UNSEEN)
    patterns = tuple((int(i & 4 != 0), int(i & 2 != 0),
                      int(i & 1 != 0)) for i in range(8))
    signatures = {}
    for family in candidates:
        signatures[family] = tuple(r101._operator_truth(
            np.array([*([2*b-1 for b in p]), -1], dtype=np.int64),
            (0, 1, 2), family) for p in patterns)
    if len(set(signatures.values())) != len(candidates):
        raise RuntimeError("R101_OPERATOR_FAMILIES_NOT_DISTINCT")
    collisions = {}
    for unseen in r101.UNSEEN:
        comparisons = []
        for familiar in r101.KNOWN:
            diff = sum(a != b for a,b in zip(
                signatures[unseen],signatures[familiar]))
            comparisons.append({
                "known":familiar,"different_patterns_of_eight":diff,
                "probability_six_iid_supports_avoid_difference":
                    ((8-diff)/8)**SUPPORT,
                "probability_query_is_different_after_undistinguishing_support":
                    ((8-diff)/8)**SUPPORT * (diff/8),
            })
        comparisons.sort(key=lambda v:(
            -v["probability_six_iid_supports_avoid_difference"],
            v["known"]))
        collisions[unseen] = comparisons
    balance = {}
    for panel,count in f["source"]["counts"].items():
        fams = {
            name:sum(operator_family(panel,index,SUPPORT)==name
                     for index in range(count))
            for name in (r101.KNOWN if panel in
                         ("FIT","VALID","DEV_FAMILIAR") else r101.UNSEEN)}
        if sum(fams.values()) != count or max(fams.values())-min(fams.values())>1:
            raise RuntimeError("R101_FROZEN_PANEL_FAMILY_BALANCE_INVALID")
        chosen = {
            str(action):sum(chosen_action(seed,index,SUPPORT)==action
                            for seed in SEEDS for index in range(count))
            for action in (r85.CALIBRATE,r85.EXPLOIT_POS)}
        if abs(chosen[str(r85.CALIBRATE)]-chosen[str(r85.EXPLOIT_POS)])>len(SEEDS):
            raise RuntimeError("R101_FROZEN_PANEL_QUERY_ACTION_BALANCE_INVALID")
        balance[panel] = {"count":count,"family_counts":fams,
                          "query_action_counts_all_three_seeds":chosen,
                          "six_support_actions_each_episode":"3 CALIBRATE + 3 EXPLOIT_POS"}
    return {
        "schema":"ttt-hybrid.r101.non-scoring-source-design-audit.v1",
        "original_scored_data_accessed":False,
        "unit_and_combinatorial_only":True,
        "operator_signatures_evaluator_only":True,
        "frozen_first_use_granted":f["authority"]["protected_scored_first_use"],
        "assumption":"six independent uniform factor triples, hypothetical exact law observations",
        "six_supports_cannot_guarantee_family_identification":True,
        "unseen_vs_known":collisions,
        "precommitted_panel_source_design":balance,
        "science_thresholds_and_frozen_source_unchanged":True,
        "independent_scientific_pass":False
    }

def operator_family(panel, index, event):
    """Evaluator-only family selection. Never pass family code to any model."""
    f = protocol()
    if panel not in ALL_PANELS or not isinstance(index, int) or not (
        0 <= index < f["source"]["counts"][panel]):
        raise PermissionError("R101_NONFROZEN_PANEL_OR_INDEX")
    if not isinstance(event, int) or not 0 <= event <= SUPPORT:
        raise PermissionError("R101_NONFROZEN_SUPPORT_EVENT")
    if panel in ("FIT", "VALID", "DEV_FAMILIAR"):
        return r101.KNOWN[index % len(r101.KNOWN)]
    target = r101.UNSEEN[index % len(r101.UNSEEN)]
    # On REVERSAL, physical law changes halfway through observed factual
    # supports. The query obeys the NEW family. All weights remain fixed.
    if panel == "REVERSAL" and event < SUPPORT // 2:
        return r101.UNSEEN[(index + 1) % len(r101.UNSEEN)]
    return target


def chosen_action(seed, index, event):
    if seed not in SEEDS or not isinstance(index, int) or index < 0:
        raise PermissionError("R101_NONFROZEN_ACTION_SEED_INDEX")
    if event < 0 or event > SUPPORT:
        raise PermissionError("R101_NONFROZEN_ACTION_EVENT")
    if event < SUPPORT:
        return r85.CALIBRATE if event % 2 == 0 else r85.EXPLOIT_POS
    return (r85.CALIBRATE if (index // 4 + SEEDS.index(seed)) % 2 == 0
            else r85.EXPLOIT_POS)


def derange_public_support(actions, slots, seed, panel, index):
    """Fixed RNG and public strata ONLY, never inspect hidden law/reward."""
    f = protocol()
    if (seed not in SEEDS or panel not in ALL_PANELS
        or not isinstance(index, int) or not 0 <= index < f["source"]["counts"][panel]
        or actions.shape != (SUPPORT,) or slots.shape != (SUPPORT, 5, 13)):
        raise PermissionError("R101_SUPPORT_DERANGEMENT_SOURCE_INVALID")
    if not np.all(np.isfinite(slots)):
        raise ValueError("R101_SUPPORT_DERANGEMENT_NONFINITE_PUBLIC")
    public_keys = [(int(actions[j]),float(slots[j,-1,12])) for j in range(SUPPORT)]
    random = np.random.default_rng(np.random.SeedSequence([
        f["source"]["salts"]["SUPPORT_SHUFFLE"], f["source"]["salts"][panel],
        seed, index]))
    perm = np.arange(SUPPORT)
    for key in sorted(set(public_keys)):
        group = np.asarray([j for j, label in enumerate(public_keys)
                            if label == key],dtype=np.int64)
        if len(group) < 2:
            raise RuntimeError("R101_SHUFFLE_CELL_HAS_NO_DERANGEMENT")
        step = int(random.integers(1, len(group)))
        perm[group] = np.roll(group,step)
    if (not np.array_equal(np.sort(perm),np.arange(SUPPORT))
        or np.any(perm == np.arange(SUPPORT))
        or any(public_keys[j] != public_keys[int(perm[j])]
               for j in range(SUPPORT))):
        raise RuntimeError("R101_SHUFFLED_SUPPORT_NOT_TRUE_PUBLIC_DERANGEMENT")
    return perm


def episode(panel, seed, index, sha, paired=False):
    """One physical operator per episode, six factual support experiences.

    FIT/VALID: query outcome of only the precommitted chosen action.
    Scoring: paired physical counterfactuals permitted only to evaluator.
    """
    f = protocol()
    if panel not in ALL_PANELS or seed not in SEEDS or not (
        isinstance(index,int) and 0 <= index < f["source"]["counts"][panel]):
        raise PermissionError("R101_EPISODE_NOT_IN_FROZEN_COHORT")
    if bool(paired) != (panel in SCORING_PANELS):
        raise PermissionError("R101_UNCHOSEN_TRAINING_RETURN_PROHIBITED")
    public, rewards, actions = [], [], []
    for event in range(SUPPORT):
        family = operator_family(panel,index,event)
        world = r101._world(panel,seed,index,sha,event=event,family=family)
        action = chosen_action(seed,index,event)
        slots,ret = r101._public_view(world,action)
        public.append(slots)
        rewards.append(ret)
        actions.append(action)
    support_slots = np.stack(public).astype(np.float32)
    support_rewards = np.asarray(rewards,np.float32)
    support_actions = np.asarray(actions,np.int8)
    perm = derange_public_support(support_actions,support_slots,seed,panel,index)
    query_world = r101._world(panel,seed,index,sha,event=SUPPORT,
                              family=operator_family(panel,index,SUPPORT))
    result = {
        "support_slots":support_slots,"support_rewards":support_rewards,
        "support_actions":support_actions,
        "support_rewards_shuffled":support_rewards[perm].copy(),
        "shuffle_perm":perm,
        "support_factual_count":SUPPORT,
    }
    if paired:
        q0,r0 = r101._public_view(query_world,r85.CALIBRATE)
        q2,r2 = r101._public_view(query_world,r85.EXPLOIT_POS)
        if (not np.array_equal(q0[:4],q2[:4])
            or not np.array_equal(q0[-1,4:],q2[-1,4:])
            or q0[-1,r85.CALIBRATE] != 1
            or q2[-1,r85.EXPLOIT_POS] != 1):
            raise RuntimeError("R101_PHYSICAL_COUNTERFACTUAL_PREACTION_LEAK")
        # Values below are NEVER in the training source.
        result.update({"query0":q0,"query2":q2,"reward0":float(r0),
                       "reward2":float(r2),"private_law":int(query_world["law_a"]),
                       "cold":bool(query_world["pivot"]==r85.COLD),
                       "changed_family":panel=="REVERSAL"})
    else:
        action = chosen_action(seed,index,SUPPORT)
        query, reward = r101._public_view(query_world,action)
        result.update({"query":query,"target":float(reward),
                       "query_factual_action":action})
        if any(key in result for key in ("query0","query2","reward0","reward2",
                                        "private_law")):
            raise RuntimeError("R101_PRIVILEGED_QUERY_RETURN_IN_FIT_VALID")
    return result


def source(panel, seed, sha):
    """No original scored panel is ever opened on engineering-only branch."""
    f = protocol()
    if panel not in ALL_PANELS or seed not in SEEDS:
        raise PermissionError("R101_SOURCE_NOT_REGISTERED")
    count = f["source"]["counts"][panel]
    paired = panel in SCORING_PANELS
    records = [episode(panel,seed,i,sha,paired=paired) for i in range(count)]
    common = {
        "support_slots":np.stack([e["support_slots"] for e in records]),
        "support_rewards":np.stack([e["support_rewards"] for e in records]),
        "support_rewards_shuffled":np.stack([e["support_rewards_shuffled"] for e in records]),
        "support_actions":np.stack([e["support_actions"] for e in records]),
    }
    if paired:
        common.update({
            "query0":np.stack([e["query0"] for e in records]),
            "query2":np.stack([e["query2"] for e in records]),
            "reward0":np.asarray([e["reward0"] for e in records],np.float32),
            "reward2":np.asarray([e["reward2"] for e in records],np.float32),
            "private_law":np.asarray([e["private_law"] for e in records],np.int8),
            "cold":np.asarray([e["cold"] for e in records],bool),
        })
    else:
        common.update({
            "query":np.stack([e["query"] for e in records]),
            "target":np.asarray([e["target"] for e in records],np.float32),
            "query_factual_action":np.asarray(
                [e["query_factual_action"] for e in records],np.int8),
        })
        if set(common) & {"query0","query2","reward0","reward2","private_law"}:
            raise RuntimeError("R101_FIT_VALID_COUNTERFACTUAL_LEAK")
    return common



# A precommitted NON-IDENTITY bijection on the original four physical
# A-factor roles. Mapping old index i -> new ROLE_RELABEL[i]. Each support
# event and the heldout query in an episode receive the SAME relabel.
ROLE_RELABEL = (2, 0, 3, 1)


def relabel_public_roles(support_slots, query):
    """Physical role-name symmetry; NEVER rewrite law, reward or action.

    The four public A events are stored by physical role index. A valid
    relabel must permute (a) observed cue, (b) semantic operand-position
    marker and (c) physical role-ID encoding TOGETHER, in every factual
    support and heldout query. The event order alone is not a role renaming.
    """
    if (support_slots.ndim != 4 or support_slots.shape[1:] != (SUPPORT,5,13)
        or query.ndim != 3 or query.shape[1:] != (5,13)
        or len(support_slots) != len(query)
        or tuple(sorted(ROLE_RELABEL)) != (0,1,2,3)):
        raise ValueError("R101_PHYSICAL_ROLE_RELABEL_INPUT_INVALID")
    order = np.argsort(np.asarray(ROLE_RELABEL,dtype=np.int64))
    slots_out = support_slots.copy()
    query_out = query.copy()
    slots_out[:,:,:4,:] = support_slots[:,:,order,:]
    query_out[:,:4,:] = query[:,order,:]
    role_tags = np.arange(1,5,dtype=np.float32) / 4.0
    slots_out[:,:,:4,9] = role_tags[None,None,:]
    query_out[:,:4,9] = role_tags[None,:]
    if (not np.array_equal(slots_out[:,:,-1,:],support_slots[:,:,-1,:])
        or not np.array_equal(query_out[:,-1,:],query[:,-1,:])
        or not np.array_equal(np.sort(slots_out[:,:,:4,8],axis=2),
                              np.sort(support_slots[:,:,:4,8],axis=2))
        or not np.array_equal(np.sort(query_out[:,:4,8],axis=1),
                              np.sort(query[:,:4,8],axis=1))):
        raise RuntimeError("R101_PHYSICAL_RELABEL_BROKE_CAUSAL_PUBLIC_STATE")
    return slots_out, query_out


def _inputs(data, indices, arm, action=None, permute=False):
    slots = data["support_slots"][indices]
    rewards = (data["support_rewards_shuffled"][indices]
               if arm == r101.SHUFFLED else data["support_rewards"][indices])
    if action is None:
        query = data["query"][indices]
    else:
        query = data["query0" if action == r85.CALIBRATE else "query2"][indices]
    if permute:
        # Apply a TRUE public role renaming, not an arbitrary event-row swap:
        # copy each cue, its semantic operand tag AND its new physical role id.
        # This is a representation-symmetry negative check; it does not
        # manufacture unseen operator families or unseen scientific data.
        slots, query = relabel_public_roles(slots, query)
    return (torch.from_numpy(np.ascontiguousarray(slots)),
            torch.from_numpy(np.ascontiguousarray(rewards)),
            torch.from_numpy(np.ascontiguousarray(query)))


def predict(model, arm, data, action=None, permute=False):
    n = len(data["support_slots"])
    results = []
    model.eval()
    with torch.no_grad():
        for offset in range(0,n,32):
            take = np.arange(offset,min(offset+32,n))
            x,y,q = _inputs(data,take,arm,action,permute=permute)
            value = model(x,y,q)
            if value.shape != (len(take),):
                raise RuntimeError("R101_UNTRAINED_QUERY_VALUE_SHAPE")
            results.append(value.detach().cpu().numpy())
    model.train()
    ret = np.concatenate(results).astype(np.float64)
    if ret.shape != (n,) or not np.all(np.isfinite(ret)):
        raise RuntimeError("R101_NONFINITE_MODEL_QUERY_VALUES")
    return ret


def fit_arm(model, arm, fit, valid, indices):
    """Full frozen budget, common FIT indices and actual query return only."""
    if (arm not in ARMS or indices.shape != (STEPS,BATCH)
        or set(fit) & {"reward0","reward2","query0","query2","private_law"}
        or set(valid) & {"reward0","reward2","query0","query2","private_law"}):
        raise PermissionError("R101_TRAINING_PRIVILEGED_DATA_OR_STEPS")
    if len(fit["target"]) != protocol()["source"]["counts"]["FIT"]:
        raise RuntimeError("R101_FIT_COUNT_DRIFT")
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    full_target = torch.from_numpy(fit["target"])
    started = time.perf_counter()
    initial_valid = float(np.mean((predict(model,arm,valid)-valid["target"])**2))
    model.train()
    for take in indices:
        slots,rewards,query = _inputs(fit,take,arm)
        value = model(slots,rewards,query)
        loss = F.mse_loss(value,full_target[take])
        if not torch.isfinite(loss):
            raise RuntimeError("R101_NONFINITE_FACTUAL_TRAIN_LOSS")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
    elapsed = time.perf_counter()-started
    train_mse = float(np.mean((predict(model,arm,fit)-fit["target"])**2))
    valid_mse = float(np.mean((predict(model,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(x) for x in (elapsed,train_mse,valid_mse,initial_valid)):
        raise RuntimeError("R101_NONFINITE_TRAINING_METRIC")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active = sum(p.numel() for p in model.parameters()
                 if p.requires_grad and p.grad is not None)
    if arm in (r101.ROLE_SET,r101.GRU,r101.MLP,r101.STATIC) and active != trainable:
        raise RuntimeError("R101_CONVENTIONAL_COMPARATOR_HAS_DORMANT_PARAMETERS")
    return {"steps":STEPS,"batches":STEPS,"factual_query_exposures":STEPS*BATCH,
            "support_factual_exposures":STEPS*BATCH*SUPPORT,
            "trainable_parameters":trainable,
            "active_gradient_parameters":active,
            "dormant_gradient_parameters":trainable-active,
            "fit_MSE":train_mse,"valid_MSE":valid_mse,
            "valid_before_MSE":initial_valid,
            "training_wall_seconds":elapsed,
            "valid_report_only":True,"flops_not_measured":True}


def score(models, panel, data):
    """Evaluator-only paired physical counterfactual; never during training."""
    if panel not in SCORING_PANELS or set(data) & {"query","target"}:
        raise PermissionError("R101_PROTECTED_EVALUATOR_ONLY_PANEL")
    n = len(data["reward0"])
    choices, rewards, metrics = {}, {}, {}
    true_choice = np.where(data["private_law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
    oracle = np.maximum(data["reward0"],data["reward2"])
    for arm in ARMS:
        v0 = predict(models[arm],arm,data,action=r85.CALIBRATE)
        v2 = predict(models[arm],arm,data,action=r85.EXPLOIT_POS)
        action = np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
        realized = np.where(action==r85.CALIBRATE,data["reward0"],data["reward2"])
        choices[arm],rewards[arm] = action,realized
        metrics[arm] = {
            "realized_reward_mean":float(np.mean(realized)),
            "regret_to_physical_oracle":float(np.mean(oracle-realized)),
            "private_true_law_action_agreement":float(np.mean(action==true_choice)),
        }
        if np.any(data["cold"]):
            metrics[arm]["cold_reward_mean"] = float(np.mean(realized[data["cold"]]))
    permute_actions = np.where(
        predict(models[r101.NATIVE],r101.NATIVE,data,action=r85.CALIBRATE,permute=True) >
        predict(models[r101.NATIVE],r101.NATIVE,data,action=r85.EXPLOIT_POS,permute=True),
        r85.CALIBRATE,r85.EXPLOIT_POS)
    return {
        "metrics":metrics,"choices":choices,"rewards":rewards,
        "role_order_permutation_action_agreement":float(
            np.mean(choices[r101.NATIVE] == permute_actions)),
        "oracle_mean":float(np.mean(oracle)),
        "actual_physical_worlds":n,
    }


def lower_bootstrap(a,b,seed):
    if a.shape != b.shape or a.ndim != 1 or seed not in SEEDS:
        raise ValueError("R101_PAIRED_BOOTSTRAP_WRONG_COHORT")
    rng = np.random.default_rng(np.random.SeedSequence(
        [protocol()["source"]["salts"]["BOOTSTRAP"],seed]))
    take = rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[take].mean(axis=1),.025))


def measures(seed, scores):
    """Separate native, mechanism, external-control, transfer, and efficiency."""
    dev,ood,cold,reverse = (scores[k] for k in
                           ("DEV_UNKNOWN","OOD_SHIFT","DELAYED_COLD","REVERSAL"))
    d = lambda s,name:s["metrics"][name]["realized_reward_mean"]
    others = (r101.EXTERNAL,r101.ROLE_SET,r101.GRU,r101.MLP,r101.STATIC)
    best = max(others,key=lambda arm:d(dev,arm))
    ood_best = max(others,key=lambda arm:d(ood,arm))
    metrics = {
        "native_minus_strongest_nonengineered_DEV_UNKNOWN_reward":
            d(dev,r101.NATIVE)-d(dev,best),
        "paired_bootstrap_lower95_native_minus_strongest":
            lower_bootstrap(dev["rewards"][r101.NATIVE],dev["rewards"][best],seed),
        "native_unknown_query_action_accuracy":
            dev["metrics"][r101.NATIVE]["private_true_law_action_agreement"],
        "native_minus_no_support_outcome_DEV_UNKNOWN":
            d(dev,r101.NATIVE)-d(dev,r101.NO_OUTCOME),
        "native_minus_shuffled_support_outcomes_DEV_UNKNOWN":
            d(dev,r101.NATIVE)-d(dev,r101.SHUFFLED),
        "native_minus_state_reset_DELAYED_COLD":
            d(cold,r101.NATIVE)-d(cold,r101.RESET),
        "native_minus_external_learned_state_DEV_UNKNOWN":
            d(dev,r101.NATIVE)-d(dev,r101.EXTERNAL),
        "native_minus_best_baseline_OOD_SHIFT":
            d(ood,r101.NATIVE)-d(ood,ood_best),
        "role_permutation_action_agreement":
            dev["role_order_permutation_action_agreement"],
        "reversal_adaptation_above_no_feedback":
            d(reverse,r101.NATIVE)-d(reverse,r101.NO_OUTCOME),
    }
    checks = protocol()["evaluation"]["checks"]
    thresholds = {
        "native_minus_strongest_nonengineered_DEV_UNKNOWN_reward":
            "native_minus_strongest_nonengineered_DEV_UNKNOWN_reward_min",
        "paired_bootstrap_lower95_native_minus_strongest":
            "paired_bootstrap_lower95_native_minus_strongest_min",
        "native_unknown_query_action_accuracy":
            "native_unknown_query_action_accuracy_min",
        "native_minus_no_support_outcome_DEV_UNKNOWN":
            "native_minus_no_support_outcome_DEV_UNKNOWN_min",
        "native_minus_shuffled_support_outcomes_DEV_UNKNOWN":
            "native_minus_shuffled_support_outcomes_DEV_UNKNOWN_min",
        "native_minus_state_reset_DELAYED_COLD":
            "native_minus_state_reset_DELAYED_COLD_min",
        "native_minus_external_learned_state_DEV_UNKNOWN":
            "native_minus_external_learned_state_DEV_UNKNOWN_min",
        "native_minus_best_baseline_OOD_SHIFT":
            "native_minus_best_baseline_OOD_SHIFT_min",
        "role_permutation_action_agreement":
            "role_permutation_action_agreement_min",
        "reversal_adaptation_above_no_feedback":
            "reversal_adaptation_above_no_feedback_min",
    }
    if set(thresholds.values()) != set(checks):
        raise RuntimeError("R101_SCIENCE_GATE_MAPPING_NOT_FROZEN")
    verdict = {key:bool(np.isfinite(metrics[key]) and metrics[key]>=checks[label])
               for key,label in thresholds.items()}
    return {"metrics":metrics,"checks":verdict,"best_nonengineered":best,
            "best_ood_baseline":ood_best,"frozen_all_checks_pass":all(verdict.values())}


def original_ci(seed):
    """Intentionally not executable: original protected first-use NOT granted.

    After separate scope/owner verification, exact-SHA/run/source custody and
    project-specific first-use admission must be implemented IN A NEW candidate
    and fully qualified BEFORE touching protected scored cohorts.
    """
    r101._admit(seed=seed)
    # No side effects before successful owner/source admission.
    freeze = protocol()
    torch.set_num_threads(1)
    sha = os.getenv("EXPECTED_CANDIDATE_SHA","")
    path = ROOT / f"local_runs/r101_unknown_operator_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R101_ORIGINAL_SCIENCE_REPLAY_OR_UNCERTAIN_EFFECT")
    started = time.perf_counter()
    report = {
        "schema":r101.SCHEMA,"candidate_SHA":sha,"seed":seed,
        "preimplementation_freeze_blob":r101.FREEZE_BLOB,
        "parent_r100_sealed_SHA":r101.PARENT,
        "status":"R101_ENGINEERING_OR_RESOURCE_INCOMPLETE",
        "scientific_pass":False,"official_GPU":False,
        "scored_heldout_first_used":False,"all_nine_arms_fully_trained":False,
        "prior_scored_source_or_weights_reused":False,
        "torch":str(torch.__version__),"platform":platform.platform(),
        "trained":{},"evaluation":{},
    }
    try:
        fit = source("FIT",seed,sha)
        valid = source("VALID",seed,sha)
        models = r101.make_models(seed)
        minibatches = np.random.default_rng(np.random.SeedSequence(
            [freeze["source"]["salts"]["BATCH"],seed])).integers(
                0,len(fit["target"]),size=(STEPS,BATCH))
        for arm in ARMS:
            report["trained"][arm] = fit_arm(models[arm],arm,fit,valid,minibatches)
            if time.perf_counter()-started > freeze["training"]["whole_process_CPU_seconds_per_seed_cap"]:
                raise TimeoutError("R101_ORIGINAL_PROCESS_BUDGET_EXCEEDED")
        report["all_nine_arms_fully_trained"] = True
        # Don't construct ANY scored DEV physical worlds before final weights.
        report["scored_heldout_first_used"] = True
        reports = {}
        for panel in SCORING_PANELS:
            reports[panel] = score(models,panel,source(panel,seed,sha))
            if time.perf_counter()-started > freeze["training"]["whole_process_CPU_seconds_per_seed_cap"]:
                raise TimeoutError("R101_ORIGINAL_PROCESS_BUDGET_EXCEEDED")
        adjudication = measures(seed,reports)
        report["evaluation"] = {
            "all_panels":{k:v["metrics"] for k,v in reports.items()},
            "oracle_means":{k:v["oracle_mean"] for k,v in reports.items()},
            "metrics":adjudication,
        }
        report["status"] = (
            "R101_UNKNOWN_OPERATOR_INFERENCE_DEVELOPMENT_SIGNAL_ONLY"
            if adjudication["frozen_all_checks_pass"] else
            "R101_UNSEEN_OPERATOR_INFERENCE_NO_GO")
    finally:
        report["whole_process_seconds"] = time.perf_counter()-started
        if path.exists():
            raise FileExistsError("R101_ORIGINAL_REPLAY_RECEIPT_EXISTS")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return report
