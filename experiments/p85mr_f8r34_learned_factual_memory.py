"""F8-R34: train neural factual-memory readouts on selected consequences.

Neural backbones are existing P85 native/GRU/SSM architectures; the same 32D
learned recurrent history and 64D value readout is attached to each. No
analytic quartic, explicit informative indicator, causal-sign key, or oracle
alternative-action effect is available to the learner. All supervised labels
are actually selected factual returns from distinct FIT trajectories.
The fitted historical R33 conventional control remains unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn
import torch.nn.functional as F

from experiments import p85mr_f8r29b_neural_information as b
from experiments import p85mr_f8r33_probe_calibration as r33

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r34_learned_factual_memory_freeze.json"
FREEZE_BLOB = "02a7528b5e1a0d9f926d6cf8bb202248490a2e70"
SCHEMA = "ttt-hybrid.f8r34-learned-factual-memory.v1"
SEEDS = (985201, 985203, 985207)
ARMS = tuple([("native_factual_memory", seed) for seed in SEEDS] +
             [(name, SEEDS[0]) for name in
              ("gru_factual_memory", "ssm_factual_memory", "history_only")])
PANELS = ("dev", "ood")
POLICIES = r33.POLICIES
UPDATES = 256
BATCH_PAIRS = 8
FIT_PAIRS = 1152
SAMPLE_SEED = 20779999
DRAW_COUNT = 10000
BOOT_SEED = 20789999
GOAL = 1.


def _blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def locked():
    data = FREEZE.read_bytes()
    if _blob(data) != FREEZE_BLOB:
        raise ValueError("F8R34_FREEZE_BLOB_DRIFT")
    f = json.loads(data)
    d, a, s, auth = f["data"], f["arms"], f["score"], f["authority"]
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_TRAINING_OR_RESULTS" or
        f["classification"] != "SOURCE_NEUTRAL_NEURAL_FACTUAL_CAUSAL_DEV_ONLY" or
        f["parent"]["head"] != "0f5e7381037b8dfdd365be222a5159f2e42fe39f" or
        f["parent"]["r33_freeze_blob"] != r33.FREEZE_BLOB or
        f["parent"]["r33_source_blob"] != _blob(
            (ROOT/"experiments/p85mr_f8r33_probe_calibration.py").read_bytes()) or
        d["fit_episodes_total"] != 3072 or
        d["fit_training_episodes"] != 2304 or
        d["fit_validation_episodes"] != 768 or
        d["fit_selection_seed"] != SAMPLE_SEED or
        d["fit_batch_pairs"] != BATCH_PAIRS or
        a["native_seeds"] != list(SEEDS) or
        a["total_arms_instances"] != len(ARMS) or
        a["train_updates_each"] != UPDATES or
        a["total_updates"] != UPDATES*len(ARMS) or
        a["checkpoints"] != [128,256] or
        a["adapter_history_hidden"] != 32 or
        a["adapter_head_hidden"] != 64 or
        s["panels"] != list(PANELS) or
        s["paired_bootstrap_draws"] != DRAW_COUNT or
        s["bootstrap_seed"] != BOOT_SEED or
        s["native_seed_both_panels_min"] != 2 or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in auth.values())):
        raise ValueError("F8R34_PROTOCOL_OR_AUTHORITY_DRIFT")
    r33.locked()
    return f


def _base_arm(arm):
    if arm == "native_factual_memory":
        return "native_dynamics"
    if arm == "gru_factual_memory":
        return "gru_dynamics"
    if arm == "ssm_factual_memory":
        return "ssm_dynamics"
    if arm == "history_only":
        return None
    raise ValueError("F8R34_UNREGISTERED_ARM")


class FactualMemory(nn.Module):
    """Trainable existing neural backbone + equal raw P80 memory readout."""

    def __init__(self, arm, seed):
        super().__init__()
        base = _base_arm(arm)
        if seed not in SEEDS or (arm != "native_factual_memory" and seed != SEEDS[0]):
            raise ValueError("F8R34_UNREGISTERED_SEED_OR_ARM")
        self.arm = arm
        self.base_arm = base
        self.backbone = b.model_for(base, seed) if base else None
        torch.manual_seed(seed + 12034)
        self.memory = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(32+16+1,64),
            nn.GELU(),
            nn.Linear(64,1),
        )

    def forward(self, actions, effects, *, ablate_backbone=False):
        if (actions.ndim != 3 or actions.shape[-1] != 16 or
            actions.shape[1] not in (14,22) or effects.shape != actions.shape):
            raise ValueError("F8R34_EXPECT_FACTUAL_P80_ACTION_EFFECT")
        # The current return effect at index -1 cannot enter any readout.
        record = torch.cat((actions[:,:-1,:], effects[:,:-1,:]), dim=-1)
        _, memory = self.memory(record)
        context = memory[-1]
        if self.backbone is None or ablate_backbone:
            native_signal = torch.zeros(
                (actions.shape[0],1), dtype=actions.dtype, device=actions.device)
        else:
            pred = b._predict(self.base_arm, self.backbone,
                              {"action":actions,"effect":effects})
            if pred.shape != effects.shape:
                raise ValueError("F8R34_BACKBONE_PREACTION_PREDICTION_SHAPE")
            native_signal = pred[:,-1,0:1]
        return self.readout(torch.cat(
            (context, actions[:,-1,:], native_signal), dim=-1)).squeeze(-1)


def fit_batch(fit, pair_ids):
    if any(type(i) is not int or not 0 <= i < FIT_PAIRS for i in pair_ids):
        raise ValueError("F8R34_INVALID_TRAIN_PAIR")
    indices = [j for i in pair_ids for j in (2*i,2*i+1)]
    rows = [fit[i] for i in indices]
    return (torch.stack([r["action"] for r in rows]),
            torch.stack([r["effect"] for r in rows]),
            torch.stack([r["direct_target"] for r in rows]))


def factual_pair_loss(prediction, actual):
    if (prediction.ndim != 1 or prediction.shape != actual.shape
        or prediction.numel() % 2):
        raise ValueError("F8R34_INVALID_PAIRED_SUPERVISION")
    # Both labels come from separate genuinely selected FIT return records.
    squared = F.mse_loss(prediction, actual)
    difference = (actual[1::2]-actual[::2]).detach()
    signed = torch.sign(difference)
    rank = F.softplus(-signed*(prediction[1::2]-prediction[::2])).mean()
    return squared + 0.1 * rank


def validation_mse(model, fit):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(2304,len(fit),64):
            rows = fit[i:i+64]
            actions = torch.stack([r["action"] for r in rows])
            effects = torch.stack([r["effect"] for r in rows])
            target = torch.stack([r["direct_target"] for r in rows])
            prediction = model(actions,effects)
            errors.extend((prediction-target).square().tolist())
    if len(errors) != 768:
        raise ValueError("F8R34_INVALID_HELDOUT_FIT")
    return statistics.fmean(errors)


def _actions(arm, model, panel, policy, *, erase=False,
             ablate_backbone=False):
    ww = r33.worlds(panel)
    inputs = [r33.candidate_inputs(w,policy,erase) for w in ww]
    actions = torch.cat([a for a,_ in inputs])
    effects = torch.cat([e for _,e in inputs])
    model.eval()
    with torch.no_grad():
        pred = model(actions,effects,ablate_backbone=ablate_backbone)
    if pred.shape != (2*len(ww),) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R34_INVALID_PREACTION_PREDICTION")
    vals = pred.detach().cpu().tolist()
    return [min((0,1),key=lambda a:(vals[2*i+a],a)) for i in range(len(ww))]


def _score(arm,model,panel,policy,*,erase=False,ablate_backbone=False):
    ww = r33.worlds(panel)
    choices = _actions(arm,model,panel,policy,erase=erase,
                       ablate_backbone=ablate_backbone)
    regrets = r33.evaluate_regret_only(ww,choices)
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[4*g:4*g+4])
                               for g in range(r33.GROUPS)],
            "sign_pair_flips":sum(choices[i]!=choices[i+1]
                                   for i in range(0,len(choices),2))}


def score_all(arm,model):
    result={}
    for panel in PANELS:
        pol={p:_score(arm,model,panel,p) for p in POLICIES}
        erasure=_score(arm,model,panel,"active_information",erase=True)
        ablated=_score(arm,model,panel,"active_information",
                       ablate_backbone=True)
        active=pol["active_information"]
        result[panel]={"policies":pol,"erased_active":erasure,
            "backbone_zeroed_active":ablated,
            "erasure_regret_penalty":
                erasure["mean_regret"]-active["mean_regret"],
            "erasure_decision_flips":sum(a!=b for a,b in zip(
                active["choices"],erasure["choices"])),
            "backbone_zeroed_regret_penalty":
                ablated["mean_regret"]-active["mean_regret"]}
    return result


def _hash(model):
    return b.r25.r22.r18.r5.f5.hash_weights(model)


def train_arm(arm,seed,fit,samples):
    model=FactualMemory(arm,seed)
    initial=_hash(model)
    opt=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm=="native_factual_memory" else .001,
        weight_decay=.01)
    trace=[]
    start=time.perf_counter()
    for update,pairs in enumerate(samples,1):
        actions,effects,targets=fit_batch(fit,pairs)
        model.train()
        opt.zero_grad(set_to_none=True)
        output=model(actions,effects)
        loss=factual_pair_loss(output,targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R34_NONFINITE_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0))
                    for g in gradients)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in gradients)):
            raise ValueError("F8R34_INVALID_LEARNED_GRADIENTS")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if update in (128,256):
            trace.append({"updates":update,
                          "last_pair_loss":float(loss.detach()),
                          "validation_mse":validation_mse(model,fit)})
    final=_hash(model)
    if initial==final or len(trace)!=2:
        raise ValueError("F8R34_MODEL_NOT_UPDATED")
    outcomes=score_all(arm,model)
    restored=FactualMemory(arm,seed)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if _hash(restored)!=final or any(
        outcomes[p]["policies"]["active_information"]["choices"] !=
        _actions(arm,restored,p,"active_information")
        for p in PANELS):
        raise ValueError("F8R34_COLD_RESTORE_DRIFT")
    return {"arm":arm,"seed":seed,
            "optimizer_updates":UPDATES,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_weight_hash":initial,"trained_weight_hash":final,
            "validation_trace":trace,
            "training_wall_seconds":time.perf_counter()-start,
            "cold_restore_actions_equal":True,
            "panels":outcomes}


def _paired_lower(current,baseline):
    rng=random.Random(BOOT_SEED)
    gains=[]
    for _ in range(DRAW_COUNT):
        sample=[rng.randrange(r33.GROUPS) for __ in range(r33.GROUPS)]
        gains.append(statistics.fmean(baseline[i]-current[i] for i in sample))
    return sorted(gains)[int(.025*DRAW_COUNT)]


def judge(rows,frozen):
    score=frozen["score"]
    baseline=next(r for r in rows if r["arm"]=="history_only")
    results=[]
    for row in rows:
        by_panel={}
        for panel in PANELS:
            rec=row["panels"][panel]
            pol=rec["policies"]
            active=pol["active_information"]
            fixed=min((pol["fixed_probe0"],pol["fixed_probe1"]),
                      key=lambda x:x["mean_regret"])
            history=baseline["panels"][panel]["policies"]["active_information"]
            gain=fixed["mean_regret"]-active["mean_regret"]
            lower=_paired_lower(active["pergroup_regret"],fixed["pergroup_regret"])
            history_gain=history["mean_regret"]-active["mean_regret"]
            qualified=bool(
                active["mean_regret"] <= score["native_active_regret_max_each"]
                and active["sign_pair_flips"] >= score["native_sign_flip_pairs_min_each"]
                and rec["erasure_regret_penalty"] >=
                    score["native_erase_regret_increase_min_each"]
                and gain >= score["native_active_vs_best_fixed_gain_min_each"]
                and lower > score["native_paired_lower_vs_best_fixed_gt"]
                and history_gain >= score["native_active_vs_history_only_gain_min_each"])
            by_panel[panel]={"active_regret":active["mean_regret"],
                "active_sign_pair_flips":active["sign_pair_flips"],
                "gain_vs_best_fixed":gain,"paired_bootstrap_lower":lower,
                "erasure_regret_penalty":rec["erasure_regret_penalty"],
                "gain_vs_history_only":history_gain,
                "backbone_zeroed_regret_penalty":
                    rec["backbone_zeroed_regret_penalty"],
                "frozen_conditions_pass":qualified}
        results.append({"arm":row["arm"],"seed":row["seed"],
                        "panels":by_panel,
                        "both_panel_pass":all(x["frozen_conditions_pass"]
                            for x in by_panel.values())})
    native=sum(x["both_panel_pass"] for x in results
               if x["arm"]=="native_factual_memory")
    return {"native_seed_both_panel_pass":native,
            "source_neutral_native_development_pass":
                native >= score["native_seed_both_panels_min"],
            "results":results,"independent_scientific_pass":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    if len(fit)!=3072:
        raise ValueError("F8R34_INCORRECT_FACTUAL_FIT_COUNT")
    generator=torch.Generator(device="cpu")
    generator.manual_seed(SAMPLE_SEED)
    samples=torch.randint(0,FIT_PAIRS,(UPDATES,BATCH_PAIRS),
                          generator=generator).tolist()
    all_rows=[train_arm(arm,seed,fit,samples) for arm,seed in ARMS]
    return {"schema":SCHEMA,"classification":freeze["classification"],
            "freeze_blob":FREEZE_BLOB,"frozen_parent":freeze["parent"]["head"],
            "factual_fit_episodes":len(fit),
            "train_factual_episodes":2304,
            "validation_factual_episodes":768,
            "neural_model_instances":len(all_rows),
            "optimizer_updates":len(all_rows)*UPDATES,
            "new_external_effects":0,"spent_F7_replay":False,
            "matched_training_samples_actions_updates":True,
            "capacity_FLOPs_matched":False,
            "independent_scientific_pass":False,
            "deployment":"NONE",
            "models":all_rows,"judgment":judge(all_rows,freeze)}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r34_learned_factual_memory_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R34_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R34_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
