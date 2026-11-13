"""F8-R45 prospective counterbalanced selected-B causal-selection falsifier.

Neither token/transformer dependence nor independent science is inferred.
The paired opposite-gate worlds have identical raw mean public keys and mean
normalized selected effects. Source-neutral DEV only, no authentic first-use.
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
from torch.nn import functional as F

from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r41_history_conditional as r41
from experiments import p85mr_f8r42_history_binding_interaction as r42
from experiments import p85mr_f8r44_gradient_gate_attribution as r44

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r45_balanced_decoy_selection_freeze.json"
FREEZE_BLOB = "2fdc7221adba9becdad752004e301c74131e2bf1"
SCHEMA = "ttt-hybrid.f8r45-balanced-decoy-selection-freeze.v1"
PARENT = "76de3d4950b5cb3125ab7908f482665daf86e691"
SALT = {"fit": 21910000, "cal": 21920000, "ood": 21930000}
SEEDS = (985201, 985207)
ARMS = tuple((a, s) for a in (
    "native_selective", "history_only_selective",
    "native_uniform", "gru_conventional") for s in SEEDS)
PANELS = ("cal", "ood")
UPDATES, BATCH = 256, 16
FIT_PAIRS, HELDOUT_PAIRS = 256, 128
DRAW_SEED, BOOT_SEED = 21949999, 21969999
BOOT_DRAWS = 10000
CUE_TAG, DECOY_TAG = .42, .18


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() +
                        b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R45_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_NEW_R45_IMPLEMENTATION_OR_SOURCE_BIRTH" or
        f["lineage"]["parent_sha"] != PARENT or
        f["source"]["new_salts"] != SALT or
        f["source"]["cohorts"]["fit"]["selected_factual_returns"] != 768 or
        f["source"]["identification"]["cue_observed_state"] != CUE_TAG or
        f["source"]["identification"]["decoy_observed_state"] != DECOY_TAG or
        f["source"]["identification"]["uniform_raw_B_effect_mean_equal_for_opposite_gates"] is not True or
        f["methods"]["training"]["updates_per_arm"] != UPDATES or
        f["methods"]["training"]["sample_seed"] != DRAW_SEED or
        f["prospective_score"]["bootstrap"]["seed"] != BOOT_SEED or
        f["prospective_score"]["independent_scientific_pass"] is not False or
        any(f["authority"][k] for k in (
            "external_original_firstuse", "spent_F7_replay",
            "spent_R44_source_as_new_science", "new_runner",
            "new_workflow", "new_controller", "paid_API", "GPU_firstuse",
            "merge", "promotion", "deployment"))):
        raise ValueError("F8R45_FREEZE_CONTRACT_DRIFT")
    if tuple(f["methods"]["seeds"]) != SEEDS or tuple(
        (a["name"], tuple(a["seeds"])) for a in f["methods"]["arms"]) != tuple(
        (a, SEEDS) for a in ("native_selective",
             "history_only_selective", "native_uniform", "gru_conventional")):
        raise ValueError("F8R45_FROZEN_ARM_DRIFT")
    r44.locked()
    return f


def positions(panel, group):
    """Private generator positions; models receive only action/effect tensors."""
    if panel not in SALT or type(group) is not int or not 0 <= group < (
        48 if panel == "fit" else 16):
        raise ValueError("F8R45_UNREGISTERED_POSITIONS")
    n = 16 if panel == "ood" else 8
    digest = hashlib.sha256(
        f"f8r45/{SALT[panel]}/{group}".encode()).digest()
    cue = int.from_bytes(digest[:4], "big") % n
    decoy = int.from_bytes(digest[4:8], "big") % (n-1)
    if decoy >= cue:
        decoy += 1
    return cue, decoy


def history(panel, group, gate):
    if panel not in SALT or gate not in (-1, 1):
        raise ValueError("F8R45_INVALID_HISTORY")
    cue, decoy = positions(panel, group)
    n = 16 if panel == "ood" else 8
    out = []
    for t in range(n):
        if t == cue:
            tag, delta = CUE_TAG, .24 * gate
        elif t == decoy:
            tag, delta = DECOY_TAG, -.24 * gate
        else:
            tag = 0.
            delta = .006 * math.sin(
                (SALT[panel]+group)*.137+t*.271)
            delta += .004 * (((t+group)%4)-1.5)
            delta = round(delta, 8)
        out.append((tag, t % 2, delta))
    return tuple(out)


def world(panel, group, context, q_sign, gate):
    if (panel not in SALT or type(group) is not int or
        not 0 <= group < (48 if panel == "fit" else 16)
        or context not in (0, 1) or q_sign not in (-1, 1)
        or gate not in (-1, 1)):
        raise ValueError("F8R45_UNREGISTERED_WORLD")
    return {"panel": panel, "group": group, "public_context": context,
            "return_state": (.61, .73)[group % 2], "goal": 1.,
            "q_evaluator_only": q_sign,
            "gate_evaluator_only": gate,
            "b_history": history(panel, group, gate)}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R45_UNREGISTERED_PANEL")
    return [world(panel, g, c, q, h)
            for g in range(48 if panel == "fit" else 16)
            for c in (0, 1) for q in (-1, 1) for h in (-1, 1)]


def _following(w, action):
    q = w["q_evaluator_only"] * w["gate_evaluator_only"]
    s, c = w["return_state"], w["public_context"]
    return (r41._law(s, action, c, q, "ood") if w["panel"] == "ood"
            else r33.r32.after(s, action, c, q))


def _selected_history(w, *, erase_b=False, erase_cue_tag=False,
                      erase_fifth=False, swap_b=False):
    c, q = w["public_context"], w["q_evaluator_only"]
    rows = [r33.b._record(s, a, r33.r32.after(s, a, c, q), c)
            for s, a in r33.r32.EARLY]
    rows.append(r33.b._record(r33.PROBE, c,
                  r33.r32.after(r33.PROBE, c, c, q), c))
    if swap_b:
        b = history(w["panel"], w["group"], -w["gate_evaluator_only"])
    else:
        b = w["b_history"]
    for tag, selected_action, delta in b:
        obs_tag = (0. if erase_cue_tag and tag == CUE_TAG else tag)
        action_key, eff = r33.b._record(
            obs_tag, selected_action, obs_tag + delta, c)
        eff = eff.clone()
        # B has a controlled selected physical change, not a reward.
        # The r33 helper's synthetic reward is a nuisance channel that
        # correlates with tagged state and gate; mask it prospectively.
        eff[4] = 0.
        rows.append((action_key, eff))
    a = torch.stack([row[0] for row in rows])
    e = torch.stack([row[1] for row in rows])
    if erase_b:
        e[5:, :].zero_()
    if erase_fifth:
        e[4].zero_()
    return a, e


def candidates(w, **ablation):
    a, e = _selected_history(w, **ablation)
    pairs = []
    for ret in (0, 1):
        key = r33.b._record(w["return_state"], ret,
                            w["return_state"], w["public_context"])[0]
        pairs.append((torch.cat((a, key[None])),
                      torch.cat((e, torch.zeros((1,16), dtype=e.dtype)))))
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def parity_certificate(panel):
    """A paired information-theoretic falsifier, evaluated on heldout sources."""
    if panel not in SALT:
        raise ValueError("F8R45_UNREGISTERED_PANEL")
    counts, worst = 0, 0.
    for g in range(48 if panel == "fit" else 16):
        w1 = world(panel, g, 0, 1, 1)
        w2 = world(panel, g, 0, 1, -1)
        a1, e1 = candidates(w1)
        a2, e2 = candidates(w2)
        assert torch.equal(a1[:, :-1, :], a2[:, :-1, :])
        mean_delta = (e1[:, 5:-1, :].mean(1) -
                      e2[:, 5:-1, :].mean(1)).abs().max().item()
        worst = max(worst, mean_delta)
        if mean_delta > 1e-7:
            raise ValueError("F8R45_UNIFORM_MEAN_GATE_INFORMATION_LEAK")
        if torch.equal(e1[:, 5:-1, :], e2[:, 5:-1, :]):
            raise ValueError("F8R45_NO_OBSERVABLE_CONDITIONAL_GATE")
        counts += 1
    return {"panel":panel,"matched_groups":counts,
            "max_paired_mean_difference":worst,
            "uniform_mean_gate_predictability":"NONE_FROM_IDENTICAL_PAIRED_MEANS"}


def factual_fit():
    locked()
    facts, labels = [], []
    for w in worlds("fit"):
        chosen_costs = []
        for ret in (0,1):
            a, e = _selected_history(w)
            following = _following(w, ret)
            key, eff = r33.b._record(
                w["return_state"], ret, following, w["public_context"])
            aa, ee = torch.cat((a, key[None])), torch.cat((e, eff[None]))
            following_from_selected = (
                w["return_state"] + r31._physical_delta(ee[-1, 0]))
            chosen_costs.append((following_from_selected-w["goal"])**2)
            facts.append({"action":aa, "effect":ee})
        labels.append(min((0,1), key=lambda k:(chosen_costs[k],k)))
    if len(facts)!=768 or len(labels)!=384 or labels.count(0)!=192:
        raise ValueError("F8R45_INVALID_FACTUAL_SOURCE")
    return facts, labels


def regret(w, choice):
    if choice not in (0,1):
        raise ValueError("F8R45_INVALID_ACTION")
    costs=[(_following(w,a)-w["goal"])**2 for a in (0,1)]
    return max(0.,costs[choice]-min(costs))


class FactualSelector(nn.Module):
    def __init__(self, arm, seed, binder):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R45_INVALID_MODEL")
        self.arm, self.seed = arm, seed
        if arm == "gru_conventional":
            self.inner = r44.ConventionalPairGate(
                "gru_conventional", seed, binder)
        else:
            source_arm = ("multiplicative_history_only_content_gate"
                if arm == "history_only_selective"
                else "multiplicative_native_content_gate")
            self.inner = r44.ContentGate(source_arm, seed, binder)

    def forward(self,a,e,*,erase_native=False,zero_selector=False,
                uniform_pool=False,zero_binder=False):
        if self.arm == "gru_conventional":
            return self.inner(a,e,zero_backbone=erase_native,
                              zero_gate=zero_selector,zero_binder=zero_binder)
        uniform = self.arm == "native_uniform" or uniform_pool
        return self.inner(a,e,zero_backbone=erase_native,
            zero_content=zero_selector,zero_binder=zero_binder,
            uniform_pool=uniform)

    def train(self,mode=True):
        super().train(mode)
        self.inner.binder.eval()
        return self


def _batch(facts, labels, ids):
    if len(ids)!=BATCH or any(type(i) is not int or
        not 0<=i<FIT_PAIRS for i in ids):
        raise ValueError("F8R45_INVALID_PAIR_BATCH")
    batch=[facts[2*i+j] for i in ids for j in (0,1)]
    return (torch.stack([w["action"] for w in batch]),
            torch.stack([w["effect"] for w in batch]),
            torch.tensor([labels[i] for i in ids],dtype=torch.long))


def heldout(model,facts,labels):
    model.eval()
    ok=0
    with torch.no_grad():
        for i in range(FIT_PAIRS,FIT_PAIRS+HELDOUT_PAIRS,BATCH):
            rows=[facts[2*j+k] for j in range(i,i+BATCH) for k in (0,1)]
            a=torch.stack([x["action"] for x in rows])
            e=torch.stack([x["effect"] for x in rows])
            labels_batch=torch.tensor(labels[i:i+BATCH],dtype=torch.long)
            ok+=int((model(a,e).argmin(-1)==labels_batch).sum())
    return ok/HELDOUT_PAIRS


def score(model,panel,*,erase_b=False,erase_cue_tag=False,
          erase_fifth=False,swap_b=False,
          erase_native=False,zero_selector=False,uniform_pool=False,
          zero_binder=False):
    if panel not in PANELS:
        raise ValueError("F8R45_UNKNOWN_SCORE_PANEL")
    ww=worlds(panel)
    source=[candidates(w,erase_b=erase_b,erase_cue_tag=erase_cue_tag,
          erase_fifth=erase_fifth,swap_b=swap_b) for w in ww]
    a=torch.cat([p[0] for p in source])
    e=torch.cat([p[1] for p in source])
    model.eval()
    with torch.no_grad():
        y=model(a,e,erase_native=erase_native,zero_selector=zero_selector,
                uniform_pool=uniform_pool,zero_binder=zero_binder)
    if y.shape!=(len(ww),2) or not bool(torch.isfinite(y).all()):
        raise ValueError("F8R45_BAD_PREACTION_ACTION_SCORE")
    choices=y.argmin(-1).tolist()
    loss=[regret(w,c) for w,c in zip(ww,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(loss),
            "pergroup_regret":[statistics.fmean(loss[8*g:8*g+8])
                                for g in range(16)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                  for i in range(0,len(choices),2))}


def train_arm(arm,seed,binder,facts,labels,draws):
    model=FactualSelector(arm,seed,binder)
    before=r40._hash(model.inner.binder)
    opt=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=.001,weight_decay=.01)
    checkpoints=[]
    t0=time.perf_counter()
    for step,ids in enumerate(draws,1):
        a,e,y=_batch(facts,labels,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R45_INVALID_TRAIN_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if not any(g is not None and bool(torch.any(g!=0)) for g in grads):
            raise ValueError("F8R45_NO_REAL_GRADIENT")
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in grads):
            raise ValueError("F8R45_NONFINITE_GRADIENT")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
        if step in (128,256):
            checkpoints.append({"updates":step,"train_ce":float(loss),
                                "heldout_accuracy":heldout(model,facts,labels)})
    if r40._hash(model.inner.binder)!=before:
        raise ValueError("F8R45_FROZEN_BINDER_CHANGED")
    panels={p:{"active":score(model,p)} for p in PANELS}
    for name,kwargs in (
       ("erase_B",{"erase_b":True}),
       ("erase_cue_tag",{"erase_cue_tag":True}),
       ("swap_B",{"swap_b":True}),
       ("erase_fifth",{"erase_fifth":True}),
       ("uniform_pool",{"uniform_pool":True}),
       ("zero_native",{"erase_native":True}),
       ("zero_selector",{"zero_selector":True}),
       ("zero_binder",{"zero_binder":True})):
        panels["ood"][name]=score(model,"ood",**kwargs)
    cold=FactualSelector(arm,seed,binder)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r40._hash(cold)!=r40._hash(model)
        or any(score(cold,p)["choices"]!=panels[p]["active"]["choices"]
               for p in PANELS)):
        raise ValueError("F8R45_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"optimizer_updates":UPDATES,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                        if p.requires_grad),
            "training_seconds":time.perf_counter()-t0,
            "checkpoints":checkpoints,"frozen_binder_hash":before,
            "panels":panels,"cold_restore":True}


def _bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED)
    values=[]
    for _ in range(BOOT_DRAWS):
        indices=[rng.randrange(16) for i in range(16)]
        values.append(statistics.fmean(
            b[i]-a[i] for i in indices))
    return sorted(values)[int(.025*BOOT_DRAWS)]


def observable_tag_reference():
    """Find selected cue via public action key; generator positions not passed."""
    def choose(w,polarity):
        a,e=candidates(w)
        observed_tags=[r31._decode_key(a[0,5+i])[0]
                       for i in range(a.shape[1]-6)]
        cue=int(max(range(len(observed_tags)),key=observed_tags.__getitem__))
        value=r31._physical_delta(e[0,5+cue,0])
        fifth=r31._physical_delta(e[0,4,0])
        return (w["public_context"] if polarity*fifth*value>0
                else 1-w["public_context"])
    fit_worlds=worlds("fit")
    facts, labels=factual_fit()
    mistakes={polarity:sum(choose(w,polarity)!=label
             for w,label in zip(fit_worlds[:FIT_PAIRS],
                                 labels[:FIT_PAIRS]))
             for polarity in (-1,1)}
    polarity=min((-1,1),key=lambda k:(mistakes[k],k))
    result={}
    for panel in PANELS:
        ww=worlds(panel)
        choices=[choose(w,polarity) for w in ww]
        result[panel]={"mean_regret":statistics.fmean(
            regret(w,c) for w,c in zip(ww,choices)),
            "choices":choices}
    return {"fit_selected_orientation":polarity,
            "fit_train_mistakes":mistakes[polarity],
            "panels":result,"selected_observable_tag_only":True}


def judge(rows,freeze):
    t=freeze["prospective_score"]
    by={(r["arm"],r["seed"]):r for r in rows}
    results=[]
    for seed in SEEDS:
        native=by["native_selective",seed]
        uniform=by["native_uniform",seed]
        strong=[r for r in rows if r["arm"] in (
            "history_only_selective","gru_conventional")]
        strongest=min(strong,key=lambda m:m["panels"]["ood"]["active"]["mean_regret"])
        active=native["panels"]["ood"]["active"]
        ref=strongest["panels"]["ood"]["active"]
        convgain=ref["mean_regret"]-active["mean_regret"]
        penalty=lambda name:native["panels"]["ood"][name]["mean_regret"]-active["mean_regret"]
        lower=_bootstrap_lower(active["pergroup_regret"],ref["pergroup_regret"])
        heldout=native["checkpoints"][-1]["heldout_accuracy"]
        cal=native["panels"]["cal"]["active"]
        fit_gate=bool(
            heldout>=t["minimum_heldout_fit_accuracy"] and
            cal["mean_regret"]<=t["maximum_cal_regret_each"] and
            active["mean_regret"]<=t["maximum_ood_regret_each"] and
            cal["gate_pair_flips"]>=t["minimum_cal_gate_pair_flips"] and
            active["gate_pair_flips"]>=t["minimum_ood_gate_pair_flips"] and
            penalty("uniform_pool")>=t["minimum_regret_penalty_uniform_pool"] and
            penalty("erase_B")>=t["minimum_regret_penalty_B_erasure"] and
            penalty("erase_cue_tag")>=t["minimum_regret_penalty_cue_tag_erasure"] and
            penalty("erase_fifth")>=t["minimum_regret_penalty_fifth_erasure"] and
            uniform["panels"]["ood"]["active"]["mean_regret"]-active["mean_regret"]
                >=t["minimum_advantage_over_native_uniform"] and
            convgain>=t["minimum_advantage_over_strongest_fully_trained_conventional"] and
            penalty("zero_native")>=t["native_backbone_ablation_penalty"] and
            lower>t["bootstrap"]["lower_95_gain_vs_strongest_gt"])
        results.append({
            "seed":seed,"frozen_development_pass":fit_gate,
            "heldout_fit":heldout,
            "cal_regret":cal["mean_regret"],
            "ood_regret":active["mean_regret"],
            "ood_gate_flips":active["gate_pair_flips"],
            "uniform_pool_penalty":penalty("uniform_pool"),
            "erase_cue_tag_penalty":penalty("erase_cue_tag"),
            "erase_B_penalty":penalty("erase_B"),
            "erase_fifth_penalty":penalty("erase_fifth"),
            "zero_native_penalty":penalty("zero_native"),
            "strongest_trained_control":strongest["arm"],
            "strongest_control_regret":ref["mean_regret"],
            "gain_vs_control":convgain,
            "bootstrap_lower":lower})
    return {"frozen_two_seed_development_pass":all(
                x["frozen_development_pass"] for x in results),
            "seeds":results,"whole_process_compute_matched":False,
            "independent_scientific_pass":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    certified={p:parity_certificate(p) for p in ("fit",)+PANELS}
    facts,labels=factual_fit()
    orig=r33.fit_cohort()
    winners=r36.fit_pairs(orig)
    pools=r37.selection_pools(orig,winners)
    generator=torch.Generator(device="cpu")
    generator.manual_seed(21249999)
    original_draws=torch.randint(0,len(pools["informative"]),
        (512,16),generator=generator).tolist()
    binder={}
    for seed in SEEDS:
        binder[seed],receipt=r39.replay_original_seed(
            seed,orig,winners,pools,original_draws)
        if not receipt["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R45_REFERENCE_BINDER_REPLAY_DRIFT")
    sampler=torch.Generator(device="cpu");sampler.manual_seed(DRAW_SEED)
    draws=torch.randint(0,FIT_PAIRS,(UPDATES,BATCH),
                         generator=sampler).tolist()
    rows=[train_arm(a,s,binder[s],facts,labels,draws) for a,s in ARMS]
    reference=observable_tag_reference()
    judgment=judge(rows,freeze)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "parent":PARENT,"source_neutral_dev_only":True,
            "factual_fit_episodes":len(facts),
            "train_pairs":FIT_PAIRS,"heldout_pairs":HELDOUT_PAIRS,
            "reference_binder_replay_updates":len(SEEDS)*512,
            "new_optimizer_updates":len(ARMS)*UPDATES,
            "source_parity":certified,
            "models":rows,"observable_tag_reference":reference,
            "judgment":judgment,
            "independent_scientific_pass":False,
            "external_original_firstuse":False,
            "deployment":"NONE"}


def retain_ci_receipt(x):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    p=ROOT/"local_runs/p85mr_f8r45_balanced_decoy_selection_dev.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():raise FileExistsError("F8R45_RECEIPT_EXISTS")
    p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R45_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
