"""F8-R47: learn a public cue-key contrast from independent factual consequences.

Fresh source-neutral synthetic DEV, NOT independent scientific acceptance.
The decisive cue role is observable in selected public action keys, not passed
as a generator position, hidden q, gate sign or an unchosen return consequence.
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
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r41_history_conditional as r41

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r47_cue_key_contrast_freeze.json"
FREEZE_BLOB = "f496f6168fbc811f2230240c455e6251da50796a"
SCHEMA = "ttt-hybrid.f8r47-cue-key-contrast-freeze.v1"
PARENT = "13186fcbf2010a89315c6d3e9b72f1d6915dbb64"
SALT = {"fit": 22110000, "cal": 22120000, "ood": 22130000}
SEEDS = (985201, 985207)
ARMS = tuple((arm, seed) for arm in (
    "learned_key_contrast", "uniform_key_control",
    "GRU_factual_control", "Native_augmented_key") for seed in SEEDS)
FIT_GROUPS, FIT_TRAIN_GROUPS, EVAL_GROUPS = 24, 16, 8
TRAIN_PAIRS, HELDOUT_PAIRS, BATCH, UPDATES = 128, 64, 16, 256
DRAW_SEED, BOOT_SEED, BOOT_DRAWS = 22149999, 22169999, 10000
PROBE, CUE, DECOY = .42, .42, .18
PANELS = ("cal", "ood")
CHECKPOINTS = (1, 32, 64, 128, 192, 256)


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() +
                        b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R47_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_IMPLEMENTATION_SOURCE_BIRTH_AND_ANY_RESULT" or
        f["parent"]["sha"] != PARENT or
        f["source"]["salts"] != SALT or
        f["source"]["train_pairs"] != TRAIN_PAIRS or
        f["source"]["fit_physically_selected_returns"] != 384 or
        f["methods"]["train_updates_each"] != UPDATES or
        f["methods"]["sample_seed"] != DRAW_SEED or
        f["frozen_dev_score"]["bootstrap"]["seed"] != BOOT_SEED or
        f["frozen_dev_score"]["independent_scientific_pass"] is not False or
        any(f["authority"][k] for k in (
            "new_controller", "new_workflow", "new_runner", "paid_API",
            "external_original_first_use",
            "spent_prior_original_or_scored_source", "GPU_first_use",
            "merge", "promotion", "deployment"))):
        raise ValueError("F8R47_FROZEN_CONTRACT_DRIFT")
    if tuple(f["methods"]["seeds"]) != SEEDS or tuple(
        (arm["name"], tuple(arm["seeds"])) for arm in f["methods"]["arms"]) != tuple(
        (name, SEEDS) for name, _ in ARMS[::2]):
        # The arm list is generated as pairs of two seeds, not deduplicated.
        expected = tuple((name, SEEDS) for name in dict.fromkeys(
            name for name, _ in ARMS))
        if tuple((arm["name"], tuple(arm["seeds"]))
                 for arm in f["methods"]["arms"]) != expected:
            raise ValueError("F8R47_FROZEN_SEEDS_OR_ARMS_DRIFT")
    return f


def positions(panel, group):
    if (panel not in SALT or type(group) is not int or
        not 0 <= group < (FIT_GROUPS if panel == "fit" else EVAL_GROUPS)):
        raise ValueError("F8R47_UNREGISTERED_SOURCE")
    length = 16 if panel == "ood" else 8
    raw = hashlib.sha256(
        f"F8-R47/{panel}/{group}/{SALT[panel]}".encode()).digest()
    cue = int.from_bytes(raw[:4], "big") % length
    decoy = int.from_bytes(raw[4:8], "big") % (length - 1)
    if decoy >= cue:
        decoy += 1
    return cue, decoy


def world(panel, group, context, sign, gate):
    positions(panel, group)
    if context not in (0, 1) or sign not in (-1, 1) or gate not in (-1, 1):
        raise ValueError("F8R47_INVALID_WORLD_FACTORS")
    return {"panel": panel, "group": group, "public_context": context,
            "q_evaluator_only": sign, "gate_evaluator_only": gate,
            "return_state": (.61, .73)[group % 2], "goal": 1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R47_UNREGISTERED_PANEL")
    return [world(panel, g, c, q, h)
            for g in range(FIT_GROUPS if panel == "fit" else EVAL_GROUPS)
            for c in (0, 1) for q in (-1, 1) for h in (-1, 1)]


def selected_B(w):
    cue, decoy = positions(w["panel"], w["group"])
    gate = w["gate_evaluator_only"]
    n = 16 if w["panel"] == "ood" else 8
    rows = []
    for t in range(n):
        if t == cue:
            state, delta = CUE, .24 * gate
        elif t == decoy:
            state, delta = DECOY, -.24 * gate
        else:
            state = 0.
            delta = round(
                .006 * math.sin((SALT[w["panel"]]+w["group"])*.137
                              + t*.271)
                + .004 * (((t+w["group"])%4)-1.5), 8)
        rows.append((state, t % 2, delta))
    return rows


def _following(w, action):
    s, c = w["return_state"], w["public_context"]
    q = w["q_evaluator_only"] * w["gate_evaluator_only"]
    return (r41._law(s, action, c, q, "ood")
            if w["panel"] == "ood" else r33.r32.after(s, action, c, q))


def selected_history(w, *, erase_B=False, erase_tag=False,
                     erase_fifth=False, swap_B=False):
    c, q = w["public_context"], w["q_evaluator_only"]
    rows = [r33.b._record(s, act, r33.r32.after(s, act, c, q), c)
            for s, act in r33.r32.EARLY]
    rows.append(r33.b._record(PROBE, c,
                r33.r32.after(PROBE, c, c, q), c))
    for tag, act, delta in selected_B(w):
        if swap_B and tag in (CUE, DECOY):
            delta = -delta
        state = 0. if erase_tag and tag == CUE else tag
        action_key, measured = r33.b._record(
            state, act, state + delta, c)
        measured = measured.clone()
        measured[4] = 0.
        rows.append((action_key, measured))
    a = torch.stack([z[0] for z in rows])
    e = torch.stack([z[1] for z in rows])
    if erase_B:
        e[5:].zero_()
    if erase_fifth:
        e[4].zero_()
    return a, e


def candidates(w, **kwargs):
    a, e = selected_history(w, **kwargs)
    aa, ee = [], []
    for choice in (0, 1):
        k = r33.b._record(w["return_state"], choice,
                          w["return_state"], w["public_context"])[0]
        aa.append(torch.cat((a, k[None])))
        ee.append(torch.cat((e, torch.zeros((1,16)))))
    return torch.stack(aa), torch.stack(ee)


def parity(panel):
    n = FIT_GROUPS if panel == "fit" else EVAL_GROUPS
    err = 0.
    for g in range(n):
        a1, e1 = candidates(world(panel, g, 0, 1, 1))
        a2, e2 = candidates(world(panel, g, 0, 1, -1))
        if not torch.equal(a1, a2):
            raise ValueError("F8R47_OBSERVED_KEY_LEAK")
        delta = (e1[:, 5:-1].mean(1)-e2[:, 5:-1].mean(1)).abs().max().item()
        err = max(err, delta)
        if delta>1e-7:
            raise ValueError("F8R47_MEAN_B_GATE_LEAK")
        if torch.equal(e1[:,5:-1],e2[:,5:-1]):
            raise ValueError("F8R47_B_CUE_NOT_OBSERVABLE")
    return {"groups": n, "max_full_B_mean_diff": err}


def factual_fit():
    locked()
    records, labels = [], []
    for w in worlds("fit"):
        cost = []
        for choice in (0, 1):
            a, e = selected_history(w)
            following = _following(w, choice)
            k, measured = r33.b._record(
                w["return_state"], choice, following, w["public_context"])
            aa = torch.cat((a, k[None]))
            ee = torch.cat((e, measured[None]))
            actual_after = (w["return_state"] +
                            r31._physical_delta(ee[-1, 0]))
            cost.append((actual_after - 1.)**2)
            records.append((aa, ee))
        labels.append(min((0, 1), key=lambda idx: (cost[idx], idx)))
    if (len(records) != 384 or len(labels) != 192 or
        labels.count(0) != 96):
        raise ValueError("F8R47_FACTUAL_FIT_SOURCE_INVALID")
    return records, labels


def fit_offset(records):
    if len(records) != 384:
        raise ValueError("F8R47_CALIBRATION_MUST_USE_FULL_FIT")
    # Only the actually selected fifth outcome for TRAIN groups, balanced q.
    observed = [PROBE + r31._physical_delta(records[2*i][1][4, 0])
                for i in range(TRAIN_PAIRS)]
    offset = statistics.fmean(observed)
    if not math.isfinite(offset):
        raise ValueError("F8R47_BAD_FIT_OFFSET")
    return offset


def _observed_components(a, e, offset):
    if (a.ndim != 3 or e.shape != a.shape or
        a.shape[-1] != 16 or a.shape[1] not in (14, 22)
        or a.shape[0] % 2):
        raise ValueError("F8R47_BAD_PAIRED_FACTS")
    obs = e.clone()
    obs[:, -1, :].zero_()
    # Vectorized inverse of the public normalized key. This is not hidden
    # metadata; every learner including GRU receives the identical public tag.
    tag = 2.4 * torch.atanh(
        (a[::2, 5:-1, 1] / a[::2, 5:-1, 0]).clamp(-.98, .98))
    b_delta = obs[::2, 5:-1, 0]
    fifth = PROBE + 2.4*torch.atanh(
        obs[::2, 4, 0].clamp(-.98, .98)) - offset
    context = (3.0*torch.atanh(
        (a[::2, -1, 2]/a[::2, -1, 0]).clamp(-.98, .98))
        ).round().to(torch.long)
    return tag, b_delta, fifth, context, obs


def _choice_cost(context, preference):
    cost0 = torch.where(context == 0, -preference, preference)
    return torch.stack((cost0, -cost0), dim=-1)


def regret(w, choice):
    if choice not in (0, 1):
        raise ValueError("F8R47_BAD_ACTION")
    vals = [(_following(w, a)-1.)**2 for a in (0, 1)]
    return max(0., vals[choice]-min(vals))


def reference_ceiling(offset):
    proof = {}
    for panel in ("fit",)+PANELS:
        ww=worlds(panel)
        errors=0
        for w in ww:
            a, e = candidates(w)
            a, e = a[0], e[0]
            fifth = PROBE + r31._physical_delta(e[4, 0]) - offset
            tagged = [(r31._decode_key(a[t])[0],
                       r31._physical_delta(e[t, 0]))
                      for t in range(5, a.shape[0]-1)]
            cue_delta = max(tagged, key=lambda row: row[0])[1]
            chosen = (w["public_context"] if fifth*cue_delta>0
                      else 1-w["public_context"])
            errors += int(regret(w, chosen) > 1e-10)
        proof[panel] = {"worlds":len(ww),"mistakes":errors}
        if errors:
            raise ValueError("F8R47_UNOBSERVABLE_TASK_STOP_BEFORE_TRAINING")
    return proof


class FactualKeyPolicy(nn.Module):
    def __init__(self, arm, seed, offset):
        super().__init__()
        if (arm, seed) not in ARMS:
            raise ValueError("F8R47_UNREGISTERED_ARM")
        self.arm, self.seed=arm, seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm != "GRU_factual_control":
            self.slope=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
            # Independent seed-dependent gain, same uniformly initialized attention.
            self.gain=nn.Parameter(torch.tensor(3. + (seed-985201)*.01,
                                                dtype=torch.float32))
        if arm == "GRU_factual_control":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22180000 + seed)
                self.gru=nn.GRU(2,24,batch_first=True)
                self.readout=nn.Linear(24,1)
            self.gain=nn.Parameter(torch.tensor(3.,dtype=torch.float32))
        if arm == "Native_augmented_key":
            self.native=r34.FactualMemory("native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.,dtype=torch.float32))

    def forward(self,a,e,*,uniform=False,zero_native=False,
                zero_key=False):
        tag, delta, fifth, context, obs = _observed_components(
            a, e, self.fit_offset)
        if self.arm == "GRU_factual_control":
            x=torch.stack((tag,delta),dim=-1)
            _, hidden=self.gru(x)
            b_sign=torch.tanh(self.readout(hidden[-1]).squeeze(-1))
        else:
            weights=(torch.full_like(tag,1./tag.shape[1])
                     if (uniform or self.arm=="uniform_key_control")
                     else torch.softmax(self.slope*tag,dim=-1))
            b_sign=(weights*delta).sum(-1)
        if zero_key:
            b_sign=torch.zeros_like(b_sign)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*b_sign)
        if self.arm == "Native_augmented_key":
            result=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref = torch.tanh(torch.atanh(pref.clamp(-.98,.98)) +
                 torch.tanh(self.native_gain)*(result[:,1]-result[:,0]))
        return _choice_cost(context,pref)


def batch(records,labels,ids):
    if len(ids)!=BATCH or any(type(i) is not int or
                              not 0<=i<TRAIN_PAIRS for i in ids):
        raise ValueError("F8R47_BAD_FIT_BATCH")
    rows=[records[2*i+j] for i in ids for j in (0,1)]
    return (torch.stack([x[0] for x in rows]),
            torch.stack([x[1] for x in rows]),
            torch.tensor([labels[i] for i in ids],dtype=torch.long))


def heldout_accuracy(model,records,labels):
    model.eval()
    correct=0
    with torch.no_grad():
        for i in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            ids=list(range(i,i+BATCH))
            rows=[records[2*j+k] for j in ids for k in (0,1)]
            a=torch.stack([x[0] for x in rows])
            e=torch.stack([x[1] for x in rows])
            y=torch.tensor([labels[j] for j in ids],dtype=torch.long)
            correct+=int((model(a,e).argmin(-1)==y).sum())
    return correct/HELDOUT_PAIRS


def score(model,panel,*,erase_B=False,erase_tag=False,
          erase_fifth=False,swap_B=False,uniform=False,
          zero_native=False,zero_key=False):
    if panel not in PANELS:
        raise ValueError("F8R47_UNREGISTERED_SCORE")
    ww=worlds(panel)
    data=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                     erase_fifth=erase_fifth,swap_B=swap_B) for w in ww]
    a=torch.cat([x[0] for x in data])
    e=torch.cat([x[1] for x in data])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,uniform=uniform,zero_native=zero_native,
                    zero_key=zero_key)
    if costs.shape!=(len(ww),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R47_BAD_SCORED_ACTION")
    choices=costs.argmin(-1).tolist()
    regrets=[regret(w,c) for w,c in zip(ww,choices)]
    return {"choices":choices,
            "mean_regret":statistics.fmean(regrets),
            "per_group_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(choices),2))}


def train_arm(arm,seed,offset,records,labels,draws):
    model=FactualKeyPolicy(arm,seed,offset)
    params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            rate=.02 if name in ("slope","gain","native_gain") else .001
            params.append({"params":[p],"lr":rate})
    optimizer=torch.optim.AdamW(params,weight_decay=0.)
    traces=[]
    started=time.perf_counter()
    for step,ids in enumerate(draws,1):
        a,e,y=batch(records,labels,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R47_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R47_GRADIENT_INVALID")
        early_grad=(float(model.slope.grad) if
                    hasattr(model,"slope") and model.slope.grad is not None
                    else None)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step in CHECKPOINTS:
            traces.append({"step":step,"loss":float(loss.detach()),
                "heldout_FIT_accuracy":heldout_accuracy(model,records,labels),
                "attention_slope":float(model.slope.detach())
                        if hasattr(model,"slope") else None,
                "slope_gradient":early_grad})
    outputs={p:{"active":score(model,p)} for p in PANELS}
    for name,args in (
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("swap_B",{"swap_B":True}),
        ("uniform",{"uniform":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_key",{"zero_key":True})
    ):
        outputs["ood"][name]=score(model,"ood",**args)
    cold=FactualKeyPolicy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r40._hash(cold)!=r40._hash(model) or
        any(score(cold,p)["choices"] != outputs[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R47_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"updates":UPDATES,
            "trainable_parameters":sum(x.numel() for x in model.parameters()
                                       if x.requires_grad),
            "train_seconds":time.perf_counter()-started,
            "traces":traces,"scores":outputs,"cold_restore":True}


def bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED)
    values=[]
    for _ in range(BOOT_DRAWS):
        sample=[rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        values.append(statistics.fmean(b[i]-a[i] for i in sample))
    return sorted(values)[int(.025*BOOT_DRAWS)]


def verdict(rows,frozen):
    t=frozen["frozen_dev_score"]
    by={(x["arm"],x["seed"]):x for x in rows}
    per_seed=[]
    for seed in SEEDS:
        sel=by["learned_key_contrast",seed]
        uni=by["uniform_key_control",seed]
        scored=sel["scores"]["ood"]["active"]
        u=uni["scores"]["ood"]["active"]
        gain=u["mean_regret"]-scored["mean_regret"]
        penalties={k:sel["scores"]["ood"][k]["mean_regret"]
                -scored["mean_regret"] for k in (
                "erase_B","erase_tag","erase_fifth","swap_B","uniform",
                "zero_native","zero_key")}
        lower=bootstrap_lower(scored["per_group_regret"],
                              u["per_group_regret"])
        valid=bool(
            sel["traces"][-1]["heldout_FIT_accuracy"]>=t["min_heldout_FIT_accuracy"]
            and sel["scores"]["cal"]["active"]["mean_regret"]<=t["max_cal_regret"]
            and scored["mean_regret"]<=t["max_ood_regret"]
            and sel["scores"]["cal"]["active"]["gate_pair_flips"]>=t["min_cal_gate_pair_flips"]
            and scored["gate_pair_flips"]>=t["min_ood_gate_pair_flips"]
            and penalties["erase_B"]>=t["min_ood_regret_penalty_B_erasure"]
            and penalties["erase_tag"]>=t["min_ood_regret_penalty_cue_tag_erasure"]
            and penalties["erase_fifth"]>=t["min_ood_regret_penalty_fifth_erasure"]
            and penalties["uniform"]>=t["min_ood_regret_penalty_uniform"]
            and gain>=t["min_ood_regret_gain_vs_uniform_control"]
            and lower>t["bootstrap"]["lower_95_gain_vs_uniform_gt"]
            and sel["traces"][-1]["attention_slope"]>=t["min_final_learned_attention_slope"])
        per_seed.append({"seed":seed,"frozen_dev_pass":valid,
            "heldout_FIT_accuracy":sel["traces"][-1]["heldout_FIT_accuracy"],
            "cal_regret":sel["scores"]["cal"]["active"]["mean_regret"],
            "ood_regret":scored["mean_regret"],
            "ood_gate_pair_flips":scored["gate_pair_flips"],
            "final_slope":sel["traces"][-1]["attention_slope"],
            "gain_vs_uniform":gain,"penalties":penalties,
            "bootstrap_lower_gain_vs_uniform":lower})
    return {"two_seed_dev_pass":all(x["frozen_dev_pass"] for x in per_seed),
            "per_seed":per_seed,
            "Native_specific_gain_verified":False,
            "strong_conventional_compute_match":False,
            "independent_scientific_pass":False}


def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    par={p:parity(p) for p in ("fit",)+PANELS}
    records,labels=factual_fit()
    offset=fit_offset(records)
    ceiling=reference_ceiling(offset)
    draw=torch.Generator(device="cpu")
    draw.manual_seed(DRAW_SEED)
    indices=torch.randint(0,TRAIN_PAIRS,(UPDATES,BATCH),
                          generator=draw).tolist()
    models=[train_arm(arm,seed,offset,records,labels,indices)
            for arm,seed in ARMS]
    result=verdict(models,frozen)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "source_parity":par,"factual_FIT_returns":len(records),
            "FIT_training_pairs":TRAIN_PAIRS,"FIT_heldout_pairs":HELDOUT_PAIRS,
            "FIT_only_probe_offset":offset,
            "observational_ceiling":ceiling,
            "neural_optimizer_updates":len(ARMS)*UPDATES,
            "models":models,"judgment":result,
            "source_neutral_dev_only":True,
            "external_first_use":False,
            "independent_scientific_pass":False,
            "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    out=ROOT/"local_runs/p85mr_f8r47_cue_key_contrast_dev.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError("F8R47_RECEIPT_ALREADY_EXISTS")
    out.write_text(json.dumps(result,sort_keys=True,indent=2,
                              allow_nan=False)+"\n")


def deny_original_first_use(*args,**kwargs):
    raise PermissionError("F8R47_ORIGINAL_FIRSTUSE_DENIED")
