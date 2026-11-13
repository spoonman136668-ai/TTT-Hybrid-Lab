"""F8-R44: frozen-before-implementation, fresh source-neutral gradient/gate study.

All R42/R43 original scientific negatives are immutable. This module only
generates separately selected synthetic factual action consequences, never
consumed R42/R43 panels or external original-source first-use.
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
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r41_history_conditional as r41
from experiments import p85mr_f8r42_history_binding_interaction as r42
from experiments import p85mr_f8r43_factorial_transfer as r43

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r44_gradient_gate_attribution_freeze.json"
FREEZE_BLOB = "049aafcb2846aa95d1e3f1cd5c5e464b9d2c71b1"
SCHEMA = "ttt-hybrid.f8r44-gradient-gate-attribution-freeze.v1"
PARENT = "f0715002a05cfce170ab7352db84e58361c4a09b"
SALT = {"fit": 21810000, "cal": 21820000, "ood": 21830000}
SEEDS = (985201, 985207)
ARMS = tuple((arm, seed) for arm in (
    "r43_additive_native_salience", "multiplicative_native_content_gate",
    "multiplicative_history_only_content_gate", "gru_conventional",
    "ssm_conventional") for seed in SEEDS)
STEPS, BATCH, TRAIN_PAIRS, HELDOUT_PAIRS = 256, 16, 256, 128
SAMPLE_SEED, BOOT_SEED, BOOT_DRAWS = 21849999, 21869999, 10000
PANELS = ("cal", "ood")
TRACE_STEPS = (1, 32, 64, 128, 192, 256)


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() +
                        b"\0" + raw).hexdigest()


def locked():
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R44_PREREG_SOURCE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != SCHEMA or
        f["stage"] != "FROZEN_BEFORE_R44_IMPLEMENTATION_SOURCE_BIRTH_OR_TRAINING" or
        f["lineage"]["parent_sha"] != PARENT or
        f["source"]["salts"] != SALT or
        f["source"]["fit"]["selected_return_factual_episodes"] != 768 or
        f["methods"]["training"]["updates_each"] != STEPS or
        f["methods"]["training"]["pairs_per_update"] != BATCH or
        f["methods"]["training"]["sample_seed"] != SAMPLE_SEED or
        f["frozen_score"]["bootstrap"]["seed"] != BOOT_SEED or
        f["frozen_score"]["bootstrap"]["group_resamples"] != BOOT_DRAWS or
        f["frozen_score"]["independent_scientific_pass"] is not False or
        any(f["authority"][k] for k in (
            "external_original_firstuse", "spent_F7_replay",
            "replay_R43_scored_sources", "new_controller", "new_workflow",
            "new_runner", "paid_API", "GPU_firstuse", "merge",
            "promotion", "deployment"))):
        raise ValueError("F8R44_PREREG_SCOPE_DRIFT")
    if tuple(f["methods"]["seeds"]) != SEEDS:
        raise ValueError("F8R44_PREREG_SEEDS_DRIFT")
    r43.locked()
    return f


def marker(g, panel):
    if panel not in SALT:
        raise ValueError("F8R44_UNREGISTERED_PANEL")
    return 5 + g % 3 if panel == "ood" else 1 + g % 4


def b_history(panel, g, hidden_gate):
    if (panel not in SALT or type(g) is not int or not
        0 <= g < (48 if panel == "fit" else 16) or
        hidden_gate not in (-1, 1)):
        raise ValueError("F8R44_INVALID_HISTORY")
    length = 16 if panel == "ood" else 8
    location = marker(g, panel)
    out = []
    for t in range(length):
        noise = .006 * math.sin((SALT[panel]+g) * .137 + t * .271)
        noise += .004 * (((t+g)%4)-1.5)
        out.append((t%2, round(noise + (.24*hidden_gate if t==location else 0.), 8)))
    return tuple(out)


def world(panel, group, context, sign, gate):
    if (panel not in SALT or type(group) is not int or not
        0 <= group < (48 if panel == "fit" else 16) or
        context not in (0,1) or sign not in (-1,1) or gate not in (-1,1)):
        raise ValueError("F8R44_UNREGISTERED_WORLD")
    return {"panel":panel, "group":group, "public_context":context,
            "q_evaluator_only":sign, "gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2], "goal":1.,
            "b_history":b_history(panel,group,gate)}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R44_UNREGISTERED_PANEL")
    return [world(panel,g,c,q,h) for g in range(48 if panel=="fit" else 16)
            for c in (0,1) for q in (-1,1) for h in (-1,1)]


def _return_following(w, choice):
    # Generative source or evaluator only. Never imported into learner/loss.
    q = w["q_evaluator_only"] * w["gate_evaluator_only"]
    s, c = w["return_state"], w["public_context"]
    return (r41._law(s,choice,c,q,"ood") if w["panel"]=="ood"
            else r33.r32.after(s,choice,c,q))


def _selected_history(w, *, erase_B=False, swap_B=False, erase_fifth=False):
    c, q = w["public_context"], w["q_evaluator_only"]
    rows = [r33.b._record(s,a,r33.r32.after(s,a,c,q),c)
            for s,a in r33.r32.EARLY]
    rows.append(r33.b._record(r33.PROBE,c,
                              r33.r32.after(r33.PROBE,c,c,q),c))
    b = (b_history(w["panel"],w["group"],-w["gate_evaluator_only"])
         if swap_B else w["b_history"])
    rows.extend(r33.b._record(0.,a,v,c) for a,v in b)
    actions = torch.stack([k[0] for k in rows])
    effects = torch.stack([k[1] for k in rows])
    if erase_B:
        # The strongest selected B effect is observable; no generator marker.
        pos = int(torch.argmax(torch.abs(effects[5:,0])))
        effects[5+pos].zero_()
    if erase_fifth:
        effects[4].zero_()
    return actions, effects


def candidates(w, **kwargs):
    a, e = _selected_history(w,**kwargs)
    pairs = []
    for ret in (0,1):
        key = r33.b._record(w["return_state"],ret,w["return_state"],
                              w["public_context"])[0]
        pairs.append((torch.cat((a,key[None])),
                      torch.cat((e,torch.zeros((1,16),dtype=e.dtype)))))
    return torch.stack([x[0] for x in pairs]),torch.stack([x[1] for x in pairs])


def factual_fit():
    locked()
    facts, labels = [], []
    for w in worlds("fit"):
        observed_costs = []
        for ret in (0,1):
            a,e = _selected_history(w)
            following = _return_following(w,ret)
            k = r33.b._record(w["return_state"],ret,following,w["public_context"])
            aa,ee = torch.cat((a,k[0][None])),torch.cat((e,k[1][None]))
            decoded_following = w["return_state"] + r31._physical_delta(ee[-1,0])
            observed_costs.append((decoded_following-w["goal"])**2)
            facts.append({"action":aa,"effect":ee})
        labels.append(min((0,1),key=lambda a:(observed_costs[a],a)))
    if len(facts)!=768 or len(labels)!=384 or labels.count(0)!=192:
        raise ValueError("F8R44_INVALID_FIT_FACTS_OR_BALANCE")
    return facts,labels


def regret(w, choice):
    if choice not in (0,1):
        raise ValueError("F8R44_INVALID_RETURN_ACTION")
    costs=[(_return_following(w,a)-w["goal"])**2 for a in (0,1)]
    return max(0.,costs[choice]-min(costs))


class ContentGate(nn.Module):
    """A learned B-dependent gate, with Native influence but no Native-only bypass."""

    def __init__(self, arm, seed, binder):
        super().__init__()
        if ((arm,seed) not in ARMS or "multiplicative" not in arm):
            raise ValueError("F8R44_INVALID_CONTENT_GATE_ARM")
        self.arm,self.seed=arm,seed
        self.binder=copy.deepcopy(binder)
        for p in self.binder.parameters():p.requires_grad_(False)
        self.binder.eval()
        self.native=(r34.FactualMemory("native_factual_memory",seed)
                     if arm=="multiplicative_native_content_gate" else None)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21880000+seed+ARMS.index((arm,seed)))
            self.attention=nn.Sequential(nn.Linear(32,24),nn.GELU(),nn.Linear(24,1))
            self.value=nn.Sequential(nn.Linear(32,24),nn.GELU(),nn.Linear(24,1))
            self.fusion=nn.Linear(2,1,bias=False)

    def train(self, mode=True):
        super().train(mode);self.binder.eval();return self

    def forward(self,a,e,*,zero_backbone=False,zero_gate=False,
                zero_binder=False,zero_content=False,uniform_pool=False):
        if (a.ndim!=3 or a.shape!=e.shape or a.shape[-1]!=16 or
            a.shape[1] not in (14,22) or a.shape[0]%2):
            raise ValueError("F8R44_INVALID_PAIRED_PREACTION_FACTS")
        obs=e.clone()
        obs[:,-1,:].zero_()
        binder=r38.predict(self.binder,"binder",a,obs).reshape(-1,2)
        b=torch.cat((a[:,5:-1,:],obs[:,5:-1,:]),dim=-1)
        logits=self.attention(b).squeeze(-1)
        weights=(torch.full_like(logits,1./logits.shape[-1])
                 if uniform_pool else logits.softmax(dim=-1))
        summary=(weights.unsqueeze(-1)*b).sum(dim=1)
        content=torch.tanh(self.value(summary).reshape(-1,2).mean(dim=-1))
        if zero_content:
            content=torch.zeros_like(content)
        if self.native is None:
            ns=torch.zeros_like(content)
        else:
            pred=self.native(a,obs,ablate_backbone=zero_backbone).reshape(-1,2)
            ns=pred[:,1]-pred[:,0]
        # Both terms contain learned selected-history content. Native cannot
        # directly select an action when the content gate is absent.
        gate=torch.tanh(self.fusion(torch.stack((content,content*ns),dim=-1)
                                    ).squeeze(-1))
        if zero_gate:gate=torch.zeros_like(gate)
        contrast=(binder[:,1]-binder[:,0])*gate
        if zero_binder:contrast=torch.zeros_like(contrast)
        return torch.stack((-.5*contrast,.5*contrast),dim=-1)


class ConventionalPairGate(nn.Module):
    """Original causal-history GRU/SSM control, independently initialized per seed."""

    def __init__(self,arm,seed,binder):
        super().__init__()
        if (arm,seed) not in ARMS or arm not in ("gru_conventional","ssm_conventional"):
            raise ValueError("F8R44_INVALID_CONVENTIONAL_ARM")
        name=("gru_factual_memory" if arm=="gru_conventional"
              else "ssm_factual_memory")
        self.binder=copy.deepcopy(binder)
        for p in self.binder.parameters():p.requires_grad_(False)
        self.binder.eval()
        # R34 restricts conventional constructors to seed 985201; retain the
        # exact architecture and deterministically initialize seed 985207.
        self.history=r34.FactualMemory(name,SEEDS[0])
        if seed!=SEEDS[0]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(21881000+seed)
                for child in self.history.modules():
                    if child is self.history:continue
                    if (not any(True for _ in child.children()) and
                        callable(getattr(child,"reset_parameters",None))):
                        child.reset_parameters()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21882000+seed+ARMS.index((arm,seed)))
            self.gate=nn.Sequential(nn.Linear(4,24),nn.GELU(),nn.Linear(24,1))

    def train(self,mode=True):
        super().train(mode);self.binder.eval();return self

    def forward(self,a,e,*,zero_backbone=False,zero_gate=False,
                zero_binder=False,zero_content=False,uniform_pool=False):
        if (a.ndim!=3 or a.shape!=e.shape or a.shape[-1]!=16 or
            a.shape[1] not in (14,22) or a.shape[0]%2):
            raise ValueError("F8R44_INVALID_CONVENTIONAL_FACTS")
        obs=e.clone();obs[:,-1,:].zero_()
        base=r38.predict(self.binder,"binder",a,obs).reshape(-1,2)
        pred=self.history(a,obs,ablate_backbone=zero_backbone).reshape(-1,2)
        features=torch.stack((pred[:,0],pred[:,1],pred[:,1]-pred[:,0],
                              .5*(pred[:,0]+pred[:,1])),dim=-1)
        gate=torch.tanh(self.gate(features)).squeeze(-1)
        contrast=(base[:,1]-base[:,0])*gate
        if zero_gate or zero_binder:contrast=torch.zeros_like(contrast)
        return torch.stack((-.5*contrast,.5*contrast),dim=-1)


class OriginalR43Gate(nn.Module):
    def __init__(self,seed,binder):
        super().__init__()
        self.inner=r43.SalienceGate("native_salience",seed,binder)

    def train(self,mode=True):
        super().train(mode);self.inner.binder.eval();return self

    def forward(self,a,e,*,zero_backbone=False,zero_gate=False,
                zero_binder=False,zero_content=False,uniform_pool=False):
        # R43 has no uniform-pool switch. Report this ablation NOT_APPLICABLE
        # rather than misrepresent an unmodified baseline as a negative.
        return self.inner(a,e,zero_backbone=zero_backbone,zero_gate=zero_gate,
                          zero_binder=zero_binder,zero_salience=zero_content)


def model_for(arm,seed,binder):
    if (arm,seed) not in ARMS:
        raise ValueError("F8R44_UNREGISTERED_MODEL")
    if arm=="r43_additive_native_salience":return OriginalR43Gate(seed,binder)
    if "multiplicative" in arm:return ContentGate(arm,seed,binder)
    return ConventionalPairGate(arm,seed,binder)


def _binder(model):
    if isinstance(model,OriginalR43Gate):return model.inner.binder
    return model.binder


def _batch(facts,labels,indices):
    if len(indices)!=BATCH or any(type(i) is not int or
            not 0<=i<TRAIN_PAIRS for i in indices):
        raise ValueError("F8R44_INVALID_FIT_BATCH")
    rows=[facts[2*i+j] for i in indices for j in (0,1)]
    return (torch.stack([x["action"] for x in rows]),
            torch.stack([x["effect"] for x in rows]),
            torch.tensor([labels[i] for i in indices],dtype=torch.long))


def _heldout_accuracy(model,facts,labels):
    model.eval();correct=0
    with torch.no_grad():
        for i in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            rows=[facts[2*j+k] for j in range(i,i+BATCH) for k in (0,1)]
            a=torch.stack([x["action"] for x in rows])
            e=torch.stack([x["effect"] for x in rows])
            y=torch.tensor(labels[i:i+BATCH],dtype=torch.long)
            correct+=int((model(a,e).argmin(-1)==y).sum())
    return correct/HELDOUT_PAIRS


def _gradient_norm(module):
    values=[float(p.grad.detach().square().sum()) for p in module.parameters()
            if p.grad is not None]
    return math.sqrt(sum(values)) if values else 0.


def _gradient_receipt(model,step,loss):
    keys={}
    for name in ("attention","reader","value","fusion","native","history","gate"):
        actual=getattr(model.inner,name,None) if isinstance(model,OriginalR43Gate) \
               else getattr(model,name,None)
        if actual is not None:keys[name]=_gradient_norm(actual)
    return {"update":step,"loss":float(loss.detach()),"gradient_norms":keys,
            "all_finite":all(math.isfinite(x) for x in keys.values())}


def score(model,panel,*,erase_B=False,swap_B=False,erase_fifth=False,
          zero_backbone=False,zero_gate=False,zero_binder=False,
          zero_content=False,uniform_pool=False):
    if panel not in PANELS:raise ValueError("F8R44_UNKNOWN_SCORE_PANEL")
    ww=worlds(panel)
    samples=[candidates(w,erase_B=erase_B,swap_B=swap_B,
                        erase_fifth=erase_fifth) for w in ww]
    a=torch.cat([x[0] for x in samples]);e=torch.cat([x[1] for x in samples])
    model.eval()
    with torch.no_grad():
        pred=model(a,e,zero_backbone=zero_backbone,
                   zero_gate=zero_gate,zero_binder=zero_binder,
                   zero_content=zero_content,uniform_pool=uniform_pool)
    if pred.shape!=(len(ww),2) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R44_INVALID_ACTION_OUTPUT")
    choices=pred.argmin(-1).tolist()
    losses=[regret(w,c) for w,c in zip(ww,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(losses),
            "pergroup_regret":[statistics.fmean(losses[8*g:8*g+8])
                               for g in range(16)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                 for i in range(0,len(choices),2))}


def train_arm(arm,seed,binder,facts,labels,draws):
    model=model_for(arm,seed,binder)
    init_hash=r40._hash(_binder(model))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=.001,weight_decay=.01)
    t0=time.perf_counter();traces=[]
    for step,ids in enumerate(draws,1):
        a,e,y=_batch(facts,labels,ids)
        model.train();opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R44_NONFINITE_TRAIN_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in grads)):
            raise ValueError("F8R44_GRADIENT_INVALID")
        if step in TRACE_STEPS:
            traces.append(_gradient_receipt(model,step,loss))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
        if step in TRACE_STEPS:
            traces[-1]["heldout_fit_accuracy"]=_heldout_accuracy(
                model,facts,labels)
    if r40._hash(_binder(model))!=init_hash:
        raise ValueError("F8R44_BINDER_CUSTODY_DRIFT")
    scored={panel:{"active":score(model,panel)} for panel in PANELS}
    for name,kwargs in (
        ("erase_B",{"erase_B":True}),("swap_B",{"swap_B":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("zero_backbone",{"zero_backbone":True}),
        ("zero_gate",{"zero_gate":True}),
        ("zero_binder",{"zero_binder":True}),
        ("zero_content",{"zero_content":True})
    ):
        scored["ood"][name]=score(model,"ood",**kwargs)
    if isinstance(model,ContentGate):
        scored["ood"]["uniform_pool"]=score(model,"ood",uniform_pool=True)
    else:
        scored["ood"]["uniform_pool"]=None
    cold=model_for(arm,seed,binder)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r40._hash(cold)!=r40._hash(model) or
        any(score(cold,p)["choices"]!=scored[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R44_COLD_RESTORE_FAIL")
    return {"arm":arm,"seed":seed,"optimizer_updates":STEPS,
            "frozen_binder_hash":init_hash,"trace":traces,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                       if p.requires_grad),
            "wall_seconds":time.perf_counter()-t0,
            "panels":scored,"cold_restore":True}


def _paired_bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED);out=[]
    for _ in range(BOOT_DRAWS):
        sample=[rng.randrange(16) for _ in range(16)]
        out.append(statistics.fmean(b[i]-a[i] for i in sample))
    return sorted(out)[int(.025*BOOT_DRAWS)]


def _fit_observed_reference(facts,labels):
    """No hidden gate or marker; calibration selects sign only from FIT observations."""
    def predict(w,polarity):
        a,e=candidates(w)
        fifth=r31._physical_delta(e[0,4,0])
        observed_b=r42._observed_b_peak(e[0])
        return (w["public_context"] if polarity*fifth*observed_b>0
                else 1-w["public_context"])
    fit=worlds("fit")[:TRAIN_PAIRS]
    mistakes={p:sum(predict(w,p)!=y for w,y in zip(fit,labels[:TRAIN_PAIRS]))
              for p in (-1,1)}
    best=min((-1,1),key=lambda p:(mistakes[p],p))
    return {"fit_only_selected_orientation":best,
            "fit_mistakes":mistakes[best],
            "panels":{panel: {"choices":[predict(w,best) for w in worlds(panel)],
                      "mean_regret":statistics.fmean(
                          regret(w,predict(w,best)) for w in worlds(panel))}
                      for panel in PANELS},
            "evaluator_marker_used":False}


def judge(rows,reference,freeze):
    by_key={(r["arm"],r["seed"]):r for r in rows}
    t=freeze["frozen_score"]
    outcomes=[]
    for seed in SEEDS:
        mul=by_key["multiplicative_native_content_gate",seed]
        old=by_key["r43_additive_native_salience",seed]
        active=mul["panels"]["ood"]["active"]
        base=old["panels"]["ood"]["active"]
        alt=[row for row in rows if row["arm"] in (
              "multiplicative_history_only_content_gate",
              "gru_conventional","ssm_conventional")]
        best=min(alt,key=lambda row:row["panels"]["ood"]["active"]["mean_regret"])
        bestreg=best["panels"]["ood"]["active"]["mean_regret"]
        penalty=lambda k:mul["panels"]["ood"][k]["mean_regret"]-active["mean_regret"]
        uniform=penalty("uniform_pool")
        gain=base["mean_regret"]-active["mean_regret"]
        convgain=bestreg-active["mean_regret"]
        lower=_paired_bootstrap_lower(active["pergroup_regret"],
                                       best["panels"]["ood"]["active"]["pergroup_regret"])
        acc=mul["trace"][-1]["heldout_fit_accuracy"]
        passed=bool(
          acc>=t["minimum_heldout_fit_accuracy"] and
          active["mean_regret"]<=t["max_ood_regret_each_seed"] and
          active["gate_pair_flips"]>=t["minimum_ood_gate_flips_each_seed"] and
          gain>=t["minimum_ood_regret_gain_vs_same_seed_r43_additive"] and
          penalty("erase_B")>=t["minimum_ood_regret_penalty_erase_B_each_seed"] and
          penalty("zero_content")>=t["minimum_ood_regret_penalty_zero_content_gate_each_seed"] and
          uniform>=t["minimum_ood_regret_penalty_uniform_pool_each_seed"] and
          penalty("zero_backbone")>=t["minimum_ood_regret_penalty_zero_native_backbone_each_seed"] and
          convgain>=t["minimum_gain_vs_best_fully_trained_conventional_each_seed"] and
          lower>t["bootstrap"]["lower_95_gain_gt"])
        outcomes.append({"seed":seed,"frozen_development_conditions_pass":passed,
                         "heldout_FIT_accuracy":acc,
                         "native_content_regret":active["mean_regret"],
                         "native_content_flips":active["gate_pair_flips"],
                         "same_seed_R43_additive_regret":base["mean_regret"],
                         "strongest_trained_control":best["arm"],
                         "strongest_trained_control_regret":bestreg,
                         "gain_vs_R43":gain,"gain_vs_strongest":convgain,
                         "bootstrap_lower":lower,
                         "penalties":{"erase_B":penalty("erase_B"),
                                      "zero_content":penalty("zero_content"),
                                      "uniform_pool":uniform,
                                      "zero_backbone":penalty("zero_backbone")}})
    return {"both_seeds_frozen_development_pass":all(
                x["frozen_development_conditions_pass"] for x in outcomes),
            "per_seed":outcomes,
            "whole_process_compute_matched":False,
            "independent_scientific_pass":False}


def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    facts,labels=factual_fit()
    fit=r33.fit_cohort();winners=r36.fit_pairs(fit)
    pools=r37.selection_pools(fit,winners)
    g=torch.Generator(device="cpu");g.manual_seed(21249999)
    old_draws=torch.randint(0,len(pools["informative"]),(512,16),
                            generator=g).tolist()
    binders={}
    for seed in SEEDS:
        binders[seed],receipt=r39.replay_original_seed(
            seed,fit,winners,pools,old_draws)
        if not receipt["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R44_REFERENCE_BINDER_MISMATCH")
    generator=torch.Generator(device="cpu");generator.manual_seed(SAMPLE_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(STEPS,BATCH),
                         generator=generator).tolist()
    models=[train_arm(arm,seed,binders[seed],facts,labels,draws)
            for arm,seed in ARMS]
    reference=_fit_observed_reference(facts,labels)
    verdict=judge(models,reference,frozen)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "parent":PARENT,"factual_fit_episodes":len(facts),
            "factual_train_pairs":TRAIN_PAIRS,
            "factual_heldout_pairs":HELDOUT_PAIRS,
            "original_binder_replay_updates":len(SEEDS)*512,
            "new_neural_updates":len(ARMS)*STEPS,
            "models":models,"observational_reference":reference,
            "judgment":verdict,"source_neutral_development_only":True,
            "spent_external_firstuse":False,
            "independent_scientific_pass":False,"deployment":"NONE"}


def retain_ci_receipt(out):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    path=ROOT/"local_runs/p85mr_f8r44_gradient_gate_attribution_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError("F8R44_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R44_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
