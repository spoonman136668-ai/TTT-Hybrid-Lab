"""F8-R40 frozen-binder/native residual: preserve successful causal binding.

Read-only baseline: exact R39 replay of R38 binder, no new external source.
Native residual and naive-sum comparison use the same independently factual
selected-return choice pairs and 256 Native-only optimizer steps per arm.
Zero-regret binder panels cannot establish further *Native* improvement.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r40_native_fusion_freeze.json"
FREEZE_BLOB="3c69015fae28a4dc7decfe52a16c323c26c3b193"
SCHEMA="ttt-hybrid.f8r40-frozen-binder-native-fusion-isolation.v1"
SEEDS=(985201,985207)
ARMS=("naive_sum","zero_gated_residual")
PANELS=("dev","ood","synthetic_cubic_transfer",
        "oscillatory_even","nonlinear_odd")
POLICIES=("active_information","fixed_probe0","fixed_probe1","seeded_random")
STEPS=256
BATCH=16


def _blob(d):
    return hashlib.sha1(b"blob "+str(len(d)).encode()+b"\0"+d).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R40_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    a,o,e,j=f["fusion_arms"],f["training"],f["evaluation"],f["frozen_judgment"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS" or
        f["parent"]["sha"]!="7fb94ecc43bf9ee0ab98f2647f35c86e15096386" or
        f["parent"]["r39_source_blob"]!=_blob(
           (ROOT/"experiments/p85mr_f8r39_binding_necessity.py").read_bytes()) or
        f["parent"]["r38_source_blob"]!=_blob(
           (ROOT/"experiments/p85mr_f8r38_learned_binding.py").read_bytes()) or
        a["arms"]!=list(ARMS) or a["seeds"]!=list(SEEDS) or
        a["model_instances"]!=len(ARMS)*len(SEEDS) or
        o["native_residual_updates_each"]!=STEPS or
        o["total_native_residual_optimizer_updates"]!=1024 or
        o["total_binder_pretraining_replay_updates"]!=1024 or
        o["train_pairs_per_step"]!=BATCH or
        o["sampler_seed"]!=21449999 or
        o["checkpoint_steps"]!=[128,256] or
        e["panels"]!=list(PANELS) or e["policies"]!=list(POLICIES) or
        e["worlds_each"]!=64 or
        j["nonregression_pass_requires_seeds"]!=2 or
        j["each_seed_panel_active_regret_max"]!=0 or
        j["each_seed_panel_hidden_sign_flips_min"]!=32 or
        j["independent_scientific_pass"] is not False):
        raise ValueError("F8R40_FROZEN_PROTOCOL_DRIFT")
    forbidden=("external_first_use","spent_F7_replay","GPU_firstuse",
               "paid_API","new_workflow","new_runner","new_scheduler",
               "new_controller","merge","promotion","deployment")
    if any(f["source_and_authority"].get(k) is not False for k in forbidden):
        raise PermissionError("F8R40_AUTHORITY_DRIFT")
    r39.locked()
    return f


class FrozenBinderResidual(nn.Module):
    def __init__(self,seed,mode,binder):
        super().__init__()
        if seed not in SEEDS or mode not in ARMS:
            raise ValueError("F8R40_UNREGISTERED_FUSION")
        self.mode=mode
        self.binder=copy.deepcopy(binder)
        for p in self.binder.parameters():
            p.requires_grad_(False)
        self.binder.eval()
        self.native=r34.FactualMemory("native_factual_memory",seed)
        self.alpha=nn.Parameter(torch.tensor(0.)) if mode=="zero_gated_residual" else None

    def train(self,mode=True):
        super().train(mode)
        self.binder.eval()
        return self

    def forward(self,actions,effects,*,zero_native=False,zero_backbone=False):
        if (actions.ndim!=3 or effects.shape!=actions.shape or
            actions.shape[-1]!=16 or actions.shape[1] not in (14,22)):
            raise ValueError("F8R40_INVALID_PREACTION_FACTS")
        obs=effects.clone()
        obs[:,-1,:].zero_()
        baseline=r38.predict(self.binder,"binder",actions,obs)
        if zero_native:
            return baseline
        native=self.native(actions,obs,ablate_backbone=zero_backbone)
        scale=1. if self.mode=="naive_sum" else .1*torch.tanh(self.alpha)
        return baseline+scale*native


def _hash(m):
    return r38._hash(m)


def _expected_binder(seed,fit,winners,pools,original_samples):
    model,receipt=r39.replay_original_seed(seed,fit,winners,pools,original_samples)
    if not receipt["replayed_R38_decision_fingerprints_match"]:
        raise ValueError("F8R40_BASELINE_CUSTODY_FAILED")
    return model,receipt


def _candidate_new(world,policy,erase=False):
    if policy not in POLICIES:
        raise ValueError("F8R40_UNREGISTERED_PROBE_POLICY")
    if policy=="active_information":
        return r39.candidate_inputs(world,erase=erase)
    context=world["public_context"]
    if policy=="fixed_probe0":
        probe=0
    elif policy=="fixed_probe1":
        probe=1
    else:
        key=(f'f8r40/{world["panel"]}/{world["group"]}/'
             f'{context}').encode()
        probe=hashlib.sha256(key).digest()[0]&1
    q=world["q_evaluator_only"]
    panel=world["panel"]
    selected=[r33.b._record(s,a,r39._generator_law(s,a,context,q,panel),
                             context) for s,a in r33.r32.EARLY]
    selected.append(r33.b._record(r33.PROBE,probe,
           r39._generator_law(r33.PROBE,probe,context,q,panel),context))
    selected.extend(r33.b._record(0.,a,v,context)
                    for a,v in world["b_history"])
    acts=torch.stack([v[0] for v in selected])
    eff=torch.stack([v[1] for v in selected])
    if erase:
        eff[4,:].zero_()
    rows=[]
    for ret in (0,1):
        key=r33.b._record(world["return_state"],ret,
                          world["return_state"],context)[0]
        rows.append((torch.cat((acts,key.unsqueeze(0))),
                     torch.cat((eff,torch.zeros((1,16))))))
    return torch.stack([a for a,e in rows]),torch.stack([e for a,e in rows])


def score(model,panel,policy,*,erase=False,zero_native=False,
          zero_backbone=False):
    if panel not in PANELS or policy not in POLICIES:
        raise ValueError("F8R40_UNREGISTERED_SCORE")
    if panel in r38.PANELS:
        worlds=r38.worlds(panel)
        observations=[r38.candidate_inputs(w,policy,erase=erase)
                      for w in worlds]
    else:
        worlds=r39.worlds(panel)
        observations=[_candidate_new(w,policy,erase=erase) for w in worlds]
    acts=torch.cat([p[0] for p in observations])
    effects=torch.cat([p[1] for p in observations])
    model.eval()
    with torch.no_grad():
        pred=model(acts,effects,zero_native=zero_native,
                   zero_backbone=zero_backbone)
    if pred.shape!=(128,) or not bool(torch.isfinite(pred).all()):
        raise ValueError("F8R40_BAD_SCORE_LOGITS")
    values=pred.detach().tolist()
    choices=[min((0,1),key=lambda a:(values[2*i+a],a))
             for i in range(64)]
    if panel in r38.PANELS:
        regret=r38.regret_evaluator_only(worlds,choices)
    else:
        regret=[]
        for w,ch in zip(worlds,choices):
            values=[(r39._generator_law(w["return_state"],a,
                w["public_context"],w["q_evaluator_only"],panel)-1.)**2
                    for a in (0,1)]
            regret.append(max(0.,values[ch]-min(values)))
    return {"mean_regret":statistics.fmean(regret),
            "choices":choices,"sign_pair_flips":
                sum(choices[i]!=choices[i+1] for i in range(0,64,2)),
            "pergroup_regret":[statistics.fmean(regret[4*g:4*g+4])
                               for g in range(16)]}


def panel_results(model,panel):
    policies={p:score(model,panel,p) for p in POLICIES}
    erased=score(model,panel,"active_information",erase=True)
    native_off=score(model,panel,"active_information",zero_native=True)
    backbone_off=score(model,panel,"active_information",zero_backbone=True)
    normal=policies["active_information"]
    return {"policies":policies,"erased":erased,
            "zero_native":native_off,"zero_backbone":backbone_off,
            "erasure_regret_penalty":
                erased["mean_regret"]-normal["mean_regret"],
            "zero_native_identity":native_off["choices"]==normal["choices"],
            "native_backbone_regret_penalty":
                backbone_off["mean_regret"]-normal["mean_regret"]}


def _validation(model,fit,winners,ids):
    model.eval()
    correct=0
    mse=[]
    with torch.no_grad():
        for p in range(0,len(ids),32):
            chunk=ids[p:p+32]
            rows=[fit[2*i+j] for i in chunk for j in (0,1)]
            a=torch.stack([x["action"] for x in rows])
            e=torch.stack([x["effect"] for x in rows])
            expected=torch.tensor([winners[i] for i in chunk],
                                  dtype=torch.long)
            logits=model(a,e).reshape(len(chunk),2)
            correct+=int((logits.argmin(dim=1)==expected).sum())
            mse.append((len(chunk),float(F.cross_entropy(-logits,expected))))
    return {"pairs":len(ids),"accuracy":correct/len(ids),
            "CE":sum(n*loss for n,loss in mse)/len(ids)}


def train_arm(model,fit,winners,pools,samples):
    original_binder=_hash(model.binder)
    initial_native=_hash(model.native)
    original_model=_hash(model)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=.001,
        weight_decay=.01)
    t0=time.perf_counter()
    checkpoints=[]
    for step,indices in enumerate(samples,1):
        ids=[pools["informative"][i] for i in indices]
        a,e,y=r38.selected_batch(fit,winners,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits=model(a,e).reshape(BATCH,2)
        loss=F.cross_entropy(-logits,y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R40_NONFINITE_TRAINING_LOSS")
        loss.backward()
        available=[p.grad for p in model.parameters() if p.requires_grad]
        if (not any(v is not None and bool(torch.any(v!=0))
                    for v in available) or any(
                    v is not None and not bool(torch.isfinite(v).all())
                    for v in available)):
            raise ValueError("F8R40_INVALID_NATIVE_GRADIENT")
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step in (128,256):
            checkpoints.append({"updates":step,"train_CE":float(loss.detach()),
                "FIT_heldout_informative":_validation(
                    model,fit,winners,pools["heldout_informative"])})
    if _hash(model.binder)!=original_binder:
        raise ValueError("F8R40_BINDER_WEIGHTS_MUTATED")
    after_native=_hash(model.native)
    if after_native==initial_native or len(checkpoints)!=2:
        raise ValueError("F8R40_NATIVE_DID_NOT_TRAIN")
    return {"optimizer_updates":STEPS,
        "frozen_binder_sha256":original_binder,
        "frozen_binder_unchanged":True,
        "initial_native_sha256":initial_native,
        "final_native_sha256":after_native,
        "initial_full_sha256":original_model,
        "final_full_sha256":_hash(model),
        "train_wall_seconds":time.perf_counter()-t0,
        "checkpoints":checkpoints}


def judge(results,freeze):
    rules=freeze["frozen_judgment"]
    output=[]
    for seed in SEEDS:
        r=next(x for x in results
               if x["seed"]==seed and x["arm"]=="zero_gated_residual")
        panels={}
        for p in PANELS:
            z=r["panels"][p]
            active=z["policies"]["active_information"]
            ok=(active["mean_regret"]<=rules["each_seed_panel_active_regret_max"]
                and active["sign_pair_flips"]>=
                    rules["each_seed_panel_hidden_sign_flips_min"]
                and z["erasure_regret_penalty"]>=
                    rules["each_seed_panel_erased_measurement_regret_penalty_min"]
                and z["zero_native_identity"] and
                z["binder_action_identity"])
            panels[p]={"regret":active["mean_regret"],
                "sign_pair_flips":active["sign_pair_flips"],
                "erasure_regret_penalty":z["erasure_regret_penalty"],
                "binder_action_identity":z["binder_action_identity"],
                "pass":bool(ok)}
        output.append({"seed":seed,"panels":panels,
                       "all_panels_pass":all(v["pass"] for v in panels.values())})
    wins=sum(x["all_panels_pass"] for x in output)
    return {"zero_gated_preservation_winning_seeds":wins,
            "source_neutral_nonregression_development_pass":
                wins>=rules["nonregression_pass_requires_seeds"],
            "native_capability_gain_unverifiable_at_zero_binder_regret":True,
            "independent_scientific_pass":False,"per_seed":output}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r33.fit_cohort()
    winners=r36.fit_pairs(fit)
    pools=r37.selection_pools(fit,winners)
    original=torch.Generator(device="cpu")
    original.manual_seed(21249999)
    pretrain=torch.randint(0,len(pools["informative"]),
                           (512,BATCH),generator=original).tolist()
    rng=torch.Generator(device="cpu")
    rng.manual_seed(21449999)
    fine=torch.randint(0,len(pools["informative"]),(STEPS,BATCH),
                       generator=rng).tolist()
    rows=[]
    for seed in SEEDS:
        binder,custody=_expected_binder(seed,fit,winners,pools,pretrain)
        if _hash(binder)!=custody["trained_weight_hash"]:
            raise ValueError("F8R40_PRETRAIN_CUSTODY_HASH")
        left=FrozenBinderResidual(seed,"naive_sum",binder)
        right=FrozenBinderResidual(seed,"zero_gated_residual",binder)
        if _hash(left.native)!=_hash(right.native):
            raise ValueError("F8R40_NATIVE_INITIAL_WEIGHT_MISMATCH")
        for panel in PANELS:
            full=score(right,panel,"active_information")
            base=score(right,panel,"active_information",zero_native=True)
            if full["choices"]!=base["choices"] or                _hash(right.binder)!=_hash(binder):
                raise ValueError("F8R40_ZERO_GATE_NOT_EXACT_BASELINE")
        for model in (left,right):
            train=train_arm(model,fit,winners,pools,fine)
            panel={}
            for p in PANELS:
                cell=panel_results(model,p)
                base=cell["zero_native"]
                cell["binder_action_identity"]=(cell["policies"]
                  ["active_information"]["choices"]==base["choices"])
                panel[p]=cell
            restored=FrozenBinderResidual(seed,model.mode,binder)
            restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
            if _hash(restored)!=_hash(model) or any(
                score(restored,p,"active_information")["choices"]!=
                panel[p]["policies"]["active_information"]["choices"]
                for p in PANELS):
                raise ValueError("F8R40_COLD_RESTORE_DRIFT")
            rows.append({**train,"seed":seed,"arm":model.mode,
                "parameters_native":sum(p.numel()
                    for p in model.native.parameters()),
                "cold_restore_equal":True,"panels":panel})
    return {"schema":SCHEMA,"classification":freeze["classification"],
        "freeze_blob":FREEZE_BLOB,"frozen_parent":freeze["parent"]["sha"],
        "pretrain_replay_seed_count":len(SEEDS),
        "pretrain_optimizer_updates":512*len(SEEDS),
        "native_arms_trained":len(rows),"native_optimizer_updates":STEPS*len(rows),
        "models":rows,"judgment":judge(rows,freeze),
        "native_improvement_proven":False,"independent_scientific_pass":False,
        "external_firstuse":False,"spent_F7_replay":False,
        "deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r40_native_fusion_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R40_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R40_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
