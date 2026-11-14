"""F8-R58: frozen five-bridge selected-intervention causal task ADMISSION.

32 sign-parity causal laws, 24 FIT/CAL and 8 independent held-out OOD laws.
Only five bridges' actually selected action consequences enter either model;
the sixth (return) action outcomes are reserved for training labels on FIT
and evaluator scores elsewhere. The task source and law split were frozen
before generating any train/evaluation data or training baselines.

This task-admission run DOES NOT claim independent learning or Hybrid Native
benefit. If qualified conventional baselines lack FIT competence or the
simplest controls explain OOD, reject rather than tune spent law cohorts.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r58_five_bridge_task_admission_freeze.json"
FREEZE_BLOB="b6549f85a8f54c69ccd76efb2614be2f04106fc8"
SCHEMA="ttt-hybrid.f8r58-five-bridge-causal-task-admission-freeze.v1"
PARENT="e499ea5fda031e30fec68a6deb5cdd19a83066e3"
LAWS=tuple(itertools.product((-1,1),repeat=5))
OOD_LAWS=(4,5,6,8,13,20,25,30)
FIT_LAWS=tuple(i for i in range(32) if i not in OOD_LAWS)
SEEDS=(985201,985207)
ARM_NAMES=("mlp_selected_terminal_proxy","gru_selected_terminal_proxy")
ARMS=tuple((name,seed) for name in ARM_NAMES for seed in SEEDS)
SALT={"FIT_train":23210000,"FIT_heldout":23215000,"CAL":23220000,"OOD":23230000}
PER_LAW={"FIT_train":24,"FIT_heldout":8,"CAL":8,"OOD":16}
AMP={"FIT_train":.16,"FIT_heldout":.16,"CAL":.12,"OOD":.08}
PROXY={"FIT_train":.24,"FIT_heldout":.30,"CAL":.48,"OOD":.72}
UPDATES,BATCH,DRAW_SEED=4096,32,23249999
CHECKPOINTS=(1,512,1024,2048,3072,4096)
PANELS=("FIT_train","FIT_heldout","CAL","OOD")


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R58_PROSPECTIVE_FREEZE_DRIFT")
    f=json.loads(raw)
    s,t,g=f["source"],f["trained_controls"],f["admission"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_BEFORE_F58_SOURCE_MODEL_OR_SCORED_RESULTS"
        or f["parent"]["r57_sha"]!=PARENT
        or f["parent"]["r57_freeze_blob"]!="caa43052c4cf74a6cae16b0e2b9033d4fa828eb1"
        or s["heldout_OOD_law_indices"]!=list(OOD_LAWS)
        or s["FIT_law_indices"]!=list(FIT_LAWS)
        or s["source_salting"]!=SALT
        or s["cases_per_law"]!=PER_LAW
        or s["terminal_action_contrast_magnitude"]!=AMP
        or s["proxy_action_contrast_magnitude"]!=PROXY
        or s["number_observed_bridges"]!=5
        or s["observed_records_per_case"]!=20
        or t["original_seeds"]!=list(SEEDS)
        or t["arm_names"]!=list(ARM_NAMES)
        or t["training_updates_each"]!=UPDATES
        or t["batch_size"]!=BATCH
        or t["training_seed_base"]!=DRAW_SEED
        or t["total_optimizer_attempts"]!=len(ARMS)*UPDATES
        or g["independent_scientific_pass"] is not False
        or any(f["authority"].values())):
        raise ValueError("F8R58_PROTOCOL_OR_AUTHORITY_DRIFT")
    return f


def target_sign(law_index):
    if not (type(law_index) is int and 0<=law_index<32):
        raise ValueError("F8R58_INVALID_LAW")
    return math.prod(LAWS[law_index])


def source_only_heuristics(freeze):
    """All proper sign-parity products (and inverse), plus majority/inverse."""
    observed=[LAWS[i] for i in OOD_LAWS]
    truth=[target_sign(i) for i in OOD_LAWS]
    outcome={}
    for k in range(1,5):
        for subset in itertools.combinations(range(5),k):
            guess=[math.prod(row[j] for j in subset) for row in observed]
            hit=sum(x==y for x,y in zip(guess,truth))/len(truth)
            outcome["degree"+str(k)+"/"+"".join(map(str,subset))]={
                "accuracy":hit,"inverted_accuracy":1-hit}
    majority=[1 if sum(row)>0 else -1 for row in observed]
    maj=sum(x==y for x,y in zip(majority,truth))/len(truth)
    outcome["majority"]={"accuracy":maj,"inverted_accuracy":1-maj}
    outcome["constant_sign"]={"accuracy":truth.count(1)/len(truth),
                              "inverted_accuracy":truth.count(-1)/len(truth)}
    max_accuracy=max(max(v.values()) for v in outcome.values())
    balanced=truth.count(1)==truth.count(-1)==4
    all_hold=max_accuracy<=freeze["admission"]["max_any_single_or_higher_degree_shortcut_OOD_accuracy"] and balanced
    # If this source-only audit ever fails, STOP BEFORE FIT training.
    if not all_hold:
        raise ValueError("F8R58_TASK_INADMISSIBLE_SIMPLE_SHORTCUT")
    return {"all_30_proper_parities_enumerated":30,
            "max_shortcut_accuracy":max_accuracy,
            "OOD_law_indices":list(OOD_LAWS),
            "OOD_target_signs":truth,
            "balanced_four_each":balanced,
            "majority_accuracy":maj,"all_shortcuts":outcome,
            "weak_source_shortcuts_admitted":all_hold}


def hashed_sign(*parts):
    raw="/".join(str(p) for p in parts).encode()
    return 1 if hashlib.sha256(raw).digest()[0]&1 else -1


def selected_case(panel,law_index,case_idx):
    """Generate only actually selected bridge action outcomes.

    For every bridge j=0..4 and both actions, reset that bridge's public
    state and actually select action, then receive terminal and proxy
    consequences. The future sixth bridge action is NOT in any feature.
    FIT return labels derive from separately selected reset trials only.
    """
    if panel not in PANELS or type(case_idx) is not int or not 0<=case_idx<PER_LAW[panel]:
        raise ValueError("F8R58_INVALID_PANEL_OR_CASE")
    if law_index not in (OOD_LAWS if panel=="OOD" else FIT_LAWS):
        raise ValueError("F8R58_LAW_SPLIT_CONTAMINATION")
    law=LAWS[law_index]
    records=[]
    features=[]
    for bridge,sign in enumerate(law):
        nuisance=(hashlib.sha256(
            f"F8R58/nuisance/{SALT[panel]}/{law_index}/{case_idx}/{bridge}".encode()
            ).digest()[1]-127.5)/127.5*.012
        proxy_sign=hashed_sign("F8R58","selected_proxy",SALT[panel],
                               law_index,case_idx,bridge)
        terminal_values=[]
        proxy_values=[]
        for action in (0,1):
            directional=1. if action==0 else -1.
            terminal=AMP[panel]*sign*directional+nuisance
            proxy=PROXY[panel]*proxy_sign*directional+nuisance*.5
            records.extend((
              {"bridge":bridge,"selected_action":action,
               "public_sensor":"terminal","observed_effect":terminal},
              {"bridge":bridge,"selected_action":action,
               "public_sensor":"proxy","observed_effect":proxy}
            ))
            terminal_values.append(terminal)
            proxy_values.append(proxy)
        features.append(((terminal_values[0]-terminal_values[1])*.5,
                         (proxy_values[0]-proxy_values[1])*.5))
    # FIT labels only, by explicitly selecting BOTH return actions from
    # independent resets. The return outcomes are NEVER returned to model.
    future=target_sign(law_index)
    if panel in ("FIT_train","FIT_heldout"):
        selected_return0=.16*future
        selected_return1=-.16*future
        label=0 if selected_return0>selected_return1 else 1
    else:
        # Evaluator-only goal sign; no selected return outcomes provided.
        label=0 if future==1 else 1
    flat=[v for p in features for v in p]
    return {"law_index_evaluator_only":law_index,
            "case_index_evaluator_only":case_idx,
            "panel_evaluator_only":panel,
            "public_selected_records":records,
            "public_pairs":features,
            "observed_model_vector":flat,
            "selected_future_return_choice_label_evaluator_only":label}


def cohort(panel):
    law_indices=OOD_LAWS if panel=="OOD" else FIT_LAWS
    samples=[selected_case(panel,law,case)
             for law in law_indices for case in range(PER_LAW[panel])]
    expected=len(law_indices)*PER_LAW[panel]
    if len(samples)!=expected:
        raise ValueError("F8R58_UNREGISTERED_COHORT")
    x=torch.tensor([s["observed_model_vector"] for s in samples],
                   dtype=torch.float32)
    y=torch.tensor([s["selected_future_return_choice_label_evaluator_only"]
                    for s in samples],dtype=torch.long)
    if x.shape!=(expected,10) or y.shape!=(expected,):
        raise ValueError("F8R58_INPUT_OR_TARGET_SHAPE")
    return x,y


class Policy(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R58_UNREGISTERED_ARM")
        self.arm,self.seed=arm,seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(DRAW_SEED+seed+(0 if arm==ARM_NAMES[0] else 100000))
            if arm=="mlp_selected_terminal_proxy":
                self.mlp=nn.Sequential(nn.Linear(10,64),nn.ReLU(),
                    nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
            else:
                self.gru=nn.GRU(input_size=2,hidden_size=48,batch_first=True)
                self.head=nn.Linear(48,2)

    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=10:
            raise ValueError("F8R58_INVALID_SELECTED_FEATURES")
        if self.arm=="mlp_selected_terminal_proxy":
            return self.mlp(x)
        _,hidden=self.gru(x.reshape(-1,5,2))
        return self.head(hidden[-1])


def accuracy(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=model(x)
    if not torch.isfinite(logits).all():
        raise ValueError("F8R58_NONFINITE_PREDICTION")
    choices=logits.argmax(-1)
    return {"accuracy":float((choices==y).float().mean()),
            "choices":choices.tolist()}


def train(arm,seed,cohorts):
    torch.manual_seed(DRAW_SEED) # train draws independent of model seed
    model=Policy(arm,seed)
    x,y=cohorts["FIT_train"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0.)
    gen=torch.Generator(device="cpu").manual_seed(DRAW_SEED)
    productive=0
    checkpoints=[]
    began=time.perf_counter()
    for step in range(1,UPDATES+1):
        indices=torch.randint(0,len(y),(BATCH,),generator=gen)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(x[indices]),y[indices])
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R58_NONFINITE_TRAIN_LOSS")
        loss.backward()
        params=list(model.parameters())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in params):
            raise ValueError("F8R58_NONFINITE_GRADIENT")
        productive+=int(any(p.grad is not None and bool(torch.any(p.grad!=0))
                            for p in params))
        torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        if step in CHECKPOINTS:
            checkpoints.append({
                "step":step,
                "loss":float(loss.detach()),
                "FIT_train_accuracy":accuracy(model,*cohorts["FIT_train"])["accuracy"],
                "FIT_heldout_accuracy":accuracy(model,*cohorts["FIT_heldout"])["accuracy"]})
    panels={p:{"active":accuracy(model,*cohorts[p])} for p in PANELS}
    scored_x,scored_y=cohorts["OOD"]
    terminal_erased=scored_x.clone();terminal_erased[:,0::2]=0
    proxy_erased=scored_x.clone();proxy_erased[:,1::2]=0
    panels["OOD"]["erase_selected_terminal"]=accuracy(model,terminal_erased,scored_y)
    panels["OOD"]["erase_selected_proxy"]=accuracy(model,proxy_erased,scored_y)
    cold=Policy(arm,seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if any(accuracy(cold,*cohorts[p])["choices"]!=panels[p]["active"]["choices"]
           for p in PANELS):
        raise ValueError("F8R58_COLD_RESTORE_ACTION_DRIFT")
    return {"arm":arm,"seed":seed,
            "training_updates":UPDATES,
            "productive_gradient_steps":productive,
            "train_seconds":time.perf_counter()-began,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "trace":checkpoints,"scores":panels,"cold_restore":True}


def judge(models,frozen,source):
    limits=frozen["admission"]
    by={(m["arm"],m["seed"]):m for m in models}
    verdict=[]
    for seed in SEEDS:
        mlp=by["mlp_selected_terminal_proxy",seed]
        gru=by["gru_selected_terminal_proxy",seed]
        ms=mlp["scores"]
        gs=gru["scores"]
        criteria={
          "MLP_FIT_train":ms["FIT_train"]["active"]["accuracy"]>=
             limits["MLP_FIT_train_accuracy_min"],
          "MLP_FIT_heldout":ms["FIT_heldout"]["active"]["accuracy"]>=
             limits["MLP_FIT_heldout_accuracy_min"],
          "GRU_FIT_train":gs["FIT_train"]["active"]["accuracy"]>=
             limits["GRU_FIT_train_accuracy_min"],
          "GRU_FIT_heldout":gs["FIT_heldout"]["active"]["accuracy"]>=
             limits["GRU_FIT_heldout_accuracy_min"],
          "MLP_OOD_min":ms["OOD"]["active"]["accuracy"]>=
             limits["MLP_OOD_accuracy_min"],
          "MLP_OOD_headroom":ms["OOD"]["active"]["accuracy"]<=
             limits["MLP_OOD_accuracy_max"],
          "terminal_erase_control":ms["OOD"]["erase_selected_terminal"]["accuracy"]<=
             limits["terminal_erasure_MLP_OOD_accuracy_max"],
          "proxy_erase_control":(
             ms["OOD"]["active"]["accuracy"]-
             ms["OOD"]["erase_selected_proxy"]["accuracy"])<=
             limits["proxy_erasure_MLP_OOD_accuracy_loss_max"]
        }
        verdict.append({"seed":seed,"criteria":criteria,
            "task_admission_seed_pass":all(criteria.values()),
            "MLP_FIT_train":ms["FIT_train"]["active"]["accuracy"],
            "MLP_FIT_heldout":ms["FIT_heldout"]["active"]["accuracy"],
            "MLP_OOD":ms["OOD"]["active"]["accuracy"],
            "GRU_FIT_train":gs["FIT_train"]["active"]["accuracy"],
            "GRU_FIT_heldout":gs["FIT_heldout"]["active"]["accuracy"],
            "GRU_OOD":gs["OOD"]["active"]["accuracy"],
            "MLP_erase_terminal_OOD":ms["OOD"]["erase_selected_terminal"]["accuracy"],
            "MLP_erase_proxy_OOD":ms["OOD"]["erase_selected_proxy"]["accuracy"]})
    return {"task_admitted":bool(source["weak_source_shortcuts_admitted"] and
            all(v["task_admission_seed_pass"] for v in verdict)),
            "original_seeds":verdict,
            "independent_scientific_pass":False,
            "no_novel_learning_mechanism_tested":True,
            "capacity_compute_matched_new_mechanism":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    audit=source_only_heuristics(freeze)
    cohorts={panel:cohort(panel) for panel in PANELS}
    for panel in PANELS:
        _,labels=cohorts[panel]
        if set(labels.tolist())!={0,1}:
            raise ValueError("F8R58_UNBALANCED_PANEL_LABELS")
    models=[train(arm,seed,cohorts) for arm,seed in ARMS]
    conclusion=judge(models,freeze,audit)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "parent":PARENT,"source_neutral_task_admission_only":True,
        "selected_factual_observations_per_case":20,
        "cohort_sizes":{p:len(cohorts[p][1]) for p in PANELS},
        "public_analytic_parity_oracle_accuracy":1.0,
        "source_audit":audit,
        "actual_optimizer_attempts":len(ARMS)*UPDATES,
        "models":models,"judgment":conclusion,
        "independent_scientific_pass":False,
        "new_unconventional_model_trained":False,
        "external_original_firstuse":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    p=ROOT/"local_runs/p85mr_f8r58_five_bridge_task_admission_dev.json"
    p.parent.mkdir(exist_ok=True)
    if p.exists():
        raise FileExistsError("F8R58_REFUSE_OVERWRITE")
    p.write_text(json.dumps(result,sort_keys=True,indent=2,
                            allow_nan=False)+"\n")


def deny_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R58_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
