"""R112: FIT/VALID-only fixed-point portable native active probe inference.

R111's already-scored UNIT_UNSEEN is never opened or regenerated here.
The R111 float model is trained freshly on R111 FIT, then quantized to a
deterministic, integer-only inference path with a frozen hard-swish surrogate.
Strict cross-OS parity is an engineering requirement, NOT scientific proof.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import torch

from experiments import r111_active_probe_diagnostic as r111
from experiments import r112_portable_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r112_portable_active_probe_fit_valid.json"
SEEDS=custody.SEEDS
Q=custody.Q
LAYERS=(5,32,16,1)
PHASE_STEPS=6


def _rdiv(z,den):
    """Nearest integer with symmetric ties, negative values included."""
    a=np.asarray(z,dtype=np.int64)
    if type(den) is not int or den<1:
        raise ValueError("R112_INTEGER_DENOMINATOR_INVALID")
    return np.where(a<0,-((-a+den//2)//den),(a+den//2)//den)


def _hard_swish(z):
    z=np.asarray(z,dtype=np.int64)
    return _rdiv(z*np.clip(z+2*Q,0,4*Q),4*Q)


def quantize_model(model):
    custody.audit_freeze()
    layers=[]
    h=hashlib.sha256(b"R112_NATIVE_INTEGER_QUANTIZED_HARDSWISH_V1|Q=384")
    for module in (model.scorer[0],model.scorer[2],model.scorer[4]):
        if not isinstance(module,torch.nn.Linear):
            raise RuntimeError("R112_UNREGISTERED_NATIVE_MODULE")
        tensors=[]
        for kind in ("weight","bias"):
            p=getattr(module,kind).detach().cpu().numpy()
            if not bool(np.isfinite(p).all()):
                raise RuntimeError("R112_NONFINITE_TRAINED_WEIGHT")
            # numpy rounds ties-to-even. Canonical little-endian packed int64.
            t=np.rint(p.astype(np.float64)*Q).astype("<i8")
            if np.any(np.abs(t)>=2**20):
                raise OverflowError("R112_QUANTIZED_VALUE_OUT_OF_RANGE")
            h.update(kind.encode());h.update(struct.pack("<I",t.ndim))
            for d in t.shape:
                h.update(struct.pack("<I",d))
            h.update(t.tobytes(order="C"))
            tensors.append(t.tolist())
        layers.append({"weight":tensors[0],"bias":tensors[1]})
    if [len(x["weight"][0]) for x in layers]!=list(LAYERS[:-1]):
        raise RuntimeError("R112_QUANTIZED_NATIVE_ARCHITECTURE_DRIFT")
    return layers,h.hexdigest()


def _integer_feature(history,offers,phase):
    if (phase not in (0,1) or len(offers)!=2
        or not isinstance(offers[0],int) or not isinstance(offers[1],int)
        or offers[0]==offers[1] or not all(0<=p<8 for p in offers)
        or len(history)>PHASE_STEPS):
        raise ValueError("R112_UNAUTHORIZED_PUBLIC_PROBE_FEATURE")
    counts=[0]*8;signed=[0]*8
    for p,r in history:
        if not isinstance(p,int) or not 0<=p<8 or r not in (-1,1):
            raise ValueError("R112_INVALID_PUBLIC_CHOSEN_FEEDBACK")
        counts[p]+=1;signed[p]+=r
    u=Q//PHASE_STEPS
    # Same physical features as R111 native scorer, converted EXACTLY from
    # integer observed counts; no torch/float inference in this path.
    return np.array([
        [u*counts[p],u*signed[p],u*abs(signed[p]),Q*phase,u*len(history)]
        for p in offers],dtype=np.int64)


def portable_scores(layers,history,offers,phase):
    if len(layers)!=3:
        raise PermissionError("R112_UNKNOWN_QUANTIZED_MODEL_ARCH")
    z=_integer_feature(history,offers,phase)
    for i,layer in enumerate(layers):
        w=np.asarray(layer["weight"],dtype=np.int64)
        b=np.asarray(layer["bias"],dtype=np.int64)
        if w.shape!=(LAYERS[i+1],LAYERS[i]) or b.shape!=(LAYERS[i+1],):
            raise RuntimeError("R112_INTEGER_LAYER_SHAPE_DRIFT")
        z=_rdiv(z@w.T,Q)+b[None,:]
        if np.any(np.abs(z)>2**25):
            raise OverflowError("R112_INTEGER_INFERENCE_RANGE_EXCEEDED")
        if i<2:
            z=_hard_swish(z)
    if z.shape!=(2,1):
        raise RuntimeError("R112_INTEGER_ACTION_VALUE_SHAPE")
    return int(z[0,0]),int(z[1,0])


def choose_portable(layers,offers,history,phase):
    s0,s1=portable_scores(layers,history,offers,phase)
    return int(offers[int(s1>s0)])


def independent_cold_integer_parity(layers):
    """Fresh Python process reads pure JSON integer weights, no GPU/owner."""
    samples=[
      ([],[0,3],0),
      ([[0,1],[3,-1]],[0,5],1),
      ([[3,1],[4,-1],[3,1],[0,-1]],[3,7],0),
      ([[0,-1],[1,-1],[2,1],[3,1],[0,1],[5,-1]],[2,5],1)]
    original=[portable_scores(layers,[tuple(v) for v in h],tuple(o),p)
              for h,o,p in samples]
    payload=json.dumps({"layers":layers,"samples":samples},sort_keys=True).encode()
    code=("import json,sys\n"
          "from experiments.r112_portable_probe_diagnostic import portable_scores\n"
          "p=json.load(sys.stdin)\n"
          "scores=[portable_scores(p['layers'],[tuple(v) for v in h],tuple(o),ph) "
          "for h,o,ph in p['samples']]\n"
          "print(json.dumps(scores,sort_keys=True))\n")
    run=subprocess.run([sys.executable,"-c",code],input=payload,cwd=ROOT,
                       capture_output=True,check=True,timeout=30)
    restored=json.loads(run.stdout)
    return sum(int(tuple(v)!=tuple(o)) for v,o in zip(original,restored))


def _validate_actor(seed,policy,models,layers):
    if seed not in SEEDS or policy not in ("portable","R111_float"):
        raise PermissionError("R112_UNAUTHORIZED_VALIDATION_ACTOR")
    custody.fit_valid_only("UNIT_VALID")
    actions=[];observed=[];query_signs=[];new_correct=old_correct=feedback_delta=0
    old_after=0
    for world_index in range(88):
        for phase in (0,1):
            world=r111.PhysicalCausalWorld("UNIT_VALID",seed,world_index,phase)
            for step in range(PHASE_STEPS):
                offers=world.offers()
                if policy=="portable":
                    action=choose_portable(layers,offers,world.history,phase)
                    # Intervention on inference input ONLY, not on environment.
                    blinded=[(p,1) for p,_ in world.history]
                    feedback_delta+=int(action!=choose_portable(
                        layers,offers,blinded,phase))
                else:
                    action=r111.choose(r111.NATIVE,models,offers,
                                       world.history,phase,seed,world_index,step)
                if action not in offers:
                    raise RuntimeError("R112_NOT_OFFERED_OR_UNCHOSEN_ACTION")
                effect=world.intervene(action)
                actions.append(action)
                observed.append(1 if effect>0 else 0)
            query,truth=world.sealed_query()
            guess=r111.fixed_readout(query,world.history)
            query_signs.append(1 if guess==truth else 0)
            if phase:
                new_correct+=int(guess==truth)
            else:
                old_correct+=int(guess==truth)
                old_after+=int(guess==truth)
    if len(actions)!=88*12 or len(query_signs)!=88*2:
        raise RuntimeError("R112_PHYSICAL_OBSERVATION_BUDGET_DRIFT")
    return {"policy":policy,"old_correct":old_correct,
            "new_correct":new_correct,"old_after_correct":old_after,
            "new_accuracy":new_correct/88,
            "total_chosen_actions":len(actions),
            "positive_observations":sum(observed),
            "feedback_sensitivity_choices":feedback_delta,
            "action_sha256":hashlib.sha256(bytes(actions)).hexdigest(),
            "outcome_sha256":hashlib.sha256(bytes(observed)).hexdigest(),
            "correctness_sha256":hashlib.sha256(bytes(query_signs)).hexdigest()}


def run_seed(seed):
    custody.audit_freeze()
    if seed not in SEEDS:
        raise PermissionError("R112_UNREGISTERED_SEED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    fit=r111.fit_examples(seed,None)
    rng=np.random.default_rng(np.random.SeedSequence(
        [r111.SALTS["UNIT_BATCH"],seed,111]))
    minibatches=rng.integers(0,len(fit[3]),size=(512,16))
    models=r111.new_models(seed)
    trained=r111.train_model(models[r111.NATIVE],r111.NATIVE,fit,minibatches,None)
    if trained["updates"]!=512 or trained["parameters"]!=737:
        raise RuntimeError("R112_NATIVE_TRAINING_BUDGET_DRIFT")
    layers,weight_hash=quantize_model(models[r111.NATIVE])
    cold=independent_cold_integer_parity(layers)
    native_float=_validate_actor(seed,"R111_float",models,layers)
    portable=_validate_actor(seed,"portable",models,layers)
    valid_drop=native_float["new_accuracy"]-portable["new_accuracy"]
    return {"seed":seed,"quantized_weight_sha256":weight_hash,
            "portable":portable,"R111_float_development":native_float,
            "portable_accuracy_drop_vs_float_valid":valid_drop,
            "local_quality_gate_pass":(
                cold==0 and valid_drop<=.10 and
                portable["total_chosen_actions"]==88*12 and
                trained["updates"]==512),
            "native_fresh_training":trained,
            "cold_integer_score_mismatches":cold,
            "wall_seconds":time.perf_counter()-start,
            "UNIT_UNSEEN_generated":False,"original_science_firstuse":False}


def run_all():
    f=custody.audit_freeze()
    if (ROOT/"experiments/r111_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R112_PROTECTED_FIRSTUSE_NOT_IN_SCOPE")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected is not None and sha!=expected:
        raise PermissionError("R112_EXACT_CANDIDATE_SHA_DRIFT")
    seeds={str(seed):run_seed(seed) for seed in SEEDS}
    local=all(x["local_quality_gate_pass"] for x in seeds.values())
    return {"schema":"ttt-hybrid.r112.fit-valid-only-portable-native.v1",
            "exact_source_sha":sha,"freeze_blob":custody.FREEZE_BLOB,
            "status":("R112_LOCAL_QUALIFICATION_PASS_PLATFORM_PARITY_UNVERIFIED"
                      if local else f["evaluation"]["fail_status"]),
            "local_fit_valid_qualification_pass":local,
            "strict_cross_OS_parity_proven_in_one_job":False,
            "by_seed":seeds,"R111_original_unseen_replayed":False,
            "R112_UNSEEN_generated":False,"original_scientific_firstuse":False,
            "scientific_pass":False,"GPU_firstuse":False,"promotion":False,
            "deployment":"HOLD"}


def main():
    data=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R112_IMMUTABLE_RECEIPT_OVERWRITE_DENIED")
    RECEIPT.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R112_PORTABILITY_FIT_VALID "+json.dumps({
        "sha":data["exact_source_sha"],"local_pass":data["local_fit_valid_qualification_pass"],
        "seeds":{k:{"quantized_sha":v["quantized_weight_sha256"],
                    "portable_accuracy":v["portable"]["new_accuracy"],
                    "float_accuracy":v["R111_float_development"]["new_accuracy"],
                    "action_sha":v["portable"]["action_sha256"],
                    "cold_mismatches":v["cold_integer_score_mismatches"]}
                 for k,v in data["by_seed"].items()},
        "science":False,"UNIT_UNSEEN_generated":False},sort_keys=True))


if __name__=="__main__":
    main()
