"""R90: original max-width native optimizer microdiagnostic, TEST fixtures only.

No protected R85-R89 cohorts, new DEV/OOD, model promotion, GPU first use,
new workflow or runner. Reuses full dual-platform G4 tests. Two seeds,
four real observed factual action returns per seed. An overfit result is a
development optimization check, never independent native scientific PASS.
"""
from __future__ import annotations

import gc
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

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r79_router_gradient as r79
from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85
from experiments import r88_causal_event_access as r88
from experiments import r89_native_event_access as r89

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r90_native_fit_freeze.json"
FREEZE_BLOB = "10f0e158bdc6534cb6f2fe50c9b37886db2a3100"
SCHEMA = "ttt-hybrid.r90-basic-factual-fit-diagnostic-v1"
PARENT = "3b75351cc09979a505c7c1ea5e95f779afbb22c1"
BRANCH = "research/r90-native-basic-fit-diagnosis-20261115"
SEEDS = (1000411, 1000423)
ARMS = ("NATIVE_HARD_TTT", "NATIVE_TTT_UPDATES_OFF", "NATIVE_NO_DEPTH",
        "NATIVE_INACTIVE_SOFT_SURROGATE", "GRU64_PUBLIC_SLOTS")
WIDTH = 1024
COUNT = 4
STEPS = 64
CHECKPOINTS = (0,1,8,16,32,64)
SALT = 25320991


def frozen():
    if r74.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R90_IMMUTABLE_FREEZE_BLOB_DRIFT")
    r89.frozen()
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    parent, src, g, s = f["parent"], f["source"], f["training"], f["thresholds"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "PROSPECTIVELY_FROZEN_BEFORE_R90_CODE_TESTS_OR_QUALIFICATION"
        or parent["R89_exact_candidate"] != PARENT
        or parent["R89_freeze_blob"] != r89.FREEZE_BLOB
        or parent["R89_original_result"] != "R89_NATIVE_ROLE_ACCESS_NO_GO_TWO_SEEDS_TWO_OF_THIRTEEN"
        or src["kind"] != "TEST_ONLY_DEVELOPMENT_FIXTURE_NOT_PROTECTED_SCIENCE"
        or src["R90_test_fixture_salt"] != SALT
        or src["worlds_per_seed"] != COUNT
        or f["seeds"] != list(SEEDS) or f["arms"] != list(ARMS)
        or g["steps"] != STEPS or g["batch"] != COUNT
        or g["checkpoint_indices"] != list(CHECKPOINTS)
        or f["control_geometry"]["native_width"] != WIDTH
        or f["control_geometry"]["native_core_parameter_count"] != 16818178
        or s["per_arm_basic_fit_final_MSE_absolute_max"] != 4.
        or s["per_arm_basic_fit_final_to_initial_MSE_ratio_max"] != .25
        or s["hard_forward_surrogate_initial_absolute_error_max"] != .000001
        or s["independent_scientific_pass"] is not False
        or any(f["authority"].values())):
        raise PermissionError("R90_FROZEN_SEMANTIC_SOURCE_OR_AUTHORITY_DRIFT")
    if SALT in set(r89.SALT.values()):
        raise RuntimeError("R90_TEST_FIXTURE_COLLIDES_WITH_SPENT_R89")
    return f


def test_only_factual(seed: int):
    """Exactly four independent legal one-intervention observations per seed."""
    if seed not in SEEDS:
        raise PermissionError("R90_NONFROZEN_TEST_SEED")
    raw, targets, actions = [], [], []
    for i in range(COUNT):
        rng = np.random.default_rng(np.random.SeedSequence([SALT, seed, i]))
        world = r85._make_synthetic_world(i % r85.N, rng)
        action = r85.CALIBRATE if i % 2 == 0 else r85.EXPLOIT_POS
        prefix, observed_return = r89._view(world, action)
        events = r88._public_slots(prefix)
        if (events.shape != (5, 13) or
            not np.array_equal(events[-1], prefix[-1]) or
            not math.isfinite(observed_return) or
            events[-1, action] != 1.0):
            raise RuntimeError("R90_TEST_FIXTURE_NOT_LEGAL_FACTUAL_INTERVENTION")
        raw.append(events)
        targets.append(observed_return)
        actions.append(action)
    x = torch.from_numpy(np.stack(raw).astype(np.float32))
    y = torch.tensor(targets, dtype=torch.float32)
    return {"input": x, "factual_return": y, "logged_actions": actions,
            "TEST_ONLY": True, "no_private_law_or_unchosen_rewards": True}


def _native(seed: int, arm: str, force_off: bool = False):
    torch.manual_seed(seed)
    m = r74.make_model("HybridPilot_FULL", WIDTH)
    old = m.ttt
    replacement = r75.TriangularFastWeightMemory(WIDTH, learning_rate=old.eta)
    replacement.load_state_dict(old.state_dict(), strict=True)
    m.ttt = replacement
    original_count = sum(p.numel() for p in m.parameters())
    if original_count != 16818178:
        raise RuntimeError("R90_MAX_WIDTH_NATIVE_CORE_DRIFT")
    if arm == "NATIVE_NO_DEPTH":
        m.depth = None
    elif arm == "NATIVE_INACTIVE_SOFT_SURROGATE":
        old_depth = m.depth
        soft = r79.InactiveSoftSurrogateRouter(WIDTH)
        soft.load_state_dict(old_depth.state_dict(), strict=True)
        m.depth = soft
    elif arm not in ("NATIVE_HARD_TTT", "NATIVE_TTT_UPDATES_OFF"):
        raise ValueError("R90_NONFROZEN_NATIVE_ARM")
    if force_off or arm == "NATIVE_INACTIVE_SOFT_SURROGATE":
        if m.depth is None:
            raise ValueError("R90_CANNOT_FORCE_MISSING_ROUTER")
        with torch.no_grad():
            m.depth.router.weight.zero_()
            m.depth.router.bias.fill_(-3.)
    return m


def _model(seed: int, arm: str):
    if arm in ARMS[:-1]:
        return _native(seed, arm)
    if arm == "GRU64_PUBLIC_SLOTS":
        torch.manual_seed(seed+64)
        return r81.GRU(64)
    raise ValueError("R90_UNREGISTERED_ARM")


def _prediction(m, arm, x):
    if arm == "NATIVE_TTT_UPDATES_OFF":
        result = m(x, ttt_update=False)
    else:
        result = m(x)
    if result.shape != (len(x), 5, 1):
        raise RuntimeError("R90_NATIVE_GRU_FIVE_PUBLIC_EVENTS_SHAPE_DRIFT")
    return result[:,-1,0]


def _mse(m,arm,x,y):
    m.eval()
    with torch.no_grad():
        loss=float(F.mse_loss(_prediction(m,arm,x),y))
    m.train()
    if not math.isfinite(loss):
        raise RuntimeError("R90_NONFINITE_ORIGINAL_TEST_FIT_LOSS")
    return loss


def _group_grad(m):
    params = dict(m.named_parameters())
    keys = {
        "stem": "stem.weight",
        "head": "head.weight",
        "ssm": "ssm.candidate.weight",
        "attention": "attn.qkv.weight",
        "ttt": "ttt.value.weight",
        "depth_router": "depth.router.weight",
        "depth_ffn": "depth.ffn.0.weight"}
    out={}
    for group,name in keys.items():
        p=params.get(name)
        out[group] = (None if p is None or p.grad is None
                      else float(p.grad.detach().abs().max()))
    return out


def _fit_arm(seed, arm, src):
    m = _model(seed, arm)
    x,y = src["input"], src["factual_return"]
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.0001)
    trajectory={"0":_mse(m,arm,x,y)}
    init_grads=None
    initial_router_utilization=None
    started=time.perf_counter()
    for step in range(1,STEPS+1):
        out=_prediction(m,arm,x)
        if step==1 and getattr(m,"depth",None) is not None:
            initial_router_utilization=float(m.depth.last_utilization)
        loss=F.mse_loss(out,y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R90_NONFINITE_SELECTED_RETURN_OBJECTIVE")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step==1:
            init_grads=_group_grad(m)
            if (init_grads["head"] is None or init_grads["head"]<=0
                or not all((v is None or math.isfinite(v)) for v in init_grads.values())):
                raise RuntimeError("R90_MISSING_OR_NONFINITE_INITIAL_HEAD_GRADIENT")
        nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
        if step in CHECKPOINTS:
            trajectory[str(step)] = _mse(m,arm,x,y)
    elapsed=time.perf_counter()-started
    if any(not math.isfinite(v) for v in trajectory.values()):
        raise RuntimeError("R90_NONFINITE_FROZEN_TRAIN_CURVE")
    final=float(trajectory[str(STEPS)])
    initial=float(trajectory["0"])
    ratio=final/max(initial,1e-12)
    t=frozen()["thresholds"]
    passes=(final<=t["per_arm_basic_fit_final_MSE_absolute_max"]
        and ratio<=t["per_arm_basic_fit_final_to_initial_MSE_ratio_max"])
    with torch.no_grad():
        errors=(_prediction(m,arm,x)-y).square().tolist()
    result={
        "arm":arm,"seed":seed,"steps":STEPS,"selected_factual_worlds":COUNT,
        "parameters":sum(p.numel() for p in m.parameters()),
        "training_wall_seconds":elapsed,"train_MSE":trajectory,
        "initial_gradient_max_abs":init_grads,
        "first_update_router_utilization":initial_router_utilization,
        "final_per_example_squared_error":errors,
        "final_over_initial_MSE":ratio,
        "basic_fit_both_frozen_thresholds":bool(passes),
        "five_public_selected_rows":True}
    del optimizer,m
    gc.collect()
    return result


def _off_router_pair(seed, src):
    """R79 exact hard forward versus inactive-only STE; no protected data."""
    x,y=src["input"],src["factual_return"]
    hard=_native(seed,"NATIVE_HARD_TTT",force_off=True)
    soft=_native(seed,"NATIVE_INACTIVE_SOFT_SURROGATE")
    a,b=dict(hard.named_parameters()),dict(soft.named_parameters())
    equal=a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    if not equal:
        raise RuntimeError("R90_FORCED_OFF_NATIVE_INITIAL_TENSOR_IDENTITY_FAIL")
    pa=_prediction(hard,"NATIVE_HARD_TTT",x)
    pb=_prediction(soft,"NATIVE_INACTIVE_SOFT_SURROGATE",x)
    diff=float((pa-pb).detach().abs().max())
    u_h,u_s=hard.depth.last_utilization,soft.depth.last_utilization
    ga={}
    for arm,m,pred in (("hard_off",hard,pa),("surrogate_off",soft,pb)):
        m.zero_grad(set_to_none=True)
        F.mse_loss(pred,y).backward()
        ga[arm]=_group_grad(m)
    t=frozen()["thresholds"]
    res={"identical_initial_core_parameters":equal,
         "initial_forward_max_abs_difference":diff,
         "hard_off_utilization":float(u_h),
         "surrogate_off_utilization":float(u_s),
         "hard_off_initial_gradients":ga["hard_off"],
         "surrogate_off_initial_gradients":ga["surrogate_off"],
         "forward_identical":diff<=t["hard_forward_surrogate_initial_absolute_error_max"],
         "hard_off_router_grad_zero":ga["hard_off"]["depth_router"] in (None,0),
         "surrogate_off_router_grad_positive":ga["surrogate_off"]["depth_router"] is not None
               and ga["surrogate_off"]["depth_router"]>0,
         "not_a_science_learning_result":True}
    del hard,soft
    gc.collect()
    if not (res["forward_identical"] and res["hard_off_router_grad_zero"]
            and res["surrogate_off_router_grad_positive"]
            and u_h==0 and u_s==0):
        raise RuntimeError("R90_ORIGINAL_SURROGATE_MECHANISM_TEST_FAILED")
    return res


def _custody() -> str:
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="cpu-qualification"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or os.getenv("RUNNER_OS") not in ("Linux","Windows")):
        raise PermissionError("R90_ORIGINAL_EXISTING_G4_ONLY")
    expected=os.getenv("EXPECTED_CANDIDATE_SHA","")
    observed=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=False,timeout=10)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
        capture_output=True,check=False,timeout=10)
    if (len(expected)!=40 or observed.returncode or
        observed.stdout.strip()!=expected or clean.returncode):
        raise PermissionError("R90_ORIGINAL_SHA_AND_CLEAN_G4_CUSTODY_FAIL")
    return expected


def original_g4_diagnostic():
    """Called by the normal full pytest suite, no new workflow/job."""
    sha=_custody()
    torch.set_num_threads(1)
    started=time.perf_counter()
    receipt={"schema":SCHEMA,"original_candidate_SHA":sha,"frozen_blob":FREEZE_BLOB,
             "R89_parent_SHA":PARENT,"runner_OS":os.getenv("RUNNER_OS"),
             "platform":platform.platform(),"torch":str(torch.__version__),
             "TEST_ONLY_no_protected_DEV_OOD_or_source":True,
             "no_main_or_native_model_mutation":True,"scientific_pass":False,
             "seeds":{},"classification":"R90_ENGINEERING_INCOMPLETE"}
    path=ROOT/f"local_runs/r90_native_fit_{os.getenv('RUNNER_OS','unknown').lower()}.json"
    if path.exists():
        raise FileExistsError("R90_UNEXPECTED_DUPLICATE_G4_RECEIPT")
    try:
        for seed in SEEDS:
            src=test_only_factual(seed)
            off=_off_router_pair(seed,src)
            arms={}
            for arm in ARMS:
                arms[arm]=_fit_arm(seed,arm,src)
            receipt["seeds"][str(seed)]={"source":"TEST_ONLY_PUBLIC_FACTUAL_FOUR_WORLD",
                "initial_hard_surrogate_identity_probe":off,"arms":arms}
        native_hard=[receipt["seeds"][str(seed)]["arms"][ARMS[0]]
                     ["basic_fit_both_frozen_thresholds"] for seed in SEEDS]
        native_ablations=[
            all(receipt["seeds"][str(seed)]["arms"][arm]
                ["basic_fit_both_frozen_thresholds"] for seed in SEEDS)
            for arm in ARMS[1:4]]
        t=frozen()["thresholds"]
        receipt["classification"]=(t["on_original_hard_fit_both_seeds"]
             if all(native_hard) else
             t["on_only_ablation_fits"] if any(native_ablations)
             else t["on_no_native_fits"])
    except BaseException as exc:
        receipt["error_type"]=type(exc).__name__
        raise
    finally:
        receipt["total_wall_seconds"]=time.perf_counter()-started
        path.parent.mkdir(exist_ok=True,parents=True)
        with path.open("x",encoding="utf-8") as f:
            json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return receipt
