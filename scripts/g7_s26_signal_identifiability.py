"""S26 detector identifiability: deterministic synthetic prequential signal controls.

This does NOT run learned models or fit/train/calibrate parameters. It tests a
FROZEN gate against an explicitly hand-crafted prediction/observation stream:
stable action contrast, persistent sign reversal after step 20, one-frame
reversal, sub-threshold contrast, and persistent initial miscalibration.

The cases are mechanism-positive/negative CONTROLS, not physical simulation,
generalization, adaptation, retention or science. The change time is part of
test data only; it is never passed to a gate. No S25/S26 policy is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g7_s25_contradiction_gate import PrequentialGate, MATCHED_UPDATES
from g7_s26_reserve_gate import QuarterReserveGate

CASES = (
    "stable",
    "persistent_reversal_at_20",
    "one_step_reversal_at_20",
    "subthreshold_reversal_at_20",
    "persistent_initial_miscalibration",
)
FROZEN_PREDICTION = (-0.05, 0.05)
MATCHED_SEED = 3101
EPS_STEPS = 32


def observation(case: str, step: int) -> tuple[float, float]:
    if case not in CASES or type(step) is not int or not 0 <= step < EPS_STEPS:
        raise ValueError("S26_SYNTHETIC_CASE_OR_TIME_NOT_REGISTERED")
    if case == "persistent_initial_miscalibration":
        return (0.05, -0.05)
    if case == "persistent_reversal_at_20" and step >= 20:
        return (0.05, -0.05)
    if case == "one_step_reversal_at_20" and step == 20:
        return (0.05, -0.05)
    if case == "subthreshold_reversal_at_20" and step >= 20:
        return (0.005, -0.005)
    return FROZEN_PREDICTION


def trace(case: str, mechanism: str) -> dict:
    """No hidden coefficients, fitted predictor or switch-step input to gate."""
    if mechanism not in ("S25", "S26"):
        raise ValueError("UNREGISTERED_PREQUENTIAL_GATE")
    gate=(PrequentialGate("contradiction", engineering_seed=MATCHED_SEED)
          if mechanism == "S25" else
          QuarterReserveGate("contradiction", engineering_seed=MATCHED_SEED))
    outcomes=[]
    for t in range(EPS_STEPS):
        actual=observation(case, t)
        d=gate.observe(
            t,predicted_minus=FROZEN_PREDICTION[0],
            predicted_plus=FROZEN_PREDICTION[1],
            observed_minus=actual[0],observed_plus=actual[1],
            primary_residual=actual[0]-FROZEN_PREDICTION[0],
        )
        outcomes.append({
            "step":t,"update":d.update,"trigger":d.trigger,
            "reason":d.reason,
            "persistent_contradiction":d.contradiction_evidence_count >= 2,
        })
    if gate.close()!=MATCHED_UPDATES:
        raise ValueError("S26_INVALID_K8_BUDGET")
    updates=[r["step"] for r in outcomes if r["update"]]
    event_selected=[r["step"] for r in outcomes if r["reason"]=="trigger"]
    fallback=[r["step"] for r in outcomes if r["reason"]=="forced_budget"]
    if len(updates)!=8 or len(event_selected)+len(fallback)!=8:
        raise ValueError("S26_SYNTHETIC_UPDATE_REASON_MISMATCH")
    return {
        "mechanism":mechanism,"case":case,
        "updates":updates,"selected_by_signal":event_selected,
        "forced_by_budget":fallback,
        "first_signal_selected":event_selected[0] if event_selected else None,
        "post20_signals":sum(x>=20 for x in event_selected),
        "post20_updates":sum(x>=20 for x in updates),
        "quarter_update_counts":[sum(start<=x<start+8 for x in updates)
                                 for start in (0,8,16,24)],
        "persistent_conflict_frames":sum(
            bool(row["persistent_contradiction"]) for row in outcomes),
        "no_trainer_or_data_fit":True,
    }


def synthetic_mechanism_audit() -> dict:
    cases=[]
    for case in CASES:
        cases.append({"case":case,"S25":trace(case,"S25"),
                      "S26":trace(case,"S26")})
    by_case={row["case"]:row for row in cases}
    strong=by_case["persistent_reversal_at_20"]
    transient=by_case["one_step_reversal_at_20"]
    weak=by_case["subthreshold_reversal_at_20"]
    stable=by_case["stable"]
    wrong=by_case["persistent_initial_miscalibration"]
    if not (
        strong["S26"]["first_signal_selected"] == 21
        and strong["S26"]["post20_signals"] >= 1
        and strong["S25"]["post20_signals"] >= 1
        and all(not stable[v]["selected_by_signal"] for v in ("S25","S26"))
        and all(not transient[v]["selected_by_signal"] for v in ("S25","S26"))
        and all(not weak[v]["selected_by_signal"] for v in ("S25","S26"))
        and wrong["S25"]["post20_updates"]==0
        and wrong["S26"]["post20_updates"]==2
        and all(row["S26"]["quarter_update_counts"]==[2,2,2,2]
                for row in cases)
    ):
        raise ValueError("S26_SYNTHETIC_MECHANISM_POSITIVE_NEGATIVE_CONTROL_FAILED")
    return {
        "schema":"ttt-hybrid.g7-s26.synthetic-signal-identifiability.v1",
        "status":"PASS_HANDCRAFTED_SIGNAL_CONTROLS_ONLY",
        "cases":cases,
        "interpretation":{
            "strong_persistent_reversal_is_mechanically_detectable":True,
            "stable_single_frame_and_subthreshold_signals_are_rejected":True,
            "miscalibration_exhausts_S25_K8_before_step20":True,
            "S26_preserves_late_K8_under_miscalibration":True,
            "causal_shift_recognition_in_actual_development_worlds":"NOT_DEMONSTRATED",
            "learned_model_competence":"NOT_ESTABLISHED",
            "generalization_and_scientific_qualification":"NOT_RUN",
        },
        "frozen_scientific_seeds_consumed":False,
        "models_trained":False,
        "gpu_used":False,
        "s25_s26_gate_implementations_modified":False,
        "full_cross_platform_regression":"UNVERIFIED",
    }


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--sha",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    proc=subprocess.run(["git","-C",str(repo),"rev-parse","HEAD"],
                        text=True,capture_output=True,timeout=10,check=False)
    if (proc.returncode!=0 or proc.stdout.strip()!=args.sha or
            len(args.sha)!=40 or
            any(c not in "0123456789abcdef" for c in args.sha)):
        raise SystemExit("S26_SYNTHETIC_AUDIT_SHA_DRIFT")
    result=synthetic_mechanism_audit()
    result["candidate_sha"]=args.sha
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S26_SYNTHETIC_SIGNAL_AUDIT="+result["status"])
    print("S26_SYNTHETIC_S25_POST20_MISFIT="+str(
        next(r for r in result["cases"] if r["case"]==
             "persistent_initial_miscalibration")["S25"]["post20_updates"]))
    print("S26_SYNTHETIC_S26_POST20_MISFIT="+str(
        next(r for r in result["cases"] if r["case"]==
             "persistent_initial_miscalibration")["S26"]["post20_updates"]))
    print("S26_SCIENTIFIC_FIRST_USE=NOT_RUN")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
