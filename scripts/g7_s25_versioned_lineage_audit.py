"""Exact-text lineage proof for derivative S25 native-portability research source.

Engineering-only: verifies that V2 differs from frozen S16/S18/S20 only in
registered imports, non-Linux science-entry guards and correctly typed Linux
RSS meters, and that S25 ledger differs only in typed memory provenance.
NEVER trains, scores, changes source pins, reads scientific seeds or promotes.
This is stricter than AST model-equivalence: only exact whitelisted edits pass.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
FROZEN_BASE="108808f7b685107a76fae1ce5605b7ff9e2bac27"
V2_PARENT="140ff8080366306be1cdd573b9fa7b7a0efec340"
FROZEN_BLOBS={
 "scripts/g7_s16_nonlinear_switch.py":"12bc2b9db686531b7cf71a58f043f655d15c1708",
 "scripts/g7_s18_causal_feature_replication.py":"cfc1959b8812e9abad581018462302bc89a31d59",
 "scripts/g7_s20_hidden_multicycle.py":"ac3f9ddd5b879e53e62030707fb6419adaf732ca",
 "scripts/g7_s25_cost_ledger.py":"174e6d027cbc54831421697c32ac217499d91e7b",
 "experiments/g7_s25_frozen_contradiction_ttt.json":"31633b341403729b7ffbb7d73f3a04098ad44380",
 "experiments/g7_s20_frozen_hidden_multicycle.json":"05a38c0161ed67fc5f839947baaf3d4153d64a18",
 "source_pins.json":"1438b51f8548db6bfe7e04805ba8c156e4b5b630",
}
DERIVATIVE_BLOBS={
 "scripts/g7_s16_nonlinear_switch.py":"868c9ffb91f62279c1a6a21cf63e874dbf501b1e",
 "scripts/g7_s18_causal_feature_replication.py":"aa09ac6bc2787691e349889f96c1b791e2df0b64",
 "scripts/g7_s20_hidden_multicycle.py":"bc3e9f255a4d1b967252ada8782ac9f8c000f7b7",
 "scripts/g7_s25_cost_ledger.py":"e2bea80e39913892c79e9ad715782df1ba0f3507",
}
SCIENTIFIC_MODULES=tuple(DERIVATIVE_BLOBS)[:3]


class LineageBlocked(ValueError):
    """Fail closed if any model, physics, seed, threshold or memory edit drifts."""


def require(condition:bool, code:str)->None:
    if not condition:
        raise LineageBlocked(code)


def git(*args:str)->str:
    p=subprocess.run(["git","-C",str(ROOT),*args],text=True,
                     capture_output=True,check=False,timeout=14)
    require(p.returncode==0,"LINEAGE_GIT_CUSTODY_UNAVAILABLE")
    return p.stdout.strip()


def blob_digest(s:str)->str:
    data=s.encode("utf-8")
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\x00"+data).hexdigest()


def approved_edits(path:str)->tuple[tuple[str,str],...]:
    imports=(
        ("import resource\n",""),
    )
    if path=="scripts/g7_s16_nonlinear_switch.py":
        return imports+(
            ("import g7_r5_oneshot as custody\n",
             "import g7_r5_oneshot as custody\n"
             "from g7_process_peak_memory import read_peak_memory\n"),
            ("def run(args):\n",
             "def run(args):\n"
             "    # Original frozen S16 peak-RSS criterion is Linux-only.\n"
             "    check(sys.platform == 'linux', 'S16_FROZEN_RSS_REQUIRES_LINUX')\n"),
            ("    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024\n",
             "    peak=read_peak_memory()\n"
             "    check(peak.metric == 'linux_ru_maxrss_bytes' and peak.comparable_to_linux_ru_maxrss,\n"
             "          'S16_FROZEN_RSS_METRIC_NOT_EQUIVALENT')\n"
             "    rss=peak.peak_bytes\n"),
        )
    if path=="scripts/g7_s18_causal_feature_replication.py":
        return imports+(
            ("from torch.nn import functional as F\n",
             "from torch.nn import functional as F\n"
             "from g7_process_peak_memory import read_peak_memory\n"),
            ("def run(args):\n",
             "def run(args):\n"
             "    # Original frozen S18 peak-RSS criterion is Linux-only.\n"
             "    ensure(sys.platform == 'linux', 'S18_FROZEN_RSS_REQUIRES_LINUX')\n"),
            ("    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024\n",
             "    peak=read_peak_memory()\n"
             "    ensure(peak.metric == 'linux_ru_maxrss_bytes' and peak.comparable_to_linux_ru_maxrss,\n"
             "           'S18_FROZEN_RSS_METRIC_NOT_EQUIVALENT')\n"
             "    rss=peak.peak_bytes\n"),
        )
    if path=="scripts/g7_s20_hidden_multicycle.py":
        return imports+(
            ("from torch.nn import functional as F\n",
             "from torch.nn import functional as F\n"
             "from g7_process_peak_memory import read_peak_memory\n"),
            ("def run(args):\n",
             "def run(args):\n"
             "    # The frozen S20 assay has a Linux ru_maxrss/2 GiB criterion. Windows\n"
             "    # working-set bytes cannot be substituted or claimed equivalent.\n"
             "    # S25 may still import the original physics functions on Windows.\n"
             "    require(sys.platform == \"linux\", \"S20_FROZEN_RSS_REQUIRES_LINUX\")\n"),
            ("    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024\n",
             "    peak=read_peak_memory()\n"
             "    require(peak.metric == \"linux_ru_maxrss_bytes\"\n"
             "            and peak.comparable_to_linux_ru_maxrss,\n"
             "            \"S20_FROZEN_RSS_METRIC_NOT_EQUIVALENT\")\n"
             "    rss=peak.peak_bytes\n"),
        )
    if path=="scripts/g7_s25_cost_ledger.py":
        return imports+(
            ("from g7_s25_numeric_ttt import (\n"
             "    ACTION_PAIR, EPISODE_STEPS, FAST_WIDTH, PrequentialTTTEpisode, WorldStep,\n"
             ")\n",
             "from g7_s25_numeric_ttt import (\n"
             "    ACTION_PAIR, EPISODE_STEPS, FAST_WIDTH, PrequentialTTTEpisode, WorldStep,\n"
             ")\n"
             "from g7_process_peak_memory import read_peak_memory\n"),
            ("        self._closed = True\n        return {\n",
             "        # Different OS metrics stay different. Linux retains the original\n"
             "        # ru_maxrss semantics. Windows reports native peak working-set bytes\n"
             "        # only, without inventing an equivalent Linux RSS measurement.\n"
             "        memory=read_peak_memory()\n"
             "        self._closed = True\n        return {\n"),
            ("            \"process_peak_rss_bytes\": (\n"
             "                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),\n",
             "            \"process_peak_rss_bytes\": (\n"
             "                memory.peak_bytes\n"
             "                if memory.comparable_to_linux_ru_maxrss else None),\n"
             "            \"process_peak_working_set_bytes\": (\n"
             "                memory.peak_bytes\n"
             "                if memory.metric == \"windows_psapi_peak_working_set_bytes\"\n"
             "                else None),\n"
             "            \"process_peak_memory_metric\": memory.metric,\n"
             "            \"process_peak_memory_comparable_to_linux_ru_maxrss\": (\n"
             "                memory.comparable_to_linux_ru_maxrss),\n"),
        )
    raise LineageBlocked("LINEAGE_UNREGISTERED_DERIVATIVE_PATH")


def expected_candidate(path:str,original:str)->str:
    current=original
    for before,after in approved_edits(path):
        require(current.count(before)==1,"LINEAGE_EXPECTED_ANCHOR_MISSING_OR_DUPLICATED_"+path)
        current=current.replace(before,after,1)
    return current


def protected_implementations(original:str,candidate:str,path:str)->list[str]:
    """Check that original data/physics/model/class definitions are not drifted."""
    a=ast.parse(original,filename=path+"@frozen")
    b=ast.parse(candidate,filename=path+"@derivative")
    def listed(tree):
        return {n.name:ast.dump(n,include_attributes=False)
                for n in tree.body
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    before=listed(a);after=listed(b)
    require(before.keys()==after.keys(),"LINEAGE_TOPLEVEL_FUNCTION_SET_DRIFT_"+path)
    allowed={"run"} if path in SCIENTIFIC_MODULES else {"EpisodeWorkLedger"}
    unchanged=[]
    for name in before:
        if name not in allowed:
            require(before[name]==after[name],
                    "LINEAGE_PHYSICS_OR_MODEL_FUNCTION_CHANGED_"+path+":"+name)
            unchanged.append(name)
    if path in SCIENTIFIC_MODULES:
        fn=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        first=fn.body[0]
        require(isinstance(first,ast.Expr) and isinstance(first.value,ast.Call),
                "LINEAGE_SCIENCE_ENTRY_NOT_GUARDED_"+path)
        call=first.value
        require(len(call.args)>=2 and
                isinstance(call.args[1],ast.Constant) and
                str(call.args[1].value).startswith(
                    "S"+path.split("/")[1].split("_")[1][1:]+
                    "_FROZEN_RSS_REQUIRES_LINUX"),
                "LINEAGE_GUARD_NOT_FIRST_"+path)
    return unchanged


def verify(path:str,original:str,candidate:str)->dict:
    require(path in DERIVATIVE_BLOBS,"LINEAGE_UNREGISTERED_PATH")
    require(blob_digest(original)==FROZEN_BLOBS[path],
            "LINEAGE_FROZEN_SOURCE_BLOB_CHANGED")
    require(blob_digest(candidate)==DERIVATIVE_BLOBS[path],
            "LINEAGE_DERIVATIVE_BLOB_CHANGED")
    require(expected_candidate(path,original)==candidate,
            "LINEAGE_UNAPPROVED_EDIT_"+path)
    unchanged=protected_implementations(original,candidate,path)
    return {
        "path":path,"frozen_blob":FROZEN_BLOBS[path],
        "derivative_blob":DERIVATIVE_BLOBS[path],
        "allowed_replacements":len(approved_edits(path)),
        "unchanged_top_level_functions_or_classes":unchanged,
        "physics_and_model_code_exactly_preserved":True,
        "original_scientific_source_pin_matches_derivative":False,
    }


def audit(expected_sha:str)->dict:
    require(len(expected_sha)==40 and
            all(x in "0123456789abcdef" for x in expected_sha),
            "LINEAGE_INVALID_EXPECTED_SHA")
    require(git("rev-parse","HEAD")==expected_sha,
            "LINEAGE_HEAD_DRIFT")
    require(git("merge-base","--is-ancestor",V2_PARENT,"HEAD")=="",
            "LINEAGE_NOT_DESCENDED_FROM_QUALIFIED_V2")
    require(not git("status","--porcelain","--untracked-files=all"),
            "LINEAGE_SOURCE_NOT_CLEAN")
    for path,digest in FROZEN_BLOBS.items():
        require(git("rev-parse",f"{FROZEN_BASE}:{path}")==digest,
                "LINEAGE_ORIGINAL_PIN_CHANGED_"+path)
    for path in ("source_pins.json",
                 "experiments/g7_s20_frozen_hidden_multicycle.json",
                 "experiments/g7_s25_frozen_contradiction_ttt.json"):
        require(git("hash-object","--",path)==FROZEN_BLOBS[path],
                "LINEAGE_FROZEN_PROTOCOL_OR_SOURCE_PINS_DRIFT")
    rows=[]
    for path in DERIVATIVE_BLOBS:
        source=subprocess.run(["git","-C",str(ROOT),"show",
                              f"{FROZEN_BASE}:{path}"],check=False,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              timeout=14)
        require(source.returncode==0,"LINEAGE_MISSING_FROZEN_ORIGINAL")
        rows.append(verify(path,source.stdout.decode("utf-8"),
                           (ROOT/path).read_text(encoding="utf-8")))
    return {
        "schema":"ttt-hybrid.g7-s25.versioned-lineage-exact-edits.v1",
        "status":"PASS_EXACT_REGISTERED_PORTABILITY_EDITS_ENGINEERING_ONLY",
        "derivative_candidate_sha":expected_sha,
        "frozen_s25_sha":FROZEN_BASE,
        "qualified_v2_parent_sha":V2_PARENT,
        "proof":rows,
        "frozen_protocols_and_source_pins_unchanged":True,
        "original_science_sources_byte_identical_to_derivative":False,
        "scientific_source_first_use_admissible":False,
        "scientific_training_or_heldout_run":False,
        "forward_model_or_world_equivalence_under_training":"NOT_PROVEN",
        "native_windows_scientific_assay":"BLOCKED_NON_EQUIVALENT_RSS",
        "merge_or_science_promotion":"NOT_AUTHORIZED",
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    output=audit(args.expected_sha)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n",
                            encoding="utf-8")
    print("S25_LINEAGE="+output["status"])
    print("S25_SCIENCE_SOURCE_REPIN=BLOCKED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
