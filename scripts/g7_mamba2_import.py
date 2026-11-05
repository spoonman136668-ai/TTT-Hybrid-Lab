"""G7 pre-GPU check: import and construct the exact official Mamba2 module.

This deliberately does NOT execute Mamba2 kernels, CUDA, or training.
Passing narrows the install/API compatibility risk before an authorized GPU run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mamba-root",type=Path,required=True)
    p.add_argument("--receipt",type=Path,default=ROOT/"local_runs"/"g7_mamba_import.json")
    args=p.parse_args()
    root=args.mamba_root.resolve()
    receipt=args.receipt.resolve()
    receipt.parent.mkdir(parents=True,exist_ok=True)
    pins_raw=(ROOT/"source_pins.json").read_bytes()
    pins=json.loads(pins_raw)
    pin=pins["sources"]["mamba2"]
    record={
        "schema":"ttt-hybrid.g7-mamba-import.v1",
        "captured_utc":datetime.now(timezone.utc).isoformat(),
        "source_pins_sha256":hashlib.sha256(pins_raw).hexdigest(),
        "required_mamba_sha":pin["sha"],
        "platform":platform.platform(),
        "authentic_cuda_forward":"NOT_RUN",
        "scientific_qualification":"UNVERIFIED",
        "status":"FAIL_OR_BLOCK",
    }
    code=1
    try:
        proc=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],capture_output=True,text=True,timeout=15,check=False)
        actual=proc.stdout.strip() if proc.returncode==0 else None
        required=os.getenv("EXPECTED_CANDIDATE_SHA")
        record["project_head"]=actual
        record["expected_head"]=required
        if not actual or len(actual)!=40 or (required and actual!=required) or (not required and os.getenv("GITHUB_ACTIONS")=="true"):
            raise RuntimeError("PROJECT_IDENTITY_DRIFT")
        # A pinned checkout is admitted before importing its Python modules.
        from authentic_preflight import source_check
        result=source_check(root,pin)
        record["source_admission"]=result
        if not result["pass"]:
            raise RuntimeError("PINNED_MAMBA_SOURCE_BLOCKED")
        # Only after checking source custody, allow its imports.
        sys.path.insert(0,str(root))
        import torch
        import triton
        from mamba_ssm.modules.mamba2 import Mamba2
        class_origin=Path(inspect.getfile(Mamba2)).resolve()
        expected=(root/"mamba_ssm/modules/mamba2.py").resolve()
        record["origin"]=str(class_origin)
        if class_origin!=expected:
            raise RuntimeError("MAMBA_CLASS_ORIGIN_MISMATCH")
        torch.manual_seed(90210)
        module=Mamba2(d_model=64,d_state=64,headdim=64,chunk_size=64,use_mem_eff_path=True)
        params=sum(p.numel() for p in module.parameters())
        if params<=0 or module.d_model!=64 or module.headdim!=64 or module.chunk_size!=64 or not module.use_mem_eff_path:
            raise RuntimeError("MAMBA_CONSTRUCTOR_INVARIANT_FAILED")
        if any(p.device.type!="cpu" for p in module.parameters()):
            raise RuntimeError("UNEXPECTED_CUDA_EXECUTION")
        record["versions"]={"python":sys.version,"torch":torch.__version__,"triton":triton.__version__,
                            "cuda_build":torch.version.cuda}
        record["constructor"]={"parameters":params,"d_model":module.d_model,
                               "d_state":module.d_state,
                               "chunk_size":module.chunk_size,
                               "headdim":module.headdim,
                               "use_mem_eff_path":module.use_mem_eff_path}
        record["status"]="PASS_PINNED_MAMBA_CPU_IMPORT_ONLY"
        code=0
    except Exception as exc:
        record["error_type"]=type(exc).__name__
        record["error_message"]=str(exc)[:600]
        record["traceback_tail"]=traceback.format_exc().splitlines()[-12:]
    finally:
        receipt.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"status":record["status"],"project_head":record.get("project_head"),
                          "constructor":record.get("constructor"),"error":record.get("error_message"),
                          "receipt":str(receipt)}))
    return code

if __name__=="__main__":
    sys.exit(main())
