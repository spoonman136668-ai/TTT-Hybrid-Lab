"""Read-only isolated G7-S25 CPU runtime and R3 provenance inspection.

No installation, torch model forward, heldout seeds, write, or science.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import sys


def cpu():
    import torch
    import pytest
    try:
        version=metadata.version("numpy")
    except metadata.PackageNotFoundError:
        version=None
    report={
        "schema":"ttt-hybrid.g7-s25.portability-cpu311-probe.v1",
        "python":list(sys.version_info[:3]),
        "torch":str(torch.__version__),
        "torch_cuda":torch.version.cuda,
        "pytest":str(pytest.__version__),
        "numpy":version,
        "science":"NOT_RUN",
    }
    if report["python"]!=[3,11,14] or report["torch"]!="2.5.1+cpu" or (
        report["pytest"]!="8.3.4" or report["torch_cuda"] is not None or
        report["numpy"] not in (None,"1.26.4")
    ):
        raise RuntimeError("PORTABILITY_CPU_RUNTIME_VERSION_DRIFT")
    return report


def r3():
    records=sorted((
        str(d.metadata.get("Name","")).lower().replace("_","-"),
        str(d.version),
    ) for d in metadata.distributions())
    digest=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    if digest!="b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df":
        raise RuntimeError("PORTABILITY_R3_PACKAGE_MANIFEST_CHANGED")
    return {"schema":"ttt-hybrid.g7-s25.portability-r3-probe.v1",
            "r3_manifest_sha256":digest,"science":"NOT_RUN"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("cpu","r3"),required=True)
    args=parser.parse_args()
    print(json.dumps(cpu() if args.mode=="cpu" else r3(),sort_keys=True))


if __name__=="__main__":
    main()
