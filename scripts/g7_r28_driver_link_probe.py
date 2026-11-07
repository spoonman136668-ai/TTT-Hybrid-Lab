"""G7-R28: one bounded, nonroot CUDA driver C-link diagnostic.
No GPU kernel/model, package installation, persistent symlink or system changes.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

result = {
    "schema": "ttt-hybrid.g7-r27.python-link-probe.v1",
    "status": "BLOCKED_NOT_STARTED",
    "uid": os.geteuid(),
    "gcc_real_executable": "UNVERIFIED",
    "cuda_library_presence": {},
    "probes": [],
}
def classify(stderr):
    data = (stderr or "").lower()[:4096]
    tags = []
    rules = {
        "CANNOT_FIND_LIBCUDA": ("cannot find -lcuda",),
        "MISSING_CRT": ("cannot find crti.o", "cannot find crtn.o"),
        "MISSING_HEADER": ("fatal error:", "no such file or directory"),
        "UNDEFINED_SYMBOL": ("undefined reference", "undefined symbol"),
        "PERMISSION": ("permission denied",),
        "DISK_FULL": ("no space left on device",),
    }
    for label, patterns in rules.items():
        if any(p in data for p in patterns):
            tags.append(label)
    return tags or ["OTHER_OR_NO_LINKER_STDERR"]
def probe(name, compiler, extra, root):
    target = root / (name + ".so")
    args = [compiler, "-x", "c", "-", "-shared", "-fPIC",
            "-o", str(target), *extra]
    try:
        p = subprocess.run(args, input="int g7_r27(void){return 1;}\n",
                           text=True, capture_output=True, check=False,
                           timeout=25)
        ok = p.returncode == 0 and target.is_file() and target.stat().st_size > 0
        result["probes"].append({
            "name": name, "outcome": "PASS" if ok else "FAIL",
            "return_code": p.returncode if p.returncode in range(0, 256) else None,
            "stderr_markers": [] if ok else classify(p.stderr),
        })
        return ok
    except (OSError, subprocess.TimeoutExpired) as e:
        result["probes"].append({"name": name, "outcome": "BLOCKED",
                                 "error_class": type(e).__name__})
        return False
try:
    if os.geteuid() != 1000:
        raise RuntimeError("UNEXPECTED_UID")
    gcc = Path("/usr/bin/gcc")
    if not gcc.is_file():
        raise RuntimeError("GCC_UNAVAILABLE")
    real = gcc.resolve().name
    if not re.fullmatch(r"[A-Za-z0-9._+-]{1,80}", real):
        raise RuntimeError("UNEXPECTED_REAL_COMPILER_NAME")
    result["gcc_real_executable"] = real
    locations = {
        "wsl_libcuda_so": Path("/usr/lib/wsl/lib/libcuda.so"),
        "wsl_libcuda_so1": Path("/usr/lib/wsl/lib/libcuda.so.1"),
        "cuda_stub_so": Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
    }
    result["cuda_library_presence"] = {
        name: path.is_file() for name, path in locations.items()
    }
    with tempfile.TemporaryDirectory(prefix="ttt-g7-r27-link-") as directory:
        root = Path(directory)
        if not probe("baseline_host_shared", str(gcc), [], root):
            raise RuntimeError("BASE_SHARED_LINK_FAILED")
        default = probe("cuda_default_link", str(gcc), ["-lcuda"], root)
        if default:
            result["status"] = "PASS_CUDA_DEFAULT_LINK"
        else:
            wsl = probe("cuda_explicit_wsl_path", str(gcc),
                        ["-L/usr/lib/wsl/lib", "-lcuda"], root)
            stub = None
            if locations["cuda_stub_so"].is_file():
                stub = probe("cuda_explicit_stub_path", str(gcc),
                             ["-L/usr/local/cuda/lib64/stubs", "-lcuda"], root)
            temporary_link = None
            if locations["wsl_libcuda_so1"].is_file():
                linkdir = root / "wsl-link"
                linkdir.mkdir()
                (linkdir / "libcuda.so").symlink_to(locations["wsl_libcuda_so1"])
                temporary_link = probe("cuda_temporary_linker_symlink", str(gcc),
                                      ["-L" + str(linkdir), "-lcuda"], root)
            result["status"] = (
                "PASS_ENVIRONMENT_ONLY_CUDA_LINK_ROUTE"
                if any(v is True for v in (wsl, stub, temporary_link))
                else "BLOCKED_NO_AVAILABLE_CUDA_LINK_TARGET"
            )
except Exception as exc:
    result["status"] = "BLOCKED_" + type(exc).__name__
print(json.dumps(result, sort_keys=True))
