# G7-R12 — isolated WSL2 host C compiler availability: prepared only

**Source baseline:** `fb3c7ff265a15b3d8dda9c9d8797ae202888acb6`; G4 and G5 Linux/Windows + pinned Mamba Ubuntu CPU qualification all PASS on that exact SHA. Original G7-R9 GPU forward [run 42300670454](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454) FAIL; G7-R11 [run 42313100239](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42313100239) SHA256-custodied original `RuntimeError` plus `triton/runtime/build.py` frame confirm `HOST_C_COMPILER_UNAVAILABLE_CONFIRMED`. Permanent proof: `docs/G7_R11_RESULT.md`.

## Minimal next decision

Determine whether an **already installed** compiler is accessible in the **original** WSL2 Ubuntu-24.04 environment, without changing the environment, installing anything or retrying GPU execution. The pinned Triton 3.5.0 host helper code resolves `CC` first, then `gcc` or `clang` on PATH. A compiler may exist as a version-suffixed binary even if neither unversioned command resolves.

A one-shot owner+runner restricted workflow (existing `TTT-HYBRID-GPU-WIN`, main-only push on creation, first run/attempt only) first verifies the unmodified runner workspace's original exact Git SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`. Then it **starts WSL2** for a max-35-second read-only Bash inventory of `command -v gcc/clang/cc`, installed `/usr/bin/gcc-*` and `clang-*`, and whether `CC` is set. It computes SHA-256 for present binaries without executing them. It emits no raw path, environment or credentials. Hashes are provenance only; `PRESENT_UNQUALIFIED` does not prove working compilation. A sanitized JSON is written only to `runner.temp` and uploaded with 30-day retention. If original identity or output fails, stop BLOCKED.

**Authority boundary:** This source-only draft PR is **not** an approval to execute. Merging starts a WSL2 process, so an explicit separate user approval is required. No CUDA, model, compiler invocation, training, package installation, downloads, system PATH edits or original checkout changes. Following a single authorized execution, retire the workflow and preserve the receipt.

If a compiler exists, the next separately authorized gate is compatibility qualification, not another GPU forward by default. If absent, decide whether to authorize a bounded compiler provisioning change; do not silently install packages.

ENGINEERING: root cause IDENTIFIED; current compiler availability UNVERIFIED; authentic GPU forward FAIL. SCIENTIFIC UNVERIFIED. DEPLOYMENT NOT AUTHORIZED.
