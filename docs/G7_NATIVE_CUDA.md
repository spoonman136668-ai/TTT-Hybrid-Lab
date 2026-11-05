# G7 — Native Mamba-2 + TTT-Linear CUDA qualification

**Scope:** An actual official-source CUDA integration smoke for the already
implemented Mamba-2 → causal attention → TTT-Linear → causal threshold depth
adapter. This is **not** a scientific learning experiment or paper-faithful
fixed-capacity Mixture-of-Depths implementation.

## Starting checkpoint and isolation

- G5 qualified parent `15be6540218ee813b33538ce0bf19476c870544e`
  (PR #2, still unmerged), intentionally **not** the G6 science branch.
- Official Mamba source
  `state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`
  with frozen `mamba_ssm/modules/mamba2.py` blob
  `36b16d4715ed2d753697167d65da5e6abba22017`.
- Official TTT-Linear source
  `test-time-training/ttt-lm-pytorch@cd831db10c8c9a0f6340f02da5613316a8a92b67`
  with frozen `ttt.py` blob
  `da267f4a88196fd4709fa1e5e62235a3fa746ec6`.
- Requires a **separately authorized, isolated Linux/WSL2 CUDA host**, actual
  driver/toolkit/native dependencies, free VRAM and verified source origins.
  **No new runner, scheduler, controller, paid GPU service or automatic
  workstation activation was created.**
- Do not consume the current FORGE controller, or Wingless/Yggdrasil
  scientific GPU capacity. The GitHub-hosted Linux/Windows CPU test runners
  are not suitable proof of authentic CUDA execution.

## Environment prerequisite — no arbitrary installation

An operator-qualified CUDA environment must contain CUDA-enabled PyTorch,
working NVIDIA GPU driver/runtime, compatible Triton and PyTorch architecture
support, the pinned official Mamba2 Python source and its native dependencies,
and pinned official TTT-LM Python source and Transformers dependencies.
Versions and compiled-binary hashes must be recorded in the runtime receipt.
Mamba's upstream README documents its source installation options and
`--no-build-isolation` semantics; a package installation **must not** silently
substitute a different source revision. Ensure source checkouts remain clean
after installation; use a separate build directory if necessary.

On an already-qualified Linux/WSL2 environment, with the exact upstream source
checkouts at `/qualified/mamba` and `/qualified/ttt`:

```bash
export PYTHONPATH="/qualified/mamba:/qualified/ttt:${PYTHONPATH:-}"
python scripts/g7_cuda_qualification.py \
  --mamba-root /qualified/mamba \
  --ttt-root /qualified/ttt \
  --min-free-gib 4 --max-peak-gib 2 \
  --expected-sha <exact-qualified-PR-commit-SHA> \
  --receipt local_runs/g7_native_cuda.json
```

The script does not fetch, install, dispatch, train, deploy, or change a ref.

## Engineering acceptance

- Read-only host gate: Linux/WSL2, CUDA availability, at least 4 GiB
  **currently free** and a 2 GiB PyTorch allocator limit. If unavailable,
  status is `BLOCKED_*`; never count this as a GPU PASS.
- Exact clean Git source SHA, file-blob, source-module and official class
  provenance; reject unpinned Python wheels or alternate source trees.
- Record GPU name, compute capability, architecture list, CUDA build,
  PyTorch version, source-module SHA-256, and optional compiled-extension
  origins/hashes. Binary provenance remains `PARTIAL_UNVERIFIED`
  until independent build-and-ABI admission is completed.
- Execute an actual CUDA matmul to catch runtime support errors.
- Delegate to the **existing** authentic G3 adapter
  (`run_gate3.py`) to test actual fused Mamba-2, causal attention,
  real TTT-Linear, and threshold routing; require finite forward/backward
  gradients, per-episode reset, strict prefix invariance, future-suffix
  negative control and measured GPU memory within the preset ceiling.
- Save the raw native G3 subreceipt and its SHA-256; fail closed on
  preflight, CUDA runtime, integration, gradients, causality or resource
  violations. A `PASS_NATIVE_CUDA_ENGINEERING_ONLY` is *not* scientific
  qualification, speedup proof, or deployment approval.

## Qualification available now

The G4 Linux/Windows full regression runs automatically on this PR and
includes `tests/test_g7_cuda_qualification.py` with both positive pure-policy
controls and negative controls for unsupported hosts, insufficient VRAM,
untrusted module source, invalid bounds and absent official checkout.
G5 native TTT CPU regression also runs automatically.

**No GitHub-hosted standard CPU runner can be claimed to have executed this
CUDA gate**. A qualified GPU host and an explicit execution authorization
are prerequisites to actual GPU runtime. This stays blocked until then,
without producing more ZIP downloads.

## Future research after engineering PASS

A real science comparison must separately preregister equivalent training
data, seeds, transfer tasks, compute/parameter/FLOP limits, causal leakage
controls, native Mamba-2 and original MoD alternatives, retention, transfer,
GPU memory and latency. G6's frozen development NO-GO remains a negative
result, not permission to retune the same cohort.


## Read-only Windows GPU inventory (added after original CPU qualification)

Before attempting a new GPU setup or installing anything, check whether the
existing machine can host an isolated WSL2 CUDA test. An intentionally small
`scripts/g7_readonly_host_probe.ps1` is now kept in the same repository.

It only queries Windows `nvidia-smi`, `wsl --version`, `wsl --list
--verbose`, and the local project's git HEAD/status. It **does not** launch a
WSL distribution, reserve VRAM, install a driver, create a VM, start a model,
dispatch a workflow or touch any other project.

From an already-updated, clean Windows checkout:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\g7_readonly_host_probe.ps1 -ProjectRoot 'C:\Research\TTT-Hybrid'
```

G7 is still stacked on G5 and unmerged. If the branch is not checked out,
read this script in the GitHub PR before bringing it into the local checkout.
Do **not** run a remote script via an unreviewed `Invoke-Expression` pipeline.

Interpretation: detection of a Windows GPU and a WSL2 distro alone is
**not CUDA qualification**. An isolated WSL2 runtime with pinned official
Mamba2 and TTT sources, compatible CUDA PyTorch, verified native dependencies,
and sufficient free VRAM is still required. Ask for a separate execution
decision before starting/installing a WSL distro or running GPU kernels.


## Candidate identity admission

The native GPU runner now **requires** an exact, full 40-digit qualified
project commit SHA, supplied through `--expected-sha` or
`EXPECTED_CANDIDATE_SHA`. It rejects missing/invalid/mismatched identity
and conflicting CLI/environment identities before any CUDA operation.
The receipt records both observed and required SHAs and the admission decision.
A different PR or merge commit needs fresh exact-head qualification; previous
CPU PASS evidence cannot be recycled against a changed tree.
