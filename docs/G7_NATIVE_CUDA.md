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


## Existing Windows runner route — least-privilege admission only

A **manual-only** workflow `.github/workflows/g7-existing-runner-admission.yml`
now prepares a safe attachment path to an existing Windows runner.
It does **not** register, install, reconfigure, or launch a runner.

**Repo and runner scoping**:

- Repository: `spoonman136668-ai/TTT-Hybrid-Lab` (private).
- Required runner labels: `self-hosted`, `Windows`, `X64`,
  and the **TTT-specific** `ttt-hybrid-gpu-isolated` label.
- DO NOT relabel or broaden the existing FORGE owner runner
  `FORGE-CODEX-REMOTE` / `forge-codex-isolated` to bypass its isolation.
  FORGE can remain running uninterrupted; TTT must have a separately
  granted/qualified runner slot and resource budget.
- The GitHub connection available to this chat cannot enumerate or edit
  GitHub runner group/repository registration settings. No eligible
  TTT-Hybrid runner connection has yet been independently observed.
- **Verified repository ownership (2026-11-05):** both TTT-Hybrid-Lab
  and FORGE are owned by the **personal GitHub account** `spoonman136668-ai`
  (GitHub owner type `User`), not an organization. Organization runner groups
  are **not an available sharing path for these current repos**.
- The existing `FORGE-CODEX-REMOTE` repo-scoped runner must remain unchanged.
  Reusing the **same physical Windows hardware** safely requires a separate
  TTT-Hybrid repository runner **registration** (different directory/work
  root, identity, and `ttt-hybrid-gpu-isolated` label), unless an existing
  TTT-registered runner can first be independently verified. A runner
  registration is **not** a new research controller or scheduler. An
  administrative machine-access route is required for that one-time host
  action; this chat's GitHub connector cannot perform it.
- A physically shared host must have an independent resource/occupancy
  admission before any TTT CUDA workload; the read-only inventory workflow
  alone does not guarantee isolation or non-interference with FORGE.

**Activation boundary**:

1. GitHub branch metadata currently reports `main` **unprotected**
   (verified 2026-11-05). Before any self-hosted job is allowed, a repository
   administrator must establish effective main-branch protection or a ruleset,
   including qualified checks and restricted changes. The workflow and the
   PowerShell probe both **fail closed** unless `github.ref_protected` /
   `GITHUB_REF_PROTECTED` are true. This is not a request to bypass FORGE.
   Then merge and accept the prerequisite G4/G5/G7 source stack under the
   project's repository acceptance policy. `workflow_dispatch` cannot
   be used on an unmerged PR workflow that is absent from the default branch.
2. Verify the resulting exact main SHA, mandatory regression, runner label,
   runner availability, and custody. Approval of the source PR does not
   authorize moving FORGE's existing runner.
3. Once the TTT runner has been attached and scoped, perform one
   owner-initiated GitHub Actions dispatch **on main**. This workflow only
   checks the exact SHA, repo/actor/attempt/OS and local clean Git worktree;
   queries `nvidia-smi`; and reads the runner user's **HKCU registry only**
   for registered distro names. It does NOT execute `wsl.exe` at all:
   `wsl --version` on systems without WSL can unexpectedly launch an
   installer, as observed on the owner's workstation.
4. Preserve `local_runs/g7_runner_inventory.json` as an Actions artifact,
   distinguishing runner-user distro registration from *actual* Linux GPU
   capability. A clean host receipt proves only Windows host admission.
5. Authentic Linux/WSL2 GPU execution remains **separate** and requires
   qualified native CUDA/PyTorch ABI and exact-source checkouts, available
   bounded resources, and explicit scoped execution authority. Do not
   infer scientific improvement or deployment admission from this host probe.

**The workflow is not triggered on PRs, pushes, comments or schedules.**
No untrusted PR code is executed on a self-hosted runner. The sole
`workflow_dispatch` route is restricted to this private repository,
the protected `main` ref, the repository owner, and first attempts.
The hosted Linux/Windows G4/G5 qualification still runs automatically
for source changes. Its PASS is not self-hosted runner admission.


## Reuse one existing registration: Mind-Palace runner (preferred over a seventh runner)

**Verified 2026-11-05:** `spoonman136668-ai/Mind-Palace` owns the
`MIND-PALACE-WIN` Windows self-hosted runner and its native regression has
completed successfully. GitHub documentation confirms that a called reusable
workflow in another repository **under the same user** can schedule a
self-hosted runner available to the **caller repository**, without a new
registration. The called workflow inherits the caller repository's GitHub
context and token; it does **not** implicitly check out TTT-Hybrid.

The **existing** `.github/workflows/g7-existing-runner-admission.yml`
needs a TTT-repository-level runner label and a protected `main`; it is
**not the selected path** when sharing an already registered Mind-Palace
runner. It must not be dispatched to bypass runner isolation.

Instead the new
`.github/workflows/g7-mind-palace-reusable-inventory.yml` is a
**read-only workflow_call**, invoked only from an owner-initiated
`workflow_dispatch` on Mind-Palace `main`. It validates runner identity
`MIND-PALACE-WIN`, exact caller owner/repository/ref, and a syntactically
valid full TTT candidate SHA. It reads only Windows NVIDIA-SMI and
requires at least 4096 MiB reported free VRAM on a single reported GPU.
It **never** invokes WSL, checks out any repository, installs anything,
or executes an actual GPU kernel. The sanitized result is written to the
*Mind-Palace caller workflow log* (no new persistent host artifact is
created). No Mind-Palace checkout, queue state, local models, retained
memory or controller records are changed. This preliminary Windows host
probe is not Linux CUDA or compiled-kernel qualification.

A prospective caller workflow, **not installed/activated by this source
change**, is:

```yaml
name: Owner-initiated TTT host inventory via existing Mind-Palace runner
on:
  workflow_dispatch:
permissions: {}
jobs:
  read-only-ttt-host:
    if: ${{ github.repository == 'spoonman136668-ai/Mind-Palace' && github.ref == 'refs/heads/main' && github.actor == 'spoonman136668-ai' && github.run_attempt == 1 }}
    uses: spoonman136668-ai/TTT-Hybrid-Lab/.github/workflows/g7-mind-palace-reusable-inventory.yml@<EXACT_40_DIGIT_QUALIFIED_TTT_SOURCE_SHA>
    with:
      ttt_candidate_sha: <SAME_EXACT_40_DIGIT_QUALIFIED_TTT_SOURCE_SHA>
```

The placeholders **must be replaced** by the same qualified immutable
40-character G7 candidate; never pin a moving branch. This is an
**operator-independent interface example**, not a run, promotion,
permission grant, approval to edit the Mind-Palace repository, or a
promise to run automatically.

Before an actual call, separately verify:
1. TTT-Hybrid's private-repo **Settings → Actions → General → Access**
   permits reusable workflows to be read by other private repositories
   owned by `spoonman136668-ai`. The current setting could not be
   inspected through this chat's GitHub connector.
2. The Mind-Palace owner approves the bounded caller source change, with
   no change to ordinary Mind-Palace regression jobs or controllers.
3. The existing runner is available and the caller `main` is the exact
   approved source; **never** run the reusable job from pull requests or
   arbitrary code. GitHub reports Mind-Palace `main` currently unprotected,
   so relying on a protected-ref guard would block this route. The gate uses
   owner-only manual dispatch and an immutable called-workflow SHA; that
   does **not** replace branch protection for eventual repository acceptance.
4. No credentials or third-party project files enter the bounded TTT host log.
   The runner shares **physical hardware** with other projects, so CUDA
   kernel allocation, WSL launch, installations and external source fetches
   remain **outside this gate** and require an independently qualified
   project-specific resource and authority decision.

**Source-only qualifier:** G4 hosted Linux/Windows full regression and
negative controls on the *exact final TTT candidate SHA*. The called
workflow's compatibility on `MIND-PALACE-WIN` is **UNVERIFIED** until a
real owner-authorized one-shot caller run produces its scoped log evidence.

Official semantics:
https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations
https://docs.github.com/en/actions/how-tos/reuse-automations/share-across-private-repositories
