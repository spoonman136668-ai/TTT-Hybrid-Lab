# G4 — Automated CPU engineering qualification

This repository now runs its **existing** 23-test engineering suite and an
intentional authentic-backend admission-negative control on GitHub-hosted Linux
and Windows after every pull request and main-branch push. No local manual
workflow dispatch, zip download, self-hosted runner allocation, or external
project controller is needed for this CPU gate.

## Scope

- Baseline starting commit: `3c3d58f0f2e670aa7e640a656224e2e59caf5ca4`.
- Runner: GitHub-hosted `ubuntu-latest` and `windows-latest`; Python 3.11,
  `torch==2.5.1+cpu`, `pytest==8.3.4`.
- Actions are pinned by exact commits. No automatic promotion or merge.
- `scripts/qualify_cpu.py` runs the entire `tests` directory. It then calls
  `run_gate3.py` with deliberately nonexistent upstream source roots and
  requires exit code 2, status `BLOCKED_SOURCE_RUNTIME_ADMISSION`, and
  `authentic_forward == NOT_RUN`. A missing negative-control receipt, a
  test failure, or unexpected pass fails the job.
- Each job produces `local_runs/cpu_qualification.json`, records its exact
  candidate SHA, pinned source-manifest hash, dependency version, timing,
  platform, test output, and admission-negative-control result. Artifacts are
  stored in the GitHub Actions run for 30 days.
- Local scratch receipts stay ignored from Git. Published immutable run
  references/SHAs, not hand-written PASS claims, are qualification evidence.

## What PASS does NOT mean

CPU engineering tests do not establish authentic Mamba2/TTTLinear CUDA
compatibility, learning improvement, causal reliability of the native kernels,
speedup, fixed-capacity Mixture-of-Depths implementation, or scientific
acceptance. The original Gate 3 CUDA gate remains blocked until a separately
qualified Linux/WSL2 GPU runner with exact-source checkouts and compatible
native extensions is available.

## Local repository

The operator already has a checkout at `C:\Research\TTT-Hybrid`. This PR
changes the remote engineering branch only. It does **not** write to the
operator's Windows filesystem. The checkout can be updated with an ordinary
`git pull --ff-only` after merge, or by a separately authorized native
checkout-sync mechanism if one is later qualified. Do not create a competing
runner/controller or interfere with active FORGE/Wingless/Yggdrasil loops.

## Next gate

After Linux and Windows CPU checks pass for the exact PR head:
1. Verify the immutable GitHub Actions run(s) and complete negative receipts.
2. Identify an independently available, isolated CUDA environment with no
   contention against active science/engineering loops.
3. Install the official pinned upstream code **in that environment only** and
   run `run_gate3.py` with exact-root source admission.
4. Save authenticated GPU/VRAM, dependency build, trace and invariant receipts
   before any matched-budget experiment. Keep SCIENTIFIC_QUALIFICATION=UNVERIFIED
   until preregistered comparative experiments and independently held-out
   evaluation complete.
