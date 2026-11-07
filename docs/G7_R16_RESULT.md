# G7-R16 — Ubuntu root-context GCC successfully installed; host C shared build FAILED

**Execution:** 2026-11-06 Eastern / 2026-11-07 UTC. **ENGINEERING:** exact candidate CPU CI 5/5 PASS, Ubuntu root GCC package installation PASS, host C shared-library compilation **FAIL**, authentic GPU forward **NOT RUN**. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**

## Authority and exact execution

- User expressly approved one root-context GCC-only installation and one bounded unprivileged authentic Mamba2+TTT forward if admitted. Original G7-R15 [run #42321056057](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42321056057) was BLOCKED by `sudo -n`, not a GPU kernel.
- [PR #36](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/pull/36) exact candidate `ecf6433ba6274b2138fb256f1663019b761dc012` qualified **5/5 mandatory G4/G5 Linux+Windows and pinned Mamba Ubuntu CPU**; merged at `4a32444ed9ce3b8246aa8b01e41c9ce83f8a5dee`.
- First and only [G7-R16 Actions run #42323786912](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42323786912), `run_number=1`, `run_attempt=1` on existing `TTT-HYBRID-GPU-WIN`. Exact checkout PASS; GCC root install executed; original unprivileged G7-R15 compiler and forward admission **FAIL**; artifact upload PASS. No retries.
- [Sanitized artifact ID 13554520874](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42323786912/artifacts/13554520874), ZIP SHA256 `179ab666e1d4368623eb22748b5cd13fe5c59352ff22e0c559e2e0986f07c19d` independently matched GitHub upload log. Included `g7_r16_root_gcc_forward.json` SHA256 `e24e53005e4afd9328b641eef2d58c9f8734080b148460a3eb7d2c5f6673fea9`; `g7_r15_compiler_forward.json` SHA256 `c769c5b6c5ec7d564caf557e5d335dbc647c3b778504b88ac4f255f49719b6d2`. All receipts are sanitized; neither original traceback nor raw GPU data uploaded.
- Repo/source-pins/R3 environment checks as normal WSL user: `source_custody=PASS`. Original G7-R9 root cause `HOST_C_COMPILER_MISSING` was independently identified in G7-R11 receipt.

## Observed sequence and stop

- Initial compiler probe under original unprivileged Ubuntu user: `compiler_before=ABSENT_IN_CHECKED_DISTRO_PATHS`.
- Explicit fixed Ubuntu-24.04 root user validation: `root_context=ROOT_UID_ZERO_CONFIRMED`.
- `apt_update=PASS`; `apt_install=PASS` with one `apt-get install -y --no-install-recommends gcc`. Log confirms Ubuntu GCC 13 (13.3.0-6ubuntu2~24.04.1) plus package-required dependencies installed. No Torch, Triton, CUDA driver, Python or model packages changed.
- Subsequent normal-user probe: `compiler_after=PRESENT_UNQUALIFIED`. This establishes existence of a distro compiler, **not** completed host-C compilation.
- Existing G7-R15 admission after normal-user source check: `source_custody=PASS`, `provisioning=SKIPPED_EXISTING_COMPILER`. First compilation test returned `status=BLOCKED_HOST_C_SHARED_BUILD_FAILED`; G7-R16 outer `status=BLOCKED_NATIVE_FORWARD_OR_HOST_COMPILER_PROBE`.
- G7-R15 inner: `compiler=null`, `native_forward=NOT_RUN`, `native_gate3_status=null`, `native_receipt_sha256=null`, `forward_checks=null`. No GPU forward was launched; optimizer, gradient, training, science qualification and deployment NOT RUN / NOT AUTHORIZED.

## Narrow technical hypothesis — not proof

The failing source probe asks GCC to compile a trivial C snippet into a shared `.so` with `-shared -fPIC`, and hides compiler stderr. Therefore **the actual linker error is UNVERIFIED**, and it is premature to claim GCC is unusable or that any GPU kernel is incompatible. One *specific testable hypothesis* is that `libc6-dev` (the GNU C runtime linker/development files) is absent: Ubuntu 24.04's official `gcc-13` package **recommends**, rather than depends on, `libc6-dev`; this workflow intentionally used `--no-install-recommends`. Package selection logs did not show libc6-dev installed in this attempt. That is a **hypothesis, not an independently observed root cause**; libc6-dev might already have existed or the compiler might have failed for a different reason.

References: [Ubuntu gcc-13 package dependencies/recommendations](https://packages.ubuntu.com/noble/gcc-13), [Ubuntu libgcc-13-dev recommendations](https://packages.ubuntu.com/noble/libgcc-13-dev).

**NO RETRY.** The owner-authorized single-use workflow is retired in source-only cleanup. The next independent task is a bounded diagnostic of GCC's sanitized failure reason plus precise `libc6-dev` package status (without installing any new package or GPU execution), or a **new explicit approval** for a constrained libc6-dev provisioning change and subsequent separately authorized GPU forward. Do not silently expand root privileges/packages based on this hypothesis.

ENGINEERING: original G7-R9 failed because compiler missing; G7-R16 installed GCC but host shared C compilation FAIL; GPU recovery UNVERIFIED. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**
