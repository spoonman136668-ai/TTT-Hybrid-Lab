# G7-R13 — read-only WSL2 compiler transport: BLOCKED at WSL_PING

**Date:** GitHub UTC 2026-11-07 (US Eastern 2026-11-06). **ENGINEERING:** source/CPU qualification PASS, one-shot WSL diagnostic **FAIL**. Original G7-R9 GPU forward remains **FAIL**. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**

## Exact authorization and original evidence

- Explicit user authorization for PR #31 only. [PR #31](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/pull/31) exact candidate SHA `c9d35baf014844dd5c8d4923569e014b800cd116`, five required G4/G5 Linux+Windows and Mamba Ubuntu CPU checks **PASS**; merged at SHA `030e381ad4894d5e1c13a1af28f7df369ce363e9`.
- [GitHub Actions run #42316405581](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42316405581), `run_number=1`, `run_attempt=1`, isolated Windows runner; the `inspect` job's read-only inventory step **FAIL**, sanitized artifact upload **PASS**. No retry.
- [Artifact ID 13551375790](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42316405581/artifacts/13551375790) ZIP SHA256 `9f4060bd68f92d806263aa98f3e2644fa232fa69af47739444c474dc1401e1da` verified against GitHub upload log. Included `g7_r13_readonly_compiler_inventory.json` SHA256 `96fc4d79fb153a2cdfddf00195c877c68ff9beecce0b6bbd3020789664f69768` (1294 bytes). Artifact is sanitized; no original WSL output or environment paths were exported.
- Receipt `source_bound=true` against the original G7-R9 forward SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`. Historical original failure remains `HOST_C_COMPILER_UNAVAILABLE_CONFIRMED` at `TRITON_HOST_C_HELPER_BUILD` per authenticated [G7-R11 run #42313100239](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42313100239).

## Observed G7-R13 result

- `status=BLOCKED_NATIVE_INVOCATION_EXCEPTION`
- `probe_phase=WSL_PING`
- `transport_exception_type=CommandNotFoundException`
- `wsl_started_for_inventory_only=true` **means invocation attempted, not successful startup**.
- `ping_exit_code=null`, `ping_output_count=null`, `inventory_exit_code=null`, `inventory_output_count=null`. The first startup marker did not return a captured child-process exit code.
- `gcc=UNVERIFIED`, `clang=UNVERIFIED`, `cc=UNVERIFIED`, `versioned_gcc=UNVERIFIED`, `versioned_clang=UNVERIFIED`, `cc_environment=UNVERIFIED`. Compiler SHA256 values are all null. The **current compiler availability remains UNVERIFIED**.
- `cuda=NOT_STARTED`, `model=NOT_RUN`, `training=NOT_RUN`, `installation=NOT_AUTHORIZED`, `deployment=NOT_AUTHORIZED`. No compiler installation, CUDA, or model execution was performed.

## Mechanistic interpretation and next substantive gate

The first Windows-side WSL invocation threw a PowerShell `CommandNotFoundException` *before* the diagnostic recorded a `ping_exit_code`. This discriminates the failure from Bash inventory parsing or a proven missing WSL compiler. The preflight `Get-Command -Name wsl.exe -CommandType Application` did not trigger its `BLOCKED_WSL_EXE_ABSENT` branch, but the call site `& $wsl.Source` may still be faulty; raw diagnostics were suppressed, so exact executable resolution is **not proven**. A child WSL process starting, succeeding or failing is not established. The captured error does **not** demonstrate absence of GCC/Clang, inability of Ubuntu to launch, or NVIDIA kernel incompatibility.

**NO RETRY.** The single-use `.github/workflows/g7-r13-one-shot-compiler-inventory.yml` is removed in this source-only cleanup. The smallest useful successor is source-only validation of Windows PowerShell `Get-Command` application path and call operator semantics, plus a distinct sanitized code for launch-target resolution; only after that may a separately authorized read-only original-runner check be considered. No automatic runner launch, compiler installation or GPU test under this scope.

**ENGINEERING:** original G7-R9 failure mechanism historically IDENTIFIED; current WSL invocation/compilers UNVERIFIED, G7-R13 transport FAIL. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**
