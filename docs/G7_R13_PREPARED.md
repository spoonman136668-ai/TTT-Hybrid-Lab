# G7-R13 — historical preparation, diagnostic consumed

**EXECUTED 2026-11-06 / GitHub UTC 2026-11-07:** PR #31 merged after user approval. The only [G7-R13 run #42316405581](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42316405581) finished with `BLOCKED_NATIVE_INVOCATION_EXCEPTION` at `WSL_PING`. The workflow was retired via source-only cleanup. See [G7_R13_RESULT.md](G7_R13_RESULT.md) for the authoritative observed result; the preparation scope below is historical.

**Baseline** `c0d743ca0174b9baf26831644952ed8bda7079ce`, full G4/G5 Linux/Windows and pinned Mamba Ubuntu CPU **5/5 PASS**. G7-R11 proved original G7-R9 native GPU failure due to `HOST_C_COMPILER_UNAVAILABLE_CONFIRMED`, not a confirmed architectural incompatibility. G7-R12 [run #42314692869](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42314692869) failed at `BLOCKED_PROBE_UNVERIFIED`, although original G7-R9 source checkout identity was bound; `gcc/clang` current presence remains unverified.

This candidate repairs the **diagnostic transport**, not the model. One owner-activated single-use workflow on isolated `TTT-HYBRID-GPU-WIN` checks original exact G7-R9 commit `cf207e3d2d3e02b2803548a0ebc74e3bd6511744` in untouched original worktree. It distinguishes missing `wsl.exe`, WSL first-command invocation, marker failure, Bash inventory invocation/exit, output schema and compiler provenance. PowerShell native-error continuation is limited to each WSL subprocess and restored in `finally`. It records only sanitized status, stage, numeric exit codes/line counts, and compiler availability/hash codes; no raw stderr, environment paths or contents. Compiler binaries are never executed.

**No CUDA, native forward, training, installer, downloads, compiler invocation, checkout, PATH changes, new runner/controller or duplicate workflow.** Single owner `main` push when this workflow file first lands, `run_number == 1`, `run_attempt == 1`, 4-minute timeout, 30-day bounded artifact. The original R12 workflow was retired.

**Execution boundary:** PR is DRAFT and must not be merged without a fresh explicit approval, because merging triggers a new WSL2 command. Exact-candidate G4/G5 Linux/Windows and pinned Mamba CPU tests are required before execution. Read-only presence is not proof of compiler compatibility. After one authorized use, retire the workflow.

ENGINEERING: original native GPU forward FAIL, original cause VERIFIED, current compiler availability UNVERIFIED. SCIENTIFIC UNVERIFIED; DEPLOYMENT NOT AUTHORIZED.
