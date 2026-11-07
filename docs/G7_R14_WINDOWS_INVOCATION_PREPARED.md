# G7-R14 — hosted-Windows native command invocation qualification (no WSL)

**Baseline** `4f66973f58470acfc4044a7114b55896d468e606`, five mandatory CPU CI checks PASS. Previously authorized G7-R13 [run #42316405581](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42316405581) failed with `CommandNotFoundException` at `WSL_PING`, before the first child exit code; exact cause unresolved. Historical G7-R9 authentic GPU forward FAIL from unavailable host compiler at original Triton helper build, while current WSL2 compiler availability is UNVERIFIED.

## Falsifiable source/hosted-CI test
Use *only* the existing Windows-hosted G4 pytest suite, not `TTT-HYBRID-GPU-WIN`. A new Python test invokes a standalone PowerShell script that executes **only trusted `cmd.exe`** with harmless echo markers. Compare (a) legacy `& $app.Source` expression to (b) validated rooted filesystem path `[string]$app.Definition` executed with `& $exe`. Test a deterministic absent-executable negative control and require `CommandNotFoundException`; do not invoke it successfully. Emit only categorical statuses into `local_runs/g7_r14_windows_native_launcher.json`, retained in existing G4 Windows artifact.

**Pass criterion:** explicit-path call succeeds, missing executable fails closed. Legacy method is **observed**, not predetermined; if legacy fails and explicit passes, original call syntax is implicated; if both pass, the isolated WSL invocation still needs separate diagnosis. This is not a model test or a compiler usability test.

No WSL startup, CUDA, GPU, compiler invocation, installation, new runner, scheduler/controller or workflow. No privilege beyond existing GitHub-hosted CI; no original workstation actions. Source/CPU test evidence alone does not authorize another isolated-runner WSL invocation. Separately approve such a check if needed, after exact-candidate qualification.

ENGINEERING: synthetic Windows command semantics pending; authentic GPU FAIL; current compiler/WSL UNVERIFIED. SCIENTIFIC UNVERIFIED; DEPLOYMENT NOT AUTHORIZED.
