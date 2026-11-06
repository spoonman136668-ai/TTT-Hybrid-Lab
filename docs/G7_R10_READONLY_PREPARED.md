# G7-R10 — historical preparation for read-only G7-R9 subreceipt diagnosis

**COMPLETED 2026-11-06:** Owner approved PR #25 merge; run #42302765535 succeeded, recovered exact hashed G3 evidence; the one-use diagnostic is retired. See [G7_R10_RESULT.md](G7_R10_RESULT.md) for the actual result. All below describe the earlier pre-execution scope, not current state.

**Parent main** `d59c024ec0eb9ea7e0383a0a6d608b3480a32b91` (G7-R9 cleanup completed, exact-main 5/5 required CI PASS). **Candidate PR does not execute the read-only runner job.**

## Existing evidence

- G7-R9 original GPU execution [run 42300670454](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454): commit `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`, authentic non-fused Mamba2+TTT native integration **FAIL** with original `RuntimeError`, sanitized category `TRITON_OR_KERNEL_COMPATIBILITY`.
- [Artifact ID 13543572395](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454/artifacts/13543572395), ZIP SHA-256 `c08de830e8d2cbd672bdbdf805ed8bd86c23865a6478b1b5c9f6411c4b9e533b` (expires after GitHub retention).
- Exact native inner receipt SHA-256 `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`, nested G3 receipt SHA-256 `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87`, frozen R3 source pins hash `44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768`.

## This gate

One-time owner-scoped original error receipt recovery on the **existing** `TTT-HYBRID-GPU-WIN` runner, with NO checkout, WSL, GPU, torch import, model execution, dependency install, new runner, controller or update to unrelated projects. The workflow checks the existing workspace original Git SHA before opening the two known local receipt files. It validates independent and nested SHA-256 custody, then records only allowlisted exception type, error signature, module and recorded SHA identities in a bounded sanitized JSON artifact. Missing/overwritten receipts return BLOCKED, not inferred diagnoses. The historical `g7_r6_native_internal.json` filename is intentional: it is the fixed filename written by the G7-R9 wrapper, though the SHA and contents correspond to G7-R9.

Single-use trigger only on owner `main` push that introduces this workflow file, with first run and first attempt guards; no `workflow_dispatch`, PR, or schedule trigger. Read-only operation runs max 4 minutes. Artifact stored for 30 days. After a separately approved execution, retire the consumed workflow and preserve exact evidence; no retry.

## Authority boundary

This pull request is **source and CPU CI qualification only**. **Merging this PR will cause a runner-side PowerShell command to execute and therefore requires separate user approval.** The user's prior authorization for G7-R9 GPU forward does not authorize this new diagnostic job. Do not merge or activate based solely on CI PASS.

ENGINEERING: original GPU FAIL, error mechanism pending. SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED. No further GPU run, installation or dependency change.
