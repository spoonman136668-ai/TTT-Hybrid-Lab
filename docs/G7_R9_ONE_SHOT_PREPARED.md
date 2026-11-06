# G7-R9 — prepared, unexecuted non-fused CUDA forward qualification

**Authority:** G7-R8 source-only PR #22 is merged at `b33fa7ba09f65d7524620e5515e8f00f852cea29`, with all five post-merge G4/G5/Mamba CPU checks PASS. The user has **not** yet approved a second GPU execution. This is *preparation and exact-source qualification only*; a merge of this workflow-containing PR is a separate runtime authorization because it triggers a one-time GPU job.

## Test hypothesis

Selecting the official pinned Mamba-2 non-fused path in `run_gate3.py` may avoid the original G7-R6 `FUSED_CAUSAL_CONV1D_NOT_CALLABLE` TypeError. G7-R7 recovered that original error with verified receipt custody. This does not establish that the unfused Triton scan/TTT-Linear integration will execute successfully.

## Execution after separately authorized acceptance

- Existing isolated `TTT-HYBRID-GPU-WIN` runner only; no new controller, runner, installation, downloads, credentials, trainer, or other project.
- Exact accepted main SHA and clean checkout; R3 local source checkouts and original Git blob pins; frozen `torch==2.9.0+cu128`, `triton==3.5.0` and original package manifest SHA-256.
- Windows owner/ref/event/runner gates, one-time first GitHub run and attempt; no other trigger. WSL2 single GPU sm_120, min 4 GiB free, max PyTorch allocator 1 GiB (not a global extension allocation cap).
- 175 s child, 240 s shell timeout, 7-minute job; no retries; 2×16×64 forward-only, finite output, episode reset, prefix/future-suffix causal checks, no gradient/optimizer/training.
- Only one sanitized outer JSON artifact uploaded with inner receipt digest, selected native `mamba_execution_path`, categorized failure type (if any), no raw exception text or traceback. Retain 30 days; retire workflow after one attempt, preserving receipt evidence.
- Explicit negative controls reject a false PASS when the fused route was accidentally selected.

**State:** ENGINEERING qualified source pending candidate CI, native GPU UNVERIFIED. SCIENTIFIC UNVERIFIED. DEPLOYMENT NOT AUTHORIZED. **Do not merge/run without a separate user approval.**
