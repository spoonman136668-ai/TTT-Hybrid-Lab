# G7-R9 — single non-fused CUDA forward attempt: FAILED

**2026-11-06. ENGINEERING:** native integration FAIL. **SCIENTIFIC:** UNVERIFIED. **DEPLOYMENT:** NOT AUTHORIZED.

## Qualified source and execution

- PR [#23](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/pull/23) merged at exact commit `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`, after 5/5 exact-candidate CPU CI PASS at `9944530a5b1e9866cd978f932466eb9124e0162d`. The owner explicitly authorized one bounded GPU forward-only attempt on 2026-11-06.
- One-time [GitHub Actions run #42300670454](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454), run_number=1/attempt=1, used the existing `TTT-HYBRID-GPU-WIN` isolated runner. Checkout PASS; bounded forward step FAIL; sanitized artifact upload PASS. GitHub workflow conclusion **failure**, no re-run.
- Sanitized artifact [ID 13543572395](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454/artifacts/13543572395), 30-day retention. Downloaded ZIP SHA-256 `c08de830e8d2cbd672bdbdf805ed8bd86c23865a6478b1b5c9f6411c4b9e533b`; contained `g7_r9_nonfused_forward.json` SHA-256 `7956fbd8544f0ef7a5914d697fbc26f56279d5b100ec2caa9668b86cee3de0ee`. Independent hash check matched GitHub job upload log.

## Original sanitized results (no invented cause)

- `status=BLOCKED_NATIVE_FORWARD_NOT_QUALIFIED`, `native_status=FAIL_NATIVE_GATE3_INTEGRATION`, `native_gate3_status=FAIL_AUTHENTIC_GPU_INTEGRATION`.
- **Runtime exception:** `RuntimeError`; sanitized heuristic category `TRITON_OR_KERNEL_COMPATIBILITY`. **Precise exception text, failing operation and root cause remain UNVERIFIED.** Categorization does not prove a Triton version bug, GPU driver mismatch, or required installation.
- `native_mamba_execution_path=PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED` matched the required route (selection verified). This is evidence that the fused path was not requested, **not evidence that the non-fused kernel completed**.
- Correct project SHA, exact official Mamba and TTT source SHAs and blobs, clean source trees, frozen R3 package manifest: PASS. `native_admission_reason=HOST_ADMITTED`; device `NVIDIA GeForce RTX 5070 Ti Laptop GPU`, capability `[12,0]`.
- Outer original native receipt SHA-256 `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`; nested G3 receipt SHA-256 `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87`. Raw G3 diagnostic not uploaded. Forward finite, suffix negative control, prefix invariance, episode reset: **all NOT REACHED**. Peak GPU allocation not recorded.
- No backward, optimizer/training, model-quality measurement, learning/generalization evidence, promotion, installation or resource-expanding work was authorized or reported.

## Disposition and next decisive question

**NO AUTOMATIC RETRY.** The consumed workflow `.github/workflows/g7-r9-one-shot-nonfused.yml` is removed by source-only cleanup, and this artifact is the audit evidence.

Next diagnosis requires the existing local G3 subreceipt from this **same original** SHA, verified against nested `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87` and native `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`; capture only sanitized exception type and whitelisted module/category without copying raw traceback/paths. The limited current signature is insufficient to choose a source fix or dependency change. Any new isolated runner job, GPU re-test, WSL launch or binary installation requires separately verified authorization. Keep FORGE/Mind-Palace/Wingless/Yggdrasil undisturbed.
