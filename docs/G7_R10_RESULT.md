# G7-R10 — original G7-R9 compiler-error custody: PASS

**Date:** 2026-11-06. **ENGINEERING:** read-only evidence recovery PASS, original GPU integration FAIL. **SCIENTIFIC:** UNVERIFIED. **DEPLOYMENT:** NOT AUTHORIZED.

## Source and authenticated execution

- Owner-authorized [PR #25](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/pull/25), exact candidate `b9907b24ee2d6ba330782bec182b1ee38e1ac026`; all 5 mandatory Linux/Windows G4/G5/Mamba CPU checks PASS, merged at `da7cadb6cd6734619494d7412039a62a37a6d4da`.
- Single authorized read-only [Actions run #42302765535](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42302765535) on `TTT-HYBRID-GPU-WIN`; run_number=1, run_attempt=1; job `recover-original` PASS, artifact upload PASS. **No checkout, WSL, CUDA, model, package installation, gradient or training.**
- Sanitized original error [artifact ID 13544313274](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42302765535/artifacts/13544313274), 30-day retention. GitHub Actions logged ZIP SHA256 `8480387bc23e5ce668800e0481dbaf37957a3582b080316beb85ce54e2f6dcb6`; downloaded archive independently matched this hash. Embedded `g7_r10_readonly_diagnostic.json` SHA256 `de8cf3cbb3ae7d5a695a36b1d71c4ac1f9fcf34e9ab3aa9741fbccc2b32c5ef6`.

## Authenticated original failure and remaining uncertainty

- `status=PASS_ORIGINAL_ERROR_CUSTODY_ONLY`, `original_source_bound=true`, `native_digest_matched=true`, `g3_digest_matched=true`; original source SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744` corresponds to the **G7-R9 native GPU failure** in [original run #42300670454](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454).
- Original native receipt SHA256 `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`; nested G3 SHA256 `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87`. Original native path `PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED`; original G3 status `FAIL_AUTHENTIC_GPU_INTEGRATION`.
- Authenticated exception type: `RuntimeError`. Sanitized signature: **`TRITON_COMPILER_EXCEPTION`**. Last allowlisted traceback source module: **`driver.py`**. The classifier uses category keywords extracted in-memory, and did not upload raw exception or traceback: `raw_error_message=NOT_EXPORTED`, `raw_traceback=NOT_EXPORTED`.
- **Root cause UNVERIFIED**: the signature is consistent with an exception in Triton compilation/driver code, but does not prove an NVIDIA Blackwell/SM120 compatibility defect, PyTorch or Triton version mismatch, PTX toolchain defect, specific failed kernel, or need for a software upgrade. Another GPU execution, binary replacement or dependency installation is not approved by this receipt.
- **STALE_RUN_ID_METADATA**: the diagnostic workflow inherited historical `original_run_id=42281970624` from G7-R7 even though its exact source SHA, two receipt hashes, and the GitHub Actions run bind it to **G7-R9 #42300670454**. This metadata defect is documented rather than altering the immutable artifact or using its stale ID as custody proof.

## Disposition

**NO GPU RETRY.** One-shot diagnostic workflow `.github/workflows/g7-r10-single-readonly-receipt.yml` is retired via qualified source-only cleanup; permanent record is preserved. A future targeted non-executing diagnosis may inspect the pinned Triton `driver.py` control flow and the classifier's boundaries, but code changes or GPU re-tests require specific evidence/authority. Full learning/generalization and deployment claims remain UNVERIFIED / NOT AUTHORIZED.
