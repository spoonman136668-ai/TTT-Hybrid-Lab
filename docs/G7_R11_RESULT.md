# G7-R11 — exact-original CUDA helper C compiler failure: verified

**2026-11-06. ENGINEERING:** original read-only source/receipt custody PASS; G7-R9 native GPU forward remains FAIL due to an identified missing host C compiler. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**

## Execution and evidence

- Pre-execution [PR #27](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/pull/27) candidate `8c7896d889f1761bb4aa971fdb867e2f13b9783e`; mandatory five G4/G5/pinned Mamba Linux+Windows CPU checks PASS.
- Owner authorized merge and one **read-only** execution. Merge SHA `028cd7057e00b7c968ee0f53a8d9074f41771dec`; [GitHub Actions run #42313100239](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42313100239), run_number=1, run_attempt=1, existing isolated `TTT-HYBRID-GPU-WIN`; job `recover-original` PASS.
- Immutable sanitized [artifact ID 13549970962](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42313100239/artifacts/13549970962) (30-day GitHub retention). Artifact ZIP SHA256 `ca384d96fd35e7b438c6af86434f641dea65e978fc0f05d93b21cfe03bbf9e20` independently matches GitHub upload log. Embedded `g7_r11_readonly_diagnostic.json` SHA256 `45a0e3090f936c31af9daefca4d37a8ba4f31c2a8c3cd837a8655fc2b6c1521b`.
- `status=PASS_ORIGINAL_ERROR_CUSTODY_ONLY`; original source bound, native digest match and nested G3 digest match all **true**. `original_run_id=42300670454` correctly identifies the original [G7-R9 attempt](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454) at source SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`. The former G7-R10 stale run-ID metadata defect was corrected in this diagnostic, without altering old artifacts.
- Original native receipt SHA256 `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`; nested G3 SHA256 `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87`; native path `PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED`; original G3 status `FAIL_AUTHENTIC_GPU_INTEGRATION`.

## Root-cause qualification

The script classified **only the original exception message**, not generic traceback words:
- `exception_type=RuntimeError`
- `cause_code=HOST_C_COMPILER_MISSING`
- `frame_phase=TRITON_HOST_C_HELPER_BUILD`
- `precise_root_cause=HOST_C_COMPILER_UNAVAILABLE_CONFIRMED`.

This matches the error raised by pinned Triton 3.5.0 `python/triton/runtime/build.py` when neither `gcc` nor `clang` is found for its host-side CUDA helper-module compilation. The matching full module path corroborated that phase. **This is a confirmed explanation of the original G7-R9 failure, not proof that any subsequent CUDA forward, kernel correctness, learning, or deployment will succeed.**

Only allowlisted codes and hashes were exported. `raw_error_message=NOT_EXPORTED` and `raw_traceback=NOT_EXPORTED` are enforced; the new artifact includes `traceback_module=NOT_EXPORTED_AMBIGUOUS_BASENAME` to prevent over-reading earlier `driver.py` labels. No installation was performed; no WSL/CUDA execution or native model work was started by G7-R11.

## Disposition and smallest next engineering gate

**NO GPU RETRY.** The one-shot `.github/workflows/g7-r11-single-readonly-receipt.yml` is removed by source-only cleanup PR; preserve these immutable hashes and results. Original G7-R9 forward and finite/causal checks remain FAIL/NOT REACHED.

Next, obtain a separately scoped authorization for **read-only toolchain discovery** inside the existing isolated WSL2 environment, bound to the exact runner/workspace. Check `command -v gcc` and `command -v clang` plus provenance for any already-qualified installed compiler. Do not install or download a compiler, modify WSL or system PATH, launch the model, or rerun a GPU experiment under this authorization. If neither compiler is available, report BLOCKED and request an explicit environment-change authorization before installation. Subsequent exact-candidate integration qualification and a separate GPU experiment authorization remain necessary.

Engineering: failure mechanism IDENTIFIED; compatible compiler availability and GPU recovery UNVERIFIED. **SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.**
