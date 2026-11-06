# G7-R6 — one-time authentic Mamba-2 + TTT-Linear CUDA forward attempt

**Date:** 2026-11-06. **Engineering:** `FAIL_NATIVE_GATE3_INTEGRATION`. **Scientific:** `UNVERIFIED`. **Deployment:** `NOT_AUTHORIZED`.

## Qualified source and execution custody

- Owner explicitly authorized one isolated forward-only GPU test. Owner declined a repository-wide branch protection change; the existing one-use workflow instead enforced first run/attempt, owner actor, main push, exact checkout SHA and runner identity.
- Source-candidate SHA `92e06a690cbed0a89350e9c0e35041d18c74ceda` passed all five mandatory G4/G5/Mamba Linux/Windows tests. No model-training execution was part of source CI.
- Execution commit `d028badf39f7e3392adc24329be4152cdcdc00cd` (PR #18 merge), same source tree as qualified candidate.
- [G7-R6 workflow run #42281970624](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42281970624): first run/attempt on existing `TTT-HYBRID-GPU-WIN`, checkout PASS, bounded native forward step FAIL, sanitized artifact upload PASS.
- [Artifact ID 13536275314](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42281970624/artifacts/13536275314) (30-day retention) contains `g7_r6_forward.json`. Downloaded ZIP SHA-256 independently verified `c9cd29f6a9531441d26f140d81a7590391a03f59f32f24a442ebd33cc9ebef89`.
- Inner native receipt SHA-256 recorded as `4d6cd2d716a4ea417d2e5a424e6878c9bad4ab93a0e7e5351ceb0c4158700f29`. The inner raw G3 subreceipt was **not uploaded** and its error classification/details have not been verified.

## Exact failure boundary

Outer one-shot status `BLOCKED_NATIVE_FORWARD_NOT_QUALIFIED`; CUDA wrapper `FAIL_NATIVE_GATE3_INTEGRATION`; nested gate `FAIL_AUTHENTIC_GPU_INTEGRATION`. This is **an authentic integration failure, not a CI or GitHub billing failure**.

The checked Mamba-2 and TTT source commits and Git blobs matched their pins and both trees were clean. CUDA host admission returned `HOST_ADMITTED`; recorded NVIDIA GeForce RTX 5070 Ti Laptop GPU compute capability `[12, 0]`. The native integration failed before finite output, episode reset, prefix invariance and suffix-leakage acceptance results were populated. GPU peak allocation also remained null. Do not claim native forward, learning, improved reasoning, transfer, retention, latency or efficiency.

## Safety and next work

No backward, optimizer, training, system installation, deployment or adjacent-project workload was authorized. The one-shot source workflow was consumed and is removed after the attempt; do not rerun or re-enable it.

Next minimal engineering action: **read-only retrieve and sanitize the existing G3 subreceipt from the isolated TTT runner**, which may explain why `FAIL_AUTHENTIC_GPU_INTEGRATION` was raised; independently verify error type, source provenance and scope, then make the smallest source-only correction and qualify it on mandatory exact-head CI. A new GPU execution requires separate scope/authority/resource approval. Preserve FORGE, Mind-Palace, Wingless, Yggdrasil and other resident loops.

## Subsequent G7-R7 diagnosis — 2026-11-06

The earlier "error classification unverified" statement above is the historical **G7-R6** status before the original inner G3 receipt was recovered. It has now been superseded by this independently checked diagnostic; the G7-R6 native forward **remains FAILED**.

- Scoped one-shot **read-only** recovery: [run #42285407805](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42285407805), execution commit `ebd0b3c38f8bc28b89087b3d22a8706b3602e439`, owner-scoped runner `TTT-HYBRID-GPU-WIN`, G7-R7 job SUCCESS.
- Sanitized [artifact ID 13536149391](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42285407805/artifacts/13536149391); independently downloaded ZIP SHA-256 `94ed67b32c164245c05fd1392be8aad3f2b2881b5c8b0a503aaf76b24a38c3b2` (30-day retention).
- Receipt `status=PASS_ORIGINAL_ERROR_CUSTODY_ONLY`; original Git checkout bound to `d028badf39f7e3392adc24329be4152cdcdc00cd`; native receipt SHA-256 matches `4d6cd2d716a4ea417d2e5a424e6878c9bad4ab93a0e7e5351ceb0c4158700f29`; original nested G3 subreceipt hash matched its parent native receipt; pinned `source_pins.json` SHA-256 `44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768`.
- **Recovered original exception:** `TypeError`. Sanitized signature: `FUSED_CAUSAL_CONV1D_NOT_CALLABLE`. Last known traceback source module: `ssd_combined.py`. Raw exception text and traceback intentionally **NOT_EXPORTED**.
- Root-cause level: the original fused Mamba-2 execution path attempted to call an unavailable causal-convolution function, based on original error signature and source location; this identifies the immediate failure mechanism. Why the binary/function is unavailable on this host and whether an official unfused path succeeds are **UNVERIFIED**. Do not infer build ABI, compatible extension, working fallback, or successful model forward without a separate test.
- **No WSL, GPU, model, source checkout, training, deployment or package installation** occurred in this diagnostic. The G7-R7 one-shot diagnostic workflow is now retired via bounded source-only cleanup, preserving the original evidence.

The **next change** should be a minimal, separately scoped source-only admission/fallback investigation on the pinned official Mamba source. One candidate is the upstream non-memory-efficient path (`use_mem_eff_path=False`) to avoid the missing fused `causal_conv1d_fwd_function`, but it needs source-level/CPU negative controls and exact-head CI first, and it does **not** establish fused CUDA compatibility. Native GPU retesting, binary installation, or deployment is separately authorized. Scientific status remains `UNVERIFIED`.
