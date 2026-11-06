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
