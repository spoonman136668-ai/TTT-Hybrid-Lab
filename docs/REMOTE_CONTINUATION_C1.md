# Hybrid C1 — cumulative training after remote recovery (engineering only)

**Purpose:** close a specific preparedness gap between the already qualified P6 remote synthetic checkpoint recovery and P2 local training continuation. This is a bounded, token-free native-state **DEV** fixture, not production training, an accepted model, a memory benchmark, a genuine off-host backup, or scientific transfer.

## Frozen source and data

The source checkpoint is **reused** unchanged from synthetic P6 producer [run #42475136686](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42475136686), original training-source SHA `3ed4a9119900c3e356794973562c9fcbbeb727a9`. Its metadata identifies a zero-step root and one completed AdamW update with all native memory, optimizer/RNG state and `sampler.position=1`. No source-world data or private model state enters this gate. The existing frozen synthetic training functions and checkpoint serializer must AST/blob-match the producer source; if they drift, **fail closed rather than silently migrate identity**.

The recovery pipeline:
1. Download the one already created, run-scoped public/synthetic GitHub Actions artifact to a new checkout. Require full P2 parent/payload validation and P6 exact cold restore.
2. In an isolated four-file checkpoint directory, create and immediately verify P7 synthetic-key HMAC inventory for the exact one-step tip. This demonstrates the code contract only, **not owner authentication**.
3. Cold load model, AdamW moments, native recurrent/fast memory/history and RNG. Require saved `position=1` and non-empty optimizer moments—**no reset to step zero**.
4. Run exactly one additional bounded factual synthetic CPU gradient update using the **unchanged producer function**. Advance fast memory and save an immutable second-step checkpoint with parent=first step, `steps=2, consumed_examples=2, sampler.position=2`.
5. Reject a stale first-step inventory after the second-step files appear. Create and verify a new test-key inventory for the six ancestor objects. Cold restore the child into fresh model/optimizer objects and require **bit-exact** equality for model weights, optimizer moments, native memory, RNG and position.
6. Do not write an `accepted` or `latest` pointer. Never grant training, deployment, FORGE research authority or model-promotion permission.

## Cross-platform qualification

The exact candidate SHA must pass mandatory full Windows and WSL CPU regressions, the existing original-source CUDA negative controls, and the C1 test against the **external previously published** synthetic P6 artifact. The source-locked owner workflow is one-use and does not generate or upload any duplicate checkpoint. Tenants and storage independence are not inferred from artifact download: the two recovery tests may run on one physical Windows+WSL host.

## Nonclaims and future work

This demonstrates remote-transport-to-next-gradient **continuity**, not open-ended autonomous training, scientific improvement, true CPU↔GPU numerical parity, cross-machine crash recovery, cryptographic issuer identity, confidential artifact transport, durable retention, or production disaster recovery. The P7 key is a deliberately public constant, **never a live owner key**. Windows/WSL floating-point differences remain allowed only for independently recomputed fixtures, never for exact serialized-state restoration.

For an authentic cumulative-learning system, use a separately approved encrypted independent store and key custodian, register real trainer cursor and native model architecture, freeze scientific retention/interference/transfer controls, qualify fresh-host recovery and owner-scoped accepted-pointer promotion. Existing scientific PR #63 and FORGE/Wingless/Yggdrasil work remain isolated.

**Disposition:** inactive draft engineering prerequisite only.
