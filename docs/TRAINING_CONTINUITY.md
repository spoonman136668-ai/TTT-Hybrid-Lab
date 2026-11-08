# Native Hybrid: preserve training across restarts (inactive engineering candidate)

`native_training_checkpoint.py` is an isolated, explicit opt-in **library**, not a trainer, scheduler, daemon, scientific first-use, or FORGE integration. It does not alter existing S24–S27 experiments, seeds, labels or accepted model state. It is intended for a future owner-authorized, token-free continuous-state training loop.

## Exact continuation contract

- Save a **zero-step root** with the full model/optimizer/scheduler (if any), native recurrent state, fast weights and bounded history, sampler/cursor, Python/Torch/CUDA/NumPy RNG state, and exact source/architecture/interface/protocol/data-split/harness/environment digests. Pass `parent_manifest_sha256=None` only at zero steps and zero consumed examples. The owner must still authenticate the source and receipt independently.
- After a completed training transaction, commit a new immutable candidate via `save_checkpoint` with the prior *verified* manifest digest, strictly greater cumulative optimizer steps and consumed examples. It publishes content-addressed payload (`.pt`) first, then immutable manifest (`.json`); incomplete/orphan payloads are never accepted as a completed checkpoint. There is deliberately no mutable `latest` pointer.
- Before resuming, use `validate_chain` and `load_checkpoint` **with an explicit tip digest and exact expected identity**. Validate all ancestors and restore optimizer moments, scheduler (if present), RNG, model and native memory. The returned sampler state must be used to restore actual dataloader order; a caller that ignores it is not qualified for continuation. Never silently initialize a fresh state when verification fails.
- Treat model/code/optimizer/interface/dataset/harness drift as a **separately qualified migration**. Preserve the previous accepted checkpoint; no migration path is implemented here. Do not use native state tensors or model parameters from one architectural variant to silently initialize a different variant.
- After restoring, perform exact deterministic cold-replay on a frozen canary and independent retention/interference tests before science. Candidate promotion must preserve earlier qualified tasks with frozen tolerances and separately establish scientific improvement; never auto-promote from this library or from a green CI badge.

### Example calls (illustrative; not executable scientific authorization)

```python
from native_training_checkpoint import save_checkpoint, load_checkpoint
# model, optimizer, native_state and sampler_state are supplied by the owner.
# identity includes source_sha plus the six SHA-256 fields defined in the module.
root_sha = save_checkpoint(checkpoint_dir, model=model, optimizer=optimizer,
    identity=identity, native_state=native_state, sampler_state=sampler_state,
    steps=0, consumed_examples=0, parent_manifest_sha256=None)
# After authorized training and verification, save with parent_manifest_sha256=root_sha.
# Cold restore must explicitly pass the selected manifest digest.
```

## Engineering evidence / limits

Standalone synthetic unit tests check exact model and AdamW optimizer restoration, Python/Torch random state replay, native fast-memory retention, sampler cursor, parent linkage, drift, tampering, reset and replay rejection. This is **local candidate engineering**. No real training, GPU, original-science data, acceptance, source-authenticated remote CI or persistent owner pointer is claimed. All current non-token Hybrid first-use work and frozen negative scientific baselines remain untouched.

### Needed before always-on continuous research

1. Qualified persistent owner storage, transactional accepted-pointer updates and crash recovery across hosts/attempts.
2. Trainer/dataloader integration enforcing step commit, sampled-data lineage, exact resume and no re-consumption; test real crash injection and determinism on Windows/WSL.
3. Frozen retention and improvement acceptance gates, independently authenticated checkpoint custody, strongest trained conventional controls, whole-process cost measurements.
4. Independently authorized original scientific first-use and separate scoped Hybrid activation; FORGE receives only minimal signed, independently verified evidence without acquiring ownership of the private runner.
