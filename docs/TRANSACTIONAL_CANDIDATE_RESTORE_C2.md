# C2 — transactional, champion-preserving Hybrid checkpoint restoration

This is an **inactive engineering defense** added after C1 remote recovery. It protects a retained model from accidental partial restores and preserves global random state. It does not provide accepted-model authority, secure issuer identity, independent backup, automatic training, or evidence of improved cognition.

## Defect addressed

The existing P2 `load_checkpoint` checks source ancestry and payload structure, then assigns model parameters, optimizer slots and RNG in sequence. A later exception could leave caller-supplied model/optimizer partly changed. Directly loading into a champion model would therefore be unsafe even with hash-verified source bytes.

## Implementation

`native_transactional_resume.prepare_isolated_candidate` performs the exact lineage/source/cursor and optional expected-parent checks **before** calling the model factory. It creates a new model and optimizer whose parameters belong exclusively to that model, rejects reuse of the caller's champion objects, and detects shared Parameter and underlying storage aliases. The P2 loader writes **only to this new candidate**, never to an accepted model.

Snapshot Python, torch CPU, initialized CUDA and optional NumPy global RNG before factory/restore; restore those globals in `finally`, including on failure after a partial candidate mutation. The caller receives an unpromoted `PreparedCandidate` object with cloned native memory, saved cursor, source SHA, restored RNG *snapshot* and immutable checkpoint digest. Crucially, merely reviewing the candidate does not activate its saved RNG in the current owner. If the caller later chooses to resume training, applying the saved RNG and any scheduler must be separately authorized, source-compatible and tested.

C1 `scripts/native_c1_remote_resume.py` now uses this API both for initial remote-restored step 1 and for the independent step-2 cold check, with exact expected parents and no-champion aliasing for the second load. The previous published P6 artifact, P2 serializer, P6 transport, P7 key API and frozen trainer source are unmodified.

## Tests and limitations

Focused positive and negative tests check retained champion weights and optimizer unchanged, ambient Python/torch RNG unchanged on valid and injected mid-restore failures, trained-root reset denial, parent/tip/cursor mismatch, alias/factory errors, and corrupt manifest refusal before the factory is invoked. The exact commit requires complete Windows+WSL project regression plus original CUDA negative controls and a project-owned C1 continuation directly from the *existing* synthetic external P6 artifact.

The factory must still be trusted owner code: arbitrary malicious factory side effects, CUDA initialization by other threads, source issuer authenticity and real off-host power-loss are outside this adapter's guarantees. No model is promoted or tagged accepted/latest. Current P2 checkpoint format and all historical results are retained unchanged.

Status: **DEV only** until separately proven by exact-source dual-platform final qualification. Keep stacked PR draft and independent scientific research running.
