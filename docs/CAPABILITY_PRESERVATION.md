# Hybrid P3 — cumulative capability registry and challenger review (inactive)

This is a **synthetic engineering candidate**, stacked on P2 checkpoint continuity. No training, scheduler, runner use, science first-use, acceptance, or promotion occurs. FORGE does not gain Hybrid execution/owner authority.

## What exists

`native_capability_registry.py` freezes an immutable, SHA-256-addressed capability policy: explicit selected champion checkpoint, protocol/harness/evaluation-split identity, ordered capability specifications with directions, competence floors, retention tolerances, meaningful-gain thresholds and sample minima, plus wall/GPU/memory ceilings. No `latest` or `accepted` pointer is written. The caller chooses an explicit policy digest.

`review_challenger` compares claimed champion/challenger measurements for **every** frozen capability, rejects missing/duplicate evidence, wrong checkpoint or eval identities, missing competence, regression, missing meaningful gain, invalid scores and out-of-budget resource claims. It returns **BLOCKED** or **CANDIDATE_REQUIRES_INDEPENDENT_QUALIFICATION**; never *accepted* or *promoted*. The receipt scores/hashes are unauthenticated claims, not measured science. This remains so even when all synthetic tests pass.

## Future scientifically admissible owner workflow

1. Select an independently authenticated P2 champion manifest and fully verify its ancestor blobs on the project-owned host; never infer the accepted model from timestamps or a mutable `latest` file.
2. Before generating/inspecting new results, freeze specific capabilities including cold retention, causal interventions, interference, unfamiliar-world transfer, and their matched competent controls; pre-register all thresholds, distinct fit/validation/heldout data, seeds, budgets and evaluation provenance.
3. Run a separately authorized *challenger* from the verified checkpoint. Integrate P2 state restoration and exact sampler re-entry; a source or architecture change requires its own migration proof. Do not overwrite the champion.
4. Independently authenticate both checkpoint blobs, source SHA, owner, signed receipts and test-data provenance; run actual cold restoration, retention, anti-leak controls, and Windows/WSL full regression on the exact candidate. This module does **not** perform those checks.
5. Supply authenticated measurements to the frozen P3 policy. Reject material regression even if a new ability improves. Separately test whole-process cost and trained conventional comparisons. Scientific acceptance and accepted-pointer CAS are separate project-owned authorities not provided here.
6. Retain every immutable negative and prior champion for audit and rollback. If capabilities diverge across architecture generations, record an independently tested state migration rather than claiming continuity.

## Additional preparedness backlog — not implemented here

- Transactional durable checkpoint store replicated off the training host, qualified crash/power-loss and host-unavailability recovery, owner-authenticated accepted pointer and bounded retention/GC.
- Production trainer/data-loader integration, no-replay data cursor, sampled-data manifests and real crash injection across Windows/WSL/GPU.
- Independent per-capability memory/causal dependency mapping and selective rehearsal/consolidation experiments; distinct frozen transfer holdouts and contamination/novelty controls.
- FORGE read-only ingestion of authenticated scientific receipts through its **existing** permitted research interface; no Hybrid runner delegation, alternate queue, new controller, or automatic promotion.

P3 enforces *policy consistency over untrusted claims*, not scientific correctness. A passing review is a candidate for independent qualification only. No monotonic learning guarantee is asserted.
