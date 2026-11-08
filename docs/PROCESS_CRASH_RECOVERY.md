# Hybrid P5 — actual process-crash recovery rehearsal (inactive engineering)

This is the **next preparation gate** after P4's immutable checkpoint copy contract. It tests a real child-process termination during replication rather than only Python exception injection. The test is a bounded, CPU-only synthetic fixture on the already isolated Hybrid project runner. It does not train original science, activate FORGE, or change an accepted model.

## Exact failure/restore experiment

- Save an explicit P2 zero-step root followed by a one-step synthetic AdamW training checkpoint with native recurrent state, fast memory, history, model weights, optimizer slots, RNG, and sampler cursor.
- A separately launched Python child executes P4 immutable replication and calls `os._exit(73)` immediately after two verified root objects are published, before the trained checkpoint tip is present.
- The supervising test proves that the incomplete candidate tip is rejected. It does not pick a fallback, create an accepted/latest pointer, or turn a partial transfer into a recovered checkpoint.
- Explicitly retry replication from the authenticated-by-local-hash primary, verify the entire immutable P2 ancestry, then deliberately corrupt the primary.
- Verify that the corrupted primary cannot be used. **Explicitly** select the intact replica, validate its parent hashes, copy to fresh staging, and cold restore model weights, AdamW state, native fast memory/history and sampler cursor. Test exact equality with the pre-crash expected values.
- Reject nonempty scratch locations. All experiment paths are private temporary directories, removed by the test process after completion.

The receipt reports source-device sameness/difference but **does not use that field to assert independent physical storage**. A different filesystem identifier is insufficient evidence of an off-host failure domain or survivability of simultaneous disk/controller failure.

## Acceptance and remaining limits

A PASS is evidence for **process-death-safe explicit checkpoint recovery on a synthetic fixture**, *not* for power-loss durability (directory flush / filesystem semantics), independent backup capacity, actual distributed replication, source/issuer authentication, scientific cognition or continuous model training. The tests run against the existing exact-source frozen P2–P4 implementation, not an altered scientific workload.

Before live use: qualify an **owner-approved, independent backup destination** with custody, resource budgets, failure-domain verification, directory and volume-specific crash consistency, access controls, fresh-host restore, real training sampler integration, and memory/retention acceptance. Do not probe or write unknown external/private volumes merely because a runner can see them.

The user-owned Hybrid research controller and prior frozen S26/S27 protocols, seeds, PR #54 training experiments, Wingless/Yggdrasil loops and FORGE boundaries remain unchanged. This PR is **stacked/draft**, not merge authority or scientific first-use.
