# Hybrid C3 — real factual native training with champion-isolated checkpoint continuation (DEV)

This is the first bridge from **C2's champion-safe candidate restoration** to the pre-existing **P3 factual native learning workload** (PR #54), rather than to the P6 public one-step linear-model fixture. It remains an offline **engineering** test, not scientific intelligence/retention acceptance or the always-on training controller.

## Source custody

The six pre-existing P3/P2 factual DEV source/test/experiment files were copied **byte-for-byte** from PR #54 exact `82d68d947242d50d4993942ba2e9d5acf6343055` (including P2 source from PR #52). The ancestor native architecture/P0 causal-order gate, P2 checkpoint library and C2 isolated loader stay unchanged. Their Git blob hashes are explicitly checked in the one-use workflow; no S26/S27 sealed worlds, first-use seeds or alternative scientific protocols are touched.

The frozen P3 pilot has exactly three engineering model initializations `46111, 46117, 46129`. Each runs **80 real AdamW gradient updates** over a fixed public analytic factual action-consequence stream. The action effect reverses at event 40. The learning target is revealed only after the prequential prediction and enters factual memory; no unobserved counterfactual is used for training. It is deliberately simple algebraic DEV data, not independently held-out evidence of intelligence.

## C3 integration

`scripts/native_c3_factual_isolated_resume.py` uses the exact unchanged P3 trainer `transact`, `save_completed`, and P0 native state constructor. For each of three initialization seeds:

1. Train and update native factual fast memory through **40 completed events**, then save an immutable root → trained 40-event checkpoint. Verify the saved memory is nonzero.
2. Continue the protected baseline model without interruption from events **40–79**, capturing every prequential error, final model weights, AdamW moments, recurrent/fast-memory/history tensors and final CPU RNG state.
3. Use the C2 transactional candidate API to restore the **exact 40-event checkpoint into fresh model/optimizer objects**, preserving the protected baseline. Require the pinned parent digest, 40 completed events, complete causal ticket, and intact fast memory.
4. Activate the *saved* candidate RNG **only inside a scoped engineering continuation**, leaving ambient RNG unchanged after the continuation. Execute the **same 40 factual gradient/memory updates** from event 40 to 79 and compare every post-checkpoint error, model parameter, optimizer moment, native memory tensor and final RNG state **bit-for-bit** against the uninterrupted baseline on the **same OS**.
5. Save the continued candidate as a new immutable **80-event child** pointing to the 40-event tip; verify exactly 80 consumed examples, no replay/reset, no `accepted` or `latest` pointer. Cold-restore that child again using C2 into a *third* fresh model/optimizer and verify exact equality and champion isolation.

No physical backup, off-host restore, ownership key, persistent training registry, source-born world, new controller/queue, paid API, model deployment or FORGE activation is introduced. This adapter is not compatible with an active production champion without separately qualifying its owner and state compatibility.

## Acceptance and reason for this gate

The question is **not** whether a checkpoint can be serialized yet again; it is whether **the existing native factual-learning machinery can continue without loss** when the checkpoint is inspected and restored via C2's champion-safe layer. Failure preserves the source, candidate and all negative results and blocks integrating with an active trainer.

The dedicated one-use project-owned qualification must pass the **complete exact-candidate Windows and Ubuntu WSL CPU regression**, original-source CUDA negative control on each, exact unchanged P3 blob custody, and independently run all three fixed P3 engineering initializations. Results are engineering-only; even all passing does **not** establish unfamiliar-world transfer, cold retention of prior learned tasks or competence beyond matched trained conventional controls.

Stacked on C2 PR #66 with no parent merge and no changes to separate P9 or FORGE science development. Reconcile hosted G4/G5/G7 failures separately; project-owned green CI alone does not mean merged or scientifically accepted.
