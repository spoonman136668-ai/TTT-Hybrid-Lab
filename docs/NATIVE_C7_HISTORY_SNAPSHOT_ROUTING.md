# Native C7 — observed-history selection of preserved native-memory snapshots

## Question

C6 proved that archived step-40 native fast memory can reduce old-task MSE on the *current* step-80 model without new outcomes or writes, if the evaluator **already knows** which saved state to choose. The prior task's error fell from 0.793203 to 0.041751 on the three-seed mean. However, oracle state choice is not autonomous retrieval. C7 tests a bridge: use *the existing P6 observed-causal-history estimator*, which gets only past observed (action, consequence) pairs, to choose between the **two intact, checkpointed native memories**.

This is a separate integration test from the independently ongoing P6/P7 research branches, not a new alternative router or replacement for the existing trained P7 context model. The implementation narrowly reuses the behavior of P6 `FactualSlopeEvidence`, recorded in PR #60 SHA `1123a8aee15572f7f2602bce135a0f38c7b9909a` source blob `7f21b73d6255f2611512969e87446352c1dee90c`. It retains P6's hand-coded fixed threshold 0.2 and initial old-route tie break. **No claim of a trained context selector** is made.

## Frozen before execution

`experiments/native_c7_observed_history_archive_dev.json` freezes all training/evaluation and hypothesis inputs:

- Existing unchanged P3 native factual training, three DEV initializations `46111, 46117, 46129`, 40 completed old-task updates followed by 40 reversed-task updates, source-locked immutable root→step40→step80 checkpoint ancestry and separate protected C2 cold loads.
- One step-80 model (its trained weights fixed throughout all evaluation) and **two actual restored fast-memory snapshots**. The old snapshot is grafted into a clone of the *step80* recurrent/history state, matching C6's old-fast-only condition. The new snapshot uses complete step80 native state. No extra fast capacity is allocated in a single prediction.
- Two public analytical DEV context sequences `old/recent/old/recent` and `recent/old/recent/old`, 16 events per block, 64 prequential events per sequence; immediate inputs always sensor zero, elapsed 0.2 and alternate action -1/+1. Only the *actual completed factual consequence* reveals the effect sign.
- **Six arms** each score *the same* events before the factual outcome is obtained: P6 inferred-history selector, oracle true-context upper bound, always-recent, always-old, action-sign shortcut, and an intentionally corrupted-action/history negative control. No-current-outcome, no-latent-label and no-schedule access are enforced in the inferred selector's signature and unit tests. The oracle's hidden label is explicitly separate and never enters the inferred function.
- Inferred P6 evidence updates **only after all arm predictions**. The negative receives precisely the same outcomes and event counts but deliberately pairs each with the opposite action sign, so its routing inference should be unreliable. The test records both the immediate 4 events and events 4–15 of each block, including switch latency and late route accuracy.
- Source-bound raw per-event squared errors, selected routes, observations, complete root/old/new checkpoint digests and canonical evidence hashes are retained. All original trained model weights, optimizer slots, native fast/recurrent/history tensors and saved accepted/latest pointers remain unchanged.

Frozen diagnostic hypotheses, **allowed to fail**: late inferred MSE beats always-recent, action-sign and wrong-action-evidence by at least 0.10; late context-choice accuracy at least 87.5%; oracle upper-bound late MSE at most inferred MSE plus 1e-5. The engineering pass tests chronology, isolation, source and evidence integrity, **not** a desired hypothesis outcome. Negative/null results must not trigger threshold edits.

## Boundaries and fair-comparison caveats

This is **observed-history-driven reuse of static archived fast memory**, but the initial history must be *acquired from new factual observations*. It is not zero-feedback cold retrieval. No gradient or fast-memory write occurs after checkpoints: the P6 estimator alone updates a small internal sign-of-effect record. The six arms have the same current action/sensor/time, native model, saved-memory storage and scored events, but P6 inference and negatives differ in compute. **A truly compute-matched trained conventional control remains absent**, so scientific performance superiority is unverified.

The two regimes are already used in the original P3 training; this is **not heldout**, unfamiliar transfer, independent original-science first-use, stable retention after long delays, learned P7 routing, true JEPA/world-model cognition, or recursive self-improvement. We are checking only whether an existing source-consistent historical inference mechanism can make the archived state useful after factual observations.

An unprovisioned durable store, owner signing credentials, distinct physical recovery host, owner acceptance pointer, authorized active training, independent S26/S27 original first-use and FORGE integration remain outside C7. No new controller, runner, queue, paid API or modification to P9 GPU/Wingless/Yggdrasil/other research.

## Qualification

C7 must run complete exact-head source-locked **Windows and Ubuntu WSL** CPU regression plus original-source CUDA negatives on the existing project-owned isolated runner, then execute all six seed/order combinations on the *same final SHA*, with per-arm raw error and route evidence. A failed hypothesis is a preserved scientific-development finding, not a test-infrastructure failure. Keep PR draft/unmerged. Hosted G4/G5/G7 whole-PR CI failures remain separate and need independent root-cause diagnosis.
