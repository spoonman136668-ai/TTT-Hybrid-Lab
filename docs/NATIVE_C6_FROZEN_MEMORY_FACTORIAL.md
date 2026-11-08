# Native C6: preserved-memory versus model-weight drift, no re-learning

## Context

C4 identified large old-task performance loss after 80 gradient updates; C5 recovered much of that performance after eight **fresh old factual outcomes** and fast-weight updates, but that was rapid reacquisition, **not retrieval from stored old memory**. Existing P5/P6/P7 context-routing studies independently own the question of learning or inferring a context router, and must not be reimplemented here.

C6 asks a smaller causal-mechanism question: **would access to the actual previously saved old native memory, without any fresh outcomes, restore prior capability when plugged into the later trained model?** It also tests whether changing model parameters instead of memory is more consequential. C6 has no context inference; the evaluator deliberately chooses the archived state using oracle knowledge. An apparent positive is a *compatibility/availability* signal, not proof that an agent knows when to select the right memory.

## Frozen development protocol

Registered before first execution at `experiments/native_c6_frozen_memory_factorial_dev.json`. Reuses exactly the old unchanged P3 analytic factual optimizer trainer, the original 3 DEV model initializations (46111, 46117, 46129), 40 old-regime events followed by 40 reversed-regime events and P2/C2 source-locked root→40→80 immutable checkpoint lineage. Never consumes official S26/S27 science seeds or world data.

For each seed, cold-restore step-40 and step-80 candidates with strict full ancestry, identity, optimizer, cursor and protected-model isolation. In a read-only `torch.no_grad()` evaluation, compare seven *declared* combinations:

1. **Old champion**: step40 model + step40 full native recurrent/fast/history state.
2. **Recent learner**: step80 model + step80 full native state.
3. **Old fast only**: step80 model + its own step80 recurrent/history and *archived step40 fast weights*.
4. **Old full native**: step80 model + step40 recurrent/fast/history.
5. **Old model + new state**: step40 model + step80 full native state.
6. **Width-rolled old fast sham**: step80 model and recent recurrent/history + same old fast scalars shifted one position along width; same storage, value multiset and norm, wrong alignment.
7. **No fast correction**: step80 model + its recent recurrent/history and zero fast weights.

Each arm receives identical 16 old-task and 16 reversed-task probes, with known old/new targets used **only to compute error after prediction**, never as model input. Each query begins from a fresh clone of the selected state; no observations are fed back, **zero new fast-memory writes and zero optimizer updates**. All original saved model/optimizer/native objects must remain bit-identical and no accepted/latest pointer can be produced. Raw squared-error vectors and source/checkpoint digests are retained.

Frozen *development diagnostics*, each allowed to fail: old-fast-only improves old MSE by at least 0.10 over recent baseline; full-old-native improves by 0.10; old model/current state improves by 0.10; old-fast-only improves over width-rolled sham by 0.10; old-fast-only gets within absolute 0.05 MSE of the complete old champion. No post-result threshold edits or selective reporting.

## Nonclaims and integration boundary

- Incompatible regimes have identical instantaneous observations/action/time, so one fixed state cannot be expected to infer which task is active. **Oracle snapshot selection is not a deployable routing mechanism.**
- An old native memory tensor written by a different model weight configuration may not remain interpretable; failure is an important mechanistic negative.
- Old model + old memory is a complete stored champion reference, not a fair same-parameter comparison against step80.
- This public in-sample analytic fixture supplies no independent held-out retention, unfamiliar transfer, fully trained conventional matched baseline, JEPA, stable lifelong learning or scientific intelligence claim.
- No new runner/controller, paid API, real backup/key custody, model promotion, live FORGE/Hybrid training activation or modification to independently active P9 GPU and P6/P7 science tracks.

Only a successful exact-source Windows and WSL full regression including original-source negatives, and measured complete seven-arm/three-seed receipt, constitutes **C6 engineering PASS**. Scientific status remains UNVERIFIED regardless of diagnostic values. Keep parent C5 and C6 draft/unmerged until separate owner acceptance and hosted CI status are resolved.
