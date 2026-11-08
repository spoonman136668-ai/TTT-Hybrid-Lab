# Hybrid C5 — prior factual history vs post-reversal adaptation (DEV only)

## Distinct question after C4 falsification

C4 PR #70 measured a factual native 40-step champion at **old-regime MSE ≈ 0.00279**, but the 80-step checkpoint had **old-regime MSE ≈ 0.79320**, while its reversed-regime MSE improved to **≈ 0.000738**. Because the old and reversed algebraic regimes have **identical instantaneous sensors/actions/times**, that failure alone does **not** distinguish inaccessible old knowledge from lack of context identification.

Existing research P6 PR #60 already inferred hidden regime from completed factual action outcomes, and P7 PR #62 trained a separate history-based router versus a conventional MLP. **C5 does not replace or rerun those scientific branches.** It answers the narrower C4-to-C2 integration question: can the exact *saved and cold-restored 80-step native P3 learner* rapidly recover the former response when presented with completed factual observations, without further optimizer updates?

## Frozen procedure and limits

Frozen DEV design in `experiments/native_c5_history_readaptation_dev_frozen.json`, prior to first C5 execution:
- Original P3/C4 developmental seeds **46111, 46117, 46129**; unchanged P3 `transact()` on the same known public 80-event analytic reversal environment; save zero-step root, 40-step champion, 80-step child with exact C2 ancestry/cursor, optimizer and native memory.
- Reconstruct **fresh protected candidate objects** for both learned checkpoints. The 80-step model parameters and optimizer moments remain fixed throughout the entire C5 probing phase. No learned model promotion.
- Start every 0-, 2- or 8-event **prefix** from an independent clone of the *same step-80 native recurrent/fast/history state*. Fixed sensor zero, elapsed 0.2, identical alternating -1/+1 actions.
- Compare **old factual** outcomes, **recent/reversed factual** outcomes, and **no-feedback** (advance identical action/recurrent/history trajectory without using outcome or writing fast memory). The old and reversed factual arms have matched action counts, full native forward/read/write operations and state size; only the already revealed outcomes differ. No-feedback is a diagnostic, **not compute-matched**.
- **Chronology:** predict current intervention before calling the original factual `analytic_consequence`; only then call original P0 `observe(ticket, observed, factual=True)`. The inferred model never receives the hidden regime label, t index or counterfactual. No gradient update or training example replay from the optimizer.
- After each prefix, use **the same C4 fixed read-only 16-action old-regime and 16-action reversed-regime MSE probes**, each from a copied native state. Require the factual old, factual recent and no-feedback arms to have **bit-identical recurrent and history tensors**, so the observation difference affects only the fast-weight store.
- Include per-seed raw squared-error evidence vectors, SHA-256 over their canonical representation, exact step-40 and step-80 parent checkpoint digests, fixed protocol digest, runtime and explicit hypothesis results. Both source code and C4 actual learned-state scoring remain unchanged.

The preregistered **development diagnostics** (not engineering pass criteria): old-factual 8-prefix reduces old-regime MSE by at least **0.10** versus no prefix, beats 8-event reversed-factual and 8-event no-feedback by at least **0.10** old-regime MSE, and recovers to within **0.05** MSE of the 40-step champion. Any and all can FAIL and must remain recorded. A failure is scientifically informative, not a permission to retune thresholds or retry seeds.

## Interpretation and authority

**Even a strong positive outcome proves only rapid re-adaptation after re-exposure to old familiar factual transitions**. P0 `observe` explicitly **writes new fast-weight correction**, so this is neither a read-only retrieval demonstration nor evidence of long-term memory preservation without learning. Previously encountered public old actions/outcomes are repeated; there is no independent held-out world, novel transfer, fully trained conventional comparator, authentic owner-issued checkpoint, independent encrypted backup, scientific first-use, new controller/scheduler or FORGE integration.

A negative outcome cannot on its own prove irreversible deletion either: P3 was never trained with a context-indexed episodic memory/routing substrate. The separate P6/P7 and eventual scientific qualification own that question. Preserve frozen results as measured, including control failures.

## Qualification

Execute a **single source-pinned project-owned full Windows/WSL CPU regression** on the already registered isolated Hybrid runner, checking original-source CUDA admission negatives and no original science first-use. Then execute the predeclared three-seed C5 experiment. A PASS signifies complete evidence integrity, causal ordering and per-platform execution—not improvement in any learning ability. Keep all parent PRs including C4 draft/unmerged and do not change accepted/latest refs, model deployment, any Wingless/Yggdrasil research, or active FORGE authority.
