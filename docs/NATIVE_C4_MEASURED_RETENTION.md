# Hybrid C4 — measured capability regression gate (DEV; not scientific acceptance)

## Why this gate, not another checkpoint experiment

C3 proved bit-exact continuation of actual *native* factual learning (40 → 80 AdamW updates) while preserving a protected reference model. P3 PR #55 already has a content-addressed retention-policy checker, but its original tests only fed **asserted synthetic scores**. C4 closes the interface gap by producing those scores from **actual C2-recovered trained native P3 checkpoint states**.

No new learning algorithm is proposed. This is a measured **no-silent-regression** screening path to prepare an eventual owner-authorized champion/challenger program.

## Predeclared engineering protocol

Frozen before the first C4 run at `experiments/native_c4_measured_retention_dev_frozen.json`:
- Exactly three already-established P3 model initializations: **46111, 46117, 46129**.
- Same source, factual-only P3 train sequence: 40 completed events → champion manifest, then through 80 completed events → challenger manifest. Parent hashes verified.
- Four pairs of 16? **Exactly two capabilities, each 16 read-only probes per checkpoint**: old pre-reversal `+0.45 × factual action + 0.04`, new post-reversal `-0.45 × factual action + 0.04`. Alternating observed actions, fixed sensor zero, dt=0.2, each probe begins from a *clone* of its own checkpoint native recurrent/fast/history state.
- Compare MSE on old regime (retained ability) and current regime (post-reversal adaptation). Fixed MSE competence floor **4.0** (extremely permissive DEV bound, NOT a scientific competence criterion), retention tolerance **0.002**, meaningful gain **0.02**, minimum 16 probes per capability, 120-second per-seed bound, CPU only, 2 GiB process RSS bound.
- Any regression larger than the tolerance blocks, even if another capability gains. No meaningful gain also blocks. This policy returns only **BLOCKED** or **CANDIDATE_REQUIRES_INDEPENDENT_QUALIFICATION**, **never** automatic acceptance.
- Compare unchanged-score control, derived from actual champion measurements, to verify it is blocked for no gain. A separate deliberately fabricated negative test proves that claimed post-reversal improvement cannot excuse old-regime regression.

Every score is a calculated mean over an included vector of sixteen actual prediction errors. Vectors and checkpoint digests are individually hashed and retained with the receipt. All evidence still lacks an independent trusted issuer and witness.

## Critical limits on interpretation

These "old/new" probes **reuse the same public algebraic regimes and actions as training**, not independently generated unseen environments. They cannot prove generalization, true holdout retention, causal comprehension, resistance to heterogeneous interference, or superiority to a fully trained conventional baseline. The old/new comparisons also conflate fixed model parameters with native memory changes; they are a **diagnostic**, not a clean mechanism attribution. Favor an observed negative rather than changing thresholds after seeing outcomes.

Reported `wall_seconds` covers this DEV per-seed measurement path; OS peak working set is a process-wide high-watermark only, not full training/storage cost and not a GPU metric. Any positive registry response is advisory and **not SCIENTIFIC_PASS**.

No altered original P3 source from PR #54, no official S26/S27 original-world first-use, no new runner/controller/queue, no external backup destination/owner key, no accepted/latest pointer, no real model promotion, no FORGE runtime integration. Separate scientific research PRs #56/#58/#60/#62/#63/#67 remain untouched.

## Qualification

The project-owned, existing isolated Windows runner executes one-source-SHA full Windows and WSL CPU regressions (including original-source negative controls), then runs the three-seed C4 experiment, retaining measured vectors, frozen policy digests, explicit blocker reasons, and full provenance. The 3-seed experiment can **pass engineering integrity even when all candidate reviews BLOCK**: the latter are substantive findings, not test-infrastructure errors.

Keep C4 stacked as a draft on C3 PR #68. Hosted G4/G5/G7 job failures remain separate and unverified as to root cause. Do not use C4 DEV results to override required independent scientific qualification or activate continuous training.
