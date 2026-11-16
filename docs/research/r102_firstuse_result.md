# R102: contradiction-gated causal memory — original scientific result

**Result: SCIENTIFIC NO-GO.** The one-time owner-authorized R102 CPU experiment was run once at source `745ce5c9ba2c7bb0dbfa3d44433e635a079a0ce2`, [original GitHub Actions run 43903611516](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43903611516). All **three independent seeds completed 11 fully trained arms** and first-use held-out scoring under the frozen 900-second-per-seed budget; none passed all 12 precommitted criteria.

| Seed | CPU whole process | Frozen checks passed | Unfamiliar decision accuracy | Native minus strongest conventional | Result |
|---|---:|---:|---:|---:|---|
| 1002711 | 228.23 s | 6/12 | 58.85% | −0.946 | NO-GO |
| 1002723 | 385.73 s | 8/12 | 71.35% | +0.095 | NO-GO |
| 1002737 | 324.31 s | 7/12 | 67.71% | −0.025 | NO-GO |

**Mean decision accuracy 65.97%**, below the **75% frozen minimum in every seed**. Mean native reward advantage vs strongest conventional **−0.292**, below the required **+0.15 in every seed**; paired 95% lower bounds **−1.375 / −0.356 / −0.546**, all below the minimum +0.02. Every seed failed the OOD improvement threshold as well. This disproves the frozen strong unfamiliar-rule-learning claim for this implementation and setup. It does NOT prove all contradiction-gated learning impossible.

## Positive bounded mechanistic findings

With identical factual observations/architecture/matched controls, the **contradiction-gated native learner beat the ungated same-capacity native** by **+0.166, +0.211, +0.103** realized reward (respectively) across all three original seeds; all three satisfy the predeclared +0.10 mechanistic check. It also beat the external matched learned-state comparator in 3/3, the zero-support-outcome comparator in 3/3, and the cold memory-reset comparator in 3/3. Physical role relabel agreement was **85.9%, 87.0%, 82.8%** (above 75%); and the reversal-vs-zero-feedback comparison passed all three. These are bounded, subhypothesis signals: overall independent scientific PASS remains false.

Crucially, **16 versus 6 factual supports** yielded reward differences **−0.519, +0.500, +0.078**, with only 1/3 seeds satisfying the frozen +0.10 gain. More observations alone did not translate to reliable transfer or stronger conventional-baseline performance. Shuffled-feedback improvement passed only 2/3. No active FLOPs were measured; no compute superiority or general-intelligence assertion is supported.

## Exact scientific and engineering custody

- **Frozen before implementation**: R102 freeze-only SHA `5c5fd2ed7481b953b8a06fba745b03e9cbcf73a7`, immutable frozen Git blob `9e4a74c4b903a8c205807f5ec0ba87d1ff028f82`; new seeds 1002711/1002723/1002737; 16 observed physical support interventions; two fresh evaluator-only unknown operator families `EXACTLY_TWO3`/`ALL_EQUAL3`; 11 frozen experimental arms; fixed 3,072 steps per arm and paired 2,000-draw bootstrap.
- **One-shot original source**: `745ce5c9ba2c7bb0dbfa3d44433e635a079a0ce2`; original G4 run `43903611516`, qualifying Linux **548 passed, 4 skipped**, Windows **549 passed, 3 skipped**; same-source G5 authentic TTT CPU and G7 pinned Mamba-2 import **PASS**. Original three scientific jobs completed, and each source-scored cohort is now **SPENT**.
- Source-bound original artifact receipts, digest and per-seed numeric checks: [`experiments/r102_scientific_firstuse_seal.json`](../experiments/r102_scientific_firstuse_seal.json). Original artifacts with full-precision results are preserved in the original run, with ZIP SHA-256:
  - [1002711](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43903611516/artifacts/14244112866): `30a3a25d28caa3d217a840ca1a6f898be7528e3bba6720848d09b0f61c54a8ca`;
  - [1002723](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43903611516/artifacts/14245616188): `574f1ea84c0e4de23120a834139bbf1549d8d7bf1414fa928820139659e36478`;
  - [1002737](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43903611516/artifacts/14244043425): `ebdee64eec7c1cfb91e2f7df613d373991240a090dce354616e1332e4086f196`.
- 3 per-seed wall times sum to **938.28 CPU-process seconds**, NOT shared elapsed runtime (jobs ran concurrently). None exceeded its 900-second per-seed limit. Original GitHub artifact retention is 30 days.
- Previously scored R101 evidence and models were not reused for training/scoring. The unchanged preimplementation R102 freeze stays immutable. Do not replay, redraw or requalify the spent experiment as if new.

## Disposition and next falsifiable direction

**ENGINEERING:** original run PASS; post-scoring seal requires exact-source full G4/G5/G7 qualification, no repeat scoring. **SCIENTIFIC: NO-GO** for the frozen strong native unfamiliar causal-rule inference hypothesis, while the 3/3 bounded gate/feedback benefit merits independent new-source study. **DEPLOYMENT: HOLD**; no GPU original use, merge, FORGE integration, model promotion, new runner, paid API or production acceptance.

The next substantive research should use an independently frozen R103 cohort (new source/salts/seeds/heldout structures) and isolate why contradiction gating improves native internal controls but fails to beat strong conventional learned role-set/MLP baselines or benefit consistently from 16 interventions. Test genuine input-latent identifiability, mechanism ablation and whole-process cost without data leakage or postresult threshold tuning. R102's scored source and original run are permanently sealed.
