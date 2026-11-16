# R103 original scientific first-use — final NO-GO

**Scientific result: NO-GO (0/3 independent seeds).** Source `427d964fb72ecb30d4212bd11b8fbb13511e88e3`, [original run #43970270969](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270969). The prospectively frozen protocol was `808aceb4389d093badfa455143cf9f94cdb71cd7` (unchanged Git blob `e9a12e3e5224abf162ae00e82b0781987c00484d`). All three independent seeds fully trained all 11 arms with 3,072 updates per arm **before** first exposure to held-out scoring; no seed exceeded the frozen 900 CPU-second budget.

| Seed | CPU whole process | Frozen checks passed | Unknown-rule decision accuracy (minimum 75%) | Native reward gain vs strongest conventional (minimum +0.15) | Paired bootstrap lower95% (minimum +0.02) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 1003011 | 486.91 s | 3/12 | 53.65% | +0.14887 | -0.50097 | NO-GO |
| 1003023 | 316.55 s | 11/12 | 63.54% | +0.81902 | +0.21525 | NO-GO |
| 1003037 | 521.11 s | 8/12 | 63.02% | +0.36960 | -0.14995 | NO-GO |
| **Mean** | — | — | **60.07%** | **+0.44583** | **-0.14522** | **NO-GO** |

**Plain meaning:** Relational factual feedback helped the native model on several measures, with positive average realized reward advantage over fully trained conventional competitors. That gain was not stable or statistically confirmed across the three seeds: all three failed unfamiliar-rule accuracy, and only one paired confidence lower bound passed. The one exceptional seed (11/12) does not prove unfamiliar-rule transfer.

All seeds passed the frozen comparisons against **no factual outcomes**, **cold memory reset**, and **physical role relabel agreement**. Improvements against an equally capacious ungated relational memory or the earlier scalar-gated native model were inconsistent. Strongest conventional baseline on DEV_UNKNOWN was STATIC_PRIOR16_NO_FEEDBACK for 1003011, GRU64_16_FACTUAL_HISTORY for 1003023, ROLE_SET64_16_FACTUAL_HISTORY for 1003037. No active FLOPs were measured; no efficiency or general-intelligence claim is justified.

## Original source and evidence custody

Original owner-scoped one-time GitHub-hosted CPU source `427d964fb72ecb30d4212bd11b8fbb13511e88e3`, same-source [G4 Linux/Windows #43970270969](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270969) PASS, [G5 #43970271016](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970271016) PASS and [G7 #43970270637](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270637) PASS. Original scientific seed jobs 133662095436 / 133662095495 / 133662095695 SUCCESS. Three source-bound original artifacts (30-day GitHub artifact retention), ZIP SHA-256 verified independently:
- [Seed 1003011 receipt](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270969/artifacts/14279525974): `639f137a406a13a29905dbfa4063982b1dc8f9e49557e8dda5e22d741619dfca`
- [Seed 1003023 receipt](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270969/artifacts/14278756758): `2e707b03807cf813acf38599ebfd804d27ea1fdf27520ff274a0fc206ceaade3`
- [Seed 1003037 receipt](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/43970270969/artifacts/14278518289): `66dcb966f9b0a547bb04ebc1bd3c4cab8be6556509acb39d37d21a3245a0aa56`

Full-precision metric values, all 12 checks, original artifacts, source hashes, and frozen thresholds are retained in [`experiments/r103_scientific_firstuse_seal.json`](../../experiments/r103_scientific_firstuse_seal.json). Total of three separate whole-process CPU durations **1324.58 seconds**, not shared wall time. No R101/R102 previously scored data or trained checkpoints were reused.

The earlier `8c4e855d6085fcf082d89eeb3fc6d8946b6fad8a` owner-approval attempt FAILED full regression due to three manifest-presence unit tests and did not consume any original data; original seed jobs were SKIPPED. The tests were fixed and fully requalified at `aecf31955cfff05f447f457ed9fc59b67e043091` before exact one-file approval at `427d964fb72ecb30d4212bd11b8fbb13511e88e3`. These are **engineering** preflight failures, not scientific negatives.

## Terminal disposition

**Engineering** original experiment SUCCESS, followed by an irreversible source replay fence and pinned spent-original workflow guard. Post-seal Linux/Windows G4, G5, G7 must still qualify the new seal commit.

**Science** NO-GO: all three seeds fail primary unfamiliar-rule decision accuracy despite bounded positive memory/feedback signs. No post-scoring threshold tuning; keep every failure.

**Deployment** HOLD: draft PR unmerged. No GPU original use, model promotion, FORGE integration, new controller/runner/scheduler or paid API.

**Research direction:** A new separately frozen R104 must isolate instability in causal-role representation and query-action decisions, compare truly matched conventional baselines and preserve independent source first-use rules. Do not replay R103 cohorts.
