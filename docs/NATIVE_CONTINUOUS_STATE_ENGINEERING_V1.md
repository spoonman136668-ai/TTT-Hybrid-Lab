# Native continuous-state Hybrid — isolated engineering v1

Status: proposed, untrained, engineering-only. This branch is stacked on the documentation-only token-free North Star PR #47 at exact base commit `74986eaf70d9aeabbcaaa8e4f9e483b73ecfba7b`. It is **not** a modification of the frozen S24/S25/S26/S27 science lineages.

## What this actually implements

- Inputs are explicit floating-point physical sensor measurements, interventions/actions, and positive elapsed time; no lexical indices, embedding table, tokenization, autoregressive next-token objective, or token-based fast memory.
- A continuous-time-gated recurrent state, attention retrieval over a bounded history of native states, explicit cross-call recurrent/fast-weight state, and a gradient-compatible continuous next-observation prediction objective.
- The fast-weight update is a separate after-outcome API, not an automatic use of future observations at prediction time. Calls must respect physical event chronology; the current prototype cannot itself authenticate when a real-world outcome was acquired.
- A caller can serialize the model weights plus explicit state and restore them for deterministic CPU inference. This is an API property, **not** experimental long-term retention, transfer, or learning.
- Synthetic smoke tests validate shape/finite/gradient behavior, intervention sensitivity, delayed updates, bounded history, batch isolation, cold restore, nonnumeric exclusion, and lack of token-layer primitives in this new module.

## Limitations and falsification requirements

- No model trained on real actions or causal environments; no measured learning advantage, generalization, transfer, robust reasoning, external JEPA, real Mamba2 kernel, native language ability, fast-throughput claim, RSI, or deployment.
- The new path is an original small differentiable mechanism reference; it is not an official Mamba2/TTT implementation or completed final cognitive architecture.
- The negative token check is scoped to the **new module**, not the entire historical repository. Existing token-conditioned pilots remain separately labeled comparison/diagnostic evidence.
- Future science must preregister: independently new source/seeds, matched conventional GRU/KRR/SSM/TTT controls, true physical interventions, cold-retention and interference, unfamiliar-world transfer, whole-process compute, action-effect error, strict stop criteria, custody and independent qualification. No S26-V2 seeds or heldout material may be reused.
- No S27 first-use, frozen protocol edit, runner installation, new controller, scheduler, merge, promotion, GPU workload or paid API is part of this code change.

## Reproducibility and acceptance

Initial local exploratory test: Python 3.13.5, PyTorch 2.10.0+cpu, `python -m pytest -q tests/test_native_continuous_state.py` = **7 PASS**. This is development evidence only, **not** cross-platform exact-head qualification.

Before accepting this candidate, complete exact final-SHA Windows plus WSL Linux full repository regression and original-source negative controls on the existing authorized project-owned environment. An engineering PASS still does not authorize scientific first-use or promotion.
