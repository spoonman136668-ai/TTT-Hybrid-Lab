# G6: TTT versus conventional memory mechanisms — frozen development screen

## Objective

Test whether **the authentic pinned TTT-Linear implementation's test-time
updates** help a small model infer an episode-specific signed slope from
observed examples. Compare against its exact same architecture with updates
disabled, causal attention, and a GRU. This is a **development screen**, not an
independent held-out discovery and not the final Mamba–attention–MoD model.

## Frozen design (before results)

- Source: official TTT-LM PyTorch commit
  `cd831db10c8c9a0f6340f02da5613316a8a92b67`, exact clean checkout
  and Git blob admission. Model source is not copied or patched.
- Frozen protocol and numeric thresholds are in `scripts/g6_protocol.py`.
- Three initial seeds: `17,29,43`. Each candidate trains for exactly
  **96 AdamW updates** with **16 episodes per batch**, learning rate `0.002`,
  clipping at 1, deterministic per-step train source. A fixed separate
  development validation source has **8 batches × 16 episodes per seed**.
- Episode: 15 observed pairs `(x_t, y_t = a*x_t)` followed by query
  `(x_q, unknown y_q)`, where each episode has a fresh randomly signed
  `a`. Inputs contain the earlier observed pairs but **never** the query
  label. Predict `a*x_q` at the last token. CPU, PyTorch float32.
- Four models: `ttt_live` (native TTT-Linear, update LR 1), `ttt_off`
  (same module/initial weights/parameter count, update LR 0), single causal
  self-attention, and conventional GRU. All see identical train/validation
  examples and training step budgets.
- Primary metric: development validation **MSE**. Secondary: scrambling
  support response values between episodes without changing queries or labels,
  training wall-time, parameter count, and paired-seed exploratory CI.
- Exact source SHA, source pins, protocol hash, PR candidate SHA, versions,
  all per-seed arm outcomes, method comparisons and negative controls are
  captured in the native GitHub Actions receipt. Results are retained even if
  the hypothesis fails. **No optimization/tuning after seeing results.**

## Prespecified development hypothesis

TTT must have mean development MSE ≤ 0.85× the frozen no-update model
and ≤ 0.90× the better of attention/GRU. It must beat no-update on
each seed by ≥ 5%, experience ≥ 10% MSE degradation on support scramble
(as evidence it actually uses context), have ≤ 1.25× parameters and
≤ 1.25× total training time versus the better conventional model.

If any condition fails, report **NO_GO_DEVELOPMENT_SCREEN**. This is not
an engineering execution failure: a clean negative result is the useful
scientific outcome. The CPU/GPU requirements, parameter count and training
times are explicitly reported, not concealed.

Even a positive development result **does not qualify scientific PASS**:
there is no sealed independent unseen sample, only three seeds, synthetic
episodic data, and no controlled FLOP-matched ablation of the full hybrid.
A future independent, preregistered original evaluation requires separate
authority and full provenance/cost controls.

## Qualification and scope

- G4 full regression and G5 native TTT tests must remain PASS on the same
  G6 SHA on Linux and Windows.
- G6 runs on an isolated GitHub-hosted Ubuntu CPU, triggered automatically by
  the PR; it never reserves the operator's RTX GPU or a FORGE runner.
- No code, workflow or controller changes in FORGE, Wingless, Yggdrasil, KTRADE
  or Mind-Palace. No merge, promotion, deployment or workstation runtime action.
- No local archive transfers; `C:\Research\TTT-Hybrid` is still an existing
  checkout managed through ordinary Git synchronization once accepted.

## Stop criterion

On clean no-go, preserve receipts and stop this specific hypothesis rather
than adding infrastructure or adjusting data/thresholds. Future high-value
work could evaluate genuine Mamba2 CUDA integration independently, with
fairer equal-compute conventional controls. Do not treat a CPU screen as GPU
evidence.
