# TTT / selective SSM / causal attention / conditional depth — research pilot

## Strict token-free cognitive North Star (target, not current implementation)

**All three research programs—Wingless, Yggdrasil and Hybrid—must be token-free in their final native cognition.** Hybrid must not use token IDs, linguistic tokenizers/vocabularies, subword/BPE segmentation, next-token objectives, token-indexed cognitive memory, token-sequence reasoning, or autoregressive token generation anywhere in its learned path. This is an explicit restriction on the architecture, not merely on which loss is reported.

Hybrid's independent mechanism family can retain **SSM/Mamba-style dynamics, causal attention, test-time learning and conditional computation only when operating on directly learned, non-tokenized perceptual/causal state and event streams**. A continuous state vector derived from token embeddings does not establish token-free cognition. The system must learn genuine raw-observation → native representation → causally conditioned decision/action/reasoning, with memory persistence, retention, unfamiliar-world transfer and resource efficiency. Text may be a modality/communication channel, but a tokenizer cannot mediate the native cognitive path.

**Existing prototypes below are token-conditioned/tokenwise research scaffolds and are NOT yet North-Star-compliant.** Frozen S24/S25/S26 and official Mamba2/TTT engineering studies remain preserved as historical baselines/mechanism diagnostics, not automatically authorized token-free scientific training. Build and preregister a separately versioned non-tokenized candidate before claiming North-Star progress. See [Hybrid token-free research North Star](docs/TOKEN_FREE_COGNITIVE_NORTH_STAR.md).

**Scope:** Isolated PyTorch CPU reference, not a pretrained model, not a scientific performance claim, and not a modification of Wingless or Yggdrasil.

This is a _non-production_ design to establish composability and invariants before costly experiments.

- `SelectiveStateBlock`: **Mamba-inspired**, token-conditioned diagonal SSM recurrence, no Mamba selective-scan kernels, causal-convolution stack, or claims of Mamba-equivalent throughput.
- `CausalAttentionBlock`: multihead scaled dot-product causal attention.
- `TTTLinearMemory`: per-episode online fast-weight update with *read before write*; not the official TTT-LM or TTT-E2E implementation.
- `CausalDepthRouter`: tokenwise **threshold** router and sparse FFN execution, not the original global top-k Mixture-of-Depths; no fixed compute budget or guaranteed speedup. Selected-route gradient is only a partial STE, so learning quality of the router remains unverified.

## Run

```bash
python -m pytest -q tests
python benchmark.py
```

The benchmarks compare random untrained models of **different parameter counts**; timings only establish engineering overhead and must **not** be interpreted as a speed comparison or quality evidence. The token router initially has unfrozen random utilization. CPU benchmark environment does not represent the operator's Windows GPU.

## Research next gate (not executed)

**Architecture gate before new North-Star training:** define, implement and separately qualify a native non-token stream/state interface for SSM, attention, TTT and conditional depth. Require a no-tokenizer-dependency negative test over the entire candidate perception/memory/reasoning/output path and a token-free learning objective. Maintain original frozen token-based pilots as separate development/comparator arms; do not silently call their success native token-free progress.


1. Pin real Mamba-2 and TTT-LM/TTT-E2E reference implementations with their dependencies and exact revision SHAs on an authorized Windows/GPU environment.
2. Preregister factorial arms (pure attention, Mamba, hybrid, hybrid+TTT, hybrid+TTT+MoD, and negative controls), **matched params and active FLOPs**, equal training data/steps, frozen seeds, and latency/VRAM limits.
3. Test delayed associative retrieval, update/reversal, long-context distractors, unseen causal composition, retention and transfer; include no-TTT, shuffled-update, zero-memory and false-context controls.
4. Freeze held-out sources before development; independently qualify on exact candidate SHA and Windows/Linux. Measure GPU peak VRAM, tokens/s, p50/p95 latency, FLOPs, model quality and capacity-aware tradeoffs.
5. Do not promote to Wingless/Yggdrasil unless transfer gains survive controls while preserving their independent architecture requirements.

## Primary references

- Sun et al. (ICML 2025), Learning to (Learn at Test Time): https://proceedings.mlr.press/v267/sun25h.html
- Tandon et al. (2025), End-to-End TTT: https://arxiv.org/abs/2512.23675
- Lenz et al. (ICLR 2025), Jamba: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a9ed43fa31dc8b4a7d7a673d713dcb5f-Abstract-Conference.html
- Raposo et al. (2024), Mixture-of-Depths: https://arxiv.org/abs/2404.02258
- Liu et al. (ICML 2026), TTT with KV binding is secretly linear attention: https://proceedings.mlr.press/v306/liu26ae.html
