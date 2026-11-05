# Gate 3: authentic-upstream CUDA integration smoke

**Scope:** Isolated experimental harness. It does **not** alter any repository,
install dependencies, dispatch a workflow, or run live models. The Gate 2 CPU
reference model and tests remain bundled unchanged. The Gate 3 adapter instantiates
pinned official `Mamba2` and `TTTLinear` classes only after exact-source
runtime admission. The conditional-depth router remains a *causal threshold*
router, **not** original fixed-capacity top-k Mixture-of-Depths.

## Current execution status

- **CPU test doubles / fail-closed controls:** see `gate3_test_receipt.json`.
- **Official Mamba-2 + TTT-LM CUDA execution:** `BLOCKED`, no Linux CUDA or
  pinned checkouts are available in this container. See `gate3_authentic_receipt.json`.
- **Model quality, transfer, speedup, efficiency, scientific qualification:** UNVERIFIED.
- **Source custody:** exact upstream Git commits and original file blobs are
  recorded in `source_pins.json`; the harness checks them at runtime. This
  does not yet qualify all compiled native-extension binary dependencies.

## Interface details

- Mamba: `mamba_ssm.modules.mamba2.Mamba2(d_model=64, d_state=64,
  headdim=64, chunk_size=64)`; CUDA fused path.
- Causal attention: native PyTorch scaled-dot-product attention.
- TTT: official `ttt.TTTLinear(ttt.TTTConfig(...), layer_idx=0)`, using
  per-forward episode reset (`cache_params=None`) and explicit causal positions.
- Depth: existing `CausalDepthRouter` conditional FFN (threshold routing),
  forced to run during the smoke test to exercise its gradient path. Not a
  measurement of learned routing, MoD capacity, or inference throughput.

## Local engineering verification (no GPU required)

```bash
python -m pytest -q tests
python run_gate3.py --mamba-root /nonexistent/mamba --ttt-root /nonexistent/ttt
# expected code 2 and receipt status BLOCKED_SOURCE_RUNTIME_ADMISSION
```

## Authentic GPU smoke (requires an existing authorized, qualified CUDA environment)

Set up an isolated Python environment with CUDA-enabled PyTorch, compatible
CUDA/NVIDIA driver, `mamba_ssm` compiled kernels, the `causal-conv1d` runtime,
and TTT-LM dependencies. The upstream TTT-LM PyTorch repository explicitly
calls its pure-PyTorch implementation a naive educational implementation and
warns against using it for performance comparisons; this is engineering smoke only.

Provide two **clean source checkouts** with exact SHAs from `source_pins.json`.
Ensure imports resolve to these checkouts, not arbitrary site-packages wheels.
The upstream Mamba2 GPU kernels may require Linux/WSL2; this gate rejects
Windows and CPU-only environments. In the eligible environment:

```bash
export PYTHONPATH="/path/to/mamba:/path/to/ttt-lm-pytorch:$PYTHONPATH"
python run_gate3.py \
  --mamba-root /path/to/mamba \
  --ttt-root /path/to/ttt-lm-pytorch \
  --receipt gate3_authentic_receipt.json
```

The command fails closed with exit code 2 if admission fails, exit code 1 on
real integration or invariant failures, and 0 only on authentic CUDA engineering
smoke PASS. It never dispatches, trains a model, or triggers acceptance or
promotion. Save the runtime receipt on the source-bound exact SHA.

## Mandatory future scientific gate

After authentic CUDA integration passes, run preregistered matched-budget
baselines (attention, Mamba2, attention+Mamba2, +TTTLinear, +causal depth),
with no-TTT and memory-scramble negative controls; freeze exact data, splits,
seeds, FLOPs/capacity, hardware/binary provenance, and measured VRAM/latency.
Add paper-exact MoD separately if wanted, including causal decoding semantics.
No Wingless or Yggdrasil changes or autonomous controller actions are implied.
