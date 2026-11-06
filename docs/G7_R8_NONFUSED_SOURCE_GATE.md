# G7-R8 — scoped official non-fused forward-only candidate (source-only)

**Frozen starting protected-source identity:** main `d1f724fbf736215da39513d3d4e23dfe03768194`.

**Evidence:** G7-R6 original execution `d028badf39f7e3392adc24329be4152cdcdc00cd` failed in the fused Mamba2 forward. One-time G7-R7 read-only recovery [run 42285407805](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42285407805) identified `TypeError`, signature `FUSED_CAUSAL_CONV1D_NOT_CALLABLE` in pinned official `ssd_combined.py`. Original receipt SHA-256 and G3 nested receipt SHA-256 were verified; raw exception/host paths were not exported.

## Hypothesis and bounded change

**Hypothesis:** For the tiny forward-only authentic hybrid, selecting pinned official `Mamba2(use_mem_eff_path=False)` can bypass the missing fused `causal_conv1d_fwd_function`. Source `state-spaces/mamba` commit `e9594ce1c732d97440f0332fdc43170a2294dbfa` has a separate non-fused path. When `causal_conv1d_fn` is unavailable, this path calls `self.conv1d(...)` through PyTorch and then the official `mamba_chunk_scan_combined` state scan. **The Triton/CUDA scan and native integration are not yet qualified.** This does not install/repair a missing binary or prove semantic equivalence to the fused path.

- In `authentic_hybrid.AuthenticHybrid`, a strict boolean `mamba_mem_eff_path` selector defaults to the original `True`. No change to official source/checkpoint.
- In `run_gate3.run`, forward-only explicitly passes `False`. The existing full engineering route still uses `True`. The receipt records which route was selected.
- Pinned official Mamba-2 CPU CI checks both constructors and the exact original upstream source markers for the branch, Conv1d fallback, and state scan. No CUDA/model forward in constructor CI.
- Native CPU engineering tests use declared doubles to check both route choices, unchanged causal/episode-reset negative controls, and no-gradient execution.

## Acceptance boundary

Required: all G4/G5 and pinned-Mamba CPU CI jobs PASS at exact candidate SHA on Linux and Windows; then separately accept/merge source-only if qualified. No new GPU run, WSL launch, runner/controller, system installation, scientific benchmark, or deployment is part of this PR.

**ENGINEERING:** source candidate only; authentic GPU forward UNVERIFIED after previously FAILING fused path. **SCIENTIFIC:** UNVERIFIED. **DEPLOYMENT:** NOT AUTHORIZED. Subsequent independent GPU retest requires a separate authorization and its own bounded source/receipt gate.
