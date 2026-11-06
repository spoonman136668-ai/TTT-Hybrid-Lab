# G7-R4 runtime admission — source-bound engineering receipt

**Result:** `PASS_CUDA_DEVICE_AND_PINNED_IMPORTS_NO_KERNELS` (engineering only).

## Immutable evidence

- Workflow: [G7-R4 first and only run #42216104266](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42216104266), `run_number=1`, `run_attempt=1`, job `runtime-admission` SUCCESS.
- Exact tested project SHA: `6fd53d073f0b1fe8180dbec47ae38c2e710cadec`.
- Artifact ID: `13500922860`, file `g7_r4_runtime.json` in ZIP; observed/published ZIP SHA-256: `6b3d34be48c9f9f89074f935acd64c09bf9aa68c09b6bbfb68c629f11453792c`.
- Previous qualified G7-R3 environment candidate: `06a01c2c3a980eeb806a7ecb7cca1979a431f80c`.
- G7-R3 pinned package-manifest SHA-256: `b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df`.
- `source_pins.json` SHA-256: `44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768`.

## Measured metadata

| Field | Result |
| --- | --- |
| Runtime | `torch 2.9.0+cu128`, CUDA build `12.8`, Triton `3.5.0` |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| Compute capability | `12.0` (`sm_120`, present in PyTorch architecture list) |
| VRAM at admission | 11,026 MiB free / 12,226 MiB total |
| Mamba-2 import | PASS, class origin matches exact pinned source `e9594ce1c732d97440f0332fdc43170a2294dbfa` |
| TTT-Linear import | PASS, class origin matches exact pinned source `cd831db10c8c9a0f6340f02da5613316a8a92b67` |
| Required Git blobs | Mamba-2 `36b16d4715ed2d753697167d65da5e6abba22017`; TTT `da267f4a88196fd4709fa1e5e62235a3fa746ec6` |

Read-only package/source checks were performed on isolated `TTT-HYBRID-GPU-WIN` under Ubuntu 24.04 WSL2. The first and only workflow was retired after its receipt; `scripts/g7_r4_runtime_probe.py` and static tests were retained.

## Qualification boundaries

- **ENGINEERING:** CUDA device metadata and exact-source imports PASS; no model forward, authentic GPU kernel, native extension compilation, or integrated Mamba-2/TTT smoke was attempted.
- **SCIENTIFIC:** UNVERIFIED — no learning, transfer, generalization, retention, efficiency, or matched control measurement.
- **DEPLOYMENT:** NOT AUTHORIZED.
- **Next gate:** separately authorize a smallest bounded native CUDA kernel and integrated source smoke with explicit VRAM and time limits, negative controls, and immutable receipts. No training, performance claim, or integration with Wingless/Yggdrasil without independent evidence.

This document retains provenance when the GitHub Actions artifact retention expires; the artifact is the original machine-readable record.
