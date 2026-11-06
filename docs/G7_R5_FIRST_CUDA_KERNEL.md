# G7-R5 — first authentic CUDA kernel smoke (engineering evidence)

**Status:** `PASS_ONE_BOUNDED_NATIVE_CUDA_MATMUL_ONLY` at source SHA `067bf951a89e7f61ad9b50cc1491566129ee0768`. **Scientific:** `UNVERIFIED`; **deployment:** `NOT_AUTHORIZED`.

## Immutable execution evidence

- Authorized first/only dedicated runner workflow: [G7-R5 run 42228676232](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42228676232), `run_number=1`, `run_attempt=1`, job `first-native-kernel` SUCCESS. Existing isolated `TTT-HYBRID-GPU-WIN`; no other runner/controller.
- Artifact ID `13506388398` with file `g7_r5_kernel.json` (30-day retention), GitHub ZIP SHA-256 **`1631b00d0cc0bf6b55c34c00e44901abbb3fe4be68a837f5ef6375b460d3500c`**. ZIP checksum independently computed and matched exactly.
- Inner native receipt digest: `28d23b6f42755e99af431c098cfb8a9a088b3aba4b3caf12aa62be6edf55ab43`; only sanitized outer receipt is uploaded.
- R3 installation/source baseline: `06a01c2c3a980eeb806a7ecb7cca1979a431f80c`; pinned environment package manifest SHA-256 `b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df`.
- Original upstream source commit and Git blob custodianship:
  - Mamba-2: `e9594ce1c732d97440f0332fdc43170a2294dbfa`, module `mamba_ssm/modules/mamba2.py` blob `36b16d4715ed2d753697167d65da5e6abba22017`.
  - TTT-Linear: `cd831db10c8c9a0f6340f02da5613316a8a92b67`, module `ttt.py` blob `da267f4a88196fd4709fa1e5e62235a3fa746ec6`.
  - Both checkout trees clean, source imports matched exact source locations, project checkout SHA matched.

## Observations and resource controls

| Field | Measured/verified |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability `12.0` (`sm_120`) |
| Runtime | Isolated PyTorch `2.9.0+cu128`, Triton `3.5.0`, CUDA build `12.8` |
| Operation | One 16×16 float32 matrix multiplication with synchronization and finite-result check |
| Native status | `PASS_NATIVE_CUDA_KERNEL_ONLY` |
| GPU memory | 8,523,264 bytes peak PyTorch allocation (about 8.13 MiB) |
| Admission | Minimum 4 GiB free VRAM; PyTorch allocator ceiling 1 GiB |
| Runtime budgets | 175-second isolated child, 240-second WSL process timeout, 7-minute job limit |
| Native Mamba/TTT forward | `NOT_RUN` |
| Training/system installation/native extension build | `NOT_RUN` |
| Scientific qualification / deployment | `UNVERIFIED` / `NOT_AUTHORIZED` |

PyTorch's per-process allocator fraction does **not** guarantee a whole-process GPU memory cap, especially for non-PyTorch allocations. The only observed proof is the reported PyTorch allocator peak in this successful run.

The finite-result check confirms basic native CUDA execution but **does not** establish numerical parity against a CPU reference, authentic Mamba-2/TTT CUDA integration, long-context retrieval, learning, retained adaptation, efficiency, or generalization.

## Post-run control

The consumed `.github/workflows/g7-r5-single-cuda-kernel.yml` one-shot execution entrypoint was removed in the subsequent bounded cleanup PR. Its source-bound wrapper `scripts/g7_r5_oneshot.py`, opt-in existing qualification script and static negative-control tests remain for audit; they do **not** authorize a second execution.

Next substantive gate, if separately authorized: smallest authentic integrated Mamba-2 + TTT forward with frozen small inputs, exact origin checks, GPU memory/time limits, negative controls, and immutable receipts. Do not train or promote models based on a kernel smoke.
