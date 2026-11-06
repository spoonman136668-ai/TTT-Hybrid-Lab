# G7-R11 — exact-message Triton cause discriminator: prepared / not executed

**Baseline:** project main `841f3bfa74a4b2e1fbb63922ae2be1786f871bec`, all five required G4/G5/Mamba Linux/Windows CPU checks PASS. No other G7 diagnostic PR is open. This is one bounded **read-only diagnostic**, not a new controller or GPU qualification.

## Why the existing evidence cannot decide a fix

G7-R9 failed native authentic forward under original SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`. G7-R10 original SHA256-verified receipt recovery [run #42302765535](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42302765535) yielded `RuntimeError`, `TRITON_COMPILER_EXCEPTION`, last allowlisted basename `driver.py`. That classifier matched generic traceback text and did not distinguish Python-side CUDA initialization, a missing C compiler, or PTX/CUDA code generation.

Pinned Triton **v3.5.0** [upstream tag commit `c3c476f357f1e9768ea4e45aa5c17528449ab9ef`](https://github.com/triton-lang/triton/tree/v3.5.0):
- `python/triton/runtime/build.py` raises a specific RuntimeError if GCC/Clang is absent; it also has a distinct failure loading a newly compiled helper.
- `python/triton/runtime/driver.py` raises a specific RuntimeError if the number of active drivers differs from one.
- `third_party/nvidia/backend/driver.py` constructs `CudaUtils` using `compile_module_from_src` on driver.c; its origin basename collides with `runtime/driver.py`.
- `python/triton/compiler/compiler.py` participates in PTX compile/kernel load; its presence in a traceback is not proof that PTX failed.

## Minimal test

One owner-authorized read-only PowerShell step on original `TTT-HYBRID-GPU-WIN` working directory, **no checkout, WSL, CUDA, Python or model execution**. The step verifies original SHA and exact native and nested G3 receipt SHA256s before processing any exception text in memory. It matches only explicit `error_message` strings against whitelisted source-defined categories, disambiguates allowed full module paths from the existing traceback tail, and exports **only sanitized codes**, no raw message, traceback or path. Unknown => `UNCLASSIFIED_OR_INSUFFICIENT_DATA` and root cause `NOT_YET_CONFIRMED`.

Original G7-R9 [run 42300670454](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42300670454) source SHA `cf207e3d2d3e02b2803548a0ebc74e3bd6511744`; native SHA256 `b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65`; G3 SHA256 `c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87`, source-pins SHA256 `44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768`. The original `g7_r6_native_internal.json` filename is historical but its SHA and contents bind to G7-R9; old stale run ID is corrected to 42300670454.

## Qualification and authority

Exact-head five G4/G5/Mamba cross-platform CPU checks must PASS. **This PR is not authorized for merge yet**: merging the main-push-only workflow starts one runner-side PowerShell job (first run and attempt only) and requires separate express approval. Never rerun an existing consumed job. No installs, downloads, GPU retry, WSL launch, scientific or deployment work. Retire workflow after any authorized single use and preserve sanitized receipt.

ENGINEERING: source candidate only, authentic GPU forward FAIL and exact root cause UNVERIFIED. SCIENTIFIC: UNVERIFIED. DEPLOYMENT: NOT AUTHORIZED.
