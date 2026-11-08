# G7-S25 R3: scoped CPU 3.11 continuation without cross-index resolution

Authorized installation on the already verified project-owned runner.
The first run (#42430242422) correctly stopped before creating Python.
The second run (#42430603843) installed isolated managed Python 3.11.14
and an empty virtual environment, then failed at uv's **dependency
resolution** before wheel installation.

Exact logs show uv's default first-index selection returned PyPI's
`torch` candidates, where `torch==2.5.1+cpu` does not exist. The
second immutable receipt `13597123535` is
`BLOCKED_DEPENDENCIES`; full tests NOT RUN, WSL install skipped.
This is an **engineering package-index failure**, not a scientific result.

R3 requires the existing pinned Python and empty venv (read-only
`importlib.metadata` check) and reuses exact uv release bytes. It
does NOT redownload, recreate or overwrite them. Installing unknown
previous packages is forbidden.

To eliminate cross-index selection, ordinary pinned `pytest` plus
explicit Torch runtime dependencies resolve **only from PyPI**.
`torch==2.5.1+cpu` then resolves **only from the official PyTorch CPU
index** with `--no-deps`. `uv pip check` and exact Python/Torch/Pytest
version probes run before original full S25 tests. The WSL fresh install
uses the same separately sourced dependencies and correct uv version
prefix allowing signed build metadata. R3 CUDA manifest and all frozen
scientific seeds remain untouched.

A PASS on both platforms is required before any acceptance claim.
No scientific first-use, GPU workload, controller, scheduler, merge or
research promotion is authorized.
