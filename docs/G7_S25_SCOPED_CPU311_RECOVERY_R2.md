# S25 CPU 3.11 pinned runtime recovery R2

This is a scoped, single-use continuation of the operator-approved
isolated Windows/WSL CPU runtime installation. Frozen S25 source:
`108808f7b685107a76fae1ce5605b7ff9e2bac27`.

Original attempted installation [#42430242422](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42430242422)
stopped at `BLOCKED_UV_BINARY` after successful both-platform read-only preflight.
Retained artifact `13596858853`: no Python or full regression ran.

An independent **read-only** diagnostic
[#42430469728](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42430469728)
artifact `13596968669` verified the exact installer ZIP SHA-256
`cdfb04512c9184939f16eea02644a7133b4e41233ead69122d49847d29a45538`,
one `uv.exe` with exit code 0, and actual version
`uv 0.9.13 (7ca92dcf6 2025-11-26)`.
The original script erroneously required an exact shorter display string.

**R2 resumes only from this verified checkpoint.** It requires the original
scoped ZIP with the same digest; exactly one executable; no Python installation
directory or venv; original approved runner and source. It does not download
or extract uv again or overwrite any earlier environment. It accepts the
pinned uv release version followed by optional build metadata.

R2 then attempts CPU Python 3.11.14 and the pinned CPU
`torch==2.5.1+cpu` and `pytest==8.3.4` environment on Windows;
and a separate nonroot WSL installation, preserving original R3 GPU package
manifest. Exact-source full tests and original negative controls must PASS on
both platforms before a qualification claim. Uncertain state blocks with no
retry. No scientific seeds, CUDA, model training, research integration,
owner/controller, scheduled jobs or other projects are changed.
