# G7-S25 isolated WSL CPU311 qualification — independent continuation

The operator authorized provisioning the missing exact Python 3.11 CPU
runtime on the existing Hybrid-owned runner, leaving R3 GPU untouched.

Windows CPU environment installation **PASS** in prior R4
[#42430972243](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42430972243):
Python 3.11.14 and torch 2.5.1+cpu, original source SHA
`108808f7b685107a76fae1ce5605b7ff9e2bac27`,
source-pin custody and authentic-source negative control all PASS.

Windows full regression **FAIL** during collection, not a runtime-provision
failure: S20/S25 source imports Unix-only `resource` in at least five test
modules; immutable receipt `13597630972`. No source portability shim,
frozen protocol patch or scientific data change was used. Missing NumPy
generated a nonfatal warning, so this WSL-only independent environment
adds the already source-pinned G5 `numpy==1.26.4`.

The original R4 Windows environment is not touched or rerun.
This independent workflow targets only normal-user Ubuntu 24.04 WSL,
requires original R3 package-manifest SHA256 without modifying it,
isolates managed Python3.11.14 and CPU torch2.5.1+cpu/pytest8.3.4 under
a new nonroot CPU-only location, and runs the original exact-S25
`scripts/qualify_cpu.py` complete suite and negative control.

**Engineering qualification:** require actual successful immutable job
receipt; do not infer PASS from correct installation or focus-only checks.
No CUDA, scientific first-use, root apt, accepted-ref edit, training,
runner re-registration, controller or scheduled job.
