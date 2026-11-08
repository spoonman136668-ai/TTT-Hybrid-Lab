# G7-S25 versioned native Windows portability — engineering candidate

Source parent: `4e3f0adaf2ece50c824e69fed3ed74937e6da1df`,
previous standalone helper from PR #40 original `108808f7b685107a76fae1ce5605b7ff9e2bac27`.
The helper independently qualified on **native Windows and WSL** in
[run #42432096031](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42432096031)
with artifact `13598455575`. This version wires it into **new engineering
source**, NOT original frozen S25 PR #40.

## Scientific custody and non-equivalence

- Original S20 frozen threshold of `max_cpu_rss_bytes=2147483648`
  remains unchanged in `experiments/g7_s20_frozen_hidden_multicycle.json`.
  The S20 `run` entry refuses Windows or macOS *before loading the frozen
  experiment or consuming any seeds*. On Linux it uses the same exact
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024` value
  through the OS-aware helper, and retains the original
  `FROZEN_RSS_CAP_EXCEEDED` check. S20 physical transition functions
  remain importable on native Windows for S25 engineering-only tests.
- S25 work ledger retains its Linux `process_peak_rss_bytes` semantics.
  Native Windows uses `GetProcessMemoryInfo` with a separate
  `process_peak_working_set_bytes` field, sets RSS to null, and
  `process_peak_memory_comparable_to_linux_ru_maxrss=false`.
  This is **not** cross-OS peak-memory fairness or scientific qualification.
- The original S20 Git blob, referenced by frozen S25 admission/controls,
  **changes in this versioned candidate**, so that original source-pin
  admission MUST remain blocked. Do not update the frozen reference,
  alter the scientific protocol, reclassify the original, or treat this
  engineering patch as a valid frozen S20/S25 scientific first-use run.
- Negative controls cover fake Windows working-set, unchanged Linux
  readings, API failure and fail-closed Windows S20 science. Existing
  original-source science source-pin checks are not relaxed.

## Exact source qualification

Use only existing `TTT-HYBRID-GPU-WIN` on its bounded CPU-only Windows and
Ubuntu-24.04 WSL runtimes. Python 3.11.14, torch 2.5.1+cpu, pytest 8.3.4
and numpy 1.26.4 are required. The isolated Windows CPU311 environment
installed successfully in [run #42430972243](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42430972243),
but had nonfatal missing NumPy and full pytest collection failure from
Unix-only imports. A one-use **isolated-only** NumPy installation is allowed
if absent; never modify R3/CUDA. WSL CPU311 already has NumPy and passed the
original full 191+1 suite in
[run #42431183003](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42431183003).

The *new candidate* requires full exact-head native Windows **and** WSL
pytest, original source-negative-control, unchanged R3 manifest, versioned
receipt, and separate explicit scientific approval before first-use.
If the original frozen source-pin admission flags the S20 change, that is
a required FAIL-CLOSED scientific custody result, **not a test-waiver**.
No new controller, scheduler, GPU experiment, accepted-ref or deployed model.
