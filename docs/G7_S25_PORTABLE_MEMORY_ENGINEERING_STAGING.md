# G7-S25 engineering-only native memory meter, staged

This branch is based exactly on the immutable S25 PR #40 source
`108808f7b685107a76fae1ce5605b7ff9e2bac27`. It adds a **standalone
standard-library helper and five synthetic tests**, but deliberately does NOT
edit or import from original S20/S25 source.

The existing native Windows full regression cannot collect five test modules:
`g7_s20_hidden_multicycle.py` and `g7_s25_cost_ledger.py` import
Unix-only `resource`. See issue #42 and artifact `13597630972`.
Previously established Linux full suite: 191 PASS, 1 SKIPPED, exact S25 SHA,
[run #42431183003](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42431183003).

## Engineering mechanism

The helper returns a typed peak size, **measurement source** and
`comparable_to_linux_ru_maxrss` flag. Linux retains the legacy
`ru_maxrss * 1024` bytes. Windows uses
`GetProcessMemoryInfo(GetCurrentProcess(), PROCESS_MEMORY_COUNTERS)`
to fetch `PeakWorkingSetSize` in bytes and marks the result **not
cross-platform equivalent**. API failure, zero measurements or unsupported
platforms BLOCK. Five isolated synthetic tests cover Unix units,
Windows peak vs current distinction, invalid API, zero and unknown platform.

No source hookup, frozen RSS threshold change or scientific claim is
allowed on this engineering-only branch. In particular:
- The original S20 strict 2 GiB `max_cpu_rss_bytes` criterion continues
  to apply only under its original Unix measurement; scientific evaluation
  on Windows must fail closed unless a separate qualified criterion is
  preregistered.
- S25 may eventually report native Windows working-set bytes *only with
  explicit provenance and non-equivalence*, never as ordinary Unix
  `ru_maxrss`.
- Source hookup to the frozen S20/S25 modules requires explicit owner
  authorization, a new versioned candidate, independent negative controls,
  and full exact-head Linux plus **native Windows** final regression.
- Missing `numpy==1.26.4` in the separately installed Windows CPU311
  environment is also a verified warning; do not claim full qualification
  without resolving it.

No new controller, scheduler, research first-use, CUDA, data, seeds, accepted
ref changes or model training. Existing frozen S25 PR #40 and S26 PR #41
are untouched.

Local development check: five synthetic tests PASS in Python 3.13, **not**
the required native Windows/WSL Python 3.11 final qualification.
