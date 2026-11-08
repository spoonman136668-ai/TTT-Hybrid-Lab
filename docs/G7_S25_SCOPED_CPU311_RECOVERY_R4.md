# G7-S25 CPU311 R4 — repair PowerShell 5.1 native Python probe quoting

R3 run [#42430864889](https://github.com/spoonman136668-ai/TTT-Hybrid-Lab/actions/runs/42430864889)
failed in the **read-only preflight** before any installation.
Windows PowerShell 5.1 stripped nested double-quotes from
`python -c` argument `print(".".join(...))`, producing `SyntaxError`
and `WINDOWS_PREFLIGHT_BLOCKED`.

This R4 source-only patch uses a quote-free Python version probe
(`print(tuple(sys.version_info[:3]))`) and a zero-installed-package
count. The later runtime confirmation is also quote-free, using four
separate output lines instead of JSON embedded in `-c`.

R4 retains the exact R2 recovery anchor: valid installer SHA256,
existing scoped Python 3.11.14 and **empty** venv, without overwriting,
re-downloading or repeating earlier package installs. Windows packages
use isolated PyPI dependencies and the CPU torch official index only.
Linux install is still unattempted and stays scoped to nonroot WSL.
Both full exact-S25 regressions and negative controls remain mandatory.

No scientific seed consumption, CUDA work, R3 environment mutation,
research-controller, scheduled task or model training.
