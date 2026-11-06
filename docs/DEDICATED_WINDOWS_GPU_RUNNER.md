# Dedicated TTT-Hybrid Windows runner — one-time registration

**Purpose:** Connect one existing Windows/NVIDIA host **only** to `spoonman136668-ai/TTT-Hybrid-Lab`, not to FORGE, Wingless, Yggdrasil, Mind-Palace or CKB. This is an execution runner registration, **not** a scientific controller, WSL CUDA admission, model run or deployment. The owner has requested one dedicated TTT runner; no new hardware, paid service, GitHub token or resource reservation has been provisioned by this PR.

## Preconditions

- Operator chooses a Windows host they are authorized to administer. Its RTX GPU must have at least 4 GiB free **at installation**, but this observation does not reserve resources or qualify CUDA.
- If it is the same physical computer used by other runners, **different runner labels/directories do not isolate GPU memory or compute**. Keep FORGE and the Wingless/Yggdrasil research loops operating undisturbed; schedule the later CUDA test only after checking actual GPU occupancy and a bounded resource lease.
- Use an **interactive runner session for the operator's Windows user** if WSL2 is required. An account such as NetworkService may not see that user's WSL distributions. This registration script does not install a Windows service or enable automatic runner startup.
- Do not register on another project's runner root. Default new root `C:\actions-runner-ttt-hybrid`. A nonempty/registered root fails closed. Exact repository `https://github.com/spoonman136668-ai/TTT-Hybrid-Lab`, runner name `TTT-HYBRID-GPU-WIN`, additional label `ttt-hybrid-gpu-isolated`.
- GitHub `main` is **currently unprotected**. The existing G7 manual-only workflow **rejects** execution until main is protected by an effective rule, required CPU checks have passed, and the separately accepted G4/G5/G7 source stack is on main. Runner registration is **not** acceptance or permission to weaken this guard.

## Register once

1. On the designated host, open the repository's **Settings → Actions → Runners → New self-hosted runner**, select Windows x64. Obtain the official GitHub runner ZIP **and its published SHA-256**. Download/extract only that trusted ZIP; do not use a random preinstalled executable. Do **not** copy registration tokens into this chat, source control, files, logs or receipts. Tokens expire.
2. Open an ordinary PowerShell terminal **as the intended WSL2 user**. From a verified checkout of this exact qualified TTT source, use:
   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\Register-TTT-GpuRunner.ps1 -RunnerArchive 'C:\Downloads\actions-runner-win-x64-<version>.zip' -ExpectedSha256 '<64-character-official-GitHub-SHA256>'
   ```
   Replace both placeholders with **verified values** shown by GitHub. The script checks ZIP hash, local NVIDIA visibility/VRAM, isolated empty directory, then requests the short-lived **repository registration token** privately and configures the runner without starting it. The GitHub registration utility receives the short-lived token on its process command line, so use a trusted local host and avoid process/command-line logging while registering; do not use a PAT.
3. Inspect the runner in the repo's Actions runner settings. It must be named `TTT-HYBRID-GPU-WIN`, have default `self-hosted, Windows, X64` labels plus `ttt-hybrid-gpu-isolated`, and be bound to **TTT-Hybrid-Lab only**. The package is not a substitute for this GitHub-side registration receipt.
4. When separately authorized to start the dedicated session, run `C:\actions-runner-ttt-hybrid\run.cmd` under the same Windows user. The shell stays open to accept jobs; close it to stop the session. Do not install a service, background scheduler, controller, automatic job or GPU workload as part of this registration.
5. Only **after** protected-main acceptance, use existing manual-only `.github/workflows/g7-existing-runner-admission.yml`. It records a read-only source/host inventory. The workflow must PASS on this runner and record a receipt before separate WSL CUDA environment qualification. It is not evidence of scientific improvement.

## Stop conditions

Invalid archive hash, nonempty root, existing runner registration, missing/ambiguous GPU, insufficient free VRAM, wrong GitHub repo/label, or missing branch protection: **STOP**. Do not borrow or relabel FORGE, Wingless or Yggdrasil runners. No substitute runner, API charge, install loop or automatic dispatch.

The PowerShell helper is source-only until independently executed on an authorized Windows host. This chat's GitHub connection cannot operate a local Windows installer, issue a GitHub registration token, or assert that the runner is online.
