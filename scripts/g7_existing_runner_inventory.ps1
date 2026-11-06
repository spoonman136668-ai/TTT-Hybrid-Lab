# G7: read-only admission on an existing Windows runner granted to TTT-Hybrid.
# NOT a CUDA dispatch, install, GPU kernel run, controller or runner registration.
# Unprotected main is allowed exactly once: owner-dispatched run number 1,
# with exact workflow SHA, a pinned inventory-script blob and source-pins blob.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedSha,
    [Parameter(Mandatory = $true)][string]$ReceiptPath
)
$ErrorActionPreference = 'Stop'

function Invoke-SafeProbe {
    param([string]$CommandName, [string[]]$Arguments)
    $command = Get-Command -Name $CommandName -ErrorAction SilentlyContinue
    if (-not $command) {
        return [ordered]@{ available = $false; exit_code = $null; output = 'NOT_INSTALLED' }
    }
    try {
        $result = & $command.Source @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $message = ($result | Out-String).Trim()
        if ($message.Length -gt 1800) { $message = $message.Substring(0, 1800) }
        return [ordered]@{ available = $true; exit_code = $exitCode; output = $message }
    } catch {
        return [ordered]@{ available = $true; exit_code = $null; output = ('PROBE_ERROR_' + $_.Exception.GetType().Name) }
    }
}

$report = [ordered]@{
    schema = 'ttt-hybrid.g7-existing-runner-admission.v1'
    utc = [DateTime]::UtcNow.ToString('o')
    project = 'spoonman136668-ai/TTT-Hybrid-Lab'
    candidate_sha = $ExpectedSha
    mode = 'HOST_READONLY_INVENTORY'
    runner_name = $env:RUNNER_NAME
    runner_os = $env:RUNNER_OS
    repository = $env:GITHUB_REPOSITORY
    ref = $env:GITHUB_REF
    event = $env:GITHUB_EVENT_NAME
    actor = $env:GITHUB_ACTOR
    run_attempt = $env:GITHUB_RUN_ATTEMPT
    workflow_run_number = $env:GITHUB_RUN_NUMBER
    admission_mode = $env:G7_ADMISSION_MODE
    one_time_exception_used = $false
    windows_gpu = $null
    wsl_registration = $null
    exact_sha = $false
    status = 'BLOCKED_UNVERIFIED'
    authentic_cuda = 'NOT_RUN'
    scientific = 'UNVERIFIED'
    controller_impact = 'NOT_PROBED'
    no_install_or_wsl_launch = $true
}
$exitCode = 2
try {
    if ($env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/TTT-Hybrid-Lab' -or
        $env:GITHUB_REF -cne 'refs/heads/main' -or
        $env:GITHUB_EVENT_NAME -cne 'workflow_dispatch' -or
        $env:GITHUB_ACTOR -cne 'spoonman136668-ai' -or
        $env:GITHUB_RUN_ATTEMPT -cne '1' -or
        $env:RUNNER_OS -cne 'Windows') {
        throw 'UNAUTHORIZED_REPO_REF_ACTOR_OR_RUNNER'
    }
    # Preserve the normal protected-main route. The exception must be
    # explicitly selected and is valid only for the first workflow run.
    $normalAdmission = ($env:GITHUB_REF_PROTECTED -ceq 'true' -and
        $env:G7_ADMISSION_MODE -ceq 'protected-main' -and
        $ExpectedSha -ceq $env:GITHUB_SHA)
    $oneTimeAdmission = ($env:GITHUB_REF_PROTECTED -cne 'true' -and
        $env:G7_ADMISSION_MODE -ceq 'g7-readonly-once' -and
        $env:GITHUB_RUN_NUMBER -ceq '1' -and
        $ExpectedSha -ceq $env:GITHUB_SHA)
    if (-not ($normalAdmission -or $oneTimeAdmission)) {
        throw 'UNAUTHORIZED_READONLY_EXCEPTION_OR_SHA'
    }
    if ($ExpectedSha -cnotmatch '^[0-9a-f]{40}    if ([IO.Path]::GetFullPath($ProjectRoot) -cne [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE)) {
        throw 'UNAUTHORIZED_WORKTREE_PATH'
    }
    $observed = (git -C $ProjectRoot rev-parse HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $observed -cne $ExpectedSha) { throw 'CHECKOUT_SHA_DRIFT' }
    if ((git -C $ProjectRoot status --porcelain | Out-String).Trim().Length -ne 0) {
        throw 'DIRTY_SOURCE_TREE'
    }
    if ($oneTimeAdmission) {
        # Do not execute an unreviewed variation of the host inventory script.
        $scriptBlob = (git -C $ProjectRoot rev-parse HEAD:scripts/g7_existing_runner_inventory.ps1 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $env:G7_REQUIRED_SCRIPT_BLOB -cnotmatch '^[0-9a-f]{40}
    # This PowerShell probe deliberately NEVER invokes wsl.exe. On a machine
    # without WSL installed, even `wsl --version` can offer to install it.
    # Registry-only inventory is safe and does not start a Linux distribution.
    $wslDistros = @()
    $key = 'Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path -LiteralPath $key) {
        foreach ($node in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
            $value = Get-ItemProperty -LiteralPath $node.PSPath -ErrorAction SilentlyContinue
            if ($value) {
                $wslDistros += [ordered]@{
                    name = [string]$value.DistributionName
                    version = $value.Version
                }
            }
        }
    }
    $report.wsl_registration = [ordered]@{
        scope = 'RUNNER_PROCESS_USER_HKCU_REGISTRY_ONLY'
        distro_count = $wslDistros.Count
        distros = $wslDistros
        runtime_cuda = 'NOT_PROBED'
    }
    $report.windows_gpu = Invoke-SafeProbe -CommandName 'nvidia-smi.exe' -Arguments @(
        '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader,nounits'
    )
    if (-not $report.windows_gpu.available -or $report.windows_gpu.exit_code -ne 0 -or
        [string]::IsNullOrWhiteSpace($report.windows_gpu.output)) {
        $report.status = 'BLOCKED_WINDOWS_GPU_NOT_VISIBLE'
    } else {
        $report.status = 'PASS_RUNNER_INVENTORY_ONLY'
        $exitCode = 0
    }
} catch {
    $report.status = 'BLOCKED_RUNNER_IDENTITY_OR_CHECKOUT'
    $report.failure = [string]$_.Exception.Message
} finally {
    # Receipt stays inside the checked-out job workspace.
    $fullReceipt = [IO.Path]::GetFullPath($ReceiptPath)
    $fullRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $fullReceipt.StartsWith($fullRoot + [IO.Path]::DirectorySeparatorChar,
                                   [StringComparison]::OrdinalIgnoreCase)) {
        throw 'RECEIPT_OUTSIDE_JOB_WORKSPACE'
    }
    $parent = Split-Path -Parent $fullReceipt
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($fullReceipt, $json + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Write-Output $json
}
exit $exitCode
) {
        throw 'INVALID_OR_MISMATCHED_GITHUB_SHA'
    }
    $report.one_time_exception_used = $oneTimeAdmission
    if ([IO.Path]::GetFullPath($ProjectRoot) -cne [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE)) {
        throw 'UNAUTHORIZED_WORKTREE_PATH'
    }
    $observed = (git -C $ProjectRoot rev-parse HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $observed -cne $ExpectedSha) { throw 'CHECKOUT_SHA_DRIFT' }
    if ((git -C $ProjectRoot status --porcelain | Out-String).Trim().Length -ne 0) {
        throw 'DIRTY_SOURCE_TREE'
    }
    $report.exact_sha = $true
    # This PowerShell probe deliberately NEVER invokes wsl.exe. On a machine
    # without WSL installed, even `wsl --version` can offer to install it.
    # Registry-only inventory is safe and does not start a Linux distribution.
    $wslDistros = @()
    $key = 'Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path -LiteralPath $key) {
        foreach ($node in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
            $value = Get-ItemProperty -LiteralPath $node.PSPath -ErrorAction SilentlyContinue
            if ($value) {
                $wslDistros += [ordered]@{
                    name = [string]$value.DistributionName
                    version = $value.Version
                }
            }
        }
    }
    $report.wsl_registration = [ordered]@{
        scope = 'RUNNER_PROCESS_USER_HKCU_REGISTRY_ONLY'
        distro_count = $wslDistros.Count
        distros = $wslDistros
        runtime_cuda = 'NOT_PROBED'
    }
    $report.windows_gpu = Invoke-SafeProbe -CommandName 'nvidia-smi.exe' -Arguments @(
        '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader,nounits'
    )
    if (-not $report.windows_gpu.available -or $report.windows_gpu.exit_code -ne 0 -or
        [string]::IsNullOrWhiteSpace($report.windows_gpu.output)) {
        $report.status = 'BLOCKED_WINDOWS_GPU_NOT_VISIBLE'
    } else {
        $report.status = 'PASS_RUNNER_INVENTORY_ONLY'
        $exitCode = 0
    }
} catch {
    $report.status = 'BLOCKED_RUNNER_IDENTITY_OR_CHECKOUT'
    $report.failure = [string]$_.Exception.Message
} finally {
    # Receipt stays inside the checked-out job workspace.
    $fullReceipt = [IO.Path]::GetFullPath($ReceiptPath)
    $fullRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $fullReceipt.StartsWith($fullRoot + [IO.Path]::DirectorySeparatorChar,
                                   [StringComparison]::OrdinalIgnoreCase)) {
        throw 'RECEIPT_OUTSIDE_JOB_WORKSPACE'
    }
    $parent = Split-Path -Parent $fullReceipt
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($fullReceipt, $json + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Write-Output $json
}
exit $exitCode
 -or
            $scriptBlob -cne $env:G7_REQUIRED_SCRIPT_BLOB) {
            throw 'EXCEPTION_SCRIPT_BLOB_DRIFT'
        }
        $pinsBlob = (git -C $ProjectRoot rev-parse HEAD:source_pins.json 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $pinsBlob -cne '1438b51f8548db6bfe7e04805ba8c156e4b5b630') {
            throw 'EXCEPTION_SOURCE_PINS_DRIFT'
        }
    }
    $report.exact_sha = $true
    # This PowerShell probe deliberately NEVER invokes wsl.exe. On a machine
    # without WSL installed, even `wsl --version` can offer to install it.
    # Registry-only inventory is safe and does not start a Linux distribution.
    $wslDistros = @()
    $key = 'Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path -LiteralPath $key) {
        foreach ($node in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
            $value = Get-ItemProperty -LiteralPath $node.PSPath -ErrorAction SilentlyContinue
            if ($value) {
                $wslDistros += [ordered]@{
                    name = [string]$value.DistributionName
                    version = $value.Version
                }
            }
        }
    }
    $report.wsl_registration = [ordered]@{
        scope = 'RUNNER_PROCESS_USER_HKCU_REGISTRY_ONLY'
        distro_count = $wslDistros.Count
        distros = $wslDistros
        runtime_cuda = 'NOT_PROBED'
    }
    $report.windows_gpu = Invoke-SafeProbe -CommandName 'nvidia-smi.exe' -Arguments @(
        '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader,nounits'
    )
    if (-not $report.windows_gpu.available -or $report.windows_gpu.exit_code -ne 0 -or
        [string]::IsNullOrWhiteSpace($report.windows_gpu.output)) {
        $report.status = 'BLOCKED_WINDOWS_GPU_NOT_VISIBLE'
    } else {
        $report.status = 'PASS_RUNNER_INVENTORY_ONLY'
        $exitCode = 0
    }
} catch {
    $report.status = 'BLOCKED_RUNNER_IDENTITY_OR_CHECKOUT'
    $report.failure = [string]$_.Exception.Message
} finally {
    # Receipt stays inside the checked-out job workspace.
    $fullReceipt = [IO.Path]::GetFullPath($ReceiptPath)
    $fullRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $fullReceipt.StartsWith($fullRoot + [IO.Path]::DirectorySeparatorChar,
                                   [StringComparison]::OrdinalIgnoreCase)) {
        throw 'RECEIPT_OUTSIDE_JOB_WORKSPACE'
    }
    $parent = Split-Path -Parent $fullReceipt
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($fullReceipt, $json + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Write-Output $json
}
exit $exitCode
) {
        throw 'INVALID_OR_MISMATCHED_GITHUB_SHA'
    }
    $report.one_time_exception_used = $oneTimeAdmission
    if ([IO.Path]::GetFullPath($ProjectRoot) -cne [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE)) {
        throw 'UNAUTHORIZED_WORKTREE_PATH'
    }
    $observed = (git -C $ProjectRoot rev-parse HEAD 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $observed -cne $ExpectedSha) { throw 'CHECKOUT_SHA_DRIFT' }
    if ((git -C $ProjectRoot status --porcelain | Out-String).Trim().Length -ne 0) {
        throw 'DIRTY_SOURCE_TREE'
    }
    $report.exact_sha = $true
    # This PowerShell probe deliberately NEVER invokes wsl.exe. On a machine
    # without WSL installed, even `wsl --version` can offer to install it.
    # Registry-only inventory is safe and does not start a Linux distribution.
    $wslDistros = @()
    $key = 'Registry::HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss'
    if (Test-Path -LiteralPath $key) {
        foreach ($node in (Get-ChildItem -LiteralPath $key -ErrorAction SilentlyContinue)) {
            $value = Get-ItemProperty -LiteralPath $node.PSPath -ErrorAction SilentlyContinue
            if ($value) {
                $wslDistros += [ordered]@{
                    name = [string]$value.DistributionName
                    version = $value.Version
                }
            }
        }
    }
    $report.wsl_registration = [ordered]@{
        scope = 'RUNNER_PROCESS_USER_HKCU_REGISTRY_ONLY'
        distro_count = $wslDistros.Count
        distros = $wslDistros
        runtime_cuda = 'NOT_PROBED'
    }
    $report.windows_gpu = Invoke-SafeProbe -CommandName 'nvidia-smi.exe' -Arguments @(
        '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader,nounits'
    )
    if (-not $report.windows_gpu.available -or $report.windows_gpu.exit_code -ne 0 -or
        [string]::IsNullOrWhiteSpace($report.windows_gpu.output)) {
        $report.status = 'BLOCKED_WINDOWS_GPU_NOT_VISIBLE'
    } else {
        $report.status = 'PASS_RUNNER_INVENTORY_ONLY'
        $exitCode = 0
    }
} catch {
    $report.status = 'BLOCKED_RUNNER_IDENTITY_OR_CHECKOUT'
    $report.failure = [string]$_.Exception.Message
} finally {
    # Receipt stays inside the checked-out job workspace.
    $fullReceipt = [IO.Path]::GetFullPath($ReceiptPath)
    $fullRoot = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $fullReceipt.StartsWith($fullRoot + [IO.Path]::DirectorySeparatorChar,
                                   [StringComparison]::OrdinalIgnoreCase)) {
        throw 'RECEIPT_OUTSIDE_JOB_WORKSPACE'
    }
    $parent = Split-Path -Parent $fullReceipt
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $json = $report | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText($fullReceipt, $json + [Environment]::NewLine,
                           [Text.UTF8Encoding]::new($false))
    Write-Output $json
}
exit $exitCode
