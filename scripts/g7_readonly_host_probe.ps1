# Read-only host admission probe for the owner's existing RTX GPU.
# No installations, WSL distro creation, driver changes, jobs, or model runs.
# PowerShell 5.1+ compatible.
[CmdletBinding()]
param(
    [string]$ProjectRoot = 'C:\Research\TTT-Hybrid',
    [string]$ReportPath = ''
)
$ErrorActionPreference = 'Continue'

function Invoke-Probe {
    param([string]$Executable, [string[]]$Arguments)
    try {
        $command = Get-Command $Executable -ErrorAction SilentlyContinue
        if (-not $command) {
            return [ordered]@{ available = $false; exit_code = $null; output = 'COMMAND_NOT_FOUND' }
        }
        $lines = & $Executable @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $output = ($lines | Out-String).Trim()
        if ($output.Length -gt 5000) { $output = $output.Substring(0, 5000) }
        return [ordered]@{ available = $true; exit_code = $exitCode; output = $output }
    } catch {
        return [ordered]@{ available = $true; exit_code = $null; output = ('PROBE_EXCEPTION_' + $_.Exception.GetType().Name) }
    }
}

$report = [ordered]@{
    schema = 'ttt-hybrid.g7-readonly-host-probe.v1'
    collected_utc = [DateTime]::UtcNow.ToString('o')
    scope = 'READ_ONLY_MACHINE_DISCOVERY'
    project_root = $ProjectRoot
    commands_install_or_execute_model = $false
    local_checkout = [ordered]@{
        exists = (Test-Path -LiteralPath $ProjectRoot -PathType Container)
    }
    nvidia_windows = Invoke-Probe 'nvidia-smi.exe' @('--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader,nounits')
    wsl_version = Invoke-Probe 'wsl.exe' @('--version')
    wsl_distributions = Invoke-Probe 'wsl.exe' @('--list','--verbose')
}

if ($report.local_checkout.exists) {
    $report.local_checkout.git_head = Invoke-Probe 'git.exe' @('-C', $ProjectRoot, 'rev-parse', 'HEAD')
    $report.local_checkout.git_status = Invoke-Probe 'git.exe' @('-C', $ProjectRoot, 'status', '--porcelain')
}

# Do not launch a WSL distribution automatically. It can have startup hooks
# or consume resources, and an existing project loop may be running there.
# Following receipt review, a separately authorized bounded WSL/CUDA probe
# may be executed without installing drivers or other programs.
$report.wsl_gpu_runtime = 'NOT_PROBED_REQUIRES_SEPARATE_APPROVAL'
$report.native_cuda_smoke = 'NOT_RUN'
$report.scientific_qualification = 'UNVERIFIED'
$json = $report | ConvertTo-Json -Depth 7
if ($ReportPath) {
    $target = [IO.Path]::GetFullPath($ReportPath)
    $parent = [IO.Path]::GetDirectoryName($target)
    if (-not [IO.Directory]::Exists($parent)) {
        throw 'REPORT_PARENT_DIRECTORY_MISSING'
    }
    [IO.File]::WriteAllText($target, $json + [Environment]::NewLine, [Text.Encoding]::UTF8)
}
Write-Output $json
