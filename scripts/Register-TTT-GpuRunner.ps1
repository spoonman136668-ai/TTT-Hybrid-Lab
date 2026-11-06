# Repo-only setup: installs ONE additional TTT runner, never a controller.
# Use only after the operator chooses the physical host and accepts its resource budget.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RunnerArchive,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedSha256,
    [string]$RunnerRoot = 'C:\actions-runner-ttt-hybrid'
)
$ErrorActionPreference = 'Stop'
$repo = 'https://github.com/spoonman136668-ai/TTT-Hybrid-Lab'
$name = 'TTT-HYBRID-GPU-WIN'
$label = 'ttt-hybrid-gpu-isolated'

# Check official GitHub-provided archive checksum *before* extraction/execution.
$zip = (Resolve-Path -LiteralPath $RunnerArchive -ErrorAction Stop).Path
$observedHash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
if ($observedHash -cne $ExpectedSha256.ToUpperInvariant()) {
    throw 'TTT_RUNNER_ARCHIVE_SHA256_MISMATCH'
}
$root = [IO.Path]::GetFullPath($RunnerRoot).TrimEnd('\')
if (-not [IO.Path]::IsPathRooted($root)) { throw 'TTT_RUNNER_ROOT_NOT_ABSOLUTE' }
if ($root -match '^[A-Za-z]:\\?$') { throw 'TTT_RUNNER_ROOT_IS_DRIVE_ROOT' }
if (Test-Path -LiteralPath (Join-Path $root '.runner')) {
    throw 'TTT_RUNNER_ALREADY_REGISTERED_NO_REPLACEMENT'
}
if (Test-Path -LiteralPath $root) {
    if (@(Get-ChildItem -LiteralPath $root -Force -ErrorAction Stop).Count -gt 0) {
        throw 'TTT_RUNNER_DIR_NOT_EMPTY_NO_REPLACEMENT'
    }
} else {
    New-Item -ItemType Directory -Path $root -Force | Out-Null
}
$nv = Get-Command -Name 'nvidia-smi.exe' -CommandType Application -ErrorAction SilentlyContinue
if (-not $nv) { throw 'TTT_WINDOWS_GPU_NOT_VISIBLE_NO_REGISTRATION' }
$inventory = @(& $nv.Source '--query-gpu=name,memory.total,memory.free' '--format=csv,noheader,nounits')
if ($LASTEXITCODE -ne 0 -or $inventory.Count -ne 1) {
    throw 'TTT_GPU_INVENTORY_NOT_SINGLE_DEVICE'
}
$cols = @($inventory[0] -split ',')
if ($cols.Count -ne 3) { throw 'TTT_GPU_INVENTORY_INVALID' }
[int]$total = 0
[int]$free = 0
if (-not [int]::TryParse($cols[1].Trim(), [ref]$total) -or
    -not [int]::TryParse($cols[2].Trim(), [ref]$free) -or
    $total -lt 4096 -or $free -lt 4096 -or $free -gt $total) {
    throw 'TTT_INSUFFICIENT_FREE_GPU_MEMORY'
}

Expand-Archive -LiteralPath $zip -DestinationPath $root -ErrorAction Stop
$config = Join-Path $root 'config.cmd'
$listener = Join-Path $root 'run.cmd'
if (-not (Test-Path -LiteralPath $config -PathType Leaf) -or
    -not (Test-Path -LiteralPath $listener -PathType Leaf)) {
    throw 'TTT_OFFICIAL_RUNNER_PAYLOAD_MISSING'
}
Write-Host "TTT_RUNNER_REPO=$repo"
Write-Host "TTT_RUNNER_NAME=$name"
Write-Host "TTT_RUNNER_LABEL=$label"
Write-Host "TTT_GPU_INVENTORY=$($cols[0].Trim()) total_mib=$total free_mib=$free"
Write-Host 'Registration is repository-scoped; the other project runners are unchanged.'
Write-Host 'Paste the short-lived registration token from the GitHub NEW RUNNER page.'
$secure = Read-Host -Prompt 'TTT repository registration token (not your PAT)' -AsSecureString
if ($secure.Length -eq 0) { throw 'TTT_REGISTRATION_TOKEN_MISSING' }
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = $null
try {
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    Push-Location -LiteralPath $root
    try {
        & $config --url $repo --token $plain --name $name --labels $label --work '_work' --unattended
        if ($LASTEXITCODE -ne 0) { throw 'TTT_RUNNER_REGISTRATION_FAILED' }
    } finally {
        Pop-Location
    }
} finally {
    $plain = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secure.Dispose()
}
Write-Host 'TTT_RUNNER_REGISTERED=CHECK_GITHUB_UI'
Write-Host "For a manually supervised session under this same Windows user, run: $listener"
Write-Host 'Not started. No Windows service, scheduler, WSL launch, CUDA kernel, model, or controller installed.'
