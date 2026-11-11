# P85-MR F1: invoke the already-authorized, ONE-SHOT FIT/CAL collection on
# the EXISTING TTT-Hybrid Windows + Ubuntu WSL project runner.
# Called ONLY by the exact branch-push first-use workflow after full
# Windows + WSL source regression. Never installs, clones, schedules, or retries.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$ExpectedSha
)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath($ProjectRoot)
$localRuns = Join-Path $project 'local_runs'
New-Item -ItemType Directory -Path $localRuns -Force | Out-Null
$receipt = Join-Path $localRuns 'p85mr_fitcal_scoped_runner.json'
$evidence = Join-Path $project 'evidence\p85mr_fitcal_firstuse\ONCE'
$report = [ordered]@{
    schema = 'ttt-hybrid.p85mr-f1-owner-route-execution.v1'
    candidate_sha = $ExpectedSha
    github_run_id = $env:GITHUB_RUN_ID
    owner_runner = $env:RUNNER_NAME
    expected_episodes = 192
    ood_opened = $false
    source_effects = 'UNVERIFIED'
    status = 'PRE_EFFECT_BLOCKED'
}
try {
    $expectedRef = 'refs/heads/research/p85mr-authorized-fitcal-firstuse-20261111'
    $expectedWorkflowRef = 'spoonman136668-ai/TTT-Hybrid-Lab/.github/workflows/p85mr-fitcal-firstuse-once.yml@' + $expectedRef
    if ($env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/TTT-Hybrid-Lab' -or
        $env:GITHUB_REF -cne $expectedRef -or
        $env:GITHUB_WORKFLOW_REF -cne $expectedWorkflowRef -or
        $env:GITHUB_ACTOR -cne 'spoonman136668-ai' -or
        $env:GITHUB_EVENT_NAME -cne 'push' -or
        $env:GITHUB_RUN_NUMBER -cne '1' -or
        $env:GITHUB_RUN_ATTEMPT -cne '1' -or
        $env:RUNNER_NAME -cne 'TTT-HYBRID-GPU-WIN' -or
        $env:RUNNER_OS -cne 'Windows' -or
        $env:RUNNER_ARCH -cne 'X64' -or
        $env:GITHUB_SHA -cne $ExpectedSha -or
        $ExpectedSha -cnotmatch '^[0-9a-f]{40}$') {
        throw 'OWNER_ROUTE_SCOPE_DENIED'
    }
    $fullWorkspace = [IO.Path]::GetFullPath($env:GITHUB_WORKSPACE).TrimEnd('\')
    if (-not $project.StartsWith($fullWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'OWNER_CHECKOUT_OUTSIDE_JOB_WORKSPACE'
    }
    $head = (& git -C $project rev-parse HEAD | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $head -cne $ExpectedSha) {
        throw 'EXACT_SOURCE_SHA_DRIFT'
    }
    $dirty = (& git -C $project status --porcelain --untracked-files=all | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $dirty) {
        throw 'DIRTY_SOURCE_REFUSED'
    }

    # The only approved CPU substrate is the existing unprivileged WSL user.
    $uid = (& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/id -u | Out-String).Trim()
    $who = (& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/id -un | Out-String).Trim()
    $linuxRoot = (& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/wslpath -u $project | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $uid -cne '1000' -or
        $who -cnotmatch '^[A-Za-z_][A-Za-z0-9_-]{0,32}$' -or
        $linuxRoot -cnotmatch '^/mnt/[a-z]/[a-zA-Z0-9._/-]{1,260}$' -or
        $linuxRoot.Contains('/../') -or $linuxRoot.Contains('//')) {
        throw 'UNQUALIFIED_WSL_OWNER_PATH'
    }
    # Never install dependencies or download sources. Probe only known,
    # narrowly scoped local checkouts, then verify exact git/blob custody
    # inside the Python collector BEFORE any real environment reset/do.
    $sourceCandidates = @(
        "/home/$who/.local/share/ttt-hybrid/Causal-Gymnasium",
        "/home/$who/.local/share/ttt-hybrid/causal-gymnasium",
        "/home/$who/Research/Causal-Gymnasium",
        "/home/$who/Causal-Gymnasium",
        '/mnt/c/Research/Causal-Gymnasium',
        '/mnt/c/Research/TTT-Hybrid/Causal-Gymnasium',
        '/mnt/d/Research/Causal-Gymnasium'
    )
    $source = $null
    foreach ($candidate in $sourceCandidates) {
        & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/test -f "$candidate/causal_gym/envs/cartpole_wind.py"
        if ($LASTEXITCODE -eq 0) {
            $observed = (& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/git -C $candidate rev-parse HEAD | Out-String).Trim()
            if ($LASTEXITCODE -eq 0 -and
                $observed -ceq 'c94e9a4cd3641588b6b8507e71ccee5c27032bdf') {
                $source = $candidate
                break
            }
        }
    }
    if (-not $source) {
        throw 'NO_EXISTING_EXACT_PINNED_CAUSALGYM_SOURCE_CHECKOUT'
    }
    $report.source_checkout = 'PREEXISTING_EXACT_SHA_FOUND'

    # The WSL CPU environment used by the already-proven TTT runner
    # is tried first; next only qualified local owner-managed environments.
    $pythonCandidates = @(
        "/home/$who/.local/share/ttt-hybrid/g7-s25-cpu311-r1/venv/bin/python",
        "/home/$who/.local/share/ttt-hybrid/p85mr-causalgym-cpu311/venv/bin/python",
        "/home/$who/.local/share/ttt-hybrid/causalgym-cpu311/venv/bin/python"
    )
    $py = $null
    $probe = "import sys,importlib.metadata as m; p={'numpy':'1.26.4','gymnasium':'0.29.1','networkx':'3.4.2','pydash':'8.0.5','toposort':'1.10'}; sys.exit(0 if sys.version_info[:2]==(3,11) and all(m.version(k)==v for k,v in p.items()) else 3)"
    foreach ($candidate in $pythonCandidates) {
        & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/test -x $candidate
        if ($LASTEXITCODE -ne 0) { continue }
        & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/timeout 20s $candidate -I -B -c $probe
        if ($LASTEXITCODE -eq 0) {
            $py = $candidate
            break
        }
    }
    if (-not $py) {
        throw 'NO_EXISTING_PINNED_PY311_CAUSALGYM_RUNTIME'
    }
    $report.runtime = 'EXISTING_PINNED_WSL311'
    $pass = @(
        "GITHUB_REPOSITORY=$env:GITHUB_REPOSITORY",
        "GITHUB_ACTOR=$env:GITHUB_ACTOR",
        "GITHUB_EVENT_NAME=$env:GITHUB_EVENT_NAME",
        "GITHUB_REF=$env:GITHUB_REF",
        "GITHUB_WORKFLOW_REF=$env:GITHUB_WORKFLOW_REF",
        "GITHUB_SHA=$env:GITHUB_SHA",
        "GITHUB_RUN_NUMBER=$env:GITHUB_RUN_NUMBER",
        "GITHUB_RUN_ATTEMPT=$env:GITHUB_RUN_ATTEMPT",
        "GITHUB_RUN_ID=$env:GITHUB_RUN_ID",
        "RUNNER_NAME=$env:RUNNER_NAME",
        "RUNNER_OS=$env:RUNNER_OS",
        "RUNNER_ARCH=$env:RUNNER_ARCH",
        'P85MR_OWNER_EXECUTION=FIT_CAL_ONLY',
        'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONHASHSEED=0',
        'CUDA_VISIBLE_DEVICES=-1'
    )
    $report.status = 'COLLECTOR_INVOKED'
    & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/timeout --signal=TERM --kill-after=15s 2550s /usr/bin/env @pass $py -I -B "$linuxRoot/experiments/p85mr_fitcal_owner_firstuse.py" --source-checkout $source --project-checkout $linuxRoot --approved-once
    if ($LASTEXITCODE -ne 0) {
        throw 'OWNER_COLLECTOR_ABORTED_NO_REPLAY'
    }
    $result = Join-Path $evidence 'RESULT.json'
    if (-not (Test-Path -LiteralPath $result -PathType Leaf)) {
        throw 'MISSING_AUTHENTIC_COLLECTOR_RESULT'
    }
    $out = Get-Content -LiteralPath $result -Raw | ConvertFrom-Json
    if ($out.status -cne 'SOURCE_COLLECTED_WITNESS_VERIFICATION_PENDING' -or
        $out.owner.project_sha -cne $ExpectedSha -or
        $out.owner.run_id -cne $env:GITHUB_RUN_ID -or
        $out.cohorts.fit.scheduled -ne 128 -or
        $out.cohorts.cal.scheduled -ne 64 -or
        $out.ood_created -ne $false -or $out.science_pass -ne $false) {
        throw 'SOURCE_RECEIPT_CUSTODY_FAILED'
    }
    $report.status = 'FITCAL_COLLECTED_INDEPENDENT_WITNESS_PENDING'
    $report.source_effects = 'FROZEN_FIT_CAL_ONLY'
    $report.actual_actions = $out.executed_actions
    Write-Output 'P85MR_F1_SOURCE_COLLECTED_FIT_CAL_ONLY_UNQUALIFIED_SCIENCE'
} catch {
    $report.failure_code = [string]$_.Exception.Message
    $report.status = if (Test-Path -LiteralPath (Join-Path $evidence 'ATTEMPT.json')) {
        'SCIENTIFIC_ATTEMPT_ABORTED_NO_REPLAY'
    } else { 'PRE_EFFECT_BLOCKED' }
    Write-Output ('P85MR_F1_STATUS=' + $report.status)
    Write-Output ('P85MR_F1_FAILURE=' + $report.failure_code)
    throw
} finally {
    $report.utc = [DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($receipt, ($report | ConvertTo-Json -Depth 6) + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false))
}
