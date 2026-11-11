# P85MR-F3: independent, first-run owner-scoped source collector launcher.
# F1 and F2 attempts remain SPENT. This script has a distinct F3 workflow
# identity, immutable single-use owner run and evidence directory.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$ExpectedSha
)
$ErrorActionPreference='Stop'
$project=[IO.Path]::GetFullPath($ProjectRoot)
$evidence=Join-Path $project 'evidence\p85mr_fitcal_f3\ONCE'
$localRuns=Join-Path $project 'local_runs'
New-Item -ItemType Directory -Path $localRuns -Force | Out-Null
$receipt=Join-Path $localRuns 'p85mr_f3_owner_result.json'
$report=[ordered]@{
    schema='ttt-hybrid.p85mr-f3-external-fitcal-owner-route.v1'
    project_sha=$ExpectedSha
    owner_run_id=$env:GITHUB_RUN_ID
    runner=$env:RUNNER_NAME
    status='PRE_EFFECT_BLOCKED'
    source_first_use=$false
    source_cohorts='FIT128_CAL64'
    ood_opened=$false
    science_pass=$false
}
try {
    $ref='refs/heads/research/p85mr-f3-repaired-fitcal-once-20261111'
    $workflow='spoonman136668-ai/TTT-Hybrid-Lab/.github/workflows/p85mr-f3-repaired-fitcal-once.yml@'+$ref
    if($env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/TTT-Hybrid-Lab' -or
       $env:GITHUB_REF -cne $ref -or $env:GITHUB_WORKFLOW_REF -cne $workflow -or
       $env:GITHUB_EVENT_NAME -cne 'push' -or
       $env:GITHUB_ACTOR -cne 'spoonman136668-ai' -or
       $env:GITHUB_RUN_NUMBER -cne '1' -or
       $env:GITHUB_RUN_ATTEMPT -cne '1' -or
       $env:RUNNER_NAME -cne 'TTT-HYBRID-GPU-WIN' -or
       $env:RUNNER_OS -cne 'Windows' -or
       $env:RUNNER_ARCH -cne 'X64' -or
       $env:GITHUB_SHA -cne $ExpectedSha -or
       $ExpectedSha -cnotmatch '^[0-9a-f]{40}$'){
        throw 'F3_OWNER_EXACT_ONCE_ROUTE_REFUSED'
    }
    $workspace=[IO.Path]::GetFullPath($env:GITHUB_WORKSPACE).TrimEnd('\')
    if(-not $project.StartsWith($workspace+'\', [StringComparison]::OrdinalIgnoreCase)){
        throw 'F3_PROJECT_OUTSIDE_WORKSPACE'
    }
    $head=(& git -C $project rev-parse HEAD | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $head -cne $ExpectedSha){throw 'F3_PROJECT_SHA_DRIFT'}
    $dirty=(& git -C $project status --porcelain --untracked-files=all | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $dirty){throw 'F3_PROJECT_DIRTY'}
    $linuxRoot=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/wslpath -u $project | Out-String).Trim()
    $who=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/id -un | Out-String).Trim()
    $uid=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/id -u | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $uid -cne '1000' -or
       $who -cnotmatch '^[A-Za-z_][A-Za-z0-9_-]{0,32}$' -or
       $linuxRoot -cnotmatch '^/mnt/[a-z]/[a-zA-Z0-9._/-]{1,260}$' -or
       -not $linuxRoot.EndsWith('/p85mr-f3',[StringComparison]::Ordinal)){
        throw 'F3_APPROVED_WSL_OWNER_OR_CHECKOUT_MISSING'
    }
    $jobRoot=$linuxRoot.Substring(0,$linuxRoot.Length-'/p85mr-f3'.Length)
    $source="$jobRoot/p85mr-upstream-f3"
    $upsha=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/git -C $source rev-parse HEAD | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $upsha -cne 'c94e9a4cd3641588b6b8507e71ccee5c27032bdf'){
        throw 'F3_PINNED_SOURCE_CHECKOUT_UNAVAILABLE'
    }
    $uporigin=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/git -C $source remote get-url origin | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $uporigin -cnotin @(
       'https://github.com/CausalAILab/Causal-Gymnasium',
       'https://github.com/CausalAILab/Causal-Gymnasium.git')){
        throw 'F3_UPSTREAM_REMOTE_DRIFT'
    }
    $sourceDirty=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/git -C $source status --porcelain --untracked-files=all | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $sourceDirty){throw 'F3_UPSTREAM_DIRTY'}
    $expectedBlobs=@{
        'causal_gym/envs/cartpole_wind.py'='d65a6a31941709dc9201259d80dd4ebbcf48b4fa'
        'causal_gym/core/pch.py'='ff37780c6b791877c70583b8d5f699be7fcd8817'
        'causal_gym/core/task.py'='668262f3f5c88fa2865b08ef10bbf96b09a3b73e'
        'causal_gym/core/scm.py'='f66f2638c797cfeaf0e96d4e13c06d4fabc2c8d5'
    }
    foreach($rel in $expectedBlobs.Keys){
        $actual=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/git -C $source hash-object -- $rel | Out-String).Trim()
        if($LASTEXITCODE -ne 0 -or $actual -cne $expectedBlobs[$rel]){
            throw 'F3_UPSTREAM_FROZEN_BLOB_MISMATCH'
        }
    }
    $py="/home/$who/.local/share/ttt-hybrid/p85mr-causalgym-cpu311-r3/venv/bin/python"
    & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/test -x $py
    if($LASTEXITCODE -ne 0){throw 'F3_R3_QUALIFIED_PYTHON_NOT_FOUND'}
    # All eight frozen package versions and basic imports, no simulator import.
    $probe="import importlib,importlib.metadata as m,sys; p={'numpy':'1.26.4','gymnasium':'0.29.1','networkx':'3.4.2','pydash':'8.0.5','toposort':'1.10','cloudpickle':'3.0.0','farama-notifications':'0.0.4','typing-extensions':'4.12.2'}; assert sys.version_info[:2]==(3,11); assert all(m.version(k)==v for k,v in p.items()); [importlib.import_module(k) for k in ('numpy','gymnasium','networkx','pydash','toposort')]; print('F3_RUNTIME_IMPORT_PASS')"
    $probeOut=(& wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/timeout 30s $py -I -B -c $probe | Out-String).Trim()
    if($LASTEXITCODE -ne 0 -or $probeOut -cne 'F3_RUNTIME_IMPORT_PASS'){
        throw 'F3_PINNED_R3_RUNTIME_VERSION_OR_IMPORT_FAILURE'
    }
    Write-Output 'P85MR_F3_PRE_EFFECT_RUNTIME_SOURCE_CUSTODY=PASS'
    $vars=@(
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
        'P85MR_F3_OWNER_SCOPE=FIT_CAL_ONLY',
        'PYTHONDONTWRITEBYTECODE=1',
        'PYTHONNOUSERSITE=1',
        'PYTHONHASHSEED=0',
        'CUDA_VISIBLE_DEVICES=-1'
    )
    $report.status='COLLECTOR_INVOKED'
    & wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/timeout --signal=TERM --kill-after=15s 2550s /usr/bin/env @vars $py -s -B "$linuxRoot/experiments/p85mr_f3_fitcal_once.py" --source-checkout $source --project-checkout $linuxRoot --approved-once
    if($LASTEXITCODE -ne 0){throw 'F3_EFFECT_ATTEMPT_ABORTED_NO_REPLAY'}
    $resultPath=Join-Path $evidence 'RESULT.json'
    if(-not(Test-Path -LiteralPath $resultPath -PathType Leaf)){
        throw 'F3_OWNER_COLLECTOR_RESULT_NOT_FOUND'
    }
    $result=Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if($result.status -cne 'SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING' -or
       $result.owner.project_sha -cne $ExpectedSha -or
       $result.owner.run_id -cne $env:GITHUB_RUN_ID -or
       $result.owner.source_route -cne 'P85MR-F3-INDEPENDENT-ONCE' -or
       $result.cohorts.fit.scheduled -ne 128 -or
       $result.cohorts.cal.scheduled -ne 64 -or
       $result.ood_created -ne $false -or
       $result.original_f1_f2_runs_replayed -ne $false -or
       $result.science_pass -ne $false){
        throw 'F3_WITNESS_AND_SOURCE_CUSTODY_NOT_ACCEPTED'
    }
    $report.status='SOURCE_COLLECTED_WITNESS_PENDING'
    $report.source_first_use=$true
    $report.actions=$result.executed_actions
    Write-Output 'P85MR_F3_SOURCE_FITCAL_COLLECTED_WITNESS_PENDING'
} catch {
    $report.failure_code=[string]$_.Exception.Message
    $report.status=if(Test-Path -LiteralPath (Join-Path $evidence 'ATTEMPT.json')){
        'ATTEMPT_CONSUMED_NO_REPLAY'
    }else{
        'PRE_EFFECT_BLOCKED'
    }
    Write-Output ('P85MR_F3_RESULT='+$report.status)
    Write-Output ('P85MR_F3_FAILURE='+$report.failure_code)
    throw
} finally {
    $report.utc=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($receipt,($report|ConvertTo-Json -Depth 8)+[Environment]::NewLine,
        [Text.UTF8Encoding]::new($false))
}
