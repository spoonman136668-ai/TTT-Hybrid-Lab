[CmdletBinding()]
param([ValidateSet('preflight','install')][string]$Mode,[string]$SourceRoot,[string]$ReceiptPath)
$ErrorActionPreference='Stop'
$sha='108808f7b685107a76fae1ce5605b7ff9e2bac27'
$scope=Join-Path $env:RUNNER_WORKSPACE '_g7_s25_cpu311_r1'
$state='BLOCKED_NOT_STARTED';$code=2
$receipt=[ordered]@{schema='ttt-hybrid.cpu311.scoped-windows.v1';source_sha=$sha;status=$state;full_regression='NOT_RUN';r3='UNCHANGED';cuda='NOT_USED';scientific_first_use='NOT_RUN'}
function AssertExit([string]$stage){if($LASTEXITCODE -ne 0){throw "BLOCKED_$stage"}}
try{
 if($env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/TTT-Hybrid-Lab' -or $env:GITHUB_ACTOR -cne 'spoonman136668-ai' -or
 $env:GITHUB_EVENT_NAME -cne 'push' -or $env:GITHUB_REF -cne 'refs/heads/engineering/g7-s25-scoped-cpu311-recovery-r3-20261107' -or
 $env:GITHUB_RUN_NUMBER -cne '1' -or $env:GITHUB_RUN_ATTEMPT -cne '1' -or
 $env:RUNNER_NAME -cne 'TTT-HYBRID-GPU-WIN' -or $env:RUNNER_OS -cne 'Windows'){throw 'BLOCKED_AUTHORITY'}
 if(-not $env:RUNNER_WORKSPACE -or -not(Test-Path -LiteralPath $SourceRoot -PathType Container)){throw 'BLOCKED_WORKSPACE'}
 $head=(& git -C $SourceRoot rev-parse HEAD | Out-String).Trim();AssertExit 'SOURCE'
 if($head -cne $sha){throw 'BLOCKED_SOURCE_SHA'}
 $protocol=(& git -C $SourceRoot hash-object -- experiments/g7_s25_frozen_contradiction_ttt.json | Out-String).Trim();AssertExit 'PROTOCOL'
 if($protocol -cne '31633b341403729b7ffbb7d73f3a04098ad44380'){throw 'BLOCKED_PROTOCOL_DRIFT'}
 $pins=(& git -C $SourceRoot hash-object -- source_pins.json | Out-String).Trim();AssertExit 'PINS'
 if($pins -cne '1438b51f8548db6bfe7e04805ba8c156e4b5b630'){throw 'BLOCKED_PINS_DRIFT'}
 if((git -C $SourceRoot status --porcelain --untracked-files=no | Out-String).Trim().Length -gt 0){throw 'BLOCKED_DIRTY_TRACKED_SOURCE'}
 if(-not(Test-Path -LiteralPath $scope -PathType Container)){throw 'BLOCKED_VERIFIED_RECOVERY_SCOPE_ABSENT'}
 # Recovery from exact R2 "No solution found" resolution failure only:
 # it never installed a wheel; the prepared interpreter must be empty.
 $python=Join-Path $scope 'venv\Scripts\python.exe'
 if(-not(Test-Path -LiteralPath (Join-Path $scope 'pythons') -PathType Container) -or
    -not(Test-Path -LiteralPath $python -PathType Leaf)){throw 'BLOCKED_MISSING_PINNED_INTERPRETER'}
 $pyVersion=(& $python -I -B -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' | Out-String).Trim()
 AssertExit 'PYTHON_RECOVERY_PROBE'
 if($pyVersion -cne '3.11.14'){throw 'BLOCKED_INTERPRETER_DRIFT'}
 $installed=(& $python -I -B -c 'import json,importlib.metadata as m;print(json.dumps(sorted((d.metadata.get("Name") or "").lower() for d in m.distributions())))' | Out-String).Trim()
 AssertExit 'PREEXISTING_WHEELS_PROBE'
 if($installed -cne '[]'){throw 'BLOCKED_UNKNOWN_PARTIAL_PACKAGE_INSTALL'}
 $zip=Join-Path $scope 'uv.zip'
 if(-not(Test-Path -LiteralPath $zip -PathType Leaf)){throw 'BLOCKED_ARCHIVE_ABSENT'}
 if((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -cne
    'cdfb04512c9184939f16eea02644a7133b4e41233ead69122d49847d29a45538'){
  throw 'BLOCKED_ARCHIVE_HASH_DRIFT'
 }
 if([IO.DriveInfo]::new([IO.Path]::GetPathRoot($scope)).AvailableFreeSpace -lt 6GB){throw 'BLOCKED_DISK_RESERVE'}
 if($Mode -eq 'preflight'){$state='PASS_WINDOWS_READONLY_PREFLIGHT';$code=0}
 else{
  # Original exact-hash release was already downloaded and read-only verified
  # in run 42430469728. No second download, extraction or uncertain replay.
  $out=Join-Path $scope 'uv_release'
  $found=@(Get-ChildItem -LiteralPath $out -Filter uv.exe -File -Recurse)
  if($found.Count -ne 1){throw 'BLOCKED_BINARY_CARDINALITY'}
  $uv=$found[0].FullName
  $actualVersion=(& $uv --version|Out-String).Trim()
  AssertExit 'UV_VERSION'
  if($actualVersion -cnotmatch '^uv 0\.9\.13(?:\s|$)'){throw 'BLOCKED_UV_IDENTITY'}
  $env:UV_PYTHON_INSTALL_DIR=Join-Path $scope 'pythons';$env:UV_CACHE_DIR=Join-Path $scope 'cache'
  $env:UV_NO_CONFIG='1';$env:UV_PYTHON_PREFERENCE='only-managed';$env:UV_PYTHON_INSTALL_BIN='false'
  $env:UV_CONCURRENT_DOWNLOADS='2';$env:UV_CONCURRENT_INSTALLS='1';$env:UV_HTTP_RETRIES='1'
  $env:CUDA_VISIBLE_DEVICES='-1';$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONHASHSEED='0'
  # No Python/venv recreation and no ambiguous cross-index resolution.
  # PyPI supplies ordinary packages; PyTorch's official CPU-only index
  # supplies torch exclusively. Require all dependency metadata to match.
  & $uv pip install --python $python --only-binary ':all:' --index-url 'https://pypi.org/simple' 'pytest==8.3.4' 'filelock>=3.13.1' 'typing-extensions>=4.8.0' 'networkx' 'jinja2' 'fsspec' 'sympy==1.13.1'
  AssertExit 'PYPI_RUNTIME_DEPENDENCIES'
  & $uv pip install --python $python --only-binary ':all:' --no-deps --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.5.1+cpu'
  AssertExit 'TORCH_CPU_INDEX_ONLY'
  & $uv pip check --python $python
  AssertExit 'DEPENDENCY_CONSISTENCY'
  $probe='import json,sys,torch,pytest;print(json.dumps({"py":list(sys.version_info[:3]),"torch":str(torch.__version__),"pytest":str(pytest.__version__),"cuda":torch.version.cuda}))'
  $v=ConvertFrom-Json ((& $python -I -B -c $probe|Out-String).Trim());AssertExit 'RUNTIME_PROBE'
  if($v.py[0] -ne 3 -or $v.py[1] -ne 11 -or $v.py[2] -ne 14 -or $v.torch -cne '2.5.1+cpu' -or $v.pytest -cne '8.3.4' -or $null -ne $v.cuda){throw 'BLOCKED_RUNTIME_VERSIONS'}
  $env:EXPECTED_CANDIDATE_SHA=$sha
  $native=Join-Path $SourceRoot 'local_runs\s25_full_regression_windows.json'
  & $python -I -B (Join-Path $SourceRoot 'scripts\qualify_cpu.py') --receipt $native;AssertExit 'WINDOWS_REGRESSION'
  $q=Get-Content -LiteralPath $native -Raw|ConvertFrom-Json
  if($q.status -cne 'PASS_CPU_ENGINEERING_ONLY' -or $q.revision -cne $sha -or $q.negative_control.pass -ne $true){throw 'BLOCKED_WINDOWS_RECEIPT'}
  $receipt.full_regression='PASS_EXACT_S25_FULL_SUITE_AND_NEGATIVE_CONTROL';$state='PASS_WINDOWS_CPU311';$code=0
 }
}catch{
 $state='BLOCKED_WINDOWS_INSTALL_OR_QUALIFICATION'
 $msg=[string]$_.Exception.Message
 if($msg -cmatch '^BLOCKED_[A-Z0-9_]+$'){$receipt.reason=$msg}
}finally{
 $receipt.status=$state;Write-Output ('WINDOWS_CPU311='+$state)
 if($Mode -eq 'install'){
  $dir=Split-Path -Parent $ReceiptPath
  if(-not(Test-Path -LiteralPath $dir)){New-Item -ItemType Directory -Path $dir -Force|Out-Null}
  $receipt|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $ReceiptPath -Encoding UTF8
 }
}
exit $code
