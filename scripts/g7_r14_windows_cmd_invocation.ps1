# G7-R14: synthetic trusted Windows cmd.exe only. No Linux/CUDA/model activity.
$ErrorActionPreference = 'Stop'
$r = [ordered]@{
  schema='ttt-hybrid.g7-r14.windows-native-launch.v1'
  scope='HOSTED_WINDOWS_CMD_ONLY'
  source_info='UNVERIFIED'; definition_info='UNVERIFIED'
  legacy_call='NOT_TESTED'; legacy_exception='NONE'
  explicit_path_call='NOT_TESTED'; explicit_exception='NONE'
  negative_missing_command='NOT_TESTED'
  original_runner='NOT_TOUCHED'; linux='NOT_RUN'
  cuda='NOT_RUN'; model='NOT_RUN'; installation='NOT_RUN'
  status='FAIL_NOT_QUALIFIED'
}
try {
  $commands = @(Get-Command -Name 'cmd.exe' -CommandType Application -ErrorAction Stop)
  if ($commands.Count -ne 1 -or $commands[0].GetType().Name -cne 'ApplicationInfo') {
    throw 'CMD_APPLICATION_RESOLUTION_FAILED'
  }
  $app = $commands[0]
  $r.source_info = if ([string]::IsNullOrWhiteSpace([string]$app.Source)) { 'EMPTY' } else { 'NONEMPTY' }
  $exe = [string]$app.Definition
  $r.definition_info = if ([string]::IsNullOrWhiteSpace($exe)) { 'EMPTY' } else { 'NONEMPTY' }
  if (-not [IO.Path]::IsPathRooted($exe) -or
      -not [IO.File]::Exists($exe) -or
      -not ([IO.Path]::GetFileName($exe) -ieq 'cmd.exe')) {
    throw 'CMD_DEFINITION_PATH_INVALID'
  }
  # Same member-access invocation syntax used by the consumed diagnostic.
  try {
    $o = @(& $app.Source /d /c 'echo G7_R14_LEGACY_OK' 2>$null)
    $r.legacy_call = if ($LASTEXITCODE -eq 0 -and $o.Count -eq 1 -and
        ([string]$o[0]).Trim() -ceq 'G7_R14_LEGACY_OK') { 'PASS' } else { 'FAIL_OUTPUT' }
  } catch {
    $r.legacy_call = 'FAILED_EXCEPTION'
    $r.legacy_exception = [string]$_.Exception.GetType().Name
  }
  # Explicit validated executable path; each argument remains a separate token.
  try {
    $o = @(& $exe /d /c 'echo G7_R14_EXPLICIT_OK' 2>$null)
    $r.explicit_path_call = if ($LASTEXITCODE -eq 0 -and $o.Count -eq 1 -and
        ([string]$o[0]).Trim() -ceq 'G7_R14_EXPLICIT_OK') { 'PASS' } else { 'FAIL_OUTPUT' }
  } catch {
    $r.explicit_path_call = 'FAILED_EXCEPTION'
    $r.explicit_exception = [string]$_.Exception.GetType().Name
  }
  $missing = Join-Path $env:TEMP 'g7_r14_absent_cmd_2ef3da1f.exe'
  if (Test-Path -LiteralPath $missing) { throw 'NEGATIVE_PATH_EXISTS' }
  try {
    $unused = @(& $missing /d /c 'echo NEVER_RUN' 2>$null)
    $r.negative_missing_command = 'FAIL_NOT_BLOCKED'
  } catch [System.Management.Automation.CommandNotFoundException] {
    $r.negative_missing_command = 'PASS_COMMAND_NOT_FOUND'
  } catch {
    $r.negative_missing_command = 'FAIL_OTHER_EXCEPTION'
  }
  if ($r.source_info -eq 'NONEMPTY' -and
      $r.definition_info -eq 'NONEMPTY' -and
      $r.explicit_path_call -eq 'PASS' -and
      $r.negative_missing_command -eq 'PASS_COMMAND_NOT_FOUND') {
    $r.status='PASS_HOSTED_WINDOWS_SYNTAX_ONLY'
  }
} catch {
  $r.status='FAIL_HOSTED_WINDOWS_SYNTAX'
}
# Emit only allowlisted categories, never executable paths/error text/env data.
$r | ConvertTo-Json -Depth 4 -Compress
if ($r.status -ne 'PASS_HOSTED_WINDOWS_SYNTAX_ONLY') { exit 2 }
