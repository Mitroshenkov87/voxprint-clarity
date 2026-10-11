# Windows entry for the rented GPU VM. Same flags, JSON, and exit codes as
# scripts/vm_test.sh. Exit 0 pass, 1 test failure, 2 setup, 3 no GPU.
# See docs/TESTING-GPU.md.
#requires -Version 5.1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$script:TranscriptStarted = $false
. (Join-Path $PSScriptRoot "gpu-common.ps1")

$Smoke = $false
$Full = $false
$Json = $args -contains "--json"
$AllowOtherGpu = $false
$AllowOtherOs = $false
$Forward = New-Object System.Collections.Generic.List[string]

function Write-VmFailure {
    param([int]$Code, [string]$Step, [string]$ErrorText)
    Write-Host ""
    Write-Host "FAILED: step ${Step}: $ErrorText"
    if ($Json) {
        $payload = @{
            type = "summary"
            app = "clarity"
            ok = $false
            exit = $Code
            step = $Step
            error = $ErrorText
        }
        Write-Output ($payload | ConvertTo-Json -Compress)
    }
    exit $Code
}

$index = 0
while ($index -lt $args.Count) {
    $item = [string]$args[$index]
    switch ($item) {
        "--smoke" { $Smoke = $true; $Forward.Add($item) }
        "--full" { $Full = $true; $Forward.Add($item) }
        "--json" { $Json = $true; $Forward.Add($item) }
        "--allow-other-gpu" { $AllowOtherGpu = $true; $Forward.Add($item) }
        "--allow-other-os" { $AllowOtherOs = $true; $Forward.Add($item) }
        "--results" {
            $index += 1
            if ($index -ge $args.Count) { Write-VmFailure 2 "args" "--results needs a directory" }
            $Forward.Add("--results")
            $Forward.Add([string]$args[$index])
        }
        default {
            Write-VmFailure 2 "args" "unknown argument $item"
        }
    }
    $index += 1
}

if (-not $Smoke -and -not $Full) {
    Write-VmFailure 2 "args" "pass --smoke or --full"
}

$Root = Split-Path -Parent $PSScriptRoot
try {
    Assert-ClarityHost -AllowOtherGpu:$AllowOtherGpu -AllowOtherOs:$AllowOtherOs
} catch {
    Write-VmFailure 3 "gpu" $_.Exception.Message
}

$Bootstrap = Join-Path $Root "vm-results\bootstrap"
New-Item -ItemType Directory -Force -Path $Bootstrap | Out-Null
try {
    $uv = Get-ClarityUv $Bootstrap
} catch {
    Write-VmFailure 2 "bootstrap" $_.Exception.Message
}
try {
    & $uv python install 3.14
    if ($LASTEXITCODE -ne 0) { Write-VmFailure 2 "bootstrap" "uv could not install CPython 3.14 (exit $LASTEXITCODE)" }
    $python = & $uv python find 3.14
    if ($LASTEXITCODE -ne 0 -or -not $python) { Write-VmFailure 2 "bootstrap" "Python 3.14 was not found after uv python install." }
} catch {
    Write-VmFailure 2 "bootstrap" $_.Exception.Message
}

$driver = Join-Path $PSScriptRoot "vm_test.py"
& $python $driver @($Forward.ToArray())
exit $LASTEXITCODE
