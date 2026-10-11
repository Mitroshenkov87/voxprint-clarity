# One script for a fresh rented RTX 4090 (Windows Server 2025, NVIDIA driver 600+).
# Installs uv, CPython 3.14, the GPU stack, an LGPL ffmpeg, and the public
# engine, then runs the benchmark and one end-to-end restore. See docs/TESTING-GPU.md.
#requires -Version 5.1
param(
    [int]$SampleFrames = 48,
    [string]$ResultsDir = "",
    [switch]$AllowOtherGpu,
    [switch]$AllowOtherOs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$script:TranscriptStarted = $false
. (Join-Path $PSScriptRoot "gpu-common.ps1")

$Root = Split-Path -Parent $PSScriptRoot
if (-not $ResultsDir) { $ResultsDir = Join-Path $Root "gpu-results" }
$Tools = Join-Path $ResultsDir "tools"
$Cache = Join-Path $ResultsDir "cache"
$Stage = Join-Path $ResultsDir "stage"
$BundleName = "clarity-gpu-test"
$BundleDir = Join-Path $Stage $BundleName
$Transcript = Join-Path $BundleDir "transcript.log"
$Archive = Join-Path $ResultsDir "$BundleName.tar.gz"

if ($SampleFrames -lt 0) {
    Fail "-SampleFrames must be 0 (the whole clip) or a positive frame count."
}
if (-not (Test-Path -LiteralPath (Join-Path $Root "clarity\bench.py"))) {
    Fail "scripts/test-gpu.ps1 must stay inside the Clarity checkout. clarity\bench.py was not found."
}
New-Item -ItemType Directory -Force -Path $BundleDir, $Tools, $Cache | Out-Null
try { Stop-Transcript | Out-Null } catch { }
Start-Transcript -LiteralPath $Transcript -Force | Out-Null
$script:TranscriptStarted = $true
Write-Host "Clarity GPU test"
Write-Host "results: $ResultsDir"

try {
    Assert-ClarityHost -AllowOtherGpu:$AllowOtherGpu -AllowOtherOs:$AllowOtherOs
} catch {
    Fail $_.Exception.Message
}

try {
    $uv = Get-ClarityUv $Tools
    $ffmpeg = Get-ClarityFfmpeg $Tools
} catch {
    Fail $_.Exception.Message
}
$ffprobe = Join-Path (Split-Path -Parent $ffmpeg) "ffprobe.exe"
$env:Path = "$(Split-Path -Parent $ffmpeg);$env:Path"
Write-Host "uv: $uv"
Write-Host "ffmpeg: $ffmpeg"

$env:PYTHONPATH = $Root
$python = Install-ClarityGpuPython -Root $Root -ResultsDir $ResultsDir -Uv $uv

Write-Host "running python -m clarity.bench"
$benchOut = Join-Path $ResultsDir "benchmark"
Invoke-Native { & $python -m clarity.bench --out $benchOut --clip-dir $Cache } "python -m clarity.bench failed"
foreach ($name in @("benchmark.md", "benchmark.json")) {
    $from = Join-Path $benchOut $name
    if (-not (Test-Path -LiteralPath $from)) { Fail "The benchmark did not write $name." }
    Copy-Item -LiteralPath $from -Destination (Join-Path $BundleDir $name) -Force
}
Get-ChildItem -Path $BundleDir -Filter "*-before.png" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path $BundleDir -Filter "*-after.png" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path $benchOut -Filter "*-before.png" | Copy-Item -Destination $BundleDir -Force
Get-ChildItem -Path $benchOut -Filter "*-after.png" | Copy-Item -Destination $BundleDir -Force
if (-not (Get-ChildItem -Path $BundleDir -Filter "*-before.png")) {
    Fail "The benchmark wrote no before/after crops."
}

$clip = Join-Path $Cache "elephants-dream-ed_1024_512kb.mp4"
if (-not (Test-Path -LiteralPath $clip)) { Fail "The benchmark cache is missing the Elephants Dream clip." }
$sampleIn = Join-Path $BundleDir "sample-source.mp4"
$sampleOut = Join-Path $BundleDir "sample.mkv"
$work = Join-Path $ResultsDir "restore-work"
if (Test-Path -LiteralPath $work) { Remove-Item -LiteralPath $work -Recurse -Force }
if (Test-Path -LiteralPath $sampleOut) { Remove-Item -LiteralPath $sampleOut -Force }
if (Test-Path -LiteralPath $sampleIn) { Remove-Item -LiteralPath $sampleIn -Force }
if ($SampleFrames -eq 0) {
    $restoreSource = $clip
    Write-Host "restoring the whole test clip"
} else {
    Write-Host "cutting the first $SampleFrames frames"
    Invoke-Native { & $ffmpeg -y -i $clip -frames:v $SampleFrames -c copy $sampleIn } "ffmpeg could not cut the sample"
    $restoreSource = $sampleIn
    Write-Host "restoring the sample end to end"
}
$env:CLARITY_RESTORE_SOURCE = $restoreSource
$env:CLARITY_RESTORE_OUTPUT = $sampleOut
$env:CLARITY_RESTORE_WORK = $work
$env:CLARITY_FFMPEG = $ffmpeg
$env:CLARITY_FFPROBE = $ffprobe
$restore = @'
import os
import sys
from pathlib import Path

from clarity.pipeline.job import Settings
from clarity.pipeline.runner import JobSpec, run_job

source = Path(os.environ["CLARITY_RESTORE_SOURCE"])
destination = Path(os.environ["CLARITY_RESTORE_OUTPUT"])
work = Path(os.environ["CLARITY_RESTORE_WORK"])

def progress(fraction: float, text: str) -> None:
    print(f"{fraction:.3f} {text}", flush=True)

try:
    run_job(
        JobSpec(
            source=source,
            destination=destination,
            settings=Settings(mode="clean", upscale=False, copy_audio=True),
            ffmpeg=os.environ["CLARITY_FFMPEG"],
            ffprobe=os.environ["CLARITY_FFPROBE"],
            work=work,
        ),
        progress=progress,
    )
except Exception as exc:
    print(f"restore failed: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc
if not destination.is_file() or destination.stat().st_size <= 0:
    print("restore failed: the output sample was not written", file=sys.stderr)
    raise SystemExit(1)
print(destination)
'@
$restorePath = Join-Path $Stage "restore_sample.py"
Set-Content -LiteralPath $restorePath -Value $restore -Encoding utf8
Invoke-Native { & $python $restorePath } "the end-to-end restore failed"
if (-not (Test-Path -LiteralPath $sampleOut)) { Fail "The restore did not write sample.mkv." }

Stop-Transcript | Out-Null
$script:TranscriptStarted = $false
if (-not (Get-Command tar.exe -ErrorAction SilentlyContinue)) {
    Fail "tar.exe was not found. Windows Server 2025 includes it; this image does not."
}
if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
Push-Location $Stage
try {
    & tar.exe -czf $Archive $BundleName
    if ($LASTEXITCODE -ne 0) { Fail "tar could not write $Archive." }
} finally {
    Pop-Location
}
Write-Host ""
Write-Host "OK: $Archive"
exit 0
