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

$Root = Split-Path -Parent $PSScriptRoot
if (-not $ResultsDir) { $ResultsDir = Join-Path $Root "gpu-results" }
$Tools = Join-Path $ResultsDir "tools"
$Cache = Join-Path $ResultsDir "cache"
$Stage = Join-Path $ResultsDir "stage"
$BundleName = "clarity-gpu-test"
$BundleDir = Join-Path $Stage $BundleName
$Transcript = Join-Path $BundleDir "transcript.log"
$Archive = Join-Path $ResultsDir "$BundleName.tar.gz"

$FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip"
$UvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$EngineUrl = "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@d11d60921448b697d832b2699e6e3e04c62c6dfd"

function Fail([string]$Reason) {
    Write-Host ""
    Write-Host "FAILED: $Reason"
    Write-Host "The GPU test stopped. Fix that and run scripts\test-gpu.ps1 again."
    if ($script:TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:TranscriptStarted = $false
    }
    exit 1
}

function Invoke-Native([scriptblock]$Command, [string]$Reason) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Fail "$Reason (exit $LASTEXITCODE)"
    }
}

function Expand-Zip([string]$Zip, [string]$Destination) {
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Destination -Force
}

if ($env:OS -ne "Windows_NT") {
    Fail "scripts/test-gpu.ps1 runs on Windows. This process is not Windows."
}
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

$caption = (Get-CimInstance Win32_OperatingSystem).Caption
Write-Host "os: $caption"
if ($caption -notmatch "Windows Server 2025" -and -not $AllowOtherOs) {
    Fail "This script expects Windows Server 2025. Found: $caption. Pass -AllowOtherOs only when the image caption differs."
}

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidia) {
    Fail "nvidia-smi was not found. Install an NVIDIA driver from the 600 series or newer."
}
$gpuLine = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
    Fail "nvidia-smi did not report a GPU. Clarity has no CPU mode."
}
$gpuLine = ($gpuLine | Select-Object -First 1).Trim()
Write-Host "gpu: $gpuLine"
$cells = @($gpuLine.Split(",") | ForEach-Object { $_.Trim() })
if ($cells.Count -lt 3) {
    Fail "nvidia-smi returned '$gpuLine', which is not name, driver, compute capability."
}
$driver = $cells[$cells.Count - 2]
$compute = $cells[$cells.Count - 1]
$gpuName = ($cells[0..($cells.Count - 3)] -join ", ")
$driverMajor = 0
if ($driver -match "^(\d+)") { $driverMajor = [int]$Matches[1] }
if ($driverMajor -lt 600) {
    Fail "NVIDIA driver $driver is older than the 600 series. Install a 600-series driver or newer."
}
$majorCap = 0
$minorCap = 0
if ($compute -match "^(\d+)\.(\d+)") {
    $majorCap = [int]$Matches[1]
    $minorCap = [int]$Matches[2]
}
if ($majorCap -lt 8 -or ($majorCap -eq 8 -and $minorCap -lt 9)) {
    Fail "GPU compute capability $compute is below 8.9. Clarity needs an RTX 40 series GPU or newer."
}
if ($gpuName -notmatch "4090" -and -not $AllowOtherGpu) {
    Fail "Expected an RTX 4090 and found '$gpuName'. Pass -AllowOtherGpu to run on another RTX 40 series card."
}

function Get-Uv {
    $existing = Get-ChildItem -Path $Tools -Filter "uv.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing) { return $existing.FullName }
    $zip = Join-Path $Tools "uv.zip"
    $dest = Join-Path $Tools "uv"
    Write-Host "downloading uv"
    Invoke-WebRequest -Uri $UvUrl -OutFile $zip -UseBasicParsing
    Expand-Zip $zip $dest
    $found = Get-ChildItem -Path $dest -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $found) { Fail "uv.exe was not in the uv archive." }
    return $found.FullName
}

function Get-Git {
    $cmd = Get-Command git -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $git = Join-Path $Tools "mingit\cmd\git.exe"
    if (Test-Path -LiteralPath $git) { return $git }
    Write-Host "downloading MinGit"
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" -Headers @{ "User-Agent" = "voxprint-clarity" }
    $asset = $release.assets | Where-Object { $_.name -like "MinGit-*-64-bit.zip" } | Select-Object -First 1
    if (-not $asset) { Fail "GitHub did not list a MinGit 64-bit zip on the latest Git for Windows release." }
    $zip = Join-Path $Tools "mingit.zip"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Zip $zip (Join-Path $Tools "mingit")
    if (-not (Test-Path -LiteralPath $git)) { Fail "MinGit did not contain cmd\git.exe." }
    return $git
}

function Get-Ffmpeg {
    $local = Join-Path $Tools "ffmpeg\bin\ffmpeg.exe"
    $probe = Join-Path $Tools "ffmpeg\bin\ffprobe.exe"
    if (-not (Test-Path -LiteralPath $local)) {
        Write-Host "downloading LGPL ffmpeg"
        $zip = Join-Path $Tools "ffmpeg-lgpl.zip"
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $zip -UseBasicParsing
        $dest = Join-Path $Tools "ffmpeg-unpack"
        Expand-Zip $zip $dest
        $bin = Get-ChildItem -Path $dest -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if (-not $bin) { Fail "The ffmpeg archive did not contain ffmpeg.exe." }
        $target = Join-Path $Tools "ffmpeg"
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
        Move-Item -LiteralPath $bin.Directory.Parent.FullName -Destination $target
    }
    if (-not (Test-Path -LiteralPath $probe)) { Fail "ffprobe.exe is missing next to the LGPL ffmpeg." }
    $build = & $local -hide_banner -buildconf 2>&1 | Out-String
    # Match enabled GPL pieces. "--disable-libx264" contains "libx264" and is the LGPL build.
    if ($build -match "--enable-gpl" -or $build -match "--enable-libx264" -or $build -match "--enable-libx265") {
        Fail "ffmpeg is not an LGPL build ($FfmpegUrl)."
    }
    $encoders = & $local -hide_banner -encoders 2>&1 | Out-String
    if ($encoders -notmatch "av1_nvenc") {
        Fail "This ffmpeg has no av1_nvenc. The restore needs an LGPL build with NVENC."
    }
    return $local
}

$uv = Get-Uv
$git = Get-Git
$ffmpeg = Get-Ffmpeg
$ffprobe = Join-Path (Split-Path -Parent $ffmpeg) "ffprobe.exe"
$env:Path = "$(Split-Path -Parent $git);$(Split-Path -Parent $ffmpeg);$env:Path"
Write-Host "uv: $uv"
Write-Host "git: $git"
Write-Host "ffmpeg: $ffmpeg"

$env:PYTHONPATH = $Root
Write-Host "installing CPython 3.14"
Invoke-Native { & $uv python install 3.14 } "uv could not install CPython 3.14"
$pythonBase = & $uv python find 3.14
if ($LASTEXITCODE -ne 0 -or -not $pythonBase) { Fail "Python 3.14 was not found after uv python install." }
$venv = Join-Path $ResultsDir "venv"
if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
    Invoke-Native { & $uv venv --python $pythonBase $venv } "uv could not create the virtualenv"
}
$python = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { Fail "The virtualenv has no Scripts\python.exe." }

Write-Host "installing the Clarity checkout"
Invoke-Native { & $uv pip install --python $python -e $Root } "the Clarity checkout failed to install"
Write-Host "installing torch 2.14.1 cu130"
Invoke-Native { & $uv pip install --python $python "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cu130" } "torch 2.14.1 cu130 failed to install"
Write-Host "installing Clarity requirements"
Invoke-Native { & $uv pip install --python $python -r (Join-Path $Root "requirements.txt") -r (Join-Path $Root "requirements-bench.txt") } "requirements failed to install"
Write-Host "installing the shared engine from d11d609"
Invoke-Native { & $uv pip install --python $python $EngineUrl } "the public engine install failed"
$lockPath = Join-Path $Root "installer\runtime_lock.json"
$lockJson = Get-Content -Raw -LiteralPath $lockPath | ConvertFrom-Json
$ortGpu = [string]$lockJson.onnxruntime_gpu
$trtLibs = [string]$lockJson.tensorrt_cu13_libs
$trtMeta = [string]$lockJson.tensorrt_cu13
if (-not $ortGpu -or -not $trtLibs -or -not $trtMeta) { Fail "installer\runtime_lock.json is missing the ONNX Runtime / TensorRT pins." }
Write-Host "replacing the CPU onnxruntime wheel and TensorRT 11 with onnxruntime-gpu $ortGpu and TensorRT $trtLibs"
Invoke-Native { & $uv pip uninstall --python $python onnxruntime onnxruntime-gpu tensorrt tensorrt-cu13 tensorrt-cu13-bindings tensorrt-cu13-libs } "could not uninstall the ONNX Runtime wheels"
Invoke-Native { & $uv pip install --python $python --extra-index-url "https://pypi.nvidia.com" "onnxruntime-gpu==$ortGpu" "tensorrt-cu13-libs==$trtLibs" } "onnxruntime-gpu / TensorRT libs failed to install"
Invoke-Native { & $uv pip install --python $python --no-deps "tensorrt-cu13==$trtMeta" } "tensorrt-cu13 metadata pin failed"

$providerCheck = @"
import importlib.metadata as metadata
import onnxruntime as ort

names = {dist.metadata["Name"].lower() for dist in metadata.distributions()}
if "onnxruntime" in names:
    raise SystemExit("The CPU onnxruntime wheel is installed and shadows onnxruntime-gpu.")
providers = ort.get_available_providers()
print("providers: " + ", ".join(providers))
missing = [name for name in ("CUDAExecutionProvider", "TensorrtExecutionProvider") if name not in providers]
if missing:
    raise SystemExit("ONNX Runtime is missing " + ", ".join(missing) + ". onnxruntime-gpu 1.31.0 needs TensorRT 10.14 (nvinfer_10).")
"@
$pcFile = Join-Path $ResultsDir "provider_check.py"
Set-Content -LiteralPath $pcFile -Value $providerCheck -Encoding ascii
Invoke-Native { & $python $pcFile } "ONNX Runtime is missing the TensorRT execution provider"

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
