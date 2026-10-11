# Shared install and GPU gate for scripts/test-gpu.ps1 and scripts/vm_test.ps1.
# Windows Server 2025, RTX 4090, NVIDIA driver 600+. See docs/TESTING-GPU.md.
#requires -Version 5.1

$script:ClarityEngineUrl = "voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.2.1.tar.gz"
$script:ClarityFfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip"
$script:ClarityUvUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"

function Fail {
    param(
        [string]$Reason,
        [int]$Code = 1
    )
    Write-Host ""
    Write-Host "FAILED: $Reason"
    if ($script:TranscriptStarted) {
        try { Stop-Transcript | Out-Null } catch { }
        $script:TranscriptStarted = $false
    }
    exit $Code
}

function Invoke-Native {
    param(
        [scriptblock]$Command,
        [string]$Reason
    )
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Fail "$Reason (exit $LASTEXITCODE)"
    }
}

function Expand-ClarityZip {
    param(
        [string]$Zip,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Expand-Archive -LiteralPath $Zip -DestinationPath $Destination -Force
}

function Assert-ClarityHost {
    param(
        [switch]$AllowOtherGpu,
        [switch]$AllowOtherOs
    )
    if ($env:OS -ne "Windows_NT") {
        throw "This PowerShell entry runs on Windows. Linux VMs run scripts/vm_test.sh."
    }
    $caption = (Get-CimInstance Win32_OperatingSystem).Caption
    Write-Host "os: $caption"
    if ($caption -notmatch "Windows Server 2025" -and -not $AllowOtherOs) {
        throw "This script expects Windows Server 2025. Found: $caption. Pass --allow-other-os when the image caption differs."
    }
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidia) {
        throw "nvidia-smi was not found. Install an NVIDIA driver from the 600 series or newer."
    }
    $gpuLine = & nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    if ($LASTEXITCODE -ne 0 -or -not $gpuLine) {
        throw "nvidia-smi did not report a GPU."
    }
    $gpuLine = ($gpuLine | Select-Object -First 1).Trim()
    Write-Host "gpu: $gpuLine"
    $cells = @($gpuLine.Split(",") | ForEach-Object { $_.Trim() })
    if ($cells.Count -lt 3) {
        throw "nvidia-smi returned '$gpuLine', which is not name, driver, compute capability."
    }
    $driver = $cells[$cells.Count - 2]
    $compute = $cells[$cells.Count - 1]
    $gpuName = ($cells[0..($cells.Count - 3)] -join ", ")
    $driverMajor = 0
    if ($driver -match "^(\d+)") { $driverMajor = [int]$Matches[1] }
    if ($driverMajor -lt 600) {
        throw "NVIDIA driver $driver is older than the 600 series. Install a 600-series driver or newer."
    }
    $majorCap = 0
    $minorCap = 0
    if ($compute -match "^(\d+)\.(\d+)") {
        $majorCap = [int]$Matches[1]
        $minorCap = [int]$Matches[2]
    }
    if ($majorCap -lt 8 -or ($majorCap -eq 8 -and $minorCap -lt 9)) {
        throw "GPU compute capability $compute is below 8.9. Clarity needs an RTX 40 series GPU or newer."
    }
    if ($gpuName -notmatch "4090" -and -not $AllowOtherGpu) {
        throw "Expected an RTX 4090 and found '$gpuName'. Pass --allow-other-gpu to run on another RTX 40 series card."
    }
}

function Get-ClarityUv {
    param([string]$Tools)
    $existing = Get-ChildItem -Path $Tools -Filter "uv.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existing) { return $existing.FullName }
    $zip = Join-Path $Tools "uv.zip"
    $dest = Join-Path $Tools "uv"
    Write-Host "downloading uv"
    Invoke-WebRequest -Uri $script:ClarityUvUrl -OutFile $zip -UseBasicParsing
    Expand-ClarityZip $zip $dest
    $found = Get-ChildItem -Path $dest -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $found) { throw "uv.exe was not in the uv archive." }
    return $found.FullName
}

function Get-ClarityFfmpeg {
    param([string]$Tools)
    $local = Join-Path $Tools "ffmpeg\bin\ffmpeg.exe"
    $probe = Join-Path $Tools "ffmpeg\bin\ffprobe.exe"
    if (-not (Test-Path -LiteralPath $local)) {
        Write-Host "downloading LGPL ffmpeg"
        $zip = Join-Path $Tools "ffmpeg-lgpl.zip"
        Invoke-WebRequest -Uri $script:ClarityFfmpegUrl -OutFile $zip -UseBasicParsing
        $dest = Join-Path $Tools "ffmpeg-unpack"
        Expand-ClarityZip $zip $dest
        $bin = Get-ChildItem -Path $dest -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if (-not $bin) { throw "The ffmpeg archive did not contain ffmpeg.exe." }
        $target = Join-Path $Tools "ffmpeg"
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
        Move-Item -LiteralPath $bin.Directory.Parent.FullName -Destination $target
    }
    if (-not (Test-Path -LiteralPath $probe)) { throw "ffprobe.exe is missing next to the LGPL ffmpeg." }
    $build = & $local -hide_banner -buildconf 2>&1 | Out-String
    # Match enabled GPL pieces. "--disable-libx264" contains "libx264" and is the LGPL build.
    if ($build -match "--enable-gpl" -or $build -match "--enable-libx264" -or $build -match "--enable-libx265") {
        throw "ffmpeg is not an LGPL build ($script:ClarityFfmpegUrl)."
    }
    $encoders = & $local -hide_banner -encoders 2>&1 | Out-String
    if ($encoders -notmatch "av1_nvenc") {
        throw "This ffmpeg has no av1_nvenc. The restore needs an LGPL build with NVENC."
    }
    return $local
}

function Install-ClarityGpuPython {
    param(
        [string]$Root,
        [string]$ResultsDir,
        [string]$Uv
    )
    Write-Host "installing CPython 3.14"
    Invoke-Native { & $Uv python install 3.14 } "uv could not install CPython 3.14"
    $pythonBase = & $Uv python find 3.14
    if ($LASTEXITCODE -ne 0 -or -not $pythonBase) { Fail "Python 3.14 was not found after uv python install." }
    $venv = Join-Path $ResultsDir "venv"
    if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
        Invoke-Native { & $Uv venv --python $pythonBase $venv } "uv could not create the virtualenv"
    }
    $python = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) { Fail "The virtualenv has no Scripts\python.exe." }
    Write-Host "installing the Clarity checkout"
    Invoke-Native { & $Uv pip install --python $python -e $Root } "the Clarity checkout failed to install"
    Write-Host "installing torch 2.14.1 cu130"
    Invoke-Native { & $Uv pip install --python $python "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cu130" } "torch 2.14.1 cu130 failed to install"
    Write-Host "installing Clarity requirements"
    Invoke-Native { & $Uv pip install --python $python -r (Join-Path $Root "requirements.txt") -r (Join-Path $Root "requirements-bench.txt") } "requirements failed to install"
    Write-Host "installing the shared engine engine-v1.2.1"
    Invoke-Native { & $Uv pip install --python $python --extra-index-url "https://pypi.nvidia.com" "$script:ClarityEngineUrl" } "the inference engine install failed"
    $pin = Join-Path $Root "installer\pin_runtime.py"
    $lock = Join-Path $Root "installer\runtime_lock.json"
    $gpuReq = Join-Path $Root "requirements-gpu.txt"
    Write-Host "pinning torch and spandrel"
    Invoke-Native { & $python $pin --uv $Uv --python $python --lock $lock --requirements $gpuReq } "the runtime pin failed"
    return $python
}
