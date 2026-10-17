# App environment for VOXPrint AI Clarity.
# The venv holds PySide, the pinned Pillow wheel, and voxprint_engine so the
# shortcut can start. Python 3.14 comes from uv. The CUDA 13 wheels, ONNX
# Runtime GPU, and TensorRT are installed for app id "clarity".
# ensure_runtime writes that ref and does not install Enlarge's default model.
# The engine pin is tag engine-v1.4.0 as a GitHub archive, with the gpu extra.
# voxprint-common 0.1.1 (Menuchah) is the release wheel. The installer downloads
# that wheel and its .sha256 file and refuses a digest that does not match.
# pyproject.toml is at the repository root, so the engine URL has no subdirectory.
# The extra's NVIDIA wheels come from the NVIDIA package index.
# A new venv and a reused venv both install that extra. pin_runtime.py then
# installs requirements-gpu.txt and spandrel. A CPU onnxruntime distribution,
# when present, is uninstalled.
# -SkipRuntime is only for the installer compile check.
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [string]$Requirements = "",
    [string]$GpuRequirements = "",
    [string]$Lock = "",
    [string]$PinScript = "",
    [string]$Log = "",
    [string]$EngineRequirement = "voxprint-engine[gpu] @ https://github.com/Mitroshenkov87/voxprint-enlarge/archive/refs/tags/engine-v1.4.0.tar.gz",
    [string]$CommonRequirement = "voxprint-common @ https://github.com/Mitroshenkov87/voxprint-common/releases/download/v0.1.1/voxprint_common-0.1.1-py3-none-any.whl",
    [switch]$SkipRuntime,
    [switch]$LinkOnly
)
$ErrorActionPreference = "Stop"
function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $line -Encoding utf8
    Write-Host $line
}
New-Item -ItemType File -Force -Path $Log | Out-Null
function Ensure-AppRuntimeLink([string]$LinkApp, [string]$VenvRoot) {
    $pythonName = Join-Path $VenvRoot "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $pythonName)) { throw "shared venv has no $pythonName" }
    New-Item -ItemType Directory -Force -Path $LinkApp | Out-Null
    $link = Join-Path $LinkApp "runtime"
    if (Test-Path -LiteralPath $link) {
        $item = Get-Item -LiteralPath $link -Force
        $reparse = [IO.FileAttributes]::ReparsePoint
        if (($item.Attributes -band $reparse) -eq 0) { throw "runtime path exists and is not a junction: $link" }
        cmd /c rmdir "$link"
        if ($LASTEXITCODE -ne 0) { throw "could not replace the runtime junction $link" }
    }
    $linked = cmd /c mklink /J "$link" "$VenvRoot" 2>&1
    Write-Log "$linked"
    if ($LASTEXITCODE -ne 0) { throw "mklink /J failed (${LASTEXITCODE}) for $link -> $VenvRoot : $linked" }
    $exposed = Join-Path $link "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $exposed)) { throw "runtime junction does not expose $exposed" }
    Write-Log "runtime junction $link -> $VenvRoot"
}
if ($SkipRuntime) {
    Write-Log "runtime skipped"
    exit 0
}
if ($env:VOXPRINT_HOME) {
    $suiteHome = $env:VOXPRINT_HOME
} else {
    $suiteHome = Join-Path $env:LOCALAPPDATA "Voxprint"
}
$shared = Join-Path $suiteHome "shared"
$venvRoot = Join-Path $shared "venv"
if ($LinkOnly) {
    Ensure-AppRuntimeLink $AppDir $venvRoot
    Write-Log "runtime junction ready"
    exit 0
}
if (-not $Requirements -or -not $Lock -or -not $GpuRequirements -or -not $PinScript) {
    throw "Requirements, GpuRequirements, Lock, and PinScript are required"
}
if (-not (Test-Path -LiteralPath $Lock)) { throw "runtime lock missing: $Lock" }
$lockJson = Get-Content -Raw -LiteralPath $Lock | ConvertFrom-Json
$pillow = [string]$lockJson.pillow
if (-not $pillow) { throw "runtime lock is missing pillow" }
New-Item -ItemType Directory -Force -Path $venvRoot | Out-Null
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    Write-Log "reusing $venvRoot"
} else {
    $uvDir = Join-Path $env:TEMP "voxprint-uv"
    New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
    $uvZip = Join-Path $uvDir "uv.zip"
    Write-Log "downloading uv"
    Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $uvZip
    Expand-Archive -LiteralPath $uvZip -DestinationPath $uvDir -Force
    $uv = Get-ChildItem -Path $uvDir -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $uv) { throw "uv.exe was not in the archive" }
    Write-Log "installing CPython 3.14"
    & $uv.FullName python install 3.14
    $python = & $uv.FullName python find 3.14
    if (-not $python) { throw "Python 3.14 was not found after uv python install" }
    Write-Log "creating venv $venvRoot"
    & $uv.FullName venv --python $python --seed "$venvRoot"
}
& $venvPython -c "import pip"
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw "pip is missing from the shared venv" }
}
$here = $PSScriptRoot
$fetch = Join-Path $here "fetch_common.py"
$stage = Join-Path $here "stage_suite_runtime.py"
if (-not (Test-Path -LiteralPath $fetch)) { throw "fetch_common.py is missing beside install-runtime.ps1" }
if (-not (Test-Path -LiteralPath $stage)) { throw "stage_suite_runtime.py is missing beside install-runtime.ps1" }
Write-Log "verifying $CommonRequirement"
$commonDir = Join-Path $env:TEMP "voxprint-clarity-common"
New-Item -ItemType Directory -Force -Path $commonDir | Out-Null
$wheelText = & $venvPython $fetch --lock $Lock --dest $commonDir
if ($LASTEXITCODE -ne 0) { throw "voxprint-common download failed" }
$wheel = "$wheelText".Trim()
if (-not $wheel) { throw "voxprint-common download did not print a wheel path" }
Write-Log "installing verified voxprint-common wheel"
& $venvPython -m pip install $wheel
if ($LASTEXITCODE -ne 0) { throw "voxprint-common install failed" }
if (-not (Test-Path -LiteralPath (Join-Path $venvRoot "runtime-key.json"))) {
    Write-Log "installing Clarity requirements"
    $filtered = Join-Path $env:TEMP "voxprint-clarity-requirements.txt"
    Get-Content -LiteralPath $Requirements | Where-Object { $_ -notmatch 'voxprint-common' } | Set-Content -LiteralPath $filtered -Encoding utf8
    & $venvPython -m pip install -r $filtered
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
}
Write-Log "installing the shared engine"
& $venvPython -m pip install --extra-index-url "https://pypi.nvidia.com" "$EngineRequirement"
if ($LASTEXITCODE -ne 0) { throw "engine install failed" }
Write-Log "staging enlarge.suite_runtime"
& $venvPython $stage --python $venvPython --lock $Lock
if ($LASTEXITCODE -ne 0) { throw "enlarge.suite_runtime install failed" }
Write-Log "ensuring the shared runtime for clarity"
# {app} does not exist yet during PrepareToInstall, so this step cannot import
# clarity. ensure_runtime("clarity") records the ref and does not install
# Enlarge's default model. --install-suite repeats that call once the app
# files are on disk.
$prepare = Join-Path $env:TEMP "voxprint-clarity-prepare-runtime.py"
@'
from enlarge.suite_runtime import SuiteInstallError, ensure_runtime

def _say(text):
    if "Enlarge" in text and "3.14" in text:
        return "Python 3.14 was not found. Clarity installs Python 3.14 with uv during setup."
    if "Enlarge" in text:
        return "VOXPrint inference engine missing, run repair."
    return text

try:
    report = ensure_runtime(
        "clarity",
        progress=lambda message: print(_say(message), flush=True),
    )
except SuiteInstallError as exc:
    print(_say(str(exc)), flush=True)
    raise SystemExit(1)
if not report.ok:
    for item in report.components:
        if item.state in {"failed", "missing"}:
            print(f"{item.id}: {_say(item.detail)}", flush=True)
    raise SystemExit(1)
raise SystemExit(0)
'@ | Set-Content -LiteralPath $prepare -Encoding utf8
& $venvPython $prepare
$prepareCode = $LASTEXITCODE
Remove-Item -LiteralPath $prepare -Force -ErrorAction SilentlyContinue
if ($prepareCode -ne 0) { throw "shared runtime install failed" }
Write-Log "installing Pillow $pillow"
& $venvPython -m pip install "Pillow==$pillow"
if ($LASTEXITCODE -ne 0) { throw "Pillow install failed" }
Write-Log "pinning torch and spandrel into the shared venv"
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$pinOutput = & $venvPython $PinScript --python $venvPython --lock $Lock --requirements $GpuRequirements 2>&1
$pinCode = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($null -ne $pinOutput) {
    foreach ($line in @($pinOutput)) {
        if ($null -ne $line -and "$line" -ne "") {
            Write-Log ([string]$line)
        }
    }
}
if ($pinCode -ne 0) { throw "runtime pin failed" }
Write-Log "installing Pillow $pillow"
& $venvPython -m pip install "Pillow==$pillow"
if ($LASTEXITCODE -ne 0) { throw "Pillow install failed" }
Ensure-AppRuntimeLink $AppDir $venvRoot
@{ pillow = $pillow; shared = $shared } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $venvRoot "runtime-key.json") -Encoding utf8
Write-Log "runtime ready"
exit 0
