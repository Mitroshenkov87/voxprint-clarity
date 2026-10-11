# App environment for Voxprint AI Clarity.
# The venv holds PySide, the pinned Pillow wheel, and voxprint_engine so the
# shortcut can start. Python 3.14, the CUDA 13 wheels, ONNX Runtime GPU,
# TensorRT, and the engine's default model are installed by
# voxprint_engine.suite_runtime.ensure_runtime (holder "clarity").
# The engine pin is tag engine-v1.1.0.
# -SkipRuntime is only for the installer compile check.
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [string]$Requirements = "",
    [string]$Lock = "",
    [string]$Log = "",
    [string]$EngineRequirement = "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.1.0",
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
if (-not $Requirements -or -not $Lock) { throw "Requirements and Lock are required" }
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
    Write-Log "installing Clarity requirements"
    & $uv.FullName pip install --python $venvPython -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Log "downloading MinGit so the public engine can be cloned"
        $gitZip = Join-Path $uvDir "mingit.zip"
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" -Headers @{ "User-Agent" = "voxprint-clarity" }
        $asset = $release.assets | Where-Object { $_.name -like "MinGit-*-64-bit.zip" } | Select-Object -First 1
        if (-not $asset) { throw "GitHub did not list a MinGit 64-bit zip" }
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $gitZip
        Expand-Archive -LiteralPath $gitZip -DestinationPath (Join-Path $uvDir "mingit") -Force
        $env:Path = "$(Join-Path $uvDir 'mingit\cmd');$env:Path"
    }
    Write-Log "installing the shared engine"
    & $uv.FullName pip install --python $venvPython $EngineRequirement
    if ($LASTEXITCODE -ne 0) { throw "engine install failed" }
}
Write-Log "ensuring the shared runtime for clarity"
& $venvPython -c "from voxprint_engine.suite_runtime import ensure_runtime; report = ensure_runtime('clarity', progress=lambda message: print(message, flush=True)); raise SystemExit(0 if report.ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "shared runtime install failed" }
Write-Log "installing Pillow $pillow"
& $venvPython -m pip install "Pillow==$pillow"
if ($LASTEXITCODE -ne 0) { throw "Pillow install failed" }
Ensure-AppRuntimeLink $AppDir $venvRoot
@{ pillow = $pillow; shared = $shared } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $venvRoot "runtime-key.json") -Encoding utf8
Write-Log "runtime ready"
exit 0
