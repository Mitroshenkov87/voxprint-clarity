# Online runtime for Voxprint AI Clarity.
# Python 3.14 and torch 2.14.1+cu130 into %LOCALAPPDATA%\Voxprint\runtime-<key>.
# There is no CPU flavor. -SkipRuntime is only for the installer compile check.
param(
    [Parameter(Mandatory = $true)][string]$AppDir,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [Parameter(Mandatory = $true)][string]$Lock,
    [Parameter(Mandatory = $true)][string]$Log,
    [string]$BenchRequirements = "",
    [string]$EngineRequirement = "voxprint-engine @ git+https://github.com/Mitroshenkov87/voxprint-enlarge@engine-v1.0.0",
    [switch]$SkipRuntime
)
$ErrorActionPreference = "Stop"
function Write-Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $Log -Value $line -Encoding utf8
    Write-Host $line
}
New-Item -ItemType File -Force -Path $Log | Out-Null
if ($SkipRuntime) {
    Write-Log "runtime skipped"
    exit 0
}
if (-not (Test-Path -LiteralPath $Lock)) { throw "runtime lock missing: $Lock" }
$hash = (Get-FileHash -LiteralPath $Lock -Algorithm SHA256).Hash.Substring(0, 16).ToLower()
$runtime = Join-Path $env:LOCALAPPDATA "Voxprint\runtime-$hash"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
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
Write-Log "creating venv $runtime"
& $uv.FullName venv --python $python "$runtime\venv"
$venvPython = Join-Path $runtime "venv\Scripts\python.exe"
Write-Log "installing torch 2.14.1 cu130"
& $uv.FullName pip install --python $venvPython "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cu130"
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
Write-Log "installing Clarity requirements"
& $uv.FullName pip install --python $venvPython -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
if ($BenchRequirements) {
    Write-Log "installing restore requirements"
    & $uv.FullName pip install --python $venvPython -r $BenchRequirements
    if ($LASTEXITCODE -ne 0) { throw "restore requirements install failed" }
}
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
Write-Log "installing onnxruntime-gpu over the engine CPU wheel"
& $uv.FullName pip install --python $venvPython --upgrade "onnxruntime-gpu"
if ($LASTEXITCODE -ne 0) { throw "onnxruntime-gpu install failed" }
$link = Join-Path $AppDir "runtime"
if (Test-Path $link) { cmd /c rmdir "$link" | Out-Null }
cmd /c mklink /J "$link" "$runtime\venv" | Out-Null
@{ python = "3.14"; torch = "2.14.1"; cuda = "cu130"; key = $hash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime "runtime-key.json") -Encoding utf8
Write-Log "runtime ready"
exit 0
