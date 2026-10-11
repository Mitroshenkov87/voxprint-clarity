# Online runtime for Voxprint AI Clarity.
# Python 3.14, torch 2.14.1+cu130, LGPL ffmpeg, and the engine go in
# %LOCALAPPDATA%\Voxprint\shared. Another suite app with the same pins reuses
# that folder. There is no CPU flavor. -SkipRuntime is only for the installer compile check.
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
function Write-Ready([string]$Shared, [string]$Name) {
    $dir = Join-Path $Shared $Name
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Set-Content -LiteralPath (Join-Path $dir "READY") -Value "ready" -Encoding utf8
}
New-Item -ItemType File -Force -Path $Log | Out-Null
if ($SkipRuntime) {
    Write-Log "runtime skipped"
    exit 0
}
if (-not (Test-Path -LiteralPath $Lock)) { throw "runtime lock missing: $Lock" }
$lockJson = Get-Content -Raw -LiteralPath $Lock | ConvertFrom-Json
$pillow = [string]$lockJson.pillow
if (-not $pillow) { throw "runtime lock is missing pillow" }
$hash = (Get-FileHash -LiteralPath $Lock -Algorithm SHA256).Hash.Substring(0, 16).ToLower()
if ($env:VOXPRINT_HOME) {
    $suiteHome = $env:VOXPRINT_HOME
} else {
    $suiteHome = Join-Path $env:LOCALAPPDATA "Voxprint"
}
$shared = Join-Path $suiteHome "shared"
$runtime = Join-Path $shared "runtime\py3.14-torch2.14.1-cu130"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$venvPython = Join-Path $runtime "venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    Write-Log "reusing $runtime"
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
    Write-Log "creating venv $runtime"
    & $uv.FullName venv --python $python "$runtime\venv"
    Write-Log "installing torch 2.14.1 cu130"
    & $uv.FullName pip install --python $venvPython "torch==2.14.1" --index-url "https://download.pytorch.org/whl/cu130"
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    Write-Log "installing Clarity requirements"
    & $uv.FullName pip install --python $venvPython -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }
    Write-Log "installing Pillow $pillow"
    & $uv.FullName pip install --python $venvPython "Pillow==$pillow"
    if ($LASTEXITCODE -ne 0) { throw "Pillow install failed" }
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
}
$ffmpeg = Join-Path $shared "ffmpeg\ffmpeg.exe"
if (Test-Path -LiteralPath $ffmpeg) {
    Write-Log "reusing $ffmpeg"
} else {
    Write-Log "installing LGPL ffmpeg"
    $ffmpegZip = Join-Path $env:TEMP "voxprint-ffmpeg-lgpl.zip"
    Invoke-WebRequest -Uri "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip" -OutFile $ffmpegZip
    $ffmpegExpand = Join-Path $env:TEMP "voxprint-ffmpeg-lgpl"
    if (Test-Path $ffmpegExpand) { Remove-Item -LiteralPath $ffmpegExpand -Recurse -Force }
    Expand-Archive -LiteralPath $ffmpegZip -DestinationPath $ffmpegExpand -Force
    $found = Get-ChildItem -Path $ffmpegExpand -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
    if (-not $found) { throw "the LGPL ffmpeg archive has no ffmpeg.exe" }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ffmpeg) | Out-Null
    Copy-Item -LiteralPath $found.FullName -Destination $ffmpeg -Force
    $probe = Get-ChildItem -Path $ffmpegExpand -Filter "ffprobe.exe" -Recurse | Select-Object -First 1
    if ($probe) { Copy-Item -LiteralPath $probe.FullName -Destination (Join-Path (Split-Path -Parent $ffmpeg) "ffprobe.exe") -Force }
}
foreach ($name in @("python", "torch", "ffmpeg", "engine")) { Write-Ready $shared $name }
$env:VOXPRINT_SHARED = $shared
& $venvPython -c "import json, os; from pathlib import Path; root = Path(os.environ['VOXPRINT_SHARED']); path = root / 'manifest.json'; data = {};
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    data = {}
comps = data.get('components') if isinstance(data.get('components'), dict) else {}
versions = {'python': '3.14', 'torch': '2.14.1+cu130', 'ffmpeg': 'lgpl', 'engine': 'engine-v1.0.0'}
for name, version in versions.items():
    entry = comps.get(name) if isinstance(comps.get(name), dict) else {'users': {}}
    users = entry.get('users') if isinstance(entry.get('users'), dict) else {}
    users['clarity'] = int(users.get('clarity') or 0) + 1
    entry['version'] = version
    entry['users'] = users
    comps[name] = entry
path.write_text(json.dumps({'format_version': 1, 'components': comps}, indent=1, sort_keys=True) + '\n', encoding='utf-8')"
if ($LASTEXITCODE -ne 0) { throw "shared manifest update failed" }
$link = Join-Path $AppDir "runtime"
if (Test-Path $link) { cmd /c rmdir "$link" | Out-Null }
cmd /c mklink /J "$link" "$runtime\venv" | Out-Null
@{ python = "3.14"; torch = "2.14.1"; cuda = "cu130"; pillow = $pillow; shared = $shared; key = $hash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime "runtime-key.json") -Encoding utf8
Write-Log "runtime ready"
exit 0
