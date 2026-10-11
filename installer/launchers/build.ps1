# Compile voxprint-clarity.exe and voxprint-clarity-cli.exe with the .NET Framework C# compiler.
$ErrorActionPreference = "Stop"
$csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
if (-not (Test-Path -LiteralPath $csc)) {
    throw "csc.exe was not found at $csc"
}
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$icon = Join-Path (Split-Path -Parent (Split-Path -Parent $here)) "branding\clarity.ico"
if (-not (Test-Path -LiteralPath $icon)) {
    throw "clarity.ico was not found at $icon"
}
$source = Join-Path $here "Launcher.cs"
$gui = Join-Path $here "voxprint-clarity.exe"
$cli = Join-Path $here "voxprint-clarity-cli.exe"
& $csc /nologo /target:winexe "/win32icon:$icon" "/out:$gui" $source
if ($LASTEXITCODE -ne 0) { throw "voxprint-clarity.exe compile failed ($LASTEXITCODE)" }
& $csc /nologo /target:exe /define:CLI "/win32icon:$icon" "/out:$cli" $source
if ($LASTEXITCODE -ne 0) { throw "voxprint-clarity-cli.exe compile failed ($LASTEXITCODE)" }
Write-Output "wrote $gui"
Write-Output "wrote $cli"
