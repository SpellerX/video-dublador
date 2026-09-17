# ---------------------------------------------------------------------------
#  Video dubber launcher (PowerShell)
#
#    .\run.ps1 check
#    .\run.ps1 input\movie.mp4 --target pt
# ---------------------------------------------------------------------------
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Args)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$python = Join-Path $Root ".python\python.exe"
if (-not (Test-Path $python)) {
    Write-Host ""
    Write-Host "  Portable Python not found at $python"
    Write-Host "  Run the bootstrap first:  python tools\bootstrap.py"
    Write-Host ""
    exit 1
}

$env:PYTHONPATH = $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# ffmpeg must be reachable by name for several libraries.
$env:PATH = (Join-Path $Root ".tools\ffmpeg\bin") + ";" + $env:PATH

& $python -m dublador @Args
exit $LASTEXITCODE
