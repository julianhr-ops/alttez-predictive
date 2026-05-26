param(
    [string]$Date = (Get-Date).ToString("yyyy-MM-dd"),
    [string]$ReportDir = "artifacts/reports",
    [string]$DatabaseUrl = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No se encontro el interprete en $python"
}

$args = @("scripts\generate_daily_brief.py", "--date", $Date, "--report-dir", $ReportDir, "--deliver")
if ($DatabaseUrl) {
    $args += @("--database-url", $DatabaseUrl)
}

& $python @args
