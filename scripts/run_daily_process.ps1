param(
    [string]$Date = ([DateTime]::UtcNow.ToString("yyyy-MM-dd")),
    [switch]$ApplyMigrations,
    [switch]$WriteSmoke,
    [string]$DatabaseUrl = $null,
    [string]$ProviderOrder = $null,
    [string]$Schema = "public"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $root "venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$preflight = Join-Path $root "scripts\shadow_readiness_check.py"
$migrations = Join-Path $root "scripts\apply_supabase_migrations.py"
$smoke = Join-Path $root "scripts\smoke_live_pipeline.py"
$brief = Join-Path $root "scripts\generate_daily_brief.py"
$challenger = Join-Path $root "scripts\run_challenger_cycle.py"

Set-Location $root

Write-Output "PROCESS: shadow readiness"
& $python $preflight
if ($LASTEXITCODE -ne 0) {
    throw "Shadow readiness check failed."
}

if ($ApplyMigrations) {
    Write-Output "PROCESS: apply migrations"
    $migrationArgs = @($migrations, "--apply")
    if ($DatabaseUrl) {
        $migrationArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @migrationArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Migration application failed."
    }
}

Write-Output "PROCESS: smoke live pipeline"
$smokeArgs = @($smoke, "--date", $Date)
if ($WriteSmoke) {
    $smokeArgs += "--write"
}
if ($ProviderOrder) {
    $smokeArgs += @("--provider-order", $ProviderOrder)
}
if ($DatabaseUrl) {
    $smokeArgs += @("--database-url", $DatabaseUrl)
}
if ($Schema) {
    $smokeArgs += @("--schema", $Schema)
}

& $python @smokeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Smoke pipeline failed."
}

Write-Output "PROCESS: daily decision brief"
$briefArgs = @($brief, "--date", $Date)
if ($DatabaseUrl) {
    $briefArgs += @("--database-url", $DatabaseUrl)
}
if ($env:ALTTEZ_DAILY_BRIEF_DELIVER -eq "true") {
    $briefArgs += "--deliver"
}
& $python @briefArgs
if ($LASTEXITCODE -ne 0) {
    throw "Daily brief generation failed."
}

Write-Output "PROCESS: challenger cycle"
$challengerArgs = @($challenger, "--date", $Date)
if ($DatabaseUrl) {
    $challengerArgs += @("--database-url", $DatabaseUrl)
}
& $python @challengerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Challenger cycle failed."
}

Write-Output "PROCESS: complete"
