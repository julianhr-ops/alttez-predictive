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
$backfill = Join-Path $root "scripts\backfill_match_results.py"
$brief = Join-Path $root "scripts\generate_daily_brief.py"
$challenger = Join-Path $root "scripts\run_challenger_cycle.py"
$pipelineAlert = Join-Path $root "scripts\send_pipeline_alert.py"
$envPath = Join-Path $root ".env"

function Get-AlttezEnvValue {
    param([string]$Name)

    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) {
        return $processValue.Trim()
    }

    if (-not (Test-Path $envPath)) {
        return $null
    }

    $line = Get-Content -Path $envPath | Where-Object {
        $_ -match "^\s*$([regex]::Escape($Name))\s*="
    } | Select-Object -First 1

    if (-not $line) {
        return $null
    }

    return (($line -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

function Clear-LoopbackProxyTrap {
    foreach ($Name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")) {
        $value = [Environment]::GetEnvironmentVariable($Name, "Process")
        if ($value -and $value -match "^https?://(127\.0\.0\.1|localhost):9/?$") {
            [Environment]::SetEnvironmentVariable($Name, $null, "Process")
            Write-Output "PROCESS: cleared invalid local proxy $Name"
        }
    }
}

Set-Location $root
Clear-LoopbackProxyTrap

try {
    $CurrentStage = "shadow readiness"
    Write-Output "PROCESS: shadow readiness"
    & $python $preflight
    if ($LASTEXITCODE -ne 0) {
        throw "Shadow readiness check failed."
    }
    if ($ApplyMigrations) {
        $CurrentStage = "apply migrations"
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

    $CurrentStage = "forced live ingestion and evaluation closure"
    Write-Output "PROCESS: forced live ingestion and evaluation closure"
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
        throw "Forced live ingestion failed."
    }

    Write-Output "PROCESS: prediction evaluations closed during ingestion finalize"

    # Recurring ESPN results backfill (keyless; covers friendlies + WC finals).
    # Closes prediction labels so the challenger evaluates on fresh outcomes.
    # Non-fatal: result enrichment must not abort the daily brief.
    $CurrentStage = "espn results backfill"
    Write-Output "PROCESS: espn results backfill"
    $backfillFrom = ([DateTime]::Parse($Date)).AddDays(-3).ToString("yyyy-MM-dd")
    $backfillArgs = @($backfill, "--source", "espn", "--date-from", $backfillFrom, "--date-to", $Date, "--apply")
    if ($DatabaseUrl) {
        $backfillArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @backfillArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Output "PROCESS: espn results backfill degraded (non-fatal)"
        & $python $pipelineAlert --stage $CurrentStage --date $Date --message "ESPN results backfill failed (non-fatal)." --level warning | Out-Null
    }

    $CurrentStage = "committee daily decision brief"
    Write-Output "PROCESS: committee daily decision brief"
    $briefArgs = @($brief, "--date", $Date)
    if ($DatabaseUrl) {
        $briefArgs += @("--database-url", $DatabaseUrl)
    }
    $deliverBrief = (Get-AlttezEnvValue "ALTTEZ_DAILY_BRIEF_DELIVER")
    if ($deliverBrief -and $deliverBrief.ToLowerInvariant() -in @("1", "true", "yes", "on")) {
        $briefArgs += "--deliver"
    }
    & $python @briefArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Daily brief generation failed."
    }

    $CurrentStage = "challenger cycle"
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
}
catch {
    $errorMessage = $_.Exception.Message
    Write-Output "PROCESS: pipeline incident alert"
    & $python $pipelineAlert --stage $CurrentStage --date $Date --message $errorMessage --level critical | Out-Null
    throw
}
