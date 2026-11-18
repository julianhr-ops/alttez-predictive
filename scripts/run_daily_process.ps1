param(
    [string]$Date = ([DateTime]::UtcNow.ToString("yyyy-MM-dd")),
    [switch]$ApplyMigrations,
    [switch]$WriteSmoke,
    [string]$DatabaseUrl = $null,
    [string]$ProviderOrder = $null,
    [string]$Schema = "public",
    [int]$ChallengerEvaluationMatches = 120
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $root "venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$preflight = Join-Path $root "scripts\shadow_readiness_check.py"
$migrations = Join-Path $root "scripts\apply_supabase_migrations.py"
$smoke = Join-Path $root "scripts\smoke_live_pipeline.py"
$backfill = Join-Path $root "scripts\backfill_match_results.py"
$calibrationFit = Join-Path $root "scripts\fit_probability_calibration.py"
$tournamentReviews = Join-Path $root "scripts\process_tournament_reviews.py"
$brief = Join-Path $root "scripts\generate_daily_brief.py"
$challenger = Join-Path $root "scripts\run_challenger_cycle.py"
$pipelineAlert = Join-Path $root "scripts\send_pipeline_alert.py"
$nbaDaily = Join-Path $root "scripts\nba_daily.py"
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

# NBA runs on free data and is independent of the soccer schedule, so it runs on the
# idle/no_schedule path too. Non-fatal: it must never abort the soccer brief.
function Invoke-NbaStage {
    $script:CurrentStage = "nba predictions"
    Write-Output "PROCESS: nba predictions"
    $nbaArgs = @($nbaDaily, "--date", $Date)
    if ($DatabaseUrl) {
        $nbaArgs += @("--database-url", $DatabaseUrl)
    }
    $deliverNba = (Get-AlttezEnvValue "ALTTEZ_DAILY_BRIEF_DELIVER")
    if ($deliverNba -and $deliverNba.ToLowerInvariant() -in @("1", "true", "yes", "on")) {
        $nbaArgs += "--deliver"
    }
    & $python @nbaArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Output "PROCESS: nba predictions degraded (non-fatal)"
        & $python $pipelineAlert --stage $script:CurrentStage --date $Date --message "NBA predictions failed (non-fatal); see nba_daily exit code $LASTEXITCODE." --level warning | Out-Null
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
    $smokeArgs = @($smoke, "--date", $Date, "--skip-analysis")
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
    $smokeExitCode = $LASTEXITCODE
    $NoSchedule = $smokeExitCode -eq 3
    if ($smokeExitCode -ne 0 -and -not $NoSchedule) {
        throw "Forced live ingestion failed."
    }

    if ($NoSchedule) {
        Write-Output "PROCESS: idle - no international matches scheduled for $Date"
    }
    else {
        Write-Output "PROCESS: prediction evaluations closed during ingestion finalize"
    }

    # Recurring ESPN results backfill (keyless; covers friendlies + WC finals).
    # Closes prediction labels so the challenger evaluates on fresh outcomes.
    # Non-fatal: result enrichment must not abort the daily brief.
    $CurrentStage = "espn results backfill"
    Write-Output "PROCESS: espn results backfill"
    $backfillFrom = ([DateTime]::Parse($Date)).AddDays(-7).ToString("yyyy-MM-dd")
    $backfillArgs = @($backfill, "--source", "espn", "--date-from", $backfillFrom, "--date-to", $Date, "--apply")
    if ($DatabaseUrl) {
        $backfillArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @backfillArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Output "PROCESS: espn results backfill degraded (non-fatal)"
        & $python $pipelineAlert --stage $CurrentStage --date $Date --message "ESPN results backfill failed (non-fatal)." --level warning | Out-Null
    }

    # Cierres de torneo registrados. Idempotente: la tabla de entregas evita
    # reenviar un informe ya entregado. Non-fatal para no bloquear el brief diario.
    $CurrentStage = "tournament review delivery"
    Write-Output "PROCESS: tournament review delivery"
    $tournamentReviewArgs = @($tournamentReviews, "--date", $Date, "--deliver")
    if ($DatabaseUrl) {
        $tournamentReviewArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @tournamentReviewArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Output "PROCESS: tournament review delivery degraded (non-fatal)"
        & $python $pipelineAlert --stage $CurrentStage --date $Date --message "Tournament review delivery failed (non-fatal)." --level warning | Out-Null
    }

    if ($NoSchedule) {
        Invoke-NbaStage
        Write-Output "PROCESS: complete (idle/no_schedule)"
        exit 0
    }

    # Re-ajusta la calibracion de probabilidad sobre las evaluaciones recien
    # cerradas, antes de generar el brief. El artefacto vive en artifacts/ (no
    # versionado), por lo que debe regenerarse cada corrida. Non-fatal: si falla,
    # el comite cae a calibracion neutra (comportamiento sin calibrar).
    $CurrentStage = "probability calibration fit"
    Write-Output "PROCESS: probability calibration fit"
    $calibrationArgs = @($calibrationFit, "--lookback-days", "365", "--apply")
    if ($DatabaseUrl) {
        $calibrationArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @calibrationArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Output "PROCESS: probability calibration fit degraded (non-fatal)"
        & $python $pipelineAlert --stage $CurrentStage --date $Date --message "Probability calibration fit failed (non-fatal); committee falls back to neutral calibration." --level warning | Out-Null
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
    $challengerArgs = @(
        $challenger,
        "--date",
        $Date,
        "--evaluation-matches",
        $ChallengerEvaluationMatches
    )
    if ($DatabaseUrl) {
        $challengerArgs += @("--database-url", $DatabaseUrl)
    }
    & $python @challengerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Challenger cycle failed."
    }

    Invoke-NbaStage

    Write-Output "PROCESS: complete"
}
catch {
    $errorMessage = $_.Exception.Message
    Write-Output "PROCESS: pipeline incident alert"
    & $python $pipelineAlert --stage $CurrentStage --date $Date --message $errorMessage --level critical | Out-Null
    throw
}
