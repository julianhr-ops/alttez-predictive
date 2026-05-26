$ErrorActionPreference = "Stop"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$python = Join-Path $root "venv\Scripts\python.exe"
$preflight = Join-Path $root "scripts\shadow_readiness_check.py"
$startScript = Join-Path $root "scripts\start_streamlit_supabase.ps1"
$stopScript = Join-Path $root "scripts\stop_streamlit_supabase.ps1"
$url = "http://127.0.0.1:8501"

Set-Location $root

Write-Output "STEP: stopping any existing Streamlit session"
& $stopScript

Write-Output "STEP: shadow_readiness_check.py"
& $python $preflight
if ($LASTEXITCODE -ne 0) {
    throw "Preflight failed. Dashboard will not start."
}

Write-Output "STEP: starting Streamlit"
& $startScript
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit start helper failed."
}

$ready = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200) {
            $ready = $true
            break
        }
    } catch {
    }

    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Output "STEP: opening browser before http check succeeded"
} else {
    Write-Output "STEP: dashboard is ready"
}

Start-Process $url
Write-Output ("URL: " + $url)
