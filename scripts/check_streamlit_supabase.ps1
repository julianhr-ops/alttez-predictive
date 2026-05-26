$ErrorActionPreference = "Stop"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$pidFile = Join-Path $root "artifacts\streamlit-supabase.pid"

if (-not (Test-Path $pidFile)) {
    Write-Output "STATUS: no-pid-file"
    exit 0
}

$streamlitPid = (Get-Content $pidFile | Select-Object -First 1).Trim()
if (-not $streamlitPid) {
    Write-Output "STATUS: empty-pid-file"
    exit 0
}

$process = Get-Process -Id $streamlitPid -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Write-Output ("STATUS: dead PID=" + $streamlitPid)
    exit 0
}

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8501" -UseBasicParsing -TimeoutSec 5
    Write-Output ("STATUS: http-" + $response.StatusCode + " PID=" + $process.Id + " URL=http://127.0.0.1:8501")
} catch {
    Write-Output ("STATUS: running-no-http PID=" + $process.Id + " DETAIL=" + $_.Exception.Message)
}
