$ErrorActionPreference = "Stop"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$python = Join-Path $root "venv\Scripts\python.exe"
$pidFile = Join-Path $root "artifacts\streamlit-supabase.pid"
$preflight = Join-Path $root "scripts\shadow_readiness_check.py"

Set-Location $root

Write-Output "PRECHECK: shadow_readiness_check.py"
& $python $preflight
if ($LASTEXITCODE -ne 0) {
    throw "Preflight failed. Streamlit will not start."
}
Write-Output "PRECHECK: ok"

$process = Start-Process `
    -FilePath $python `
    -WindowStyle Hidden `
    -PassThru `
    -ArgumentList @(
        "-m",
        "streamlit",
        "run",
        "alttez/dashboard/app.py",
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8501"
    )

Set-Content -Path $pidFile -Value $process.Id -Encoding ascii

Write-Output ("PID: " + $process.Id)
Write-Output ("PID_FILE: " + $pidFile)
Write-Output ("URL: http://127.0.0.1:8501")
