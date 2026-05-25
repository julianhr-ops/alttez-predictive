$ErrorActionPreference = "Stop"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $root ("artifacts\streamlit-supabase-" + $stamp + ".out.log")
$err = Join-Path $root ("artifacts\streamlit-supabase-" + $stamp + ".err.log")

Set-Location $root

$process = Start-Process `
    -FilePath "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -WindowStyle Hidden `
    -PassThru `
    -ArgumentList @(
        "-NoProfile",
        "-Command",
        "Set-Location '$root'; & 'venv\Scripts\python.exe' -m streamlit run alttez/dashboard/app.py --server.headless true --server.address 127.0.0.1 --server.port 8501"
    ) `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err

Write-Output ("PID: " + $process.Id)
Write-Output ("OUT: " + $out)
Write-Output ("ERR: " + $err)
