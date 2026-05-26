$ErrorActionPreference = "Stop"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$pidFile = Join-Path $root "artifacts\streamlit-supabase.pid"

$stopped = $false

if (Test-Path $pidFile) {
    $streamlitPid = (Get-Content $pidFile | Select-Object -First 1).Trim()
    if ($streamlitPid) {
        $process = Get-Process -Id $streamlitPid -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            Stop-Process -Id $process.Id -Force
            Write-Output ("STOPPED_PID: " + $process.Id)
            $stopped = $true
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -eq 8501 } |
    Select-Object -ExpandProperty OwningProcess -Unique

foreach ($listenerPid in $listeners) {
    $process = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
    if ($null -ne $process) {
        Stop-Process -Id $process.Id -Force
        Write-Output ("STOPPED_PORT_8501_PID: " + $process.Id)
        $stopped = $true
    }
}

if (-not $stopped) {
    Write-Output "STOPPED: none"
}
