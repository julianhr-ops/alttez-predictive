$ErrorActionPreference = "Stop"

param(
    [string]$LogPath
)

if (-not $LogPath) {
    $latest = Get-ChildItem -Path "c:\Alttez predictor\alttez-predictive-engine\artifacts" -Filter "statsbomb-download-*.log" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latest) {
        throw "No StatsBomb download log was found."
    }

    $LogPath = $latest.FullName
}

Write-Output ("LOG: " + $LogPath)
Get-Content -LiteralPath $LogPath -Tail 50
