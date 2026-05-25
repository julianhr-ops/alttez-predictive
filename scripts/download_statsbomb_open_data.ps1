$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = "c:\Alttez predictor\alttez-predictive-engine"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $root ("artifacts\statsbomb-download-" + $stamp + ".log")
$zip = Join-Path $root "artifacts\statsbomb-open-data-master.zip"
$destBase = Join-Path $root "storage\network"
$extract = Join-Path $destBase "open-data-master"
$dest = Join-Path $destBase "statsbomb-open-data-archive"

$worker = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'

function Write-Log([string]`$message) {
    `$timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -LiteralPath '$log' -Value ('[' + `$timestamp + '] ' + `$message)
}

Write-Log 'Starting StatsBomb Open Data download job.'

if (Test-Path '$zip') {
    Remove-Item -LiteralPath '$zip' -Force
    Write-Log 'Removed previous ZIP file.'
}

if (Test-Path '$extract') {
    Remove-Item -LiteralPath '$extract' -Recurse -Force
    Write-Log 'Removed previous extracted temp folder.'
}

if (Test-Path '$dest') {
    Remove-Item -LiteralPath '$dest' -Recurse -Force
    Write-Log 'Removed previous destination folder.'
}

Write-Log 'Downloading ZIP from codeload.github.com...'
curl.exe -L --retry 3 --retry-delay 5 --output '$zip' https://codeload.github.com/statsbomb/open-data/zip/refs/heads/master
if (`$LASTEXITCODE -ne 0) {
    throw 'curl download failed.'
}

Write-Log 'Download complete. Expanding archive...'
Expand-Archive -LiteralPath '$zip' -DestinationPath '$destBase' -Force

Write-Log 'Renaming extracted folder...'
Rename-Item -LiteralPath '$extract' -NewName 'statsbomb-open-data-archive'

Write-Log 'StatsBomb Open Data ready.'
"@

$process = Start-Process `
    -FilePath "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -WindowStyle Hidden `
    -PassThru `
    -ArgumentList @(
        "-NoProfile",
        "-Command",
        $worker
    )

Write-Output ("PID: " + $process.Id)
Write-Output ("LOG: " + $log)
Write-Output ("DEST: " + $dest)
