$ErrorActionPreference = "Stop"

$root = (git rev-parse --show-toplevel).Trim()
$hooksDir = Join-Path $root ".git\hooks"
$sourceHook = Join-Path $root ".githooks\pre-push"
$targetHook = Join-Path $hooksDir "pre-push"

if (-not (Test-Path $hooksDir)) {
    New-Item -ItemType Directory -Path $hooksDir | Out-Null
}

try {
    Copy-Item -Path $sourceHook -Destination $targetHook -Force
    Write-Output ("INSTALLED: " + $targetHook)
} catch {
    Write-Output "INSTALL_FAILED: cannot write to .git/hooks in this workspace."
    Write-Output "FALLBACK: run `git config core.hooksPath .githooks` if your checkout allows local config writes."
    Write-Output "FALLBACK: or copy `.githooks/pre-push` to `.git/hooks/pre-push` manually."
    Write-Output ("DETAIL: " + $_.Exception.Message)
}
