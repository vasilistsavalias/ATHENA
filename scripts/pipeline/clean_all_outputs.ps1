# Stop script on first error
$ErrorActionPreference = "Stop"

$FORCE = $args -contains "--force"
$DEEP = $args -contains "--deep"

$DIRS_TO_CLEAN = @(
    "outputs",
    "data/intermediate",
    "smoke_tests",
    ".pytest_cache",
    "MagicMock",
    "data/scraper_dumps"
)

if ($DEEP) {
    # Optional: reclaim virtualenv disk usage (rebuildable via setup scripts).
    $DIRS_TO_CLEAN += ".venv"
}

Write-Host "===================================================================="
Write-Host "Clean All Project Artifacts & Intermediate Data"
Write-Host "===================================================================="
Write-Host "This script will delete:"
foreach ($d in $DIRS_TO_CLEAN) {
    Write-Host "  - $d"
}
Write-Host "  - All .tar.gz, .zip, .tar, and .gz files in the root"
if ($DEEP) {
    Write-Host ""
    Write-Host "Deep clean enabled (--deep): also deletes .venv"
}
Write-Host ""

if ($FORCE) {
    $response = "y"
} else {
    $response = Read-Host "Are you sure you want to continue? (y/N)"
}

if ($response -ne "y") {
    Write-Host "Aborted."
    exit 1
}

foreach ($d in $DIRS_TO_CLEAN) {
    if (Test-Path -Path $d) {
        Write-Host "Deleting: $d"
        Remove-Item -Path $d -Recurse -Force
    }
}

Write-Host "Cleaning archive files..."
Get-ChildItem -Path "." -Include "*.tar.gz", "*.zip", "*.tar", "*.gz" -File | Remove-Item -Force

Write-Host ""
Write-Host "===================================================================="
Write-Host "Project cleaning complete. Ready for a fresh start."
Write-Host "===================================================================="
Write-Host ""
