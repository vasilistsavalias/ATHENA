Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$forbidden = @(
    '^AGENTS\.md$',
    '^debate\.md$',
    '^\.github/copilot-instructions\.md$',
    '^\.vscode/',
    '^outputs/',
    '^00_logs/',
    '^smoke_tests/',
    '^data/01_raw/',
    '^docs/',
    '^plan_v2\.md$',
    '^0\.READMEs/gcp_execution_guide_local\.md$',
    '^0\.READMEs/outputs_v3\.md$',
    '^0\.READMEs/v3_analysis\.md$',
    '^docs/thesis_latex/',
    '^docs/presentations/_archive/',
    '^docs/presentations/.*\.(pptx|pdf)$',
    '^docs/reports/',
    '^powerpoint/',
    '^data/models/weights/.*\.(pt|pth)$',
    '(^|/)\.env$'
)

$tracked = git ls-files
$hits = @()
foreach ($line in $tracked) {
    foreach ($pattern in $forbidden) {
        if ($line -match $pattern) {
            $hits += $line
            break
        }
    }
}

if ($hits.Count -gt 0) {
    Write-Host "Forbidden tracked artifacts detected:" -ForegroundColor Red
    $hits | Sort-Object -Unique | ForEach-Object { Write-Host " - $_" }
    exit 1
}

Write-Host "Private artifact check passed."
