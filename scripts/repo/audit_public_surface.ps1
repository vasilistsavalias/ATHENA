param(
    [string]$OutDir = ".rewrite_audit"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$tracked = Join-Path $OutDir "tracked_files.txt"
$large = Join-Path $OutDir "largest_blobs.txt"
$sensitive = Join-Path $OutDir "sensitive_hits.txt"

git ls-files | Out-File -Encoding utf8 $tracked

git rev-list --objects --all |
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
    Select-String '^blob ' |
    ForEach-Object { $_.Line } |
    Sort-Object { [int]($_.Split(' ')[2]) } -Descending |
    Select-Object -First 100 |
    Out-File -Encoding utf8 $large

$pattern = '(^|/)\.env|google_compute_engine|id_rsa|credentials|secret|token|key|\.pem|\.pfx|\.p12'
git log --all --name-only --pretty=format: |
    Select-String -Pattern $pattern -CaseSensitive:$false |
    ForEach-Object { $_.Line } |
    Sort-Object -Unique |
    Out-File -Encoding utf8 $sensitive

Write-Host "Audit reports written to $OutDir"
