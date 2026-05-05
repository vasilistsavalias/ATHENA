param(
    [string]$OutDir = "external_weights"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$targets = @(
    @{
        Name = "yolov8n.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        Sha256 = ""
    }
)

foreach ($target in $targets) {
    $dst = Join-Path $OutDir $target.Name
    Write-Host "Downloading $($target.Name) ..."
    Invoke-WebRequest -Uri $target.Url -OutFile $dst

    if ($target.Sha256 -and $target.Sha256.Trim().Length -gt 0) {
        $actual = (Get-FileHash -Algorithm SHA256 -Path $dst).Hash.ToLowerInvariant()
        $expected = $target.Sha256.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "Checksum mismatch for $($target.Name). expected=$expected actual=$actual"
        }
    } else {
        Write-Host "No pinned SHA256 configured for $($target.Name). Verify manually if needed."
    }
}

Write-Host "Done. Weights saved in: $OutDir"
