# Define output directory
$OUTPUT_DIR = "outputs/00_reproducibility"
if (!(Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR
}

Write-Host "Dumping environment to $OUTPUT_DIR..."

# 1. Pip freeze
pip freeze | Out-File -FilePath "$OUTPUT_DIR/pip_freeze.txt" -Encoding utf8

# 2. System info
$sysInfo = @{
    os             = (Get-WmiObject Win32_OperatingSystem).Caption
    python_version = (python --version 2>&1)
    gpu            = "none"
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $sysInfo.gpu = (nvidia-smi --query-gpu=name --format=csv, noheader)
}

$sysInfo | ConvertTo-Json | Out-File -FilePath "$OUTPUT_DIR/system_info.json" -Encoding utf8

Write-Host "Done."
