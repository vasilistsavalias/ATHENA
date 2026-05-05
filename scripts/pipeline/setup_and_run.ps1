# Stop script on first error
$ErrorActionPreference = "Stop"
$env:PIP_NO_INPUT = "1"

# --- 1. Setup Virtual Environment ---
Write-Host ">>> Setting up Python virtual environment..."
if (-not (Test-Path -Path ".venv")) {
    python -m venv .venv
    Write-Host "Virtual environment created."
} else {
    Write-Host "Virtual environment already exists."
}

# --- 2. Activate Virtual Environment ---
Write-Host ">>> Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

# --- 3. Install Dependencies ---
Write-Host ">>> Installing dependencies from requirements.txt..."
pip install --upgrade --no-input -r pipeline_requirements.txt

# --- 4. Install Project Package ---
Write-Host ">>> Installing project package in editable mode..."
pip install --no-input -e .

# --- 5. Run the Pipeline ---
Write-Host ">>> Running the main pipeline..."
$env:PYTHONPATH = "src"
python src/thesis_pipeline/main.py $args

Write-Host ">>> Pipeline execution finished."
