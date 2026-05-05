#!/bin/bash

# Define output directory
OUTPUT_DIR="outputs/00_reproducibility"
mkdir -p "$OUTPUT_DIR"

echo "Dumping environment to $OUTPUT_DIR..."

# 1. Pip freeze
pip freeze > "$OUTPUT_DIR/pip_freeze.txt"

# 2. Conda export (if available)
if command -v conda &> /dev/null
then
    conda env export > "$OUTPUT_DIR/environment.yml"
fi

# 3. System info (minimal)
echo "{" > "$OUTPUT_DIR/system_info.json"
echo "  \"os\": \"$(uname -a)\"," >> "$OUTPUT_DIR/system_info.json"
echo "  \"python_version\": \"$(python --version)\"," >> "$OUTPUT_DIR/system_info.json"
if command -v nvidia-smi &> /dev/null
then
    echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader)\"" >> "$OUTPUT_DIR/system_info.json"
else
    echo "  \"gpu\": \"none\"" >> "$OUTPUT_DIR/system_info.json"
fi
echo "}" >> "$OUTPUT_DIR/system_info.json"

echo "Done."
