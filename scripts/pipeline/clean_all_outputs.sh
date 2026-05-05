#!/bin/bash
set -e

# Flags (accept in any order)
FORCE=0
DEEP=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --deep) DEEP=1 ;;
  esac
done

# Resolve the directory where the script resides
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Resolve Project Root
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Define directories to clean
DIRS_TO_CLEAN=(
    "$PROJECT_ROOT/outputs"
    "$PROJECT_ROOT/data/intermediate"
    "$PROJECT_ROOT/smoke_tests"
    "$PROJECT_ROOT/.pytest_cache"
    "$PROJECT_ROOT/MagicMock"
    "$PROJECT_ROOT/data/scraper_dumps"
)

if [[ "$DEEP" -eq 1 ]]; then
  # Optional: reclaim virtualenv disk usage (rebuildable via setup scripts).
  DIRS_TO_CLEAN+=("$PROJECT_ROOT/.venv")
fi

echo "===================================================================="
echo "Clean All Project Artifacts & Intermediate Data"
echo "===================================================================="
echo "Working from Project Root: $PROJECT_ROOT"
echo "This script will delete:"
for d in "${DIRS_TO_CLEAN[@]}"; do
    echo "  - $d"
done
echo "  - All .tar.gz, .zip, .tar, and .gz files in $PROJECT_ROOT"
if [[ "$DEEP" -eq 1 ]]; then
  echo
  echo "Deep clean enabled (--deep): also deletes .venv"
fi
echo

# Check if --force flag is provided
if [[ "$FORCE" -eq 1 ]]; then
    REPLY="y"
else
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
fi

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Clean Directories
for d in "${DIRS_TO_CLEAN[@]}"; do
    if [ -d "$d" ]; then
        echo "Deleting Directory: $d"
        rm -rf "$d"
    else
        echo "Skipping (Not found): $d"
    fi
done

# Clean archive files in project root
echo "Cleaning archive files in project root..."
find "$PROJECT_ROOT" -maxdepth 1 -name "*.tar.gz" -type f -delete
find "$PROJECT_ROOT" -maxdepth 1 -name "*.zip" -type f -delete
find "$PROJECT_ROOT" -maxdepth 1 -name "*.tar" -type f -delete
find "$PROJECT_ROOT" -maxdepth 1 -name "*.gz" -type f -delete

echo
echo "===================================================================="
echo "Project cleaning complete. Ready for a fresh start."
echo "===================================================================="
echo
