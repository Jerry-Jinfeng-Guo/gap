#!/bin/bash
# clean.sh - Clean build artifacts and directories

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_ROOT="$(pwd)"
fi

echo "======================================"
echo "GAP Clean Script"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Directories to clean
BUILD_DIRS=(
    "build"
    "build_*"
    ".venv"  # Optional: clean virtual environment
)

# Files to clean
CLEAN_FILES=(
    "*.so"
    "*.pyc"
    "__pycache__"
    "CMakeCache.txt"
    "compile_commands.json"
)

# Function to ask user confirmation
confirm() {
    while true; do
        read -p "$1 [y/N] " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            "" ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

cd "$PROJECT_ROOT"

# Clean build directories
echo "Found build directories:"
for pattern in "${BUILD_DIRS[@]}"; do
    if [[ "$pattern" == ".venv" ]]; then
        # Special handling for virtual environment
        if [ -d ".venv" ]; then
            echo "  .venv/ (Python virtual environment)"
            if confirm "Remove Python virtual environment (.venv)?"; then
                rm -rf .venv
                echo "  ✓ Removed .venv/"
            else
                echo "  → Keeping .venv/"
            fi
        fi
    else
        # Handle build directory patterns
        for dir in $pattern; do
            if [ -d "$dir" ] && [ "$dir" != "$pattern" ]; then
                echo "  $dir/"
            fi
        done
    fi
done

echo ""
if confirm "Remove all build directories?"; then
    for pattern in "${BUILD_DIRS[@]}"; do
        if [[ "$pattern" != ".venv" ]]; then
            for dir in $pattern; do
                if [ -d "$dir" ] && [ "$dir" != "$pattern" ]; then
                    rm -rf "$dir"
                    echo "  ✓ Removed $dir/"
                fi
            done
        fi
    done
else
    echo "  → Keeping build directories"
fi

# Clean individual files
echo ""
echo "Cleaning build artifacts..."
find . -name "*.so" -type f -delete 2>/dev/null && echo "  ✓ Removed *.so files" || true
find . -name "*.pyc" -type f -delete 2>/dev/null && echo "  ✓ Removed *.pyc files" || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null && echo "  ✓ Removed __pycache__ directories" || true
find . -name "CMakeCache.txt" -type f -delete 2>/dev/null && echo "  ✓ Removed CMakeCache.txt files" || true

echo ""
echo "✓ Cleanup completed!"
echo ""
echo "To rebuild:"
echo "  ./build.sh                 # CPU build"
echo "  ./build.sh --cuda ON       # CUDA build"