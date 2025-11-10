#!/bin/bash
# build_with_cuda.sh - Build GAP with CUDA enabled

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Virtual environment path
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_EXE="$VENV_PATH/bin/python"

echo "======================================"
echo "GAP Build with CUDA (Virtual Environment)"
echo "======================================"

# Check if virtual environment exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"

# Check for CUDA
if ! nvcc --version >/dev/null 2>&1; then
    echo "Error: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA version: $(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')"

# Build directory
BUILD_DIR="$PROJECT_ROOT/build_cuda"

# Clean build directory if requested
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake configuration with CUDA enabled
echo ""
echo "Configuring with CMake (CUDA enabled)..."

cmake \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -DGAP_BUILD_PYTHON_BINDINGS=ON \
    -DGAP_ENABLE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ..

echo ""
echo "Building..."
make -j$(nproc)