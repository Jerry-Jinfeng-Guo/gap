#!/bin/bash
# build_with_venv.sh - Build GAP with proper virtual environment Python

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Virtual environment path
VENV_PATH="$PROJECT_ROOT/.venv"
PYTHON_EXE="$VENV_PATH/bin/python"

echo "======================================"
echo "GAP Build with Virtual Environment"
echo "======================================"

# Check if virtual environment exists
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it first with:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install pybind11[global]"
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"

# Check if pybind11 is installed
if ! $PYTHON_EXE -m pybind11 --version >/dev/null 2>&1; then
    echo "Installing pybind11..."
    $PYTHON_EXE -m pip install "pybind11[global]"
fi

echo "pybind11 version: $($PYTHON_EXE -m pybind11 --version)"
echo "pybind11 CMake dir: $($PYTHON_EXE -m pybind11 --cmakedir)"

# Build directory
BUILD_DIR="$PROJECT_ROOT/build"

# Clean build directory if requested
if [ "$1" = "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# CMake configuration with proper Python
echo ""
echo "Configuring with CMake..."

cmake \
    -DPython3_EXECUTABLE="$PYTHON_EXE" \
    -DGAP_BUILD_PYTHON_BINDINGS=ON \
    -DGAP_ENABLE_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    ..

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "✓ Build completed successfully!"
echo "Python module built at: $BUILD_DIR/bindings/gap_solver*.so"