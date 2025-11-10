#!/bin/bash
# build_test.sh - Quick test script for GAP Python bindings build

set -e  # Exit on any error

echo "======================================"
echo "GAP Python Bindings - Build Test"
echo "======================================"

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Check CMake  
cmake --version || { echo "Error: CMake not found"; exit 1; }

# Check for CUDA (optional)
if nvcc --version >/dev/null 2>&1; then
    echo "✓ CUDA found:"
    nvcc --version | head -n 1
    CUDA_AVAILABLE=true
else
    echo "! CUDA not found - building CPU-only version"
    CUDA_AVAILABLE=false
fi

# Create test build directory
BUILD_DIR="build_test"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

echo ""
echo "Testing CMake configuration..."

# Test CMake configuration
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS="-DGAP_BUILD_PYTHON_BINDINGS=ON"
if [ "$CUDA_AVAILABLE" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGAP_ENABLE_CUDA=ON"
else  
    CMAKE_ARGS="$CMAKE_ARGS -DGAP_ENABLE_CUDA=OFF"
fi

echo "Running: cmake $CMAKE_ARGS .."
cmake $CMAKE_ARGS .. || { echo "Error: CMake configuration failed"; exit 1; }

echo ""
echo "Testing build system..."

# Test if build targets exist
echo "Checking for gap_solver target..."
make help | grep gap_solver >/dev/null || { echo "Error: gap_solver target not found"; exit 1; }

echo ""
echo "✓ Build system configuration successful!"
echo ""
echo "To complete the build, run:"
echo "  cd $BUILD_DIR"
echo "  make gap_solver -j$(nproc)"
echo ""
echo "Or to build and install Python package:"
echo "  pip install -e ."

cd ..
echo ""
echo "Build test completed successfully!"