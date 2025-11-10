# GAP Python Bindings - Build Guide

## Quick Start

### Option 1: Use Unified Build Script (Recommended)

```bash
# CPU-only build (default)
./build.sh

# CUDA-enabled build  
./build.sh --cuda ON

# Clean CUDA build
./build.sh --cuda ON --clean

# Debug build without Python bindings
./build.sh --python OFF --build-type Debug

# Show all options
./build.sh --help
```

### Option 2: Use Specific Scripts (Alternative)

```bash
# CPU-only build
./scripts/build_with_venv.sh

# CUDA-enabled build (if CUDA available)
./scripts/build_with_cuda.sh
```

### Option 2: Manual Build

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install "pybind11[global]"

# Build with proper Python
mkdir build && cd build
cmake -DPython3_EXECUTABLE="../.venv/bin/python" \
      -DGAP_BUILD_PYTHON_BINDINGS=ON \
      -DGAP_ENABLE_CUDA=OFF \
      ..
make gap_solver -j$(nproc)
```

## Understanding pybind11 Detection

GAP's CMake configuration tries to find pybind11 in this order:

### ✅ **Best Case: Virtual Environment Detection**
```
-- Found pybind11 via Python: /path/to/venv/lib/python3.x/site-packages/pybind11/share/cmake/pybind11
-- ✓ pybind11 found via Python and CMake CONFIG
-- pybind11 source: Python virtual environment
```

### ⚠️ **Fallback 1: System Installation**
```
-- pybind11 not found via Python
-- ✓ pybind11 found via CMake CONFIG (system installation)  
-- pybind11 source: System CMake
```

### ⚠️ **Fallback 2: GitHub Download**
```
-- pybind11 not found via Python
-- Could NOT find pybind11 (missing: pybind11_DIR)
-- pybind11 not found via CMake CONFIG, fetching from GitHub...
-- ✓ pybind11 fetched from GitHub
-- pybind11 source: GitHub (FetchContent)
```

## Why GitHub Fallback Happens

The GitHub fallback occurs when:

1. **Virtual environment not activated properly**
   - Solution: Use build scripts or set `Python3_EXECUTABLE` explicitly

2. **pybind11 not installed in current Python environment**
   ```bash
   pip install "pybind11[global]"
   ```

3. **Wrong Python interpreter detected**
   ```bash
   # Force specific Python
   cmake -DPython3_EXECUTABLE="/path/to/specific/python" ..
   ```

4. **System CMake cache issues**
   ```bash
   # Clear cache and reconfigure
   rm -rf build && mkdir build
   ```

## CUDA Build Issues

### Missing CUDA Headers Error
```
error: 'cudaError_t' was not declared in this scope
```
**Fixed in**: Added `#include <cuda_runtime.h>` to `cuda_bindings.cpp`

### Type Registration Conflict
```
ImportError: generic_type: type "GPUNewtonRaphson" is already registered!
```
**Fixed in**: Changed to factory functions instead of direct class binding

## Testing Your Build

### CPU-only Build
```bash
cd /path/to/gap
python -c "
import sys; sys.path.insert(0, 'build/lib')
import gap_solver
print('Available backends:', gap_solver.get_available_backends())
print('CUDA available:', gap_solver.is_cuda_available())
"
```

### CUDA Build
```bash
cd /path/to/gap  
python -c "
import sys; sys.path.insert(0, 'build_cuda/lib')
import gap_solver
print('Available backends:', gap_solver.get_available_backends())
print('CUDA available:', gap_solver.is_cuda_available())
print('CUDA devices:', gap_solver.get_cuda_device_count())
"
```

## Installation for Development

```bash
# Install in development mode
pip install -e .

# Then import directly
python -c "import gap; print(gap.get_build_info())"
```