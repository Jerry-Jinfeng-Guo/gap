#!/bin/bash
# build.sh - Unified build script for GAP with Python bindings
# Supports both CPU and CUDA builds with command-line options

set -e

# Default configuration
ENABLE_CUDA="OFF"
BUILD_TYPE="Release"
CLEAN_BUILD=false
PYTHON_BINDINGS="ON"
VERBOSE=false
PARALLEL_JOBS=$(nproc)

# Get the directory of this script and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    # Running from scripts/ directory
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    # Running from project root via symlink
    PROJECT_ROOT="$(pwd)"
fi

# Function to print usage
print_usage() {
    cat << EOF
GAP Build Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --cuda ON|OFF           Enable/disable CUDA support (default: OFF)
    --build-type TYPE       Build type: Debug, Release, RelWithDebInfo (default: Release)
    --python ON|OFF         Enable/disable Python bindings (default: ON)
    --clean                 Clean build directory before building
    --verbose               Enable verbose build output
    --jobs N                Number of parallel build jobs (default: $(nproc))
    -h, --help              Show this help message

EXAMPLES:
    $0                      # CPU-only build with Python bindings
    $0 --cuda ON            # CUDA-enabled build
    $0 --cuda ON --clean    # Clean CUDA build
    $0 --python OFF         # Build without Python bindings
    $0 --build-type Debug   # Debug build

REQUIREMENTS:
    - Python 3.8+ with virtual environment at .venv/
    - For CUDA builds: CUDA Toolkit 11.0+
    - CMake 3.18+
    - C++20 compatible compiler
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            if [[ "$2" == "ON" || "$2" == "on" || "$2" == "1" || "$2" == "true" ]]; then
                ENABLE_CUDA="ON"
            elif [[ "$2" == "OFF" || "$2" == "off" || "$2" == "0" || "$2" == "false" ]]; then
                ENABLE_CUDA="OFF"
            else
                echo "Error: Invalid CUDA option '$2'. Use ON or OFF."
                exit 1
            fi
            shift 2
            ;;
        --build-type)
            if [[ "$2" =~ ^(Debug|Release|RelWithDebInfo|MinSizeRel)$ ]]; then
                BUILD_TYPE="$2"
            else
                echo "Error: Invalid build type '$2'. Use Debug, Release, RelWithDebInfo, or MinSizeRel."
                exit 1
            fi
            shift 2
            ;;
        --python)
            if [[ "$2" == "ON" || "$2" == "on" || "$2" == "1" || "$2" == "true" ]]; then
                PYTHON_BINDINGS="ON"
            elif [[ "$2" == "OFF" || "$2" == "off" || "$2" == "0" || "$2" == "false" ]]; then
                PYTHON_BINDINGS="OFF"
            else
                echo "Error: Invalid Python bindings option '$2'. Use ON or OFF."
                exit 1
            fi
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --jobs)
            if [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
                PARALLEL_JOBS="$2"
            else
                echo "Error: Invalid job count '$2'. Must be a positive integer."
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            print_usage
            exit 1
            ;;
    esac
done

# Print configuration header
echo "======================================"
echo "        GAP Unified Build Script      "
echo "======================================"
echo "Configuration:"
echo "  CUDA support: $ENABLE_CUDA"
echo "  Build type: $BUILD_TYPE"
echo "  Python bindings: $PYTHON_BINDINGS"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Clean build: $CLEAN_BUILD"
echo "  Verbose: $VERBOSE"
echo "======================================"

# Set build directory based on CUDA setting
if [ "$ENABLE_CUDA" = "ON" ]; then
    BUILD_DIR="$PROJECT_ROOT/build_cuda"
    echo "Building with CUDA support..."
else
    BUILD_DIR="$PROJECT_ROOT/build"
    echo "Building CPU-only version..."
fi

# Check for virtual environment if Python bindings are enabled
if [ "$PYTHON_BINDINGS" = "ON" ]; then
    VENV_PATH="$PROJECT_ROOT/.venv"
    PYTHON_EXE="$VENV_PATH/bin/python"

    if [ ! -f "$PYTHON_EXE" ]; then
        echo "Error: Virtual environment not found at $VENV_PATH"
        echo "Please create it first:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install 'pybind11[global]'"
        exit 1
    fi

    echo "Using Python: $PYTHON_EXE"
    echo "Python version: $($PYTHON_EXE --version)"

    # Check if pybind11 is installed
    if ! $PYTHON_EXE -c "import pybind11" >/dev/null 2>&1; then
        echo "Installing pybind11..."
        $PYTHON_EXE -m pip install "pybind11[global]"
    fi

    echo "pybind11 version: $($PYTHON_EXE -c "import pybind11; print(pybind11.version_info())" 2>/dev/null || echo "unknown")"
fi

# Check for CUDA if enabled
if [ "$ENABLE_CUDA" = "ON" ]; then
    if ! nvcc --version >/dev/null 2>&1; then
        echo "Error: CUDA not found. Please install CUDA toolkit."
        echo "To build without CUDA, use: $0 --cuda OFF"
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',' || echo "unknown")
    echo "CUDA version: $CUDA_VERSION"
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Prepare CMake arguments
CMAKE_ARGS=(
    "-DGAP_ENABLE_CUDA=$ENABLE_CUDA"
    "-DGAP_BUILD_PYTHON_BINDINGS=$PYTHON_BINDINGS"
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
)

# Add Python executable if bindings are enabled
if [ "$PYTHON_BINDINGS" = "ON" ]; then
    CMAKE_ARGS+=("-DPython3_EXECUTABLE=$PYTHON_EXE")
fi

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=ON")
fi

# Configure with CMake
echo ""
echo "Configuring with CMake..."
echo "Command: cmake ${CMAKE_ARGS[*]} .."

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo ""
echo "Building with $PARALLEL_JOBS parallel jobs..."

MAKE_ARGS=("-j$PARALLEL_JOBS")
if [ "$VERBOSE" = true ]; then
    MAKE_ARGS+=("VERBOSE=1")
fi

make "${MAKE_ARGS[@]}"

echo ""
echo "✓ Build completed successfully!"

# Print results
echo ""
echo "Build results:"
echo "  Executables: $BUILD_DIR/bin/"
echo "  Libraries: $BUILD_DIR/lib/"

if [ "$PYTHON_BINDINGS" = "ON" ]; then
    PYTHON_MODULE=$(find "$BUILD_DIR/lib" -name "gap_solver*.so" 2>/dev/null | head -n1)
    if [ -n "$PYTHON_MODULE" ]; then
        echo "  Python module: $(basename "$PYTHON_MODULE")"
        echo ""
        echo "To test the Python module:"
        echo "  cd $PROJECT_ROOT"
        echo "  $PYTHON_EXE -c \"import sys; sys.path.insert(0, '$(basename "$BUILD_DIR")/lib'); import gap_solver; print('GAP loaded:', gap_solver.get_available_backends())\""
    fi
fi

echo ""
echo "To run tests:"
echo "  cd $BUILD_DIR"
echo "  make test"