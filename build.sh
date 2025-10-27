# Build script for GAP Power Flow Calculator
#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_TYPE="Release"
BUILD_DIR="build"
ENABLE_CUDA="AUTO"
CUDA_ARCH=""
VERBOSE=false
CLEAN=false
RUN_TESTS=false
INSTALL=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for GAP Power Flow Calculator

OPTIONS:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -d, --debug             Build in debug mode (default: Release)
    -v, --verbose           Enable verbose build output
    -t, --test              Run tests after building
    -i, --install           Install after building
    --cuda OPTION           CUDA support: ON, OFF, AUTO (default: AUTO)
    --cuda-arch ARCH        CUDA architecture (e.g., 70, 75, 80)
    --build-dir DIR         Build directory (default: build)

EXAMPLES:
    $0                      # Basic release build with auto CUDA detection
    $0 -c -t               # Clean build and run tests
    $0 -d --cuda OFF       # Debug build without CUDA
    $0 --cuda-arch 75      # Build for specific CUDA architecture

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        -i|--install)
            INSTALL=true
            shift
            ;;
        --cuda)
            ENABLE_CUDA="$2"
            shift 2
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate CUDA option
if [[ "$ENABLE_CUDA" != "ON" && "$ENABLE_CUDA" != "OFF" && "$ENABLE_CUDA" != "AUTO" ]]; then
    print_error "Invalid CUDA option: $ENABLE_CUDA"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]]; then
    print_error "CMakeLists.txt not found. Please run this script from the project root."
    exit 1
fi

# Clean build directory if requested
if [[ "$CLEAN" == true ]]; then
    print_status "Cleaning build directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
if [[ ! -d "$BUILD_DIR" ]]; then
    print_status "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# Detect CUDA availability if AUTO mode
if [[ "$ENABLE_CUDA" == "AUTO" ]]; then
    print_status "Auto-detecting CUDA availability..."
    if command -v nvcc &> /dev/null; then
        print_success "CUDA compiler found: $(nvcc --version | grep "release")"
        ENABLE_CUDA="ON"
        
        # Auto-detect CUDA architecture if not specified
        if [[ -z "$CUDA_ARCH" ]]; then
            if command -v nvidia-smi &> /dev/null; then
                print_status "Auto-detecting CUDA architecture..."
                # This is a simplified detection; in practice, you'd want more sophisticated detection
                CUDA_ARCH="70"  # Default to compute capability 7.0
                print_status "Using CUDA architecture: $CUDA_ARCH"
            fi
        fi
    else
        print_warning "CUDA compiler not found. Building CPU-only version."
        ENABLE_CUDA="OFF"
    fi
fi

# Prepare CMake arguments
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ "$ENABLE_CUDA" == "ON" ]]; then
    # Check GCC version and use GCC 13 if current version is too new for CUDA
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$GCC_VERSION" -gt 13 ]]; then
        if command -v gcc-13 &> /dev/null; then
            print_warning "GCC version $GCC_VERSION is too new for CUDA. Using gcc-13 for CUDA compilation."
            CMAKE_ARGS+=(-DCMAKE_C_COMPILER=gcc-13)
            CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER=g++-13)
            CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER=g++-13)
        else
            print_warning "GCC version $GCC_VERSION detected but gcc-13 not found. Adding -allow-unsupported-compiler flag for CUDA."
            CMAKE_ARGS+=(-DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler")
        fi
    fi
    
    # Use project's CUDA configuration
    CMAKE_ARGS+=(-DGAP_ENABLE_CUDA=ON)
    if [[ -n "$CUDA_ARCH" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    fi
elif [[ "$ENABLE_CUDA" == "OFF" ]]; then
    CMAKE_ARGS+=(-DGAP_ENABLE_CUDA=OFF)
    CMAKE_ARGS+=(-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE)
fi

# Configure with CMake
print_status "Configuring project with CMake..."
print_status "Build type: $BUILD_TYPE"
print_status "CUDA support: $ENABLE_CUDA"
if [[ -n "$CUDA_ARCH" ]]; then
    print_status "CUDA architecture: $CUDA_ARCH"
fi

cmake .. "${CMAKE_ARGS[@]}"

# Build the project
print_status "Building project..."

MAKE_ARGS=()
if [[ "$VERBOSE" == true ]]; then
    MAKE_ARGS+=(VERBOSE=1)
fi

# Determine number of cores for parallel build
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MAKE_ARGS+=(-j"$NPROC")

make "${MAKE_ARGS[@]}"

print_success "Build completed successfully!"

# List built targets
print_status "Built targets:"
ls -la bin/ 2>/dev/null || echo "  No executables found in bin/"
ls -la lib/ 2>/dev/null || echo "  No libraries found in lib/"

# Run tests if requested
if [[ "$RUN_TESTS" == true ]]; then
    print_status "Running tests..."
    
    if ctest --output-on-failure; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed!"
        exit 1
    fi
fi

# Install if requested
if [[ "$INSTALL" == true ]]; then
    print_status "Installing..."
    make install
    print_success "Installation completed!"
fi

# Show usage example
print_success "Build completed successfully!"
echo
print_status "Usage examples:"
echo "  # CPU backend:"
echo "  ./bin/gap_main -i ../data/sample/simple_3bus.json -o results.json"
echo
if [[ "$ENABLE_CUDA" == "ON" ]]; then
    echo "  # GPU backend:"
    echo "  ./bin/gap_main -i ../data/sample/simple_3bus.json -o results.json -b gpu"
    echo
fi
echo "  # Run tests:"
echo "  ./bin/gap_unit_tests"
echo "  ./bin/gap_validation_tests"
if [[ "$ENABLE_CUDA" == "ON" ]]; then
    echo "  ./bin/gap_gpu_tests"
fi
echo