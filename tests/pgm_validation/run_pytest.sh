#!/usr/bin/env bash
#
# Convenience wrapper to run PGM validation tests with pytest.
# Automatically sets up PYTHONPATH and runs pytest with appropriate options.
#
# Usage:
#     ./run_pytest.sh                                    # Run all tests
#     ./run_pytest.sh -v                                 # Verbose output
#     ./run_pytest.sh -k radial_3feeder                  # Run specific test
#     ./run_pytest.sh --markers                          # List available markers
#     ./run_pytest.sh -m "not slow"                      # Skip slow tests
#

# Get the script directory and GAP root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_LIB="$GAP_ROOT/build/lib"

# Check if build exists
if [ ! -d "$BUILD_LIB" ]; then
    echo "❌ GAP solver not found. Please build the project first:"
    echo "   Expected location: $BUILD_LIB"
    echo ""
    echo "Build instructions:"
    echo "   cd $GAP_ROOT"
    echo "   mkdir -p build && cd build"
    echo "   cmake .."
    echo "   make gap_solver"
    exit 1
fi

# Set up PYTHONPATH
export PYTHONPATH="$BUILD_LIB:$PYTHONPATH"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Please install it:"
    echo "   pip install pytest"
    echo ""
    echo "Or install dev dependencies:"
    echo "   pip install -e '.[dev]'"
    exit 1
fi

# Run pytest with all arguments passed through
echo "🧪 Running PGM validation tests with pytest..."
echo "   PYTHONPATH=$PYTHONPATH"
echo "   Test directory: $SCRIPT_DIR"
echo ""

cd "$SCRIPT_DIR"
pytest test_pgm_validation.py "$@"
