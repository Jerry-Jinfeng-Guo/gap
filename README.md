# GAP: GPU-Accelerated Power Flow Calculator

A high-performance power flow calculation tool with **Power Grid Model (PGM) compliance** and configurable CPU and CUDA GPU backends.

## Overview

GAP (GPU-Accelerated Power flow) is a modern C++20 power flow solver designed for professional power system analysis. Built on the **Power Grid Model standard**, it features a modular architecture with pluggable backends, comprehensive electrical parameter support, and production-ready capabilities for both research and industrial applications.

## Features

- **Power Grid Model (PGM) Compliance** - Full support for modern electrical standards
- **Modern C++20 codebase** with clean interfaces and modular design  
- **Configurable backends**: CPU and CUDA GPU implementations
- **Newton-Raphson power flow solver** with optimized sparse linear algebra
- **Comprehensive PGM JSON IO** - Native support for lines, transformers, and generic branches
- **Enhanced admittance matrix** with shunt appliance support (capacitors, reactors)
- **Robust test framework** - 28 passing tests including PGM validation and CSR format verification
- **Modern toolchain** - GCC 14 + CUDA 13 compatibility
- **Cross-platform compatibility** (Linux, Windows, macOS)

## Architecture

### Core Components

- **Main Executable**: Command-line interface with backend selection and PGM support
- **PGM JSON IO Module**: Power Grid Model compliant data parsing with scientific notation support
- **Enhanced Admittance Matrix**: CPU and GPU backends with shunt appliance integration
- **CSR Matrix Validation**: Comprehensive sparse matrix format verification
- **LU Solver**: CPU and GPU sparse linear solvers with robust factorization
- **Newton-Raphson Power Flow Solver**: CPU and GPU implementations with convergence optimization
- **Comprehensive Test Framework**: PGM validation, unit tests, and electrical parameter verification

### Power Grid Model Support

- **Lines**: Standard transmission lines with r1/x1/g1/b1 parameters
- **Transformers**: Two-winding transformers with tap control and phase shift
- **Generic Branches**: Flexible branch modeling with arbitrary parameters
- **Shunt Appliances**: Capacitors and reactors with proper admittance integration
- **Bus Types**: Automatic inference (slack, PV, PQ) from appliance configuration

### Backend Types

- **CPU Backend**: Optimized C++ with enhanced sparse linear algebra and CSR format validation
- **GPU Backend**: CUDA-accelerated with cuBLAS, cuSPARSE, and cuSOLVER integration

## Requirements

### Minimum Requirements
- CMake 3.18+
- C++20 compatible compiler (GCC 14+, Clang 12+, MSVC 2019+)
- Linux, Windows, or macOS

### Recommended Development Environment
- **GCC 14** - Modern C++20 features and optimizations
- **CUDA 13.0+** - Latest CUDA features and performance improvements
- System alternatives configuration for easy toolchain switching

### GPU Requirements (Optional)
- NVIDIA GPU with Compute Capability 6.0+
- **CUDA Toolkit 13.0+** (recommended for best performance)
- cuBLAS, cuSPARSE, cuSOLVER libraries

## Building

### Clone and Setup
```bash
git clone <repository-url>
cd gap
```

### Recommended Build Method (using build script)
```bash
# CPU-only build
./build.sh --cuda OFF -c

# GPU-enabled build (with CUDA 13 + GCC 14)
./build.sh --cuda ON -c

# Debug build with full validation
./build.sh --cuda ON -c -t Debug
```

### CMake Presets (VS Code Integration)
```bash
# Using CMake presets for consistent builds
cmake --preset default          # CPU-only configuration
cmake --preset debug            # Debug configuration  
cmake --preset cpu-only         # Explicit CPU-only build

# Build with preset
cmake --build --preset default
```

### Manual Build Method
```bash
mkdir build && cd build

# CPU-only build
cmake ..
make -j$(nproc)

# GPU-enabled build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70  # Adjust for your GPU
make -j$(nproc)
```

### Build Targets
- `gap_main` - Main executable
- `gap_*` - Individual library components
- `gap_unit_tests` - Unit test executable
- `gap_validation_tests` - Validation test executable

## Usage

### Basic Power Flow Calculation
```bash
# CPU backend with PGM-compliant data
./build/bin/gap_main -i data/pgm/network_1.json -o results.json

# GPU backend (if available)
./build/bin/gap_main -i data/pgm/network_1.json -o results.json -b gpu

# Test different PGM component types
./build/bin/gap_main -i data/pgm/network_2.json -o results.json -v  # Transformers
./build/bin/gap_main -i data/pgm/network_3.json -o results.json -v  # Generic branches

# Custom solver settings with enhanced precision
./build/bin/gap_main -i data/pgm/network_1.json -o results.json -t 1e-8 -m 100
```

### Command Line Options
```
-i, --input FILE      Input JSON file (required)
-o, --output FILE     Output JSON file (required)
-b, --backend TYPE    Backend: cpu, gpu (default: cpu)
-t, --tolerance VAL   Convergence tolerance (default: 1e-6)
-m, --max-iter NUM    Maximum iterations (default: 50)
-v, --verbose         Enable verbose output
--benchmark           Enable benchmarking
--flat-start          Use flat start initialization
-h, --help            Show help message
```

## Input Format

Power system data follows the **Power Grid Model (PGM)** standard in JSON format:

```json
{
  "version": "1.0",
  "type": "input", 
  "is_batch": false,
  "node": [
    {
      "id": 1,
      "u_rated": 400000.0
    },
    {
      "id": 2, 
      "u_rated": 400000.0
    }
  ],
  "line": [
    {
      "id": 3,
      "from_node": 1,
      "to_node": 2,
      "from_status": 1,
      "to_status": 1,
      "r1": 0.0206,
      "x1": 0.0079,
      "c1": 0.0,
      "tan1": 0.0,
      "i_n": 1000.0
    }
  ],
  "source": [
    {
      "id": 4,
      "node": 1,
      "status": 1,
      "u_ref": 1.0
    }
  ],
  "sym_load": [
    {
      "id": 5,
      "node": 2,
      "status": 1,
      "type": 0,
      "p_specified": 60000.0,
      "q_specified": 0.0
    }
  ]
}
```

### PGM Component Types
- **node**: Bus/node definitions with rated voltage (`u_rated`)
- **line**: Transmission lines with electrical parameters (r1, x1, c1, i_n)
- **transformer**: Two-winding transformers with tap control
- **link**: Generic branches with flexible parameters
- **source**: Voltage sources (slack buses)
- **sym_load**: Symmetric loads (PQ buses)
- **shunt**: Capacitors and reactors with g1/b1 parameters

### Electrical Parameters
- **r1, x1**: Positive-sequence resistance and reactance (Ω)
- **g1, b1**: Positive-sequence conductance and susceptance (S)
- **c1**: Positive-sequence capacitance (F)
- **i_n**: Rated current (A)
- **u_rated**: Rated voltage (V)
- **sn**: Rated power (VA)

## Testing

### Run Unit Tests
```bash
# Comprehensive unit test suite (28/28 passing)
./build/bin/gap_unit_tests

# Individual validation tests
./build/bin/gap_validation_tests

# Or from build directory
cd build
./bin/gap_unit_tests
./bin/gap_validation_tests
```

### Test Coverage ✅
- **PGM JSON IO** - Complete Power Grid Model parsing and validation
- **Admittance Matrix** - Enhanced construction with shunt appliance support  
- **CSR Format Validation** - Comprehensive sparse matrix structure verification
- **LU Solver** - Robust factorization and solution accuracy
- **Power Flow Convergence** - Newton-Raphson algorithm validation
- **Component Types** - Lines, transformers, generic branches, and appliances
- **Backend Functionality** - CPU and GPU implementation verification
- **Electrical Parameters** - Scientific notation support and parameter validation

### Test Results
- **Total Tests**: 28 ✅
- **Success Rate**: 100%
- **Coverage**: Full PGM compliance validation
- **Matrix Testing**: CSR format correctness across multiple topologies
- **Component Testing**: All PGM component types (lines, transformers, generic branches)

**Status**: All tests pass successfully with comprehensive validation of Power Grid Model compliance and enhanced functionality.

## Development

### Project Structure
```
gap/
├── CMakeLists.txt           # Root build configuration
├── CMakePresets.json        # Shared build presets
├── README.md               # This file  
├── src/                    # Source code
│   ├── main/              # Main executable
│   ├── core/              # Core interfaces and factory
│   ├── io/                # PGM JSON IO module
│   ├── admittance/        # Enhanced admittance matrix backends
│   │   ├── cpu/          # CPU implementation with shunt support
│   │   └── gpu/          # GPU implementation
│   └── solver/            # Solver backends
│       ├── lu/           # LU solvers with CSR validation
│       │   ├── cpu/     # CPU implementation
│       │   └── gpu/     # GPU implementation  
│       └── powerflow/    # Newton-Raphson power flow solvers
│           ├── cpu/     # CPU implementation
│           └── gpu/     # GPU implementation
├── include/gap/           # Header files with PGM types
│   ├── core/             # Core interfaces with BackendFactory
│   ├── io/               # PGM JSON IO interfaces
│   ├── admittance/       # Admittance interfaces
│   └── solver/           # Solver interfaces
├── tests/                 # Comprehensive test suite (28 tests)
│   ├── unit/             # Unit tests with PGM validation
│   └── validation/       # Validation tests
├── data/                  # PGM test data files
│   └── pgm/              # Power Grid Model JSON files
└── docs/                  # Documentation
```

### Adding New Backends

1. Implement the appropriate interface (`IAdmittanceMatrix`, `ILUSolver`, or `IPowerFlowSolver`)
2. Add factory method in `BackendFactory`
3. Update CMake configuration
4. Add corresponding tests

### Code Style
- Modern C++20 features encouraged
- RAII and smart pointers for memory management
- Clear interface segregation
- Comprehensive error handling

## Performance

### Benchmarking
Enable benchmarking with `--benchmark` flag:
```bash
./bin/gap_main -i large_system.json -o results.json --benchmark
```

### Expected Performance
- **CPU Backend**: Suitable for small to medium systems (< 10,000 buses)
- **GPU Backend**: Optimized for large systems (> 1,000 buses)
- **Memory Usage**: Scales with system size and sparsity

## Troubleshooting

### CUDA Issues
```bash
# Check CUDA availability
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check compute capability
deviceQuery  # From CUDA samples
```

### Build Issues
```bash
# Clean build
rm -rf build && mkdir build && cd build

# Verbose build output
make VERBOSE=1

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Runtime Issues
```bash
# Check library dependencies
ldd ./bin/gap_main

# Enable verbose output
./bin/gap_main -i network.json -o results.json -v

# Run with debugger
gdb ./bin/gap_main
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Follow the existing code style
5. Submit a pull request

## License

[Specify your license here]

## References

- IEEE Power Flow Test Cases
- CUDA Programming Guide
- Modern C++ Best Practices
- Sparse Linear Algebra Libraries
