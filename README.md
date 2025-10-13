# GAP: GPU-Accelerated Power Flow Calculator

A high-performance power flow calculation tool with configurable CPU and CUDA GPU backends.

## Overview

GAP (GPU-Accelerated Power flow) is a modern C++20 power flow solver designed for power system analysis. It features a modular architecture with pluggable backends, allowing users to choose between CPU and GPU execution depending on their hardware and performance requirements.

## Features

- **Modern C++20 codebase** with clean interfaces and modular design
- **Configurable backends**: CPU and CUDA GPU implementations
- **Newton-Raphson power flow solver** with sparse linear algebra
- **JSON-based input/output** for power system data
- **Comprehensive test suite** with unit and validation tests
- **CMake build system** with CUDA support
- **Cross-platform compatibility** (Linux, Windows, macOS)

## Architecture

### Core Components

- **Main Executable**: Command-line interface with backend selection
- **IO Module**: JSON-based data input/output (CPU-based)
- **Admittance Matrix Preparation**: CPU and GPU backends
- **LU Solver**: CPU and GPU sparse linear solvers  
- **Newton-Raphson Power Flow Solver**: CPU and GPU implementations
- **Test Framework**: Unit tests and IEEE validation cases

### Backend Types

- **CPU Backend**: Uses standard C++ with optimized sparse linear algebra
- **GPU Backend**: CUDA-accelerated with cuBLAS, cuSPARSE, and cuSOLVER

## Requirements

### Minimum Requirements
- CMake 3.18+
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- Linux, Windows, or macOS

### GPU Requirements (Optional)
- NVIDIA GPU with Compute Capability 6.0+
- CUDA Toolkit 11.0+
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

# GPU-enabled build (when CUDA available) 
./build.sh --cuda ON -c
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
# CPU backend (from project root)
./build/bin/gap_main -i data/sample/simple_3bus.json -o results.json

# GPU backend (if available)
./build/bin/gap_main -i data/sample/simple_3bus.json -o results.json -b gpu

# With verbose output
./build/bin/gap_main -i data/sample/simple_3bus.json -o results.json -v

# Custom solver settings
./build/bin/gap_main -i data/sample/simple_3bus.json -o results.json -t 1e-8 -m 100
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

Power system data should be provided in JSON format:

```json
{
  "base_mva": 100.0,
  "buses": [
    {
      "id": 1,
      "type": 2,
      "voltage_magnitude": 1.05,
      "voltage_angle": 0.0,
      "active_power": 0.0,
      "reactive_power": 0.0
    }
  ],
  "branches": [
    {
      "from_bus": 1,
      "to_bus": 2,
      "impedance": {"real": 0.01, "imag": 0.1},
      "susceptance": 0.02,
      "status": true
    }
  ]
}
```

### Bus Types
- `0`: PQ bus (load bus)
- `1`: PV bus (generator bus)
- `2`: Slack bus (reference bus)

## Testing

### Run Unit Tests
```bash
# Individual test suites (recommended)
./build/bin/gap_unit_tests
./build/bin/gap_validation_tests

# Or from build directory
cd build
./bin/gap_unit_tests
./bin/gap_validation_tests

# CTest integration (may show failures from stub implementations)
make test  # Note: Some tests may fail initially due to stub implementations
```

### Test Coverage
- IO module functionality
- Admittance matrix construction
- LU solver correctness  
- Power flow convergence
- IEEE test cases validation
- Backend comparison tests

**Note**: Some tests may initially fail or show non-convergence due to stub implementations. This is expected behavior until the full algorithms are implemented.

## Development

### Project Structure
```
gap/
├── CMakeLists.txt           # Root build configuration
├── README.md               # This file
├── src/                    # Source code
│   ├── main/              # Main executable
│   ├── core/              # Core interfaces and factory
│   ├── io/                # Input/output module
│   ├── admittance/        # Admittance matrix backends
│   │   ├── cpu/          # CPU implementation
│   │   └── gpu/          # GPU implementation
│   └── solver/            # Solver backends
│       ├── lu/           # LU solvers
│       │   ├── cpu/     # CPU implementation
│       │   └── gpu/     # GPU implementation
│       └── powerflow/    # Power flow solvers
│           ├── cpu/     # CPU implementation
│           └── gpu/     # GPU implementation
├── include/gap/           # Header files
│   ├── core/             # Core interfaces
│   ├── io/               # IO interfaces
│   ├── admittance/       # Admittance interfaces
│   └── solver/           # Solver interfaces
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── validation/       # Validation tests
├── data/                  # Sample data files
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