# GAP Python Bindings

Python bindings for the GAP (GPU-Accelerated Power Flow) Calculator, providing high-performance power system analysis capabilities with both CPU and GPU (CUDA) backends.

## Features

- **Newton-Raphson Power Flow**: High-performance solver with CPU and GPU backends
- **Modern C++20**: Built on a robust, modern C++ foundation
- **PGM Compliance**: Compatible with Power Grid Model standards
- **Flexible Backends**: Runtime selection between CPU and GPU execution
- **Python Integration**: Seamless integration with Python scientific ecosystem

## Installation

### Prerequisites

- Python 3.8+
- CMake 3.18+
- C++20 compatible compiler (GCC 10+, Clang 10+, or MSVC 2019+)
- Optional: CUDA Toolkit 11.0+ for GPU support

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Jerry-Jinfeng-Guo/gap.git
cd gap

# Install Python bindings
pip install .

# Or for development (editable install)
pip install -e .
```

### Build Configuration

The build system automatically detects CUDA availability:

```bash
# Force CUDA support (fails if CUDA not found)
GAP_ENABLE_CUDA=ON pip install .

# Force CPU-only build
GAP_ENABLE_CUDA=OFF pip install .

# Control build parallelism
GAP_BUILD_JOBS=8 pip install .
```

## Quick Start

```python
import gap

# Check available backends
print(gap.get_build_info())

# Create a power flow solver
config = gap.PowerFlowConfig()
config.tolerance = 1e-6
config.max_iterations = 50

# CPU backend
cpu_solver = gap.create_solver(gap.BackendType.CPU)

# GPU backend (if available)
if gap.is_cuda_available():
    gpu_solver = gap.create_solver(gap.BackendType.GPU_CUDA)

# Load network data
network = gap.load_network_from_json("network.json")

# Solve power flow
result = cpu_solver.solve_power_flow(network, admittance_matrix, config)

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Bus voltages: {len(result.bus_voltages)}")
```

## Validation with Power Grid Model

GAP includes comprehensive validation against Power Grid Model:

```python
# Install validation dependencies
pip install gap-solver[validation]

# Run validation pipeline
from gap.tests.pgm_validation.validation_demo import ValidationPipeline

pipeline = ValidationPipeline("validation_workspace")
results = pipeline.run_single_test_demo(num_buses=10)
```

## API Reference

### Core Types

- `BackendType`: CPU, GPU_CUDA
- `PowerFlowConfig`: Solver configuration
- `PowerFlowResult`: Solution results  
- `NetworkData`: Power system network definition

### Solver Interface

- `create_solver(backend)`: Create solver for specified backend
- `IPowerFlowSolver.solve_power_flow()`: Main solution method
- `load_network_from_json()`: Load network from JSON file

### CUDA Support

When CUDA is available:

- `gap.is_cuda_available()`: Check CUDA availability
- `gap.get_cuda_device_count()`: Number of CUDA devices
- `gap.get_cuda_device_properties()`: Device information

## Performance

GAP is optimized for performance:

- **CPU**: Vectorized operations with modern sparse linear algebra
- **GPU**: CUDA acceleration for large networks (1000+ buses)
- **Memory**: Efficient data structures with minimal Python overhead

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black .
flake8 .
```

## License

MPL-2 License - see LICENSE file for details.