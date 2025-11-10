"""
GAP Python Bindings Package

This package provides Python bindings for the GAP (GPU-Accelerated Power Flow) calculator.
Supports both CPU and GPU (CUDA) backends for high-performance power system analysis.
"""

__version__ = "1.0.0"
__author__ = "GAP Development Team"
__email__ = "gap@powerflow.com"

# Import the compiled C++ extension
try:
    from .gap_solver import *

    _cpp_module_available = True
except ImportError as e:
    _cpp_module_available = False
    _import_error = str(e)

if not _cpp_module_available:
    raise ImportError(
        f"Failed to import GAP C++ extension module: {_import_error}\n"
        "This usually means:\n"
        "1. The module wasn't compiled (run: python setup.py build_ext)\n"
        "2. Missing dependencies (CUDA libraries if GPU support was enabled)\n"
        "3. Incompatible Python version or architecture"
    )

# Convenience imports and version info
from .gap_solver import (  # Core types; Data structures; Solvers (availability depends on build configuration); Factory functions
    ApplianceType,
    BackendType,
    BranchType,
    BusType,
    CPUNewtonRaphson,
    NetworkData,
    PowerFlowConfig,
    PowerFlowResult,
    create_solver,
    get_available_backends,
    is_cuda_available,
)

# Conditional CUDA imports
try:
    from .gap_solver import GPUNewtonRaphson

    _cuda_available = True
except ImportError:
    _cuda_available = False


# Module information
def get_build_info():
    """Get information about how GAP was built."""
    info = {
        "version": __version__,
        "cuda_available": _cuda_available,
        "backends": get_available_backends(),
        "python_bindings": True,
    }
    return info


def print_build_info():
    """Print GAP build information."""
    info = get_build_info()
    print("GAP Power Flow Calculator - Python Bindings")
    print(f"Version: {info['version']}")
    print(f"CUDA Support: {'Available' if info['cuda_available'] else 'Not Available'}")
    print(f"Available Backends: {', '.join([str(b) for b in info['backends']])}")


# For backwards compatibility and convenience
__all__ = [
    # Version info
    "__version__",
    "get_build_info",
    "print_build_info",
    # Core enums
    "BackendType",
    "BusType",
    "ApplianceType",
    "BranchType",
    # Data structures
    "PowerFlowConfig",
    "PowerFlowResult",
    "NetworkData",
    # Solver classes
    "CPUNewtonRaphson",
    # Factory functions
    "create_solver",
    "get_available_backends",
    "is_cuda_available",
]

# Conditionally add CUDA exports
if _cuda_available:
    __all__.append("GPUNewtonRaphson")
