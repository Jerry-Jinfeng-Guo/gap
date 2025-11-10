#!/usr/bin/env python3
"""
GAP Solver Module Check

Simple utility to check if GAP solver Python bindings are available
and show import status with helpful diagnostics.
"""

import os
from pathlib import Path
import sys


def check_gap_module():
    """Check GAP module availability and show import status."""

    print("GAP Solver Module Check")
    print("=" * 40)

    # Add the validation script path to test the import logic
    repo_root = Path(__file__).parent.parent  # Go up from scripts/ to repo root
    validation_path = repo_root / "tests" / "pgm_validation"

    if validation_path.exists():
        sys.path.append(str(validation_path))

        try:
            from validation_demo import GAP_AVAILABLE, GAP_IMPORT_ERROR, gap_solver

            print(f"Status: {'✅ AVAILABLE' if GAP_AVAILABLE else '❌ NOT AVAILABLE'}")

            if GAP_AVAILABLE:
                print(f"Module: {gap_solver}")
                try:
                    backends = gap_solver.get_available_backends()
                    print(f"Backends: {backends}")
                    cuda_available = (
                        gap_solver.is_cuda_available()
                        if hasattr(gap_solver, "is_cuda_available")
                        else False
                    )
                    print(f"CUDA Support: {'✅ Yes' if cuda_available else '❌ No'}")
                except Exception as e:
                    print(f"Backend check failed: {e}")
            else:
                print(f"Error: {GAP_IMPORT_ERROR}")

                # Show available build directories
                build_dirs = [
                    repo_root / "build_cuda" / "lib",
                    repo_root / "build" / "lib",
                    repo_root / "build_release" / "lib",
                    repo_root / "build_debug" / "lib",
                ]

                print("\nBuild directory status:")
                for build_dir in build_dirs:
                    status = "✅ exists" if build_dir.exists() else "❌ missing"
                    print(f"  {build_dir.name}: {status}")
                    if build_dir.exists():
                        gap_modules = list(build_dir.glob("gap_solver*.so"))
                        if gap_modules:
                            print(f"    Python module: ✅ {gap_modules[0].name}")
                        else:
                            print(f"    Python module: ❌ not found")

                print("\n💡 Solutions:")
                print("  1. Build with Python bindings: ./build.sh --cuda ON")
                print(
                    "  2. Set custom path: export GAP_MODULE_PATH=/path/to/gap/module"
                )
                print("  3. Install as package: pip install -e .")

        except ImportError as e:
            print(f"❌ Failed to import validation_demo: {e}")
    else:
        print(f"❌ Validation script not found at: {validation_path}")

    print("\nEnvironment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  GAP_MODULE_PATH: {os.environ.get('GAP_MODULE_PATH', 'not set')}")


if __name__ == "__main__":
    check_gap_module()
