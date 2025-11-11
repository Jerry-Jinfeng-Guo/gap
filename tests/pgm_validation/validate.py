#!/usr/bin/env python3
"""
Helper script to set up PYTHONPATH and run validation.
This makes it easier to run validation without manually setting environment variables.

Usage:
    ./validate.sh              # Run all tests
    ./validate.sh radial_3feeder  # Run specific test
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    # Get the GAP build directory
    gap_root = Path(__file__).parent.parent.parent
    build_lib = gap_root / "build" / "lib"

    if not build_lib.exists():
        print("❌ GAP solver not found. Please build the project first:")
        print(f"   Expected location: {build_lib}")
        print("\nBuild instructions:")
        print("   cd <gap_root>")
        print("   mkdir build && cd build")
        print("   cmake ..")
        print("   make gap_solver")
        sys.exit(1)

    # Set up environment
    env = os.environ.copy()
    pythonpath = str(build_lib)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath

    # Build command
    cmd = [sys.executable, "run_validation.py"]

    # Pass through arguments
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("-"):
            # It's a flag
            cmd.extend(sys.argv[1:])
        else:
            # It's a test case name
            cmd.extend(["--test-case", sys.argv[1]])
            if len(sys.argv) > 2:
                cmd.extend(sys.argv[2:])

    # Run validation
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
