#!/usr/bin/env python3
"""
Quick example of using the validation framework.

Run this to test a single case:
    python example_validation.py
"""

from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set PYTHONPATH for GAP solver
sys.path.insert(0, str(Path.home() / "rcdt/repos/gap/build/lib"))

from run_validation import ValidationRunner


def main():
    # Create validator
    runner = ValidationRunner(
        test_data_dir=Path(__file__).parent / "test_data",
        base_power=10e6,  # 10 MVA for distribution systems
        verbose=True,
    )

    # Run validation for radial_3feeder
    print("\n" + "=" * 80)
    print("Example: Validating radial_3feeder test case")
    print("=" * 80)

    runner.run_all(specific_test="radial_3feeder")
    runner.print_summary()


if __name__ == "__main__":
    main()
