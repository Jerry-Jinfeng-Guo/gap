"""
Test script to verify auto-detection of bus ID indexing scheme.

This test validates that the GAP solver can automatically handle both:
- 0-based bus IDs (e.g., buses 0, 1, 2, ...)
- 1-based bus IDs (e.g., buses 1, 2, 3, ...)

The admittance matrix builder auto-detects the ID scheme and creates proper mappings.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import numpy as np

import gap_solver


def test_0_based_ids():
    """Test with 0-based bus IDs (like radial_3feeder)"""
    print("=" * 60)
    print("Test 1: 0-based Bus IDs (buses 0, 1)")
    print("=" * 60)

    # 2-bus system with 0-based IDs
    bus_data = np.array(
        [
            # id, u_rated, type, u_pu, u_angle, active_power, reactive_power
            [0, 230000, 2, 1.05, 0, 0, 0],  # Bus 0: Slack (type=2)
            [1, 230000, 0, 1.0, 0, -100e6, -50e6],  # Bus 1: Load (type=0=PQ)
        ]
    )

    branch_data = np.array(
        [
            # id, from, to, r1, x1, b1
            [1, 0, 1, 5.29, 52.9, 0],
        ]
    )

    result = gap_solver.solve_simple_power_flow(
        bus_data, branch_data, tolerance=1e-6, max_iterations=50
    )

    # Parse result: list of [real, imag, mag, angle], last row is metadata
    voltages = result[:-1]
    metadata = result[-1]
    converged = bool(metadata[0])
    iterations = int(metadata[1])
    final_mismatch = metadata[2]

    print(f"  Status: {'✅ CONVERGED' if converged else '❌ FAILED'}")
    print(f"  Iterations: {iterations}")
    print(f"  Mismatch: {final_mismatch:.2e}")
    print(f"  Bus 0: {voltages[0][2]:.4f} pu")
    print(f"  Bus 1: {voltages[1][2]:.4f} pu")
    print()

    assert converged, "Failed to converge with 0-based IDs"
    print("✅ Test passed: 0-based IDs handled correctly\n")


def test_1_based_ids():
    """Test with 1-based bus IDs (like simple_2bus)"""
    print("=" * 60)
    print("Test 2: 1-based Bus IDs (buses 1, 2)")
    print("=" * 60)

    # 2-bus system with 1-based IDs
    bus_data = np.array(
        [
            # id, u_rated, type, u_pu, u_angle, active_power, reactive_power
            [1, 230000, 2, 1.05, 0, 0, 0],  # Bus 1: Slack (type=2)
            [2, 230000, 0, 1.0, 0, -100e6, -50e6],  # Bus 2: Load (type=0=PQ)
        ]
    )

    branch_data = np.array(
        [
            # id, from, to, r1, x1, b1
            [1, 1, 2, 5.29, 52.9, 0],
        ]
    )

    result = gap_solver.solve_simple_power_flow(
        bus_data, branch_data, tolerance=1e-6, max_iterations=50
    )

    # Parse result: list of [real, imag, mag, angle], last row is metadata
    voltages = result[:-1]
    metadata = result[-1]
    converged = bool(metadata[0])
    iterations = int(metadata[1])
    final_mismatch = metadata[2]

    print(f"  Status: {'✅ CONVERGED' if converged else '❌ FAILED'}")
    print(f"  Iterations: {iterations}")
    print(f"  Mismatch: {final_mismatch:.2e}")
    print(f"  Bus 1: {voltages[0][2]:.4f} pu")
    print(f"  Bus 2: {voltages[1][2]:.4f} pu")
    print()

    assert converged, "Failed to converge with 1-based IDs"
    print("✅ Test passed: 1-based IDs handled correctly\n")


def test_mixed_ids():
    """Test with non-sequential bus IDs"""
    print("=" * 60)
    print("Test 3: Non-sequential Bus IDs (buses 10, 20)")
    print("=" * 60)

    # 2-bus system with non-sequential IDs
    bus_data = np.array(
        [
            # id, u_rated, type, u_pu, u_angle, active_power, reactive_power
            [10, 230000, 2, 1.05, 0, 0, 0],  # Bus 10: Slack (type=2)
            [20, 230000, 0, 1.0, 0, -100e6, -50e6],  # Bus 20: Load (type=0=PQ)
        ]
    )

    branch_data = np.array(
        [
            # id, from, to, r1, x1, b1
            [1, 10, 20, 5.29, 52.9, 0],
        ]
    )

    result = gap_solver.solve_simple_power_flow(
        bus_data, branch_data, tolerance=1e-6, max_iterations=50
    )

    # Parse result: list of [real, imag, mag, angle], last row is metadata
    voltages = result[:-1]
    metadata = result[-1]
    converged = bool(metadata[0])
    iterations = int(metadata[1])
    final_mismatch = metadata[2]

    print(f"  Status: {'✅ CONVERGED' if converged else '❌ FAILED'}")
    print(f"  Iterations: {iterations}")
    print(f"  Mismatch: {final_mismatch:.2e}")
    print(f"  Bus 10: {voltages[0][2]:.4f} pu")
    print(f"  Bus 20: {voltages[1][2]:.4f} pu")
    print()

    assert converged, "Failed to converge with non-sequential IDs"
    print("✅ Test passed: Non-sequential IDs handled correctly\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GAP Bus ID Auto-Detection Test Suite")
    print("=" * 60)
    print()

    try:
        test_0_based_ids()
        test_1_based_ids()
        test_mixed_ids()

        print("=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        print("\nSummary:")
        print("  ✅ 0-based bus IDs work correctly")
        print("  ✅ 1-based bus IDs work correctly")
        print("  ✅ Non-sequential bus IDs work correctly")
        print("\nThe admittance matrix builder automatically detects")
        print("the bus ID scheme and creates proper ID-to-index mappings.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
