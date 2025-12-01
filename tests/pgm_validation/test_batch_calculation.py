#!/usr/bin/env python3
"""
Test GAP batch calculation with generated test data.

This script:
1. Loads batch scenario data from update.json
2. Runs GAP batch calculation (via Python bindings)
3. Compares against PGM batch reference in batch_output.json
4. Reports accuracy and performance
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from json_io.gap_json_parser import PGMJSONParser
from reference_solutions.pgm_reference import PGM_AVAILABLE

# Try to import GAP solver
GAP_AVAILABLE = False
try:
    import gap_solver

    GAP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  GAP solver not available: {e}")


def load_test_case(test_dir: Path):
    """Load test case data."""
    input_file = test_dir / "input.json"
    update_file = test_dir / "update.json"
    batch_output_file = test_dir / "batch_output.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not update_file.exists():
        raise FileNotFoundError(f"Update file not found: {update_file}")
    if not batch_output_file.exists():
        raise FileNotFoundError(f"Batch output file not found: {batch_output_file}")

    with open(input_file) as f:
        input_data = json.load(f)

    with open(update_file) as f:
        update_data = json.load(f)

    with open(batch_output_file) as f:
        batch_reference = json.load(f)

    return input_data, update_data, batch_reference


def run_gap_batch_sequential(parser, gap_network, update_data, config):
    """
    Run GAP solver sequentially for each scenario (without batch API).

    This is the fallback when batch API is not available in Python bindings.
    """
    if not GAP_AVAILABLE:
        raise RuntimeError("GAP solver not available")

    n_scenarios = update_data["n_scenarios"]
    load_ids = update_data["sym_load"]["id"]
    p_matrix = update_data["sym_load"]["p_specified"]
    q_matrix = update_data["sym_load"]["q_specified"]

    results = []
    total_time = 0.0

    print(f"  Running {n_scenarios} scenarios sequentially...")

    for scenario_idx in range(n_scenarios):
        # Prepare bus and branch data (simplified - would need full conversion)
        # For now, just use the simple API

        # Build bus data with updated loads
        bus_data = []
        for i in range(gap_network.n_node):
            bus_id = int(gap_network.node_ids[i])
            u_rated = gap_network.u_rated[i]

            # Determine bus type
            bus_type = 0  # PQ
            if i in gap_network.source_node:
                bus_type = 2  # SLACK

            voltage_pu = 1.0
            angle_rad = 0.0
            if i in gap_network.source_node:
                source_idx = np.nonzero(gap_network.source_node == i)[0][0]
                voltage_pu = gap_network.u_ref_pu[source_idx]

            # Calculate net power injection with scenario loads
            p_load = 0.0
            q_load = 0.0
            if i in gap_network.load_node:
                load_indices = np.nonzero(gap_network.load_node == i)[0]
                for load_idx in load_indices:
                    if gap_network.load_status[load_idx]:
                        # Find this load in update data
                        load_id = gap_network.load_ids[load_idx]
                        if load_id in load_ids:
                            update_idx = load_ids.index(load_id)
                            p_load -= p_matrix[scenario_idx][update_idx]
                            q_load -= q_matrix[scenario_idx][update_idx]

            bus_data.append(
                [bus_id, u_rated, bus_type, voltage_pu, angle_rad, p_load, q_load]
            )

        # Build branch data
        branch_data = []
        base_impedance = (gap_network.u_rated[0] ** 2) / gap_network.base_power
        for i in range(gap_network.n_branch):
            if gap_network.branch_status[i]:
                branch_id = int(gap_network.branch_ids[i])
                from_bus = int(gap_network.from_node[i])
                to_bus = int(gap_network.to_node[i])
                r_ohms = float(gap_network.r_pu[i]) * base_impedance
                x_ohms = float(gap_network.x_pu[i]) * base_impedance
                b_siemens = float(gap_network.b_pu[i]) / base_impedance
                branch_data.append(
                    [branch_id, from_bus, to_bus, r_ohms, x_ohms, b_siemens]
                )

        # Run single power flow
        start = time.time()
        result = gap_solver.solve_simple_power_flow(
            bus_data,
            branch_data,
            tolerance=config["tolerance"],
            max_iterations=config["max_iterations"],
            verbose=False,
        )
        elapsed = time.time() - start
        total_time += elapsed

        # Extract voltages
        n_bus = len(result) - 1
        voltages = []
        for i in range(n_bus):
            _, _, magnitude, angle_rad = result[i]
            voltages.append((magnitude, angle_rad))

        # Extract metadata
        metadata = result[-1]
        converged = bool(metadata[0])
        iterations = int(metadata[1])

        results.append(
            {
                "scenario_id": scenario_idx,
                "converged": converged,
                "iterations": iterations,
                "voltages": voltages,
                "time": elapsed,
            }
        )

    return results, total_time


def compare_with_reference(gap_results, batch_reference):
    """Compare GAP results with PGM batch reference."""
    n_scenarios = len(gap_results)

    voltage_errors = []
    angle_errors = []

    print(f"\n  Comparing {n_scenarios} scenarios:")

    for i, gap_result in enumerate(gap_results):
        if i >= len(batch_reference["scenarios"]):
            print(f"    Scenario {i}: ⚠️  No reference data")
            continue

        ref_scenario = batch_reference["scenarios"][i]
        ref_u_pu = np.array(ref_scenario["node"]["u_pu"])
        ref_u_angle = np.array(ref_scenario["node"]["u_angle"])

        gap_voltages = gap_result["voltages"]
        gap_u_pu = np.array([v[0] for v in gap_voltages])
        gap_u_angle = np.array([v[1] for v in gap_voltages])

        # Calculate errors
        v_err = np.max(np.abs(gap_u_pu - ref_u_pu[: len(gap_u_pu)]))
        a_err = np.max(
            np.abs(
                np.degrees(gap_u_angle) - np.degrees(ref_u_angle[: len(gap_u_angle)])
            )
        )

        voltage_errors.append(v_err)
        angle_errors.append(a_err)

        status = "✅" if v_err < 1e-4 else "⚠️"
        print(f"    Scenario {i}: {status} V_err={v_err:.2e} pu, A_err={a_err:.2e}°")

    return {
        "max_voltage_error": max(voltage_errors) if voltage_errors else 0,
        "avg_voltage_error": np.mean(voltage_errors) if voltage_errors else 0,
        "max_angle_error": max(angle_errors) if angle_errors else 0,
        "avg_angle_error": np.mean(angle_errors) if angle_errors else 0,
    }


def test_batch_calculation(test_case_name: str):
    """Test batch calculation for a specific test case."""
    print(f"\n{'='*70}")
    print(f"Testing Batch Calculation: {test_case_name}")
    print(f"{'='*70}")

    test_dir = Path(__file__).parent / "test_data" / test_case_name

    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False

    # Load test data
    print("1. Loading test data...")
    try:
        input_data, update_data, batch_reference = load_test_case(test_dir)
        print(f"   ✓ Network: {len(input_data.get('data', input_data)['node'])} nodes")
        print(f"   ✓ Scenarios: {update_data['n_scenarios']}")
        print(f"   ✓ PGM reference: {batch_reference['n_scenarios']} scenarios")
        print(
            f"   ✓ PGM batch time: {batch_reference['total_calculation_time_s']*1000:.2f} ms"
        )
    except Exception as e:
        print(f"   ❌ Failed to load test data: {e}")
        return False

    # Parse network
    print("\n2. Parsing network...")
    try:
        parser = PGMJSONParser()
        input_file = test_dir / "input.json"
        gap_network = parser.parse_network(input_file)
        print(
            f"   ✓ Parsed: {gap_network.n_node} nodes, {gap_network.n_branch} branches"
        )
    except Exception as e:
        print(f"   ❌ Failed to parse network: {e}")
        return False

    # Check if GAP is available
    if not GAP_AVAILABLE:
        print("\n3. Running GAP batch calculation...")
        print("   ❌ GAP solver not available - cannot run batch calculation")
        print("\n💡 To enable GAP solver:")
        print("   1. Build with Python bindings: cmake -DGAP_BUILD_PYTHON_BINDINGS=ON")
        print("   2. Make sure build/lib is in PYTHONPATH")
        return False

    # Run GAP batch calculation (sequential for now)
    print("\n3. Running GAP batch calculation (sequential)...")
    config = {"tolerance": 1e-8, "max_iterations": 20}

    try:
        gap_results, total_time = run_gap_batch_sequential(
            parser, gap_network, update_data, config
        )
        print(f"   ✓ Completed: {len(gap_results)} scenarios")
        print(f"   ✓ Total time: {total_time*1000:.2f} ms")
        print(f"   ✓ Avg time: {total_time*1000/len(gap_results):.2f} ms/scenario")
    except Exception as e:
        print(f"   ❌ Failed to run GAP calculation: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Compare with reference
    print("\n4. Comparing with PGM reference...")
    try:
        comparison = compare_with_reference(gap_results, batch_reference)
        print(f"\n   Summary:")
        print(f"   Max voltage error: {comparison['max_voltage_error']:.2e} pu")
        print(f"   Avg voltage error: {comparison['avg_voltage_error']:.2e} pu")
        print(f"   Max angle error: {comparison['max_angle_error']:.2e}°")

        # Performance comparison
        pgm_time = batch_reference["total_calculation_time_s"] * 1000
        gap_time = total_time * 1000
        speedup = pgm_time / gap_time if gap_time > 0 else 0

        print(f"\n   Performance:")
        print(f"   PGM batch time: {pgm_time:.2f} ms")
        print(f"   GAP sequential time: {gap_time:.2f} ms")
        print(
            f"   Relative speed: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
        )

        # Pass/fail criteria
        passed = comparison["max_voltage_error"] < 1e-4
        if passed:
            print(f"\n   ✅ VALIDATION PASSED")
        else:
            print(f"\n   ⚠️  VALIDATION WARNING: Errors exceed tolerance")

        return passed

    except Exception as e:
        print(f"   ❌ Failed to compare results: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run batch calculation tests."""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║   GAP Batch Calculation Test Suite                            ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # Test cases
    test_cases = [
        "radial_1feeder_2nodepf",
        "radial_1feeder_4nodepf",
        "radial_2feeder_4nodepf",
        "radial_3feeder_4nodepf",
        "radial_10feeder_10nodepf",
    ]

    results = []
    for test_case in test_cases:
        passed = test_batch_calculation(test_case)
        results.append((test_case, passed))

    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")

    for test_case, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}  {test_case}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} passed")

    if total_passed == len(results):
        print("\n🎉 All batch calculation tests passed!")
    else:
        print("\n⚠️  Some tests failed - check implementation")

    return total_passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
