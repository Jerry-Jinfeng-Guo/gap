#!/usr/bin/env python3
"""
Test script to verify the batch power flow API works correctly.
Uses the newly added Python bindings for batch calculation.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np

# Import GAP solver
import gap_solver


def test_batch_api():
    """Test the batch calculation API with a simple test case."""
    print("=" * 80)
    print("Testing GAP Batch Calculation API")
    print("=" * 80)

    # Load test data
    test_case = "radial_10feeder_10nodepf"
    data_dir = Path(__file__).parent / "test_data" / test_case

    assert (
        data_dir.exists()
    ), f"Test data not found at {data_dir}. Please run generate_test_data.py first"

    print(f"\nTest case: {test_case}")

    # Load network topology
    with open(data_dir / "input.json", "r") as f:
        input_json = json.load(f)
        input_data = input_json["data"] if "data" in input_json else input_json

    # Load batch scenarios
    with open(data_dir / "update.json", "r") as f:
        update_json = json.load(f)
        update_data = update_json["data"] if "data" in update_json else update_json

    # Load PGM reference results
    with open(data_dir / "batch_output.json", "r") as f:
        pgm_batch_output = json.load(f)
        pgm_results = pgm_batch_output["scenarios"]  # List of scenario results
        pgm_time_ms = pgm_batch_output["total_calculation_time_s"] * 1000.0

    # Load metadata for PGM timing
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Parse base network
    buses = []
    for bus in input_data["node"]:
        # [id, u_rated, bus_type, u_pu, u_angle, p_load, q_load]
        buses.append(
            [
                bus["id"],
                bus["u_rated"],
                0,  # PQ bus type (will be updated for slack)
                1.0,  # initial voltage magnitude
                0.0,  # initial voltage angle
                0.0,  # will be set per scenario
                0.0,  # will be set per scenario
            ]
        )

    # Mark source bus (connected node) as SLACK
    source_node_id = input_data["source"][0]["node"]
    source_u_ref = input_data["source"][0]["u_ref"]
    for bus in buses:
        if bus[0] == source_node_id:
            bus[2] = 2  # SLACK type
            bus[3] = source_u_ref  # Set slack voltage
            break

    branches = []
    for line in input_data["line"]:
        # [id, from_bus, to_bus, r, x, b]
        branches.append(
            [
                line["id"],
                line["from_node"],
                line["to_node"],
                line["r1"],
                line["x1"],
                0.0,  # susceptance
            ]
        )

    # Parse batch scenarios
    sym_load = update_data["sym_load"]
    load_ids = sym_load["id"]
    p_scenarios = sym_load["p_specified"]
    q_scenarios = sym_load["q_specified"]

    # Create load ID to node ID mapping
    load_to_node = {}
    for load in input_data["sym_load"]:
        load_to_node[load["id"]] = load["node"]

    n_scenarios = len(p_scenarios)
    n_loads = len(load_ids)

    print(f"  Buses: {len(buses)}")
    print(f"  Branches: {len(branches)}")
    print(f"  Scenarios: {n_scenarios}")
    print(f"  Loads per scenario: {n_loads}")

    # Create scenarios in GAP format
    # Note: PGM uses generation convention (loads positive), GAP uses load convention (loads negative)
    scenarios_data = []
    for scenario_idx in range(n_scenarios):
        scenario_updates = []
        for load_idx, load_id in enumerate(load_ids):
            p_load_pgm = p_scenarios[scenario_idx][load_idx]
            q_load_pgm = q_scenarios[scenario_idx][load_idx]
            # Convert from PGM convention (positive=load) to GAP convention (negative=load)
            p_load = -p_load_pgm
            q_load = -q_load_pgm
            # Map load ID to node ID
            node_id = load_to_node[load_id]
            scenario_updates.append([node_id, p_load, q_load])
        scenarios_data.append(scenario_updates)

    # Test 1: Run batch calculation with Y-bus caching
    print("\n" + "-" * 80)
    print("Test 1: Batch calculation WITH Y-bus factorization caching")
    print("-" * 80)

    # PGM uses 1 MVA as default base power
    base_power_va = 1e6

    start_time = time.time()
    voltages, stats = gap_solver.solve_simple_power_flow_batch(
        buses,
        branches,
        scenarios_data,
        tolerance=1e-6,
        max_iterations=50,
        verbose=False,
        base_power=base_power_va,
        backend="cpu",
        reuse_y_bus_factorization=True,
        warm_start=False,
        verbose_summary=True,
    )
    gap_time_ms = (time.time() - start_time) * 1000.0

    total_iters, total_time_ms, avg_time_ms, converged, failed = stats

    print(f"\n✅ Batch calculation completed")
    print(f"  Total iterations: {int(total_iters)}")
    print(f"  Converged scenarios: {int(converged)}/{n_scenarios}")
    print(f"  Failed scenarios: {int(failed)}")
    print(f"  Total solve time: {total_time_ms:.4f} ms")
    print(f"  Avg time per scenario: {avg_time_ms:.4f} ms")
    print(f"  Python overhead: {gap_time_ms - total_time_ms:.4f} ms")

    # Test 2: Run batch calculation WITHOUT Y-bus caching (for comparison)
    print("\n" + "-" * 80)
    print("Test 2: Batch calculation WITHOUT Y-bus factorization caching")
    print("-" * 80)

    start_time = time.time()
    voltages_no_cache, stats_no_cache = gap_solver.solve_simple_power_flow_batch(
        buses,
        branches,
        scenarios_data,
        tolerance=1e-6,
        max_iterations=50,
        verbose=False,
        base_power=base_power_va,
        backend="cpu",
        reuse_y_bus_factorization=False,  # Disable caching
        warm_start=False,
        verbose_summary=True,
    )
    gap_time_no_cache_ms = (time.time() - start_time) * 1000.0

    total_iters_nc, total_time_nc_ms, avg_time_nc_ms, converged_nc, failed_nc = (
        stats_no_cache
    )

    print(f"\n✅ Batch calculation completed (no cache)")
    print(f"  Total iterations: {int(total_iters_nc)}")
    print(f"  Converged scenarios: {int(converged_nc)}/{n_scenarios}")
    print(f"  Failed scenarios: {int(failed_nc)}")
    print(f"  Total solve time: {total_time_nc_ms:.4f} ms")
    print(f"  Avg time per scenario: {avg_time_nc_ms:.4f} ms")

    # Validate against PGM results
    print("\n" + "-" * 80)
    print("Validation against PGM Reference")
    print("-" * 80)

    max_voltage_error = 0.0
    max_angle_error = 0.0

    for scenario_idx in range(n_scenarios):
        gap_voltages = voltages[scenario_idx]

        # Get PGM results for this scenario (columnar format)
        pgm_scenario_result = pgm_results[scenario_idx]
        pgm_node_data = pgm_scenario_result["node"]

        # Convert columnar to dict by id
        pgm_node_dict = {}
        for i, node_id in enumerate(pgm_node_data["id"]):
            pgm_node_dict[node_id] = {
                "u_pu": pgm_node_data["u_pu"][i],
                "u_angle": pgm_node_data["u_angle"][i],
            }

        for bus_idx, gap_voltage in enumerate(gap_voltages):
            # Find corresponding PGM result
            bus_id = buses[bus_idx][0]
            if bus_id not in pgm_node_dict:
                continue

            pgm_result = pgm_node_dict[bus_id]

            # GAP voltage: [real, imag, magnitude, angle_rad]
            gap_v_mag = gap_voltage[2]
            gap_v_angle = gap_voltage[3]

            # PGM voltage
            pgm_v_mag = pgm_result["u_pu"]
            pgm_v_angle_rad = pgm_result["u_angle"]

            # Compute errors
            v_error = abs(gap_v_mag - pgm_v_mag)
            angle_error = abs(gap_v_angle - pgm_v_angle_rad)

            max_voltage_error = max(max_voltage_error, v_error)
            max_angle_error = max(max_angle_error, angle_error)

    print(f"\n  Maximum voltage magnitude error: {max_voltage_error:.3e} p.u.")
    print(f"  Maximum voltage angle error: {max_angle_error:.3e} radians")

    # Accuracy check (realistic thresholds considering different solution methods)
    voltage_threshold = 1e-3  # 0.1% voltage error
    angle_threshold = 1e-3  # ~0.06 degrees

    accuracy_pass = (
        max_voltage_error < voltage_threshold and max_angle_error < angle_threshold
    )

    if accuracy_pass:
        print(
            f"  ✅ ACCURACY: GOOD (max error {max_voltage_error:.2e} p.u. < {voltage_threshold:.0e})"
        )
    else:
        print(f"  ⚠️  ACCURACY: Errors exceed threshold")

    # Performance comparison
    print("\n" + "-" * 80)
    print("Performance Comparison")
    print("-" * 80)

    print(f"\n  PGM Batch:           {pgm_time_ms:8.2f} ms")
    print(
        f"  GAP Batch (cached):  {total_time_ms:8.2f} ms  ({total_time_ms/pgm_time_ms:.2f}x)"
    )
    print(
        f"  GAP Batch (no cache):{total_time_nc_ms:8.2f} ms  ({total_time_nc_ms/pgm_time_ms:.2f}x)"
    )
    print(f"\n  Speedup from caching: {total_time_nc_ms/total_time_ms:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    if accuracy_pass and int(converged) == n_scenarios:
        print("✅ ALL TESTS PASSED")
        print(
            f"   - Accuracy: Good ({max_voltage_error:.2e} p.u. < {voltage_threshold:.0e})"
        )
        print(f"   - Convergence: {int(converged)}/{n_scenarios} scenarios")
        print(
            f"   - Performance: {total_time_ms/pgm_time_ms:.2f}x vs PGM (with caching)"
        )
        print(f"   - Caching benefit: {total_time_nc_ms/total_time_ms:.2f}x speedup")
    else:
        print("❌ TESTS FAILED")
        if int(converged) != n_scenarios:
            print(f"   - Convergence issue: {int(converged)}/{n_scenarios} scenarios")
        if not accuracy_pass:
            print(f"   - Accuracy issue: errors exceed threshold")

    # Use assertions for pytest
    assert (
        int(converged) == n_scenarios
    ), f"Only {int(converged)}/{n_scenarios} scenarios converged"
    assert (
        accuracy_pass
    ), f"Accuracy check failed: max error {max_voltage_error:.2e} exceeds {voltage_threshold:.2e}"


if __name__ == "__main__":
    test_batch_api()
    print("\n✅ Test completed successfully")
