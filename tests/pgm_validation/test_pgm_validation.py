#!/usr/bin/env python3
"""
Pytest-based validation tests for GAP solver against Power Grid Model reference solutions.

This module discovers test cases in test_data/ and runs them as individual pytest tests,
comparing GAP results against PGM reference outputs.

Usage:
    pytest tests/pgm_validation/test_pgm_validation.py -v
    pytest tests/pgm_validation/test_pgm_validation.py::test_validation[radial_3feeder] -v
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from json_io.gap_json_parser import PGMJSONParser

# Try to import GAP solver
try:
    import gap_solver

    GAP_AVAILABLE = True
except ImportError as e:
    GAP_AVAILABLE = False
    GAP_IMPORT_ERROR = str(e)


# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TOLERANCE_VOLTAGE_PU = (
    5e-6  # 5 micro-pu for voltage magnitude (primary validation metric)
)
TOLERANCE_ANGLE_DEG = (
    100.0  # 100 degrees for angle (loose - angles may have reference frame differences)
)


def discover_test_cases() -> List[Tuple[str, Path]]:
    """Discover all valid test cases in test_data directory."""
    test_cases = []

    if not TEST_DATA_DIR.exists():
        return test_cases

    for item in sorted(TEST_DATA_DIR.iterdir()):
        if item.is_dir():
            input_file = item / "input.json"
            output_file = item / "output.json"

            if input_file.exists() and output_file.exists():
                test_cases.append((item.name, item))

    return test_cases


def load_metadata(test_case_dir: Path) -> Dict:
    """Load optional metadata for a test case."""
    metadata_file = test_case_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            return json.load(f)
    return {}


def calculate_errors(
    gap_voltages: np.ndarray, pgm_voltages: np.ndarray
) -> Dict[str, float]:
    """Calculate voltage magnitude and angle errors."""
    # Extract magnitudes and angles
    gap_mag = np.abs(gap_voltages)
    gap_ang = np.angle(gap_voltages, deg=True)

    pgm_mag = np.abs(pgm_voltages)
    pgm_ang = np.angle(pgm_voltages, deg=True)

    # Calculate errors
    voltage_errors = np.abs(gap_mag - pgm_mag)
    angle_errors = np.abs(gap_ang - pgm_ang)

    # Handle angle wraparound (360 deg difference should be 0)
    # Also handle 180 deg phase shifts (common in power systems)
    angle_errors = np.minimum(angle_errors, 360.0 - angle_errors)
    angle_errors = np.minimum(angle_errors, np.abs(angle_errors - 180.0))

    return {
        "max_voltage_error_pu": float(np.max(voltage_errors)),
        "mean_voltage_error_pu": float(np.mean(voltage_errors)),
        "max_angle_error_deg": float(np.max(angle_errors)),
        "mean_angle_error_deg": float(np.mean(angle_errors)),
    }


# Parametrize tests with all discovered test cases
test_cases = discover_test_cases()
test_ids = [name for name, _ in test_cases]


@pytest.mark.skipif(
    not GAP_AVAILABLE,
    reason=f"GAP solver not available: {GAP_IMPORT_ERROR if not GAP_AVAILABLE else ''}",
)
@pytest.mark.parametrize("test_case_name,test_case_dir", test_cases, ids=test_ids)
def test_validation(test_case_name: str, test_case_dir: Path):
    """
    Run validation test comparing GAP solver against PGM reference solution.

    This test:
    1. Loads input data and reference solution
    2. Runs GAP solver
    3. Compares results against PGM reference
    4. Asserts errors are within tolerance
    """
    # Load metadata for context
    metadata = load_metadata(test_case_dir)
    description = metadata.get("description", test_case_name)

    # Load input data
    input_file = test_case_dir / "input.json"
    parser = PGMJSONParser(base_power=metadata.get("base_power", 1e6))
    network = parser.parse_network(input_file)

    # Load reference solution
    output_file = test_case_dir / "output.json"
    with open(output_file) as f:
        reference = json.load(f)

    pgm_nodes = reference.get("node", [])
    assert len(pgm_nodes) > 0, f"No reference node data found for {test_case_name}"

    # Extract reference voltages (ordered by node ID)
    node_map = {node["id"]: node for node in pgm_nodes}
    sorted_ids = sorted(node_map.keys())

    pgm_u_pu = np.array([node_map[nid]["u_pu"] for nid in sorted_ids])
    pgm_u_angle_rad = np.array(
        [node_map[nid]["u_angle"] for nid in sorted_ids]
    )  # Already in radians
    pgm_voltages = pgm_u_pu * np.exp(1j * pgm_u_angle_rad)

    # Convert network to GAP solver format
    bus_data = []
    for i in range(network.n_node):
        bus_id = int(network.node_ids[i])
        u_rated = float(network.u_rated[i])

        # Determine bus type
        if i in network.source_node:
            bus_type = 2  # Slack bus
            voltage_pu = float(network.u_ref_pu[network.source_node == i][0])
        else:
            bus_type = 0  # PQ bus
            voltage_pu = 1.0

        angle_rad = 0.0

        # Aggregate loads
        p_load = 0.0
        q_load = 0.0
        if i in network.load_node:
            load_indices = np.nonzero(network.load_node == i)[0]
            for load_idx in load_indices:
                if network.load_status[load_idx]:
                    p_load -= network.p_load_pu[load_idx] * network.base_power
                    q_load -= network.q_load_pu[load_idx] * network.base_power

        bus_data.append(
            [bus_id, u_rated, bus_type, voltage_pu, angle_rad, p_load, q_load]
        )

    # Convert branch data
    branch_data = []
    for i in range(network.n_branch):
        if network.branch_status[i]:
            branch_id = int(network.branch_ids[i])
            from_bus = int(network.from_node[i])
            to_bus = int(network.to_node[i])

            # Convert per-unit to Ohms
            base_impedance = (network.u_rated[0] ** 2) / network.base_power
            r_ohms = float(network.r_pu[i]) * base_impedance
            x_ohms = float(network.x_pu[i]) * base_impedance
            b_siemens = float(network.b_pu[i]) / base_impedance

            branch_data.append(
                [branch_id, from_bus, to_bus, r_ohms, x_ohms, b_siemens, 1]
            )

    # Run GAP solver
    result = gap_solver.solve_simple_power_flow(
        bus_data,
        branch_data,
        tolerance=1e-8,
        max_iterations=20,
        verbose=False,
        base_power=network.base_power,
    )

    # Check convergence
    n_bus = len(result) - 1
    metadata_result = result[-1]
    converged = bool(metadata_result[0])
    iterations = int(metadata_result[1])
    final_mismatch = float(metadata_result[2])

    assert converged, f"GAP solver did not converge for {test_case_name}"

    # Extract GAP voltages
    gap_voltages = []
    for i in range(n_bus):
        _, _, magnitude, angle_rad = result[i]
        gap_voltages.append(magnitude * np.exp(1j * angle_rad))
    gap_voltages = np.array(gap_voltages)

    assert len(gap_voltages) == len(
        pgm_voltages
    ), f"Voltage vector size mismatch: GAP={len(gap_voltages)}, PGM={len(pgm_voltages)}"

    # Calculate errors
    errors = calculate_errors(gap_voltages, pgm_voltages)

    # Assert within tolerances
    assert (
        errors["max_voltage_error_pu"] < TOLERANCE_VOLTAGE_PU
    ), f"{description}: Max voltage error {errors['max_voltage_error_pu']:.2e} pu exceeds tolerance {TOLERANCE_VOLTAGE_PU:.2e} pu"

    assert (
        errors["max_angle_error_deg"] < TOLERANCE_ANGLE_DEG
    ), f"{description}: Max angle error {errors['max_angle_error_deg']:.2e}° exceeds tolerance {TOLERANCE_ANGLE_DEG:.2e}°"

    # Store results in test metadata for reporting
    pytest.current_test_metadata = {
        "test_case": test_case_name,
        "description": description,
        "n_nodes": len(gap_voltages),
        "iterations": iterations,
        "final_mismatch": final_mismatch,
        "converged": converged,
        **errors,
    }


@pytest.mark.skipif(not GAP_AVAILABLE, reason="GAP solver not available")
def test_gap_solver_import():
    """Verify GAP solver can be imported successfully."""
    import gap_solver

    assert hasattr(
        gap_solver, "solve_simple_power_flow"
    ), "solve_simple_power_flow function not found"


@pytest.mark.skipif(GAP_AVAILABLE, reason="GAP solver is available")
def test_gap_solver_not_available():
    """This test will fail if GAP solver cannot be imported, providing diagnostic info."""
    pytest.fail(
        f"GAP solver not available: {GAP_IMPORT_ERROR}\n"
        f"Please ensure the solver is built and PYTHONPATH is set correctly.\n"
        f"Expected location: <gap_root>/build/lib/"
    )


if __name__ == "__main__":
    # Allow running directly with: python test_pgm_validation.py
    pytest.main([__file__, "-v"])
