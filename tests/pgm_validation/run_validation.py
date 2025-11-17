#!/usr/bin/env python3
"""
GAP Solver Validation Runner

This script discovers and runs validation tests for all test cases in the test_data directory.
Each test case should have:
  - input.json: Power Grid Model format input
  - output.json: Reference solution from PGM
  - metadata.json (optional): Test case metadata

The script compares GAP solver results against PGM reference solutions and generates
a comprehensive validation report.

Usage:
    python run_validation.py [--test-case NAME] [--verbose] [--base-power MVA]
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from json_io.gap_json_parser import PGMJSONParser

# Try to import GAP solver
GAP_AVAILABLE = False
GAP_IMPORT_ERROR = None
try:
    import gap_solver

    GAP_AVAILABLE = True
except ImportError as e:
    GAP_IMPORT_ERROR = str(e)


@dataclass
class ValidationResult:
    """Results from a single validation test."""

    test_case: str
    success: bool
    converged: bool
    iterations: int
    final_mismatch: float
    calculation_time: float
    max_voltage_error_pu: float
    max_angle_error_deg: float
    mean_voltage_error_pu: float
    mean_angle_error_deg: float
    pgm_calculation_time: Optional[float] = None
    n_nodes: Optional[int] = None
    n_lines: Optional[int] = None
    per_node_errors: Optional[List[Dict]] = None
    error_message: Optional[str] = None


class ValidationRunner:
    """Runs validation tests for GAP solver against PGM reference solutions."""

    def __init__(
        self, test_data_dir: Path, base_power: float = None, verbose: bool = False
    ):
        self.test_data_dir = test_data_dir
        self.base_power = base_power
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def discover_test_cases(self) -> List[Path]:
        """Discover all test cases in the test_data directory."""
        test_cases = []

        if not self.test_data_dir.exists():
            print(f"❌ Test data directory not found: {self.test_data_dir}")
            return test_cases

        # Look for directories containing input.json and output.json
        for item in sorted(self.test_data_dir.iterdir()):
            if item.is_dir():
                input_file = item / "input.json"
                output_file = item / "output.json"

                if input_file.exists() and output_file.exists():
                    test_cases.append(item)
                    if self.verbose:
                        print(f"  Found test case: {item.name}")

        return test_cases

    def load_metadata(self, test_case_dir: Path) -> Dict:
        """Load test case metadata if available."""
        metadata_file = test_case_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        return {}

    def run_gap_solver(self, network, input_file: Path) -> tuple:
        """Run GAP solver on the network."""
        # Convert to GAP format
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

        # Run solver
        start_time = time.time()
        result = gap_solver.solve_simple_power_flow(
            bus_data,
            branch_data,
            tolerance=1e-8,
            max_iterations=20,
            verbose=self.verbose,
            base_power=network.base_power,
        )
        calc_time = time.time() - start_time

        # Extract results
        n_bus = len(result) - 1
        metadata = result[-1]
        converged = bool(metadata[0])
        iterations = int(metadata[1])
        final_mismatch = metadata[2]

        gap_results = {
            "converged": converged,
            "iterations": iterations,
            "final_mismatch": final_mismatch,
            "calculation_time": calc_time,
            "buses": [],
        }

        for i in range(n_bus):
            _, _, magnitude, angle_rad = result[i]
            gap_results["buses"].append(
                {
                    "id": int(network.node_ids[i]),
                    "u_pu": magnitude,
                    "u_angle_deg": np.degrees(angle_rad),
                    "u_angle_rad": angle_rad,
                }
            )

        return gap_results

    def compare_results(self, gap_results: Dict, pgm_reference: Dict) -> tuple:
        """Compare GAP results with PGM reference and calculate errors."""
        # Create mapping from bus ID to results
        gap_buses = {b["id"]: b for b in gap_results["buses"]}
        pgm_buses = {b["id"]: b for b in pgm_reference["node"]}

        voltage_errors = []
        angle_errors = []
        per_node_errors = []

        for bus_id in sorted(gap_buses.keys()):
            if bus_id not in pgm_buses:
                continue

            gap_bus = gap_buses[bus_id]
            pgm_bus = pgm_buses[bus_id]

            # Voltage magnitude error
            v_error = abs(gap_bus["u_pu"] - pgm_bus["u_pu"])
            voltage_errors.append(v_error)

            # Angle error (PGM output is already in degrees)
            pgm_angle_deg = pgm_bus[
                "u_angle"
            ]  # Already in degrees from reference solution
            angle_error = abs(gap_bus["u_angle_deg"] - pgm_angle_deg)
            angle_errors.append(angle_error)

            # Store per-node details
            per_node_errors.append(
                {
                    "bus_id": bus_id,
                    "gap_v_pu": gap_bus["u_pu"],
                    "pgm_v_pu": pgm_bus["u_pu"],
                    "v_error_pu": v_error,
                    "gap_angle_deg": gap_bus["u_angle_deg"],
                    "pgm_angle_deg": pgm_angle_deg,
                    "angle_error_deg": angle_error,
                }
            )

        max_v_error = max(voltage_errors) if voltage_errors else 0.0
        mean_v_error = np.mean(voltage_errors) if voltage_errors else 0.0
        max_angle_error = max(angle_errors) if angle_errors else 0.0
        mean_angle_error = np.mean(angle_errors) if angle_errors else 0.0

        return (
            max_v_error,
            mean_v_error,
            max_angle_error,
            mean_angle_error,
            per_node_errors,
        )

    def run_test_case(self, test_case_dir: Path) -> ValidationResult:
        """Run validation for a single test case."""
        test_name = test_case_dir.name

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Running: {test_name}")
            print(f"{'='*80}")

        try:
            # Load input and reference
            input_file = test_case_dir / "input.json"
            output_file = test_case_dir / "output.json"

            with open(output_file) as f:
                pgm_reference = json.load(f)

            # Determine base power
            metadata = self.load_metadata(test_case_dir)
            base_power = self.base_power

            if base_power is None:
                # Try to get from metadata
                base_power = metadata.get("base_power_va")

                if base_power is None:
                    # Default: Use 10 MVA for distribution, 100 MVA for transmission
                    # Simple heuristic: if total load < 10 MW, use 10 MVA
                    with open(input_file) as f:
                        input_data = json.load(f)
                    loads = input_data.get("data", {}).get("sym_load", [])
                    total_p = sum(abs(load.get("p_specified", 0)) for load in loads)
                    base_power = 10e6 if total_p < 10e6 else 100e6

            # Parse network
            parser = PGMJSONParser(base_power=base_power)
            network = parser.parse_network(input_file)

            # Run GAP solver
            gap_results = self.run_gap_solver(network, input_file)

            if not gap_results["converged"]:
                return ValidationResult(
                    test_case=test_name,
                    success=False,
                    converged=False,
                    iterations=gap_results["iterations"],
                    final_mismatch=gap_results["final_mismatch"],
                    calculation_time=gap_results["calculation_time"],
                    max_voltage_error_pu=0.0,
                    max_angle_error_deg=0.0,
                    mean_voltage_error_pu=0.0,
                    mean_angle_error_deg=0.0,
                    error_message="Solver did not converge",
                )

            # Compare results
            (
                max_v_err,
                mean_v_err,
                max_a_err,
                mean_a_err,
                per_node,
            ) = self.compare_results(gap_results, pgm_reference)

            # Extract metadata
            pgm_calc_time = metadata.get("pgm_reference", {}).get("calculation_time_s")
            n_nodes = metadata.get("n_node")
            n_lines = metadata.get("n_line")

            # Determine success criteria (voltage magnitude is primary metric)
            success = max_v_err < 1e-4  # 0.01% error threshold

            return ValidationResult(
                test_case=test_name,
                success=success,
                converged=True,
                iterations=gap_results["iterations"],
                final_mismatch=gap_results["final_mismatch"],
                calculation_time=gap_results["calculation_time"],
                max_voltage_error_pu=max_v_err,
                max_angle_error_deg=max_a_err,
                mean_voltage_error_pu=mean_v_err,
                mean_angle_error_deg=mean_a_err,
                pgm_calculation_time=pgm_calc_time,
                n_nodes=n_nodes,
                n_lines=n_lines,
                per_node_errors=per_node,
            )

        except Exception as e:
            return ValidationResult(
                test_case=test_name,
                success=False,
                converged=False,
                iterations=0,
                final_mismatch=0.0,
                calculation_time=0.0,
                max_voltage_error_pu=0.0,
                max_angle_error_deg=0.0,
                mean_voltage_error_pu=0.0,
                mean_angle_error_deg=0.0,
                error_message=str(e),
            )

    def run_all(self, specific_test: Optional[str] = None) -> List[ValidationResult]:
        """Run validation for all test cases or a specific one."""
        test_cases = self.discover_test_cases()

        if specific_test:
            test_cases = [tc for tc in test_cases if tc.name == specific_test]
            if not test_cases:
                print(f"❌ Test case '{specific_test}' not found")
                return []

        print(f"\n{'='*80}")
        print(f"GAP SOLVER VALIDATION")
        print(f"{'='*80}")
        print(f"Found {len(test_cases)} test case(s)")

        if not GAP_AVAILABLE:
            print(f"\n❌ GAP solver not available: {GAP_IMPORT_ERROR}")
            print("Please ensure the solver is built and PYTHONPATH is set correctly.")
            return []

        for test_case_dir in test_cases:
            result = self.run_test_case(test_case_dir)
            self.results.append(result)

        return self.results

    def print_summary(self, detailed: bool = False):
        """Print validation summary."""
        if not self.results:
            print("\n⚠️  No validation results to display")
            return

        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}\n")

        # Summary table header
        if detailed:
            print(
                f"{'Test Case':<25} {'Status':<10} {'Nodes':<6} {'Iter':<6} "
                f"{'Max V Err':<12} {'Max ∠ Err':<12} {'GAP (ms)':<10} {'PGM (ms)':<10}"
            )
            print("-" * 110)
        else:
            print(
                f"{'Test Case':<25} {'Status':<10} {'Iter':<6} "
                f"{'Max V Err':<12} {'Max ∠ Err':<12} {'Time (ms)':<10}"
            )
            print("-" * 80)

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            if not result.converged:
                status = "❌ NO CONV"

            if detailed:
                pgm_time_str = (
                    f"{result.pgm_calculation_time*1000:.2f}"
                    if result.pgm_calculation_time
                    else "N/A"
                )
                nodes_str = f"{result.n_nodes}" if result.n_nodes else "N/A"
                print(
                    f"{result.test_case:<25} {status:<10} {nodes_str:<6} {result.iterations:<6} "
                    f"{result.max_voltage_error_pu:<12.2e} {result.max_angle_error_deg:<12.4f} "
                    f"{result.calculation_time*1000:<10.2f} {pgm_time_str:<10}"
                )
            else:
                print(
                    f"{result.test_case:<25} {status:<10} {result.iterations:<6} "
                    f"{result.max_voltage_error_pu:<12.2e} {result.max_angle_error_deg:<12.4f} "
                    f"{result.calculation_time*1000:<10.2f}"
                )

            if result.error_message:
                print(f"  └─ Error: {result.error_message}")

        if detailed:
            print("-" * 110)
        else:
            print("-" * 80)

        # Overall statistics
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed

        print(f"\n📊 Overall Results: {passed}/{len(self.results)} PASSED")

        if failed > 0:
            print(f"   ⚠️  {failed} test(s) failed")

        # Detailed statistics for passed tests
        if passed > 0:
            passed_results = [r for r in self.results if r.success]
            avg_iterations = np.mean([r.iterations for r in passed_results])
            avg_gap_time = np.mean([r.calculation_time for r in passed_results])
            max_v_err = max([r.max_voltage_error_pu for r in passed_results])

            print(f"\n   Average GAP iterations: {avg_iterations:.1f}")
            print(f"   Average GAP time: {avg_gap_time*1000:.2f} ms")
            print(f"   Maximum voltage error: {max_v_err:.2e} pu")

            # PGM timing stats if available
            pgm_times = [
                r.pgm_calculation_time for r in passed_results if r.pgm_calculation_time
            ]
            if pgm_times:
                avg_pgm_time = np.mean(pgm_times)
                print(f"   Average PGM time: {avg_pgm_time*1000:.2f} ms")
                speedup = avg_pgm_time / avg_gap_time
                if speedup > 1:
                    print(f"   GAP speedup: {speedup:.2f}x faster than PGM")
                else:
                    print(f"   PGM speedup: {1/speedup:.2f}x faster than GAP")

    def print_detailed_results(self, test_case_name: Optional[str] = None):
        """Print detailed per-node comparison for a specific test case or all."""
        results_to_print = self.results
        if test_case_name:
            results_to_print = [
                r for r in self.results if r.test_case == test_case_name
            ]
            if not results_to_print:
                print(f"❌ No results found for test case: {test_case_name}")
                return

        for result in results_to_print:
            if not result.per_node_errors:
                continue

            print(f"\n{'='*100}")
            print(f"Detailed Results: {result.test_case}")
            print(f"{'='*100}")
            print(f"Network: {result.n_nodes} nodes, {result.n_lines} lines")
            print(
                f"Convergence: {'✅ Yes' if result.converged else '❌ No'} ({result.iterations} iterations)"
            )
            print(f"GAP Time: {result.calculation_time*1000:.3f} ms")
            if result.pgm_calculation_time:
                print(f"PGM Time: {result.pgm_calculation_time*1000:.3f} ms")
            print()
            print(
                f"{'Bus':<6} {'GAP V (pu)':<12} {'PGM V (pu)':<12} {'V Err (pu)':<12} "
                f"{'GAP ∠ (°)':<12} {'PGM ∠ (°)':<12} {'∠ Err (°)':<12}"
            )
            print("-" * 100)

            for node in result.per_node_errors:
                print(
                    f"{node['bus_id']:<6} "
                    f"{node['gap_v_pu']:<12.8f} {node['pgm_v_pu']:<12.8f} {node['v_error_pu']:<12.2e} "
                    f"{node['gap_angle_deg']:<12.6f} {node['pgm_angle_deg']:<12.6f} {node['angle_error_deg']:<12.6f}"
                )

            print("-" * 100)
            print(f"Max Voltage Error: {result.max_voltage_error_pu:.2e} pu")
            print(f"Mean Voltage Error: {result.mean_voltage_error_pu:.2e} pu")
            print(f"Max Angle Error: {result.max_angle_error_deg:.6f}°")
            print(f"Mean Angle Error: {result.mean_angle_error_deg:.6f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Run GAP solver validation tests against PGM reference solutions"
    )
    parser.add_argument("--test-case", help="Run specific test case (default: run all)")
    parser.add_argument(
        "--base-power",
        type=float,
        help="Base power in VA (default: auto-detect from metadata or load size)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed comparison including PGM timing",
    )
    parser.add_argument(
        "--per-node",
        action="store_true",
        help="Show per-node voltage and angle comparisons",
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        default=Path(__file__).parent / "test_data",
        help="Path to test data directory",
    )

    args = parser.parse_args()

    runner = ValidationRunner(
        test_data_dir=args.test_data_dir,
        base_power=args.base_power,
        verbose=args.verbose,
    )

    runner.run_all(specific_test=args.test_case)
    runner.print_summary(detailed=args.detailed)

    if args.per_node:
        runner.print_detailed_results(test_case_name=args.test_case)

    # Exit with error code if any tests failed
    if any(not r.success for r in runner.results):
        sys.exit(1)


if __name__ == "__main__":
    main()
