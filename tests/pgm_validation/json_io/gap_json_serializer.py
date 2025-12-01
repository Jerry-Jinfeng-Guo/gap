"""
JSON Output Serializer for GAP NRPF Results

This module serializes GAP Newton-Raphson power flow results to
Power Grid Model JSON format for benchmark comparison.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import GAP types if available
try:
    from ...types import Complex, Float, Index
except ImportError:
    # Fallback type definitions
    Float = np.float64
    Complex = np.complex128
    Index = np.int32


@dataclass
class GAPPowerFlowResults:
    """
    GAP power flow calculation results.

    Contains all results from Newton-Raphson power flow solution
    in formats compatible with PGM benchmark output structure.
    """

    # Convergence information
    converged: bool
    n_iterations: int
    max_mismatch: float
    calculation_time: float

    # Node results (voltages)
    node_ids: np.ndarray  # [n_node] Node IDs
    u_pu: np.ndarray  # [n_node] Voltage magnitude (p.u.)
    u_angle_deg: np.ndarray  # [n_node] Voltage angle (degrees)
    u_volt: np.ndarray  # [n_node] Voltage magnitude (V)

    # Branch results (power flows)
    branch_ids: np.ndarray  # [n_branch] Branch IDs
    p_from_mw: np.ndarray  # [n_branch] Active power from (MW)
    q_from_mvar: np.ndarray  # [n_branch] Reactive power from (MVAr)
    p_to_mw: np.ndarray  # [n_branch] Active power to (MW)
    q_to_mvar: np.ndarray  # [n_branch] Reactive power to (MVAr)
    i_from_a: np.ndarray  # [n_branch] Current from (A)
    i_to_a: np.ndarray  # [n_branch] Current to (A)
    loading: np.ndarray  # [n_branch] Loading factor (p.u.)

    # Load results (power consumption)
    load_ids: np.ndarray  # [n_load] Load IDs
    p_load_mw: np.ndarray  # [n_load] Active power consumed (MW)
    q_load_mvar: np.ndarray  # [n_load] Reactive power consumed (MVAr)

    # Source results (power generation)
    source_ids: np.ndarray  # [n_source] Source IDs
    p_source_mw: np.ndarray  # [n_source] Active power generated (MW)
    q_source_mvar: np.ndarray  # [n_source] Reactive power generated (MVAr)

    # Base values
    base_power: float  # Base power (VA)
    base_voltage: float  # Base voltage (V)


class GAPJSONSerializer:
    """
    Serializer to convert GAP NRPF results to PGM JSON output format.

    Creates standardized output.json files compatible with Power Grid Model
    benchmark comparison tools and analysis pipelines.
    """

    def __init__(self):
        """Initialize serializer."""
        pass

    def serialize_results(
        self,
        results: GAPPowerFlowResults,
        output_file: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Serialize GAP power flow results to PGM JSON format.

        Args:
            results: GAP power flow results
            output_file: Path to output.json file
            metadata: Optional metadata to include
        """
        # Create PGM-compatible output structure
        output_data = {
            "node": self._serialize_node_results(results),
            "line": self._serialize_line_results(results),
            "sym_load": self._serialize_load_results(results),
            "source": self._serialize_source_results(results),
            "meta": self._serialize_metadata(results, metadata),
        }

        # Write to JSON file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=self._json_serializer)

    def _serialize_node_results(
        self, results: GAPPowerFlowResults
    ) -> List[Dict[str, Any]]:
        """Serialize node voltage results."""
        node_results = []

        for i in range(len(results.node_ids)):
            node_results.append(
                {
                    "id": int(results.node_ids[i]),
                    "energized": True,  # Assume all nodes energized if converged
                    "u_pu": float(results.u_pu[i]),
                    "u_angle": float(results.u_angle_deg[i]),
                    "u": float(results.u_volt[i]),
                    "p": float(0.0),  # Node injection computed separately
                    "q": float(0.0),  # Node injection computed separately
                }
            )

        return node_results

    def _serialize_line_results(
        self, results: GAPPowerFlowResults
    ) -> List[Dict[str, Any]]:
        """Serialize line/branch power flow results."""
        line_results = []

        for i in range(len(results.branch_ids)):
            line_results.append(
                {
                    "id": int(results.branch_ids[i]),
                    "energized": True,  # Assume energized if converged
                    "loading": float(results.loading[i]),
                    "p_from": float(results.p_from_mw[i]),
                    "q_from": float(results.q_from_mvar[i]),
                    "i_from": float(results.i_from_a[i]),
                    "s_from": float(
                        np.sqrt(results.p_from_mw[i] ** 2 + results.q_from_mvar[i] ** 2)
                    ),
                    "p_to": float(results.p_to_mw[i]),
                    "q_to": float(results.q_to_mvar[i]),
                    "i_to": float(results.i_to_a[i]),
                    "s_to": float(
                        np.sqrt(results.p_to_mw[i] ** 2 + results.q_to_mvar[i] ** 2)
                    ),
                }
            )

        return line_results

    def _serialize_load_results(
        self, results: GAPPowerFlowResults
    ) -> List[Dict[str, Any]]:
        """Serialize load consumption results."""
        load_results = []

        for i in range(len(results.load_ids)):
            load_results.append(
                {
                    "id": int(results.load_ids[i]),
                    "energized": True,
                    "p": float(results.p_load_mw[i]),
                    "q": float(results.q_load_mvar[i]),
                    "s": float(
                        np.sqrt(results.p_load_mw[i] ** 2 + results.q_load_mvar[i] ** 2)
                    ),
                    "pf": float(
                        results.p_load_mw[i]
                        / max(
                            1e-12,
                            np.sqrt(
                                results.p_load_mw[i] ** 2 + results.q_load_mvar[i] ** 2
                            ),
                        )
                    ),
                }
            )

        return load_results

    def _serialize_source_results(
        self, results: GAPPowerFlowResults
    ) -> List[Dict[str, Any]]:
        """Serialize source generation results."""
        source_results = []

        for i in range(len(results.source_ids)):
            source_results.append(
                {
                    "id": int(results.source_ids[i]),
                    "energized": True,
                    "p": float(results.p_source_mw[i]),
                    "q": float(results.q_source_mvar[i]),
                    "s": float(
                        np.sqrt(
                            results.p_source_mw[i] ** 2 + results.q_source_mvar[i] ** 2
                        )
                    ),
                    "pf": float(
                        results.p_source_mw[i]
                        / max(
                            1e-12,
                            np.sqrt(
                                results.p_source_mw[i] ** 2
                                + results.q_source_mvar[i] ** 2
                            ),
                        )
                    ),
                    "i": float(0.0),  # Source current (computed from power)
                }
            )

        return source_results

    def _serialize_metadata(
        self, results: GAPPowerFlowResults, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Serialize calculation metadata."""
        meta = {
            "solver": "GAP_NRPF",
            "calculation_method": "newton_raphson",
            "converged": results.converged,
            "n_iterations": results.n_iterations,
            "max_mismatch": float(results.max_mismatch),
            "calculation_time_s": float(results.calculation_time),
            "base_power_va": float(results.base_power),
            "base_voltage_v": float(results.base_voltage),
            "symmetric": True,  # GAP NRPF is symmetric solver
        }

        # Add custom metadata if provided
        if metadata:
            meta.update(metadata)

        return meta

    def create_comparison_summary(
        self,
        gap_results: GAPPowerFlowResults,
        reference_file: Path,
        output_file: Path,
        tolerance: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Create detailed comparison summary between GAP and reference results.

        Args:
            gap_results: GAP calculation results
            reference_file: Path to reference output.json
            output_file: Path to save comparison summary
            tolerance: Tolerance values for different metrics

        Returns:
            Comparison summary dictionary
        """
        if tolerance is None:
            tolerance = {
                "voltage_pu": 1e-6,
                "angle_deg": 1e-4,
                "power_mw": 1e-3,
                "power_mvar": 1e-3,
            }

        # Load reference results
        with open(reference_file, "r") as f:
            reference_data = json.load(f)

        # Compare node results
        node_comparison = self._compare_node_results(
            gap_results, reference_data.get("node", []), tolerance
        )

        # Compare line results
        line_comparison = self._compare_line_results(
            gap_results, reference_data.get("line", []), tolerance
        )

        # Compare load results
        load_comparison = self._compare_load_results(
            gap_results, reference_data.get("sym_load", []), tolerance
        )

        # Create summary
        summary = {
            "comparison_timestamp": pd.Timestamp.now().isoformat(),
            "gap_solver_info": {
                "converged": gap_results.converged,
                "n_iterations": gap_results.n_iterations,
                "max_mismatch": gap_results.max_mismatch,
                "calculation_time": gap_results.calculation_time,
            },
            "reference_solver_info": reference_data.get("meta", {}),
            "node_comparison": node_comparison,
            "line_comparison": line_comparison,
            "load_comparison": load_comparison,
            "tolerance_settings": tolerance,
            "overall_match": self._evaluate_overall_match(
                node_comparison, line_comparison, load_comparison
            ),
        }

        # Save comparison summary
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)

        return summary

    def _compare_node_results(
        self,
        gap_results: GAPPowerFlowResults,
        reference_nodes: List[Dict],
        tolerance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare node voltage results."""
        # Create reference lookup by ID
        ref_by_id = {node["id"]: node for node in reference_nodes}

        voltage_errors = []
        angle_errors = []
        max_voltage_error = 0.0
        max_angle_error = 0.0

        for i, node_id in enumerate(gap_results.node_ids):
            if int(node_id) in ref_by_id:
                ref_node = ref_by_id[int(node_id)]

                # Voltage magnitude error
                v_error = abs(gap_results.u_pu[i] - ref_node["u_pu"])
                voltage_errors.append(v_error)
                max_voltage_error = max(max_voltage_error, v_error)

                # Voltage angle error
                a_error = abs(gap_results.u_angle_deg[i] - ref_node["u_angle"])
                angle_errors.append(a_error)
                max_angle_error = max(max_angle_error, a_error)

        return {
            "n_nodes_compared": len(voltage_errors),
            "max_voltage_error_pu": float(max_voltage_error),
            "rms_voltage_error_pu": float(
                np.sqrt(np.mean(np.array(voltage_errors) ** 2))
            ),
            "max_angle_error_deg": float(max_angle_error),
            "rms_angle_error_deg": float(np.sqrt(np.mean(np.array(angle_errors) ** 2))),
            "voltage_within_tolerance": bool(
                max_voltage_error <= tolerance["voltage_pu"]
            ),
            "angle_within_tolerance": bool(max_angle_error <= tolerance["angle_deg"]),
        }

    def _compare_line_results(
        self,
        gap_results: GAPPowerFlowResults,
        reference_lines: List[Dict],
        tolerance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare line power flow results."""
        ref_by_id = {line["id"]: line for line in reference_lines}

        p_errors = []
        q_errors = []
        max_p_error = 0.0
        max_q_error = 0.0

        for i, line_id in enumerate(gap_results.branch_ids):
            if int(line_id) in ref_by_id:
                ref_line = ref_by_id[int(line_id)]

                # Active power error
                p_error = abs(gap_results.p_from_mw[i] - ref_line["p_from"])
                p_errors.append(p_error)
                max_p_error = max(max_p_error, p_error)

                # Reactive power error
                q_error = abs(gap_results.q_from_mvar[i] - ref_line["q_from"])
                q_errors.append(q_error)
                max_q_error = max(max_q_error, q_error)

        return {
            "n_lines_compared": len(p_errors),
            "max_p_error_mw": float(max_p_error),
            "rms_p_error_mw": float(np.sqrt(np.mean(np.array(p_errors) ** 2))),
            "max_q_error_mvar": float(max_q_error),
            "rms_q_error_mvar": float(np.sqrt(np.mean(np.array(q_errors) ** 2))),
            "p_within_tolerance": bool(max_p_error <= tolerance["power_mw"]),
            "q_within_tolerance": bool(max_q_error <= tolerance["power_mvar"]),
        }

    def _compare_load_results(
        self,
        gap_results: GAPPowerFlowResults,
        reference_loads: List[Dict],
        tolerance: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare load consumption results."""
        ref_by_id = {load["id"]: load for load in reference_loads}

        p_errors = []
        q_errors = []
        max_p_error = 0.0
        max_q_error = 0.0

        for i, load_id in enumerate(gap_results.load_ids):
            if int(load_id) in ref_by_id:
                ref_load = ref_by_id[int(load_id)]

                # Active power error
                p_error = abs(gap_results.p_load_mw[i] - ref_load["p"])
                p_errors.append(p_error)
                max_p_error = max(max_p_error, p_error)

                # Reactive power error
                q_error = abs(gap_results.q_load_mvar[i] - ref_load["q"])
                q_errors.append(q_error)
                max_q_error = max(max_q_error, q_error)

        return {
            "n_loads_compared": len(p_errors),
            "max_p_error_mw": float(max_p_error),
            "rms_p_error_mw": float(np.sqrt(np.mean(np.array(p_errors) ** 2))),
            "max_q_error_mvar": float(max_q_error),
            "rms_q_error_mvar": float(np.sqrt(np.mean(np.array(q_errors) ** 2))),
            "p_within_tolerance": bool(max_p_error <= tolerance["power_mw"]),
            "q_within_tolerance": bool(max_q_error <= tolerance["power_mvar"]),
        }

    def _evaluate_overall_match(
        self, node_comp: Dict, line_comp: Dict, load_comp: Dict
    ) -> Dict[str, Any]:
        """Evaluate overall match quality."""
        all_within_tolerance = (
            node_comp.get("voltage_within_tolerance", False)
            and node_comp.get("angle_within_tolerance", False)
            and line_comp.get("p_within_tolerance", False)
            and line_comp.get("q_within_tolerance", False)
            and load_comp.get("p_within_tolerance", False)
            and load_comp.get("q_within_tolerance", False)
        )

        return {
            "all_within_tolerance": all_within_tolerance,
            "max_errors": {
                "voltage_pu": node_comp.get("max_voltage_error_pu", 999),
                "angle_deg": node_comp.get("max_angle_error_deg", 999),
                "power_mw": max(
                    line_comp.get("max_p_error_mw", 999),
                    load_comp.get("max_p_error_mw", 999),
                ),
                "power_mvar": max(
                    line_comp.get("max_q_error_mvar", 999),
                    load_comp.get("max_q_error_mvar", 999),
                ),
            },
        }

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Example usage and testing
if __name__ == "__main__":
    # Create example results for testing
    n_node = 5
    n_branch = 4
    n_load = 4
    n_source = 1

    example_results = GAPPowerFlowResults(
        # Convergence
        converged=True,
        n_iterations=4,
        max_mismatch=1e-8,
        calculation_time=0.005,
        # Node results
        node_ids=np.arange(n_node),
        u_pu=np.array([1.05, 1.02, 1.01, 1.00, 0.99]),
        u_angle_deg=np.array([0.0, -2.1, -4.2, -6.1, -7.8]),
        u_volt=np.array([10500, 10200, 10100, 10000, 9900]),
        # Branch results
        branch_ids=np.arange(n_branch),
        p_from_mw=np.array([2.5, 1.2, 0.8, 0.4]),
        q_from_mvar=np.array([0.8, 0.4, 0.3, 0.1]),
        p_to_mw=np.array([-2.4, -1.1, -0.7, -0.3]),
        q_to_mvar=np.array([-0.7, -0.3, -0.2, -0.1]),
        i_from_a=np.array([150, 75, 50, 25]),
        i_to_a=np.array([145, 70, 45, 20]),
        loading=np.array([0.15, 0.08, 0.05, 0.03]),
        # Load results
        load_ids=np.arange(n_load),
        p_load_mw=np.array([0.6, 0.5, 0.4, 0.3]),
        q_load_mvar=np.array([0.2, 0.15, 0.12, 0.08]),
        # Source results
        source_ids=np.array([999]),
        p_source_mw=np.array([2.6]),
        q_source_mvar=np.array([0.9]),
        # Base values
        base_power=1e6,
        base_voltage=10e3,
    )

    # Test serialization
    serializer = GAPJSONSerializer()
    test_output = Path("test_output.json")

    try:
        serializer.serialize_results(example_results, test_output)
        print(f"Successfully serialized results to {test_output}")

        # Verify the JSON is valid
        with open(test_output, "r") as f:
            data = json.load(f)
            print(f"Output contains {len(data)} main sections")
            print(f"Nodes: {len(data.get('node', []))}")
            print(f"Lines: {len(data.get('line', []))}")
            print(f"Loads: {len(data.get('sym_load', []))}")
            print(f"Sources: {len(data.get('source', []))}")

    except Exception as e:
        print(f"Error during serialization: {e}")
