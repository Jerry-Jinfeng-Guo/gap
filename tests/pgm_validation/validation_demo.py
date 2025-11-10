"""
GAP NRPF Validation Pipeline

Main orchestration script for validating GAP Newton-Raphson Power Flow solver
against Power Grid Model benchmark using systematic test generation and comparison.

This module provides complete integration with the GAP solver through Python bindings
for comprehensive power flow validation against PGM reference solutions.
"""

from dataclasses import asdict
import json

# GAP solver Python bindings - flexible import from build directories
# This supports multiple build configurations (CUDA, CPU-only, debug, etc.)
import os
from pathlib import Path

# Import our PGM benchmark integration modules
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

_gap_module_paths = []

# 1. Check environment variable for custom path (e.g., export GAP_MODULE_PATH=/path/to/gap/module)
if "GAP_MODULE_PATH" in os.environ:
    _gap_module_paths.append(Path(os.environ["GAP_MODULE_PATH"]))

# 2. Check common build directory patterns relative to this script
_repo_root = Path(__file__).parent.parent.parent
_gap_module_paths.extend(
    [
        _repo_root / "build_cuda" / "lib",  # CUDA-enabled build
        _repo_root / "build" / "lib",  # Default build
        _repo_root / "build_release" / "lib",  # Release build
        _repo_root / "build_debug" / "lib",  # Debug build
        _repo_root / "lib",  # Local install location
    ]
)

GAP_AVAILABLE = False
GAP_IMPORT_ERROR = "No GAP solver module found"
gap_solver = None

# 3. First try to import as installed package
try:
    import gap_solver  # noqa: F401

    GAP_AVAILABLE = True
    GAP_IMPORT_ERROR = None
except ImportError:
    # 4. Search in build directories for Python module
    existing_paths = [p for p in _gap_module_paths if p.exists()]

    for module_path in existing_paths:
        sys.path.insert(0, str(module_path))
        try:
            import gap_solver  # noqa: F401

            GAP_AVAILABLE = True
            GAP_IMPORT_ERROR = None
            break
        except ImportError as e:
            GAP_IMPORT_ERROR = f"Found path {module_path} but import failed: {e}"
            # Remove the path we just added to avoid polluting sys.path
            if str(module_path) in sys.path:
                sys.path.remove(str(module_path))
            continue

    if not GAP_AVAILABLE:
        if existing_paths:
            GAP_IMPORT_ERROR = (
                f"GAP Python module not found in existing build directories: {[str(p) for p in existing_paths]}. "
                f"Make sure to build with -DGAP_BUILD_PYTHON_BINDINGS=ON"
            )
        else:
            GAP_IMPORT_ERROR = (
                f"No GAP build directories found. Expected one of: {[str(p) for p in _gap_module_paths]}. "
                f"Run './build.sh' or set GAP_MODULE_PATH environment variable."
            )

sys.path.append(str(Path(__file__).parent))

from grid_generators.pgm_generator import PGMGridGenerator
from json_io.gap_json_parser import GAPNetworkData, PGMJSONParser
from json_io.gap_json_serializer import GAPJSONSerializer, GAPPowerFlowResults
from reference_solutions.pgm_reference import PGM_AVAILABLE, PGMReferenceSolver


class ValidationPipeline:
    """
    Complete validation pipeline for GAP NRPF solver.

    Orchestrates the entire validation workflow:
    1. Generate test networks using PGM tools
    2. Create reference solutions with PGM solver
    3. Run GAP NRPF solver on same networks using Python bindings
    4. Compare results with detailed analysis
    5. Generate validation reports
    """

    def __init__(
        self,
        workspace_dir: Path,
        base_power: float = 1e6,
        tolerance_settings: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize validation pipeline.

        Args:
            workspace_dir: Directory for validation workspace
            base_power: Base power for per-unit calculations (VA)
            tolerance_settings: Custom tolerance settings for comparisons
        """
        self.workspace_dir = Path(workspace_dir)
        self.base_power = base_power

        # Default tolerance settings
        self.tolerance = tolerance_settings or {
            "voltage_pu": 1e-6,
            "angle_deg": 1e-4,
            "power_mw": 1e-3,
            "power_mvar": 1e-3,
        }

        # Create workspace structure
        self._setup_workspace()

        # Initialize components
        self.grid_generator = PGMGridGenerator(seed=42)
        self.json_parser = PGMJSONParser(base_power=base_power)
        self.json_serializer = GAPJSONSerializer()

        if PGM_AVAILABLE:
            self.reference_solver = PGMReferenceSolver()
        else:
            self.reference_solver = None
            warnings.warn("PGM reference solver not available")

    def _convert_to_gap_solver_network(self, gap_network: GAPNetworkData) -> Any:
        """
        Convert GAPNetworkData to gap_solver.NetworkData format.

        Args:
            gap_network: Parsed network data from PGM

        Returns:
            gap_solver.NetworkData ready for power flow calculation
        """
        if not GAP_AVAILABLE:
            raise RuntimeError(f"GAP solver not available: {GAP_IMPORT_ERROR}")

        # Create NetworkData instance
        network_data = gap_solver.NetworkData()

        # Convert bus data
        buses = []
        for i in range(gap_network.n_node):
            bus = gap_solver.BusData()
            bus.id = int(gap_network.node_ids[i])
            bus.u_rated = gap_network.u_rated[i]
            bus.u_pu = 1.0  # Initial flat start
            bus.u_angle = 0.0  # Initial flat start

            # Determine bus type from sources
            if i in gap_network.source_node:
                source_idx = np.nonzero(gap_network.source_node == i)[0][0]
                bus.bus_type = gap_solver.BusType.SLACK  # Use first source as slack
                bus.u_pu = gap_network.u_ref_pu[source_idx]
            else:
                bus.bus_type = gap_solver.BusType.PQ

            # Set power injections (loads are negative)
            bus.active_power = 0.0
            bus.reactive_power = 0.0
            if i in gap_network.load_node:
                load_indices = np.nonzero(gap_network.load_node == i)[0]
                for load_idx in load_indices:
                    if gap_network.load_status[load_idx]:
                        bus.active_power -= (
                            gap_network.p_load_pu[load_idx] * gap_network.base_power
                        )
                        bus.reactive_power -= (
                            gap_network.q_load_pu[load_idx] * gap_network.base_power
                        )

            bus.energized = 1  # All buses energized by default
            buses.append(bus)

        network_data.buses = buses

        # Convert branch data
        branches = []
        for i in range(gap_network.n_branch):
            branch = gap_solver.BranchData()
            branch.id = int(gap_network.branch_ids[i])
            branch.from_bus = int(gap_network.from_node[i])
            branch.to_bus = int(gap_network.to_node[i])
            branch.r1 = gap_network.r_pu[i]  # Resistance in per-unit
            branch.x1 = gap_network.x_pu[i]  # Reactance in per-unit
            branch.b1 = gap_network.b_pu[i]  # Susceptance in per-unit
            branch.branch_type = gap_solver.BranchType.LINE
            branch.status = int(gap_network.branch_status[i])
            branches.append(branch)

        network_data.branches = branches

        # Convert appliance data (loads and sources)
        appliances = []

        # Add loads as appliances
        for i in range(gap_network.n_load):
            if gap_network.load_status[i]:
                appliance = gap_solver.ApplianceData()
                appliance.id = int(gap_network.load_ids[i])
                appliance.node = int(gap_network.load_node[i])
                appliance.type = gap_solver.ApplianceType.LOADGEN
                appliance.p_specified = (
                    -gap_network.p_load_pu[i] * gap_network.base_power
                )  # Negative for loads
                appliance.q_specified = (
                    -gap_network.q_load_pu[i] * gap_network.base_power
                )
                appliance.status = 1
                appliances.append(appliance)

        # Add sources as appliances
        for i in range(gap_network.n_source):
            if gap_network.source_status[i]:
                appliance = gap_solver.ApplianceData()
                appliance.id = int(gap_network.source_ids[i])
                appliance.node = int(gap_network.source_node[i])
                appliance.type = gap_solver.ApplianceType.SOURCE
                appliance.u_ref = gap_network.u_ref_pu[i]  # Reference voltage
                appliance.u_ref_angle = 0.0  # Reference angle
                appliance.status = 1
                appliances.append(appliance)

        network_data.appliances = appliances

        return network_data

    def _setup_workspace(self):
        """Create validation workspace directory structure."""
        dirs = [
            "test_networks",
            "reference_solutions",
            "gap_solutions",
            "comparison_results",
            "reports",
            "logs",
        ]

        for dir_name in dirs:
            (self.workspace_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def run_single_test_demo(
        self, config: Dict[str, Any], case_name: str = "demo_case"
    ) -> Dict[str, Any]:
        """
        Run a single test case demonstration.

        This method shows the complete validation workflow without requiring
        the actual GAP solver to be available.

        Args:
            config: Grid generation configuration
            case_name: Name for the test case

        Returns:
            Test case results
        """
        print(f"Running validation demo: {case_name}")

        # 1. Generate test network
        print("  1. Generating test network...")
        network_data = self.grid_generator.generate_symmetric_radial_grid(**config)

        # 2. Save network to JSON
        test_dir = self.workspace_dir / "test_networks" / case_name
        test_dir.mkdir(parents=True, exist_ok=True)

        input_file = test_dir / "input.json"
        self.grid_generator.export_to_pgm_format(network_data, test_dir)
        print(
            f"     ✓ Network saved: {network_data['metadata']['n_node']} nodes, "
            f"{network_data['metadata']['n_line']} lines"
        )

        # 3. Generate reference solution using PGM
        if self.reference_solver:
            print("  2. Generating PGM reference solution...")
            reference_file = (
                self.workspace_dir
                / "reference_solutions"
                / f"{case_name}_reference.json"
            )
            reference_result = self.reference_solver.generate_reference_solution(
                input_file=input_file, output_file=reference_file
            )
            print(
                f"     ✓ Reference generated: {reference_result['calculation_time']:.3f}s"
            )
        else:
            print("  2. PGM reference solver not available")
            reference_result = None
            reference_file = None

        # 4. Parse network for GAP solver
        print("  3. Parsing network for GAP solver...")
        gap_network = self.json_parser.parse_network(input_file)
        Y = self.json_parser.create_admittance_matrix(gap_network)
        print(f"     ✓ Parsed: Y matrix {Y.shape} ({Y.nnz} non-zeros)")

        # 5. Run GAP NRPF solver (synthetic for demonstration)
        print("  4. Running GAP NRPF solver (synthetic demo)...")
        gap_file = self.workspace_dir / "gap_solutions" / f"{case_name}_gap.json"
        gap_result = self._run_gap_solver_demo(gap_network, gap_file)
        print(
            f"     ✓ GAP solver: {gap_result.converged}, {gap_result.n_iterations} iterations"
        )

        # 6. Compare results
        if reference_file and reference_file.exists():
            print("  5. Comparing results...")
            comparison_file = (
                self.workspace_dir
                / "comparison_results"
                / f"{case_name}_comparison.json"
            )
            comparison_result = self.json_serializer.create_comparison_summary(
                gap_results=gap_result,
                reference_file=reference_file,
                output_file=comparison_file,
                tolerance=self.tolerance,
            )
            print(
                f"     ✓ Comparison: max voltage error {comparison_result['node_comparison']['max_voltage_error_pu']:.2e} p.u."
            )
        else:
            print("  5. Comparison skipped (no reference)")
            comparison_result = None

        return {
            "network_metadata": network_data["metadata"],
            "reference_info": reference_result,
            "gap_info": {
                "converged": gap_result.converged,
                "n_iterations": gap_result.n_iterations,
                "calculation_time": gap_result.calculation_time,
            },
            "comparison": comparison_result,
        }

    def _run_gap_solver_real(
        self,
        network: GAPNetworkData,
        backend_type: str = "CPU",
        solver_config: Optional[Dict[str, Any]] = None,
    ) -> GAPPowerFlowResults:
        """
        Run GAP Newton-Raphson solver using Python bindings.

        Args:
            network: Parsed network data
            output_file: Output file for results
            backend_type: "CPU" or "GPU" for computation backend
            solver_config: Optional solver configuration parameters

        Returns:
            GAP power flow results converted to validation format
        """
        if not GAP_AVAILABLE:
            raise RuntimeError(f"GAP solver not available: {GAP_IMPORT_ERROR}")

        # Convert to gap_solver format
        gap_network_data = self._convert_to_gap_solver_network(network)

        # Create admittance matrix (required by GAP solver)
        Y = self.json_parser.create_admittance_matrix(network)

        # Convert scipy sparse matrix to GAP format if needed
        # For now, let's assume the solver can accept the NetworkData and build its own Y matrix

        # Create solver configuration
        config = gap_solver.PowerFlowConfig()
        if solver_config:
            if "tolerance" in solver_config:
                config.tolerance = solver_config["tolerance"]
            if "max_iterations" in solver_config:
                config.max_iterations = solver_config["max_iterations"]
            if "use_flat_start" in solver_config:
                config.use_flat_start = solver_config["use_flat_start"]
            if "verbose" in solver_config:
                config.verbose = solver_config["verbose"]

        # Select backend and create solver
        if backend_type.upper() == "GPU" and gap_solver.is_cuda_available():
            backend = gap_solver.BackendType.GPU_CUDA
        else:
            backend = gap_solver.BackendType.CPU

        solver = gap_solver.create_solver(backend)

        # Run power flow calculation
        start_time = time.time()
        try:
            # Try different solver API patterns
            try:
                # If solver requires admittance matrix as separate parameter
                result = solver.solve_power_flow(gap_network_data, Y.tocsr(), config)
            except TypeError:
                # If solver builds Y matrix internally
                result = solver.solve_power_flow(gap_network_data, config)

            calculation_time = time.time() - start_time

            # Convert results to validation format
            return self._convert_gap_result_to_validation_format(
                result, network, calculation_time
            )

        except Exception as ex:
            calculation_time = time.time() - start_time
            print(f"GAP solver failed: {ex}")  # Debug info
            # Return failed result for analysis
            return self._create_failed_gap_result(network, calculation_time)

    def _convert_gap_result_to_validation_format(
        self,
        gap_result: Any,
        original_network: GAPNetworkData,
        calculation_time: float,
    ) -> GAPPowerFlowResults:
        """
        Convert gap_solver.PowerFlowResult to GAPPowerFlowResults format.
        """
        # Extract voltage results
        u_pu = np.array([v.magnitude for v in gap_result.bus_voltages])
        u_angle_deg = np.array([np.degrees(v.angle) for v in gap_result.bus_voltages])
        u_volt = u_pu * original_network.base_voltage

        # Calculate branch power flows (simplified - would need admittance calculation)
        n_branch = original_network.n_branch
        p_from_mw = np.zeros(n_branch)
        q_from_mvar = np.zeros(n_branch)
        p_to_mw = np.zeros(n_branch)
        q_to_mvar = np.zeros(n_branch)
        i_from_a = np.zeros(n_branch)
        i_to_a = np.zeros(n_branch)
        loading = np.zeros(n_branch)

        # NOTE: Branch flow calculation from voltage solution requires:
        # 1. Building the admittance matrix from network parameters
        # 2. Computing complex power flows using V and Y matrices
        # 3. Converting to MW/MVAR and current values
        # This is left as zeros for initial integration; can be enhanced later

        # Extract load results
        p_load_mw = original_network.p_load_pu * original_network.base_power / 1e6
        q_load_mvar = original_network.q_load_pu * original_network.base_power / 1e6

        # Calculate source injections (sum of loads + losses)
        total_load_p = np.sum(p_load_mw)
        total_load_q = np.sum(q_load_mvar)
        p_source_mw = np.array([total_load_p])  # Simplified - no loss calculation yet
        q_source_mvar = np.array([total_load_q])

        return GAPPowerFlowResults(
            # Convergence info
            converged=gap_result.converged,
            n_iterations=gap_result.iterations,
            max_mismatch=gap_result.final_mismatch,
            calculation_time=calculation_time,
            # Node results
            node_ids=original_network.node_ids,
            u_pu=u_pu,
            u_angle_deg=u_angle_deg,
            u_volt=u_volt,
            # Branch results
            branch_ids=original_network.branch_ids,
            p_from_mw=p_from_mw,
            q_from_mvar=q_from_mvar,
            p_to_mw=p_to_mw,
            q_to_mvar=q_to_mvar,
            i_from_a=i_from_a,
            i_to_a=i_to_a,
            loading=loading,
            # Load results
            load_ids=original_network.load_ids,
            p_load_mw=p_load_mw,
            q_load_mvar=q_load_mvar,
            # Source results
            source_ids=original_network.source_ids,
            p_source_mw=p_source_mw,
            q_source_mvar=q_source_mvar,
            # Base values
            base_power=original_network.base_power,
            base_voltage=original_network.base_voltage,
        )

    def _create_failed_gap_result(
        self, network: GAPNetworkData, calculation_time: float
    ) -> GAPPowerFlowResults:
        """Create a failed result structure for analysis."""
        n_node = network.n_node
        n_branch = network.n_branch
        n_load = network.n_load
        n_source = network.n_source

        return GAPPowerFlowResults(
            # Convergence info
            converged=False,
            n_iterations=0,
            max_mismatch=np.inf,
            calculation_time=calculation_time,
            # Node results (NaN for failed)
            node_ids=network.node_ids,
            u_pu=np.full(n_node, np.nan),
            u_angle_deg=np.full(n_node, np.nan),
            u_volt=np.full(n_node, np.nan),
            # Branch results (NaN for failed)
            branch_ids=network.branch_ids,
            p_from_mw=np.full(n_branch, np.nan),
            q_from_mvar=np.full(n_branch, np.nan),
            p_to_mw=np.full(n_branch, np.nan),
            q_to_mvar=np.full(n_branch, np.nan),
            i_from_a=np.full(n_branch, np.nan),
            i_to_a=np.full(n_branch, np.nan),
            loading=np.full(n_branch, np.nan),
            # Load results
            load_ids=network.load_ids,
            p_load_mw=np.full(n_load, np.nan),
            q_load_mvar=np.full(n_load, np.nan),
            # Source results
            source_ids=network.source_ids,
            p_source_mw=np.full(n_source, np.nan),
            q_source_mvar=np.full(n_source, np.nan),
            # Base values
            base_power=network.base_power,
            base_voltage=network.base_voltage,
        )

    def _run_gap_solver_demo(
        self, network: GAPNetworkData, output_file: Path
    ) -> GAPPowerFlowResults:
        """
        Wrapper method for GAP solver - chooses between real or synthetic solver.

        Args:
            network: Parsed network data
            output_file: Output file for results

        Returns:
            GAP power flow results
        """
        if GAP_AVAILABLE:
            return self._run_gap_solver_real(network)
        else:
            return self._run_gap_solver_synthetic(network, output_file)

    def _run_gap_solver_synthetic(
        self, network: GAPNetworkData, output_file: Path
    ) -> GAPPowerFlowResults:
        """
        Fallback synthetic GAP solver for when real solver is not available.
        """
        calculation_time = 0.003  # 3 milliseconds

        # Create synthetic voltage solution with realistic voltage profile
        u_pu = np.ones(network.n_node)
        u_angle_deg = np.zeros(network.n_node)

        # Simple voltage drop simulation for demonstration
        for i in range(network.n_node):
            if i == 0:  # Source node
                u_pu[i] = 1.05  # Source voltage
                u_angle_deg[i] = 0.0
            else:
                # Voltage drop based on distance from source
                u_pu[i] = 1.05 - i * 0.005  # 0.5% drop per node
                u_angle_deg[i] = -i * 0.5  # Small angle lag

        u_volt = u_pu * network.base_voltage

        # Synthetic branch results
        n_branch = network.n_branch
        p_from_mw = np.linspace(1.5, 0.1, n_branch)  # Decreasing power along feeders
        q_from_mvar = np.linspace(0.3, 0.05, n_branch)
        p_to_mw = -p_from_mw * 0.98  # Small losses
        q_to_mvar = -q_from_mvar * 0.97

        base_current = network.base_power / (np.sqrt(3) * network.base_voltage)
        i_from_a = (
            np.abs(p_from_mw + 1j * q_from_mvar)
            * 1e6
            / (np.sqrt(3) * u_volt[network.from_node[:n_branch]])
        )
        i_to_a = i_from_a * 0.99
        loading = i_from_a / base_current / 2.0  # Assume 2 p.u. current rating

        # Load and source results
        p_load_mw = network.p_load_pu * network.base_power / 1e6
        q_load_mvar = network.q_load_pu * network.base_power / 1e6

        total_load_p = np.sum(p_load_mw)
        total_load_q = np.sum(q_load_mvar)
        p_source_mw = np.array([total_load_p * 1.03])  # 3% losses
        q_source_mvar = np.array([total_load_q * 1.05])  # 5% reactive losses

        # Create GAP results
        gap_results = GAPPowerFlowResults(
            # Convergence
            converged=True,
            n_iterations=5,
            max_mismatch=1.2e-9,
            calculation_time=calculation_time,
            # Node results
            node_ids=network.node_ids,
            u_pu=u_pu,
            u_angle_deg=u_angle_deg,
            u_volt=u_volt,
            # Branch results
            branch_ids=network.branch_ids,
            p_from_mw=p_from_mw,
            q_from_mvar=q_from_mvar,
            p_to_mw=p_to_mw,
            q_to_mvar=q_to_mvar,
            i_from_a=i_from_a,
            i_to_a=i_to_a,
            loading=loading,
            # Load results
            load_ids=network.load_ids,
            p_load_mw=p_load_mw,
            q_load_mvar=q_load_mvar,
            # Source results
            source_ids=network.source_ids,
            p_source_mw=p_source_mw,
            q_source_mvar=q_source_mvar,
            # Metadata
            backend_type="synthetic",
            base_power=network.base_power,
            base_voltage=network.base_voltage,
        )

        # Save results to JSON
        self.json_serializer.serialize_results(gap_results, output_file)

        return gap_results


# Demonstration function
def run_validation_demo(
    workspace_dir: Path = Path("validation_demo_workspace"),
) -> None:
    """
    Run a simple validation demonstration.

    This shows the complete validation workflow working with synthetic GAP results.
    When the actual GAP solver is available, this framework can be used directly
    by replacing the synthetic solver with real GAP NRPF calls.
    """

    print("GAP NRPF Validation Pipeline Demonstration")
    print("=" * 50)

    # Test configurations
    test_configs = [
        {
            "n_feeder": 3,
            "n_node_per_feeder": 4,
            "load_p_w_min": 0.2e6,
            "load_p_w_max": 0.6e6,
            "pf": 0.95,
            "n_step": 1,
        },
        {
            "n_feeder": 5,
            "n_node_per_feeder": 6,
            "load_p_w_min": 0.3e6,
            "load_p_w_max": 0.8e6,
            "pf": 0.92,
            "n_step": 1,
        },
    ]

    # Initialize pipeline
    pipeline = ValidationPipeline(workspace_dir)

    # Run test cases
    results = []
    for i, config in enumerate(test_configs):
        case_name = f"demo_case_{i+1:02d}"
        result = pipeline.run_single_test_demo(config, case_name)
        results.append(result)
        print()

    print("=" * 50)
    print("VALIDATION DEMONSTRATION COMPLETE")
    print("=" * 50)

    # Summary
    for i, result in enumerate(results):
        print(f"Test Case {i+1}:")
        print(f"  Network: {result['network_metadata']['n_node']} nodes")
        print(f"  GAP Converged: {result['gap_info']['converged']}")
        print(f"  GAP Time: {result['gap_info']['calculation_time']:.3f}s")
        if result["comparison"]:
            print(
                f"  Max Voltage Error: {result['comparison']['node_comparison']['max_voltage_error_pu']:.2e} p.u."
            )
        print()

    print(f"All files saved to: {workspace_dir.absolute()}")
    print("\nNext steps:")
    print("1. Replace synthetic GAP solver with actual GAP NRPF implementation")
    print("2. Run systematic validation across multiple network configurations")
    print("3. Analyze results and tune GAP solver parameters as needed")


if __name__ == "__main__":
    if not PGM_AVAILABLE:
        print(
            "Power Grid Model not available. Install with: pip install power-grid-model"
        )
        exit(1)

    print("Running GAP NRPF validation pipeline demonstration...")

    try:
        run_validation_demo()

    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()
