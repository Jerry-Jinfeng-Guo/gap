"""
PGM Grid Generator for GAP NRPF Validation

This module adapts the Power Grid Model benchmark grid generation tools 
for use with the GAP Newton-Raphson Power Flow solver validation.

Based on generate_fictional_grid_pgm_tpf from:
https://github.com/PowerGridModel/power-grid-model-benchmark
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path
import json

# Standard grid parameters (from PGM benchmark)
U_RATED = 10e3  # 10 kV
FREQUENCY = 50.0  # Hz
SOURCE_SK = 1e20  # Short circuit power
SOURCE_RX = 0.1  # R/X ratio
SOURCE_01 = 1.0  # Zero sequence ratio
SOURCE_U_REF = 1.05  # Reference voltage (p.u.)
SOURCE_NODE = 0  # Source node ID

# Cable parameters (630Al XLPE 10kV from PGM benchmark)
CABLE_PARAM = {
    "r1": 0.063,      # Positive sequence resistance (Ohm/km)
    "x1": 0.103,      # Positive sequence reactance (Ohm/km) 
    "c1": 0.4e-6,     # Positive sequence capacitance (F/km)
    "r0": 0.156,      # Zero sequence resistance (Ohm/km)
    "x0": 0.1,        # Zero sequence reactance (Ohm/km)
    "c0": 0.66e-6,    # Zero sequence capacitance (F/km)
    "tan1": 0.0,      # Positive sequence loss tangent
    "tan0": 0.0,      # Zero sequence loss tangent
    "i_n": 1e3,       # Nominal current (A)
}


class PGMGridGenerator:
    """
    Grid generator adapted from Power Grid Model benchmark tools.
    Creates symmetric radial networks suitable for power flow validation.
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_symmetric_radial_grid(
        self,
        n_feeder: int,
        n_node_per_feeder: int,
        cable_length_km_min: float = 0.8,
        cable_length_km_max: float = 1.2,
        load_p_w_min: float = 0.4e6 * 0.8,
        load_p_w_max: float = 0.4e6 * 1.2,
        pf: float = 0.95,
        n_step: int = 1,
        load_scaling_min: float = 0.5,
        load_scaling_max: float = 1.5
    ) -> Dict[str, Any]:
        """
        Generate a symmetric radial grid network.
        
        Network topology:
        source --- source_node ---| ---line--- node ---line--- node ... (n_node_per_feeder)
                                  |              |              |
                                  |            load            load ...
                                  |
                                  | ---line--- node ---line--- node ... (feeder 2)
                                  |              |              |
                                  |            load            load ...
                                  | ... (n_feeder feeders)
        
        Args:
            n_feeder: Number of feeders
            n_node_per_feeder: Number of nodes per feeder
            cable_length_km_min: Minimum cable length (km)
            cable_length_km_max: Maximum cable length (km)
            load_p_w_min: Minimum load active power (W)
            load_p_w_max: Maximum load active power (W)
            pf: Power factor
            n_step: Number of time steps for load profiles
            load_scaling_min: Minimum load scaling factor
            load_scaling_max: Maximum load scaling factor
            
        Returns:
            Dictionary containing:
            - 'nodes': Node data (id, u_rated)
            - 'lines': Line data (id, from_node, to_node, parameters)
            - 'loads': Load data (id, node, p_specified, q_specified)
            - 'source': Source data (id, node, parameters)
            - 'load_profiles': Time series load data
            - 'metadata': Grid metadata
        """
        n_node = n_feeder * n_node_per_feeder + 1
        n_line = n_node - 1
        n_load = n_line  # One load per non-source node
        
        # Generate nodes
        nodes = self._generate_nodes(n_node)
        
        # Generate lines (radial topology)
        lines = self._generate_lines(n_node, n_feeder, n_node_per_feeder, 
                                   cable_length_km_min, cable_length_km_max)
        
        # Generate loads
        loads = self._generate_loads(n_load, load_p_w_min, load_p_w_max, pf)
        
        # Generate source
        source = self._generate_source()
        
        # Generate load profiles
        load_profiles = self._generate_load_profiles(
            n_load, n_step, load_scaling_min, load_scaling_max, loads
        )
        
        # Create metadata
        metadata = {
            'n_node': n_node,
            'n_line': n_line,
            'n_load': n_load,
            'n_feeder': n_feeder,
            'n_node_per_feeder': n_node_per_feeder,
            'seed': self.seed,
            'topology': 'radial',
            'frequency': FREQUENCY,
            'base_voltage': U_RATED
        }
        
        return {
            'nodes': nodes,
            'lines': lines,
            'loads': loads,
            'source': source,
            'load_profiles': load_profiles,
            'metadata': metadata
        }

    def _generate_nodes(self, n_node: int) -> pd.DataFrame:
        """Generate node data."""
        return pd.DataFrame({
            'id': np.arange(n_node, dtype=np.int32),
            'u_rated': np.full(n_node, U_RATED, dtype=np.float64)
        })

    def _generate_lines(self, n_node: int, n_feeder: int, n_node_per_feeder: int,
                       cable_length_km_min: float, cable_length_km_max: float) -> pd.DataFrame:
        """Generate line data with radial topology."""
        n_line = n_node - 1
        
        # Create topology: feeder structure
        # to_node: nodes 1 to n_node_per_feeder for each feeder
        to_node_feeder = np.arange(1, n_node_per_feeder + 1, dtype=np.int32)
        to_node_feeder = (
            to_node_feeder.reshape(1, -1) + 
            np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
        )
        to_node = to_node_feeder.ravel()
        
        # from_node: source node (0) connects to first node of each feeder,
        # then chain connection within each feeder
        from_node_feeder = np.arange(1, n_node_per_feeder, dtype=np.int32)
        from_node_feeder = (
            from_node_feeder.reshape(1, -1) + 
            np.arange(0, n_feeder).reshape(-1, 1) * n_node_per_feeder
        )
        from_node_feeder = np.concatenate(
            (np.zeros(shape=(n_feeder, 1), dtype=np.int32), from_node_feeder), axis=1
        )
        from_node = from_node_feeder.ravel()
        
        # Generate random line lengths
        length = self.rng.uniform(cable_length_km_min, cable_length_km_max, size=n_line)
        
        # Create line DataFrame
        lines_data = {
            'id': np.arange(n_node, n_node + n_line, dtype=np.int32),
            'from_node': from_node,
            'to_node': to_node,
            'from_status': np.ones(n_line, dtype=np.int8),
            'to_status': np.ones(n_line, dtype=np.int8),
            'length_km': length
        }
        
        # Add cable parameters (scaled by length)
        for param_name, param_value in CABLE_PARAM.items():
            if param_name in ["i_n", "tan1", "tan0"]:
                lines_data[param_name] = np.full(n_line, param_value, dtype=np.float64)
            else:
                lines_data[param_name] = param_value * length
                
        return pd.DataFrame(lines_data)

    def _generate_loads(self, n_load: int, load_p_w_min: float, 
                       load_p_w_max: float, pf: float) -> pd.DataFrame:
        """Generate symmetric load data."""
        # Generate random load powers (per phase for symmetric load)
        p_specified = self.rng.uniform(load_p_w_min / 3.0, load_p_w_max / 3.0, size=n_load)
        q_specified = p_specified * np.sqrt(1 - pf**2) / pf
        
        # Use high ID range to avoid conflicts with nodes and lines
        load_id_offset = 100000
        
        return pd.DataFrame({
            'id': np.arange(load_id_offset, load_id_offset + n_load, dtype=np.int32),
            'node': np.arange(1, n_load + 1, dtype=np.int32),  # Loads on nodes 1 to n_load
            'status': np.ones(n_load, dtype=np.int8),
            'type': np.full(n_load, 'const_power', dtype=object),
            'p_specified': p_specified,
            'q_specified': q_specified
        })

    def _generate_source(self) -> Dict[str, Any]:
        """Generate source data."""
        return {
            'id': 999999,  # High ID to avoid conflicts
            'node': SOURCE_NODE,
            'status': 1,
            'u_ref': SOURCE_U_REF,
            'sk': SOURCE_SK,
            'rx_ratio': SOURCE_RX,
            'z01_ratio': SOURCE_01
        }

    def _generate_load_profiles(self, n_load: int, n_step: int,
                              load_scaling_min: float, load_scaling_max: float,
                              loads: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate time series load profiles."""
        scaling = self.rng.uniform(load_scaling_min, load_scaling_max, size=(n_step, n_load))
        
        p_profile = loads['p_specified'].values.reshape(1, -1) * scaling
        q_profile = loads['q_specified'].values.reshape(1, -1) * scaling
        
        return {
            'time_steps': np.arange(n_step),
            'p_specified': p_profile,
            'q_specified': q_profile,
            'load_ids': loads['id'].values
        }

    def export_to_pgm_format(self, grid_data: Dict[str, Any], output_path: Path) -> None:
        """
        Export grid data to Power Grid Model JSON format.
        
        Args:
            grid_data: Grid data from generate_symmetric_radial_grid()
            output_path: Output file path for JSON
        """
        pgm_dataset = {
            'node': grid_data['nodes'].to_dict('records'),
            'line': grid_data['lines'].to_dict('records'),
            'sym_load': grid_data['loads'].to_dict('records'),
            'source': [grid_data['source']]
        }
        
        # Add load profiles as update dataset if time series data exists
        if grid_data['load_profiles']['time_steps'].size > 1:
            pgm_update = {
                'sym_load': {
                    'id': grid_data['load_profiles']['load_ids'],
                    'p_specified': grid_data['load_profiles']['p_specified'],
                    'q_specified': grid_data['load_profiles']['q_specified']
                }
            }
        else:
            pgm_update = None
        
        # Export input dataset
        with open(output_path / 'input.json', 'w') as f:
            json.dump(pgm_dataset, f, indent=2, default=self._json_serializer)
        
        # Export update dataset if exists
        if pgm_update:
            with open(output_path / 'update.json', 'w') as f:
                json.dump(pgm_update, f, indent=2, default=self._json_serializer)
        
        # Export metadata
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(grid_data['metadata'], f, indent=2, default=self._json_serializer)

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Example usage and testing
if __name__ == "__main__":
    generator = PGMGridGenerator(seed=42)
    
    # Generate a small test grid
    grid = generator.generate_symmetric_radial_grid(
        n_feeder=2,
        n_node_per_feeder=3,
        n_step=1
    )
    
    print("Generated grid summary:")
    print(f"Nodes: {len(grid['nodes'])}")
    print(f"Lines: {len(grid['lines'])}")
    print(f"Loads: {len(grid['loads'])}")
    print(f"Topology: {grid['metadata']['topology']}")
    
    # Export to test directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    generator.export_to_pgm_format(grid, test_dir)
    print(f"Exported to {test_dir}")