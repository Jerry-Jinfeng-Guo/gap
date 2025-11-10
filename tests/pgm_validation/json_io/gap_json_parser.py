"""
JSON Input Parser for GAP NRPF Solver

This module parses Power Grid Model JSON format input data and converts it
to GAP internal data structures for Newton-Raphson power flow calculation.
"""

import json
import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

# Import GAP types if available
try:
    from ...cpu.power_flow.power_system import PowerSystem
    from ...cpu.power_flow.network_data import NetworkData
    from ...types import Float, Complex, Index
except ImportError:
    # Fallback type definitions for standalone usage
    Float = np.float64
    Complex = np.complex128
    Index = np.int32


@dataclass
class GAPNetworkData:
    """
    GAP network data structure compatible with Newton-Raphson solver.
    
    This structure contains all the data needed for power flow calculation
    in the format expected by the GAP NRPF implementation.
    """
    # Network topology
    n_node: int
    n_branch: int
    n_load: int
    n_source: int
    
    # Node data
    node_ids: np.ndarray  # [n_node] Node IDs
    u_rated: np.ndarray   # [n_node] Rated voltages (V)
    
    # Branch data
    branch_ids: np.ndarray     # [n_branch] Branch IDs
    from_node: np.ndarray      # [n_branch] From node indices
    to_node: np.ndarray        # [n_branch] To node indices
    branch_status: np.ndarray  # [n_branch] Branch status (1=in service)
    
    # Branch parameters
    r_pu: np.ndarray          # [n_branch] Resistance (p.u.)
    x_pu: np.ndarray          # [n_branch] Reactance (p.u.)
    b_pu: np.ndarray          # [n_branch] Susceptance (p.u.)
    
    # Load data
    load_ids: np.ndarray      # [n_load] Load IDs
    load_node: np.ndarray     # [n_load] Load node indices
    load_status: np.ndarray   # [n_load] Load status (1=in service)
    p_load_pu: np.ndarray     # [n_load] Active power (p.u.)
    q_load_pu: np.ndarray     # [n_load] Reactive power (p.u.)
    
    # Source data
    source_ids: np.ndarray    # [n_source] Source IDs
    source_node: np.ndarray   # [n_source] Source node indices
    source_status: np.ndarray # [n_source] Source status (1=in service)
    u_ref_pu: np.ndarray      # [n_source] Reference voltage (p.u.)
    
    # Base values
    base_power: float         # Base power (VA)
    base_voltage: float       # Base voltage (V)
    base_impedance: float     # Base impedance (Ohm)
    frequency: float          # System frequency (Hz)


class PGMJSONParser:
    """
    Parser to convert Power Grid Model JSON input to GAP network data.
    
    Handles the conversion from PGM standardized JSON format to GAP's
    internal data structures, including per-unit conversion and node indexing.
    """
    
    def __init__(self, base_power: float = 1e6):  # 1 MVA base
        """
        Initialize parser with base values.
        
        Args:
            base_power: Base power for per-unit conversion (VA)
        """
        self.base_power = base_power
        
    def parse_network(self, input_file: Path, update_file: Optional[Path] = None) -> GAPNetworkData:
        """
        Parse PGM JSON input file to GAP network data.
        
        Args:
            input_file: Path to input.json file
            update_file: Optional path to update.json file for time series data
            
        Returns:
            GAPNetworkData structure ready for NRPF calculation
        """
        # Load JSON data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
            
        update_data = None
        if update_file and update_file.exists():
            with open(update_file, 'r') as f:
                update_data = json.load(f)
        
        # Parse components
        nodes_data = self._parse_nodes(input_data.get('node', []))
        lines_data = self._parse_lines(input_data.get('line', []), nodes_data)
        loads_data = self._parse_loads(input_data.get('sym_load', []), nodes_data, update_data)
        sources_data = self._parse_sources(input_data.get('source', []), nodes_data)
        
        # Calculate base values
        base_voltage = nodes_data['u_rated'].max()  # Use max voltage as base
        base_impedance = base_voltage**2 / self.base_power
        
        # Convert to per-unit
        lines_pu = self._convert_lines_to_pu(lines_data, base_impedance)
        loads_pu = self._convert_loads_to_pu(loads_data, self.base_power)
        sources_pu = self._convert_sources_to_pu(sources_data)
        
        # Create GAP network data structure
        return GAPNetworkData(
            # Topology
            n_node=len(nodes_data['id']),
            n_branch=len(lines_data['id']),
            n_load=len(loads_data['id']),
            n_source=len(sources_data['id']),
            
            # Node data
            node_ids=nodes_data['id'],
            u_rated=nodes_data['u_rated'],
            
            # Branch data
            branch_ids=lines_data['id'],
            from_node=lines_data['from_node_idx'],
            to_node=lines_data['to_node_idx'],
            branch_status=lines_data['status'],
            r_pu=lines_pu['r'],
            x_pu=lines_pu['x'],
            b_pu=lines_pu['b'],
            
            # Load data
            load_ids=loads_data['id'],
            load_node=loads_data['node_idx'],
            load_status=loads_data['status'],
            p_load_pu=loads_pu['p'],
            q_load_pu=loads_pu['q'],
            
            # Source data
            source_ids=sources_data['id'],
            source_node=sources_data['node_idx'],
            source_status=sources_data['status'],
            u_ref_pu=sources_pu['u_ref'],
            
            # Base values
            base_power=self.base_power,
            base_voltage=base_voltage,
            base_impedance=base_impedance,
            frequency=50.0  # Default 50 Hz
        )
    
    def _parse_nodes(self, nodes_json: List[Dict]) -> Dict[str, np.ndarray]:
        """Parse node data from JSON."""
        if not nodes_json:
            raise ValueError("No nodes found in input data")
            
        node_df = pd.DataFrame(nodes_json)
        
        # Create node ID to index mapping (important for GAP indexing)
        node_ids = node_df['id'].values.astype(Index)
        sorted_indices = np.argsort(node_ids)
        
        return {
            'id': node_ids[sorted_indices],
            'u_rated': node_df['u_rated'].values[sorted_indices].astype(Float),
            'id_to_idx': {node_id: idx for idx, node_id in enumerate(node_ids[sorted_indices])}
        }
    
    def _parse_lines(self, lines_json: List[Dict], nodes_data: Dict) -> Dict[str, np.ndarray]:
        """Parse line/branch data from JSON."""
        if not lines_json:
            raise ValueError("No lines found in input data")
            
        line_df = pd.DataFrame(lines_json)
        id_to_idx = nodes_data['id_to_idx']
        
        # Map node IDs to indices
        from_node_idx = np.array([id_to_idx[node_id] for node_id in line_df['from_node']], dtype=Index)
        to_node_idx = np.array([id_to_idx[node_id] for node_id in line_df['to_node']], dtype=Index)
        
        # Calculate status (both from and to must be active)
        status = (line_df.get('from_status', 1).values * 
                 line_df.get('to_status', 1).values).astype(Index)
        
        return {
            'id': line_df['id'].values.astype(Index),
            'from_node_idx': from_node_idx,
            'to_node_idx': to_node_idx,
            'status': status,
            'r1': line_df['r1'].values.astype(Float),
            'x1': line_df['x1'].values.astype(Float),
            'c1': line_df.get('c1', np.zeros(len(line_df))).values.astype(Float),
            'length_km': line_df.get('length_km', np.ones(len(line_df))).values.astype(Float)
        }
    
    def _parse_loads(self, loads_json: List[Dict], nodes_data: Dict, 
                    update_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Parse load data from JSON."""
        if not loads_json:
            # No loads - create empty arrays
            return {
                'id': np.array([], dtype=Index),
                'node_idx': np.array([], dtype=Index),
                'status': np.array([], dtype=Index),
                'p_specified': np.array([], dtype=Float),
                'q_specified': np.array([], dtype=Float)
            }
            
        load_df = pd.DataFrame(loads_json)
        id_to_idx = nodes_data['id_to_idx']
        
        # Map load nodes to indices
        node_idx = np.array([id_to_idx[node_id] for node_id in load_df['node']], dtype=Index)
        
        # Get load values (use update data if available)
        p_specified = load_df['p_specified'].values.astype(Float)
        q_specified = load_df['q_specified'].values.astype(Float)
        
        if update_data and 'sym_load' in update_data:
            update_loads = update_data['sym_load']
            # Use first time step if time series data
            if 'p_specified' in update_loads:
                p_update = np.array(update_loads['p_specified'])
                if p_update.ndim > 1:
                    p_specified = p_update[0, :]  # First time step
                else:
                    p_specified = p_update
                    
            if 'q_specified' in update_loads:
                q_update = np.array(update_loads['q_specified'])
                if q_update.ndim > 1:
                    q_specified = q_update[0, :]  # First time step
                else:
                    q_specified = q_update
        
        return {
            'id': load_df['id'].values.astype(Index),
            'node_idx': node_idx,
            'status': load_df.get('status', np.ones(len(load_df))).values.astype(Index),
            'p_specified': p_specified,
            'q_specified': q_specified
        }
    
    def _parse_sources(self, sources_json: List[Dict], nodes_data: Dict) -> Dict[str, np.ndarray]:
        """Parse source data from JSON."""
        if not sources_json:
            raise ValueError("No sources found in input data")
            
        source_df = pd.DataFrame(sources_json)
        id_to_idx = nodes_data['id_to_idx']
        
        # Map source nodes to indices
        node_idx = np.array([id_to_idx[node_id] for node_id in source_df['node']], dtype=Index)
        
        return {
            'id': source_df['id'].values.astype(Index),
            'node_idx': node_idx,
            'status': source_df.get('status', np.ones(len(source_df))).values.astype(Index),
            'u_ref': source_df['u_ref'].values.astype(Float),
            'sk': source_df.get('sk', np.full(len(source_df), 1e20)).values.astype(Float),
            'rx_ratio': source_df.get('rx_ratio', np.full(len(source_df), 0.1)).values.astype(Float)
        }
    
    def _convert_lines_to_pu(self, lines_data: Dict, base_impedance: float) -> Dict[str, np.ndarray]:
        """Convert line parameters to per-unit."""
        r_pu = lines_data['r1'] / base_impedance
        x_pu = lines_data['x1'] / base_impedance
        
        # Susceptance (capacitive) - convert C to B
        omega = 2 * np.pi * 50.0  # 50 Hz
        b_pu = omega * lines_data['c1'] * base_impedance
        
        return {
            'r': r_pu.astype(Float),
            'x': x_pu.astype(Float),
            'b': b_pu.astype(Float)
        }
    
    def _convert_loads_to_pu(self, loads_data: Dict, base_power: float) -> Dict[str, np.ndarray]:
        """Convert load values to per-unit."""
        return {
            'p': (loads_data['p_specified'] / base_power).astype(Float),
            'q': (loads_data['q_specified'] / base_power).astype(Float)
        }
    
    def _convert_sources_to_pu(self, sources_data: Dict) -> Dict[str, np.ndarray]:
        """Convert source values to per-unit."""
        return {
            'u_ref': (sources_data['u_ref']).astype(Float)  # Already in p.u.
        }
    
    def create_admittance_matrix(self, network: GAPNetworkData) -> sp.csc_matrix:
        """
        Create nodal admittance matrix from network data.
        
        Args:
            network: GAP network data
            
        Returns:
            Complex admittance matrix [n_node x n_node]
        """
        n_node = network.n_node
        Y = sp.lil_matrix((n_node, n_node), dtype=Complex)
        
        # Add branch impedances
        for i in range(network.n_branch):
            if network.branch_status[i] == 0:
                continue  # Skip out-of-service branches
                
            from_idx = network.from_node[i]
            to_idx = network.to_node[i]
            
            # Branch impedance
            z = complex(network.r_pu[i], network.x_pu[i])
            if abs(z) < 1e-12:
                raise ValueError(f"Branch {i} has zero impedance")
                
            y = 1.0 / z
            
            # Shunt admittance (half at each end)
            y_shunt = complex(0, network.b_pu[i] / 2.0)
            
            # Fill admittance matrix
            Y[from_idx, to_idx] -= y
            Y[to_idx, from_idx] -= y
            Y[from_idx, from_idx] += y + y_shunt
            Y[to_idx, to_idx] += y + y_shunt
        
        return Y.tocsc()
    
    def create_power_injection_vector(self, network: GAPNetworkData) -> np.ndarray:
        """
        Create power injection vector (S = P + jQ).
        
        Args:
            network: GAP network data
            
        Returns:
            Complex power injection vector [n_node]
        """
        S = np.zeros(network.n_node, dtype=Complex)
        
        # Add loads (negative injection)
        for i in range(network.n_load):
            if network.load_status[i] == 0:
                continue  # Skip out-of-service loads
                
            node_idx = network.load_node[i]
            S[node_idx] -= complex(network.p_load_pu[i], network.q_load_pu[i])
        
        return S


# Example usage and testing
if __name__ == "__main__":
    from pathlib import Path
    
    # Test the parser
    parser = PGMJSONParser(base_power=1e6)  # 1 MVA base
    
    # Example: parse a test file
    test_dir = Path("test_data")
    if (test_dir / "input.json").exists():
        try:
            network = parser.parse_network(test_dir / "input.json")
            print("Successfully parsed network:")
            print(f"  Nodes: {network.n_node}")
            print(f"  Branches: {network.n_branch}")
            print(f"  Loads: {network.n_load}")
            print(f"  Sources: {network.n_source}")
            print(f"  Base power: {network.base_power/1e6:.1f} MVA")
            print(f"  Base voltage: {network.base_voltage/1e3:.1f} kV")
            
            # Test admittance matrix creation
            Y = parser.create_admittance_matrix(network)
            print(f"  Admittance matrix: {Y.shape} ({Y.nnz} non-zeros)")
            
            # Test power injection vector
            S = parser.create_power_injection_vector(network)
            print(f"  Total load: P={np.sum(S.real)*network.base_power/1e6:.2f} MW, "
                  f"Q={np.sum(S.imag)*network.base_power/1e6:.2f} MVAr")
                  
        except Exception as e:
            print(f"Error parsing test file: {e}")
    else:
        print("No test file found. Use PGM grid generator to create test data first.")