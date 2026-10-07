"""Design vector assembly and node-distribution calculations."""

from dataclasses import dataclass, field
import numpy as np

from aeromesh.topology.classify import TopologyInfo

@dataclass
class DesignVector:
    """The optimization variables and mesh constraints.
    
    In Phase C, this is the vector x that the optimizer varies.
    In Phase B, this provides the instructions to build the blocks.
    """
    # 1D array of length equal to the number of block interfaces
    nodes_per_edge: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    
    # Wall-normal spacing (first cell height)
    dy1: float = 1e-5
    
    # Expansion ratio
    r: float = 1.15
    
    # Coordinates of singularities (s_i along the medial axis)
    # For now, we store their physical (x, y) coordinates.
    singularities: list[np.ndarray] = field(default_factory=list)


def calculate_nodes_per_edge(singularities: list[np.ndarray], target_resolution: int = 50) -> np.ndarray:
    """Calculate the number of nodes per interface edge.
    
    For a fully structured multi-block grid, opposite edges of a block must 
    have the same number of nodes. For now, as a robust default for Phase B,
    we assign a constant target resolution to all topological edges.
    
    Parameters
    ----------
    singularities : list of np.ndarray
        The block corners.
    target_resolution : int
        The default number of nodes to place along block edges.
        
    Returns
    -------
    np.ndarray
        Array of node counts for the interfaces.
    """
    # Placeholder: assign constant resolution to all interfaces.
    # A true C-domain has 5 primary interface lines spanning from the medial axis to the boundaries.
    # In a generalized blocking algorithm, we will trace the interfaces dynamically.
    # We return a generic array that can be queried by the blocking module.
    
    # We will assume a maximum of 20 interfaces for any topology right now.
    return np.full(20, target_resolution, dtype=int)


def build_design_vector(topology: TopologyInfo, dy1: float = 1e-5, r: float = 1.15) -> DesignVector:
    """Assemble the design vector from the topology and flow constraints.
    
    Parameters
    ----------
    topology : TopologyInfo
        The classified topology and singularity locations.
    dy1 : float
        Target first-cell height.
    r : float
        Target expansion ratio.
        
    Returns
    -------
    DesignVector
    """
    n_nodes = calculate_nodes_per_edge(topology.singularities, target_resolution=40)
    
    return DesignVector(
        nodes_per_edge=n_nodes,
        dy1=dy1,
        r=r,
        singularities=topology.singularities
    )
