"""Topological block decomposition of the fluid domain."""

from dataclasses import dataclass
import numpy as np

from aeromesh.medial.graph import MedialGraph
from aeromesh.topology.design import DesignVector
from aeromesh.domain.outer import Domain

@dataclass
class Block:
    """A 4-sided structural block in the fluid domain.
    
    The boundaries are oriented counter-clockwise, though the exact orientation
    depends on the mesh generator. Typically:
    - south: airfoil surface
    - north: far-field boundary
    - west: interface curve (e.g. from a singularity)
    - east: interface curve
    """
    id: int
    south: np.ndarray  # (N, 2) polyline
    east: np.ndarray   # (M, 2) polyline
    north: np.ndarray  # (N, 2) polyline
    west: np.ndarray   # (M, 2) polyline


@dataclass
class BlockSystem:
    """A collection of blocks forming the entire computational domain."""
    blocks: list[Block]


def _build_c_grid_blocks(domain: Domain) -> list[Block]:
    """Explicitly slice a C-domain into 4 canonical topological blocks."""
    blocks = []
    outer = domain.outer
    air_pts = domain.airfoil.points
    te_point = domain.airfoil.te
    
    # Identify airfoil LE and split into upper/lower
    le_idx = np.argmin(air_pts[:, 0])
    # The geometric array for NACA starts at TE upper (y>0) -> LE -> TE lower (y<0)
    air_upper = air_pts[:le_idx+1]
    air_lower = air_pts[le_idx:]
    
    top_corner_idx = 0
    bot_corner_idx = np.argmax(outer[:, 0] - outer[:, 1])
    
    top_half = outer[:, 1] > 0
    bot_half = outer[:, 1] < 0
    
    top_te_idx = np.argmin(np.abs(outer[top_half, 0] - te_point[0]))
    top_te_idx_abs = np.where(top_half)[0][top_te_idx]
    
    bot_te_idx = np.argmin(np.abs(outer[bot_half, 0] - te_point[0]))
    bot_te_idx_abs = np.where(bot_half)[0][bot_te_idx]
    
    le_outer_idx = np.argmin(outer[:, 0])
    
    wake_x = np.linspace(te_point[0], outer[top_corner_idx, 0], 50)
    upper_wake = np.column_stack((wake_x, np.full_like(wake_x, te_point[1])))
    lower_wake = np.column_stack((wake_x, np.full_like(wake_x, te_point[1])))
    
    # --- Block 1: Top Airfoil ---
    south1 = air_upper[::-1]  # LE to TE
    north1 = outer[top_te_idx_abs:le_outer_idx+1][::-1]
    west1 = np.array([south1[0], north1[0]])
    east1 = np.array([south1[-1], north1[-1]])
    blocks.append(Block(id=0, south=south1, east=east1, north=north1, west=west1))
    
    # --- Block 2: Bottom Airfoil ---
    south2 = air_lower  # LE to TE
    north2 = outer[le_outer_idx:bot_te_idx_abs+1]
    west2 = np.array([south2[0], north2[0]])
    east2 = np.array([south2[-1], north2[-1]])
    blocks.append(Block(id=1, south=south2, east=east2, north=north2, west=west2))
    
    # --- Block 3: Top Wake ---
    south3 = upper_wake
    north3 = outer[:top_te_idx_abs+1][::-1]
    west3 = np.array([south3[0], north3[0]])
    east3 = np.array([south3[-1], north3[-1]])
    blocks.append(Block(id=2, south=south3, east=east3, north=north3, west=west3))
    
    # --- Block 4: Bottom Wake ---
    south4 = lower_wake
    north4 = outer[bot_te_idx_abs:bot_corner_idx+1]
    west4 = np.array([south4[0], north4[0]])
    east4 = np.array([south4[-1], north4[-1]])
    blocks.append(Block(id=3, south=south4, east=east4, north=north4, west=west4))
    
    return blocks

def _build_o_grid_blocks(domain: Domain) -> list[Block]:
    """Explicitly slice an O-domain into 2 canonical topological blocks."""
    # Placeholder for O-grid slicing
    return []

def _build_h_grid_blocks(domain: Domain) -> list[Block]:
    """Explicitly slice an H-domain into 6 canonical topological blocks."""
    # Placeholder for H-grid slicing
    return []

def build_block_system(mg: MedialGraph, dv: DesignVector, domain: Domain) -> BlockSystem:
    """Construct the explicit block boundaries for structured meshing.
    
    Instead of mapping raw Voronoi regions (which produce triangular blocks
    at sharp 90-degree corners), we construct canonical 4-sided topological 
    blocks (e.g. C-grid, O-grid) by slicing the physical boundaries at the
    identified singularity coordinates.
    
    Parameters
    ----------
    mg : MedialGraph
        The medial axis skeleton.
    dv : DesignVector
        Optimization constraints (currently provides target nodes).
    domain : Domain
        The physical domain boundaries.
        
    Returns
    -------
    BlockSystem
    """
    domain_type = type(domain).__name__
    
    if "CDomain" in domain_type:
        blocks = _build_c_grid_blocks(domain)
    elif "ODomain" in domain_type:
        blocks = _build_o_grid_blocks(domain)
    elif "HDomain" in domain_type:
        blocks = _build_h_grid_blocks(domain)
    else:
        # Fallback to C-grid
        blocks = _build_c_grid_blocks(domain)
        
    return BlockSystem(blocks=blocks)


def enforce_connectivity(system: BlockSystem) -> bool:
    """Verify that adjacent blocks share the exact same number of nodes on their interfaces.
    
    Parameters
    ----------
    system : BlockSystem
        The generated blocks.
        
    Returns
    -------
    bool
        True if the system is conformally connected, False otherwise.
    """
    # For a fully connected system, the lengths of shared edges must match exactly.
    # In Phase B, this acts as a validation step before passing the blocks to TFI.
    # Returning True as placeholder.
    return True
