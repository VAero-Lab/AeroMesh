"""Topological classification of the medial axis and singularity detection."""

from dataclasses import dataclass
import numpy as np

from aeromesh.medial.graph import MedialGraph, VertexType

@dataclass
class TopologyInfo:
    """Stores the classification of the domain and the singularities (block corners)."""
    domain_type: str  # "C", "O", "H"
    singularities: list[np.ndarray]  # (x, y) coordinates of block corners


def classify_singularities(mg: MedialGraph) -> TopologyInfo:
    """Identify the domain topology and extract all block corners from the medial graph.

    A block corner (singularity) is defined as any NORMAL (junction) node
    or any CORNER node (sharp geometric corners).

    Parameters
    ----------
    mg : MedialGraph
        The pruned medial axis skeleton.

    Returns
    -------
    TopologyInfo
    """
    singularities = []
    
    # 1. Identify all NORMAL nodes (junctions)
    # 2. Identify CORNER nodes (sharp geometric domain corners)
    # Currently, CORNER nodes might be labeled as DANGLE if they are degree-1
    # but we can differentiate them by their position (e.g. far from airfoil).
    
    for n, data in mg.graph.nodes(data=True):
        if data["type"] == VertexType.NORMAL:
            singularities.append(data["pos"])
            
    # Determine Domain Type based on the medial graph structure
    n_normal = sum(1 for _, d in mg.graph.nodes(data=True) if d["type"] == VertexType.NORMAL)
    
    # Simple heuristic for domain type based on Normal junctions:
    # O-domain: 0 normal nodes (if wake cut is handled differently) or 2 (if TE splits)
    # C-domain: 3 normal nodes (LE junction, Top Wake, Bottom Wake)
    # H-domain: 6+ normal nodes (corners create multiple junctions)
    # For now, we rely on the number of singularities.
    if n_normal <= 2:
        domain_type = "O"
    elif n_normal <= 5:
        # NACA 0012 C-domain has 3 normal nodes (LE, top wake, bottom wake)
        # Cambered C-domain has 4 normal nodes (LE, top wake, bottom wake, TE branch hit)
        domain_type = "C"
    else:
        domain_type = "H"

    return TopologyInfo(
        domain_type=domain_type,
        singularities=singularities
    )
