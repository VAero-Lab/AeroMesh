"""MedialGraph data structure.

Wraps a networkx.Graph with the medial axis fields (r_m, θ_m, normals)
stored as node and edge attributes. Includes TopMaker classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx
import numpy as np


class VertexType(Enum):
    """TopMaker medial vertex classification based on degree."""
    DANGLE = auto()  # Degree 1 (e.g., LE)
    NORMAL = auto()  # Degree 3+ (junction)
    CORNER = auto()  # E.g., sharp TE or domain corner


class EdgeType(Enum):
    """TopMaker medial edge classification based on endpoints."""
    PRIMARY = auto() # Normal to Normal
    FLARE = auto()   # Normal to Corner
    DANGLE = auto()  # Normal to Dangle


@dataclass
class MedialGraph:
    """The medial axis of a computational domain, stored as a graph.

    Nodes represent the key medial vertices (Normal, Dangle, Corner).
    Edges represent the medial curves connecting them.
    Degree-2 nodes from the raw Voronoi diagram are collapsed into
    polyline geometry stored on the edges.

    Attributes
    ----------
    graph : networkx.Graph
        The underlying graph. Node attributes:
            - ``'pos'``: (2,) array, the medial vertex coordinates.
            - ``'type'``: VertexType (NORMAL, DANGLE, CORNER).
            - ``'touch_points'``: list of (2,) arrays (nearest boundary points).
        Edge attributes:
            - ``'polyline'``: (N, 2) array of points along the curve.
            - ``'r_m'``: (N,) array, medial radius sampled along the edge.
            - ``'theta_m'``: (N,) array, opening angle along the edge.
            - ``'n1'``: (N, 2) array, unit normals toward touch point 1.
            - ``'n2'``: (N, 2) array, unit normals toward touch point 2.
            - ``'type'``: EdgeType.
            - ``'arc_length'``: float, total length of this edge.
    """

    graph: nx.Graph = field(default_factory=nx.Graph)

    @classmethod
    def from_raw(
        cls,
        circumcentres: np.ndarray,
        edges: list[tuple[int, int]],
    ) -> MedialGraph:
        """Build a raw MedialGraph (all circumcentres as nodes).
        
        This is typically followed by a topological collapse of degree-2 nodes.

        Parameters
        ----------
        circumcentres : np.ndarray, shape (T, 2)
            All circumcentres (including those outside the domain).
        edges : list of (int, int)
            Pairs of triangle indices forming medial edges.

        Returns
        -------
        MedialGraph
        """
        G = nx.Graph()

        node_set = set()
        for i, j in edges:
            node_set.add(i)
            node_set.add(j)

        for idx in node_set:
            G.add_node(idx, pos=circumcentres[idx])

        for i, j in edges:
            length = np.linalg.norm(circumcentres[i] - circumcentres[j])
            G.add_edge(i, j, length=length)

        return cls(graph=G)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    def positions(self) -> np.ndarray:
        """Return an (N, 2) array of all medial point positions."""
        nodes = sorted(self.graph.nodes)
        return np.array([self.graph.nodes[n]["pos"] for n in nodes])

