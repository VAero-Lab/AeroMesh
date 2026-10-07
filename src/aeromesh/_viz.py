"""Internal plotting utilities for AeroMesh.

Provides quick-look visualization functions used by examples and tests.
Not part of the public API.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_boundary(boundary, ax=None, **kwargs):
    """Plot an airfoil boundary."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))
    defaults = {"color": "k", "linewidth": 1.5, "label": boundary.name}
    defaults.update(kwargs)
    ax.plot(boundary.points[:, 0], boundary.points[:, 1], **defaults)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    return ax


def plot_domain(domain, ax=None, show_markers=True, **kwargs):
    """Plot the computational domain (airfoil + outer boundary)."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Airfoil
    air = domain.airfoil.points
    ax.plot(air[:, 0], air[:, 1], "k-", linewidth=2, label="Airfoil")

    # Outer boundary
    outer_closed = np.vstack([domain.outer, domain.outer[:1]])
    ax.plot(outer_closed[:, 0], outer_closed[:, 1], "b-",
            linewidth=1, alpha=0.7, label="Far-field")

    # Hole point
    ax.plot(*domain.hole_point, "rx", markersize=8, label="Hole point")

    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"{domain.domain_type.value}-domain")
    return ax


def plot_triangulation(tri_result, domain=None, ax=None):
    """Plot the CDT triangulation."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    verts = tri_result["vertices"]
    tris = tri_result["triangles"]

    ax.triplot(verts[:, 0], verts[:, 1], tris, "b-", linewidth=0.3, alpha=0.5)

    if domain is not None:
        air = domain.airfoil.points
        ax.plot(air[:, 0], air[:, 1], "k-", linewidth=2)

    ax.set_aspect("equal")
    ax.set_title(f"CDT: {len(tris)} triangles, {len(verts)} vertices")
    return ax


def plot_medial_graph(mg, domain=None, ax=None):
    """Plot the skeletal medial axis graph and polylines."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    from aeromesh.medial.graph import VertexType, EdgeType

    # Draw medial polylines
    for u, v, data in mg.graph.edges(data=True):
        polyline = data["polyline"]
        ax.plot(polyline[:, 0], polyline[:, 1], color="#e74c3c", linewidth=1.0, alpha=0.9, zorder=4)

    # Draw nodes with distinct shapes and clear labels
    for n, data in mg.graph.nodes(data=True):
        vtype = data["type"]
        pos = data["pos"]
        if vtype == VertexType.NORMAL:
            # Junction point (Degree 3+)
            ax.plot(pos[0], pos[1], marker="o", color="#27ae60", markersize=4, zorder=5,
                    label="Normal (Junction)" if "Normal (Junction)" not in ax.get_legend_handles_labels()[1] else "")
        elif vtype == VertexType.DANGLE:
            # End of a branch (Degree 1, e.g. at Leading Edge)
            ax.plot(pos[0], pos[1], marker="s", color="#8e44ad", markersize=4, zorder=5,
                    label="Dangle (End)" if "Dangle (End)" not in ax.get_legend_handles_labels()[1] else "")
        elif vtype == VertexType.CORNER:
            # Corner point (Sharp TE or Domain corners)
            ax.plot(pos[0], pos[1], marker="^", color="#2980b9", markersize=4, zorder=5,
                    label="Corner" if "Corner" not in ax.get_legend_handles_labels()[1] else "")

    if domain is not None:
        air = domain.airfoil.points
        ax.plot(air[:, 0], air[:, 1], "k-", linewidth=1.0, zorder=3)
        outer_closed = np.vstack([domain.outer, domain.outer[:1]])
        ax.plot(outer_closed[:, 0], outer_closed[:, 1], color="#7f8c8d", linestyle="--",
                linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_aspect("equal")
    ax.set_title(f"Medial Skeleton: {mg.n_nodes} nodes, {mg.n_edges} edges")
    ax.legend()
    return ax
