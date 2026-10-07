"""Constrained Delaunay Triangulation and medial axis extraction.

Uses Shewchuk's Triangle library for the CDT (needed later for mesh generation),
and scipy.spatial.Voronoi on the boundary points for exact medial-axis extraction.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import triangle as tr
from scipy.spatial import Voronoi
from matplotlib.path import Path

from aeromesh.domain.outer import Domain
from aeromesh.medial.graph import MedialGraph, VertexType, EdgeType


# ---------------------------------------------------------------------------
# CDT (still needed for later mesh generation steps)
# ---------------------------------------------------------------------------

def triangulate_domain(
    domain: Domain,
    max_area: float | None = None,
    min_angle: float = 20.0,
) -> dict:
    """Run a Constrained Delaunay Triangulation on the domain."""
    tri_input = domain.to_triangle_input()

    if max_area is None:
        max_area = domain.farfield**2 / 5000.0

    switches = f"pq{min_angle:.0f}a{max_area:.6f}D"
    return tr.triangulate(tri_input, switches)


def compute_circumcentres(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Compute the circumcentre of every triangle (vectorised)."""
    A = verts[tris[:, 0]]
    B = verts[tris[:, 1]]
    C = verts[tris[:, 2]]

    D = 2.0 * (A[:, 0] * (B[:, 1] - C[:, 1])
               + B[:, 0] * (C[:, 1] - A[:, 1])
               + C[:, 0] * (A[:, 1] - B[:, 1]))
    D = np.where(np.abs(D) < 1e-15, 1e-15, D)

    ux = ((A[:, 0]**2 + A[:, 1]**2) * (B[:, 1] - C[:, 1])
          + (B[:, 0]**2 + B[:, 1]**2) * (C[:, 1] - A[:, 1])
          + (C[:, 0]**2 + C[:, 1]**2) * (A[:, 1] - B[:, 1])) / D

    uy = ((A[:, 0]**2 + A[:, 1]**2) * (C[:, 0] - B[:, 0])
          + (B[:, 0]**2 + B[:, 1]**2) * (A[:, 0] - C[:, 0])
          + (C[:, 0]**2 + C[:, 1]**2) * (B[:, 0] - A[:, 0])) / D

    return np.column_stack((ux, uy))


# ---------------------------------------------------------------------------
# Boundary-point labelling
# ---------------------------------------------------------------------------

def _label_boundary_points(domain: Domain) -> np.ndarray:
    """Assign a macro-curve label to each boundary point.

    Labels used:
        10 = upper airfoil surface (TE → LE)
        11 = lower airfoil surface (LE → TE)
        20, 21, 22, 23 = outer boundary macro-curves
    """
    n_air = len(domain.airfoil.points) - 1  # open polygon count
    n_out = len(domain.outer)
    n_total = n_air + n_out

    labels = np.zeros(n_total, dtype=np.int32)

    # Airfoil: upper vs lower split at le_index
    le_idx = domain.airfoil.le_index
    labels[:le_idx] = 10       # upper surface
    labels[le_idx:n_air] = 11  # lower surface

    # Outer boundary: reuse the per-segment macro-curve markers from domain
    outer_seg_markers = domain.markers[n_air:]
    for i in range(n_out):
        labels[n_air + i] = outer_seg_markers[i]

    return labels


# ---------------------------------------------------------------------------
# Voronoi-based medial axis extraction (replaces the broken CDT approach)
# ---------------------------------------------------------------------------

def extract_medial_axis(tri_result: dict, domain: Domain) -> MedialGraph:
    """Extract the medial axis skeleton using scipy.spatial.Voronoi.

    Algorithm
    ---------
    1. Compute the Voronoi diagram of ALL boundary points (airfoil + outer).
    2. Each Voronoi ridge separates two generator points.
       A ridge is part of the medial axis iff its two generators belong to
       **different** macro-curves.
    3. Keep only ridges whose Voronoi vertices both lie inside the fluid domain.
    4. Collapse degree-2 chains and prune short dangle branches.

    Parameters
    ----------
    tri_result : dict
        Output from ``triangulate_domain()`` (currently unused but kept for
        API compatibility and future use).
    domain : Domain
        The assembled computational domain.

    Returns
    -------
    MedialGraph
        The pruned, collapsed medial axis skeleton.
    """
    bnd_pts = domain.vertices          # (N_bnd, 2) all boundary points
    labels = _label_boundary_points(domain)

    # --- Step 1: Initialize paths and arrays ---------------------------
    bnd_pts = domain.vertices

    # --- Step 2 & 3: filter ridges and project crossing ridges ---------
    air_path = Path(domain.airfoil.points)
    outer_path = Path(np.vstack([domain.outer, domain.outer[:1]]))
    
    in_air = air_path.contains_points(bnd_pts)
    in_outer = outer_path.contains_points(bnd_pts)
    in_fluid = in_outer & ~in_air
    
    # To fix the Voronoi angle bisector at a sharp trailing edge (which otherwise 
    # creates a perpendicular artifact), we omit the singular TE point (index 0) 
    # from the Voronoi generators. This creates an infinitesimal gap that forces 
    # the algorithm to generate a ridge exactly along the true angle bisector.
    use_indices = np.arange(1, len(bnd_pts))
    vor_pts = bnd_pts[use_indices]
    vor_labels = labels[use_indices]
    
    vor = Voronoi(vor_pts)
    
    # We map nodes using the original boundary points
    node_positions = {i: vor.vertices[i] for i in range(len(vor.vertices))}
    medial_edges: list[tuple[int, int, int, int]] = []
    
    # Evaluate fluid containment for Voronoi vertices
    v_in_air = air_path.contains_points(vor.vertices)
    v_in_outer = outer_path.contains_points(vor.vertices)
    v_in_fluid = v_in_outer & ~v_in_air
    
    for ridge_idx, (v1, v2) in enumerate(vor.ridge_vertices):
        if v1 == -1 or v2 == -1:
            continue  # infinite ridge

        p1, p2 = vor.ridge_points[ridge_idx]
        
        orig_p1 = use_indices[p1]
        orig_p2 = use_indices[p2]
        
        if labels[orig_p1] == labels[orig_p2]:
            continue  # same macro-curve -> not medial

        if v_in_fluid[v1] and v_in_fluid[v2]:
            medial_edges.append((v1, v2, orig_p1, orig_p2))
        elif v_in_fluid[v1] and v_in_air[v2]:
            # Crosses into airfoil, project inner node to boundary generator midpoint
            mid = (bnd_pts[orig_p1] + bnd_pts[orig_p2]) / 2.0
            new_v = len(node_positions)
            node_positions[new_v] = mid
            medial_edges.append((v1, new_v, orig_p1, orig_p2))
        elif v_in_fluid[v2] and v_in_air[v1]:
            # Crosses into airfoil, project inner node to boundary generator midpoint
            mid = (bnd_pts[orig_p1] + bnd_pts[orig_p2]) / 2.0
            new_v = len(node_positions)
            node_positions[new_v] = mid
            medial_edges.append((new_v, v2, orig_p1, orig_p2))

    # --- Build raw graph -----------------------------------------------
    raw_G = nx.Graph()
    for v1, v2, p1, p2 in medial_edges:
        pos1 = node_positions[v1]
        pos2 = node_positions[v2]
        raw_G.add_node(v1, pos=pos1)
        raw_G.add_node(v2, pos=pos2)
        raw_G.add_edge(v1, v2, weight=float(np.linalg.norm(pos1 - pos2)), p1=p1, p2=p2)

    if raw_G.number_of_nodes() == 0:
        return MedialGraph(graph=nx.Graph())

    # --- Compute medial radius for each node ---------------------------
    # r_m = distance from the Voronoi vertex to its nearest boundary point
    for n in raw_G.nodes:
        pos = raw_G.nodes[n]["pos"]
        dists = np.linalg.norm(bnd_pts - pos, axis=1)
        raw_G.nodes[n]["radius"] = float(np.min(dists))

    # --- Collapse degree-2 chains into polyline edges ------------------
    clean_G = nx.MultiGraph()

    key_nodes = [n for n, d in raw_G.degree() if d != 2]
    for n in key_nodes:
        deg = raw_G.degree(n)
        vtype = VertexType.NORMAL if deg >= 3 else VertexType.DANGLE
        clean_G.add_node(
            n,
            pos=raw_G.nodes[n]["pos"],
            type=vtype,
            radius=raw_G.nodes[n]["radius"],
        )

    visited_edges: set[tuple[int, int]] = set()
    for start_node in key_nodes:
        for neighbor in raw_G.neighbors(start_node):
            canon = tuple(sorted((start_node, neighbor)))
            if canon in visited_edges:
                continue

            path = [start_node]
            curr = neighbor
            prev = start_node

            while raw_G.degree(curr) == 2:
                path.append(curr)
                visited_edges.add(tuple(sorted((prev, curr))))
                nxt = [nb for nb in raw_G.neighbors(curr) if nb != prev][0]
                prev = curr
                curr = nxt

            path.append(curr)
            visited_edges.add(tuple(sorted((prev, curr))))
            end_node = curr

            # Polyline geometry
            polyline = np.array([raw_G.nodes[n]["pos"] for n in path])
            r_m = np.array([raw_G.nodes[n]["radius"] for n in path])
            arc_length = float(
                np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))
            )

            # Edge type
            t1 = clean_G.nodes[start_node]["type"]
            t2 = clean_G.nodes[end_node]["type"]
            if t1 == VertexType.NORMAL and t2 == VertexType.NORMAL:
                etype = EdgeType.PRIMARY
            elif t1 == VertexType.DANGLE or t2 == VertexType.DANGLE:
                etype = EdgeType.DANGLE
            else:
                etype = EdgeType.FLARE

            # Boundary generators
            gen1 = []
            gen2 = []
            for i in range(len(path) - 1):
                edata = raw_G.edges[path[i], path[i+1]]
                gen1.append(edata["p1"])
                gen2.append(edata["p2"])

            clean_G.add_edge(
                start_node, end_node,
                polyline=polyline,
                r_m=r_m,
                arc_length=arc_length,
                type=etype,
                gen1=np.array(gen1, dtype=int),
                gen2=np.array(gen2, dtype=int)
            )

    # --- Prune short dangle branches -----------------------------------
    to_remove = []
    for u, v, k, data in list(clean_G.edges(keys=True, data=True)):
        if data["type"] != EdgeType.DANGLE:
            continue
        dangle_node = u if clean_G.nodes[u]["type"] == VertexType.DANGLE else v
        normal_node = v if dangle_node == u else u
        local_scale = clean_G.nodes[normal_node]["radius"]
        if data["arc_length"] < 0.5 * local_scale:
            to_remove.append((u, v, k))

    for u, v, k in to_remove:
        clean_G.remove_edge(u, v, k)

    clean_G.remove_nodes_from(list(nx.isolates(clean_G)))

    return MedialGraph(graph=clean_G)
