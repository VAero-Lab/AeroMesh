"""Computational domain construction around an airfoil.

Builds the outer boundary Γ_out that, together with the airfoil boundary Γ,
defines the fluid domain Ω for meshing.

Domain types
------------
- **C-domain**: semicircular arc upstream of the LE, rectangular wake downstream.
- **O-domain**: full ellipse/circle around the airfoil with a wake cut.
- **H-domain**: full rectangle enclosing the airfoil.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from aeromesh.geometry.boundary import Boundary, _signed_area


class DomainType(Enum):
    """Supported computational domain topologies."""

    C = "C"
    O = "O"
    H = "H"


@dataclass
class Domain:
    """A complete computational domain for meshing.

    Contains the inner boundary (airfoil), outer boundary, segment markers,
    and a hole point to tell the triangulator which region is the airfoil
    interior (to be excluded).

    Attributes
    ----------
    airfoil : Boundary
        The inner boundary (wall).
    outer : np.ndarray, shape (M, 2)
        The outer boundary points, ordered CCW.
    domain_type : DomainType
        The topology class (C, O, or H).
    farfield : float
        Far-field distance in chord lengths.
    wake_length : float
        Wake extension behind the TE in chord lengths.
    hole_point : np.ndarray, shape (2,)
        A point inside the airfoil, used by the triangulator to identify
        the airfoil interior as a hole.
    vertices : np.ndarray, shape (N+M, 2)
        Combined vertices (airfoil + outer) for the triangulator.
    segments : np.ndarray, shape (K, 2)
        Edge connectivity (index pairs) defining the constrained edges.
    markers : np.ndarray, shape (K,)
        Segment markers: 1 = airfoil wall, 2 = farfield, 3 = wake, etc.
    """

    airfoil: Boundary
    outer: np.ndarray
    domain_type: DomainType
    farfield: float
    wake_length: float
    hole_point: np.ndarray
    vertices: np.ndarray
    segments: np.ndarray
    markers: np.ndarray

    def to_triangle_input(self) -> dict:
        """Return a dict formatted for ``triangle.triangulate()``.

        Returns
        -------
        dict
            Keys: ``'vertices'``, ``'segments'``, ``'holes'``,
            ``'segment_markers'``.
        """
        return {
            "vertices": self.vertices,
            "segments": self.segments,
            "holes": self.hole_point.reshape(1, 2),
            "segment_markers": self.markers.reshape(-1, 1),
        }


def build_domain(
    airfoil: Boundary,
    domain_type: DomainType | str = DomainType.C,
    farfield: float = 15.0,
    wake_length: float = 5.0,
    n_pts: int = 150,
) -> Domain:
    """Construct a computational domain around an airfoil.

    Parameters
    ----------
    airfoil : Boundary
        The airfoil boundary (unit chord, LE at origin).
    domain_type : DomainType or str
        The topology class (C, O, or H).
    farfield : float
        Far-field radius/distance in chord lengths.
    wake_length : float
        Wake extension behind the TE in chord lengths (not applicable for O).
    n_pts : int
        Approximate number of points to distribute on the outer boundary.

    Returns
    -------
    Domain
        The fully assembled domain ready for triangulation.
    """
    if isinstance(domain_type, str):
        domain_type = DomainType(domain_type.upper())

    if domain_type == DomainType.C:
        return build_c_domain(airfoil, farfield, wake_length, n_pts)
    elif domain_type == DomainType.O:
        return build_o_domain(airfoil, farfield, n_pts)
    elif domain_type == DomainType.H:
        return build_h_domain(airfoil, farfield, wake_length, n_pts)
    else:
        raise ValueError(f"Unsupported domain type: {domain_type}")


def _assemble_domain(
    airfoil: Boundary,
    outer_pts: np.ndarray,
    domain_type: DomainType,
    farfield: float,
    wake_length: float,
    out_markers: np.ndarray | None = None,
) -> Domain:
    """Helper to assemble the Domain dataclass from points."""
    # Ensure CCW winding
    outer_closed = np.vstack([outer_pts, outer_pts[:1]])
    if _signed_area(outer_closed) < 0:
        outer_pts = outer_pts[::-1]
        if out_markers is not None:
            out_markers = out_markers[::-1]

    # Airfoil boundary
    airfoil_pts = airfoil.points[:-1]  # drop the closing duplicate
    n_air = len(airfoil_pts)
    n_out = len(outer_pts)

    vertices = np.vstack([airfoil_pts, outer_pts])

    # Airfoil segments
    air_segs = np.column_stack((
        np.arange(n_air),
        np.roll(np.arange(n_air), -1),
    ))
    
    # Airfoil markers: Upper=10, Lower=11
    # points 0 to le_index are upper surface.
    le_idx = airfoil.le_index
    air_markers = np.full(n_air, 11, dtype=np.int32)
    air_markers[:le_idx] = 10

    # Outer segments
    out_indices = np.arange(n_air, n_air + n_out)
    out_segs = np.column_stack((
        out_indices,
        np.roll(out_indices, -1),
    ))
    if out_markers is None:
        out_markers = np.full(n_out, 20, dtype=np.int32)

    segments = np.vstack([air_segs, out_segs])
    markers = np.concatenate([air_markers, out_markers])

    # Hole point (inside the airfoil)
    hole_point = airfoil_pts.mean(axis=0)

    return Domain(
        airfoil=airfoil,
        outer=outer_pts,
        domain_type=domain_type,
        farfield=farfield,
        wake_length=wake_length,
        hole_point=hole_point,
        vertices=vertices,
        segments=segments,
        markers=markers,
    )


def build_c_domain(
    airfoil: Boundary,
    farfield: float = 15.0,
    wake_length: float = 5.0,
    n_pts: int = 150,
) -> Domain:
    """Construct a C-shaped computational domain.
    
    Upstream semicircle centred on LE, rectangular wake downstream.
    """
    te = airfoil.te
    le = airfoil.le
    te_x, _ = te[0], te[1]
    le_x, le_y = le[0], le[1]
    wake_x = te_x + wake_length
    R = farfield

    # Allocate points roughly proportionally
    total_len = np.pi * R + 2 * (wake_x - le_x) + 2 * R
    n_arc = max(10, int(n_pts * (np.pi * R) / total_len))
    n_wake = max(10, int(n_pts * (wake_x - le_x) / total_len))
    n_back = max(10, int(n_pts * (2 * R) / total_len))

    # Upper edge: wake_x back to le_x (going left)
    upper_edge = np.column_stack((
        np.linspace(wake_x, le_x, n_wake + 2)[1:-1],
        np.full(n_wake, R),
    ))

    # Upstream semicircle: top to bottom, going CCW
    theta_arc = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)
    arc_pts = np.column_stack((
        le_x + R * np.cos(theta_arc),
        le_y + R * np.sin(theta_arc),
    ))

    # Lower edge: le_x to wake_x (going right)
    lower_edge = np.column_stack((
        np.linspace(le_x, wake_x, n_wake + 2)[1:-1],
        np.full(n_wake, -R),
    ))

    # Back edge: bottom to top (going up)
    back_edge = np.column_stack((
        np.full(n_back, wake_x),
        np.linspace(-R, R, n_back + 2)[1:-1],
    ))

    # Corner points
    top_right = np.array([[wake_x, R]])
    bot_right = np.array([[wake_x, -R]])

    outer_pts = np.vstack([
        top_right,
        upper_edge,
        arc_pts,
        lower_edge,
        bot_right,
        back_edge,
    ])

    # Markers for outer boundary macro-curves
    # 20 = Upper, 21 = Arc, 22 = Lower, 23 = Back
    out_markers = np.concatenate([
        np.full(1 + len(upper_edge), 20, dtype=np.int32),
        np.full(len(arc_pts), 21, dtype=np.int32),
        np.full(len(lower_edge) + 1, 22, dtype=np.int32),
        np.full(len(back_edge), 23, dtype=np.int32),
    ])

    return _assemble_domain(airfoil, outer_pts, DomainType.C, farfield, wake_length, out_markers)


def build_o_domain(
    airfoil: Boundary,
    farfield: float = 15.0,
    n_pts: int = 150,
) -> Domain:
    """Construct an O-shaped computational domain (full circle)."""
    # Centered at mid-chord (0.5, 0)
    cx, cy = 0.5, 0.0
    R = farfield

    # Full circle, starting from TE (theta=0) going CCW
    theta = np.linspace(0, 2 * np.pi, n_pts + 1)[:-1]
    
    outer_pts = np.column_stack((
        cx + R * np.cos(theta),
        cy + R * np.sin(theta),
    ))

    # O-domain outer is a single macro-curve (20)
    out_markers = np.full(len(outer_pts), 20, dtype=np.int32)

    return _assemble_domain(airfoil, outer_pts, DomainType.O, farfield, 0.0, out_markers)


def build_h_domain(
    airfoil: Boundary,
    farfield: float = 15.0,
    wake_length: float = 5.0,
    n_pts: int = 150,
) -> Domain:
    """Construct an H-shaped computational domain (rectangle)."""
    x_min = -farfield
    x_max = 1.0 + wake_length
    y_min = -farfield
    y_max = farfield

    # Allocate points
    w = x_max - x_min
    h = y_max - y_min
    total_len = 2 * w + 2 * h
    n_w = max(10, int(n_pts * w / total_len))
    n_h = max(10, int(n_pts * h / total_len))

    # Top edge: right to left
    top_edge = np.column_stack((
        np.linspace(x_max, x_min, n_w + 2)[1:-1],
        np.full(n_w, y_max),
    ))
    # Left edge: top to bottom
    left_edge = np.column_stack((
        np.full(n_h, x_min),
        np.linspace(y_max, y_min, n_h + 2)[1:-1],
    ))
    # Bottom edge: left to right
    bottom_edge = np.column_stack((
        np.linspace(x_min, x_max, n_w + 2)[1:-1],
        np.full(n_w, y_min),
    ))
    # Right edge: bottom to top
    right_edge = np.column_stack((
        np.full(n_h, x_max),
        np.linspace(y_min, y_max, n_h + 2)[1:-1],
    ))

    top_right = np.array([[x_max, y_max]])
    top_left = np.array([[x_min, y_max]])
    bot_left = np.array([[x_min, y_min]])
    bot_right = np.array([[x_max, y_min]])

    outer_pts = np.vstack([
        top_right, top_edge,
        top_left, left_edge,
        bot_left, bottom_edge,
        bot_right, right_edge,
    ])

    out_markers = np.concatenate([
        np.full(1 + len(top_edge), 20, dtype=np.int32),
        np.full(1 + len(left_edge), 21, dtype=np.int32),
        np.full(1 + len(bottom_edge), 22, dtype=np.int32),
        np.full(1 + len(right_edge), 23, dtype=np.int32),
    ])

    return _assemble_domain(airfoil, outer_pts, DomainType.H, farfield, wake_length, out_markers)
