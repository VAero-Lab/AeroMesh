"""Geometry: the primitives the whole pipeline is built on.

A ``Loop`` is a single closed curve; a ``Region`` is one outer loop and any
number of bodies. Corner detection and cusp opening are stated in terms of
curves and arc length, so nothing here knows what an airfoil is -- the AeroShape
intake in :mod:`aeromesh.geometry.airfoil` is the only module that does, and it
stops at returning a ``Loop``.
"""

from aeromesh.geometry.airfoil import load_airfoil, profile_to_loop
from aeromesh.geometry.corners import (
    Corners,
    detect_corners,
    fluid_interior_angle,
    vertex_turns,
)
from aeromesh.geometry.cusp import find_cusps, open_cusp
from aeromesh.geometry.loop import Loop, signed_area
from aeromesh.geometry.region import (
    BoundaryLoop,
    Region,
    largest_inscribed_circle,
    prepare_boundary,
)

__all__ = [
    "Loop", "signed_area",
    "Corners", "detect_corners", "fluid_interior_angle", "vertex_turns",
    "find_cusps", "open_cusp",
    "load_airfoil", "profile_to_loop",
    "BoundaryLoop", "Region", "largest_inscribed_circle", "prepare_boundary",
]
