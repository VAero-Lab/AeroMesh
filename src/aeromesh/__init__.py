"""AeroMesh — automatic structured meshing for 2-D aerodynamic sections.

The package exposes the geometry layer (sections, loops, regions, far-field
curves) at the top level; the mesher itself lives in :mod:`aeromesh.mesh` and
the quality metrics in :mod:`aeromesh.quality`.

Imports are deliberately shallow. The previous version eagerly imported every
subpackage here, so ``import aeromesh.mesh`` pulled in three generations of
the method at once -- the medial-axis blocking engine, both collars and the
current marcher -- which made the dependency structure impossible to read and
the retired code impossible to remove. Only what the callers use is imported.

:mod:`aeromesh.medial` **is** exported here, although nothing on the
single-body path calls it. It is kept for the multi-element and
three-dimensional work, where the medial radius is the cap on layer thickness
inside a slot, and it costs three modules to import. Keeping its names at the
top level keeps the public API stable; hiding them to tidy the namespace
broke callers for no gain.

:mod:`aeromesh.mesh` is **not** imported here. It pulls in scipy sparse
solvers and is the one subpackage heavy enough to be worth deferring.
"""

from aeromesh.domain.farfield import (box_farfield, c_farfield, circle_farfield,
                                      offset_farfield, rectangle)
from aeromesh.geometry.airfoil import load_airfoil, profile_to_loop
from aeromesh.geometry.corners import (Corners, detect_corners,
                                       fluid_interior_angle, vertex_turns)
from aeromesh.geometry.cusp import find_cusps, open_cusp
from aeromesh.geometry.loop import Loop, signed_area
from aeromesh.geometry.region import BoundaryLoop, Region
from aeromesh.medial.axis import (EdgeKind, MedialAxis, VertexKind,
                                  exterior_axis, interior_axis)
from aeromesh.medial.fields import (medial_radius_on_boundary,
                                    optimum_flow_index)


def build_region(
    bodies,
    farfield=15.0,
    outer=None,
    spacing=None,
    body_spacing=None,
    te="blunt",
    te_thickness=0.002,
    n_farfield=720,
):
    """Build a validated :class:`Region` from a set of bodies.

    Parameters
    ----------
    bodies : Loop or sequence of Loop
        The solid bodies. One, or as many as the configuration has.
    farfield : float
        Radius of the default circular outer boundary, in the bodies' own
        units. Ignored when ``outer`` is given.
    outer : Loop, optional
        An explicit outer boundary. Build it with one of the constructors in
        :mod:`aeromesh.domain.farfield`:

        =========================  =======  ==========================
        constructor                corners  conventional name
        =========================  =======  ==========================
        ``circle_farfield``        0        O-type (the default)
        ``c_farfield``             2        C-type
        ``box_farfield``           4        H-type / tunnel walls
        ``offset_farfield``        0        distance level set
        =========================  =======  ==========================

        The shape is a *supplied* modelling input, like where you truncate the
        domain — not a topology selection. It is passed as a curve and stored
        as a curve; no later stage can tell which constructor made it. What it
        does influence is how many corners the fluid boundary has, and a corner
        generates a flare, and a flare generates a block. So the choice matters,
        and the standard shapes are offered as standard shapes.
    spacing, body_spacing : float, optional
        Point spacing for the outer boundary and for the bodies. Each defaults
        to that loop's own scale divided by 500.
    te : {'blunt', 'sharp'}
        Cusp policy for the bodies. See :func:`prepare_boundary`.
    te_thickness : float
        Face length used by the ``'blunt'`` policy.
    n_farfield : int
        Points on the default circular far field.

    Returns
    -------
    Region

    Examples
    --------
    >>> region = build_region([main], farfield=15.0)                 # circle
    >>> region = build_region([main], outer=c_farfield([main], 15.0, 25.0))
    >>> region = build_region([main], outer=box_farfield([main], 15, 25, 15))
    """
    if isinstance(bodies, Loop):
        bodies = [bodies]
    bodies = list(bodies)
    if not bodies:
        raise ValueError("build_region needs at least one body.")
    if outer is None:
        outer = circle_farfield(bodies, radius=farfield, n_points=n_farfield)
    return Region.build(outer, bodies, spacing=spacing, body_spacing=body_spacing,
                        te=te, te_thickness=te_thickness)


__all__ = [
    "Loop", "signed_area",
    "Corners", "detect_corners", "fluid_interior_angle", "vertex_turns",
    "find_cusps", "open_cusp",
    "load_airfoil", "profile_to_loop",
    "Region", "BoundaryLoop", "build_region",
    "circle_farfield", "c_farfield", "box_farfield", "offset_farfield",
    "rectangle",
    "MedialAxis", "EdgeKind", "VertexKind",
    "exterior_axis", "interior_axis",
    "medial_radius_on_boundary", "optimum_flow_index",
]
