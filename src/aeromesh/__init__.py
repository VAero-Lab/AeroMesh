"""AeroMesh — automatic structured multiblock meshing via the medial axis.

Implements the MAPS method: medial-axis parameterised structured meshing.

The input is a **set of closed loops** — one outer boundary and any number of
bodies. There is no domain type to choose and no topology to select. A single
airfoil, a three-element high-lift configuration and a wing–body crossflow cut
are the same kind of object, and every stage sees them identically.

    import aeromesh as am

    main   = am.load_airfoil("2412")
    flap   = main.transform(scale=0.32, angle=-0.49, dx=1.02, dy=-0.10, name="flap")
    region = am.build_region([main, flap], farfield=15.0, downstream=25.0)

Pipeline status
---------------
S0  geometry, corners, cusp policy, region        available
S1  medial engine (r_m, theta_m, normals, typing)  available
S2  singularity solver and certificates            available (4/6 certify)
S3  decomposition                                  not started
S4  mesh construction                              not started

See PROJECT_TRACKER.md for the current state.
"""

__version__ = "0.4.0"

from aeromesh.geometry.loop import Loop, signed_area
from aeromesh.geometry.corners import (
    Corners,
    detect_corners,
    fluid_interior_angle,
    vertex_turns,
)
from aeromesh.geometry.cusp import find_cusps, open_cusp
from aeromesh.geometry.airfoil import load_airfoil, profile_to_loop
from aeromesh.geometry.region import (
    BoundaryLoop,
    Region,
    largest_inscribed_circle,
    prepare_boundary,
)
from aeromesh.medial.axis import (
    EdgeKind,
    MedialAxis,
    VertexKind,
    exterior_axis,
    interior_axis,
)
from aeromesh.medial.fields import optimum_flow_index
from aeromesh.topology.singularities import (
    Corner,
    Singularity,
    SingularityField,
    flux_residual,
    solve_singularities,
)
from aeromesh.domain.farfield import (
    box_farfield,
    c_farfield,
    circle_farfield,
    offset_farfield,
    rectangle,
)


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
    "BoundaryLoop", "Region", "largest_inscribed_circle", "prepare_boundary",
    "circle_farfield", "c_farfield", "box_farfield", "offset_farfield", "rectangle",
    "MedialAxis", "VertexKind", "EdgeKind", "exterior_axis", "interior_axis",
    "optimum_flow_index",
    "Corner", "Singularity", "SingularityField", "flux_residual",
    "solve_singularities",
    "build_region",
]
