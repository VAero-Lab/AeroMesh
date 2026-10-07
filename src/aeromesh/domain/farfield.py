"""Far-field boundary generation.

Where the domain is truncated is a modelling choice -- not derivable from the
geometry, and one of the few things the pipeline is told rather than computes.

Every function here returns a plain ``Loop``. **The shape is never recorded and
never reaches any later stage**: ``Region`` stores curves, not names, so nothing
downstream can branch on which constructor was used. That is the line that
matters. Choosing a C-shaped outer curve does not select a "C topology" -- it
supplies two corners, and the flux balance decides what to do with them.

The shape does, however, *influence* the topology, through its corners: a
corner of the fluid boundary generates a medial flare, and a flare generates a
block. Measured on a NACA 2412 at five chords -- circle 0 flares, C-shape 2,
box 4. So the choice is a real modelling decision and the standard shapes are
offered as standard shapes.

| constructor          | corners | conventional name |
|----------------------|---------|-------------------|
| ``circle_farfield``  | 0       | O-type outer boundary (the default) |
| ``c_farfield``       | 2       | C-type outer boundary |
| ``box_farfield``     | 4       | H-type outer boundary / tunnel walls |
| ``offset_farfield``  | 0       | none -- a distance level set, for tight or multi-body domains |
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from aeromesh.geometry.loop import Loop


def _body_extent(bodies):
    """Bounding box and centre of a body set, with a guard on emptiness."""
    if not bodies:
        raise ValueError("A far field needs at least one body.")
    pts = np.vstack([b.points for b in bodies])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return lo, hi, 0.5 * (lo + hi)


def _check_encloses(loop: Loop, bodies, what: str) -> Loop:
    for b in bodies:
        if not bool(loop.contains(b.points).all()):
            raise ValueError(
                f"{what} does not enclose body {b.name!r}. Increase the "
                f"far-field extent, or pass an explicit outer Loop."
            )
    return loop


def circle_farfield(
    bodies: list[Loop] | tuple[Loop, ...],
    radius: float,
    n_points: int = 720,
    centre: np.ndarray | None = None,
) -> Loop:
    """A circular outer boundary -- the conventional O-type far field.

    The default for :func:`aeromesh.build_region`. Smooth, so it contributes no
    corners of its own and no flares: the entire topology then comes from the
    bodies. This is what pyHyp and most O-mesh studies use, and it is the
    easiest curve to compare published results against.

    Parameters
    ----------
    bodies : sequence of Loop
        Used to place and check the circle; all of them must end up inside.
    radius : float
        Circle radius, in the bodies\' own units (chords, for a unit-chord
        section).
    n_points : int
        Points on the curve.
    centre : np.ndarray, optional
        Defaults to the centre of the body set\'s bounding box.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    _, _, c = _body_extent(bodies)
    if centre is not None:
        c = np.asarray(centre, dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    loop = Loop.from_points(
        np.column_stack([c[0] + radius * np.cos(t), c[1] + radius * np.sin(t)]),
        name=f"circle(R={radius:g})")
    return _check_encloses(loop, bodies, f"A circle of radius {radius:g}")


def c_farfield(
    bodies: list[Loop] | tuple[Loop, ...],
    radius: float,
    wake_length: float,
    n_arc: int = 400,
    n_straight: int = 220,
) -> Loop:
    """A C-type outer boundary: upstream semicircle, straight sides, outflow plane.

    The semicircle is centred on the body set\'s leading point and joins the
    straight sides tangentially, so the only corners are the two downstream
    ones where the sides meet the outflow plane. Those two corners are what
    generate the two flares of a C-type blocking -- but nothing here decides
    that; the flux balance does.

    Parameters
    ----------
    bodies : sequence of Loop
    radius : float
        Semicircle radius and half-height of the straight sides.
    wake_length : float
        Distance from the body set\'s trailing point to the outflow plane.
    n_arc, n_straight : int
        Points on the arc and on each straight run.
    """
    if radius <= 0 or wake_length <= 0:
        raise ValueError("radius and wake_length must be positive.")
    lo, hi, c = _body_extent(bodies)
    x_nose, x_out, y_c = lo[0], hi[0] + wake_length, c[1]

    t = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc)
    arc = np.column_stack([x_nose + radius * np.cos(t), y_c + radius * np.sin(t)])
    bottom = np.column_stack([np.linspace(x_nose, x_out, n_straight, endpoint=False),
                              np.full(n_straight, y_c - radius)])
    outflow = np.column_stack([np.full(n_straight, x_out),
                               np.linspace(y_c - radius, y_c + radius, n_straight,
                                           endpoint=False)])
    top = np.column_stack([np.linspace(x_out, x_nose, n_straight, endpoint=False),
                           np.full(n_straight, y_c + radius)])
    loop = Loop.from_points(np.vstack([arc[1:-1], bottom, outflow, top]),
                            name=f"c-shape(R={radius:g},wake={wake_length:g})")
    return _check_encloses(loop, bodies, f"A C-shape of radius {radius:g}")


def box_farfield(
    bodies: list[Loop] | tuple[Loop, ...],
    upstream: float,
    downstream: float,
    lateral: float,
    n_points: int = 720,
) -> Loop:
    """A rectangular outer boundary -- the conventional H-type far field.

    Also the right choice for wind-tunnel walls, where the rectangle is
    physical rather than a truncation. Its four corners generate four flares,
    which is the mechanism behind the familiar ring-plus-four-blocks topology.

    Margins are measured from the body set\'s bounding box, in the bodies\'
    own units.
    """
    if min(upstream, downstream, lateral) <= 0:
        raise ValueError("All margins must be positive.")
    lo, hi, _ = _body_extent(bodies)
    loop = rectangle(lo[0] - upstream, lo[1] - lateral,
                     hi[0] + downstream, hi[1] + lateral,
                     n_points=n_points,
                     name=f"box(-{upstream:g},+{downstream:g},±{lateral:g})")
    return _check_encloses(loop, bodies, "The box")


def offset_farfield(
    bodies: list[Loop] | tuple[Loop, ...],
    distance: float,
    n_points: int = 720,
    downstream: float | None = None,
    wake_direction: np.ndarray | None = None,
    centre: np.ndarray | None = None,
    hull: bool = True,
) -> Loop:
    """Far-field boundary as a level set of the distance to the body set.

    Not the default. Use it when a conventional shape wraps the configuration
    badly -- a tight domain, or bodies spread far apart -- where a circle large
    enough to contain everything wastes cells. Like a circle it contributes no
    corners, so it yields the same topology a circle would; its advantage is
    extent, not structure.

    Parameters
    ----------
    bodies : sequence of Loop
        The solid bodies. All of them are taken into account, so the far field
        of a three-element configuration wraps the group.
    distance : float
        Truncation distance from the nearest body, in the bodies' own units.
    n_points : int
        Number of points on the returned curve.
    downstream : float, optional
        Truncation distance in the wake direction, if it should differ from
        ``distance``. Blended smoothly with the ray angle, so the curve stays
        C1 and picks up no artificial corners.
    wake_direction : np.ndarray, optional
        Unit vector the ``downstream`` distance applies along. Defaults to +x.
    centre : np.ndarray, optional
        Origin for the radial construction. Defaults to the midpoint of the
        body set's bounding box.
    hull : bool
        Measure the distance to the convex hull of the body set rather than to
        the bodies themselves. The outward offset of a convex set is C1, so the
        result is smooth; offsetting a union of separated bodies instead leaves
        a concave crease wherever the nearest-body branch switches, and that
        crease is a real corner that would drive real topology. Since where the
        domain is truncated is arbitrary, the smooth choice is the honest one.
        Set False to get the true per-body level set.

    Returns
    -------
    Loop
        A closed CCW curve enclosing every body.

    Raises
    ------
    ValueError
        If ``distance`` is too small for the level set to be a single curve
        around the whole group -- which happens when widely separated bodies
        each get their own contour. Increase ``distance``.
    """
    if not bodies:
        raise ValueError("offset_farfield needs at least one body.")
    if distance <= 0:
        raise ValueError("distance must be positive.")

    pts = np.vstack([b.points for b in bodies])
    if hull and len(pts) >= 3:
        ring = pts[ConvexHull(pts).vertices]
        # Densify the hull edges so a point-based distance query approximates
        # the distance to the hull boundary rather than to its vertices.
        edge = np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1)
        step = max(edge.min(), edge.sum() / 4000.0)
        dense = [ring[i] + np.outer(np.linspace(0, 1, max(2, int(edge[i] / step)) + 1)[:-1],
                                    ring[(i + 1) % len(ring)] - ring[i])
                 for i in range(len(ring))]
        query_pts = np.vstack(dense)
    else:
        query_pts = pts
    tree = cKDTree(query_pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(np.hypot(*(hi - lo)))
    if centre is None:
        centre = 0.5 * (lo + hi)
    centre = np.asarray(centre, dtype=np.float64)

    if downstream is None:
        downstream = distance
    if wake_direction is None:
        wake_direction = np.array([1.0, 0.0])
    w = np.asarray(wake_direction, dtype=np.float64)
    w = w / np.linalg.norm(w)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    rays = np.column_stack([np.cos(theta), np.sin(theta)])
    # Smooth blend: full downstream weight along w, none at 90 degrees or more.
    align = np.clip(rays @ w, 0.0, 1.0) ** 2
    target = distance + (downstream - distance) * align

    far = span + 2.0 * max(distance, downstream)
    out = np.empty((n_points, 2))
    for i, (d, tgt) in enumerate(zip(rays, target)):
        lo_t, hi_t = 0.0, far
        for _ in range(70):
            mid = 0.5 * (lo_t + hi_t)
            if tree.query(centre + mid * d)[0] < tgt:
                lo_t = mid
            else:
                hi_t = mid
        out[i] = centre + 0.5 * (lo_t + hi_t) * d

    loop = Loop.from_points(out, name=f"farfield(d={distance:g})")

    # The radial construction only recovers a star-shaped level set. Verify
    # rather than assume: every body must end up strictly inside.
    for b in bodies:
        if not bool(loop.contains(b.points).all()):
            raise ValueError(
                f"The level set at distance {distance:g} is not a single "
                f"star-shaped curve around the body set (body {b.name!r} is not "
                f"enclosed). Increase distance, or pass an explicit outer Loop."
            )
    return loop


def rectangle(x_min: float, y_min: float, x_max: float, y_max: float,
              n_points: int = 720, name: str = "rectangle") -> Loop:
    """A rectangular curve, for wind-tunnel walls or a prescribed outer box.

    This is a curve constructor, not a domain type: the four corners it carries
    are found by the corner detector like any others, and what topology they
    produce is derived downstream.
    """
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("rectangle needs x_max > x_min and y_max > y_min.")
    w, h = x_max - x_min, y_max - y_min
    per = 2.0 * (w + h)
    n_w = max(2, int(round(n_points * w / per)))
    n_h = max(2, int(round(n_points * h / per)))
    bottom = np.column_stack([np.linspace(x_min, x_max, n_w, endpoint=False),
                              np.full(n_w, y_min)])
    right = np.column_stack([np.full(n_h, x_max),
                             np.linspace(y_min, y_max, n_h, endpoint=False)])
    top = np.column_stack([np.linspace(x_max, x_min, n_w, endpoint=False),
                           np.full(n_w, y_max)])
    left = np.column_stack([np.full(n_h, x_min),
                            np.linspace(y_max, y_min, n_h, endpoint=False)])
    return Loop.from_points(np.vstack([bottom, right, top, left]), name=name)
