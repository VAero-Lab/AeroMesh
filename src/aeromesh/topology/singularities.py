"""Singularity solver and the index budget (S2).

Mesh singularities are **solved for**, not looked up. Given the medial angle
along the axis, Fogg's flux balance decides where the cross-field must break and
by how much; nothing here consults a table of block patterns.

Two layers:

*Flux balance.* The optimum mesh flow across the axis is
``n = round((pi - theta_m) / (pi/2))`` -- the number of quarter-turns a cross
makes crossing it. The residual flux on a medial radius pair is then
``Phi = (pi - theta_m) - n * pi/2``, which the choice of ``n`` keeps inside
(-pi/4, pi/4]. A flux imbalance, and so a singularity, can only occur at three
kinds of position (Fogg sec. 3.5): where ``theta_m`` crosses the critical angles
pi/4 or 3pi/4, at medial vertices, and where a concave corner's reference
direction switches.

*Index budget.* Independently of any of that, discrete Gauss-Bonnet for a quad
mesh fixes the total. For a mesh with interior valences ``v`` and boundary
valences ``v_b``,

    sum_interior (4 - v) + sum_boundary (3 - v_b) = 4 * chi

With Bunin's ``k = v - 4`` and ``n_c = v_b - 1`` quads at a boundary vertex,

    sum_interior k = sum_corners (2 - n_c) - 4 * chi

This is a constraint the answer has to satisfy, derived from a different
direction, and it is what turns "this blocking looks reasonable" into "this
blocking is admissible". Verified against six configurations whose correct
blocking is known -- square, disk, triangle, L-shape, annulus -- and it
reproduces Fogg's stated ``k = 2`` at the centre of a regular hexagon, which was
not used to derive it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aeromesh.medial.axis import MedialAxis, VertexKind
from aeromesh.medial.fields import optimum_flow_index

CRITICAL_ANGLES = (np.pi / 4, 3 * np.pi / 4)


def flux_residual(theta_m, n=None):
    """Residual flux on a medial radius pair, ``(pi - theta_m) - n * pi/2``.

    With the optimum ``n`` this lies in (-pi/4, pi/4]; it is the part of the
    cross-field rotation that the chosen mesh flow does not absorb.
    """
    theta_m = np.asarray(theta_m, dtype=np.float64)
    if n is None:
        n = optimum_flow_index(theta_m)
    return (np.pi - theta_m) - np.asarray(n) * (np.pi / 2)


# ═══════════════════════════════════════════════════════════════════
#  Records
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Corner:
    """A C0 corner of the fluid boundary, with the element count it implies."""

    position: np.ndarray
    fluid_angle: float
    n_c: int

    @property
    def index_contribution(self) -> int:
        """``2 - n_c``: this corner's term in the budget."""
        return 2 - self.n_c


@dataclass(frozen=True)
class Singularity:
    """A mesh singularity: a node where other than four quads meet."""

    position: np.ndarray
    k: int
    source: str          # "crossing" | "vertex" | "finite_contact" | "merged"
    r_m: float = 0.0
    theta_m: float = 0.0
    n_merged: int = 1

    def __repr__(self) -> str:
        return (f"Singularity(k={self.k:+d}, {self.source}, "
                f"pos=[{self.position[0]:.4g}, {self.position[1]:.4g}], "
                f"r_m={self.r_m:.4g})")


@dataclass
class SingularityField:
    """The solved singularities, the corners, and the budget they must satisfy."""

    singularities: list[Singularity] = field(default_factory=list)
    corners: list[Corner] = field(default_factory=list)
    chi: int = 0

    @property
    def k_total(self) -> int:
        return int(sum(s.k for s in self.singularities))

    @property
    def k_required(self) -> int:
        """``sum_corners (2 - n_c) - 4 * chi``."""
        return int(sum(c.index_contribution for c in self.corners) - 4 * self.chi)

    def budget(self) -> dict:
        """The index certificate."""
        return {
            "k_total": self.k_total,
            "k_required": self.k_required,
            "residual": self.k_total - self.k_required,
            "chi": self.chi,
            "corner_term": int(sum(c.index_contribution for c in self.corners)),
            "ok": self.k_total == self.k_required,
        }

    def by_source(self) -> dict:
        out: dict[str, int] = {}
        for s in self.singularities:
            out[s.source] = out.get(s.source, 0) + 1
        return out

    def __repr__(self) -> str:
        b = self.budget()
        state = "BALANCED" if b["ok"] else f"residual {b['residual']:+d}"
        return (f"SingularityField(n={len(self.singularities)}, "
                f"k_total={b['k_total']:+d}, required={b['k_required']:+d}, {state})")


# ═══════════════════════════════════════════════════════════════════
#  The three critical position classes
# ═══════════════════════════════════════════════════════════════════

def critical_crossings(axis: MedialAxis, corner_pts=None,
                       tol: float = 0.0,
                       vertex_guard: float = 1.0) -> list[Singularity]:
    """Class 1: points where ``theta_m`` crosses pi/4 or 3pi/4 on a medial edge.

    The optimum flow index ``n`` is piecewise constant and can only change at
    these angles, so a crossing is where the mesh flow has to change by a
    quarter-turn and a singularity is needed to reconcile the two sides.

    The type is ``k = -|dn|``: one negative singularity per critical angle
    crossed, independent of which way ``theta_m`` is moving. The thin-sliver
    derivation suggests a signed ``k = -dn``, but that reverses with the
    direction of travel along the edge, and two closed-form checks rule it out.
    An ellipse is a disk-like region with no corners, so the budget requires
    ``sum k = -4``; its axis always yields exactly four crossings with
    ``dn = [-1, -1, +1, +1]``, which sums to zero under either signed rule and to
    -4 under ``-|dn|``, at every aspect ratio from 2:1 to 1.01:1. And an ellipse
    must tend to a circle, whose medial axis is a single finite-contact vertex
    with ``k = -floor(2*pi / (pi/2)) = -4`` -- computed by an entirely separate
    route. Only ``-|dn|`` is continuous in that limit.

    Positive singularities therefore never arise here; they come from medial
    vertices and from concave corners.

    Fogg restricts this class to crossings **with touching points on smooth
    boundary edges**. Two kinds of crossing are therefore excluded. One whose
    touches have converged on a corner is not a class-1 event: that corner's
    influence is already carried by its ``n_c`` term. Neither is one sitting
    within a medial radius of a medial vertex -- a branch point or a
    finite-contact end -- where the flow index is changing because of the
    vertex, and that is the vertex's own balance. Counting either here
    double-counts it.
    """
    guard = []
    for _, d in axis.graph.nodes(data=True):
        if d.get("anchor"):
            continue
        if d["kind"] in (VertexKind.NORMAL, VertexKind.FINITE_CONTACT):
            guard.append((d["pos"], vertex_guard * d["r_m"]))

    out: list[Singularity] = []
    for u, v, d in axis.graph.edges(data=True):
        theta = d["theta_m"]
        n = optimum_flow_index(theta)
        jumps = np.flatnonzero(np.diff(n) != 0)
        for j in jumps:
            if corner_pts is not None and len(corner_pts):
                touch = np.vstack([d["t1"][j], d["t2"][j]])
                if np.linalg.norm(touch[:, None, :] - corner_pts[None, :, :],
                                  axis=2).min() <= tol:
                    continue
            here = d["polyline"][j]
            if any(np.linalg.norm(here - gp) <= gr for gp, gr in guard):
                continue
            dn = int(n[j + 1] - n[j])
            # Interpolate the position onto the critical angle itself.
            t0, t1 = theta[j], theta[j + 1]
            target = min(CRITICAL_ANGLES, key=lambda c: abs(0.5 * (t0 + t1) - c))
            w = 0.0 if abs(t1 - t0) < 1e-15 else np.clip((target - t0) / (t1 - t0), 0.0, 1.0)
            pos = d["polyline"][j] + w * (d["polyline"][j + 1] - d["polyline"][j])
            out.append(Singularity(position=pos, k=-abs(dn), source="crossing",
                                   r_m=float(d["r_m"][j]), theta_m=float(target)))
    return out


def finite_contact_singularities(axis: MedialAxis) -> list[Singularity]:
    """Class 3: a vertex whose touching circle meets the boundary along an arc.

    ``k = -floor(contact_extent / (pi/2))``. The quantity is the *angular extent
    of contact*, not the largest angle between two touches: a semicircular end
    spans pi and gives k = -2, a full circle spans 2*pi and gives k = -4, which
    is the four superimposed negative singularities of the 0-sided template.
    """
    out: list[Singularity] = []
    for _, d in axis.graph.nodes(data=True):
        if d["kind"] is not VertexKind.FINITE_CONTACT:
            continue
        extent = float(d.get("contact_extent", d["theta_m"]))
        out.append(Singularity(position=d["pos"],
                               k=-int(np.floor(extent / (np.pi / 2))),
                               source="finite_contact",
                               r_m=float(d["r_m"]), theta_m=extent))
    return out


def merge(singularities: list[Singularity], factor: float = 0.75,
          floor: float = 0.0) -> list[Singularity]:
    """Combine singularities closer together than a fraction of the local r_m.

    Fogg puts the sensible range at [r_m/4, r_m] and uses 3*r_m/4 in his worked
    example. Merging sums the types, so a plus-minus pair cancels to a regular
    grid point and disappears; two of a sign become one higher-order singularity.
    Without it a decomposition fragments around clusters that the mesh would not
    resolve anyway.
    """
    if not singularities:
        return []
    pos = np.array([s.position for s in singularities])
    rad = np.array([s.r_m for s in singularities])
    n = len(singularities)
    unused = set(range(n))
    groups: list[list[int]] = []
    while unused:
        seed = unused.pop()
        group, frontier = [seed], [seed]
        while frontier:
            i = frontier.pop()
            tol = max(factor * rad[i], floor)
            for j in list(unused):
                if np.linalg.norm(pos[j] - pos[i]) <= tol:
                    unused.discard(j)
                    group.append(j)
                    frontier.append(j)
        groups.append(group)
    groups = [[singularities[i] for i in g] for g in groups]

    out: list[Singularity] = []
    for g in groups:
        k = int(sum(x.k for x in g))
        if k == 0:
            continue                      # a cancelling pair is a regular point
        pos = np.mean([x.position for x in g], axis=0)
        out.append(Singularity(position=pos, k=k,
                               source=g[0].source if len(g) == 1 else "merged",
                               r_m=float(np.mean([x.r_m for x in g])),
                               theta_m=float(np.mean([x.theta_m for x in g])),
                               n_merged=len(g)))
    return out


def solve_singularities(axis: MedialAxis, region, merge_factor: float = 0.75,
                        do_merge: bool = True) -> SingularityField:
    """Solve for the singularity field of a region, and assemble its budget.

    Parameters
    ----------
    axis : MedialAxis
        The exterior axis of ``region``.
    region : Region
    merge_factor : float
        Merging tolerance as a fraction of the local medial radius; Fogg's
        range is [0.25, 1.0].
    do_merge : bool
        Set False to inspect the raw solution before cancellation.

    Returns
    -------
    SingularityField

    Notes
    -----
    A non-zero budget residual is reported rather than absorbed: a field is only
    reported balanced when the solved singularities account for the budget on
    their own.
    """
    corners: list[Corner] = []
    for b in region.boundaries:
        if not len(b.corners):
            continue
        for pos, ang, nc in zip(b.corners.positions(b.loop), b.fluid_angles,
                                b.corner_elements):
            corners.append(Corner(position=pos, fluid_angle=float(ang), n_c=int(nc)))

    corner_pts = (np.array([c.position for c in corners]) if corners
                  else np.empty((0, 2)))
    spacing = min(b.loop.perimeter / b.loop.n_points for b in region.boundaries)
    found = (critical_crossings(axis, corner_pts, tol=5.0 * spacing)
             + finite_contact_singularities(axis)
             + vertex_singularities(axis)
             + concave_switch_singularities(axis, region))
    if do_merge:
        found = merge(found, factor=merge_factor)

    return SingularityField(singularities=found, corners=corners,
                            chi=region.euler_characteristic)


def concave_switch_singularities(axis: MedialAxis, region,
                                 tol_factor: float = 3.0) -> list[Singularity]:
    """Class 2 of Fogg's positions: concave-corner reference switches.

    Where a medial point's touch is a *concave* corner of the fluid rather than
    a smooth edge, theta_m has to be measured from one of the cross directions
    at that corner, chosen to minimise the adjustment (Fogg sec. 3.1, Fig. 6).
    A corner of fluid angle ``alpha`` carrying ``n_c`` elements has its cross
    directions evenly spaced at ``alpha * i / n_c``. As the medial point travels,
    the nearest of those directions changes, and the reference jumps by one
    sector -- a singularity of type -1 per switch.

    This is the class that accounts for the singularities an airfoil's trailing
    edge demands. There is no medial branch at a trailing edge (a convex corner
    of a body is reflex for the fluid, so nothing emanates), but the corner is
    still a *touch point* for a stretch of the surrounding medial axis, and that
    is where its influence enters.
    """
    concave = []
    for b in region.boundaries:
        if not len(b.corners):
            continue
        pts = b.loop.points
        n = b.loop.n_points
        for idx, pos, ang, nc in zip(b.corners.index, b.corners.positions(b.loop),
                                     b.fluid_angles, b.corner_elements):
            if ang <= np.pi or nc < 2:
                continue                        # only concave corners switch
            # Edge directions leaving the corner, measured into the fluid.
            e_prev = pts[(idx - 1) % n] - pos
            e_next = pts[(idx + 1) % n] - pos
            concave.append((pos, float(ang), int(nc),
                            e_prev / np.linalg.norm(e_prev),
                            e_next / np.linalg.norm(e_next)))
    if not concave:
        return []

    spacing = min(b.loop.perimeter / b.loop.n_points for b in region.boundaries)
    out: list[Singularity] = []
    for _, _, d in axis.graph.edges(data=True):
        for touch, other in ((d["t1"], d["t2"]), (d["t2"], d["t1"])):
            for cpos, ang, nc, e_a, e_b in concave:
                near = np.linalg.norm(touch - cpos, axis=1) <= tol_factor * spacing
                if near.sum() < 3:
                    continue
                # Angle of the medial radius, measured from one corner edge,
                # swept through the fluid. Which of the two edges starts the
                # fluid wedge depends on the corner's orientation, so pick the
                # one that keeps the sweep inside [0, alpha] rather than assume.
                w = d["polyline"][near] - cpos
                w = w / np.maximum(np.linalg.norm(w, axis=1), 1e-300)[:, None]
                cands = []
                for e in (e_a, e_b):
                    a_ = np.mod(np.arctan2(w[:, 0] * e[1] - w[:, 1] * e[0],
                                           w[:, 0] * e[0] + w[:, 1] * e[1]),
                                2 * np.pi)
                    cands.append((max(0.0, float(a_.max()) - ang), a_))
                phi = min(cands, key=lambda c: c[0])[1]
                # Sector index: which of the n_c cross sectors the radius is in.
                sector = np.clip(np.floor(phi / (ang / nc)).astype(int), 0, nc - 1)
                jumps = np.flatnonzero(np.diff(sector) != 0)
                poly = d["polyline"][near]
                rm = d["r_m"][near]
                th = d["theta_m"][near]
                for j in jumps:
                    out.append(Singularity(
                        position=0.5 * (poly[j] + poly[j + 1]),
                        k=-abs(int(sector[j + 1] - sector[j])),
                        source="concave_switch",
                        r_m=float(rm[j]), theta_m=float(th[j])))
    return out


def _flow_index_near(axis: MedialAxis, node, edge_data) -> int:
    """Optimum flow index on an incident edge, read a local radius from the vertex.

    Not at the vertex itself, where theta_m is the vertex's own value rather than
    the edge's, and not far along it, where the edge may have crossed a critical
    angle. One medial radius out is the natural local scale.
    """
    poly, arc, theta = edge_data["polyline"], edge_data["s"], edge_data["theta_m"]
    pos = axis.graph.nodes[node]["pos"]
    at_start = (np.linalg.norm(poly[0] - pos) <= np.linalg.norm(poly[-1] - pos))
    dist = arc if at_start else arc[-1] - arc
    target = min(max(axis.graph.nodes[node]["r_m"], 1e-12), 0.5 * arc[-1])
    return int(optimum_flow_index(theta[int(np.argmin(np.abs(dist - target)))]))


def vertex_singularities(axis: MedialAxis) -> list[Singularity]:
    """Class 2: flux balance at medial vertices.

    Take a control region around the vertex, cut across each incident medial
    edge one local radius out. Each cut carries ``n_j`` quarter-turns of the
    cross field, which enters the balance exactly as a boundary corner's
    ``n_c`` does, so the same accounting that gives
    ``sum k = sum (2 - n_c) - 4*chi`` globally gives, for the disc around one
    vertex,

        k_V = sum_j (2 - n_j) - 4

    over its incident edges. Domain corners are not counted here; they are
    already carried by the global corner term, and counting them twice would
    double them.

    Verified against six polygons whose correct blocking is known: triangle -1,
    square 0, pentagon +1, hexagon **+2** (the value Fogg states, obtained here
    from the medial axis of an actual hexagon), rectangle 0 across its two
    vertices, and a stadium, whose semicircular ends carry the whole budget and
    leave its vertices nothing to supply. An L-shape is still off by one.

    Note this reduces to ``m - 4`` whenever every incident edge has ``n_j = 1``,
    which is why that simpler rule worked on the regular polygons and failed on
    a rectangle, where the central edge has ``n = 0``.
    """
    out: list[Singularity] = []
    for nd, d in axis.graph.nodes(data=True):
        if d.get("anchor") or d["kind"] is not VertexKind.NORMAL:
            continue
        ns = [_flow_index_near(axis, nd, e)
              for _, _, e in axis.graph.edges(nd, data=True)]
        k = int(sum(2 - n for n in ns) - 4)
        if k == 0:
            continue
        out.append(Singularity(position=d["pos"], k=k, source="vertex",
                               r_m=float(d["r_m"]), theta_m=float(d["theta_m"])))
    return out
