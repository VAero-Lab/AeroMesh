"""Medial axis extraction.

Two entry points, and they differ only in which region is being skeletonised:

    exterior_axis(region)   the fluid: inside the outer loop, outside every body
    interior_axis(loop)     one body's own interior

The exterior axis supplies everything between and around the bodies. The
interior axis supplies each body's own shape, and unlike the exterior one it is
scale-free -- it does not care where the domain is truncated, which is what makes
it survive a far field at fifty chords.

Membership is geometric, never label-based. A Voronoi ridge is medial when its
two generators lie on different loops, or far enough apart on the same one. The
first iteration accepted a ridge because its generators carried different
*markers*, which made the skeleton depend on how the boundary happened to be
labelled and invented a branch of radius zero at every leading edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi, cKDTree

from aeromesh.geometry.loop import Loop
from aeromesh.geometry.region import Region
from aeromesh.medial.fields import optimum_flow_index, segment_distance, vertex_fields


class VertexKind(Enum):
    """Medial vertex classification, after Rigby's Table 1."""

    NORMAL = auto()          # three or more distinct touches: a branch point
    CORNER = auto()          # terminates at a C0 corner of the boundary, r_m -> 0
    DANGLE = auto()          # curvature contact: one touch, theta_m -> 0
    FINITE_CONTACT = auto()  # the touching circle meets the boundary along an arc


class EdgeKind(Enum):
    """Medial edge classification, after Rigby's Table 2."""

    PRIMARY = auto()         # NORMAL - NORMAL
    FLARE = auto()           # NORMAL - CORNER
    DANGLE = auto()          # NORMAL - DANGLE
    CORNER_CORNER = auto()
    CORNER_DANGLE = auto()
    DANGLE_DANGLE = auto()
    LOOP = auto()            # a closed edge with no vertices at all


_EDGE_KIND = {
    frozenset({VertexKind.NORMAL}): EdgeKind.PRIMARY,
    frozenset({VertexKind.NORMAL, VertexKind.CORNER}): EdgeKind.FLARE,
    frozenset({VertexKind.NORMAL, VertexKind.DANGLE}): EdgeKind.DANGLE,
    frozenset({VertexKind.CORNER}): EdgeKind.CORNER_CORNER,
    frozenset({VertexKind.CORNER, VertexKind.DANGLE}): EdgeKind.CORNER_DANGLE,
    frozenset({VertexKind.DANGLE}): EdgeKind.DANGLE_DANGLE,
}


@dataclass
class MedialAxis:
    """A medial axis, as a graph carrying its geometric fields.

    Nodes are the key medial vertices (degree != 2, plus finite-contact points).
    Degree-2 chains are collapsed onto the edges, which carry the sampled fields.

    Node attributes
        ``pos``      (2,) position
        ``r_m``      radius of the maximal inscribed circle
        ``theta_m``  largest angle subtended by two distinct touches
        ``kind``     :class:`VertexKind`
        ``touch``    (n, 2) touch points
        ``touch_index`` (n,) index of the nearest boundary sample to each
        ``normals``  (n, 2) unit vectors toward each touch point

    Edge attributes
        ``polyline`` (N, 2) the medial curve
        ``s``        (N,) arc length along it, from 0
        ``r_m``      (N,)
        ``theta_m``  (N,)
        ``n1``,``n2``(N, 2) unit vectors to the two nearest touches
        ``t1``,``t2``(N, 2) the touch points themselves
        ``i1``,``i2``(N,) boundary sample index of each touch, so a block can
                     walk the arc between two of them
        ``kind``     :class:`EdgeKind`
        ``length``   float
    """

    graph: nx.MultiGraph = field(default_factory=nx.MultiGraph)
    n_bodies: int = 0

    # ── structure ───────────────────────────────────────────────────

    @property
    def n_vertices(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def n_components(self) -> int:
        return nx.number_connected_components(self.graph) if self.n_vertices else 0

    @property
    def betti_1(self) -> int:
        """Number of independent cycles, ``E - V + C``.

        The medial axis deformation-retracts onto its region, so for the fluid
        region this must equal the number of bodies. That identity is the
        cheapest correctness check in the pipeline and it is exact.
        """
        if self.n_vertices == 0:
            return 0
        return self.n_edges - self.n_vertices + self.n_components

    def certificate(self) -> dict:
        """The retract check: ``betti_1 == n_bodies``."""
        return {
            "betti_1": self.betti_1,
            "n_bodies": self.n_bodies,
            "ok": self.betti_1 == self.n_bodies,
        }

    def kinds(self) -> dict:
        """Count of each vertex kind.

        Representation anchors are excluded: a bare loop has no medial vertices,
        so this is empty for the doughnut case even though the graph holds one
        node to carry the closed edge.
        """
        out: dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            if d.get("anchor"):
                continue
            out[d["kind"].name] = out.get(d["kind"].name, 0) + 1
        return out

    @property
    def n_real_vertices(self) -> int:
        """Vertices that are genuine medial vertices, anchors excluded."""
        return sum(1 for _, d in self.graph.nodes(data=True) if not d.get("anchor"))

    def edge_kinds(self) -> dict:
        out: dict[str, int] = {}
        for _, _, d in self.graph.edges(data=True):
            out[d["kind"].name] = out.get(d["kind"].name, 0) + 1
        return out

    # ── fields ──────────────────────────────────────────────────────

    def sample_points(self) -> np.ndarray:
        """Every medial point, vertices and edge samples, as one (M, 2) array."""
        chunks = [np.atleast_2d(d["pos"]) for _, d in self.graph.nodes(data=True)]
        chunks += [d["polyline"] for _, _, d in self.graph.edges(data=True)]
        return np.vstack(chunks) if chunks else np.empty((0, 2))

    def theta_range(self) -> tuple[float, float]:
        vals = [d["theta_m"] for _, _, d in self.graph.edges(data=True)]
        if not vals:
            return (float("nan"), float("nan"))
        allv = np.concatenate(vals)
        return float(allv.min()), float(allv.max())

    def flow_index_histogram(self) -> dict:
        """Fogg's ``n`` over every edge sample. Non-zero entries are where the
        flux balance will place singularities."""
        vals = [optimum_flow_index(d["theta_m"]) for _, _, d in self.graph.edges(data=True)]
        if not vals:
            return {}
        v, c = np.unique(np.concatenate(vals), return_counts=True)
        return dict(zip(v.tolist(), c.tolist()))

    def __repr__(self) -> str:
        cert = self.certificate()
        return (f"MedialAxis(V={self.n_vertices}, E={self.n_edges}, "
                f"C={self.n_components}, b1={cert['betti_1']}, "
                f"bodies={self.n_bodies}, {'certified' if cert['ok'] else 'FAILS RETRACT'})")


# ═══════════════════════════════════════════════════════════════════
#  Construction
# ═══════════════════════════════════════════════════════════════════

def _raw_graph(pts, segment_id, segment_closed, inside, min_sep_frac, min_sep_points):
    """Voronoi vertices inside the region, joined by geometrically medial ridges.

    The test is on *smooth boundary elements*, not loops. Two generators on
    different elements always give a medial ridge, however close together they
    are -- that is what lets a branch terminate at a trailing edge. Two on the
    same element have to be far apart, because otherwise their bisector is the
    sampling artefact between adjacent samples, not a feature of the shape.

    Within a closed element (a loop with no corners at all) the index distance
    wraps. Within an open one it does not, so the two samples either side of a
    corner are maximally far apart and their ridge is kept.
    """
    vor = Voronoi(pts)
    V = vor.vertices
    ok = inside(V)

    counts = np.bincount(segment_id, minlength=len(segment_closed))
    sep = np.maximum(min_sep_points, np.ceil(min_sep_frac * counts)).astype(int)
    # Index of each sample within its own segment.
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    local = np.arange(len(pts)) - starts[segment_id]

    G = nx.Graph()
    for ri, (v1, v2) in enumerate(vor.ridge_vertices):
        if v1 == -1 or v2 == -1 or not (ok[v1] and ok[v2]):
            continue
        a, b = vor.ridge_points[ri]
        sa, sb = segment_id[a], segment_id[b]
        if sa == sb:
            d = abs(int(local[a]) - int(local[b]))
            if segment_closed[sa]:
                d = min(d, counts[sa] - d)
            if d < sep[sa]:
                continue
        for v in (v1, v2):
            if v not in G:
                G.add_node(v, pos=V[v], gens=set())
        G.nodes[v1]["gens"].update((int(a), int(b)))
        G.nodes[v2]["gens"].update((int(a), int(b)))
        G.add_edge(v1, v2)
    return G


def _annotate(G, pts, loop_id, offsets, tree):
    """Attach r_m, theta_m, touches and normals to every raw node.

    Nodes with a single touch are then dropped. A medial point is by definition
    equidistant from two or more *distinct* boundary elements; a node whose
    generators all fall in one element is not on the medial axis, it is an
    artefact of the discretisation. These cluster at sharp convex corners of a
    body -- a trailing edge -- where the corner is the unique nearest point for
    a whole fan of directions and no medial branch exists at all.
    """
    nodes = list(G.nodes)
    if not nodes:
        return
    pos = np.array([G.nodes[n]["pos"] for n in nodes])
    r = segment_distance(pos, pts, loop_id, offsets, tree=tree)
    drop = []
    for n, p, rm in zip(nodes, pos, r):
        f = vertex_fields(p, np.fromiter(G.nodes[n]["gens"], dtype=int),
                          pts, loop_id, offsets, r_m=rm)
        if f["n_touch"] < 2:
            drop.append(n)
            continue
        G.nodes[n].update(f)
    G.remove_nodes_from(drop)
    G.remove_nodes_from(list(nx.isolates(G)))


def _corner_tol(r_m: float, spacing: float) -> float:
    """How close a branch tip must be to a known corner to be that corner's flare.

    A flare into a corner whose fluid interior angle is ``alpha`` reaches radius
    ``r_m`` at a distance ``r_m / sin(alpha / 2)`` from the corner, so the
    tolerance has to scale with ``r_m``; the factor 8 covers angles down to about
    15 degrees. The spacing term covers the last few samples, where the Voronoi
    simply stops short of the corner.

    A genuine dangle is never caught by this: its radius is the local osculating
    radius and the nearest corner is a whole feature away.
    """
    return 8.0 * r_m + 4.0 * spacing


def _prune(G, corner_pts, min_theta, spacing):
    """Drop branches whose medial angle never rises above ``min_theta``.

    A branch produced by boundary perturbation subtends a small object angle
    along its whole length; a real feature does not. Branches anchored at a known
    C0 corner are never pruned: a flare into a sharp corner legitimately has a
    small angle, and the corner is a fact about the geometry, not noise.
    """
    if G.number_of_nodes() == 0:
        return
    protected = cKDTree(corner_pts) if len(corner_pts) else None

    changed = True
    while changed:
        changed = False
        for leaf in [n for n in G.nodes if G.degree(n) == 1]:
            chain, prev, cur = [leaf], leaf, next(iter(G.neighbors(leaf)))
            while G.degree(cur) == 2:
                chain.append(cur)
                nxt = [q for q in G.neighbors(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if max(G.nodes[n]["theta_m"] for n in chain) >= min_theta:
                continue
            if protected is not None:
                if protected.query(G.nodes[leaf]["pos"])[0] <= _corner_tol(
                        G.nodes[leaf]["r_m"], spacing):
                    continue
            G.remove_nodes_from(chain)
            changed = True
    G.remove_nodes_from(list(nx.isolates(G)))


def contact_extent(position, r_m, pts, tol=0.002) -> float:
    """Angular extent over which the touching circle meets the boundary.

    Zero for a point contact, pi for a semicircular end, 2*pi for a full circle.

    The tolerance is deliberately tight. True finite contact means the boundary
    *is* a circular arc there, so the deviation from the touching circle is zero;
    a merely near-circular feature such as an airfoil nose departs from its
    osculating circle quickly and must not be mistaken for it.
    This is the quantity Fogg's ``k = -floor(extent / (pi/2))`` consumes at a
    finite-contact vertex (his Fig. 10), and it is not the same thing as the
    ``theta_m`` of a medial edge -- it is not capped at pi.
    """
    if r_m <= 0:
        return 0.0
    v = pts - position
    d = np.linalg.norm(v, axis=1)
    on = np.abs(d - r_m) <= tol * r_m
    if on.sum() < 3:
        return 0.0
    u = v[on] / d[on][:, None]
    ang = np.sort(np.arctan2(u[:, 1], u[:, 0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    # Contact all the way round shows up as every gap being the sampling step.
    # Compare against the median gap, not against 2*pi/n: the latter grows as
    # fewer points are in contact and so calls a small arc a full circle.
    if gaps.max() <= 4.0 * float(np.median(gaps)):
        return 2 * np.pi
    return float(2 * np.pi - gaps.max())


def _classify(G, corner_pts, spacing):
    """Type every key vertex: NORMAL, CORNER or DANGLE."""
    tree = cKDTree(corner_pts) if len(corner_pts) else None
    kinds, anchors = {}, {}
    for n in G.nodes:
        deg = G.degree(n)
        if deg >= 3:
            kinds[n] = VertexKind.NORMAL
        elif deg == 1:
            kinds[n] = VertexKind.DANGLE
            if tree is not None:
                d, i = tree.query(G.nodes[n]["pos"])
                if d <= _corner_tol(G.nodes[n]["r_m"], spacing):
                    kinds[n] = VertexKind.CORNER
                    anchors[n] = int(i)
    return kinds, anchors


def _corner_theta(alpha: float) -> float:
    """Medial angle at a flare tip, from the corner's fluid interior angle.

    Approaching a convex corner of angle ``alpha`` along its bisector, the two
    touch points are the feet on the two edges and the directions to them
    subtend ``pi - alpha``. Taking it analytically avoids the worst-conditioned
    quantity in the extraction: at the tip both touch points converge on the
    corner, so the measured angle is decided by the last surviving Voronoi
    vertex and moves with the sampling.
    """
    return float(np.clip(np.pi - alpha, 0.0, np.pi))


def _collapse(G, kinds, anchors, corner_pts, corner_ang, n_bodies):
    """Collapse degree-2 chains into edges carrying the sampled fields."""
    M = nx.MultiGraph()
    keys = [n for n in G.nodes if G.degree(n) != 2]

    for n in keys:
        d = G.nodes[n]
        pos = d["pos"]
        theta = d["theta_m"]
        if kinds[n] is VertexKind.CORNER and n in anchors:
            pos = corner_pts[anchors[n]]
            if corner_ang is not None:
                theta = _corner_theta(float(corner_ang[anchors[n]]))
        M.add_node(n, pos=pos, r_m=0.0 if kinds[n] is VertexKind.CORNER else d["r_m"],
                   theta_m=theta, kind=kinds[n],
                   touch=d["touch"], normals=d["normals"],
                   n_touch=int(d.get("n_touch", len(d["touch"]))),
                   touch_index=d.get("touch_index"),
                   contact_extent=float(d.get("contact_extent", 0.0)))

    def pack(chain, u, v):
        poly = np.array([G.nodes[n]["pos"] for n in chain])
        # A flare ends at the corner itself, where r_m is exactly zero.
        rm = np.array([G.nodes[n]["r_m"] for n in chain])
        th = np.array([G.nodes[n]["theta_m"] for n in chain])
        n1 = np.array([G.nodes[n]["normals"][0] for n in chain])
        n2 = np.array([G.nodes[n]["normals"][min(1, len(G.nodes[n]["normals"]) - 1)]
                       for n in chain])
        t1 = np.array([G.nodes[n]["touch"][0] for n in chain])
        t2 = np.array([G.nodes[n]["touch"][min(1, len(G.nodes[n]["touch"]) - 1)]
                       for n in chain])
        i1 = np.array([G.nodes[n]["touch_index"][0] for n in chain], dtype=int)
        i2 = np.array([G.nodes[n]["touch_index"][min(1, len(G.nodes[n]["touch_index"]) - 1)]
                       for n in chain], dtype=int)
        for end, idx in ((u, 0), (v, -1)):
            if end in kinds and kinds[end] is VertexKind.CORNER and end in anchors:
                poly[idx] = corner_pts[anchors[end]]
                rm[idx] = 0.0
                if corner_ang is not None:
                    th[idx] = _corner_theta(float(corner_ang[anchors[end]]))
        s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))))
        kind = (_EDGE_KIND.get(frozenset({kinds[u], kinds[v]}), EdgeKind.PRIMARY)
                if u in kinds and v in kinds else EdgeKind.LOOP)
        return dict(polyline=poly, s=s, r_m=rm, theta_m=th, n1=n1, n2=n2,
                    t1=t1, t2=t2, i1=i1, i2=i2, kind=kind, length=float(s[-1]))

    seen: set[frozenset] = set()
    for start in keys:
        for nb in list(G.neighbors(start)):
            if frozenset((start, nb)) in seen and G.degree(nb) != 2:
                continue
            chain, prev, cur = [start], start, nb
            while G.degree(cur) == 2:
                chain.append(cur)
                nxt = [q for q in G.neighbors(cur) if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            chain.append(cur)
            key = frozenset((start, nb)) if G.degree(nb) != 2 else frozenset(chain[:2])
            if key in seen:
                continue
            seen.add(key)
            seen.add(frozenset(chain[-2:]))
            M.add_edge(start, cur, **pack(chain, start, cur))

    # A skeleton with no key vertices at all is a bare loop -- the doughnut case.
    if not keys and G.number_of_nodes():
        cyc = nx.cycle_basis(G)
        if cyc:
            ring = cyc[0]
            anchor = ring[0]
            d = G.nodes[anchor]
            # A bare loop has no medial vertices at all -- Rigby's doughnut.
            # The node exists only so the graph can carry the closed edge, and
            # it is flagged so it is not reported as a vertex that is really
            # there. Betti-1 is unaffected: V = E = C = 1 gives b1 = 1.
            M.add_node(anchor, pos=d["pos"], r_m=d["r_m"], theta_m=d["theta_m"],
                       kind=VertexKind.NORMAL, touch=d["touch"],
                       normals=d["normals"], anchor=True,
                       n_touch=int(d.get("n_touch", len(d["touch"]))),
                       contact_extent=float(d.get("contact_extent", 0.0)))
            M.add_edge(anchor, anchor, **pack(ring + [anchor], anchor, anchor))
            M.edges[anchor, anchor, 0]["kind"] = EdgeKind.LOOP

    return MedialAxis(graph=M, n_bodies=n_bodies)


def _finite_contact(loop_or_region, pts, loop_id, offsets, n_bodies, kind_pos):
    """Fallback for a boundary whose medial axis collapses to a single point.

    An exact circle has a medial axis of one point with the touching circle in
    contact all the way round. A discrete Voronoi cannot represent that as a
    graph and returns nothing, so the vertex is emitted explicitly, carrying the
    largest angle it subtends -- which is what Fogg's
    ``k = -floor(max(theta_m) / (pi / 2))`` consumes at such a vertex.
    """
    centre, radius = kind_pos
    vec = pts - centre
    d = np.linalg.norm(vec, axis=1)
    near = d <= radius * 1.02
    if not near.any():
        near = d <= np.percentile(d, 5.0)
    u = vec[near] / d[near][:, None]
    # Largest angle subtended by any two contact directions.
    ang = np.arctan2(u[:, 1], u[:, 0])
    ang.sort()
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    # The *angular extent of contact*, which is not the same quantity as the
    # theta_m of a medial edge and is not capped at pi: a semicircular end spans
    # pi and gives k = -2, a full circle spans 2*pi and gives k = -4, the four
    # superimposed negatives of the 0-sided template.
    if len(ang) <= 1:
        theta_max = 0.0
    elif gaps.max() <= 4.0 * (2 * np.pi / len(ang)):
        theta_max = 2 * np.pi          # contact all the way round, up to sampling
    else:
        theta_max = float(2 * np.pi - gaps.max())

    M = nx.MultiGraph()
    M.add_node(0, pos=centre, r_m=float(radius), theta_m=theta_max,
               kind=VertexKind.FINITE_CONTACT,
               touch=pts[near], normals=u,
               n_touch=int(near.sum()), contact_extent=theta_max)
    return MedialAxis(graph=M, n_bodies=n_bodies)


def _build(pts, loop_id, offsets, segment_id, segment_closed, inside, corner_pts,
           corner_ang, n_bodies, spacing, min_sep_frac, min_sep_points, min_theta,
           contact_fallback=None):
    tree = cKDTree(pts)
    G = _raw_graph(pts, segment_id, segment_closed, inside, min_sep_frac, min_sep_points)
    _annotate(G, pts, loop_id, offsets, tree)
    _prune(G, corner_pts, min_theta, spacing)

    if G.number_of_nodes() == 0:
        if contact_fallback is not None:
            return _finite_contact(None, pts, loop_id, offsets, n_bodies, contact_fallback)
        return MedialAxis(graph=nx.MultiGraph(), n_bodies=n_bodies)

    kinds, anchors = _classify(G, corner_pts, spacing)
    for nd in G.nodes:
        G.nodes[nd]["contact_extent"] = contact_extent(
            G.nodes[nd]["pos"], G.nodes[nd]["r_m"], pts)
    # A branch that ends on an arc rather than a point is a finite-contact
    # vertex, not a plain curvature-contact dangle (Fogg, Fig. 10).
    for nd, kind in list(kinds.items()):
        if kind is VertexKind.DANGLE and G.nodes[nd]["contact_extent"] >= np.pi / 2:
            kinds[nd] = VertexKind.FINITE_CONTACT
    return _collapse(G, kinds, anchors, corner_pts, corner_ang, n_bodies)


def exterior_axis(
    region: Region,
    min_sep_frac: float = 0.01,
    min_sep_points: int = 4,
    min_theta: float = np.deg2rad(15.0),
) -> MedialAxis:
    """Medial axis of the fluid region: inside the outer loop, outside the bodies.

    Supplies everything between and around the bodies -- the ring against each
    wall, the flares into any corners of the truncation curve, the branches
    through the gaps of a multi-element configuration.

    Parameters
    ----------
    region : Region
    min_sep_frac, min_sep_points : float, int
        Two generators on the *same* loop must be at least this far apart, as a
        fraction of that loop's sample count and as an absolute count, for their
        ridge to count as medial. Closer than that is the bisector between
        adjacent samples, which is an artefact of sampling.
    min_theta : float
        Branches whose medial angle never reaches this are pruned, unless they
        terminate at a known C0 corner.

    Returns
    -------
    MedialAxis
        Whose ``betti_1`` must equal ``region.n_bodies``; see
        :meth:`MedialAxis.certificate`.
    """
    pts, loop_id, offsets = region.sample()
    corners = [b.corners.positions(b.loop) for b in region.boundaries if len(b.corners)]
    angles = [b.fluid_angles for b in region.boundaries if len(b.corners)]
    corner_pts = np.vstack(corners) if corners else np.empty((0, 2))
    corner_ang = np.concatenate(angles) if angles else np.empty(0)
    spacing = min(b.loop.perimeter / b.loop.n_points for b in region.boundaries)
    return _build(pts, loop_id, offsets, region.segment_ids(),
                  region.segment_is_closed(), region.contains, corner_pts, corner_ang,
                  region.n_bodies, spacing, min_sep_frac, min_sep_points, min_theta)


def interior_axis(
    loop: Loop,
    corners: np.ndarray | None = None,
    corner_idx: np.ndarray | None = None,
    corner_angles: np.ndarray | None = None,
    min_sep_frac: float = 0.01,
    min_sep_points: int = 4,
    min_theta: float = np.deg2rad(15.0),
) -> MedialAxis:
    """Medial axis of one body's own interior.

    Scale-free: it does not change when the far field moves, which is what makes
    it usable where the exterior axis degenerates. For an airfoil it is the
    camber line, running from a dangle vertex behind the leading edge to the
    trailing edge.

    Parameters
    ----------
    loop : Loop
        The body. Resample it first; the fields are computed from the samples.
    corners : np.ndarray, shape (K, 2), optional
        Known C0 corner positions, so flares terminate exactly, are never
        pruned, and divide the loop into smooth boundary elements. Pass
        ``BoundaryLoop.corners.positions(loop)``.
    corner_idx : np.ndarray, optional
        The same corners as vertex indices. Derived from ``corners`` if omitted.
    corner_angles : np.ndarray, optional
        Interior angle of the *body* at each corner, so a flare tip gets its
        medial angle analytically instead of from the worst-conditioned point in
        the extraction.
    min_sep_frac, min_sep_points, min_theta
        As for :func:`exterior_axis`.

    Returns
    -------
    MedialAxis
        With ``n_bodies = 0``: a simply connected interior has no cycles, so the
        retract check requires ``betti_1 == 0``.
    """
    pts = loop.points
    loop_id = np.zeros(len(pts), dtype=int)
    offsets = np.array([0, len(pts)])
    corner_pts = np.empty((0, 2)) if corners is None else np.atleast_2d(corners)
    spacing = loop.perimeter / loop.n_points

    # Segment the loop at its corners, exactly as a Region does.
    if corner_idx is None and corners is not None:
        corner_idx = np.argmin(
            np.linalg.norm(pts[:, None, :] - corner_pts[None, :, :], axis=2), axis=0)
    if corner_idx is None or len(corner_idx) == 0:
        segment_id = np.zeros(len(pts), dtype=int)
        segment_closed = np.array([True])
    else:
        idx = np.sort(np.asarray(corner_idx, dtype=int))
        shift = int(idx[0])
        cuts = np.sort((idx - shift) % len(pts))
        seg = np.searchsorted(cuts, np.arange(len(pts)), side="right") - 1
        segment_id = np.roll(seg, shift)
        segment_closed = np.zeros(len(idx), dtype=bool)

    from aeromesh.geometry.region import largest_inscribed_circle
    fallback = largest_inscribed_circle(loop)

    return _build(pts, loop_id, offsets, segment_id, segment_closed, loop.contains,
                  corner_pts, corner_angles, 0, spacing, min_sep_frac,
                  min_sep_points, min_theta, contact_fallback=fallback)
