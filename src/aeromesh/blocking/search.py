"""Choosing where to cut.

The block builder will decompose a region given any set of cuts; this is what
decides which set. The objective, following Fogg (sec. 4.2), is that each
singularity ends up alone inside its own subregion -- that is what "logically
convex polygon subregions around the mesh singularities" means once the regions
are the spans and vertex regions of the medial graph.

The rule is simple to state. On a medial edge carrying singularities at arc
positions s1 < s2 < ... < sk, cuts go in the **gaps between them** -- one in
(0, s1), one in (s1, s2), and so on up to one in (sk, L). That yields k+1 cuts,
which separate the k singularities into k spans and leave the two end cuts
bounding the vertex regions at either end of the edge. An edge carrying no
singularity still needs one cut, or the vertex regions at its two ends would run
into each other.

Within each gap the candidate cuts are ranked as Fogg ranks them: n = 0 before
n = 1, because a straight-through cut leaves no concave corner, and within a
class the medial angle nearer the ideal for that class. The best-ranked
candidate in the gap is taken.

Nothing here is specific to any body or truncation shape: the input is a medial
graph, a singularity field and a ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeromesh.blocking.splits import candidate_splits


def _cross2(a, b) -> float:
    """2-D scalar cross product. ``np.cross`` on 2-vectors is deprecated."""
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class Placement:
    """Where a singularity sits on the medial axis."""

    edge: tuple | None          # None when it belongs to a vertex
    node: int | None
    index: int                  # sample index along the edge
    s: float                    # arc length along the edge
    k: int
    source: str
    distance: float             # how far the singularity was from the axis


def locate(axis, position, tol_factor: float = 2.0) -> Placement | None:
    """Find the medial edge or vertex a singularity belongs to.

    Singularities are reported as points; the decomposition needs to know which
    part of the skeleton they sit on. A singularity closer to a vertex than to
    any edge interior belongs to that vertex's region and places no cut.
    """
    best = None
    for u, v, key, d in axis.graph.edges(keys=True, data=True):
        dist = np.linalg.norm(d["polyline"] - position, axis=1)
        i = int(np.argmin(dist))
        if best is None or dist[i] < best[0]:
            best = (float(dist[i]), (u, v, key), i, float(d["s"][i]))
    if best is None:
        return None
    dist, edge, i, s = best

    node_best = None
    for n, nd in axis.graph.nodes(data=True):
        if nd.get("anchor"):
            continue
        dn = float(np.linalg.norm(nd["pos"] - position))
        if node_best is None or dn < node_best[0]:
            node_best = (dn, n, nd)
    if node_best is not None:
        dn, n, nd = node_best
        if dn <= tol_factor * max(nd["r_m"], 1e-12) and dn <= dist * 1.5:
            return Placement(edge=None, node=n, index=-1, s=float("nan"),
                             k=0, source="", distance=dn)
    return Placement(edge=edge, node=None, index=i, s=s, k=0, source="",
                     distance=dist)


def place_singularities(axis, field) -> dict:
    """Group a singularity field onto the medial edges it will need cuts around.

    Returns ``{edge_key: [Placement, ...]}`` sorted by arc length. Singularities
    that belong to a vertex are left out -- the vertex region already contains
    them, so they need no cut of their own.
    """
    out: dict = {}
    for sing in field.singularities:
        p = locate(axis, sing.position)
        if p is None or p.edge is None:
            continue
        p = Placement(edge=p.edge, node=None, index=p.index, s=p.s,
                      k=sing.k, source=sing.source, distance=p.distance)
        out.setdefault(p.edge, []).append(p)
    for key in out:
        out[key] = sorted(out[key], key=lambda q: q.s)
    return out


def _corner_frame(loop, ci: int, alpha: float):
    """The fluid wedge at a boundary corner, as a reference ray and a sense.

    Returns ``(a0, sense)``: the ray back along the incoming edge, and +1 or -1
    for the rotational direction in which the fluid wedge of angle ``alpha``
    opens. The sense is calibrated against ``alpha`` itself rather than deduced
    from the loop's handedness, so it is right for an outer boundary and for a
    hole without the caller having to know which it has.
    """
    q = loop.points
    n = len(q)
    e_in = q[ci] - q[(ci - 1) % n]
    e_out = q[(ci + 1) % n] - q[ci]
    e_in /= max(np.linalg.norm(e_in), 1e-300)
    e_out /= max(np.linalg.norm(e_out), 1e-300)
    a0 = -e_in
    ccw = float(np.arctan2(_cross2(a0, e_out), np.dot(a0, e_out))) % (2 * np.pi)
    sense = 1.0 if abs(ccw - alpha) <= abs((2 * np.pi - ccw) - alpha) else -1.0
    return a0, sense


def corner_cuts(axis, region, only=None, need: dict | None = None) -> dict:
    """Cuts that resolve every reflex boundary corner.

    Two things are wrong with a reflex corner left alone.

    A C0 corner of the wall inside a block *side* is a kink in a curve the
    mesher treats as smooth: transfinite interpolation maps a straight logical
    edge onto it and puts a badly skewed cell exactly at the trailing edge,
    which is the one place on an aerofoil where that is least affordable. So the
    corner has to be a corner of the blocking too.

    And one cut is not enough. A corner of fluid angle ``alpha`` carries
    ``n_c = round(alpha / (pi/2))`` elements, so ``n_c - 1`` grid lines leave it,
    and they want to divide ``alpha`` evenly -- Fogg's cross directions at
    ``alpha * i / n_c``. Cutting anywhere else splits the corner into wildly
    unequal sectors: on a NACA blunt trailing edge of 261.8 degrees, taking the
    middle of each run of candidates gives block angles of 7.8, 37.1 and 216.9
    degrees where 87.3 each is available.

    Convex corners of the fluid need nothing done: ``r_m -> 0`` there, a medial
    branch runs out to the corner and terminates, and the corner is already a
    polygon vertex of the region at that end -- which is why a box corner never
    shows this defect. A reflex corner is the problem: the medial axis sweeps
    past it with the corner as one of its two touches, so nothing in the graph
    marks it, and on a smooth far field an aerofoil's two blunt trailing-edge
    corners end up buried in the middle of a single ring block's wall.

    Every candidate is a medial radius whose foot is the corner, so the cuts
    stay curves of the medial coordinate system like every other cut. Their feet
    all coincide at the corner; what distinguishes them is the direction they
    leave it in, and that is what is matched to the cross directions. The medial
    wedge at a reflex corner is only ``alpha - pi`` wide, so the achievable
    directions sit inside the wanted ones and the match is to the nearest
    available -- on the trailing edge above, within 3 degrees.

    ``need``, when given, is what the caller is short of:
    ``{global corner index: [cross direction, ...]}`` from
    :meth:`~aeromesh.blocking.blocks.BlockSystem.unresolved_corners`. Only those
    corners are cut, and only towards the directions listed. That is the whole
    point: most corners need nothing. On a box or C far field a flare already
    runs out to each trailing-edge corner and the two wall arcs of a vertex
    region meet there, so the corner has its lines already, and forcing a cut
    anyway splits a sector into slivers and costs the certificate -- it is what
    turns a 90 degree block angle into a 3.6 degree one. ``only`` is the cruder
    form, restricting by corner but choosing the directions itself.

    Returns ``{edge_key: [sample index, ...]}``.
    """
    _, _, offsets = region.sample()
    if need is not None:
        only = set(int(k) for k in need)
    else:
        only = None if only is None else {int(x) for x in only}
    want: list = []
    tol = 1e-7 * region.outer.loop.scale
    for li, b in enumerate(region.boundaries):
        if not len(b.corners):
            continue
        for ci, ang, nc in zip(b.corners.index, b.fluid_angles, b.corner_elements):
            alpha = float(ang)
            gi = int(offsets[li]) + int(ci)
            if alpha <= np.pi or int(nc) < 2:
                continue
            if only is not None and gi not in only:
                continue
            a0, sense = _corner_frame(b.loop, int(ci), alpha)
            want.append((gi, b.loop.points[int(ci)], alpha, int(nc), a0, sense))
    if not want:
        return {}

    def direction(d, i, pos, a0, sense) -> float:
        w = d["polyline"][int(i)] - pos
        nw = float(np.linalg.norm(w))
        if nw <= 0.0:
            return np.nan
        w = w / nw
        return (sense * float(np.arctan2(_cross2(a0, w),
                                         np.dot(a0, w)))) % (2 * np.pi)

    out: dict = {}
    for gi, pos, alpha, nc, a0, sense in want:
        # Candidates, grouped by the contiguous run of one medial edge they come
        # from. A run is the stretch of one edge over which the corner is the
        # touch, and the runs of the edges meeting there tile the achievable
        # range of directions between them. At most one cut per run: two cuts in
        # one run would bound a span of almost no width.
        runs: list = []
        for u, v, key, d in axis.graph.edges(keys=True, data=True):
            n = len(d["polyline"])
            for foot in (d["t1"], d["t2"]):
                # Matched on the exact **foot**, not on the touch index. The
                # touch index is the boundary sample nearest the foot, and two
                # things go wrong with it. It can be filed one sample over -- on
                # a regular pentagon three of five runs were, and matching the
                # corner's own index exactly then found no candidates at all and
                # dropped those corners without a note. And relaxing it to the
                # index's two neighbours is worse than useless: it admits the
                # sample at each *end* of the run, where the foot has already
                # left the corner for the adjacent edge, so the cut's foot is a
                # spacing away and the corner stays buried anyway. That is
                # exactly what happened on an aerofoil in a circular far field.
                #
                # The foot is unambiguous. At a reflex corner the corner is the
                # nearest boundary point for the whole of its normal cone, so
                # every sample in the run has its foot *exactly* there, and the
                # run ends precisely where the foot moves off.
                dist = np.linalg.norm(np.asarray(foot, dtype=float) - pos,
                                      axis=1)
                idx = np.flatnonzero(dist <= tol)
                if not len(idx):
                    continue
                for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    if not len(run):
                        continue
                    phi = np.array([direction(d, i, pos, a0, sense) for i in run])
                    ok = np.isfinite(phi)
                    if not ok.any():
                        continue
                    runs.append(((u, v, key), n, run[ok], phi[ok]))
        if not runs:
            continue

        # Match the cross directions to the runs that can reach them, nearest
        # first, then take the sample in that run closest to the direction.
        #
        # One cut, not the n_c - 1 the corner's element count asks for. The
        # outermost cross directions are only achievable at the very end of a
        # run -- that is, *at* a medial vertex, where the radius pair is
        # degenerate -- because the medial wedge at a reflex corner is only
        # alpha - pi wide while the directions wanted span alpha. A grid line
        # leaving the corner in that direction is already supplied by the vertex
        # region's own arc, and cutting there as well duplicates it and leaves a
        # sliver span: it costs the box and C cases their tiling. So the corner
        # is resolved -- it becomes a block corner, which is what removes the
        # kink -- and the remaining sectors are whatever the surrounding
        # structure provides, reported by ``BlockSystem.quality`` rather than
        # forced here.
        targets = (list(need[gi]) if need is not None and gi in need
                   else [alpha / nc])
        taken: set = set()
        for target in targets:
            best = None
            for j, (ek, n, run, phi) in enumerate(runs):
                if j in taken:
                    continue
                gap = float(np.min(np.abs(phi - target)))
                if best is None or gap < best[0]:
                    best = (gap, j)
            if best is None:
                break
            j = best[1]
            taken.add(j)
            ek, n, run, phi = runs[j]
            i = int(run[int(np.argmin(np.abs(phi - target)))])
            out.setdefault(ek, set()).add(int(np.clip(i, 1, n - 2)))
    return {k: sorted(v) for k, v in out.items()}


def select_splits(axis, region, field=None, margin: float = 0.03,
                  stride: int = 1, forced: dict | None = None) -> dict:
    """Choose the cut positions on every medial edge.

    Parameters
    ----------
    axis : MedialAxis
    region : Region
    forced : dict, optional
        Extra cuts that must be made, ``{edge_key: [sample index, ...]}``. Used
        for the second pass that resolves buried boundary corners; see
        :func:`corner_cuts`.
    field : SingularityField, optional
        When given, cuts are placed in the gaps between singularities so that
        each span holds at most one, and a flare carrying no singularity is left
        uncut. Without it, a single best-ranked cut is placed on each edge, which
        is enough to separate the vertex regions but makes no attempt to isolate
        anything.
    margin : float
        Keeps candidates clear of the edge ends, where the radius pair is
        degenerate and the vertex's own balance applies instead.
    stride : int
        Sub-sampling of candidate positions, for speed on long edges.

    Returns
    -------
    dict
        ``{edge_key: np.ndarray of sample indices}``, sorted and unique.
    """
    placed = place_singularities(axis, field) if field is not None else {}
    forced = dict(forced or {})
    chosen: dict = {}

    for u, v, key in axis.graph.edges(keys=True):
        ek = (u, v, key)
        d = axis.graph.edges[u, v, key]
        n = len(d["polyline"])
        if d["kind"].name == "LOOP":
            # A closed edge has no vertices, so with nothing to separate it is
            # one ring block -- an O-grid, found rather than assumed, and right
            # for a smooth body in a smooth far field. It still has to be cut
            # where a reflex corner demands it.
            if ek in forced:
                chosen[ek] = np.unique(np.asarray(forced[ek], dtype=int))
            continue
        bounds_here = [q.s for q in placed.get(ek, [])]

        # A flare runs out to a convex corner of the fluid. Along it theta_m is
        # exactly pi - alpha at every sample, so every candidate has identical
        # rank, and r_m is exactly linear in arc length, so the wedge a cut
        # leaves behind is a scaled copy of itself wherever the cut goes. No
        # local criterion can prefer a position, and none is needed: Fogg marks
        # a convex corner with n_c != 0 as *resolved*, so no split is required
        # between a singularity and that corner. The wedge therefore merges into
        # the neighbouring vertex region and the corner becomes one of its
        # polygon vertices. A flare is only cut when a singularity sits on it.
        if d["kind"].name == "FLARE" and not bounds_here and ek not in forced:
            chosen[ek] = np.empty(0, dtype=int)
            continue

        cands = candidate_splits(axis, ek, stride=stride, margin=margin)
        if not cands:
            chosen[ek] = np.array([max(1, n // 2)], dtype=int)
            continue

        gaps = list(zip([0.0] + bounds_here,
                        bounds_here + [float(d["s"][-1])]))

        picks = []
        for lo, hi in gaps:
            inside = [c for c in cands if lo < c.s < hi]
            if not inside:
                # No admissible cut in this gap: fall back to its midpoint, and
                # let the tiling check report the consequence rather than
                # silently dropping a separation.
                mid = 0.5 * (lo + hi)
                inside = [min(cands, key=lambda c: abs(c.s - mid))]
            picks.append(min(inside, key=lambda c: c.rank).index)

        chosen[ek] = np.unique(np.asarray(picks + forced.get(ek, []),
                                          dtype=int))

    return chosen
