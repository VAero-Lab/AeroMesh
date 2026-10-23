"""Turning an m-sided block into quadrilaterals.

Transfinite interpolation needs four sides. The decomposition does not produce
four-sided blocks and should not be made to: forcing every region to be a
quadrilateral is exactly the kind of prescribed structure this method exists to
avoid, and the medial graph's vertex regions are m-gons for a reason -- m is the
number of medial branches meeting there, which is a property of the geometry.
Measured over the test matrix after the near-wall layer goes in, 37 of 52 blocks
have m != 4, with m ranging over 3, 5, 6 and 7.

**Midpoint subdivision** is the bridge. An m-gon with corners ``V_i``, side
midpoints ``M_i`` and centre ``C`` splits into m quadrilaterals

    (V_i, M_i, C, M_{i-1})

and nothing else is needed: every sub-block has four sides, the subdivision is
conformal with whatever lies across each side because it only ever splits a side
at its own midpoint, and it is a general construction rather than a template
chosen per geometry. This is the one prescribed construction the method allows,
and it is named as such.

It is also exactly the mesh model the index certificate already assumes. The
centre ``C`` is a single node where all m quads meet, so its valence is m and its
index is ``m - 4`` -- which is the term
:meth:`~aeromesh.blocking.blocks.BlockSystem.index_certificate` charges to an
m-gon block. So subdividing must leave the certificate unchanged, and that is a
sharp test of both: see ``tests/test_templates.py``.

The one thing midpoint subdivision needs is that the block be **logically
convex**. At a corner whose interior angle exceeds pi the sub-block there is
non-convex, and at a corner near 2*pi it folds. Those corners are reported
rather than silently meshed; on this test matrix they are the 261.8 degree
reflex corners of a blunt trailing edge, which is the open limitation recorded
in the tracker.
"""

from __future__ import annotations

import numpy as np

from aeromesh.blocking.blocks import (Block, BlockSystem, arc_length,
                                      clean_sides, local_spacing, orient_ccw,
                                      subcurve)


def side_midpoint(side: np.ndarray) -> np.ndarray:
    """The point halfway along a side by **arc length**.

    By arc length, not by index: a side's points are boundary samples or a
    resampled cut, and neither is evenly spaced. Splitting by index puts the
    join in a different place on the two blocks that share the side, which is a
    hanging node.
    """
    s = arc_length(side)
    return subcurve(side, 0.0, 0.5 * float(s[-1]))[-1]


def centre_of(block: Block, axis=None) -> np.ndarray:
    """Where the m sub-blocks meet.

    For a vertex region, the medial vertex it was built around. That is where
    the singularity is -- the region's m sides are its m incident medial
    branches -- and the region is assembled as legs radiating from it, so it is
    the point the spokes can reach without leaving the block. Measured, it
    removes the folding the inscribed centre leaves on 6 of 11 vertex regions.

    The centre of the block's largest inscribed circle, not its area centroid.
    The centroid is the obvious choice and it is the wrong one here: a near-wall
    layer strip is a long thin polygon whose centroid sits within a rounding
    error of its own boundary -- measured, the centroid of one 5-sided strip
    landed *exactly on one of its corners* -- and the spokes drawn to it then
    cross each other and the sub-blocks fold. The inscribed-circle centre is by
    construction the interior point furthest from the boundary, which is the
    most room the spokes can be given, and it is interior for any simple
    polygon including a strongly non-convex one.
    """
    from aeromesh.geometry.loop import Loop
    from aeromesh.geometry.region import largest_inscribed_circle

    if (axis is not None and block.kind == "vertex" and len(block.source) == 1
            and block.source[0] in axis.graph.nodes):
        return np.asarray(axis.graph.nodes[block.source[0]]["pos"], dtype=float)

    outline = block.outline()
    try:
        return largest_inscribed_circle(Loop.from_points(outline))[0]
    except Exception:
        p = outline
        x, y = p[:, 0], p[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = 0.5 * float(np.sum(cross))
        if abs(a) < 1e-300:
            return p.mean(axis=0)
        return np.array([
            float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * a),
            float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * a)])


def _spoke(a: np.ndarray, b: np.ndarray, n: int = 9) -> np.ndarray:
    """The straight run from a side midpoint in to the centre.

    Straight is the right starting point and not the finishing one: the interior
    edges are what elliptic smoothing in S4 moves, and TFI only needs them to
    bound a valid quad.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a + t * (b - a)


def _at(side: np.ndarray, t: float) -> np.ndarray:
    """The point a fraction ``t`` along a side, by arc length."""
    total = float(arc_length(side)[-1])
    if total <= 0.0:
        return np.asarray(side[0], dtype=float)
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return np.asarray(side[0], dtype=float)
    if t >= 1.0:
        return np.asarray(side[-1], dtype=float)
    return subcurve(side, 0.0, t * total)[-1]


def tfi_spokes(block: Block, n_spoke: int = 33):
    """The spokes of a four-sided block, by transfinite interpolation.

    A harmonic map onto a disc is the right tool for an m-gon and the wrong one
    for a quadrilateral strip. The near-wall layer produces blocks of aspect
    ratio 54, 137, 345, 402; sending the boundary of a 54:1 strip to a circle by
    arc length gives each long side half the circumference and each short end
    almost none, so the map is violently distorted and the straight radii come
    back as curves whose polyline samples cut across the strip. Measured, a
    perfectly well-shaped quad -- interior angles 92, 88, 82, 102 degrees --
    still produced a self-intersecting sub-block that way.

    For four sides there is no need for any of it. Bilinear transfinite
    interpolation maps the unit square onto the block from its own four sides,

        P(u,v) = (1-v)B(u) + vT(u) + (1-u)L(v) + uR(v) - (bilinear corners),

    and the subdivision is the image of the square's two midlines -- which
    cannot cross, because ``u = 1/2`` and ``v = 1/2`` cross once in the square
    and the map is injective for any quad that is not folded. It is also exactly
    the map S4 will use to fill the block, so the subdivision and the mesh agree
    by construction rather than by accident.
    """
    if block.wraps or block.n_sides != 4:
        return None, None
    S = [np.asarray(x, dtype=float) for x in block.sides]
    if any(float(arc_length(x)[-1]) <= 0.0 for x in S):
        return None, None

    V = [S[i][0] for i in range(4)]

    def P(u, v):
        B = _at(S[0], u)
        R = _at(S[1], v)
        T = _at(S[2], 1.0 - u)
        L = _at(S[3], 1.0 - v)
        return ((1 - v) * B + v * T + (1 - u) * L + u * R
                - ((1 - u) * (1 - v) * V[0] + u * (1 - v) * V[1]
                   + u * v * V[2] + (1 - u) * v * V[3]))

    centre = P(0.5, 0.5)
    t = np.linspace(0.0, 1.0, max(3, n_spoke))
    spokes = [
        np.array([P(0.5, 0.5 * x) for x in t]),            # from side 0
        np.array([P(1.0 - 0.5 * x, 0.5) for x in t]),       # from side 1
        np.array([P(0.5, 1.0 - 0.5 * x) for x in t]),       # from side 2
        np.array([P(0.5 * x, 0.5) for x in t]),             # from side 3
    ]
    for i, path in enumerate(spokes):
        path[0] = _at(S[i], 0.5)
        path[-1] = centre
    return spokes, centre


def harmonic_spokes(block: Block, n_spoke: int = 81, per_side: int = 48,
                    triangles: int = 400):
    """The m spokes from the side midpoints to one interior point, by a
    harmonic map onto a disc.

    Straight spokes to a centre do not work here, at any choice of centre. The
    area centroid of a near-wall layer strip lands within a rounding error of
    its own boundary; the inscribed-circle centre is interior by construction
    but the straight lines to it still leave the block, because a vertex region
    wrapping a quarter of the domain is strongly non-convex; and even a
    four-sided span around half an aerofoil folds. Measured over the test matrix
    with the inscribed centre, 5 of 8 blocks produced a self-intersecting
    sub-block, and 2 of those were already four-sided.

    Tutte's theorem gives the construction that cannot fold. A discrete harmonic
    map of a triangulated disc onto a **convex** region, with the boundary
    mapped homeomorphically to the boundary, is injective. So:

    1. triangulate the block (constrained Delaunay, no boundary Steiner points);
    2. send its boundary to the unit circle by cumulative arc length, which is a
       homeomorphism;
    3. solve the discrete Laplace system for the interior vertices, giving an
       injective embedding in the disc;
    4. in the disc the spokes are straight radii from each side's mid-arc-length
       point to the origin, which plainly do not cross;
    5. carry them back through the map.

    Injectivity makes the images non-crossing curves meeting at one interior
    point, which is exactly what the subdivision needs, and only ``m * n_spoke``
    points have to be mapped -- the corners and side midpoints are already on
    the physical boundary and are used exactly.

    Returns ``(spokes, centre)`` with ``spokes[i]`` running from the midpoint of
    side ``i`` to the centre, or ``(None, None)`` if the map could not be built.
    """
    try:
        import triangle as tr
    except ImportError:                           # pragma: no cover
        return None, None

    # Never hand a non-simple polygon to Triangle. It is a C library, and on a
    # self-crossing outline it reports "Topological inconsistency after
    # splitting a segment" and -- in a longer-running process -- segfaults,
    # which takes the interpreter down with it. The caller's own validity check
    # used to run *after* this call, which is exactly backwards.
    from shapely.geometry import LineString, Polygon
    try:
        o = block.outline()
        if len(o) < 3:
            return None, None
        if not LineString(np.vstack([o, o[:1]])).is_simple:
            return None, None
        if not Polygon(o).is_valid or Polygon(o).area <= 0.0:
            return None, None
    except Exception:
        return None, None

    m = block.n_sides
    sides = [np.asarray(x, dtype=float) for x in block.sides]
    lengths = [float(arc_length(x)[-1]) for x in sides]
    if min(lengths) <= 0.0:
        return None, None
    perimeter = float(sum(lengths))

    # The canonical domain is the unit **disc**, with the boundary mapped by
    # cumulative arc length. A disc is maximally convex, so Tutte applies, and
    # arc length keeps the map close to isometric -- which matters: a near-wall
    # layer strip has an aspect ratio near 40, and forcing its five sides onto a
    # *regular* pentagon is a violent distortion that the spokes do not survive.
    def theta(dist):
        return 2.0 * np.pi * dist / perimeter

    start = np.concatenate(([0.0], np.cumsum(lengths)))

    verts, bc = [], []
    for i, side in enumerate(sides):
        k = max(4, per_side)
        for j in range(k):                        # [0, 1) -- the corner is j=0
            t = j / k
            d = t * lengths[i]
            verts.append(subcurve(side, 0.0, max(d, 1e-300))[-1] if t > 0.0
                         else side[0])
            a = theta(start[i] + d)
            bc.append([np.cos(a), np.sin(a)])
    V = np.asarray(verts, dtype=float)
    bc = np.asarray(bc, dtype=float)
    n_b = len(V)
    seg = np.column_stack([np.arange(n_b), (np.arange(n_b) + 1) % n_b])

    area = abs(block.area)
    try:
        # 'Y' prohibits Steiner points on the boundary. Without it Triangle
        # splits boundary segments -- measured, 200 input segments came back as
        # 216 to 231 -- and each inserted vertex has no Dirichlet value, so it is
        # solved as a free vertex and leaves the canonical boundary. The map is
        # then not a homeomorphism on the boundary and Tutte's guarantee does
        # not apply, which is why refining the spokes did not converge: folding
        # went 10, 8, 4, 7 out of 42 as the sampling was increased.
        mesh = tr.triangulate({"vertices": V, "segments": seg},
                              f"pq30Ya{area / max(triangles, 1):.12g}")
    except Exception:                             # pragma: no cover
        return None, None
    if "triangles" not in mesh or not len(mesh["triangles"]):
        return None, None
    P = np.asarray(mesh["vertices"], dtype=float)
    T = np.asarray(mesh["triangles"], dtype=int)
    if len(P) < n_b:
        return None, None
    # Triangle keeps the input vertices first and in order, so the boundary
    # condition still lines up; bail out rather than guess if it did not.
    if not np.allclose(P[:n_b], V, atol=1e-12):
        return None, None

    # ── 3. uniform-weight Laplace, Dirichlet on the boundary ──
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    n = len(P)
    rows, cols, vals = [], [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for u, v in ((T[:, a], T[:, b]), (T[:, b], T[:, a])):
            rows.append(u)
            cols.append(v)
            vals.append(np.ones(len(u)))
    W = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(n, n)).tocsr()
    W.data[:] = 1.0                               # one entry per distinct edge
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = (coo_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
         - W)

    free = np.ones(n, dtype=bool)
    free[:n_b] = False
    if not free.any():
        return None, None
    Q = np.zeros((n, 2))
    Q[:n_b] = bc
    A = L[free][:, free]
    rhs = -L[free][:, ~free] @ Q[~free]
    try:
        sol = spsolve(A.tocsc(), rhs)
    except Exception:                             # pragma: no cover
        return None, None
    sol = np.asarray(sol)
    if sol.ndim == 1:
        sol = sol.reshape(-1, 2)
    if not np.all(np.isfinite(sol)):
        return None, None
    Q[free] = sol

    # ── 4-5. straight radii in the canonical m-gon, mapped back ──
    canon_tri = Q[T]                              # (nT, 3, 2)
    a0 = canon_tri[:, 0]
    e1 = canon_tri[:, 1] - a0
    e2 = canon_tri[:, 2] - a0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    good = np.abs(det) > 1e-300
    if not good.any():
        return None, None

    def to_physical(q):
        """Locate q in the canonical triangulation and map it back."""
        d = q - a0
        u = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / np.where(good, det, 1.0)
        v = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / np.where(good, det, 1.0)
        ok = good & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if not ok.any():
            # outside every triangle: take the nearest by barycentric slack
            slack = (np.maximum(0, -u) + np.maximum(0, -v)
                     + np.maximum(0, u + v - 1.0))
            slack[~good] = np.inf
            k = int(np.argmin(slack))
        else:
            k = int(np.flatnonzero(ok)[0])
        w = np.array([1.0 - u[k] - v[k], u[k], v[k]])
        return w @ P[T[k]]

    centre = to_physical(np.zeros(2))
    spokes = []
    for i in range(m):
        a = theta(start[i] + 0.5 * lengths[i])
        rim = np.array([np.cos(a), np.sin(a)])
        t = np.linspace(0.0, 1.0, max(3, n_spoke))[:, None]
        path = np.array([to_physical(q) for q in rim * (1.0 - t)])
        path[-1] = centre
        spokes.append(path)
    return spokes, centre


def _spokes_for(block: Block):
    """Transfinite interpolation for a quadrilateral, the harmonic map
    otherwise; and the harmonic map as a fallback if TFI folds."""
    if block.n_sides == 4 and not block.wraps:
        spokes, centre = tfi_spokes(block)
        if spokes is not None:
            return spokes, centre
    return harmonic_spokes(block)


def _raw_quads(block: Block, C: np.ndarray | None = None,
               spokes=None) -> list:
    """The m four-sided outlines of the midpoint subdivision, unoriented."""
    m = block.n_sides
    sides = [np.asarray(s, dtype=float) for s in block.sides]
    halves = []
    for s in sides:
        total = float(arc_length(s)[-1])
        mid = 0.5 * total
        halves.append((subcurve(s, 0.0, mid), subcurve(s, mid, total)))

    out = []
    for i in range(m):
        # V_i is the corner between side i-1 and side i, so the quad there is
        # bounded by the second half of side i-1 and the first half of side i.
        second_prev = halves[(i - 1) % m][1]
        first_here = halves[i][0]
        if spokes is not None:
            inward = spokes[i]
            outward = spokes[(i - 1) % m][::-1]
        else:
            inward = _spoke(first_here[-1], C)
            outward = _spoke(C, second_prev[0])
        inward = np.vstack([first_here[-1:], inward[1:]])
        outward = np.vstack([outward[:-1], second_prev[:1]])
        out.append((second_prev, first_here, inward, outward))
    return out


def subdivide(block: Block, centre: np.ndarray | None = None,
              first_id: int = 0, tol_at=None, axis=None) -> list:
    """Split one m-gon into m four-sided blocks.

    A four-sided block is returned unchanged -- there is nothing to do and
    subdividing it would add an unnecessary interior node of valence 4.
    """
    if block.wraps:
        return [block]

    spokes, harmonic = _spokes_for(block)
    C = (harmonic if harmonic is not None
         else (centre_of(block, axis) if centre is None
               else np.asarray(centre, dtype=float)))
    out = []
    for quad in _raw_quads(block, C, spokes):
        out.append(Block(
            id=first_id + len(out),
            sides=orient_ccw(clean_sides(quad, tol_at=tol_at,
                                         pinned=(True, True, True, True))),
            kind="quad", source=block.source))
    return out


def quad_faults(block: Block, centre: np.ndarray | None = None,
                axis=None) -> list:
    """Why midpoint subdivision would not be valid on this block.

    Not a convexity proxy -- the subdivision is actually carried out and every
    sub-quad is checked. A centroid-inside test is far too weak: it passes on
    thin layer strips and on non-convex vertex regions that then produce
    self-intersecting sub-blocks, and a decomposition that tiles is exactly the
    thing an area check cannot tell apart from one that folds.

    Returns a list of complaints, empty when the block is fit to subdivide.
    """
    from shapely.geometry import LineString, Point, Polygon

    if block.wraps:
        return []
    if block.n_sides < 3:
        return [f"only {block.n_sides} sides"]

    try:
        parent = Polygon(block.outline())
        if not parent.is_valid:
            return ["outline is not simple"]
    except Exception as exc:                      # pragma: no cover
        return [f"outline cannot be polygonised: {exc}"]

    # Validity first, *then* the harmonic map -- see the guard in
    # `harmonic_spokes` for why the order matters.
    spokes, harmonic = _spokes_for(block)
    C = (harmonic if harmonic is not None
         else (centre_of(block, axis) if centre is None else centre))
    if not parent.contains(Point(C)):
        return ["centre falls outside the block"]

    faults = []
    for k, q in enumerate(_raw_quads(block, C, spokes)):
        o = np.vstack([np.asarray(q[0])] + [np.asarray(x)[1:] for x in q[1:]])
        gap = max(float(np.linalg.norm(np.asarray(q[i])[-1]
                                       - np.asarray(q[(i + 1) % 4])[0]))
                  for i in range(4))
        if gap > 1e-9 * block.area ** 0.5 + 1e-12:
            faults.append(f"sub-block {k} does not close (gap {gap:.2e})")
            continue
        if not LineString(np.vstack([o, o[:1]])).is_simple:
            faults.append(f"sub-block {k} self-intersects")
            continue
        x, y = o[:, 0], o[:, 1]
        a = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if abs(a) <= 0.0:
            faults.append(f"sub-block {k} has no area")
    return faults


def to_quads(system: BlockSystem, strict: bool = True) -> BlockSystem:
    """Midpoint-subdivide a whole system into four-sided blocks.

    **All or nothing.** Subdivision splits every side at its own midpoint, so it
    is conformal only when both blocks that share a side do it. Subdividing the
    m-gons and leaving the quads alone -- or skipping the m-gons that would fold
    -- puts a node in the middle of a side whose neighbour runs straight past
    it, which is a hanging node; it shows up as a non-conformal interface
    report, and on two aerofoils in a box it did. So a quad is subdivided too,
    into four, which costs an interior node of valence 4 and therefore no index.

    With ``strict`` the system is returned untouched, with the reason in
    ``notes``, if *any* block would fold. That is the honest failure: a
    partially subdivided system is not a mesh.

    Rings are the exception and need no care: a ring has no corners, is already
    one structured block, and shares its sides with nobody.
    """
    tol_at = local_spacing(system.region)
    faults = {b.id: quad_faults(b, axis=system.axis) for b in system.blocks}
    bad = {i: f for i, f in faults.items() if f}

    if bad and strict:
        notes = list(system.notes)
        notes.append(
            f"not subdivided: {len(bad)} of {system.n_blocks} blocks would "
            f"fold, and a partial subdivision is not conformal")
        for i, f in sorted(bad.items()):
            blk = next(b for b in system.blocks if b.id == i)
            notes.append(f"  block {i} (m={blk.n_sides}, {blk.kind}): "
                         + "; ".join(f))
        return BlockSystem(blocks=list(system.blocks), region=system.region,
                           axis=system.axis, notes=notes)

    blocks, notes, bid = [], list(system.notes), 0
    for b in system.blocks:
        if b.wraps:
            blocks.append(Block(id=bid, sides=b.sides, kind=b.kind,
                                source=b.source, wraps=True))
            bid += 1
            continue
        if faults[b.id]:
            notes.append(f"block {b.id} (m={b.n_sides}, {b.kind}): "
                         + "; ".join(faults[b.id]) + " -- left whole")
            blocks.append(Block(id=bid, sides=b.sides, kind=b.kind,
                                source=b.source))
            bid += 1
            continue
        for q in subdivide(b, first_id=bid, tol_at=tol_at, axis=system.axis):
            blocks.append(Block(id=bid, sides=q.sides, kind="quad",
                                source=q.source))
            bid += 1
    return BlockSystem(blocks=blocks, region=system.region, axis=system.axis,
                       notes=notes)
