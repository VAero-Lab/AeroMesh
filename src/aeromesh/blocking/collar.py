"""The near-wall collar: zone 1 of a zoned blocking.

Why a zone at all, when the premise is that nothing is prescribed. Because
"prescribed" has two meanings and only one of them was ever forbidden.
Prescribing *per geometry* -- recognise an aerofoil, apply a C-grid template --
is what this method exists to avoid. Being *forced by topology* is different: a
thin band around a simple closed curve is an annulus, and an annulus is an
O-grid because there is nothing else it could be. No choice is made, so none can
be wrong. The same argument already licenses midpoint subdivision.

What a zone buys is that one medial axis stops being asked to do three jobs at
three scales. The exterior axis is measurably useless in the far field --
theta_m runs 172.9 to 180.0 degrees at ten chords, so Fogg's index is zero at
every sampled point -- and it is at its best between bodies, where a slot drives
theta_m down to 10.9 degrees. Using it everywhere spends its authority where it
has none.

And the collar **absorbs the geometric singularities**. Every hard case in this
project lives at the wall: the reflex trailing-edge corner, the flares running
into it with ``r_m -> 0``, the touch arrays that stop being monotonic where the
feet jitter. Wrap the body in a collar and the mid-field analysis never sees any
of it -- it sees one smooth, well-separated closed curve per body.

The construction, which takes no decisions of its own:

* **Thickness** is ``delta(s) = min(delta_target, rho_max * r_m(s))``: the flow
  sets it, the medial radius caps it to the room available. ``delta < r_m``
  pointwise is sufficient for the offset to be simple.
* **The outer curve** is smoothed and then *re-projected* back to distance
  ``delta``, alternately, until both hold. Offsetting per sample alone gives a
  curve jagged enough to grow medial branches of its own; smoothing alone
  shortens it and drags it onto the wall. Doing both, alternately, gives a
  smooth curve that is still at the right distance -- which is what the earlier
  ``peel`` attempt got wrong.
* **The cuts** are the grid lines a reflex wall corner demands. A corner of
  fluid angle ``alpha`` carries ``n_c = round(alpha / (pi/2))`` elements, so
  ``n_c - 1`` lines leave it, at the cross directions ``alpha * i / n_c``. They
  are straight rays from the corner to the outer curve. For a blunt trailing
  edge that is two corners at 261.8 degrees, two lines each; for a sharp one,
  a single corner at 343.5 degrees and three lines. Nothing names a trailing
  edge: the wake falls out of the corner's element count.

Inside the collar the ``alpha - pi`` restriction that defeats the mid-field does
not apply. There, a cut must reach the medial axis and continue to the opposite
wall, so it has to be a medial radius and the outermost cross directions are
provably unreachable. A ray that stops at ``delta`` has no such obligation, so
the cross directions are achieved exactly.
"""

from __future__ import annotations

import numpy as np

from aeromesh.geometry.loop import Loop


# ═══════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def nearest_on_polyline(query: np.ndarray, poly: np.ndarray):
    """Foot, distance and segment index of the nearest point on a closed
    polyline, for each query point. Exact, not nearest-sample."""
    q = np.atleast_2d(np.asarray(query, dtype=float))
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    d = b - a
    L2 = np.einsum("ij,ij->i", d, d)
    L2 = np.where(L2 > 0.0, L2, 1.0)
    # (Q, P) parameter of the projection onto each segment
    t = np.einsum("qij,ij->qi", q[:, None, :] - a[None, :, :], d) / L2
    t = np.clip(t, 0.0, 1.0)
    feet = a[None, :, :] + t[:, :, None] * d[None, :, :]
    dist = np.linalg.norm(feet - q[:, None, :], axis=2)
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    return feet[rows, k], dist[rows, k], k


def outward_normals(points: np.ndarray) -> np.ndarray:
    """Unit normals of a closed CCW polyline, pointing out of it."""
    p = np.asarray(points, dtype=float)
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
    return np.column_stack([t[:, 1], -t[:, 0]])


def smooth_offset(wall: np.ndarray, delta: np.ndarray, n_out: int = 400,
                  passes: int = 60) -> np.ndarray:
    """A smooth closed curve at distance ``delta`` from ``wall``.

    Smoothing and offsetting fight each other and neither works alone. Offsetting
    each sample along its own normal gives a curve whose segment lengths span
    four orders of magnitude, because the wall's sampling is not the offset's;
    its own medial axis then sprouts branches and the region outside it decomposes
    into three times as many blocks as it should. Smoothing that curve shortens
    it -- plain curve shortening -- and drags it onto the wall exactly where the
    curvature is highest, which on one measurement took the wall distance from a
    1.5x range to 66x.

    Alternating the two converges to what is wanted: smooth, and at the right
    distance. Each pass relaxes the curve and then pushes every point back out
    to ``delta`` along the direction from its own nearest wall point -- which is
    the wall normal where the wall is smooth, and the radial direction about a
    corner where it is not, so a convex corner is rounded rather than chamfered.
    """
    wall = np.asarray(wall, dtype=float)
    delta = np.asarray(delta, dtype=float)
    curve = wall + delta[:, None] * outward_normals(wall)
    curve = Loop.from_points(curve, name="collar").resample(n_out).points

    for _ in range(passes):
        relaxed = 0.25 * (np.roll(curve, 1, axis=0) + 2.0 * curve
                          + np.roll(curve, -1, axis=0))
        feet, dist, _ = nearest_on_polyline(relaxed, wall)
        # the thickness wanted at each foot, read off the wall field
        j = nearest_on_polyline(feet, wall)[2]
        want = delta[j]
        step = relaxed - feet
        norm = np.maximum(np.linalg.norm(step, axis=1), 1e-300)
        curve = feet + want[:, None] * step / norm[:, None]

    return Loop.from_points(curve, name="collar").resample(n_out).points


def corner_frame(wall: np.ndarray, ci: int, alpha: float):
    """Reference ray and rotational sense of the fluid wedge at a wall corner."""
    n = len(wall)
    e_in = wall[ci] - wall[(ci - 1) % n]
    e_out = wall[(ci + 1) % n] - wall[ci]
    e_in /= max(np.linalg.norm(e_in), 1e-300)
    e_out /= max(np.linalg.norm(e_out), 1e-300)
    a0 = -e_in
    cross = float(a0[0] * e_out[1] - a0[1] * e_out[0])
    ccw = float(np.arctan2(cross, float(np.dot(a0, e_out)))) % (2 * np.pi)
    sense = 1.0 if abs(ccw - alpha) <= abs((2 * np.pi - ccw) - alpha) else -1.0
    return a0, sense


def ray_hit(origin: np.ndarray, direction: np.ndarray,
            curve: np.ndarray) -> tuple:
    """Where a ray first crosses a closed polyline: the point and the segment."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    a = np.asarray(curve, dtype=float)
    b = np.roll(a, -1, axis=0)
    e = b - a
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-300
    rhs = a - o
    t = np.where(ok, (rhs[:, 0] * e[:, 1] - rhs[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), -1.0)
    u = np.where(ok, (rhs[:, 0] * d[1] - rhs[:, 1] * d[0]) / np.where(ok, den, 1.0), -1.0)
    good = ok & (t > 1e-12) & (u >= -1e-9) & (u <= 1.0 + 1e-9)
    if not good.any():
        raise ValueError("the corner ray never meets the collar's outer curve")
    k = int(np.flatnonzero(good)[np.argmin(t[good])])
    return o + t[k] * d, k, float(u[k])


# ═══════════════════════════════════════════════════════════════════
#  The collar itself
# ═══════════════════════════════════════════════════════════════════

def _arc_between(curve: np.ndarray, k_from: int, u_from: float,
                 p_from: np.ndarray, k_to: int, u_to: float,
                 p_to: np.ndarray, forward: bool) -> np.ndarray:
    """The run of a closed polyline between two points lying on its segments.

    Addressed by **position** along the curve, ``segment + parameter``, not by
    walking segment indices. Two points can share a segment -- at a blunt
    trailing edge the two wake rays converge and land 0.02 apart on the *same*
    one -- and an index walk then has no way to say "the short piece between
    them" and goes the whole way round instead, swallowing the entire collar.
    """
    n = len(curve)
    out = [np.asarray(p_from, dtype=float)]
    same = (k_from % n) == (k_to % n)
    if forward:
        if not (same and u_to >= u_from):
            j, steps = (k_from + 1) % n, 0
            while True:
                out.append(curve[j])
                if j == k_to % n or steps > n:
                    break
                j = (j + 1) % n
                steps += 1
    else:
        if not (same and u_to <= u_from):
            j, steps = k_from % n, 0
            while True:
                out.append(curve[j])
                if j == (k_to + 1) % n or steps > n:
                    break
                j = (j - 1) % n
                steps += 1
    out.append(np.asarray(p_to, dtype=float))
    q = np.vstack(out)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(q, axis=0), axis=1) > 0))
    q = q[keep]
    if len(q) < 2:
        q = np.vstack([p_from, p_to])
    q[0], q[-1] = p_from, p_to
    return q


def _wall_run(wall: np.ndarray, i_from: int, i_to: int,
              whole: bool = False) -> np.ndarray:
    """Wall samples from one index to another, walking forward with wrap.

    ``whole`` asks for the entire loop back to the start, which is what a body
    with a single corner needs: all of its rays leave the same point, so the
    piece of collar that spans the body runs from that corner the whole way
    round and back to it. Comparing corner *indices* cannot tell that piece from
    the wedges between adjacent rays -- they have identical endpoints.
    """
    n = len(wall)
    if whole:
        idx = [(i_from + k) % n for k in range(n + 1)]
        return wall[np.asarray(idx, dtype=int)]
    if i_from == i_to:
        return wall[[i_from]]
    idx, j = [i_from], i_from
    while j != i_to:
        j = (j + 1) % n
        idx.append(j)
        if len(idx) > n + 1:
            break
    return wall[np.asarray(idx, dtype=int)]


def build_collar(region, body: int = 0, delta_target: float | None = None,
                 rho_max: float = 0.05, n_outer: int = 400,
                 smoothing: int = 60):
    """Blocks for the near-wall collar of one body.

    Returns ``{'wall', 'outer', 'blocks', 'rays', 'delta'}`` where ``blocks`` is
    a list of side-tuples, counter-clockwise. A block between two rays of
    *different* corners is four-sided -- wall, ray, outer, ray -- and one between
    two rays of the *same* corner is the three-sided wedge that corner's element
    count demands, with its apex on the corner.
    """
    from aeromesh.blocking.blocks import clean_sides, orient_ccw
    from aeromesh.blocking.layers import wall_distance
    from aeromesh.medial.axis import exterior_axis

    bl = region.boundaries[body + 1]
    wall = bl.loop.points
    axis = exterior_axis(region)
    delta = wall_distance(axis, region, delta_target=delta_target,
                          rho_max=rho_max)[body + 1]
    outer = smooth_offset(wall, delta, n_out=n_outer, passes=smoothing)

    # ── the rays a reflex corner demands ──
    rays = []
    for ci, alpha, n_c in zip(bl.corners.index, bl.fluid_angles,
                              bl.corner_elements):
        alpha = float(alpha)
        if alpha <= np.pi or int(n_c) < 2:
            continue
        a0, sense = corner_frame(wall, int(ci), alpha)
        # The **boundaries of the corner's normal cone**, not the interior cross
        # directions. The cone is the set of directions in which the corner is
        # the nearest boundary point; it spans ``[pi/2, alpha - pi/2]``, so its
        # two edges are the perpendiculars to the two wall edges forming the
        # corner. This is Fogg's own clamp (sec. 4.1): the radii "are almost
        # aligned and perpendicular to the edges forming the corner".
        #
        # The cross directions ``alpha * i / n_c`` are wrong *here* for a
        # concrete, measurable reason. At a blunt trailing edge they point
        # downstream from both corners and **converge** -- measured, they cross
        # 0.021 chord behind the trailing edge, and a layer 0.02 thick puts its
        # outer curve essentially at that crossing. The base face, 0.002 long,
        # then maps to 0.0001 of outer curve: a band that contracts 20x instead
        # of propagating, which is a block connecting an edge to a point. No
        # amount of resampling fixes it -- it stays 0.0001 at four times the
        # resolution -- because the rays really do meet there.
        #
        # Perpendiculars to a *straight* edge are parallel, so the two rays
        # bounding the base face never converge and the face propagates to its
        # own offset, one to one. Each wall segment maps to exactly its own
        # offset and each corner owns exactly its rounding arc.
        # Fogg's rule, applied properly: the ideal cross directions
        # ``alpha * i / n_c``, each **clamped into the corner's normal cone**
        # ``[pi/2, alpha - pi/2]``, plus the cone's own two boundaries.
        #
        # Clamping alone is not enough and neither is taking the boundaries
        # alone -- the two cases need different things and the one rule gives
        # both. At a blunt trailing edge (alpha = 261.8, n_c = 3) both ideal
        # directions fall outside the cone and clamp onto its boundaries, which
        # is what stops the two corners' rays converging. At a sharp one
        # (alpha = 343.5, n_c = 4) the middle direction at 171.8 degrees is
        # comfortably inside, and keeping it matters: without it the corner's
        # whole 163.2 degree cone is spanned by a single sub-block and the cell
        # at the apex inherits all 163 degrees of it.
        lo, hi = 0.5 * np.pi, alpha - 0.5 * np.pi
        wanted = [lo, hi] + [alpha * i / int(n_c) for i in range(1, int(n_c))]
        phis, seen = [], []
        for w in sorted(float(np.clip(w, lo, hi)) for w in wanted):
            if not seen or abs(w - seen[-1]) > 1e-6 * max(alpha, 1.0):
                seen.append(w)
                phis.append(w)
        for phi_local in phis:
            phi = sense * phi_local
            c, s = np.cos(phi), np.sin(phi)
            d = np.array([c * a0[0] - s * a0[1], s * a0[0] + c * a0[1]])
            hit, k, u = ray_hit(wall[int(ci)], d, outer)
            rays.append({"ci": int(ci), "hit": hit, "k": k, "u": u,
                         "pos": k + u})
    if not rays:
        return {"wall": wall, "outer": outer, "blocks": [], "rays": [],
                "delta": delta}

    rays.sort(key=lambda r: r["pos"])
    outer_perimeter = float(np.sum(np.linalg.norm(
        np.roll(outer, -1, axis=0) - outer, axis=1)))

    # ── one block between each consecutive pair of rays ──
    blocks, kinds, labels = [], [], []
    m = len(rays)
    for i in range(m):
        A, B = rays[i], rays[(i + 1) % m]
        outer_arc = _arc_between(outer, B["k"], B["u"], B["hit"],
                                 A["k"], A["u"], A["hit"], forward=False)

        # How much wall this piece of collar covers is decided by how much of the
        # outer curve its arc subtends, not by comparing the two corner indices.
        # Where a body has a single corner -- a sharp trailing edge -- every ray
        # leaves the same point, so index comparison calls every piece a wedge,
        # including the one that wraps the whole aerofoil. The two populations
        # are orders apart: a wedge's arc is the layer thickness across, a
        # body-spanning arc is most of the perimeter.
        arc_len = float(np.sum(np.linalg.norm(np.diff(outer_arc, axis=0), axis=1)))
        whole = (A["ci"] == B["ci"]) and arc_len > 0.25 * outer_perimeter
        wall_arc = _wall_run(wall, A["ci"], B["ci"], whole=whole)

        ray_b = np.vstack([wall[B["ci"]], B["hit"]])
        ray_a = np.vstack([A["hit"], wall[A["ci"]]])
        if len(wall_arc) > 1:
            sides = (wall_arc, ray_b, outer_arc, ray_a)
            tags = ("wall", "ray", "outer", "ray")
        else:
            sides = (ray_b, outer_arc, ray_a)
            tags = ("ray", "outer", "ray")
        cleaned = clean_sides(sides)
        if len(cleaned) != len(sides):          # a side collapsed
            tags = tuple(t for t, sd in zip(tags, sides) if len(sd) > 1)
        oriented = orient_ccw(cleaned)
        if oriented is not cleaned and len(oriented) == len(cleaned) \
                and not np.array_equal(oriented[0], cleaned[0]):
            tags = tuple(reversed(tags))        # orient_ccw reversed the chain
        blocks.append(oriented)
        labels.append(tags)
        kinds.append("band" if len(wall_arc) > 1 else "wedge")

    return {"wall": wall, "outer": outer, "blocks": blocks, "kinds": kinds,
            "labels": labels, "rays": rays, "delta": delta}


# ═══════════════════════════════════════════════════════════════════
#  Meshing the collar
# ═══════════════════════════════════════════════════════════════════

def resample_by(side: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points at the given fractions of a side's arc length."""
    p = np.asarray(side, dtype=float)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0),
                                                        axis=1))))
    if s[-1] <= 0.0:
        return np.repeat(p[:1], len(t), axis=0)
    return np.column_stack([np.interp(t * s[-1], s, p[:, 0]),
                            np.interp(t * s[-1], s, p[:, 1])])


def wall_clustering(n: int, first: float, total: float) -> np.ndarray:
    """Fractions along the wall-normal direction, clustered at the wall.

    Hyperbolic tangent stretching: the first cell is ``first`` thick and the
    rest grow smoothly out to ``total``. This is the only place ``Delta_y1``
    enters -- the blocking knows nothing about it, which is the point of
    separating the two.
    """
    if n < 2:
        return np.linspace(0.0, 1.0, max(n, 2))
    target = max(min(first / max(total, 1e-300), 0.49), 1e-9)
    lo, hi = 1e-6, 20.0
    for _ in range(80):                          # solve for the stretch factor
        b = 0.5 * (lo + hi)
        t = 1.0 + np.tanh(b * (np.linspace(0, 1, n) - 1.0)) / np.tanh(b)
        if t[1] - t[0] > target:
            lo = b
        else:
            hi = b
    t = 1.0 + np.tanh(b * (np.linspace(0, 1, n) - 1.0)) / np.tanh(b)
    return (t - t[0]) / (t[-1] - t[0])


def tfi(sides, n_u: int, n_v: int) -> np.ndarray:
    """Bilinear transfinite interpolation on four sides, ordered CCW.

    ``sides[0]`` runs along u at v = 0 and the rest follow round, so
    ``sides[2]`` is traversed backwards and ``sides[3]`` likewise. Returns an
    ``(n_u, n_v, 2)`` grid.
    """
    u = np.linspace(0.0, 1.0, n_u)
    v = np.linspace(0.0, 1.0, n_v)
    B = resample_by(sides[0], u)
    R = resample_by(sides[1], v)
    T = resample_by(sides[2], 1.0 - u)
    L = resample_by(sides[3], 1.0 - v)
    # The corner a bilinear patch subtracts at (1,1) is `T[-1]`, and at (0,1)
    # it is `T[0]`. `T` is the opposite side read *backwards*, so `T[0]` is that
    # side's **end** and `T[-1]` is its **start** -- and in a CCW chain the
    # side's start is the (1,1) corner. Swapping the two is a sign-flip on half
    # the correction: the mesh folds over itself, covering 204 per cent of the
    # region it is meant to fill and straying twice the layer thickness outside
    # it. Every aggregate measure survived it -- all Jacobians positive,
    # worst angle 25.6 degrees -- because each patch is individually fine and it
    # is their *placement* that is wrong. Only a coverage check sees it.
    V = [B[0], B[-1], T[-1], T[0]]
    uu, vv = u[:, None, None], v[None, :, None]
    return ((1 - vv) * B[:, None, :] + vv * T[:, None, :]
            + (1 - uu) * L[None, :, :] + uu * R[None, :, :]
            - ((1 - uu) * (1 - vv) * V[0] + uu * (1 - vv) * V[1]
               + uu * vv * V[2] + (1 - uu) * vv * V[3]))


def cell_quality(grid: np.ndarray) -> dict:
    """Minimum interior angle, worst aspect ratio and smallest signed area."""
    p = np.asarray(grid, dtype=float)
    a, b = p[:-1, :-1], p[1:, :-1]
    c, d = p[1:, 1:], p[:-1, 1:]
    quad = np.stack([a, b, c, d], axis=2)             # (I, J, 4, 2)
    e = np.roll(quad, -1, axis=2) - quad
    L = np.linalg.norm(e, axis=3)
    prev = np.roll(e, 1, axis=2)
    cosang = (np.einsum("ijkl,ijkl->ijk", -prev, e)
              / np.maximum(np.linalg.norm(prev, axis=3) * L, 1e-300))
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    area = 0.5 * np.abs(
        np.cross(quad[:, :, 2] - quad[:, :, 0], quad[:, :, 3] - quad[:, :, 1]))
    signed = 0.5 * (np.cross(b - a, d - a) + np.cross(d - c, b - c))
    return {
        "cells": int(area.size),
        "min_angle": float(np.min(ang)),
        "max_angle": float(np.max(ang)),
        "max_aspect": float(np.max(L.max(axis=2)
                                   / np.maximum(L.min(axis=2), 1e-300))),
        "min_signed_area": float(np.min(signed)),
        "all_positive": bool(np.all(signed > 0.0)),
    }


def patch_from_nodes(s0, s1, s2, s3) -> np.ndarray:
    """Transfinite interpolation on four sides given as **exact node lists**.

    Not resampled. The sides handed in are already the nodes the neighbouring
    blocks put there -- the wall clustering on a ray, the arc's own division --
    and resampling them by arc length would quietly replace a clustered ray
    with an evenly spaced one, which is the mismatch this is here to avoid.
    Opposite sides must carry equal counts; that is the interval assignment,
    and it is the caller's job.
    """
    B = np.asarray(s0, dtype=float)
    R = np.asarray(s1, dtype=float)
    T = np.asarray(s2, dtype=float)[::-1]
    L = np.asarray(s3, dtype=float)[::-1]
    if len(B) != len(T) or len(R) != len(L):
        raise ValueError(f"opposite sides disagree: {len(B)}/{len(T)} and "
                         f"{len(R)}/{len(L)}")
    n_u, n_v = len(B), len(R)
    u = np.linspace(0.0, 1.0, n_u)[:, None, None]
    v = np.linspace(0.0, 1.0, n_v)[None, :, None]
    V = [B[0], B[-1], T[-1], T[0]]
    return ((1 - v) * B[:, None, :] + v * T[:, None, :]
            + (1 - u) * L[None, :, :] + u * R[None, :, :]
            - ((1 - u) * (1 - v) * V[0] + u * (1 - v) * V[1]
               + u * v * V[2] + (1 - u) * v * V[3]))


def fermat_point(a, b, c, iters: int = 200) -> np.ndarray:
    """The point joined to three others by segments meeting at 120 degrees.

    A wedge is divided about its centre into three sub-blocks, so that centre is
    a **valence-three node**: three quadrilaterals share it, and their interior
    angles there sum to 360. The best they can do is 120 each, and the point at
    which the three spokes meet at 120 degrees is the Fermat point of the
    triangle formed by the three side midpoints.

    The area centroid is not that point and was measurably not good enough: it
    left one of the three cells at 166 degrees and the other two near 97. Nor
    can elliptic smoothing repair it, because the centre is a *corner* of all
    three sub-blocks and a patch smoother holds its boundary.

    Found by Weiszfeld's iteration, which minimises the total distance to the
    three vertices -- the same point. If a vertex angle reaches 120 degrees the
    minimiser is that vertex itself, and the iteration walks to it.
    """
    pts = np.vstack([a, b, c]).astype(float)
    p = pts.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(pts - p, axis=1)
        if np.any(d < 1e-14):
            break
        w = 1.0 / d
        nxt = (w[:, None] * pts).sum(axis=0) / w.sum()
        if np.linalg.norm(nxt - p) < 1e-15:
            p = nxt
            break
        p = nxt
    return p


def _area_centroid(outline: np.ndarray) -> np.ndarray:
    """Area centroid, not the mean of the sampled points -- a point mean is
    dragged onto whichever side is sampled most finely."""
    x, y = outline[:, 0], outline[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a2 = float(np.sum(cross))
    if abs(a2) < 1e-300:
        return outline.mean(axis=0)
    return np.array([float(np.sum((x + np.roll(x, -1)) * cross)) / (3.0 * a2),
                     float(np.sum((y + np.roll(y, -1)) * cross)) / (3.0 * a2)])


def _inside(point, outline) -> bool:
    """Is a point strictly inside a closed outline?"""
    from shapely.geometry import Point, Polygon
    try:
        poly = Polygon(outline)
        return bool((poly if poly.is_valid else poly.buffer(0))
                    .contains(Point(point)))
    except Exception:                             # pragma: no cover
        return False


def curvature_spacing(side, n: int, strength: float = 1.0) -> np.ndarray:
    """Parameters along a wall side, clustered where it curves.

    Equidistributes ``1 + strength * |kappa| * L / n`` rather than arc length
    alone, so a leading edge of 1 per cent radius gets the points it needs
    without the whole band paying for them. ``strength = 0`` is uniform.

    The weight is scaled by the *mean* cell size so that it is dimensionless
    and the same ``strength`` means the same thing on any chord.
    """
    p = np.asarray(side, dtype=float)
    if len(p) < 5 or n < 3 or strength <= 0.0:
        return np.linspace(0.0, 1.0, max(n, 2))
    seg = np.diff(p, axis=0)
    ds = np.linalg.norm(seg, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(ds)))
    total = cum[-1]
    if total <= 0.0:
        return np.linspace(0.0, 1.0, max(n, 2))

    # discrete curvature at each interior vertex: turn angle over arc length
    t_hat = seg / np.maximum(ds, 1e-300)[:, None]
    cross = (t_hat[:-1, 0] * t_hat[1:, 1] - t_hat[:-1, 1] * t_hat[1:, 0])
    dot = np.einsum("ij,ij->i", t_hat[:-1], t_hat[1:])
    turn = np.abs(np.arctan2(cross, dot))
    mid = 0.5 * (ds[:-1] + ds[1:])
    kappa = np.concatenate(([0.0], turn / np.maximum(mid, 1e-300), [0.0]))
    kappa = np.convolve(kappa, np.ones(5) / 5.0, mode="same")

    w = 1.0 + strength * kappa * total / max(n - 1, 1)
    w_seg = 0.5 * (w[:-1] + w[1:])
    m = np.concatenate(([0.0], np.cumsum(w_seg * ds)))
    m /= max(m[-1], 1e-300)
    return np.interp(np.linspace(0.0, 1.0, n), m, cum / total)


def _corresponding(outer_s, wall_s, t):
    """Parameters along ``outer_s`` that face the wall at parameters ``t``.

    Each point of the offset belongs to the wall point it is nearest to, so
    reading that off gives a monotone map from the outer curve's own parameter
    to the wall's. Inverting it says where to sample the outer side so that
    node ``i`` of each really is across the layer from node ``i`` of the other.
    """
    w = np.asarray(wall_s, dtype=float)
    o = np.asarray(outer_s, dtype=float)
    if len(w) < 2 or len(o) < 2:
        return np.asarray(t, dtype=float)
    seg = w[1:] - w[:-1]
    L2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-300)
    u = np.clip(np.einsum("qij,ij->qi", o[:, None, :] - w[None, :-1, :], seg)
                / L2, 0.0, 1.0)
    foot = w[None, :-1, :] + u[:, :, None] * seg[None, :, :]
    k = np.argmin(np.linalg.norm(foot - o[:, None, :], axis=2), axis=1)

    cw = np.concatenate(([0.0], np.cumsum(np.linalg.norm(seg, axis=1))))
    if cw[-1] <= 0.0:
        return np.asarray(t, dtype=float)
    rows = np.arange(len(o))
    on_wall = (cw[k] + u[rows, k] * (cw[k + 1] - cw[k])) / cw[-1]

    co = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(o, axis=0),
                                                         axis=1))))
    own = co / max(co[-1], 1e-300)
    order = np.argsort(on_wall)
    xs, ys = on_wall[order], own[order]
    keep = np.concatenate(([True], np.diff(xs) > 0))
    out = np.interp(np.asarray(t, dtype=float), xs[keep], ys[keep])

    # Pin the ends and force the map monotone. The two sides are opposite sides
    # of a quadrilateral, so they run in opposite senses and the pairing at the
    # block corners is known exactly: the wall's start faces the outer side's
    # end. Reading it off the geometry instead fails where the band is
    # **pinched** -- a sharp trailing edge makes the wall side start and finish
    # at the same point, so the nearest-foot map is multivalued there and the
    # last parameter jumped from 0.025 back to 1.0, folding the patch and
    # covering 147 per cent of the collar.
    out[0], out[-1] = 1.0, 0.0
    return np.clip(np.minimum.accumulate(out), 0.0, 1.0)


def mesh_collar(collar: dict, n_across: int = 25, target_spacing: float = 0.01,
                first_cell: float = 1e-4,
                wedge_split: float | None = None,
                curvature: float = 2.0) -> list:
    """Fill every collar block by transfinite interpolation.

    **Every ray carries the same nodes.** A ray is shared by the two blocks
    either side of it, so both must put the same number of points on it, in the
    same places. The first version did not: a band gave its ray 24 nodes with
    the wall clustering and the wedge beside it gave the same ray 12, evenly
    spaced -- a hanging node on every ray, and a mesh whose boundary-layer lines
    visibly failed to meet the trailing edge's.

    Making them agree is an **interval assignment**, the integer problem that
    governs any structured multiblock mesh. Working it through, the three
    sub-blocks of a wedge need the spokes to the centre to carry ``a`` nodes,
    ``a``, and ``b``, and the arc ``2a - 1``, where ``a`` and ``b`` are the two
    halves of a split ray. Nothing forces ``a == b``: an earlier reading imposed
    that and so split the ray at its **node-index** midpoint, which under wall
    clustering is 18 per cent of the way along its *length*. The apex sub-block
    was then tiny, the other two huge, and the Fermat point was dragged toward
    the apex. ``wedge_split`` is the fraction of ray *length* to divide at, and
    ``None`` -- the default -- divides at the **node-index** midpoint instead.

    Which is worth a word, because the obvious reasoning is wrong. Under wall
    clustering the node-index midpoint sits 18 per cent along the ray's length,
    so the apex sub-block is small and the other two large, and that looks like
    a defect to be corrected. Dividing by length instead was measured and is
    *worse*: the wedge's smallest angle runs 59.0 degrees at the node midpoint,
    57.4 at 30 per cent of length, 53.6 at 40, 52.0 at 50, 47.2 at 80. The
    clustered split wins because the sub-block that has to turn the corner is
    the one beside the arc, and giving it more of the ray is what costs the
    angle.
    """
    if n_across < 5:
        n_across = 5
    out = []

    # One source of truth per ray. Computing the clustering separately on each
    # side gave the same count and *different positions* -- up to 1.3 per cent
    # of the ray apart -- because the two sides solve the stretching from their
    # own copy of it. Conformality is not "both sides put 25 nodes on it"; it is
    # "both sides put the same 25 nodes on it".
    wall_pts = collar["wall"]
    ray_cache: dict = {}

    def ray_nodes(side):
        """The clustered nodes of a ray, from its wall end outwards."""
        a, b = np.asarray(side[0]), np.asarray(side[-1])
        key = tuple(np.round(np.sort(np.concatenate([a, b])) / 1e-12)
                    .astype(np.int64))
        if key not in ray_cache:
            da = nearest_on_polyline(a[None, :], wall_pts)[1][0]
            db = nearest_on_polyline(b[None, :], wall_pts)[1][0]
            base = side if da <= db else side[::-1]
            thick = float(np.sum(np.linalg.norm(np.diff(base, axis=0), axis=1)))
            v = wall_clustering(n_across, first_cell, thick)
            ray_cache[key] = resample_by(base, v)      # wall -> outer
        return ray_cache[key]

    for sides, tags, kind in zip(collar["blocks"], collar["labels"],
                                 collar["kinds"]):
        if kind == "band":
            w = tags.index("wall")
            wall_s, ray_b, outer_s, ray_a = (sides[(w + j) % 4]
                                             for j in range(4))
            length = float(np.sum(np.linalg.norm(np.diff(wall_s, axis=0),
                                                 axis=1)))
            n_u = max(8, int(round(length / target_spacing)) + 1)
            # Curvature-adapted, not uniform by arc length. Worth it mainly
            # where the wall's curvature varies most: on eight aerofoils the
            # band's minimum angle goes 67.0 -> 73.1 degrees and coverage
            # 99.89 -> 99.97 per cent, and on the high-camber NACA 6409 the
            # whole collar's minimum goes 53.0 -> 58.7, which brings the worst
            # case into line with the rest. It costs 3 per cent on peak aspect
            # ratio. Strength 4 buys a little more in the band (74.5) and 8
            # less, so the optimum is flat and 2 sits inside it.
            t = curvature_spacing(wall_s, n_u, curvature)
            # The outer side is sampled **in correspondence with the wall**,
            # not uniformly along its own arc length. The two are not the same
            # curve length: around a leading edge the offset is 1.83 times the
            # wall it came from, and 1.39 even at mid-chord. Sampling each
            # uniformly makes node i of one the partner of node i of the other
            # when they are nowhere near each other across the layer, and the
            # grid lines fan. Measured, that was 1400 cells -- 24 per cent of
            # the whole mesh -- below 45 degrees, spread over the forward half
            # of the aerofoil and nothing to do with the trailing edge.
            #
            # The correspondence is known, because the outer curve was built as
            # the offset: each of its points belongs to the wall point it is
            # nearest. Invert that and sample where the wall nodes project.
            outer_t = _corresponding(outer_s, wall_s, t)
            # `ray_a` is traversed outer-to-wall in the chain, so its shared
            # nodes are reversed into chain order -- the nodes themselves come
            # from the cache either way.
            out.append(patch_from_nodes(resample_by(wall_s, t),
                                        ray_nodes(ray_b),
                                        # `C[j]` faces wall node `j`; the
                                        # patch reads this side reversed, so
                                        # hand it over reversed.
                                        resample_by(outer_s, outer_t)[::-1],
                                        ray_nodes(ray_a)[::-1]))
            continue

        # ── a corner wedge: apex on the wall, the corner's rounding arc out ──
        ray_1, arc, ray_2 = sides
        apex = (ray_1[0] if np.allclose(ray_1[0], ray_2[-1])
                else ray_1[-1])
        A1 = ray_1 if np.allclose(ray_1[0], apex) else ray_1[::-1]
        A2 = ray_2 if np.allclose(ray_2[0], apex) else ray_2[::-1]
        arc_s = arc if np.allclose(arc[0], A1[-1]) else arc[::-1]

        N1, N2 = ray_nodes(A1), ray_nodes(A2)              # apex -> outer
        # Split the ray at the node nearest `wedge_split` of its *length*.
        cum = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(N1, axis=0),
                                                              axis=1))))
        frac = cum / max(cum[-1], 1e-300)
        j = (len(N1) // 2 if wedge_split is None
             else int(np.argmin(np.abs(frac - wedge_split))))
        j = int(np.clip(j, 1, len(N1) - 2))
        a, b = j + 1, len(N1) - j
        Narc = resample_by(arc_s, np.linspace(0.0, 1.0, 2 * a - 1))

        Ma, Mb, Mc = N1[j], N2[j], Narc[a - 1]
        # The centre is where the three spokes should meet at 120 degrees.
        centre = fermat_point(Ma, Mb, Mc)
        if not _inside(centre, np.vstack([sides[0]]
                                         + [x[1:] for x in sides[1:]])):
            centre = _area_centroid(np.vstack([sides[0]]
                                              + [x[1:] for x in sides[1:]]))
        sp_a = np.linspace(Ma, centre, a)
        sp_b = np.linspace(centre, Mb, a)
        sp_c = np.linspace(Mc, centre, b)

        # chain: apex -> Ma -> centre -> Mb -> apex
        out.append(patch_from_nodes(N1[:j + 1], sp_a, sp_b, N2[:j + 1][::-1]))
        out.append(patch_from_nodes(N1[j:], Narc[:a], sp_c, sp_a[::-1]))
        out.append(patch_from_nodes(Narc[a - 1:], N2[j:][::-1], sp_b[::-1],
                                    sp_c[::-1]))
    return out


def validate_mesh(collar: dict, grids: list, sample: int = 7) -> dict:
    """Does the mesh actually fill the collar, once, and stay inside it?

    Written because it was missing, and its absence cost a round. Every
    *aggregate* measure of the first meshed collar was reassuring -- all
    Jacobians positive, worst interior angle 25.6 degrees -- and the mesh was
    folded over itself: it covered **204 per cent** of the annulus and strayed
    0.04 outside it, twice the layer thickness. Each patch was individually
    fine. It was their placement that was wrong, and no per-patch metric can see
    that.

    So the three questions a per-cell check cannot answer are asked here:
    does the union of the patches equal the region they are meant to fill, do
    any two patches overlap, and does any point lie outside.
    """
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union

    annulus = Polygon(collar["outer"]).difference(Polygon(collar["wall"]))
    hulls = []
    for g in grids:
        ring = np.vstack([g[:, 0], g[-1, :], g[::-1, -1], g[0, ::-1]])
        poly = Polygon(ring)
        hulls.append(poly if poly.is_valid else poly.buffer(0))
    union = unary_union(hulls)
    total = float(sum(h.area for h in hulls))

    outside = 0.0
    for g in grids:
        pts = g.reshape(-1, 2)[::sample]
        d = max(Point(p).distance(annulus) for p in pts)
        outside = max(outside, float(d))

    q = [cell_quality(g) for g in grids]
    out = {
        "coverage": float(union.area / annulus.area),
        "overlap": float((total - union.area) / annulus.area),
        "outside": outside / max(float(np.max(collar["delta"])), 1e-300),
        "cells": int(sum(x["cells"] for x in q)),
        "min_angle": float(min(x["min_angle"] for x in q)),
        "max_angle": float(max(x["max_angle"] for x in q)),
        "max_aspect": float(max(x["max_aspect"] for x in q)),
        "all_positive": bool(all(x["all_positive"] for x in q)),
    }
    out["ok"] = bool(abs(out["coverage"] - 1.0) < 1e-3
                     and abs(out["overlap"]) < 1e-6
                     and out["outside"] < 1e-6
                     and out["all_positive"])
    return out


def smooth_patch(grid: np.ndarray, sweeps: int = 200,
                 omega: float = 1.0) -> np.ndarray:
    """Winslow elliptic smoothing of one structured patch, boundary fixed.

    Thompson, Thames and Mastin's construction: require the curvilinear
    coordinates to be harmonic, which in the computational plane is

        a x_ee - 2 b x_en + c x_nn = 0,
        a = x_n^2 + y_n^2,  b = x_e x_n + y_e y_n,  c = x_e^2 + y_e^2,

    solved by point Gauss-Seidel. The boundary is held, so a smoothed patch
    still meets its neighbours exactly and the interval assignment is untouched.

    This is the remedy for *shear*, which is what the collar's corner wedges
    have: measured, their worst cells sit at the wedge's centre with near-equal
    sides and a 17.8 degree included angle -- rhombi, not slivers. The shear is
    inherent to filling a three-way subdivision algebraically, because the
    spokes are required to carry the wall clustering by one neighbouring
    sub-block and a uniform distribution by the other, and no choice of
    parameterisation satisfies both. An elliptic solve does not have to choose.
    """
    g = np.array(grid, dtype=float)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return g
    for _ in range(max(0, sweeps)):
        xe = 0.5 * (g[2:, 1:-1] - g[:-2, 1:-1])
        xn = 0.5 * (g[1:-1, 2:] - g[1:-1, :-2])
        a = np.einsum("ijk,ijk->ij", xn, xn)
        c = np.einsum("ijk,ijk->ij", xe, xe)
        bb = np.einsum("ijk,ijk->ij", xe, xn)
        den = 2.0 * (a + c)
        den = np.where(np.abs(den) < 1e-300, 1.0, den)
        cross = 0.25 * (g[2:, 2:] - g[2:, :-2] - g[:-2, 2:] + g[:-2, :-2])
        new = ((a[:, :, None] * (g[2:, 1:-1] + g[:-2, 1:-1])
                + c[:, :, None] * (g[1:-1, 2:] + g[1:-1, :-2])
                - 2.0 * bb[:, :, None] * cross) / den[:, :, None])
        g[1:-1, 1:-1] += omega * (new - g[1:-1, 1:-1])
    return g


def smooth_assembly(grids: list, fixed: list, sweeps: int = 60,
                    omega: float = 0.5, tol: float = 1e-12) -> list:
    """Laplacian smoothing across several patches at once, on shared nodes.

    A per-patch smoother holds its own boundary, so a node that is a *corner*
    of several patches can never move -- and the collar's worst node is exactly
    that: the centre of a wedge, where three sub-blocks meet. Assembling them
    first lets it move.

    ``fixed`` lists the curves whose nodes must not move; everything else is
    free. For a wedge those are its two rays, which it shares with the bands,
    and its arc, which is the collar's outer boundary. The wall clustering is
    therefore held at the boundary and only the interior relaxes.

    Deliberately not applied to a whole collar: the bands are anisotropic by
    design, ``Delta_y1`` over wall spacing being a hundred to one, and plain
    Laplacian smoothing equalises spacing -- it would undo the clustering it is
    meant to preserve.
    """
    pts, index, quads = [], {}, []
    scale = max(1e-12, float(np.max([np.ptp(g.reshape(-1, 2), axis=0).max()
                                     for g in grids])))
    tol_key = max(tol * scale, 1e-12)

    def node(p):
        key = tuple(np.round(np.asarray(p, dtype=float) / max(tol_key, 1e-300))
                    .astype(np.int64))
        if key not in index:
            index[key] = len(pts)
            pts.append(np.asarray(p, dtype=float))
        return index[key]

    shapes = []
    for g in grids:
        ids = np.empty(g.shape[:2], dtype=int)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ids[i, j] = node(g[i, j])
        shapes.append(ids)
        for i in range(g.shape[0] - 1):
            for j in range(g.shape[1] - 1):
                quads.append((ids[i, j], ids[i + 1, j],
                              ids[i + 1, j + 1], ids[i, j + 1]))

    P = np.array(pts, dtype=float)
    # Held nodes are found by distance to the fixed *curves*, not by matching
    # their sample positions. The first version matched positions and held
    # nothing at all: the mesh nodes are resampled along each side and almost
    # never land on the side's own samples, so the boundary was free and drifted
    # -- which is what made the first measurement of this function say that
    # assembled smoothing wrecks a wedge. It was the loose boundary doing that.
    held = np.zeros(len(P), dtype=bool)
    near = tol_key
    for curve in fixed:
        q = np.asarray(curve, dtype=float)
        if len(q) < 2:
            continue
        held |= nearest_on_polyline(P, q)[1] <= near

    nbr: list = [[] for _ in range(len(P))]
    for q in quads:
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            nbr[q[a]].append(q[b])
            nbr[q[b]].append(q[a])
    nbr = [np.unique(n) for n in nbr]

    for _ in range(max(0, sweeps)):
        for i in range(len(P)):
            if held[i] or not len(nbr[i]):
                continue
            P[i] += omega * (P[nbr[i]].mean(axis=0) - P[i])

    return [P[ids] for ids in shapes]
