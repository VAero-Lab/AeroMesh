"""Constant-rho splits: the near-wall layer.

The medial coordinates are ``(s, rho)`` with ``rho = d_wall / r_m(s)``, so
``rho = 0`` is the wall and ``rho = 1`` is the medial axis. A **constant-s**
curve is a medial radius pair -- the cuts the decomposition already uses. A
**constant-rho** curve is the other family: a closed curve offset from a body,

    C(s) = T(s) + rho * ( p(s) - T(s) )

read straight off the touch points and the medial polyline that every axis edge
already carries.

Why it matters. Without it a span runs wall to wall through the medial axis, so
for a body in a far field one block reaches from the surface to the truncation
boundary -- an aspect ratio no boundary layer survives. Peeling a layer at
``rho*`` gives a ring hugging the body that can carry the ``Delta_y1``
clustering, and leaves an outer region that cannot contaminate it.

Because ``rho`` is normalised by the local medial radius, the layer is thicker
where the domain is thicker and thins into a slot by itself. That is the
property the medial parameterisation exists for, and ``rho*`` is also the design
variable the optimisation stage varies.
"""

from __future__ import annotations

import numpy as np

from aeromesh.blocking.blocks import arc_length as _arc_length
from aeromesh.blocking.blocks import subcurve
from aeromesh.geometry.loop import Loop


def _loop_of(index: int, offsets: np.ndarray) -> int:
    return int(np.searchsorted(offsets, index, side="right") - 1)


def medial_radius_on_boundary(axis, region) -> dict:
    """The medial radius seen from each boundary sample.

    For boundary sample ``j`` this is the radius of the maximal inscribed circle
    that touches there -- the smallest ``r_m`` among the medial samples whose
    touch is ``j``, since a larger circle through the same point is not the
    maximal one *at* that point. Samples no medial point happens to touch are
    filled in by interpolation along the boundary -- and so are samples whose
    only toucher is degenerate, which is the same situation.

    Degenerate means a medial sample whose circle is smaller than the boundary
    sampling that produced it, or whose separation angle has collapsed. Neither
    describes a maximal inscribed circle; both describe the medial axis of the
    *polygon* rather than of the curve the polygon approximates, and that axis
    grows a spurious branch at essentially every vertex. Because the rule above
    is a **minimum** over touchers, one such sample is enough to carry the whole
    field to zero at that boundary sample, and nothing downstream recovers.

    Measured: NACA 6409 and 6412 -- 6 per cent camber, where the lower surface
    turns concave and the exterior axis comes close to the wall -- each produce
    exactly one medial sample with ``r_m = 0`` and ``theta_m = 0``, out of about
    2075. That one sample set ``r_m = 0`` on the wall near x = 0.38; the
    thickness cap ``rho_max * r_m`` took ``delta`` to zero with it; and the
    collar's outer curve then touched the wall -- aspect ratio 8715, negative
    Jacobians, minimum angle 22 degrees. Every other aerofoil tested was clean,
    which is why this went unseen until the battery was widened to high camber.

    Returns ``{loop_index: array of r_m, one per sample of that loop}``.
    """
    pts, _, offsets = region.sample()
    n_loops = len(region.boundaries)
    best = [np.full(int(offsets[i + 1] - offsets[i]), np.inf) for i in range(n_loops)]

    # Mean point spacing of each loop: the length below which that loop cannot
    # resolve a circle at all.
    floor = np.empty(n_loops)
    for i in range(n_loops):
        q = pts[int(offsets[i]):int(offsets[i + 1])]
        d_i = np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1)
        floor[i] = float(np.mean(d_i)) if len(d_i) else 0.0

    for _, _, d in axis.graph.edges(data=True):
        theta = np.asarray(d.get("theta_m", np.full(len(d["r_m"]), np.inf)),
                           dtype=float)
        for idx in ("i1", "i2"):
            J = d[idx].astype(int)
            for j, r, th in zip(J, d["r_m"], theta):
                li = _loop_of(int(j), offsets)
                if not np.isfinite(r) or r <= floor[li] or th <= 1e-9:
                    continue
                k = int(j) - int(offsets[li])
                if r < best[li][k]:
                    best[li][k] = r

    out = {}
    for li, arr in enumerate(best):
        known = np.flatnonzero(np.isfinite(arr))
        if len(known) < 4:
            continue
        n = len(arr)
        # Interpolate the gaps around the loop, wrapping.
        xp = np.concatenate([known - n, known, known + n])
        fp = np.concatenate([arr[known]] * 3)
        out[li] = np.interp(np.arange(n), xp, fp)
    return out


def _smooth_closed(values: np.ndarray, passes: int) -> np.ndarray:
    """Three-point averaging passes around a closed array."""
    q = np.asarray(values, dtype=float).copy()
    for _ in range(max(0, passes)):
        q = 0.25 * (np.roll(q, 1, axis=0) + 2.0 * q + np.roll(q, -1, axis=0))
    return q


def outward_normals(loop: Loop) -> np.ndarray:
    """Unit normals pointing out of a CCW loop -- into the fluid, for a body."""
    p = loop.points
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
    return np.column_stack([t[:, 1], -t[:, 0]])


def wall_distance(axis, region, delta_target: float | None = None,
                  rho_max: float = 0.05, bodies_only: bool = True,
                  smooth: int = 12) -> dict:
    """The near-wall layer thickness at each boundary sample.

        delta(s) = min( delta_target , rho_max * r_m(s) )

    Why not constant ``rho``. A layer at constant ``rho`` is as thick a fraction
    of the local medial radius everywhere, and for a lone body that is very
    nearly a uniform layer without having been told anything about the geometry
    -- measured over a chord, ``r_m`` varies by 1.1x on a circle in a box and
    1.5x on an aerofoil, so the layer varies by about as little. That is the
    medial parameterisation doing exactly what it is for.

    Put a flap next to the aerofoil and ``r_m`` collapses in the slot: 0.05 to
    2.38, a factor of 47, and a constant fraction of it collapses with it, 42x
    from end to end. The coordinate is behaving correctly and is the wrong
    thickness law. A boundary layer's thickness is set by the flow -- by
    ``Delta_y1``, the growth rate and the wall point count -- not by how much
    room happens to be available; it must merely *fit* in the room available.

    So ``delta_target`` is the thickness asked for and the medial term is a
    ceiling that keeps the layer inside the space there is. ``rho_max`` stops
    being the law and becomes the safety margin -- while remaining the
    continuous quantity the optimisation stage varies, which is the reason for
    introducing it.

    With ``delta_target=None`` the cap is the whole law, which is the old
    constant-``rho`` behaviour.

    Returns ``{loop_index: array of delta, one per sample of that loop}``.
    """
    if not 0.0 < rho_max < 1.0:
        raise ValueError(f"rho_max must lie strictly in (0, 1), got {rho_max}.")
    if delta_target is not None and delta_target <= 0.0:
        raise ValueError(f"delta_target must be positive, got {delta_target}.")

    out = {}
    for li, r_m in medial_radius_on_boundary(axis, region).items():
        if bodies_only and li == 0:
            continue
        # Smooth, then clip back to the raw field. Smoothing alone *raises* the
        # cap exactly where the cap is the only thing preventing a collision:
        # the slot between an aerofoil and its flap has an r_m minimum of 0.0485
        # held over one or two samples, and 12 passes of (1,2,1)/4 diffuse it to
        # 0.606 -- a cap 12.5x too generous, and the two layers then intersect
        # from rho_max = 0.3 upwards. Clipping keeps what smoothing is for,
        # which is removing upward spikes, and restores the guarantee.
        #
        # The guarantee is worth stating, because delta < r_m pointwise is not
        # merely prudent, it is sufficient for the offset to be simple: if the
        # normals from two distinct wall points crossed at X with t_1 < r_m(s_1),
        # X would lie strictly inside the maximal inscribed disc at s_1, whose
        # interior meets the wall nowhere -- contradiction. Body-to-body
        # collision is covered by the same argument, since r_m is the distance
        # to the *nearest* boundary whichever body that is.
        cap = rho_max * np.minimum(_smooth_closed(r_m, smooth), r_m)
        out[li] = cap if delta_target is None else np.minimum(delta_target, cap)
    return out


def rho_offset(axis, region, rho: float = 0.05, bodies_only: bool = True,
               smooth: int = 12, delta_target: float | None = None) -> dict:
    """Closed offset curves at ``rho``, one per boundary the axis touches.

    Each boundary sample is moved along its own outward normal by
    ``rho * r_m``, with ``r_m`` the medial radius seen from that sample.

    Parameters
    ----------
    axis : MedialAxis
        The exterior axis of ``region``.
    region : Region
    rho : float
        The ``rho_max`` of :func:`wall_distance`: with no ``delta_target`` it is
        the offset as a fraction of the local medial radius, and with one it is
        the fraction that caps it.
    delta_target : float, optional
        The thickness the flow asks for. See :func:`wall_distance`.
    bodies_only : bool
        Skip the outer boundary. Offsetting it is well defined but pinches to
        nothing at any corner it has, since a flare's ``r_m`` goes to zero there.
    smooth : int
        Averaging passes applied to the *radius field*, not to the curve.

    Returns
    -------
    dict
        ``{loop_index: Loop}``. Index 0 is the outer boundary; 1.. are bodies in
        the order they were given.

    Notes
    -----
    Offsetting along the boundary's own normal, and smoothing the scalar radius
    rather than the curve, matters. Reading the offset points straight off the
    medial samples gives a curve whose segment lengths span four orders of
    magnitude, because medial sampling has nothing to do with boundary sampling;
    and smoothing *that curve* to tidy it up shortens it, dragging the offset
    onto the wall where the curvature is high. A scalar field can be smoothed
    without moving anything in the wrong direction, and the normal is exact.
    """
    deltas = wall_distance(axis, region, delta_target=delta_target,
                           rho_max=rho, bodies_only=bodies_only, smooth=smooth)
    tag = f"rho={rho:g}" if delta_target is None else f"d={delta_target:g}"
    out = {}
    for li, d in deltas.items():
        loop = region.boundaries[li].loop
        pts = loop.points + d[:, None] * outward_normals(loop)
        out[li] = Loop.from_points(pts, name=f"{loop.name}@{tag}")
    return out


def peel(axis, region, rho: float, spacing: float | None = None):
    """Split a region into near-wall layers and the region that remains.

    Returns ``(inner_region, layers)`` where ``inner_region`` has the same outer
    boundary but each body replaced by its ``rho`` offset, and ``layers`` maps a
    body index to ``(body_loop, offset_loop)`` -- the annulus between them, which
    is a ring block and meshes as an O-grid.

    The remaining region is decomposed by exactly the same machinery as before:
    nothing downstream needs to know a layer was taken off. The number of bodies
    is unchanged, so the retract and index certificates read the same.
    """
    from aeromesh.geometry.region import Region

    offsets = rho_offset(axis, region, rho)
    if len(offsets) != region.n_bodies:
        missing = [i for i in range(1, region.n_bodies + 1) if i not in offsets]
        raise ValueError(
            f"No rho-offset could be built for body index {missing}; the medial "
            f"axis touches too little of it. Refine the boundary sampling."
        )

    bodies = [offsets[i] for i in range(1, region.n_bodies + 1)]
    inner = Region.build(region.outer.loop, bodies, body_spacing=spacing,
                         te="sharp", validate=False)
    layers = {i: (region.holes[i - 1].loop, offsets[i])
              for i in range(1, region.n_bodies + 1)}
    return inner, layers


# ═══════════════════════════════════════════════════════════════════
#  Inserting the layer into the blocking
# ═══════════════════════════════════════════════════════════════════

def _side_kinds(system, tol_factor: float = 1e-7):
    """Label every side of every block ``wall``, ``outer`` or ``cut``.

    Read off the geometry rather than recorded at construction: a side lies on
    the boundary when all of it does, and which boundary decides wall from outer.
    """
    from scipy.spatial import cKDTree

    from aeromesh.medial.fields import segment_distance

    region = system.region
    pts, loop_id, offsets = region.sample()
    tol = tol_factor * region.outer.loop.scale
    # One tree for the whole system. It was being rebuilt inside the inner loop,
    # once per *side*, which on a three-element case is several hundred builds
    # over a few thousand points apiece and was most of the layer stage's cost.
    tree = cKDTree(pts)
    out = {}
    for b in system.blocks:
        kinds = []
        for side in b.sides:
            d = segment_distance(side, pts, loop_id, offsets, tree=tree)
            if float(np.max(d)) > tol:
                kinds.append("cut")
                continue
            # By arc length, not by index: a side's points are boundary samples
            # or a resampled cut and neither is evenly spaced.
            mid = subcurve(side, 0.0, 0.5 * float(_arc_length(side)[-1]))[-1]
            j = int(tree.query(mid[None, :])[1][0])
            kinds.append("outer" if int(loop_id[j]) == 0 else "wall")
        out[b.id] = kinds
    return out


def _normals_along(arc: np.ndarray, region, step: float) -> np.ndarray:
    """Unit normals of a wall arc, signed to point into the fluid.

    The sign cannot be taken from the loop's handedness, because ``orient_ccw``
    may have reversed the side, and it must not be taken by comparing against a
    reference direction read at a different point of the arc: on a closed wall
    curve the normal at the first sample and a radial direction at the midpoint
    are very nearly opposite, which silently offsets the layer *into* the body.

    So it is settled by asking the region which side is fluid -- and the two
    details of *how* both matter.

    **The probe must be short.** Stepping off the wall by a fraction of the
    layer thickness only works if the body is thicker than that. A 0.3-chord
    flap is 0.036 thick while the layer reaches 0.194 on its outboard side, so a
    probe scaled to the *largest* thickness on the arc steps clean through the
    body and finds fluid on both sides -- at every sample. The vote then ties,
    the tie went to the unflipped normal, and the whole near-wall layer was
    built inside the flap. The area check still passed: `tiles=True`,
    relative error 4.5e-4, nothing in the notes. So the probe is scaled to the
    *smallest* thickness and capped by the arc's own sampling.

    **The vote must discard the ambiguous samples.** A sample where the probe
    finds fluid on both sides, or on neither, has stepped across something and
    carries no information; counting it is what produced the tie. Only samples
    that answer unambiguously vote, and they vote over the whole arc rather
    than a fixed nine.
    """
    p = np.asarray(arc, dtype=float)
    t = np.empty_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-300)[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])

    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    seg = seg[seg > 0.0]
    probe = max(min(step, 0.5 * float(np.min(seg)) if len(seg) else step),
                1e-15)

    # The vote is sub-sampled, but nowhere near as far as the version that got
    # this wrong. That one polled 9 points and tied; this polls up to 256, which
    # is 28 times as many, and still discards the ambiguous ones. Polling every
    # point of every arc was most of the layer stage's cost.
    j = (np.arange(len(p)) if len(p) <= 256 else
         np.unique(np.linspace(0, len(p) - 1, 256).astype(int)))
    out_side = region.contains(p[j] + probe * n[j])
    in_side = region.contains(p[j] - probe * n[j])
    clear = out_side != in_side                # the unambiguous samples only
    if not clear.any():
        return n
    return n if int(np.sum(out_side[clear])) >= int(np.sum(in_side[clear])) \
        else -n


def _fluid_arc(centre: np.ndarray, a: np.ndarray, b: np.ndarray, region,
               n: int = 17) -> np.ndarray:
    """A circular arc about ``centre`` from ``a`` to ``b``, taken through the fluid.

    Needed where a block touches the wall at a single point rather than along an
    arc -- the apex of a wake region at a sharp or blunt trailing edge. The wall
    there is one point, so the curve at distance ``delta`` from it is an arc of
    radius ``delta``, not an offset of anything. Which of the two ways round is
    settled by asking the region, because at a trailing edge the fluid is on the
    reflex side and the short way is inside the body.
    """
    r = 0.5 * (float(np.linalg.norm(a - centre)) + float(np.linalg.norm(b - centre)))
    t0 = float(np.arctan2(*(a - centre)[::-1]))
    t1 = float(np.arctan2(*(b - centre)[::-1]))

    def sweep(t0, t1):
        t = np.linspace(t0, t1, n)
        return centre + r * np.column_stack([np.cos(t), np.sin(t)])

    short = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
    for delta in (short, short - np.sign(short or 1.0) * 2 * np.pi):
        arc = sweep(t0, t0 + delta)
        if bool(np.all(region.contains(arc[1:-1]))):
            arc[0], arc[-1] = a, b
            return arc
    arc = sweep(t0, t0 + short)
    arc[0], arc[-1] = a, b
    return arc


def insert_layer(system, delta_target: float | None = None,
                 rho_max: float = 0.05):
    """Split the wall-adjacent blocks to carry a near-wall layer.

    The layer goes in as a **split of the blocking**, not as a new region to be
    decomposed. That matters, and it is the second attempt: peeling the layer off
    and re-running the medial analysis on what was left recomputed a topology it
    should have inherited. The offset curve carries enough small-scale structure
    to grow medial branches of its own, so a body in a box went from 4 blocks to
    15, and smoothing the curve to tame it is worse still -- averaging a curve
    shortens it, and it dragged the offset onto the wall where the curvature is
    high, taking the wall distance from a 1.5x range to 66x.

    Splitting blocks has none of that. Topology is inherited by construction and
    the split is exact rather than a re-derivation that has to agree with the
    original to within a tolerance.

    What makes it exact is that the two families of medial curves are
    perpendicular at the wall. A cut is a medial radius, and a medial radius
    meets the boundary at a right angle; a constant-``rho`` curve is parallel to
    the wall. So the offset meets a cut at the point reached by travelling
    ``delta`` along it from its wall foot -- an arc-length step, not a search,
    with no tolerance anywhere in it.

    **Trimming is a property of the cut, not of the block.** Whether a cut loses
    its last ``delta`` is decided from its endpoint alone -- is that endpoint on
    a body wall, and how thick is the layer there -- so every block sharing the
    cut necessarily decides the same way. Driving it from "this block has a wall
    run next to this cut" instead is wrong, and wrong in a way that is hard to
    see: the wake region of a C-type far field touches the wall only at the
    trailing-edge point, so it has no wall side, kept its cuts at full length,
    and overlapped its neighbours' layer by 2e-8 of the region area -- two orders
    below what the tiling check resolves, and caught by the index certificate.

    A block that touches the wall only at a point still gets a layer: the curve
    at distance ``delta`` from a point is a circular arc, and the wedge between
    it and the two cut ends is that block's share of the band.

    No singularity is created. Where the offset crosses a cut, four blocks meet
    -- the inner and outer halves of the two blocks either side -- so the node
    has valence four and the index certificate is unchanged.

    Parameters
    ----------
    system : BlockSystem
    delta_target : float, optional
        The thickness the flow asks for; see :func:`wall_distance`.
    rho_max : float
        The fraction of the local medial radius that caps it.

    Returns
    -------
    BlockSystem
        A new system. ``notes`` records any wall run or apex left unsplit.
    """
    from scipy.spatial import cKDTree

    from aeromesh.blocking.blocks import (Block, BlockSystem, clean_sides,
                                          local_spacing, orient_ccw)
    from aeromesh.medial.fields import segment_distance

    region, axis = system.region, system.axis
    deltas = wall_distance(axis, region, delta_target=delta_target,
                           rho_max=rho_max)
    pts, loop_id, offsets = region.sample()
    tree = cKDTree(pts)
    tol_at = local_spacing(region)
    wall_tol = 1e-7 * region.outer.loop.scale

    def delta_many(points):
        """The layer thickness at each point; zero where it is not on a wall.

        Keyed to the nearest boundary sample, so every block that shares a cut
        gets the same number for its foot and trims it in the same place.

        Vectorised over the whole arc. Called one point at a time it did its own
        `segment_distance` and tree query per sample, several hundred times per
        side, and that was most of the layer stage's cost.
        """
        q = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.zeros(len(q))
        on_wall = segment_distance(q, pts, loop_id, offsets) <= wall_tol
        if not on_wall.any():
            return out
        j = tree.query(q[on_wall])[1].astype(int)
        li = loop_id[j].astype(int)
        vals = np.zeros(len(j))
        for loop in np.unique(li):
            arr = deltas.get(int(loop))
            if arr is None:
                continue
            m = li == loop
            vals[m] = arr[j[m] - int(offsets[int(loop)])]
        out[on_wall] = vals
        return out

    def delta_at(point):
        """The layer thickness at one point. See :func:`delta_many`."""
        return float(delta_many(np.asarray(point, dtype=float)[None, :])[0])

    kinds = _side_kinds(system)
    blocks, notes, bid = [], list(system.notes), 0

    for b in system.blocks:
        ks = kinds[b.id]

        if b.wraps:
            wall = [i for i, k in enumerate(ks) if k == "wall"]
            if len(wall) != 1:
                blocks.append(Block(id=bid, sides=b.sides, kind=b.kind,
                                    source=b.source, wraps=True))
                bid += 1
                continue
            w = np.asarray(b.sides[wall[0]], dtype=float)
            other = np.asarray(b.sides[1 - wall[0]], dtype=float)
            d = delta_many(w)
            if not np.all(d > 0.0):
                blocks.append(Block(id=bid, sides=b.sides, kind=b.kind,
                                    source=b.source, wraps=True))
                bid += 1
                continue
            n = _normals_along(w, region, 0.25 * float(np.min(d)))
            off = w + d[:, None] * n
            off[-1] = off[0]
            blocks.append(Block(id=bid, sides=(off, w), kind="ring",
                                source=b.source, wraps=True))
            bid += 1
            blocks.append(Block(id=bid, sides=(other, off), kind="ring",
                                source=b.source, wraps=True))
            bid += 1
            continue

        m = b.n_sides
        sides = [np.asarray(x, dtype=float) for x in b.sides]
        lengths = [float(_arc_length(x)[-1]) for x in sides]

        # Every cut end that sits on a body wall loses the layer's thickness
        # there. Decided per cut end, so neighbours agree without being asked.
        trim_head = [0.0] * m
        trim_tail = [0.0] * m
        for i in range(m):
            if ks[i] != "cut":
                continue
            trim_head[i] = delta_at(sides[i][0])
            trim_tail[i] = delta_at(sides[i][-1])
            if trim_head[i] + trim_tail[i] >= lengths[i]:
                notes.append(f"block {b.id}: cut {i} is shorter than the layer "
                             f"it would lose; no layer on it")
                trim_head[i] = trim_tail[i] = 0.0

        def crossing(i, at_start):
            """Where the layer's outer edge meets cut ``i``."""
            d = trim_head[i] if at_start else lengths[i] - trim_tail[i]
            return subcurve(sides[i], 0.0, max(d, 1e-300))[-1] if at_start \
                else subcurve(sides[i], d, lengths[i])[0]

        pieces = []            # (first side index, replacement side)
        consumed = set()       # side indices folded into a replacement

        # ── wall runs: the layer follows an arc of wall ──
        for k0, k1 in _wall_runs(ks):
            prev_i, next_i = (k0 - 1) % m, (k1 + 1) % m
            if ks[prev_i] != "cut" or ks[next_i] != "cut":
                notes.append(f"block {b.id}: wall run {k0}..{k1} is bounded by "
                             f"{ks[prev_i]} and {ks[next_i]}, not cuts; no layer")
                continue
            run = [sides[(k0 + t) % m] for t in range(_run_len(k0, k1, m))]
            if trim_tail[prev_i] <= 0.0 or trim_head[next_i] <= 0.0:
                continue
            Q = crossing(prev_i, False)
            P = crossing(next_i, True)

            offs = []
            for arc in run:
                d = delta_many(arc)
                d[d <= 0.0] = np.min(d[d > 0.0]) if np.any(d > 0.0) else 0.0
                n = _normals_along(arc, region, 0.25 * float(np.min(d)))
                offs.append(arc + d[:, None] * n)
            off = offs[0]
            for nxt in offs[1:]:
                off = np.vstack([off, nxt[1:]])
            off[0], off[-1] = Q, P

            prev_near = subcurve(sides[prev_i],
                                 lengths[prev_i] - trim_tail[prev_i],
                                 lengths[prev_i])
            next_near = subcurve(sides[next_i], 0.0, trim_head[next_i])
            strip = tuple(run) + (next_near, off[::-1], prev_near)
            pin = (False,) * len(run) + (True, False, True)
            blocks.append(Block(
                id=bid, sides=orient_ccw(clean_sides(strip, tol_at=tol_at,
                                                     pinned=pin)),
                kind="layer", source=b.source))
            bid += 1
            pieces.append((k0, off))
            consumed.update((k0 + t) % m for t in range(_run_len(k0, k1, m)))

        # ── apexes: the layer rounds a point of wall ──
        for i in range(m):
            j = (i + 1) % m
            if ks[i] != "cut" or ks[j] != "cut":
                continue
            if trim_tail[i] <= 0.0 or trim_head[j] <= 0.0:
                continue
            apex = sides[i][-1]
            Q = crossing(i, False)
            P = crossing(j, True)
            arc = _fluid_arc(apex, Q, P, region)
            # near_i runs Q -> apex, near_j runs apex -> P, and the arc runs
            # Q -> P, so the wedge closes as Q -> apex -> P -> Q.
            near_i = subcurve(sides[i], lengths[i] - trim_tail[i], lengths[i])
            near_j = subcurve(sides[j], 0.0, trim_head[j])
            blocks.append(Block(
                id=bid,
                sides=orient_ccw(clean_sides((near_i, near_j, arc[::-1]),
                                             tol_at=tol_at,
                                             pinned=(True, True, False))),
                kind="layer", source=b.source))
            bid += 1
            # In the parent, cut i now ends at Q and cut j starts at P, so the
            # arc goes between them in its own direction.
            pieces.append((j, arc))

        if not pieces and not any(trim_head) and not any(trim_tail):
            blocks.append(Block(id=bid, sides=b.sides, kind=b.kind,
                                source=b.source))
            bid += 1
            continue

        # ── what is left of the parent ──
        insert_at: dict = {}
        for k0, curve in pieces:
            insert_at.setdefault(k0, []).append(curve)
        new_sides, new_pin = [], []
        for i in range(m):
            for curve in insert_at.get(i, ()):
                new_sides.append(curve)
                new_pin.append(True)
            if i in consumed:
                continue
            if trim_head[i] > 0.0 or trim_tail[i] > 0.0:
                try:
                    new_sides.append(subcurve(sides[i], trim_head[i],
                                              lengths[i] - trim_tail[i]))
                except ValueError as exc:
                    notes.append(f"block {b.id}: cut {i} {exc}")
                    new_sides.append(sides[i])
                new_pin.append(True)
                continue
            new_sides.append(sides[i])
            new_pin.append(ks[i] == "cut")

        blocks.append(Block(
            id=bid, sides=orient_ccw(clean_sides(tuple(new_sides),
                                                 tol_at=tol_at,
                                                 pinned=tuple(new_pin))),
            kind=b.kind, source=b.source))
        bid += 1

    return BlockSystem(blocks=blocks, region=region, axis=axis, notes=notes)


def _run_len(k0: int, k1: int, m: int) -> int:
    """How many sides a cyclic run from ``k0`` to ``k1`` spans."""
    return (k1 - k0) % m + 1


def _wall_runs(kinds) -> list:
    """Maximal cyclic runs of ``wall`` sides, as ``(first, last)`` index pairs.

    A run can be longer than one side: two wall arcs meeting at a body corner
    are two sides with no cut between them, and the layer follows both.
    """
    m = len(kinds)
    if all(k == "wall" for k in kinds) or not any(k == "wall" for k in kinds):
        return []
    start = next(i for i in range(m) if kinds[i] != "wall")
    runs, i = [], 0
    while i < m:
        j = (start + i) % m
        if kinds[j] != "wall":
            i += 1
            continue
        k = i
        while k < m and kinds[(start + k) % m] == "wall":
            k += 1
        runs.append(((start + i) % m, (start + k - 1) % m))
        i = k
    return runs
