"""Structured meshes around a single body, marched from the wall.

Two constructions, both returning a :class:`MultiBlock`.

``o_mesh``
    One block wrapped on itself. A band around a closed curve is an annulus,
    so nothing is chosen. It rounds every corner, which is what limits it.

``c_mesh``
    The topology that the corner **element counts** ask for. A reflex corner of
    fluid angle ``alpha`` wants ``n_c = round(alpha / 90deg)`` elements
    (``BoundaryLoop.corner_elements``), hence ``n_c - 1`` grid lines leaving
    it. Each wall face next to the corner supplies one -- its own normal -- and
    whatever is left over has to leave the body:

    * a sharp trailing edge is one corner of 343deg, ``n_c = 4``: two wall
      normals and **one line into the fluid**, the wake cut;
    * a blunt trailing edge is two corners of 262deg, ``n_c = 3`` each: the
      surface normal, and the normal of the base face, which is a line into
      the fluid from each corner. The base face sits between them and gets a
      strip of its own.

    So every reflex corner emits one wake line, the wall the long way round
    between wake corners is marched as one C block, and any face left *between*
    wake lines becomes a strip. The code below never asks whether the trailing
    edge is sharp or blunt: a sharp one is the case where the aft face has zero
    length and its strip has zero cells.

    With the outer boundary this produces (two outflow corners), the index
    budget ``sum k = sum (2 - n_c) - 4 chi`` is zero for both trailing edges,
    so there is no irregular node anywhere -- in particular none inside the
    boundary layer, where the collar of October left one at y+ ~ 890.

What is supplied: the body, the first cell height, the far-field distance and
wake length, and target spacings. What is derived: the corners, the element
counts, the wake lines, the block count and every node count.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from aeromesh.mesh.hyperbolic import march
from aeromesh.mesh.spacing import arclength, graded_parameters, layer_heights
from aeromesh.mesh.structured import Block, MultiBlock


# --------------------------------------------------------------------------
# the wall as smooth segments between corners
# --------------------------------------------------------------------------

class Segment:
    """A smooth piece of wall between two corners, as a cubic spline.

    The loop the region stores is a polyline; at y+ = 1 the first cell is
    thinner than the sagitta of its facets at a leading edge, so nodes are
    placed on a spline through it rather than on its chords.
    """

    def __init__(self, pts: np.ndarray, closed: bool = False):
        self.pts = np.asarray(pts, dtype=float)
        self.s = arclength(self.pts)
        self.length = float(self.s[-1])
        if len(self.pts) >= 4 and self.length > 0.0:
            self._spline = CubicSpline(self.s, self.pts,
                                       bc_type="periodic" if closed
                                       else "not-a-knot")
        else:
            self._spline = None

    def __call__(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self._spline is None:
            f = s / max(self.length, 1e-300)
            return self.pts[0] + f[:, None] * (self.pts[-1] - self.pts[0])
        out = self._spline(s)
        out[np.isclose(s, 0.0)] = self.pts[0]
        out[np.isclose(s, self.length)] = self.pts[-1]
        return out

    def dense(self, n: int = 4000) -> np.ndarray:
        return self(np.linspace(0.0, self.length, n))

    def tangent(self, s: float) -> np.ndarray:
        if self._spline is None:
            t = self.pts[-1] - self.pts[0]
        else:
            t = self._spline(s, 1)
        return t / max(np.linalg.norm(t), 1e-300)


def wall_segments(boundary):
    """Clockwise smooth segments of a body, split at its corners.

    Returns ``(segments, corner_elements)`` where segment ``k`` runs from
    corner ``k`` to corner ``k + 1`` (cyclically) and ``corner_elements[k]`` is
    the element count at corner ``k``. A corner-free body gives one closed
    segment and an empty count array.
    """
    pts = np.asarray(boundary.loop.points, dtype=float)[::-1]     # clockwise
    n = len(pts)
    idx = np.sort((n - 1 - np.asarray(boundary.corners.index, dtype=int)) % n)
    if len(idx) == 0:
        closed = np.vstack([pts, pts[:1]])
        return [Segment(closed, closed=True)], np.zeros(0, int)
    # element counts, re-ordered to match the clockwise corner order
    orig = (n - 1 - idx) % n
    lookup = {int(i): int(e) for i, e in zip(boundary.corners.index,
                                             boundary.corner_elements)}
    elems = np.array([lookup[int(i)] for i in orig])
    segs = []
    for k in range(len(idx)):
        a, b = idx[k], idx[(k + 1) % len(idx)]
        ring = np.arange(a, b + 1) if b > a else np.concatenate(
            [np.arange(a, n), np.arange(0, b + 1)])
        segs.append(Segment(pts[ring % n]))
    return segs, elems


def _nodes_on(seg: Segment, **kw) -> np.ndarray:
    dense_s = np.linspace(0.0, seg.length, 4000)
    s_nodes = graded_parameters(seg(dense_s), **kw)
    return seg(s_nodes)


# --------------------------------------------------------------------------
# O-mesh
# --------------------------------------------------------------------------

def o_mesh(boundary, first_cell: float, radius: float = 50.0,
           n_layers: int = 120, growth: float = 1.15,
           h_wall: float = 0.012, h_corner: float = 2.5e-4,
           turn_per_cell: float = np.deg2rad(5.0), ratio: float = 1.15,
           eps: float = 0.5, volume_smoothing: float = 0.3,
           n_wall: int | None = None,
           allow_cusp: bool = False, max_growth: float = 1.3) -> MultiBlock:
    """One annular block marched from the wall to ``radius``.

    An O-mesh gives every boundary node exactly two cells. A corner whose
    element count is four or more (a sharp trailing edge: 343 degrees, four
    elements) cannot be met that way: the march has to turn the grid through
    the whole corner within one cell, and the cell at the cusp degenerates.
    Measured on NACA 0012: a face at 180 degrees of non-orthogonality
    (``checkMesh``) and a floating-point exception in the solver within 220
    iterations. Such a body is therefore refused unless ``allow_cusp`` is set,
    which exists only to reproduce that failure as a baseline. The element
    count asks for a C-mesh there, which is what ``c_mesh`` builds.

    ``n_wall`` pins the number of wall nodes (the size field is scaled to
    give it), so meshes from different generators can be compared at equal
    grid dimensions.
    """
    segs, elems = wall_segments(boundary)
    if not allow_cusp and len(elems) and int(np.max(elems)) >= 4:
        raise ValueError(
            "o_mesh: a corner asks for %d elements; an O-mesh gives it 2 and "
            "degenerates there (180-degree face). Use c_mesh." % np.max(elems))
    closed = len(segs) == 1 and segs[0]._spline is not None and \
        np.allclose(segs[0].pts[0], segs[0].pts[-1])
    parts = []
    alloc = [None] * len(segs)
    if n_wall is not None:
        # share the pinned count in proportion to the count each segment
        # would take unpinned -- not to its length: by length a 0.002-chord
        # blunt base got 2 cells of 383 and the O-mesh's worst boundary-layer
        # skew went 42 -> 76 degrees. At least 2 cells per segment; the
        # longest segment absorbs the rounding.
        nat = []
        for seg in segs:
            hc = None if closed else h_corner
            nat.append(len(_nodes_on(seg, h_max=h_wall, h_start=hc, h_end=hc,
                                     ratio=ratio,
                                     turn_per_cell=turn_per_cell)) - 1)
        alloc = [max(2, int(round(n_wall * k / sum(nat)))) for k in nat]
        big = int(np.argmax([s.length for s in segs]))
        alloc[big] += n_wall - sum(alloc)
    for k, seg in enumerate(segs):
        hc = None if closed else h_corner
        parts.append(_nodes_on(seg, h_max=h_wall, h_start=hc, h_end=hc,
                               ratio=ratio, turn_per_cell=turn_per_cell,
                               n=alloc[k])[:-1])
    wall = np.concatenate(parts)
    heights = layer_heights(np.full(len(wall), first_cell), n_layers,
                            growth=growth, distance=radius,
                            max_growth=max_growth)
    grid = march(wall, heights, periodic=True, eps=eps,
                 volume_smoothing=volume_smoothing)
    grid = np.concatenate([grid, grid[:1]], axis=0)        # close the seam
    return MultiBlock([Block(grid, {"jmin": "wall", "jmax": "farfield"},
                             name="O")])


# --------------------------------------------------------------------------
# C-mesh
# --------------------------------------------------------------------------

def wake_centreline(start: np.ndarray, direction: np.ndarray, length: float,
                    axis=(1.0, 0.0), relax: float = 1.0,
                    n: int = 4000) -> np.ndarray:
    """A wake line leaving ``start`` along ``direction`` and turning onto
    ``axis`` over a distance ``relax``.

    The departure direction is the geometry's (the trailing-edge bisector);
    the axis is the modelling input (the freestream direction, which a mesher
    does not know and is told). ``relax`` is a continuous parameter the
    optimisation layer may own.
    """
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    phi0 = np.arctan2(axis[0] * d[1] - axis[1] * d[0], float(axis @ d))
    # geometric stepping: fine near the corner, coarse far away
    s = np.concatenate(([0.0], np.geomspace(1e-6, float(length), n - 1)))
    phi = phi0 * np.exp(-s / max(relax, 1e-12))
    base = np.arctan2(axis[1], axis[0])
    ang = base + phi
    mid = 0.5 * (ang[:-1] + ang[1:])
    step = np.diff(s)[:, None] * np.column_stack([np.cos(mid), np.sin(mid)])
    return np.vstack([start, start + np.cumsum(step, axis=0)])


def c_mesh(boundary, first_cell: float, radius: float = 50.0,
           wake_length: float = 50.0, n_layers: int = 120,
           growth: float = 1.15, h_wall: float = 0.012,
           h_te: float | None = None, te_aspect: float = 135.0,
           h_wake: float | None = None,
           turn_per_cell: float = np.deg2rad(5.0), ratio: float = 1.15,
           wake_axis=(1.0, 0.0), wake_relax: float = 1.0,
           wake_spread: float = 1e6, wake_soft_cap: float | None = None,
           spread_ref: str = "first",
           aft_edge_factor: float = 4.0, aft_divergence: float = 0.2,
           eps: float = 0.5,
           volume_smoothing: float = 0.3, base_aspect: float | None = None,
           n_wall: int | None = None,
           n_wake: int | None = None,
           max_growth: float = 1.3) -> MultiBlock:
    """C block marched from the wall and its wake lines, plus aft strips.

    Along the wake the first cell height (and the width of an aft strip) grows
    **in proportion to the streamwise spacing**, so a wake-cut cell keeps the
    aspect ratio it has at the trailing edge instead of becoming a sliver four
    chords long. ``wake_spread`` caps that growth. The first version saturated
    it at 40 and OpenFOAM's ``checkMesh`` -- not our own metrics -- reported 26
    faces at 87 degrees of non-orthogonality: on a cell of aspect ratio 20000 a
    one per cent taper in height moves the centroid along the cell by more
    than the cell is tall.
    """
    segs, elems = wall_segments(boundary)
    if len(elems) == 0:
        raise ValueError("a corner-free body emits no wake line; use o_mesh")
    # The streamwise spacing at the trailing edge is tied to the first cell
    # height through the aspect ratio of the corner cell. It is not a free
    # number: finite-volume non-orthogonality around the wake cut grows with
    # that aspect ratio (measured: 45.7 degrees at 600, 12.8 at 135, 6.3 at
    # 67), so a finer first cell has to bring a finer trailing edge with it.
    if h_te is None:
        h_te = min(te_aspect * first_cell, 0.5 * h_wall)
    # every reflex corner (three or more elements) emits a wake line
    wake_corner = elems >= 3
    if not wake_corner.any():
        raise ValueError("no reflex corner asks for a line into the fluid")
    # the wall proper is the long way round between wake corners; faces left
    # between wake lines are 'aft' and get strips.
    lengths = np.array([s.length for s in segs])
    main = int(np.argmax(lengths))
    # A blunt base is wall too, and its wall-normal direction is the wake's
    # streamwise one: the first wake spacing is the base's first cell. Tied to
    # the trailing edge's aspect it is 135 first-cell heights, and the solver
    # sees y+ = 25 on the base against 0.6 on the rest of the wall.
    # ``base_aspect`` refines the first wake spacing alone. It is OFF by
    # default because the trade was measured and is bad in this construction:
    # on NACA 0012 the worst boundary-layer skew goes 4.8 -> 14.9 / 18.0 /
    # 22.8 / 25.7 degrees at base aspects 32 / 24 / 16 / 12, and cells invert
    # at 8 (at 12 on NACA 2424). The spacing jump at the corner, from the
    # wall's trailing-edge cell to the wake's first, is what the corner cells
    # cannot absorb; refining the wall towards the corner as well destabilised
    # the march below the section. Closing the base needs that grading done
    # properly, which is open work.
    h_base = h_te
    if len(segs) > 1 and base_aspect is not None:
        h_base = min(h_te, base_aspect * first_cell)
    order = [(main + k) % len(segs) for k in range(len(segs))]
    aft = [segs[k] for k in order[1:]]
    wall_seg = segs[main]

    # aft gap: from the end of the wall round to its start
    a_pt, b_pt = wall_seg.pts[-1], wall_seg.pts[0]      # end, start of wall
    gap = float(np.linalg.norm(b_pt - a_pt))
    u_start = wall_seg.tangent(0.0)                     # into the wall from b
    u_end = -wall_seg.tangent(wall_seg.length)          # into the wall from a
    d0 = -(u_start + u_end)
    d0 /= np.linalg.norm(d0)
    centre0 = 0.5 * (a_pt + b_pt)

    if h_wake is None:
        h_wake = radius / 12.0
    cl = wake_centreline(centre0, d0, wake_length, wake_axis, wake_relax)
    s_w = graded_parameters(cl, h_max=h_wake, h_start=h_base, ratio=ratio,
                            n=n_wake)
    s_cl = arclength(cl)
    centre = np.column_stack([np.interp(s_w, s_cl, cl[:, 0]),
                              np.interp(s_w, s_cl, cl[:, 1])])
    tang = np.gradient(cl, s_cl, axis=0, edge_order=1)
    tang = np.column_stack([np.interp(s_w, s_cl, tang[:, 0]),
                            np.interp(s_w, s_cl, tang[:, 1])])
    tang /= np.linalg.norm(tang, axis=1)[:, None]
    left = np.column_stack([-tang[:, 1], tang[:, 0]])
    ds_w = np.gradient(s_w)
    # Widening is measured against the default trailing-edge spacing
    # (135 first cells), fixed whatever h_te and h_base are, not against
    # the first wake spacing: that one is the base's first cell, and when it
    # is refined to resolve the base (base_aspect) a ratio to it fans the two
    # wake lines apart at once -- 17x the width at base aspect 8, which is
    # what inverted cells above the section. ``spread_ref='first'`` keeps the
    # old law.
    ref = (ds_w[0] if spread_ref == "first" else
           max(min(135.0 * first_cell, 0.5 * h_wall), ds_w[0]))
    spread = np.clip(ds_w / ref, 1.0, wake_spread)
    if wake_soft_cap is not None:
        spread = 1.0 + wake_soft_cap * np.tanh((spread - 1.0) / wake_soft_cap)
    spread[0] = 1.0
    # One widening law for both the first cell on the wake cut and the aft
    # strip. They must share it: a cell that tapers has its centroid moved
    # along it, and if the cells either side of a wake line taper at different
    # rates their centroids part company -- 35 degrees of non-orthogonality
    # across the line, measured, when the first cell grew at the full rate and
    # the strip at a fifth of it. The gentle rate is taken for both, because at
    # the full rate the two wake lines leave the corners 14 degrees apart.
    spread = 1.0 + aft_divergence * (spread - 1.0)

    # which side of the centreline is the wall's start, which its end
    side_b = np.sign(float((b_pt - centre0) @ left[0])) if gap > 0 else 1.0
    half = 0.5 * gap * spread
    line_b = centre + side_b * half[:, None] * left    # leaves from wall start
    line_a = centre - side_b * half[:, None] * left    # leaves from wall end
    line_b[0], line_a[0] = b_pt, a_pt

    # wall nodes
    wall = _nodes_on(wall_seg, h_max=h_wall, h_start=h_te, h_end=h_te,
                     ratio=ratio, turn_per_cell=turn_per_cell,
                     n=None if n_wall is None else n_wall - 1)
    n_w = len(s_w) - 1
    curve = np.vstack([line_b[::-1], wall[1:-1], line_a])
    first = np.concatenate([first_cell * spread[::-1],
                            np.full(len(wall) - 2, first_cell),
                            first_cell * spread])
    heights = layer_heights(first, n_layers, growth=growth, distance=radius,
                            max_growth=max_growth)
    # the two ends slide on the outflow plane, normal to the wake there
    end_dir = np.array([left[-1] * side_b * -1.0, left[-1] * side_b])
    # region lies to the left of the curve: at the first end the curve runs
    # back up the wake, so its left is the centreline's right on that side.
    t0 = curve[1] - curve[0]
    n0 = np.array([-t0[1], t0[0]])
    if end_dir[0] @ n0 < 0:
        end_dir = -end_dir
    t1 = curve[-1] - curve[-2]
    n1 = np.array([-t1[1], t1[0]])
    if end_dir[1] @ n1 < 0:
        end_dir[1] = -end_dir[1]
    grid = march(curve, heights, periodic=False, eps=eps,
                 volume_smoothing=volume_smoothing, end_directions=end_dir)

    n_c = len(curve)
    jmin = ([None] * n_w + ["wall"] * (n_c - 1 - 2 * n_w) + [None] * n_w)
    blocks = [Block(grid, {"jmin": jmin, "jmax": "farfield",
                           "imin": "outflow", "imax": "outflow"}, name="C")]

    # aft strips: one per face between wake lines (none for a sharp edge)
    if gap > 0.0 and aft:
        face = aft[0] if len(aft) == 1 else Segment(
            np.vstack([s.pts for s in aft]))
        # the face runs from the wall's end (a) to its start (b)
        edge = aft_edge_factor * first_cell
        dense_s = np.linspace(0.0, face.length, 2000)
        s_f = graded_parameters(face(dense_s), h_max=face.length / 6.0,
                                h_start=edge, h_end=edge, ratio=1.2)
        frac = s_f / face.length
        row0 = face(s_f)
        row0[0], row0[-1] = a_pt, b_pt
        strip = (line_a[:, None, :] * (1.0 - frac)[None, :, None]
                 + line_b[:, None, :] * frac[None, :, None])
        strip[0] = row0
        # orient so cells are counter-clockwise
        e1 = strip[1, 0] - strip[0, 0]
        e2 = strip[0, 1] - strip[0, 0]
        if e1[0] * e2[1] - e1[1] * e2[0] < 0:
            strip = strip[:, ::-1]
        blocks.append(Block(strip, {"imin": "wall", "imax": "outflow"},
                            name="aft"))
    return MultiBlock(blocks)
