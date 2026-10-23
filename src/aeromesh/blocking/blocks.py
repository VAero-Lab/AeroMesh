"""Block construction.

The decomposition is constructive, straight off the medial graph, rather than a
planar arrangement computed from a soup of curves:

* cutting a medial edge at a set of positions makes each span between
  consecutive cuts a **four-sided block** -- two radius pairs and two boundary
  arcs, with the medial axis running through its middle;
* each medial vertex becomes an **m-sided region** bounded by the innermost cut
  on each incident edge and the boundary arcs between them;
* a medial edge that closes on itself with no vertices at all is a **ring**, one
  block wrapping around on itself -- the doughnut case of Rigby's rules.

So the face structure is the medial graph's, and no arrangement has to be
computed. Nothing here is specific to any body or any truncation shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from aeromesh.blocking.splits import Split, radius_split


# ═══════════════════════════════════════════════════════════════════
#  Boundary arcs
# ═══════════════════════════════════════════════════════════════════

def _edge_key(u, v, key) -> tuple:
    """A direction-independent key for a medial edge.

    ``graph.edges(keys=True)`` yields ``(u, v, key)`` in one order, but
    ``graph.edges(node, keys=True)`` yields ``(node, other, key)`` -- reversed
    for one of the two endpoints. Anything keyed on the raw tuple therefore
    misses half its lookups.
    """
    return (min(u, v), max(u, v), key)


def _loop_of(index: int, offsets: np.ndarray) -> int:
    return int(np.searchsorted(offsets, index, side="right") - 1)


def _short_path(i_from: int, i_to: int, start: int, n: int) -> np.ndarray:
    """Boundary indices from ``i_from`` to ``i_to`` the short way round a loop."""
    a, b = i_from - start, i_to - start
    fwd = (b - a) % n
    if fwd <= n - fwd:
        steps = np.arange(a, a + fwd + 1)
    else:
        steps = np.arange(a, a - (n - fwd) - 1, -1)
    return (steps % n) + start


def boundary_arc(touch_index: np.ndarray, pts: np.ndarray, offsets: np.ndarray,
                 start: np.ndarray | None = None,
                 end: np.ndarray | None = None) -> np.ndarray:
    """The boundary polyline traced by a run of touch indices.

    The touches of consecutive medial samples are consecutive-ish boundary
    samples, so the arc is recovered by stitching the short path between each
    pair. That handles direction and wrap-around without having to work either
    out, and it fills in any boundary samples the medial sampling stepped over.

    ``start`` and ``end`` replace the first and last vertex. They have to be
    given, because a split ends at the *exact* perpendicular foot on the
    boundary while the arc is built from boundary *samples*, and the two differ
    by up to half a sample spacing. Leaving that mismatch open leaves a gap of
    exactly that size at every join between a split and an arc, which is both a
    hole in the block outline and the cause of the outline crossing itself when
    it is closed.
    """
    loop = _loop_of(int(touch_index[0]), offsets)
    loop_start = int(offsets[loop])
    n = int(offsets[loop + 1] - offsets[loop])
    path = [int(touch_index[0])]
    for a, b in zip(touch_index[:-1], touch_index[1:]):
        seg = _short_path(int(a), int(b), loop_start, n)
        path.extend(int(x) for x in seg[1:])
    # drop immediate backtracking left by a touch that paused
    out = [path[0]]
    for x in path[1:]:
        if len(out) >= 2 and x == out[-2]:
            out.pop()
        elif x != out[-1]:
            out.append(x)

    arc = pts[np.asarray(out, dtype=int)].astype(float).copy()
    if start is not None:
        arc[0] = start
    if end is not None:
        arc[-1] = end
    return arc


def arc_length(curve: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def subcurve(curve: np.ndarray, a: float, b: float) -> np.ndarray:
    """The piece of a polyline between arc lengths ``a`` and ``b``.

    Returned in the curve's own orientation, with both cut points exact, so two
    pieces of one curve share their common endpoint to the last bit and the
    blocks either side of it meet without a gap.
    """
    c = np.asarray(curve, dtype=float)
    s = arc_length(c)
    if not (0.0 <= a < b <= s[-1] + 1e-12):
        raise ValueError(f"cannot take [{a:.6g}, {b:.6g}] of a curve of "
                         f"length {s[-1]:.6g}")
    b = min(b, s[-1])

    def at(dist):
        if dist <= 0.0:
            return c[0], 0
        if dist >= s[-1]:
            return c[-1], len(c)
        j = int(np.searchsorted(s, dist))
        t = (dist - s[j - 1]) / max(s[j] - s[j - 1], 1e-300)
        return c[j - 1] + t * (c[j] - c[j - 1]), j

    pa, ja = at(a)
    pb, jb = at(b)
    middle = c[ja:jb]
    out = [pa[None, :]]
    if len(middle):
        out.append(middle)
    out.append(pb[None, :])
    q = np.vstack(out)
    keep = np.concatenate(([True], np.linalg.norm(np.diff(q, axis=0), axis=1) > 0))
    q = q[keep]
    if len(q) < 2:
        q = np.vstack([pa, pb])
    q[0], q[-1] = pa, pb
    return q


# ═══════════════════════════════════════════════════════════════════
#  Blocks
# ═══════════════════════════════════════════════════════════════════

def local_spacing(region):
    """A callable giving the boundary's own sample spacing near a point.

    A single tolerance for the whole region does not work: the spacing is set
    per loop, and taking the largest lets the far field decide what counts as a
    feature on a body. A box 40 across sampled at 720 points has a spacing of
    0.055, and twice that erases real sides on a 0.3-chord flap. Taking the
    smallest instead just moves the failure to the far field. So the tolerance
    has to be read where the point is.

    Returns ``f(point) -> 2 * local spacing``, or 0.0 for a point that is not on
    the boundary at all, which must never be snapped.
    """
    pts, _, offsets = region.sample()
    spacing = np.empty(len(pts))
    for i in range(len(offsets) - 1):
        a, b = int(offsets[i]), int(offsets[i + 1])
        q = pts[a:b]
        step = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        spacing[a:b] = 0.5 * (step + np.roll(step, 1))
    tree = cKDTree(pts)

    def tol_at(point) -> float:
        d, j = tree.query(np.asarray(point, dtype=float))
        h = float(spacing[int(j)])
        return 2.0 * h if d <= 2.0 * h else 0.0

    return tol_at


def clean_sides(sides, tol: float = 1e-12, snap_tol: float = 0.0,
                tol_at=None, pinned=None) -> tuple:
    """Drop sides that have collapsed to a point, and repeated vertices.

    Two cuts can terminate at exactly the same boundary point -- most often a
    sharp corner on a body, where the medial radii from either side both land on
    the corner. The arc between them is then empty, and an m-sided region is
    really (m-1)-sided. Keeping the empty side leaves a repeated vertex in the
    outline, which makes it non-simple for no geometric reason.

    ``tol_at``, when given, replaces ``snap_tol`` with a tolerance read at the
    point in question -- see :func:`local_spacing`.

    ``pinned`` marks the sides whose endpoints are analytic and must not move.
    A cut is pinned: it is a medial radius, its feet are exact perpendicular
    feet, and the two blocks either side of it compute it from the same data, so
    it is identical in both. A boundary arc is not pinned: it is traced through
    boundary *samples*, and its ends are already overwritten with the exact feet
    where it meets a cut. Closing a gap at a cut/arc join by moving both ends to
    their midpoint therefore moves the one endpoint that was right, and moves it
    by an amount that depends on the other sides of *this* block -- so the
    neighbour sharing that cut does not follow, and the shared side disagrees at
    one end. The crack that opens is a lens about a sample spacing wide: far too
    small for an area test (1e-7 of the region) and plainly visible to the index
    certificate, which counts the two ends as two block corners.

    ``snap_tol`` closes joins narrower than itself by moving both ends to their
    midpoint. Two arcs that meet at a vertex's shared touch can land on adjacent
    boundary *samples*, half a spacing apart; that is a representation artefact
    of the same point, not a gap in the geometry. The tolerance has to stay well
    below any real feature, so it is set from the boundary sampling.
    """
    if pinned is None:
        pinned = [False] * len(sides)
    out, pin = [], []
    for side, is_pin in zip(sides, pinned):
        side = np.asarray(side, dtype=float)
        if len(side) > 1:
            ends = (side[0].copy(), side[-1].copy())
            keep = np.concatenate(
                ([True], np.linalg.norm(np.diff(side, axis=0), axis=1) > tol))
            side = side[keep]
            if is_pin and len(side) >= 2:
                side = side.copy()
                side[0], side[-1] = ends
        if len(side) < 2:
            continue
        out.append(side)
        pin.append(bool(is_pin))

    if (snap_tol > 0.0 or tol_at is not None) and len(out) >= 2:
        for i in range(len(out)):
            j = (i + 1) % len(out)
            a, b = out[i][-1], out[j][0]
            gap = float(np.linalg.norm(a - b))
            lim = (min(tol_at(a), tol_at(b)) if tol_at is not None else snap_tol)
            if 0.0 < gap <= lim:
                # Move the approximate end onto the analytic one; only fall back
                # to the midpoint when neither end is authoritative.
                if pin[j] and not pin[i]:
                    mid = b
                elif pin[i] and not pin[j]:
                    mid = a
                else:
                    mid = 0.5 * (a + b)
                out[i] = np.vstack([out[i][:-1], mid])
                out[j] = np.vstack([mid, out[j][1:]])
    return tuple(out)


def orient_ccw(sides) -> tuple:
    """Wind a block's sides counter-clockwise.

    Blocks are assembled by walking the medial graph, which does not fix a
    handedness, so some come out clockwise. Consistency matters twice over: two
    blocks sharing a side must traverse it in opposite directions for the
    interface to make sense, and transfinite interpolation needs to know which
    way round the four sides go.
    """
    sides = list(sides)
    if not sides:
        return tuple(sides)
    pts = np.vstack([sides[0]] + [s[1:] for s in sides[1:]])
    x, y = pts[:, 0], pts[:, 1]
    if 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) >= 0.0:
        return tuple(sides)
    return tuple(s[::-1] for s in reversed(sides))


@dataclass(frozen=True)
class Block:
    """One block of the decomposition.

    Attributes
    ----------
    id : int
    sides : tuple of np.ndarray
        The logical sides in order; consecutive sides share an endpoint and the
        last closes onto the first. A ``span`` block has four, a ``vertex``
        region has one per incident medial edge, a ``ring`` has two (the two
        boundaries it separates) and wraps.
    kind : {'span', 'vertex', 'ring'}
    source : tuple
        The medial edge key or vertex id the block came from.
    wraps : bool
        True for a ring, whose sides are closed curves rather than a chain.
    """

    id: int
    sides: tuple
    kind: str
    source: tuple
    wraps: bool = False

    @property
    def n_sides(self) -> int:
        return len(self.sides)

    def outline(self) -> np.ndarray:
        """The block's closed outline, for area and plotting."""
        if self.wraps:
            return np.vstack(self.sides[:1])
        parts = [self.sides[0]]
        for s in self.sides[1:]:
            parts.append(s[1:] if len(s) > 1 else s)
        return np.vstack(parts)

    @property
    def area(self) -> float:
        """Unsigned area of the outline. For a ring this is the outer loop only;
        :meth:`BlockSystem.covered_area` handles the hole."""
        p = self.outline()
        x, y = p[:, 0], p[:, 1]
        return abs(0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    def __repr__(self) -> str:
        return (f"Block({self.id}, {self.kind}, sides={self.n_sides}, "
                f"area={self.area:.4g})")


@dataclass(frozen=True)
class Interface:
    """A side shared by exactly two blocks.

    ``flipped`` records that the two blocks traverse the shared side in opposite
    directions, which they must: each block walks its own outline the same way
    round, so a side they share is walked one way by one and the other way by
    the other.
    """

    block_a: int
    side_a: int
    block_b: int
    side_b: int
    flipped: bool


@dataclass
class BlockSystem:
    """The blocks of one region, with the checks that say they are a tiling."""

    blocks: list = field(default_factory=list)
    region: object = None
    axis: object = None
    notes: list = field(default_factory=list)

    # ── connectivity ────────────────────────────────────────────────

    def _side_key(self, side, tol):
        """A key two blocks sharing a side both compute, whichever way they
        walk it.

        The midpoint has to be taken at half the **arc length**, not at
        ``side[len(side) // 2]``. The index midpoint is not invariant under
        reversal: for a side of 118 points one block takes index 59 and the
        neighbour, walking the other way, takes 58, so the two keys differ and
        the side pairs with nothing. Cuts escaped it by luck -- ``radius_split``
        resamples to 33 points, and 33 is odd -- but a near-wall layer's offset
        curve has whatever length the wall arc gave it, and on an aerofoil in a
        box ten sides went unpaired. They were then filed as boundary sides,
        where the old ``accounted`` tautology could not see them.
        """
        a, b = side[0], side[-1]
        lo, hi = (a, b) if tuple(a) <= tuple(b) else (b, a)
        total = float(arc_length(side)[-1])
        mid = (subcurve(side, 0.0, 0.5 * total)[-1] if total > 0.0
               else np.asarray(side[0], dtype=float))
        return (tuple(np.round(lo / tol).astype(np.int64)),
                tuple(np.round(hi / tol).astype(np.int64)),
                tuple(np.round(mid / tol).astype(np.int64)))

    def connectivity(self, tol: float | None = None) -> dict:
        """Work out which blocks share which sides.

        Adjacent blocks are built from the *same* ``radius_split`` call, so a
        shared cut is byte-identical in both and matching on endpoints plus a
        midpoint is enough to pair them without geometric search.

        Returns ``interfaces`` (sides shared by two blocks), ``boundary``
        (sides lying on the region boundary, belonging to one block), and
        ``over_shared`` (any side claimed by three or more, which would mean the
        decomposition is not a manifold).
        """
        if tol is None:
            tol = 1e-7 * self.region.outer.loop.scale
        buckets: dict = {}
        for b in self.blocks:
            if b.wraps:
                continue
            for i, side in enumerate(b.sides):
                buckets.setdefault(self._side_key(side, tol), []).append((b.id, i))

        interfaces, boundary, over = [], [], []
        by_id = {b.id: b for b in self.blocks}
        for members in buckets.values():
            if len(members) == 1:
                boundary.append(members[0])
            elif len(members) == 2:
                (ba, sa), (bb, sb) = members
                flipped = bool(np.allclose(by_id[ba].sides[sa][0],
                                           by_id[bb].sides[sb][-1], atol=tol))
                interfaces.append(Interface(ba, sa, bb, sb, flipped))
            else:
                over.append(members)
        return {"interfaces": interfaces, "boundary": boundary,
                "over_shared": over}

    def interface_report(self) -> dict:
        """Is the decomposition conformal?

        Every side must be either on the region boundary or shared with exactly
        one other block. A side claimed by three blocks, or a side that should
        pair but does not, is a non-conformal decomposition -- which no area
        test can see, and which would make the mesh fail to match across the
        interface in S4.
        """
        from aeromesh.medial.fields import segment_distance

        c = self.connectivity()
        n_sides = sum(b.n_sides for b in self.blocks if not b.wraps)
        paired = 2 * len(c["interfaces"])

        # A side that pairs with nobody is filed as "boundary", and then
        # `paired + boundary == n_sides` holds whatever happened -- it is a
        # tautology once `over_shared` has taken the sides claimed three times.
        # So check the name: a boundary side has to *be* on the boundary. A cut
        # whose shared end was nudged by 0.01 leaves two orphaned sides sitting
        # 3.0 away from the nearest wall, and without this the report still said
        # conformal.
        pts, loop_id, offsets = self.region.sample()
        tol = 1e-6 * self.region.outer.loop.scale
        by_id = {b.id: b for b in self.blocks}
        stranded = []
        for bid, si in c["boundary"]:
            side = by_id[bid].sides[si]
            d = float(np.max(segment_distance(side, pts, loop_id, offsets)))
            if d > tol:
                stranded.append((bid, si, d))

        return {
            "n_interfaces": len(c["interfaces"]),
            "n_boundary_sides": len(c["boundary"]),
            "over_shared": c["over_shared"],
            "accounted": paired + len(c["boundary"]) == n_sides,
            "all_flipped": all(i.flipped for i in c["interfaces"]),
            "stranded": stranded,
            "conformal": (not c["over_shared"]
                          and not stranded
                          and paired + len(c["boundary"]) == n_sides
                          and all(i.flipped for i in c["interfaces"])),
        }

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def kinds(self) -> dict:
        out: dict = {}
        for b in self.blocks:
            out[b.kind] = out.get(b.kind, 0) + 1
        return out

    def region_area(self) -> float:
        """Area of the fluid: the outer loop less every body."""
        a = abs(self.region.outer.loop.area)
        for h in self.region.holes:
            a -= abs(h.loop.area)
        return a

    def covered_area(self) -> float:
        def ring_area(side_a, side_b):
            def poly(p):
                x, y = p[:, 0], p[:, 1]
                return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            return abs(abs(poly(side_a)) - abs(poly(side_b)))

        total = 0.0
        for b in self.blocks:
            total += ring_area(*b.sides[:2]) if b.kind == "ring" else b.area
        return total

    def validate(self, area_tol: float = 1e-3, join_tol: float = 1e-9) -> dict:
        """Check that the blocks really are a decomposition of the region.

        An area sum alone is a weak test: blocks could overlap in one place and
        leave a gap of the same size elsewhere and it would still pass, and a
        zero-area spike -- an outline vertex flung far from the region -- changes
        no area at all. So four things are checked, from cheapest to strictest:

        * **joins** -- consecutive sides of a block must share an endpoint
          exactly, or the outline has a hole in it;
        * **simplicity** -- an outline must not cross itself;
        * **containment** -- no outline vertex may lie outside the region's
          bounding box, which is what catches a spike;
        * **tiling** -- with shapely, the union of the blocks must equal the
          fluid region: overlap, gap and spill are each measured separately
          rather than inferred from one area total.
        """
        want = self.region_area()
        got = self.covered_area()
        rel = abs(got - want) / want if want else float("inf")

        worst_join, open_blocks = 0.0, []
        for b in self.blocks:
            if b.kind == "ring" or b.n_sides < 2:
                continue
            for i in range(b.n_sides):
                g = float(np.linalg.norm(
                    b.sides[i][-1] - b.sides[(i + 1) % b.n_sides][0]))
                worst_join = max(worst_join, g)
                if g > join_tol * self.region.outer.loop.scale:
                    open_blocks.append(b.id)

        lo, hi = self.region.outer.loop.bbox
        pad = 1e-6 * self.region.outer.loop.scale
        stray = [b.id for b in self.blocks
                 if ((b.outline() < lo - pad).any() or (b.outline() > hi + pad).any())]

        degenerate = [b.id for b in self.blocks if b.area <= 0.0 or b.n_sides < 2]

        report = {
            "n_blocks": self.n_blocks,
            "region_area": want,
            "covered_area": got,
            "relative_error": rel,
            "worst_join": worst_join,
            "open_blocks": sorted(set(open_blocks)),
            "stray_blocks": stray,
            "degenerate": degenerate,
            "notes": list(self.notes),
        }
        report["tiles"] = (rel < area_tol and not degenerate
                           and not open_blocks and not stray)
        return report

    # ── the realised index certificate ──────────────────────────────

    def corner_angles(self) -> dict:
        """Interior angle at every logical block corner, in degrees.

        The sides are wound CCW, so at a corner the ray back along the incoming
        side and the ray forward along the outgoing side enclose the interior,
        and the signed angle from the first to the second is its negative.

        Returns ``{(block_id, corner_index): angle}``. A ring has none.
        """
        out = {}
        for b in self.blocks:
            if b.wraps:
                continue
            m = b.n_sides
            for i in range(m):
                a, c = b.sides[i], b.sides[(i + 1) % m]
                u, v = a[-2] - a[-1], c[1] - c[0]
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if nu == 0.0 or nv == 0.0:
                    continue
                ang = np.degrees(np.arctan2(float(u[0] * v[1] - u[1] * v[0]),
                                            float(np.dot(u, v))))
                out[(b.id, i)] = float((-ang) % 360.0)
        return out

    def corner_lines(self, tol: float | None = None) -> dict:
        """What the blocking actually does at each reflex boundary corner.

        A corner of fluid angle ``alpha`` carrying ``n_c`` elements wants
        ``n_c - 1`` grid lines leaving it, dividing ``alpha`` into equal
        sectors. This reports the lines that are there, as directions measured
        in the corner's own frame, so the second pass can ask for the ones that
        are missing instead of cutting blindly.

        Returns ``{global corner index: (alpha, n_c, [phi of each interior
        line])}``. A corner with no line at all is buried inside a side: it has
        one sector, of the full ``alpha``, and a kink in the middle of it.
        """
        from aeromesh.blocking.search import _corner_frame

        if tol is None:
            tol = self.coincidence_tol()
        _, _, offsets = self.region.sample()
        out = {}
        for li, b in enumerate(self.region.boundaries):
            for ci, ang, nc in zip(getattr(b.corners, "index", []),
                                   b.fluid_angles, b.corner_elements):
                alpha = float(ang)
                if alpha <= np.pi or int(nc) < 2:
                    continue
                pos = b.loop.points[int(ci)]
                a0, sense = _corner_frame(b.loop, int(ci), alpha)
                phis = []
                for blk in self.blocks:
                    if blk.wraps:
                        continue
                    for side in blk.sides:
                        for end, nxt in ((side[0], side[1]),
                                         (side[-1], side[-2])):
                            if np.linalg.norm(end - pos) > tol:
                                continue
                            w = nxt - pos
                            nw = float(np.linalg.norm(w))
                            if nw <= 0.0:
                                continue
                            w = w / nw
                            phi = (sense * float(np.arctan2(
                                a0[0] * w[1] - a0[1] * w[0],
                                np.dot(a0, w)))) % (2 * np.pi)
                            # Drop the two boundary tangents: they bound the
                            # wedge, they are not lines through it.
                            edge = 0.05 * alpha
                            if edge < phi < alpha - edge:
                                phis.append(phi)
                uniq = []
                for phi in sorted(phis):
                    if not uniq or abs(phi - uniq[-1]) > 1e-6 * max(alpha, 1.0):
                        uniq.append(phi)
                out[int(offsets[li]) + int(ci)] = (alpha, int(nc), uniq)
        return out

    def unresolved_corners(self, tol: float | None = None) -> dict:
        """Reflex corners short of the grid lines their element count wants.

        Returns ``{global corner index: [phi of each missing cross direction]}``,
        empty when every corner is resolved. A corner is short either because it
        is buried in a side -- no line at all -- or because the surrounding
        structure happened to supply fewer than ``n_c - 1``.
        """
        need = {}
        for gi, (alpha, nc, have) in self.corner_lines(tol).items():
            span = alpha / nc
            missing = [alpha * i / nc for i in range(1, nc)
                       if not any(abs(alpha * i / nc - h) < 0.5 * span
                                  for h in have)]
            if missing:
                need[gi] = missing
        return need

    def buried_corners(self, tol: float | None = None) -> list:
        """C0 boundary corners that lie strictly inside a block side.

        A kink in a curve the mesher will treat as smooth. Transfinite
        interpolation maps a straight logical edge onto that side, so the cell
        at the kink is skewed by the whole corner angle -- and on an aerofoil
        the kink is the trailing edge.
        """
        if tol is None:
            tol = self.coincidence_tol()
        pts, _, offsets = self.region.sample()
        ids, pos = [], []
        for li, b in enumerate(self.region.boundaries):
            for ci in getattr(b.corners, "index", []):
                ids.append(int(offsets[li]) + int(ci))
                pos.append(b.loop.points[int(ci)])
        if not ids:
            return []
        pos = np.asarray(pos, dtype=float)
        buried = set()
        for b in self.blocks:
            for side in b.sides:
                if len(side) < 3:
                    continue
                inner = side[1:-1]
                d = np.linalg.norm(inner[:, None, :] - pos[None, :, :], axis=2)
                for j in np.flatnonzero(d.min(axis=0) <= tol):
                    buried.add(ids[int(j)])
        return sorted(buried)

    def canonicalise(self, flat_tol: float = 12.0):
        """Merge sides that meet at a corner which is not a corner.

        ``n_sides`` is a count of the *representation*, not the logical degree
        of the polygon, and the two differ. A vertex region's two boundary arcs
        meet at the medial vertex's shared touch at exactly 180 degrees, on a
        smooth stretch of wall, with no other block having a corner there. One
        aerofoil-in-a-box blocking carries eight such joins out of twenty-three
        block corners, and its blocks report sides of ``[8, 8, 8, 7]`` where
        their logical degrees are ``[6, 6, 6, 5]``.

        It matters because everything downstream reads ``n_sides`` as the degree.
        Midpoint subdivision puts one node of valence ``m`` at the centre of an
        m-gon, so a spurious side means a spurious unit of index, and it puts a
        180-degree one-quad cell on the wall at the join. The index certificate
        charges ``m - 4`` per block and so carries the same error; the two
        cancel, which is why the budget still balanced while being wrong twice.

        A join is merged when its interior angle is within ``flat_tol`` of 180
        degrees and the point is not a detected C0 corner of the boundary. Both
        conditions are read from the geometry, so two blocks sharing the point
        reach the same verdict and the result stays conformal -- and a real
        corner of the wall is never merged away, which is the defect
        :meth:`buried_corners` exists to catch.
        """
        from aeromesh.medial.fields import segment_distance

        pts, loop_id_a, offsets = self.region.sample()
        corner_pos = []
        for li, b in enumerate(self.region.boundaries):
            for ci in getattr(b.corners, "index", []):
                corner_pos.append(b.loop.points[int(ci)])
        corner_pos = (np.asarray(corner_pos, dtype=float) if corner_pos
                      else np.empty((0, 2)))
        # The corner guard is measured against the boundary's own **sampling**,
        # not against `coincidence_tol`. That tolerance is 1e-9 of the scale --
        # it exists for points that should be bit-identical -- and a detected
        # corner position can sit a fraction of a sample from the point the
        # blocking uses. Guarding at 1e-9 let a genuine 120 degree wall corner
        # on a triangular body be merged away, which cost that blocking its
        # conformality.
        tol_at = local_spacing(self.region)
        scale = self.region.outer.loop.scale

        blocks, merged = [], 0
        for b in self.blocks:
            if b.wraps or b.n_sides <= 4:
                blocks.append(b)
                continue
            sides = [np.asarray(x, dtype=float) for x in b.sides]
            # Only two *boundary arcs* may merge. That is the whole shape of the
            # defect -- two arcs of one vertex region meeting at the medial
            # vertex's shared touch -- and anything wider is unsafe: a cut's
            # foot at a reflex corner can leave almost tangentially, so a
            # cut-into-wall join can read as flat (179.6 degrees measured) and
            # merging it destroys a real block corner.
            on_bnd = [bool(np.max(segment_distance(x, pts, loop_id_a, offsets))
                           <= 1e-7 * scale) for x in sides]
            changed = True
            while changed and len(sides) > 4:
                changed = False
                m = len(sides)
                for i in range(m):
                    j = (i + 1) % m
                    a, c = sides[i], sides[j]
                    p = c[0]
                    if not (on_bnd[i] and on_bnd[j]):
                        continue                 # not two boundary arcs
                    lim = tol_at(p)
                    if len(corner_pos) and np.min(
                            np.linalg.norm(corner_pos - p, axis=1)) <= max(
                                lim, 1e-9 * scale):
                        continue                 # a real corner of the wall
                    u, v = a[-2] - a[-1], c[1] - c[0]
                    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                    if nu == 0.0 or nv == 0.0:
                        continue
                    ang = (-np.degrees(np.arctan2(
                        float(u[0] * v[1] - u[1] * v[0]),
                        float(np.dot(u, v))))) % 360.0
                    if abs(ang - 180.0) > flat_tol:
                        continue
                    sides[i] = np.vstack([a, c[1:]])
                    sides.pop(j)
                    on_bnd.pop(j)
                    merged += 1
                    changed = True
                    break
            blocks.append(Block(id=b.id, sides=tuple(sides), kind=b.kind,
                                source=b.source, wraps=b.wraps))
        # No note: ``notes`` carries a specific contract -- something was left
        # unhandled -- and a completed merge is not that. The effect is visible
        # in ``n_sides`` and in ``quality()["m_gons"]``, and the before-and-after
        # is one call away.
        return BlockSystem(blocks=blocks, region=self.region, axis=self.axis,
                           notes=list(self.notes))

    def design_index(self, tol: float | None = None) -> dict:
        """The index budget of the mesh the blocking's *angles* ask for.

        A corner of interior angle ``theta`` wants ``n = round(theta / (pi/2))``
        elements -- Fogg's rule at a boundary corner, applied at a block corner.
        Charging every corner with its own ``n`` gives a second, independent
        reading of the same identity:

            block term     sum over blocks of [ sum_corners (2 - n_i) - 4 ]
            interior term  sum over interior nodes of ( sum n_i - 4 )
            boundary term  sum over boundary nodes of ( 2 - sum n_i )

        and ``block + interior == boundary - 4 chi`` identically, for any
        consistent choice of the ``n_i``. All three terms have to use the design
        counts. The first version of this method took the block term from the
        angles and the required term from ``index_certificate``, whose boundary
        term counts one element per corner -- so it was differencing two
        different models and its "residual" was really
        ``-sum_corners (n_i - 1)``: a quality number wearing an identity's
        clothes. It also dropped the interior term, which is not optional,
        because ``round`` is not additive: three corners of 100, 130 and 130
        degrees meeting at an interior node ask for 1 + 1 + 1 = 3 elements, not
        the 4 that 360 degrees would.

        What the comparison with
        :meth:`index_certificate` is *for* is the gap between the two models:
        midpoint subdivision puts one element at every corner, so wherever a
        corner asks for more than one the delivered mesh carries index the
        design would have placed elsewhere. On this test matrix that gap is
        entirely the unresolved reflex corners of a blunt trailing edge.
        """
        if tol is None:
            tol = self.coincidence_tol()
        ang = self.corner_angles()

        def elements(theta: float) -> int:
            return max(1, int(np.floor(theta / 90.0 + 0.5)))

        per_block: dict = {}
        for b in self.blocks:
            if not b.wraps:
                per_block.setdefault(b.id, 0)
        at_point: dict = {}
        for (bid, i), a in ang.items():
            n = elements(a)
            per_block[bid] = per_block.get(bid, 0) + (2 - n)
            blk = next(x for x in self.blocks if x.id == bid)
            p = blk.sides[(i + 1) % blk.n_sides][0]
            key = tuple(np.round(p / tol).astype(np.int64))
            at_point[key] = at_point.get(key, 0) + n

        corners = self.corner_valences(tol)
        k_blocks = sum(v - 4 for v in per_block.values())
        k_interior = sum(at_point.get(key, 0) - 4
                         for key, (_, _, on_b) in corners.items() if not on_b)
        b_term = sum(2 - at_point.get(key, 0)
                     for key, (_, _, on_b) in corners.items() if on_b)
        chi = self.region.euler_characteristic
        realised = self.index_certificate(tol)
        return {
            "k_blocks": int(k_blocks),
            "k_interior": int(k_interior),
            "k_design": int(k_blocks + k_interior),
            "boundary_term": int(b_term),
            "chi": int(chi),
            "k_required": int(b_term - 4 * chi),
            "residual": int(k_blocks + k_interior - (b_term - 4 * chi)),
            "ok": int(k_blocks + k_interior) == int(b_term - 4 * chi),
            "k_realised": int(realised["k_realised"]),
            "gap_to_realised": int(realised["k_realised"]
                                   - (k_blocks + k_interior)),
        }

    def is_sound(self) -> bool:
        """Does this blocking pass the checks a mesh can be built on?

        All four: it tiles by area, no block outline crosses itself and the
        union is the fluid region, every side is either on the boundary or
        shared with exactly one other block, and the index budget balances.

        ``validate_strict`` is included despite the cost. It was left out at
        first on the grounds that the certificate resolves cracks two orders
        finer, which is true but not sufficient: on a triangular and on a
        square body in a circular far field, corner resolution produced a
        blocking that balanced its index, was conformal and tiled by area, and
        still had an overlap shapely could see. The four checks are independent
        and none subsumes another.
        """
        try:
            strict = self.validate_strict()["ok"]
        except Exception:
            return False                  # invalid geometry is not sound
        return bool(strict
                    and self.index_certificate()["ok"]
                    and self.interface_report()["conformal"]
                    and self.validate()["tiles"])

    def quality(self) -> dict:
        """Mesh-quality measures of the blocking, before any mesh exists.

        None of these can fail the tiling checks, and all of them decide whether
        the mesh will be usable: a block corner near zero degrees is a sliver
        cell, one past 180 means the block is not logically convex so the m-gon
        template and transfinite interpolation both misbehave, and a buried
        corner is a kink inside a side.
        """
        ang = np.array(list(self.corner_angles().values()))
        sides = [float(np.sum(np.linalg.norm(np.diff(sd, axis=0), axis=1)))
                 for b in self.blocks for sd in b.sides]
        scale = self.region.outer.loop.scale
        return {
            "min_angle": float(np.min(ang)) if len(ang) else float("nan"),
            "max_angle": float(np.max(ang)) if len(ang) else float("nan"),
            "n_sliver": int(np.sum(ang < 20.0)) if len(ang) else 0,
            "n_reflex": int(np.sum(ang > 181.0)) if len(ang) else 0,
            "min_side": min(sides) / scale if sides else float("nan"),
            "buried_corners": self.buried_corners(),
            "unresolved_corners": {k: len(v)
                                   for k, v in self.unresolved_corners().items()},
            "m_gons": sorted({b.n_sides for b in self.blocks if not b.wraps}),
            "design_gap": self.design_index()["gap_to_realised"],
        }

    def coincidence_tol(self) -> float:
        """Below what separation two points are the same point.

        The blocking's own resolution, not machine epsilon: a cut ends at an
        exact perpendicular foot but a boundary arc is traced through boundary
        *samples*, so two constructions of one point can differ by a fraction of
        a sample spacing. A combinatorial count must not be more discriminating
        than the geometry it counts, or it reads representation noise as
        topology. Same rule as the ``snap_tol`` ``decompose`` uses.
        """
        return 1e-9 * self.region.outer.loop.scale

    def corner_valences(self, tol: float | None = None) -> dict:
        """Group the blocks' logical corners by position.

        Returns ``{point_key: (position, n_blocks, on_boundary)}``. ``n_blocks``
        is how many blocks have a logical corner there, which for a mesh built
        by transfinite interpolation (or the m-gon template) is exactly how many
        quads meet at that node.
        """
        if tol is None:
            tol = self.coincidence_tol()
        groups: dict = {}
        for b in self.blocks:
            if b.wraps:
                continue
            for i in range(b.n_sides):
                p = b.sides[i][0]
                key = tuple(np.round(p / tol).astype(np.int64))
                if key not in groups:
                    groups[key] = [p, 0]
                groups[key][1] += 1

        # Distance to the boundary *polyline*, not to its samples: every cut
        # ends at an exact perpendicular foot, which lies between two samples.
        # Measured to the nearest sample a foot sits half a spacing off the
        # wall, and every wall corner of every block is then misread as an
        # interior node -- which breaks the certificate by several units per
        # block while the blocking itself is correct.
        from aeromesh.medial.fields import segment_distance

        pts, loop_id, offsets = self.region.sample()
        q = np.array([p for p, _ in groups.values()], dtype=float)
        if not len(q):
            return {}
        d = segment_distance(q, pts, loop_id, offsets)
        on = d <= 1e-7 * self.region.outer.loop.scale
        return {key: (p, n, bool(flag))
                for (key, (p, n)), flag in zip(groups.items(), on)}

    def hanging_nodes(self, tol: float | None = None) -> list:
        """Block corners that land in the interior of another block's side.

        The defect a hanging node is: one block treats the point as a corner,
        its neighbour runs straight past it. The mesh then has a node on one
        side of the interface with nothing to match on the other, and no area
        test can see it -- the two blocks still tile, to within the sliver
        between the corner and the side it should have landed on.

        It is reported rather than repaired because the repair is not local: the
        point either belongs in the neighbour's side, which has to be split, or
        it is a duplicate of a corner a fraction of a sample spacing away and
        the two should be one. Which of those it is cannot be read off the
        lengths involved -- a side 0.02 long on a far field sampled every 0.055
        is real, and one 0.002 long on a body sampled every 0.002 is not.

        Returns a list of ``(position, corner_blocks, side_block, side_index,
        distance)``.
        """
        if tol is None:
            tol = self.coincidence_tol()
        corners = self.corner_valences(tol)
        out = []
        for key, (p, n, _on_b) in corners.items():
            for b in self.blocks:
                if b.wraps:
                    continue
                for i, side in enumerate(b.sides):
                    if (np.linalg.norm(side[0] - p) <= tol
                            or np.linalg.norm(side[-1] - p) <= tol):
                        continue            # it is an endpoint here, not hanging
                    seg = side[:-1]
                    d = side[1:] - seg
                    L2 = np.einsum("ij,ij->i", d, d)
                    t = np.clip(np.einsum("ij,ij->i", p - seg, d)
                                / np.maximum(L2, 1e-300), 0.0, 1.0)
                    dist = float(np.min(np.linalg.norm(seg + t[:, None] * d - p,
                                                       axis=1)))
                    if dist <= tol:
                        out.append((p, n, b.id, i, dist))
        return out

    def index_certificate(self, tol: float | None = None) -> dict:
        """The index budget of the blocking *as built*.

        The discrete Gauss-Bonnet identity for a quadrilateral mesh,

            sum_interior (valence - 4) = sum_boundary (2 - q) - 4 * chi,

        holds for every valid quad mesh -- it is Euler's formula rearranged, and
        it is verified in the tests on an explicitly constructed mesh. So it is
        not a prediction to be checked against the geometry; it is a consistency
        test of *this* blocking's own bookkeeping, and the thing it catches is a
        hanging node: a block corner that lands in the middle of a neighbour's
        side contributes to one side of the identity and not the other.

        The mesh's irregular nodes are read straight off the blocks:

        * a logical m-gon block meshed by midpoint subdivision puts one node of
          valence ``m`` at its centre, so ``k = m - 4``;
        * a point where ``q`` blocks meet has ``q`` quads around it, so an
          interior one has ``k = q - 4`` and a boundary one contributes
          ``2 - q``;
        * a ring block is an O-grid and contributes nothing.

        This is the certificate the mesher has to satisfy, and unlike the
        singularity field it is computed from the product rather than from local
        estimates of ``theta_m``.
        """
        corners = self.corner_valences(tol)
        k_blocks = sum(b.n_sides - 4 for b in self.blocks
                       if not b.wraps and b.n_sides != 4)
        k_interior = sum(n - 4 for _, n, on_b in corners.values() if not on_b)
        b_term = sum(2 - n for _, n, on_b in corners.values() if on_b)
        chi = self.region.euler_characteristic
        realised = k_blocks + k_interior
        required = b_term - 4 * chi
        return {
            "k_blocks": int(k_blocks),
            "k_interior_corners": int(k_interior),
            "k_realised": int(realised),
            "boundary_term": int(b_term),
            "chi": int(chi),
            "k_required": int(required),
            "residual": int(realised - required),
            "n_corner_points": len(corners),
            "hanging": self.hanging_nodes(tol),
            "ok": realised == required,
        }

    def validate_strict(self) -> dict:
        """The shapely checks: self-intersection, overlap, gap and spill.

        Kept separate from :meth:`validate` because it costs a polygon
        intersection per block pair, but it is the test that actually settles
        whether the blocks tile.
        """
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union

        polys, non_simple = [], []
        for b in self.blocks:
            if b.kind == "ring":
                poly = Polygon(b.sides[0], [b.sides[1]])
            else:
                o = b.outline()
                poly = Polygon(o)
                if not LineString(np.vstack([o, o[:1]])).is_simple:
                    non_simple.append(b.id)
            # Repair before measuring. A self-touching outline makes an invalid
            # polygon, and GEOS raises rather than returning an answer -- so the
            # check that exists to *measure* the defect was the one thing that
            # could not survive it. `buffer(0)` resolves the ring into valid
            # parts and the overlap, gap and spill are then real numbers again.
            # The block is still named in `non_simple`, which is what fails the
            # check.
            if not poly.is_valid:
                try:
                    poly = poly.buffer(0)
                except Exception:          # pragma: no cover
                    poly = Polygon()
            polys.append(poly)

        fluid = Polygon(self.region.outer.loop.closed_points,
                        [h.loop.closed_points for h in self.region.holes])
        try:
            union = unary_union(polys)
        except Exception:                  # pragma: no cover
            union = unary_union([q for q in polys if q.is_valid])
        area = fluid.area
        out = {
            "non_simple": non_simple,
            "overlap_fraction": (sum(p.area for p in polys) - union.area) / area,
            "gap_fraction": fluid.difference(union).area / area,
            "spill_fraction": union.difference(fluid).area / area,
        }
        out["ok"] = (not non_simple
                     and abs(out["overlap_fraction"]) < 1e-6
                     and out["gap_fraction"] < 1e-6
                     and out["spill_fraction"] < 1e-6)
        return out

    def __repr__(self) -> str:
        v = self.validate()
        state = "tiles" if v["tiles"] else f"area error {v['relative_error']:.2e}"
        return f"BlockSystem({self.n_blocks} blocks, {self.kinds()}, {state})"


# ═══════════════════════════════════════════════════════════════════
#  Decomposition
# ═══════════════════════════════════════════════════════════════════

def roll_loop_edge(d: dict, start: int) -> dict:
    """Re-origin a closed medial edge so sample ``start`` becomes sample 0.

    A LOOP edge's polyline is closed -- its last point repeats its first -- so
    cutting it is a matter of choosing where it begins. Rolling to the first cut
    turns the wrap-around span into an ordinary one from the last cut to the end
    of the array, and every builder downstream works unchanged on index ranges.

    Arc length is recomputed from the new origin; everything else is a rotation.
    """
    n = len(d["polyline"]) - 1          # the closing duplicate
    k = int(start) % n
    out = dict(d)
    for name in ("polyline", "r_m", "theta_m", "n1", "n2", "t1", "t2", "i1", "i2"):
        arr = np.asarray(d[name])
        body = np.roll(arr[:n], -k, axis=0)
        out[name] = np.concatenate([body, body[:1]], axis=0)
    seg = np.linalg.norm(np.diff(out["polyline"], axis=0), axis=1)
    out["s"] = np.concatenate(([0.0], np.cumsum(seg)))
    out["length"] = float(out["s"][-1])
    return out


def _span_block(bid, edge_data, edge_key, a, b, pts, offsets, tol_at=None):
    """The four-sided block between samples ``a`` and ``b`` of one medial edge."""
    cut_a = radius_split(edge_data, a, edge_key)
    cut_b = radius_split(edge_data, b, edge_key)
    # Each arc starts and ends exactly where the cuts do, so the four sides
    # chain without a gap.
    arc2 = boundary_arc(edge_data["i2"][a:b + 1], pts, offsets,
                        start=cut_a.curve[-1], end=cut_b.curve[-1])
    arc1 = boundary_arc(edge_data["i1"][b:a - 1 if a else None:-1], pts, offsets,
                        start=cut_b.curve[0], end=cut_a.curve[0])
    return Block(id=bid,
                 sides=orient_ccw(clean_sides((cut_a.curve, arc2,
                                               cut_b.curve[::-1], arc1),
                                              tol_at=tol_at,
                                              pinned=(True, False, True, False))),
                 kind="span", source=edge_key)


def _is_simple(block) -> bool:
    """Does the block's outline avoid crossing itself?"""
    from shapely.geometry import LineString
    try:
        o = block.outline()
        return bool(LineString(np.vstack([o, o[:1]])).is_simple
                    and block.area > 0.0)
    except Exception:                     # pragma: no cover
        return False


def _vertex_block(bid, axis, node, inner, pts, offsets, snap_tol=0.0,
                  tol_at=None):
    """The m-sided region around a medial vertex.

    Bounded by the innermost cut on each incident edge and, between them, the
    boundary arcs running from those cuts back in to the vertex's own touches.

    Those arcs are traced from the edges' touch arrays, the same way a span
    block's are -- not by walking the boundary and taking whichever run to the
    next cut end is shorter. The shorter run is only the right one while the
    region is small compared to the boundary, and once cuts are placed by the
    ranked search a vertex region can span half the domain.

    Incident edges are chained by the touch they share at the vertex: each edge
    arrives at one of the vertex's touches and leaves by another, and for a
    generic vertex that makes a single cycle.
    """
    legs = []
    for entry in inner:
        (u, v, key), cut = entry[0], entry[1]
        forced_start = entry[2] if len(entry) > 2 else None
        d = axis.graph.edges[u, v, key]
        if cut is None:
            # An uncut flare: the region runs all the way out to the corner,
            # which becomes one of its polygon vertices. The "cut" is that
            # single point, and clean_sides drops the degenerate side.
            at_start = (forced_start if forced_start is not None else
                        (np.linalg.norm(d["polyline"][0]
                                        - axis.graph.nodes[node]["pos"])
                         <= np.linalg.norm(d["polyline"][-1]
                                           - axis.graph.nodes[node]["pos"])))
            tip = d["polyline"][-1] if at_start else d["polyline"][0]
            i1 = d["i1"] if at_start else d["i1"][::-1]
            i2 = d["i2"] if at_start else d["i2"][::-1]
            # Both arcs run from their vertex touch to the corner by the
            # **short boundary path**, not by following the touch array.
            #
            # The touch array cannot be followed here. Near a flare's corner
            # r_m goes to zero and the exact feet jitter, so the array is not
            # monotonic -- measured on one body it ran 2993 -> 2949 -> 2954,
            # covering five samples twice. The region's outline then touches
            # itself, and invisibly: the overshoot and the return cancel, so the
            # area is right and only a simplicity test sees it.
            #
            # Measured honestly, this is a real defect and a rare one: it moved
            # one case of 35 in a randomised sweep. The self-touching vertex
            # regions that dominate that sweep have a different cause.
            #
            # The corner lies between the two touches, so the two short paths to
            # it partition that stretch of wall by construction -- the property
            # the arcs must have and the touch array does not. A path that is a
            # single sample (the touch already *is* the corner) makes a
            # one-point side, which `clean_sides` drops; its two neighbours
            # already meet there, so the chain closes.
            corner = int(np.argmin(np.linalg.norm(pts - tip, axis=1)))
            loop_i = _loop_of(int(i1[0]), offsets)
            loop_start = int(offsets[loop_i])
            n_loop = int(offsets[loop_i + 1] - offsets[loop_i])
            legs.append({"cut": None, "tip": pts[corner],
                         "arc1": _short_path(int(i1[0]), corner,
                                             loop_start, n_loop),
                         "arc2": _short_path(int(i2[0]), corner,
                                             loop_start, n_loop),
                         "t1": int(i1[0]), "t2": int(i2[0])})
            continue
        at_start = (forced_start if forced_start is not None else
                    (np.linalg.norm(d["polyline"][0]
                                    - axis.graph.nodes[node]["pos"])
                     <= np.linalg.norm(d["polyline"][-1]
                                       - axis.graph.nodes[node]["pos"])))
        i1 = d["i1"] if at_start else d["i1"][::-1]
        i2 = d["i2"] if at_start else d["i2"][::-1]
        c = cut.index if at_start else len(d["polyline"]) - 1 - cut.index
        c = int(np.clip(c, 1, len(i1) - 1))
        legs.append({
            "cut": cut, "tip": None,
            "arc1": i1[:c + 1], "arc2": i2[:c + 1],
            "t1": int(i1[0]), "t2": int(i2[0]),
        })

    # Chain the legs by the vertex touch they share.
    by_touch: dict = {}
    for j, leg in enumerate(legs):
        by_touch.setdefault(leg["t1"], []).append((j, 1))
        by_touch.setdefault(leg["t2"], []).append((j, 2))

    def assemble(order):
        sides, pinned = [], []
        for j, side_in, side_out in order:
            leg = legs[j]
            arc_in = leg["arc1"] if side_in == 1 else leg["arc2"]
            arc_out = leg["arc1"] if side_out == 1 else leg["arc2"]
            if leg["cut"] is None:
                end_in = end_out = leg["tip"]
                curve = np.vstack([leg["tip"], leg["tip"]])
            else:
                end_in = (leg["cut"].curve[0] if side_in == 1
                          else leg["cut"].curve[-1])
                end_out = (leg["cut"].curve[0] if side_out == 1
                           else leg["cut"].curve[-1])
                curve = (leg["cut"].curve if side_in == 1
                         else leg["cut"].curve[::-1])
            sides.append(boundary_arc(arc_in, pts, offsets, end=end_in))
            sides.append(curve)
            sides.append(boundary_arc(arc_out[::-1], pts, offsets,
                                      start=end_out))
            # The cut is analytic and shared with the neighbouring region; the
            # two arcs are traced. Only an uncut flare's "cut" is a boundary
            # sample, and that one is a point, so it drops out either way.
            pinned.extend((False, leg["cut"] is not None, False))
        return orient_ccw(clean_sides(sides, snap_tol=snap_tol, tol_at=tol_at,
                                      pinned=pinned))

    # Every way the legs chain, not the first one that presents itself. A touch
    # shared by more than two legs leaves a genuine choice, and taking
    # `nxt[0]` is a coin flip: on a three-body C far field it walked one
    # boundary arc twice -- the region came out with eight sides for a
    # degree-three vertex, two of them the same curve reversed, and its outline
    # pinched against itself. That is an invalid block, and it passed every
    # check except the shapely one.
    orders: list = []

    def walk(j, side_in, used, order):
        if len(orders) >= 24:
            return
        used = used | {j}
        side_out = 2 if side_in == 1 else 1
        order = order + [(j, side_in, side_out)]
        if len(order) == len(legs):
            orders.append(order)
            return
        t_out = legs[j]["t2"] if side_out == 2 else legs[j]["t1"]
        for k, sd in by_touch.get(t_out, []):
            if k not in used:
                walk(k, sd, used, order)

    for first in (1, 2):
        walk(0, first, frozenset(), [])
    if not orders:
        return None                      # not a single cycle; caller reports it

    from shapely.geometry import LineString

    fallback = None
    for order in orders:
        sides = assemble(order)
        if not sides:
            continue
        blk = Block(id=bid, sides=sides, kind="vertex", source=(node,))
        if fallback is None:
            fallback = blk
        if _is_simple(blk):
            return blk
    # No chaining gives a simple region. The first one is still returned, and
    # the caller notes it: dropping the block instead is worse, not better. On a
    # randomised sweep the rejection fired on 45 per cent of configurations and
    # each time left a hole of 3 to 47 per cent of the region area, against a
    # self-touching outline that costs a few parts per billion. A region the
    # tiling check can measure beats a region that is not there.
    return fallback


def _corner_block(bid, node, data, cut, pts, offsets, tol_at=None):
    """The wedge between a flare's innermost cut and the corner it ends at.

    A flare terminates at a convex corner of the fluid with r_m = 0, so there is
    no annular span left over -- just a three-sided sliver bounded by the cut and
    the two boundary arcs running into the corner. Without it the decomposition
    is short by one wedge per corner, which is exactly the area a tiling check
    catches.
    """
    corner = int(np.argmin(np.linalg.norm(pts - data["pos"], axis=1)))
    loop = _loop_of(corner, offsets)
    start, n = int(offsets[loop]), int(offsets[loop + 1] - offsets[loop])
    arc_b = boundary_arc(_short_path(int(cut.touch_index[1]), corner, start, n),
                         pts, offsets, start=cut.curve[-1])
    arc_a = boundary_arc(_short_path(corner, int(cut.touch_index[0]), start, n),
                         pts, offsets, end=cut.curve[0])
    return Block(id=bid,
                 sides=orient_ccw(clean_sides((cut.curve, arc_b, arc_a),
                                              tol_at=tol_at,
                                              pinned=(True, False, False))),
                 kind="corner", source=(node,))


def decompose(axis, region, splits_per_edge: int = 1,
              vertex_margin: float = 0.12, splits: dict | None = None,
              field=None, resolve_corners: bool = True,
              forced: dict | None = None, canonical: bool = True
              ) -> BlockSystem:
    """Build the blocks of a region from its exterior medial axis.

    Parameters
    ----------
    axis : MedialAxis
    region : Region
    splits_per_edge : int
        Interior cuts per medial edge, used only when neither ``splits`` nor
        ``field`` is given. Cuts are then evenly spaced, which tiles but makes no
        attempt to isolate anything.
    vertex_margin : float
        Where those evenly spaced cuts sit, as a fraction of the edge's length
        from each end.
    splits : dict, optional
        Explicit cut sample indices per medial edge, as returned by
        :func:`aeromesh.blocking.search.select_splits`.
    field : SingularityField, optional
        When given and ``splits`` is not, the cuts are chosen by the ranked
        search so that each block holds at most one singularity.
    resolve_corners : bool
        Run a second pass that cuts to any C0 boundary corner left buried inside
        a block side by the first. Only the corners that need it are cut, which
        is why it takes two passes: whether a corner ends up on a block corner
        of its own accord depends on the whole blocking, not on anything local
        to the corner. Most do -- a flare already runs out to each trailing-edge
        corner on a box or C far field -- and cutting the rest anyway turns 90
        degree block angles into 3 degree ones.
    forced : dict, optional
        Cuts that must be made regardless, ``{edge_key: [sample index, ...]}``.
    canonical : bool
        Merge side joins that are flat and not boundary corners, so a block's
        ``n_sides`` is its logical degree. See
        :meth:`BlockSystem.canonicalise`.

    Returns
    -------
    BlockSystem
    """
    if splits is None and field is not None:
        from aeromesh.blocking.search import select_splits
        splits = select_splits(axis, region, field, forced=forced)

    pts, _, offsets = region.sample()
    # Cut positions are kept local, not stashed on the medial graph: decompose
    # must not mutate its input, or a second call sees the first call's cuts.
    edge_cuts: dict = {}
    # Two arcs meeting at a vertex's shared touch can land on adjacent samples;
    # anything below a couple of spacings is the same point, not a gap.
    tol_at = local_spacing(region)
    snap_tol = 2.0 * max(b.loop.perimeter / b.loop.n_points
                         for b in region.boundaries)
    blocks, notes, bid = [], [], 0
    # The cuts actually used, per edge. The second pass starts from these rather
    # than from `splits`, so it is a strict refinement: without it an edge that
    # gains a corner cut loses whatever the first pass had put on it, which for
    # the even-spacing mode means losing the only cut it had.
    used_cuts: dict = {}

    for u, v, key in axis.graph.edges(keys=True):
        d = axis.graph.edges[u, v, key]
        n = len(d["polyline"])
        ek = (u, v, key)

        if d["kind"].name == "LOOP":
            cuts = (np.unique(np.asarray(splits[ek], dtype=int))
                    if splits is not None and ek in splits else np.empty(0, int))
            used_cuts[ek] = cuts
            # One cut is enough. A closed edge cut once gives a single block
            # whose wall arc runs the whole way round and whose two remaining
            # sides are the two faces of that one cut -- the classic C-type
            # arrangement. Treating it as "not enough" kept the ring and threw
            # the cut away, which left a body with a single reflex corner buried
            # with nothing reported.
            if len(cuts) < 1:
                # Nothing demands a cut: the ring is the block. An O-grid,
                # arrived at rather than assumed.
                outer_arc = boundary_arc(d["i1"], pts, offsets)
                inner_arc = boundary_arc(d["i2"], pts, offsets)
                blocks.append(Block(id=bid, sides=(outer_arc, inner_arc),
                                    kind="ring", source=ek, wraps=True))
                bid += 1
                continue
            # Re-origin at the first cut so the spans close the ring without a
            # wrap-around case.
            dd = roll_loop_edge(d, int(cuts[0]))
            m = len(dd["polyline"]) - 1
            local = np.unique((cuts - int(cuts[0])) % m)
            local = np.append(local, m)          # the origin again, closing up
            for a, b in zip(local[:-1], local[1:]):
                if b <= a:
                    continue
                blocks.append(_span_block(bid, dd, ek, int(a), int(b), pts,
                                          offsets, tol_at=tol_at))
                bid += 1
            continue

        if splits is not None and ek in splits:
            if not len(splits[ek]):
                used_cuts[ek] = np.empty(0, dtype=int)
                continue          # deliberately uncut, e.g. a flare
            cuts = np.unique(np.clip(np.asarray(splits[ek], dtype=int), 1, n - 2))
        else:
            lo = max(1, int(round(vertex_margin * (n - 1))))
            hi = min(n - 2, int(round((1.0 - vertex_margin) * (n - 1))))
            if hi <= lo:
                lo, hi = 1, n - 2
            cuts = np.unique(np.linspace(lo, hi, splits_per_edge + 1).round().astype(int))
        used_cuts[ek] = cuts
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            blocks.append(_span_block(bid, d, ek, int(a), int(b), pts, offsets,
                                      tol_at=tol_at))
            bid += 1
        edge_cuts[_edge_key(u, v, key)] = (int(cuts[0]), int(cuts[-1]))

    for node, data in axis.graph.nodes(data=True):
        if data.get("anchor"):
            continue
        kind = data["kind"].name
        if kind not in ("NORMAL", "CORNER"):
            notes.append(f"vertex {node}: kind {kind} has no region builder yet")
            continue
        inner = []
        for uu, vv, kk, dd in axis.graph.edges(node, keys=True, data=True):
            if dd["kind"].name == "LOOP":
                continue
            # An edge that leaves this vertex and comes back to it contributes
            # **two** legs to the region, one per end. `networkx` reports a
            # self-loop once, so one leg per reported edge is one too few and
            # the region is built with the wrong degree.
            #
            # Measured: it corrects the degree -- a vertex that reported two
            # incident edges now reports its real three -- and it does *not* on
            # its own make that region simple. So it is kept because the leg
            # count is right, not because it fixes the self-touching outline,
            # whose cause is still open.
            ends = (True, False) if uu == vv else (None,)
            for forced_start in ends:
                if _edge_key(uu, vv, kk) not in edge_cuts:
                    inner.append(((uu, vv, kk), None, forced_start))
                    continue
                at_start = (forced_start if forced_start is not None else
                            (np.linalg.norm(dd["polyline"][0] - data["pos"])
                             <= np.linalg.norm(dd["polyline"][-1]
                                               - data["pos"])))
                lo_i, hi_i = edge_cuts[_edge_key(uu, vv, kk)]
                idx = lo_i if at_start else hi_i
                inner.append(((uu, vv, kk),
                              radius_split(dd, idx, (uu, vv, kk)), at_start))
        if kind == "CORNER":
            if len(inner) == 1 and inner[0][1] is None:
                continue          # uncut flare: the wedge is part of the vertex region
            if len(inner) != 1:
                notes.append(f"corner {node}: {len(inner)} cuts, expected 1")
                continue
            blocks.append(_corner_block(bid, node, data, inner[0][1], pts, offsets,
                                        tol_at=tol_at))
            bid += 1
            continue
        if len(inner) < 2:
            notes.append(f"vertex {node}: only {len(inner)} cuts, skipped")
            continue
        blk = _vertex_block(bid, axis, node, inner, pts, offsets,
                            snap_tol=snap_tol, tol_at=tol_at)
        if blk is None:
            notes.append(f"vertex {node}: its {len(inner)} incident edges do "
                         f"not chain into a cycle")
            continue
        if not _is_simple(blk):
            notes.append(f"vertex {node}: no chaining of its {len(inner)} "
                         f"incident edges gives a simple region; the region is "
                         f"emitted as built and the tiling checks will show it")
        blocks.append(blk)
        bid += 1

    system = BlockSystem(blocks=blocks, region=region, axis=axis, notes=notes)
    if canonical:
        canon = system.canonicalise()
        if canon.is_sound() or not system.is_sound():
            system = canon
        else:
            # Not silent. Everything downstream reads `n_sides` as the logical
            # degree, so a blocking kept in its non-canonical form reports
            # `m_gons` and `design_gap` on a representation with spurious sides,
            # and that has to be said.
            system.notes.append(
                "canonicalisation rejected: merging the flat side joins broke "
                "the decomposition, so the blocking keeps its spurious sides "
                "and n_sides overstates the logical degree")

    # `is not False` rather than truthiness: the budget counts down to 0, and 0
    # is falsy, so the last pass skipped the whole block -- including the note
    # that says the corners could not be resolved.
    if resolve_corners is not False:
        # Driven by burial, which is a correctness defect -- a kink inside a
        # side -- and not by `unresolved_corners`, which is a quality one.
        #
        # Resolving a corner's full sector count is the better-shaped answer and
        # it is measurably better shaped: it lifts the worst block angle on an
        # aerofoil in a box from 7.8 to 77.9 degrees. It is also not yet
        # consistent. The medial structure near a short blunt trailing-edge face
        # is dense, the extra cuts land their far feet within 5e-4 of existing
        # ones on the lower surface, and the two then count as two block corners
        # instead of one -- an index residual of -4, which the certificate
        # catches and no area test would. So the sector count is reported by
        # `quality` and left alone until the cuts can be placed without that
        # collision. What a corner is short of is in `unresolved_corners`.
        passes = 4 if resolve_corners is True else int(resolve_corners)
        buried = system.buried_corners()
        if buried:
            from aeromesh.blocking.search import corner_cuts
            extra = corner_cuts(axis, region, only=buried) if passes > 0 else {}
            if extra:
                merged = {k: np.asarray(v, dtype=int)
                          for k, v in used_cuts.items()}
                for k, v in extra.items():
                    merged[k] = np.unique(np.concatenate(
                        [merged.get(k, np.empty(0, dtype=int)),
                         np.asarray(v, dtype=int)]))
                refined = decompose(axis, region, splits_per_edge,
                                    vertex_margin, splits=merged, field=field,
                                    resolve_corners=passes - 1,
                                    canonical=canonical)
                # Resolving a corner is a refinement, and a refinement that
                # breaks the decomposition is not an improvement. On a polygonal
                # body in a box far field the extra cuts interact with the
                # flares and the result stops tiling -- certificate -4 to -8,
                # non-conformal -- while the unrefined blocking was sound. So
                # keep whichever is sound, and say what was given up.
                if refined.is_sound() or not system.is_sound():
                    return refined
                system.notes.append(
                    f"corner resolution rejected: cutting to corners "
                    f"{buried} broke the decomposition "
                    f"(certificate {refined.index_certificate()['residual']:+d}, "
                    f"conformal "
                    f"{refined.interface_report()['conformal']}), so the "
                    f"blocking without it is kept and those corners stay "
                    f"inside a block side")
                return system
            # Nothing more can be cut, so say so. One pass is not always enough
            # -- a regular pentagon came out of the first with five buried
            # corners and out of the second with two -- and a fixed point that
            # still has a kink in the middle of a wall side must not be reported
            # as success.
            system.notes.append(
                f"corners {buried} remain buried inside a block side after "
                f"corner resolution; the wall has a kink inside a block side "
                f"there")
    return system
