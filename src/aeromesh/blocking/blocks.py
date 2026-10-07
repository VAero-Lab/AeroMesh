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

from aeromesh.blocking.splits import Split, radius_split


# ═══════════════════════════════════════════════════════════════════
#  Boundary arcs
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  Blocks
# ═══════════════════════════════════════════════════════════════════

def clean_sides(sides, tol: float = 1e-12) -> tuple:
    """Drop sides that have collapsed to a point, and repeated vertices.

    Two cuts can terminate at exactly the same boundary point -- most often a
    sharp corner on a body, where the medial radii from either side both land on
    the corner. The arc between them is then empty, and an m-sided region is
    really (m-1)-sided. Keeping the empty side leaves a repeated vertex in the
    outline, which makes it non-simple for no geometric reason.
    """
    out = []
    for side in sides:
        side = np.asarray(side, dtype=float)
        if len(side) > 1:
            keep = np.concatenate(
                ([True], np.linalg.norm(np.diff(side, axis=0), axis=1) > tol))
            side = side[keep]
        if len(side) < 2:
            continue
        out.append(side)
    return tuple(out)


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


@dataclass
class BlockSystem:
    """The blocks of one region, with the checks that say they are a tiling."""

    blocks: list = field(default_factory=list)
    region: object = None
    axis: object = None
    notes: list = field(default_factory=list)

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
            polys.append(poly)

        fluid = Polygon(self.region.outer.loop.closed_points,
                        [h.loop.closed_points for h in self.region.holes])
        union = unary_union(polys)
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

def _span_block(bid, edge_data, edge_key, a, b, pts, offsets):
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
                 sides=clean_sides((cut_a.curve, arc2, cut_b.curve[::-1], arc1)),
                 kind="span", source=edge_key)


def _vertex_block(bid, axis, node, inner, pts, offsets):
    """The m-sided region around a medial vertex.

    Bounded by the innermost cut on each incident edge and the boundary arcs
    between consecutive cut ends. The arcs are found by walking the boundary
    both ways from each cut end and keeping the shorter run to the next one --
    the region is local, so the short way is the one that bounds it.
    """
    ends = []                       # (boundary index, point, cut id, which end)
    for cid, (key, cut) in enumerate(inner):
        ends.append((cut.touch_index[0], cut.curve[0], cid, 0))
        ends.append((cut.touch_index[1], cut.curve[-1], cid, 1))

    used, sides, cid, side_end = set(), [], 0, 1
    cuts = [c for _, c in inner]
    sides.append(cuts[0].curve)
    used.add((0, 0))
    used.add((0, 1))
    here = ends[1]

    for _ in range(len(cuts) * 2):
        loop = _loop_of(int(here[0]), offsets)
        start, n = int(offsets[loop]), int(offsets[loop + 1] - offsets[loop])
        best = None
        for e in ends:
            if (e[2], e[3]) in used or _loop_of(int(e[0]), offsets) != loop:
                continue
            path = _short_path(int(here[0]), int(e[0]), start, n)
            if best is None or len(path) < len(best[0]):
                best = (path, e)
        if best is None:
            break
        path, nxt = best
        nxt_curve = cuts[nxt[2]].curve
        sides.append(boundary_arc(path, pts, offsets, start=here[1],
                                  end=nxt_curve[0] if nxt[3] == 0 else nxt_curve[-1]))
        curve = cuts[nxt[2]].curve
        sides.append(curve if nxt[3] == 0 else curve[::-1])
        used.add((nxt[2], 0))
        used.add((nxt[2], 1))
        here = ends[2 * nxt[2] + (1 - nxt[3])]
        if len(used) >= 2 * len(cuts):
            loop = _loop_of(int(here[0]), offsets)
            start, n = int(offsets[loop]), int(offsets[loop + 1] - offsets[loop])
            sides.append(boundary_arc(
                _short_path(int(here[0]), int(ends[0][0]), start, n),
                pts, offsets, start=here[1], end=sides[0][0]))
            break

    return Block(id=bid, sides=clean_sides(sides), kind="vertex", source=(node,))


def _corner_block(bid, node, data, cut, pts, offsets):
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
    return Block(id=bid, sides=clean_sides((cut.curve, arc_b, arc_a)),
                 kind="corner", source=(node,))


def decompose(axis, region, splits_per_edge: int = 1,
              vertex_margin: float = 0.12) -> BlockSystem:
    """Build the blocks of a region from its exterior medial axis.

    Parameters
    ----------
    axis : MedialAxis
    region : Region
    splits_per_edge : int
        Interior cuts per medial edge, in addition to the two that bound the
        vertex regions at its ends.
    vertex_margin : float
        Where those bounding cuts sit, as a fraction of the edge's length from
        each end.

    Returns
    -------
    BlockSystem
    """
    pts, _, offsets = region.sample()
    blocks, notes, bid = [], [], 0

    for u, v, key in axis.graph.edges(keys=True):
        d = axis.graph.edges[u, v, key]
        n = len(d["polyline"])
        ek = (u, v, key)

        if d["kind"].name == "LOOP":
            outer_arc = boundary_arc(d["i1"], pts, offsets)
            inner_arc = boundary_arc(d["i2"], pts, offsets)
            blocks.append(Block(id=bid, sides=(outer_arc, inner_arc),
                                kind="ring", source=ek, wraps=True))
            bid += 1
            continue

        lo = max(1, int(round(vertex_margin * (n - 1))))
        hi = min(n - 2, int(round((1.0 - vertex_margin) * (n - 1))))
        if hi <= lo:
            lo, hi = 1, n - 2
        cuts = np.unique(np.linspace(lo, hi, splits_per_edge + 1).round().astype(int))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            blocks.append(_span_block(bid, d, ek, int(a), int(b), pts, offsets))
            bid += 1
        axis.graph.edges[u, v, key]["_cut_lo"] = int(cuts[0])
        axis.graph.edges[u, v, key]["_cut_hi"] = int(cuts[-1])

    for node, data in axis.graph.nodes(data=True):
        if data.get("anchor"):
            continue
        kind = data["kind"].name
        if kind not in ("NORMAL", "CORNER"):
            notes.append(f"vertex {node}: kind {kind} has no region builder yet")
            continue
        inner = []
        for uu, vv, kk, dd in axis.graph.edges(node, keys=True, data=True):
            if "_cut_lo" not in dd:
                continue
            at_start = (np.linalg.norm(dd["polyline"][0] - data["pos"])
                        <= np.linalg.norm(dd["polyline"][-1] - data["pos"]))
            idx = dd["_cut_lo"] if at_start else dd["_cut_hi"]
            inner.append(((uu, vv, kk), radius_split(dd, idx, (uu, vv, kk))))
        if kind == "CORNER":
            if len(inner) != 1:
                notes.append(f"corner {node}: {len(inner)} cuts, expected 1")
                continue
            blocks.append(_corner_block(bid, node, data, inner[0][1], pts, offsets))
            bid += 1
            continue
        if len(inner) < 2:
            notes.append(f"vertex {node}: only {len(inner)} cuts, skipped")
            continue
        blocks.append(_vertex_block(bid, axis, node, inner, pts, offsets))
        bid += 1

    return BlockSystem(blocks=blocks, region=region, axis=axis, notes=notes)
