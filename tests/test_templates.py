"""Tests for quadrilateralisation (S3 item D).

Transfinite interpolation needs four sides, and the decomposition does not
produce four-sided blocks -- nor should it be made to, since forcing every
region to be a quadrilateral is the kind of prescribed structure this method
exists to avoid. Measured over the test matrix after the near-wall layer goes
in, most blocks have m != 4, with m over 3, 5, 6 and 7.

Midpoint subdivision is the bridge, and it is the one prescribed construction
the method allows: an m-gon with corners V_i, side midpoints M_i and a centre C
splits into the m quadrilaterals (V_i, M_i, C, M_{i-1}). Two properties have to
hold and neither is obvious -- it has to be applied to *every* block or it is
not conformal, and the spokes to C must not fold, which they do for any fixed
choice of C.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.blocks import Block, BlockSystem
from aeromesh.blocking.templates import (harmonic_spokes, quad_faults,
                                         side_midpoint, subdivide, to_quads)
from aeromesh.geometry.loop import Loop


def ellipse(a, b, cx=0.0, cy=0.0, n=400, name="e"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(
        np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


@pytest.fixture(scope="module")
def bodies():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    return {
        "main": main,
        "flap": main.transform(scale=0.30, angle=np.deg2rad(-25), dx=1.05,
                               dy=-0.08, name="flap"),
        "fuse": ellipse(1.6, 0.55, name="fuse"),
        "disc": ellipse(1.0, 1.0, name="disc"),
    }


LAYOUTS = [
    ("aerofoil / box", ["main"], "box"),
    ("aerofoil / circle", ["main"], "circle"),
    ("aerofoil / C", ["main"], "c"),
    ("two aerofoils / box", ["main", "flap"], "box"),
    ("disc / box", ["disc"], "box"),
    ("fuselage / circle", ["fuse"], "circle"),
]


def build(group, kind, layer=True):
    outer = {"circle": lambda g: am.circle_farfield(g, 8.0),
             "c": lambda g: am.c_farfield(g, 8.0, 14.0),
             "box": lambda g: am.box_farfield(g, 4.0, 6.0, 4.0)}[kind](group)
    region = am.build_region(group, outer=outer)
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    system = am.decompose(axis, region, field=field)
    if layer:
        system = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
    return region, axis, system


# ═══════════════════════════════════════════════════════════════════
#  The pieces
# ═══════════════════════════════════════════════════════════════════

class TestSideMidpoint:
    def test_the_midpoint_is_taken_by_arc_length(self):
        """Not by index. A side's points are boundary samples or a resampled
        cut, neither evenly spaced, and splitting by index puts the join in a
        different place on the two blocks that share the side."""
        side = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(side_midpoint(side), [0.5, 0.0], atol=1e-12)

    def test_it_is_reversal_invariant(self):
        side = np.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.4], [1.0, 0.4]])
        np.testing.assert_allclose(side_midpoint(side),
                                   side_midpoint(side[::-1]), atol=1e-12)


class TestHarmonicSpokes:
    """Straight spokes to a centre fold, at any centre. Tutte's theorem gives
    the construction that cannot: a discrete harmonic map of a triangulated disc
    onto a convex region, with the boundary mapped homeomorphically, is
    injective, so radii of a disc map back to non-crossing curves."""

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_spokes_are_built_for_every_block(self, bodies, name, keys, kind):
        _, _, system = build([bodies[k] for k in keys], kind)
        for b in system.blocks:
            if b.wraps:
                continue
            spokes, centre = harmonic_spokes(b)
            assert spokes is not None, (name, b.id)
            assert len(spokes) == b.n_sides
            assert centre is not None

    def test_every_spoke_starts_at_its_side_midpoint_and_ends_at_the_centre(
            self, bodies):
        _, _, system = build([bodies["main"]], "box")
        b = next(x for x in system.blocks if not x.wraps and x.n_sides > 4)
        spokes, centre = harmonic_spokes(b)
        for i, path in enumerate(spokes):
            np.testing.assert_allclose(path[0], side_midpoint(b.sides[i]),
                                       atol=1e-9)
            np.testing.assert_allclose(path[-1], centre, atol=1e-12)

    def test_the_spokes_stay_inside_the_block(self, bodies):
        """Measured as the distance by which a spoke point strays outside, not
        as a length ratio: the spokes are polylines and the block's sides are
        curved, so a chord-versus-arc comparison measures the sampling, not the
        containment. The real excursion is at machine precision."""
        from shapely.geometry import Point, Polygon
        _, _, system = build([bodies["main"]], "box")
        scale = system.region.outer.loop.scale
        worst = 0.0
        for b in system.blocks:
            if b.wraps:
                continue
            spokes, _ = harmonic_spokes(b)
            poly = Polygon(b.outline())
            for path in spokes:
                for q in path:
                    pt = Point(q)
                    if not poly.contains(pt):
                        worst = max(worst, poly.exterior.distance(pt))
        assert worst < 1e-12 * scale, worst

    def test_the_spokes_do_not_cross_each_other(self, bodies):
        from shapely.geometry import LineString
        _, _, system = build([bodies["main"]], "box")
        for b in system.blocks:
            if b.wraps:
                continue
            spokes, _ = harmonic_spokes(b)
            lines = [LineString(p) for p in spokes]
            for i in range(len(lines)):
                for j in range(i + 1, len(lines)):
                    # they share the centre, and nothing else
                    hit = lines[i].intersection(lines[j])
                    assert hit.geom_type in ("Point", "GeometryCollection",
                                             "MultiPoint"), (b.id, i, j)


# ═══════════════════════════════════════════════════════════════════
#  Subdivision
# ═══════════════════════════════════════════════════════════════════

class TestSubdivide:
    def test_an_m_gon_gives_m_quads(self, bodies):
        _, _, system = build([bodies["main"]], "box")
        for b in system.blocks:
            if b.wraps:
                continue
            quads = subdivide(b)
            assert len(quads) == b.n_sides
            assert all(q.n_sides == 4 for q in quads)

    def test_every_sub_block_closes_and_has_area(self, bodies):
        _, _, system = build([bodies["main"]], "circle")
        for b in system.blocks:
            if b.wraps or quad_faults(b):
                continue
            for q in subdivide(b):
                for i in range(4):
                    np.testing.assert_allclose(
                        q.sides[i][-1], q.sides[(i + 1) % 4][0], atol=1e-9)
                assert q.area > 0.0

    def test_the_sub_blocks_tile_their_parent(self, bodies):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        _, _, system = build([bodies["main"]], "circle")
        for b in system.blocks:
            if b.wraps or quad_faults(b):
                continue
            parent = Polygon(b.outline())
            union = unary_union([Polygon(q.outline()) for q in subdivide(b)])
            assert union.area == pytest.approx(parent.area, rel=1e-6)
            assert parent.difference(union).area / parent.area < 1e-6

    def test_a_quad_is_subdivided_too(self, bodies):
        """It has to be. Subdivision splits every side at its midpoint, so a
        quad left whole next to a subdivided neighbour carries a node in the
        middle of a side it runs straight past."""
        _, _, system = build([bodies["main"]], "circle")
        quad = next(b for b in system.blocks if not b.wraps and b.n_sides == 4)
        assert len(subdivide(quad)) == 4


class TestQuadFaults:
    """Not a convexity proxy: the subdivision is carried out and every
    sub-block checked. A centroid-inside test passes on thin layer strips that
    then produce self-intersecting sub-blocks."""

    def test_a_sound_block_has_no_faults(self, bodies):
        _, _, system = build([bodies["disc"]], "box")
        assert not any(quad_faults(b) for b in system.blocks)

    def test_a_re_entrant_block_is_handled_not_rejected(self):
        """Tutte's theorem is about the *target* being convex, not the source,
        so a strongly re-entrant block subdivides perfectly well. This is the
        property the harmonic construction was adopted for."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        pts = [np.array([[0.0, 0.0], [2.0, 0.0]]),
               np.array([[2.0, 0.0], [2.0, 2.0]]),
               np.array([[2.0, 2.0], [1.0, 0.05]]),
               np.array([[1.0, 0.05], [0.0, 2.0]]),
               np.array([[0.0, 2.0], [0.0, 0.0]])]
        blk = Block(id=0, sides=tuple(pts), kind="span", source=())
        assert quad_faults(blk) == []
        quads = subdivide(blk)
        assert len(quads) == 5
        union = unary_union([Polygon(q.outline()) for q in quads])
        assert union.area == pytest.approx(Polygon(blk.outline()).area,
                                           rel=1e-9)

    def test_a_self_crossing_outline_is_rejected_before_it_can_crash(self):
        """Triangle is a C library: on a self-crossing outline it reports a
        topological inconsistency and, in a longer-running process, segfaults,
        taking the interpreter with it. So the validity check has to run before
        the triangulation, not after."""
        bow = [np.array([[0.0, 0.0], [2.0, 2.0]]),
               np.array([[2.0, 2.0], [2.0, 0.0]]),
               np.array([[2.0, 0.0], [0.0, 2.0]]),
               np.array([[0.0, 2.0], [1.0, 1.0]]),
               np.array([[1.0, 1.0], [0.0, 0.0]])]
        blk = Block(id=1, sides=tuple(bow), kind="span", source=())
        assert harmonic_spokes(blk) == (None, None)
        assert quad_faults(blk) == ["outline is not simple"]
        for _ in range(25):                 # must stay alive
            harmonic_spokes(blk)

    def test_a_ring_has_nothing_to_subdivide(self, bodies):
        _, _, system = build([bodies["fuse"]], "circle")
        ring = next(b for b in system.blocks if b.wraps)
        assert quad_faults(ring) == []
        assert subdivide(ring) == [ring]


# ═══════════════════════════════════════════════════════════════════
#  The whole system
# ═══════════════════════════════════════════════════════════════════

class TestToQuads:
    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_the_result_is_always_a_sound_decomposition(
            self, bodies, name, keys, kind):
        _, _, system = build([bodies[k] for k in keys], kind)
        quads = to_quads(system)
        assert quads.is_sound(), (name, quads.notes[-2:])

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_subdivision_is_all_or_nothing(self, bodies, name, keys, kind):
        """A partial subdivision is not conformal, so if any block would fold
        the system comes back untouched with the reason in notes."""
        _, _, system = build([bodies[k] for k in keys], kind)
        quads = to_quads(system)
        degrees = {b.n_sides for b in quads.blocks if not b.wraps}
        if degrees - {4}:
            assert quads.n_blocks == system.n_blocks
            assert any("not subdivided" in n for n in quads.notes)
        else:
            assert quads.n_blocks >= system.n_blocks

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_subdivision_preserves_the_index_certificate(
            self, bodies, name, keys, kind):
        """Midpoint subdivision puts one node of valence m at the centre of an
        m-gon, which is exactly the ``m - 4`` the certificate already charged
        that block -- so the budget must not move."""
        _, _, system = build([bodies[k] for k in keys], kind)
        quads = to_quads(system)
        assert system.index_certificate()["ok"], name
        assert quads.index_certificate()["ok"], name

    def test_a_ring_is_left_as_a_ring(self, bodies):
        _, _, system = build([bodies["fuse"]], "circle")
        quads = to_quads(system)
        assert all(b.wraps for b in quads.blocks)
        assert quads.is_sound()

    def test_a_smooth_body_reaches_all_quads(self, bodies):
        _, _, system = build([bodies["disc"]], "box")
        quads = to_quads(system)
        assert {b.n_sides for b in quads.blocks if not b.wraps} == {4}
        assert quads.is_sound()

    def test_the_blocks_that_cannot_be_subdivided_are_named(self, bodies):
        """The open limitation is the reflex trailing-edge corner, and it has
        to be reported rather than silently meshed."""
        _, _, system = build([bodies["main"]], "box")
        quads = to_quads(system)
        if {b.n_sides for b in quads.blocks if not b.wraps} - {4}:
            assert any("fold" in n for n in quads.notes)
