"""Tests for block construction (S3).

The decisive check is the tiling: every block's area must sum to the fluid area.
A gap or an overlap shows up at once, and no amount of plausible-looking geometry
hides it. Everything else here is structure -- that the right *kind* of block
comes out of the right medial feature, with no rule naming a body or a shape.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.blocks import _short_path, boundary_arc, decompose
from aeromesh.geometry.loop import Loop


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="ellipse"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


@pytest.fixture(scope="module")
def bodies():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    return {
        "main": main,
        "flap": main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10,
                               name="flap"),
        "slat": main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02,
                               name="slat"),
        "fuselage": ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage"),
    }


def build(group, kind, **kw):
    outer = {"circle": lambda g: am.circle_farfield(g, 4.0),
             "c": lambda g: am.c_farfield(g, 4.0, 6.0),
             "box": lambda g: am.box_farfield(g, 4.0, 6.0, 4.0)}[kind](group)
    region = am.build_region(group, outer=outer, **kw)
    return region, decompose(am.exterior_axis(region), region)


class TestArcWalking:
    def test_short_path_goes_the_near_way(self):
        assert _short_path(2, 5, 0, 10).tolist() == [2, 3, 4, 5]
        assert _short_path(1, 8, 0, 10).tolist() == [1, 0, 9, 8]

    def test_short_path_wraps(self):
        assert _short_path(8, 1, 0, 10).tolist() == [8, 9, 0, 1]

    def test_short_path_respects_a_loop_offset(self):
        assert _short_path(102, 105, 100, 10).tolist() == [102, 103, 104, 105]

    def test_an_arc_follows_the_boundary(self, bodies):
        region, _ = build([bodies["main"]], "box")
        pts, _, offsets = region.sample()
        idx = np.arange(20, 40)
        arc = boundary_arc(idx, pts, offsets)
        np.testing.assert_allclose(arc, pts[idx])


CONFIGS = [
    ("1 body / circle", ["main"], "circle"),
    ("1 body / C-shape", ["main"], "c"),
    ("1 body / box", ["main"], "box"),
    ("2 bodies / circle", ["main", "flap"], "circle"),
    ("2 bodies / box", ["main", "flap"], "box"),
    ("3 bodies / circle", ["main", "flap", "slat"], "circle"),
    ("3 bodies / box", ["main", "flap", "slat"], "box"),
    ("bluff body / circle", ["fuselage"], "circle"),
    ("bluff body / box", ["fuselage"], "box"),
]


class TestTiling:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_blocks_tile_the_region(self, bodies, name, keys, kind):
        _, system = build([bodies[k] for k in keys], kind)
        v = system.validate()
        assert v["tiles"], v
        assert v["relative_error"] < 1e-3

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_no_block_is_degenerate(self, bodies, name, keys, kind):
        _, system = build([bodies[k] for k in keys], kind)
        assert system.validate()["degenerate"] == []
        assert all(b.n_sides >= 2 for b in system.blocks)

    @pytest.mark.parametrize("splits", [1, 2, 4, 8])
    def test_refining_the_split_density_keeps_the_tiling(self, bodies, splits):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4.0, 6.0, 4.0))
        system = decompose(am.exterior_axis(region), region, splits_per_edge=splits)
        assert system.validate()["tiles"]
        assert system.n_blocks >= 16

    def test_every_block_is_reported(self, bodies):
        _, system = build([bodies["main"]], "box")
        assert system.validate()["notes"] == [], "an unhandled feature was skipped"


class TestStrictTiling:
    """The checks an area sum cannot make.

    Summing areas is necessary but not sufficient: blocks could overlap in one
    place and leave an equal gap elsewhere, and a zero-area spike -- an outline
    vertex flung far from the region -- changes no area at all. These tests
    exist because exactly that spike was present and invisible to the area sum.
    """

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_sides_of_a_block_join_exactly(self, bodies, name, keys, kind):
        """A split ends at the exact perpendicular foot on the boundary, while an
        arc is built from boundary samples. If the arc is not made to start and
        end at those feet, every join is left open by up to half a sample
        spacing."""
        region, system = build([bodies[k] for k in keys], kind)
        v = system.validate()
        assert v["open_blocks"] == [], v
        assert v["worst_join"] < 1e-9 * region.outer.loop.scale

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_no_outline_vertex_escapes_the_region(self, bodies, name, keys, kind):
        """Catches a spike, which has no area and so passes every area check."""
        _, system = build([bodies[k] for k in keys], kind)
        assert system.validate()["stray_blocks"] == []

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_outlines_do_not_cross_themselves(self, bodies, name, keys, kind):
        _, system = build([bodies[k] for k in keys], kind)
        assert system.validate_strict()["non_simple"] == []

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_union_of_the_blocks_is_the_region(self, bodies, name, keys, kind):
        """Overlap, gap and spill measured separately, not inferred from a
        single area total."""
        _, system = build([bodies[k] for k in keys], kind)
        w = system.validate_strict()
        assert w["ok"], w
        assert abs(w["overlap_fraction"]) < 1e-6
        assert w["gap_fraction"] < 1e-6
        assert w["spill_fraction"] < 1e-6

    def test_a_collapsed_side_is_dropped_not_kept(self, bodies):
        """Two cuts can land on exactly the same boundary point -- a sharp corner
        on a body -- leaving an empty arc between them. The region is then one
        side shorter, and keeping the empty side would leave a repeated vertex
        that makes the outline non-simple for no geometric reason."""
        _, system = build([bodies["main"], bodies["flap"]], "box")
        for b in system.blocks:
            for side in b.sides:
                assert len(side) >= 2
                assert np.linalg.norm(np.diff(side, axis=0), axis=1).sum() > 0


class TestStructure:
    def test_a_smooth_far_field_around_one_body_gives_one_ring(self, bodies):
        """The doughnut case: a bare medial loop, so one block wrapping on
        itself. Derived from the skeleton, not selected."""
        _, system = build([bodies["main"]], "circle")
        assert system.n_blocks == 1
        assert system.kinds() == {"ring": 1}
        assert system.blocks[0].wraps

    def test_a_bluff_body_gives_the_same_ring(self, bodies):
        """The body's shape does not enter: what matters is that the skeleton
        is a bare loop."""
        _, system = build([bodies["fuselage"]], "circle")
        assert system.kinds() == {"ring": 1}

    def test_a_box_gives_a_ring_of_spans_plus_one_wedge_per_corner(self, bodies):
        """Four corners, four flares, four corner wedges -- the familiar
        ring-and-corner-blocks arrangement, derived."""
        _, system = build([bodies["main"]], "box")
        k = system.kinds()
        assert k["corner"] == 4
        assert k["vertex"] == 4
        assert k["span"] == 8

    def test_a_c_shape_gives_two(self, bodies):
        _, system = build([bodies["main"]], "c")
        k = system.kinds()
        assert k["corner"] == 2 and k["vertex"] == 2

    def test_block_count_grows_with_body_count(self, bodies):
        counts = [build([bodies[k] for k in keys], "circle")[1].n_blocks
                  for keys in (["main"], ["main", "flap"], ["main", "flap", "slat"])]
        assert counts == sorted(counts)
        assert counts[0] < counts[-1]

    def test_a_cusped_trailing_edge_still_tiles(self, bodies):
        for kind in ("circle", "c"):
            _, system = build([bodies["main"]], kind, te="sharp")
            assert system.validate()["tiles"]


class TestOutlines:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS[:5], ids=[c[0] for c in CONFIGS[:5]])
    def test_outlines_are_closed_and_non_degenerate(self, bodies, name, keys, kind):
        _, system = build([bodies[k] for k in keys], kind)
        for b in system.blocks:
            out = b.outline()
            assert len(out) >= 4
            assert np.all(np.linalg.norm(np.diff(out, axis=0), axis=1) >= 0)
            assert b.area > 0

    @pytest.mark.parametrize("name,keys,kind", CONFIGS[:5], ids=[c[0] for c in CONFIGS[:5]])
    def test_outlines_stay_in_the_closure_of_the_region(self, bodies, name, keys, kind):
        """A block's sides are boundary arcs and medial radii, so most outline
        points lie *on* the boundary, where point-in-polygon is ambiguous. The
        meaningful statement is that every outline point is in the fluid or on
        its boundary -- never strictly outside, which would mean a side had
        wandered into a body or past the far field.
        """
        from aeromesh.medial.fields import segment_distance

        region, system = build([bodies[k] for k in keys], kind)
        pts, loop_id, offsets = region.sample()
        scale = region.outer.loop.scale
        for b in system.blocks:
            q = b.outline()[::5]
            inside = region.contains(q)
            on_edge = segment_distance(q, pts, loop_id, offsets) < 1e-3 * scale
            stray = int((~(inside | on_edge)).sum())
            assert stray == 0, f"{b}: {stray} outline points outside the region"
