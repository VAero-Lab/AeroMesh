"""Tests for constant-rho offsets -- the near-wall layer (S3 item C).

rho = d_wall / r_m, so rho = 0 is the wall and rho = 1 the medial axis. The
offset curve at rho* is the outer edge of a near-wall block. What these tests
pin down is the construction: each boundary sample moves along its *own* normal
by rho times the medial radius seen from it, with the radius field smoothed
rather than the curve.
"""

import numpy as np
import pytest
from shapely.geometry import LinearRing, Polygon

import aeromesh as am
from aeromesh.blocking.layers import (
    medial_radius_on_boundary,
    outward_normals,
    rho_offset,
)
from aeromesh.geometry.loop import Loop


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="e"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


@pytest.fixture(scope="module")
def single():
    body = am.load_airfoil("2412", n_points=400).transform(name="main")
    region = am.build_region([body], outer=am.box_farfield([body], 4.0, 6.0, 4.0))
    return region, am.exterior_axis(region)


@pytest.fixture(scope="module")
def bodies():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    return {
        "main": main,
        "flap": main.transform(scale=0.30, angle=np.deg2rad(-25), dx=1.05,
                               dy=-0.08, name="flap"),
        "fuselage": ellipse(1.6, 0.55, n=400, name="fuselage"),
    }


@pytest.fixture(scope="module")
def pair():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10,
                          name="flap")
    region = am.build_region([main, flap],
                             outer=am.box_farfield([main, flap], 4.0, 6.0, 4.0))
    return region, am.exterior_axis(region)


class TestNormals:
    def test_normals_point_out_of_a_ccw_loop(self):
        c = ellipse(1.0, 1.0, n=400)
        n = outward_normals(c)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-9)
        # on a unit circle the outward normal is the position itself
        np.testing.assert_allclose(n, c.points, atol=1e-3)

    def test_stepping_along_a_normal_leaves_the_body(self):
        c = ellipse(1.0, 0.6, n=600)
        moved = c.points + 0.05 * outward_normals(c)
        assert not c.contains(moved).any()


class TestRadiusField:
    def test_one_radius_per_boundary_sample(self, single):
        region, axis = single
        radii = medial_radius_on_boundary(axis, region)
        for li, arr in radii.items():
            assert len(arr) == region.boundaries[li].loop.n_points
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0)

    def test_the_radius_is_positive_on_a_body_but_vanishes_at_an_outer_corner(
            self, single):
        """A flare runs out to a corner of the truncation curve with r_m -> 0,
        so the outer boundary's radius field legitimately touches zero there --
        which is why offsetting it would pinch, and why it is skipped."""
        region, axis = single
        radii = medial_radius_on_boundary(axis, region)
        assert np.all(radii[1] > 0)
        assert radii[0].min() < 1e-2 * region.outer.loop.scale

    def test_the_radius_matches_the_annulus_value(self):
        """For a body centred in a circular far field the medial radius is half
        the gap, everywhere."""
        inner, outer = ellipse(1.0, 1.0, n=1200, name="inner"), ellipse(3.0, 3.0, n=2000, name="outer")
        from aeromesh.geometry.region import Region
        region = Region.build(outer, [inner], te="sharp", validate=False)
        radii = medial_radius_on_boundary(am.exterior_axis(region), region)
        np.testing.assert_allclose(radii[1], 1.0, rtol=2e-3)


class TestOffsetCurve:
    @pytest.mark.parametrize("rho", [0.02, 0.05, 0.10, 0.25])
    def test_the_offset_is_a_simple_closed_curve(self, single, rho):
        region, axis = single
        for loop in rho_offset(axis, region, rho).values():
            assert LinearRing(loop.closed_points).is_simple

    @pytest.mark.parametrize("rho", [0.02, 0.05, 0.10])
    def test_the_offset_encloses_its_body_and_stays_in_the_fluid(self, single, rho):
        region, axis = single
        for li, loop in rho_offset(axis, region, rho).items():
            body = region.boundaries[li].loop
            assert Polygon(loop.closed_points).buffer(0).contains(
                Polygon(body.closed_points).buffer(0))
            assert region.outer.loop.contains(loop.points).all()

    def test_the_offset_distance_scales_with_rho(self, single):
        from scipy.spatial import cKDTree

        region, axis = single
        body = region.holes[0].loop
        tree = cKDTree(body.points)
        means = []
        for rho in (0.02, 0.04, 0.08):
            loop = rho_offset(axis, region, rho)[1]
            means.append(float(np.mean(tree.query(loop.points)[0])))
        ratios = [means[1] / means[0], means[2] / means[1]]
        for r in ratios:
            assert 1.8 < r < 2.2, f"doubling rho should roughly double the offset: {ratios}"

    def test_rho_must_lie_between_the_wall_and_the_axis(self, single):
        region, axis = single
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="rho"):
                rho_offset(axis, region, bad)

    def test_the_outer_boundary_is_skipped_by_default(self, single):
        """Offsetting it is well defined but pinches to nothing at its corners,
        where a flare's r_m goes to zero."""
        region, axis = single
        assert 0 not in rho_offset(axis, region, 0.05)
        assert 0 in rho_offset(axis, region, 0.05, bodies_only=False)


class TestThicknessLaw:
    """What a constant-rho layer does, and where it is the wrong law.

    These are measurements rather than requirements -- they record the behaviour
    the next step has to accommodate.
    """

    @staticmethod
    def spread(region, axis, rho=0.05):
        from scipy.spatial import cKDTree

        loop = rho_offset(axis, region, rho)[1]
        d = cKDTree(region.holes[0].loop.points).query(loop.points)[0]
        return float(d.max() / d.min())

    def test_for_one_body_the_layer_is_nearly_uniform(self, single):
        """r_m barely varies around a lone body, so a constant fraction of it is
        a sensible thickness."""
        region, axis = single
        assert self.spread(region, axis) < 2.0

    def test_in_a_slot_the_layer_thins_sharply(self, pair):
        """r_m collapses between two bodies, so a constant fraction of it does
        too -- correct for the coordinate, wrong for a boundary layer, which is
        why the thickness law needs a cap rather than rho alone."""
        region, axis = pair
        assert self.spread(region, axis) > 10.0


# ═══════════════════════════════════════════════════════════════════
#  Inserting the layer into the blocking (S3 item C, second attempt)
# ═══════════════════════════════════════════════════════════════════

class TestInsertLayer:
    """The layer goes in as a split of the blocking, not as a new region to be
    decomposed. Topology is then inherited rather than recomputed, which is what
    went wrong the first way round: the offset curve grows medial branches of its
    own, and a body in a box went from 4 blocks to 15."""

    LAYOUTS = [
        ("aerofoil / box", ["main"], "box"),
        ("aerofoil / circle", ["main"], "circle"),
        ("aerofoil / C", ["main"], "c"),
        ("two aerofoils / box", ["main", "flap"], "box"),
        ("two aerofoils / circle", ["main", "flap"], "circle"),
        ("two aerofoils / C", ["main", "flap"], "c"),
        ("fuselage / circle", ["fuselage"], "circle"),
        ("fuselage / box", ["fuselage"], "box"),
    ]

    @staticmethod
    def _build(group, kind):
        outer = {"circle": lambda g: am.circle_farfield(g, 8.0),
                 "c": lambda g: am.c_farfield(g, 8.0, 14.0),
                 "box": lambda g: am.box_farfield(g, 4.0, 6.0, 4.0)}[kind](group)
        region = am.build_region(group, outer=outer)
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        return region, axis, am.decompose(axis, region, field=field)

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_the_layered_blocking_is_still_a_decomposition(
            self, bodies, name, keys, kind):
        _, _, system = self._build([bodies[k] for k in keys], kind)
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert layered.validate()["tiles"], name
        assert layered.validate_strict()["ok"], name
        assert layered.interface_report()["conformal"], name

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_the_layer_creates_no_singularity(self, bodies, name, keys, kind):
        """Where the offset crosses a cut, four blocks meet -- the inner and
        outer halves of the two either side -- so the node has valence four and
        the index budget is untouched."""
        _, _, system = self._build([bodies[k] for k in keys], kind)
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert system.index_certificate()["ok"], name
        assert layered.index_certificate()["ok"], name

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    def test_every_block_gains_a_layer_and_nothing_is_skipped(
            self, bodies, name, keys, kind):
        _, _, system = self._build([bodies[k] for k in keys], kind)
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert len(layered.notes) == len(system.notes), layered.notes[-3:]
        assert layered.n_blocks > system.n_blocks, name
        assert any(b.kind == "layer" or b.kind == "ring" for b in layered.blocks)

    def test_the_layer_hugs_the_wall_at_the_thickness_asked_for(self, bodies):
        from scipy.spatial import cKDTree
        region, _, system = self._build([bodies["main"]], "box")
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        wall = cKDTree(region.holes[0].loop.points)
        for b in layered.blocks:
            if b.kind != "layer":
                continue
            d = wall.query(b.outline())[0]
            assert d.max() <= 0.02 * 1.6, b
            assert d.min() <= 1e-9

    def test_a_ring_splits_into_two_rings(self, bodies):
        """A smooth body in a smooth far field is one ring block; the layer makes
        it two, both still rings, and the inner one is the O-grid band."""
        _, _, system = self._build([bodies["fuselage"]], "circle")
        assert system.kinds() == {"ring": 1}
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert layered.kinds() == {"ring": 2}
        assert layered.validate()["tiles"]

    def test_the_offset_goes_into_the_fluid_not_into_the_body(self, bodies):
        """The sign of the wall normal cannot be read from the loop's handedness,
        because orient_ccw may have reversed the side, nor by comparing against a
        reference direction taken at a different point of a closed arc -- on a
        ring those are very nearly opposite, and the layer lands inside the body.
        """
        region, _, system = self._build([bodies["fuselage"]], "circle")
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        inner = [b for b in layered.blocks if b.wraps][0]
        outer_area = abs(Loop.from_points(inner.sides[0]).area)
        wall_area = abs(Loop.from_points(inner.sides[1]).area)
        assert outer_area > wall_area

    def test_a_block_touching_the_wall_at_one_point_still_gets_a_layer(self, bodies):
        """The wake region of a C-type far field meets the wall only at the
        trailing-edge point, so it has no wall side at all. Driving the trim from
        "this block has a wall run" left its cuts at full length while its
        neighbours trimmed theirs, and it overlapped their layer by 2e-8 of the
        region -- two orders below what the tiling check resolves, and caught by
        the certificate."""
        _, _, system = self._build([bodies["main"]], "c")
        wedge = [b for b in system.blocks if b.n_sides == 3]
        assert wedge, "expected a pinched wake region"
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert layered.validate_strict()["overlap_fraction"] < 1e-12
        assert layered.index_certificate()["ok"]

    def test_the_thickness_asked_for_scales_the_layer(self, bodies):
        _, _, system = self._build([bodies["main"]], "box")
        areas = []
        for target in (0.01, 0.02, 0.04):
            layered = am.insert_layer(system, delta_target=target, rho_max=0.05)
            areas.append(sum(b.area for b in layered.blocks
                             if b.kind == "layer"))
        assert areas[0] < areas[1] < areas[2]
        assert areas[1] / areas[0] == pytest.approx(2.0, rel=0.1)


class TestSubcurve:
    def test_a_piece_keeps_its_orientation_and_exact_ends(self):
        from aeromesh.blocking.layers import subcurve
        c = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        q = subcurve(c, 0.5, 2.5)
        np.testing.assert_allclose(q[0], [0.5, 0.0])
        np.testing.assert_allclose(q[-1], [2.5, 0.0])
        assert np.all(np.diff(q[:, 0]) > 0)

    def test_two_pieces_share_their_endpoint_exactly(self):
        from aeromesh.blocking.layers import subcurve
        c = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 2.0]])
        total = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        a = subcurve(c, 0.0, 1.3)
        b = subcurve(c, 1.3, total)
        np.testing.assert_array_equal(a[-1], b[0])

    def test_the_whole_curve_round_trips(self):
        from aeromesh.blocking.layers import subcurve
        c = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        total = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        q = subcurve(c, 0.0, total)
        np.testing.assert_allclose(q[0], c[0])
        np.testing.assert_allclose(q[-1], c[-1])

    def test_an_impossible_range_is_rejected(self):
        from aeromesh.blocking.layers import subcurve
        c = np.array([[0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            subcurve(c, 0.5, 0.4)
        with pytest.raises(ValueError):
            subcurve(c, -0.1, 0.5)


class TestWallRuns:
    def test_a_single_run(self):
        from aeromesh.blocking.layers import _wall_runs
        assert _wall_runs(["outer", "cut", "wall", "wall", "cut"]) == [(2, 3)]

    def test_two_runs(self):
        from aeromesh.blocking.layers import _wall_runs
        runs = _wall_runs(["cut", "wall", "wall", "cut", "wall", "wall",
                           "cut", "outer"])
        assert sorted(runs) == [(1, 2), (4, 5)]

    def test_a_run_that_wraps(self):
        from aeromesh.blocking.layers import _wall_runs
        assert _wall_runs(["wall", "cut", "outer", "cut", "wall"]) == [(4, 0)]

    def test_no_wall_at_all(self):
        from aeromesh.blocking.layers import _wall_runs
        assert _wall_runs(["outer", "cut", "cut"]) == []

    def test_all_wall(self):
        from aeromesh.blocking.layers import _wall_runs
        assert _wall_runs(["wall", "wall"]) == []

    def test_run_length_is_cyclic(self):
        from aeromesh.blocking.layers import _run_len
        assert _run_len(2, 3, 9) == 2
        assert _run_len(4, 0, 5) == 2
        assert _run_len(0, 0, 5) == 1
