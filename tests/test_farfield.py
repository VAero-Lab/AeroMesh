"""Invariant tests for far-field generation (S0).

Where the domain is truncated is a modelling choice. What it must never be is a
topology choice, so the generated curve must not carry corners that were not
asked for -- a crease in the truncation boundary would drive real blocking.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.domain.farfield import (
    box_farfield,
    c_farfield,
    circle_farfield,
    offset_farfield,
    rectangle,
)
from aeromesh.geometry.corners import detect_corners
from aeromesh.geometry.loop import Loop


@pytest.fixture(scope="module")
def pair():
    main = am.load_airfoil("2412", n_points=300).transform(name="main")
    flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10,
                          name="flap")
    return main, flap


class TestOffsetFarfield:
    @pytest.mark.parametrize("distance", [2.0, 5.0, 15.0, 50.0])
    def test_encloses_every_body(self, pair, distance):
        ff = offset_farfield(list(pair), distance)
        for b in pair:
            assert bool(ff.contains(b.points).all())

    def test_distance_is_honoured(self, pair):
        from scipy.spatial import ConvexHull, cKDTree

        pts = np.vstack([b.points for b in pair])
        hull = pts[ConvexHull(pts).vertices]
        dense = Loop.from_points(hull).resample(n=8000)
        ff = offset_farfield(list(pair), 7.0, n_points=360)
        d = cKDTree(dense.points).query(ff.points)[0]
        assert d.min() == pytest.approx(7.0, rel=2e-3)
        assert d.max() == pytest.approx(7.0, rel=2e-3)

    def test_hull_offset_is_smooth(self, pair):
        """A crease in the truncation boundary is a corner that drives topology."""
        ff = offset_farfield(list(pair), 3.0)
        assert len(detect_corners(ff.resample(spacing=ff.scale / 500))) == 0

    def test_per_body_offset_creases_between_separated_bodies(self, pair):
        """Documents why the hull is the default, rather than hiding it."""
        ff = offset_farfield(list(pair), 3.0, hull=False)
        assert len(detect_corners(ff.resample(spacing=ff.scale / 500))) > 0

    def test_downstream_stretches_the_wake_side_only(self, pair):
        """Wake extension on the level set. A circle has no wake direction --
        use c_farfield or box_farfield for that."""
        base = offset_farfield(list(pair), 3.0)
        long = offset_farfield(list(pair), 3.0, downstream=9.0)
        assert long.bbox[1][0] > base.bbox[1][0] + 5.0     # +x extends
        assert long.bbox[0][0] == pytest.approx(base.bbox[0][0], abs=1e-6)  # -x does not

    def test_scales_to_body_count_without_special_casing(self, pair):
        main, flap = pair
        slat = main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02,
                              name="slat")
        for group in ([main], [main, flap], [main, flap, slat]):
            ff = offset_farfield(group, 4.0)
            assert all(bool(ff.contains(b.points).all()) for b in group)

    def test_rejects_bad_arguments(self, pair):
        with pytest.raises(ValueError, match="at least one body"):
            offset_farfield([], 3.0)
        with pytest.raises(ValueError, match="positive"):
            offset_farfield(list(pair), 0.0)


class TestConventionalShapes:
    """The standard CFD truncation curves, and the corner counts that make
    them differ. A corner of the fluid boundary generates a medial flare and a
    flare generates a block, so these counts are the whole reason the choice of
    shape is a real modelling decision rather than a cosmetic one."""

    EXPECTED_CORNERS = {"circle": 0, "c": 2, "box": 4, "offset": 0}

    @staticmethod
    def build(kind, bodies):
        if kind == "circle":
            return circle_farfield(bodies, 15.0)
        if kind == "c":
            return c_farfield(bodies, 15.0, 25.0)
        if kind == "box":
            return box_farfield(bodies, 15.0, 25.0, 15.0)
        return offset_farfield(bodies, 15.0)

    @pytest.mark.parametrize("kind", ["circle", "c", "box", "offset"])
    def test_corner_count(self, pair, kind):
        import aeromesh as am
        region = am.build_region(list(pair), outer=self.build(kind, list(pair)))
        assert len(region.outer.corners) == self.EXPECTED_CORNERS[kind]

    @pytest.mark.parametrize("kind", ["circle", "c", "box"])
    def test_every_corner_is_a_right_angle(self, pair, kind):
        import aeromesh as am
        region = am.build_region(list(pair), outer=self.build(kind, list(pair)))
        if len(region.outer.corners):
            np.testing.assert_allclose(
                np.degrees(region.outer.fluid_angles), 90.0, atol=1.0)

    @pytest.mark.parametrize("kind", ["circle", "c", "box", "offset"])
    def test_encloses_every_body(self, pair, kind):
        ff = self.build(kind, list(pair))
        for b in pair:
            assert bool(ff.contains(b.points).all())

    def test_the_c_shape_arc_joins_its_sides_tangentially(self, pair):
        """Only the two downstream corners, never four."""
        ff = c_farfield(list(pair), 15.0, 25.0)
        c = detect_corners(ff.resample(spacing=ff.scale / 500))
        assert len(c) == 2
        assert (ff.points[c.index][:, 0] > 0).all()      # both downstream

    def test_circle_radius_is_exact(self, pair):
        ff = circle_farfield(list(pair), 9.0)
        centre = 0.5 * (ff.bbox[0] + ff.bbox[1])
        r = np.linalg.norm(ff.points - centre, axis=1)
        np.testing.assert_allclose(r, 9.0, rtol=1e-9)

    def test_box_margins_are_measured_from_the_body_box(self, pair):
        lo_b = np.vstack([b.points for b in pair]).min(axis=0)
        hi_b = np.vstack([b.points for b in pair]).max(axis=0)
        ff = box_farfield(list(pair), 3.0, 7.0, 4.0)
        lo, hi = ff.bbox
        assert lo[0] == pytest.approx(lo_b[0] - 3.0)
        assert hi[0] == pytest.approx(hi_b[0] + 7.0)
        assert lo[1] == pytest.approx(lo_b[1] - 4.0)
        assert hi[1] == pytest.approx(hi_b[1] + 4.0)

    @pytest.mark.parametrize("kind", ["circle", "c", "box"])
    def test_rejects_nonsense_extents(self, pair, kind):
        with pytest.raises(ValueError):
            if kind == "circle":
                circle_farfield(list(pair), -1.0)
            elif kind == "c":
                c_farfield(list(pair), 15.0, -1.0)
            else:
                box_farfield(list(pair), 15.0, 25.0, 0.0)

    def test_refuses_a_far_field_that_does_not_enclose(self, pair):
        with pytest.raises(ValueError, match="does not enclose"):
            circle_farfield(list(pair), 0.2)


class TestShapeDoesNotLeak:
    """The architectural line: a curve is passed, a curve is stored. No later
    stage can branch on which constructor made the outer boundary."""

    def test_region_records_no_shape(self, pair):
        import aeromesh as am
        from dataclasses import fields
        region = am.build_region(list(pair), outer=c_farfield(list(pair), 15.0, 25.0))
        names = {f.name for f in fields(region)} | {f.name for f in fields(region.outer)}
        assert not names & {"shape", "kind", "domain_type", "topology", "family"}

    def test_two_shapes_give_structurally_identical_objects(self, pair):
        import aeromesh as am
        a = am.build_region(list(pair), outer=circle_farfield(list(pair), 15.0))
        b = am.build_region(list(pair), outer=box_farfield(list(pair), 15, 25, 15))
        assert type(a) is type(b)
        assert a.n_bodies == b.n_bodies
        assert a.euler_characteristic == b.euler_characteristic

    def test_an_arbitrary_hand_built_curve_is_accepted(self, pair):
        """Nothing requires the outer boundary to come from a constructor."""
        import aeromesh as am
        t = np.linspace(0, 2 * np.pi, 900, endpoint=False)
        r = 14.0 + 2.0 * np.sin(3 * t)                  # a shape with no name
        odd = Loop.from_points(np.column_stack([r * np.cos(t), r * np.sin(t)]), "odd")
        region = am.build_region(list(pair), outer=odd)
        assert region.n_bodies == 2


class TestRectangle:
    def test_has_exactly_four_right_angle_corners(self):
        r = rectangle(-4, -3, 8, 3).resample(spacing=0.05)
        c = detect_corners(r)
        assert len(c) == 4
        np.testing.assert_allclose(np.degrees(c.turn), 90.0, atol=0.5)

    def test_dimensions_are_exact(self):
        r = rectangle(-4, -3, 8, 3)
        lo, hi = r.bbox
        np.testing.assert_allclose(lo, [-4, -3])
        np.testing.assert_allclose(hi, [8, 3])
        assert r.area == pytest.approx(12 * 6)

    def test_rejects_an_inverted_box(self):
        with pytest.raises(ValueError):
            rectangle(1, 0, 0, 1)
