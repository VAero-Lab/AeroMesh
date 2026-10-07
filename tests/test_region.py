"""Invariant tests for Region (S0).

The point of these is that one code path serves one body, three bodies, a
bluff body and a mixture. Anything that branches on body type or count fails
here.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.geometry.loop import Loop
from aeromesh.geometry.region import Region, largest_inscribed_circle


def ellipse(a, b, cx=0.0, cy=0.0, n=500, name="ellipse"):
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
        "circle": ellipse(0.5, 0.5, name="circle"),
    }


CONFIGS = [
    ("single airfoil", ["main"]),
    ("main + flap", ["main", "flap"]),
    ("slat + main + flap", ["main", "flap", "slat"]),
    ("airfoil + fuselage", ["main", "fuselage"]),
    ("fuselage alone", ["fuselage"]),
    ("circle alone", ["circle"]),
]


class TestLargestInscribedCircle:
    def test_exact_for_a_circle(self):
        c, r = largest_inscribed_circle(ellipse(2.0, 2.0, n=2000))
        assert r == pytest.approx(2.0, rel=2e-5)
        np.testing.assert_allclose(c, [0, 0], atol=1e-3)

    def test_exact_for_a_square(self):
        sq = Loop.from_points([[0, 0], [4, 0], [4, 4], [0, 4]]).resample(n=1200)
        c, r = largest_inscribed_circle(sq)
        assert r == pytest.approx(2.0, rel=1e-3)
        np.testing.assert_allclose(c, [2, 2], atol=1e-2)

    def test_interior_for_a_strongly_cambered_section(self):
        """A sample centroid can fall outside such a shape; this must not."""
        lp = am.load_airfoil("8412", n_points=400).resample(n=1200)
        c, r = largest_inscribed_circle(lp)
        assert bool(lp.contains(c[None, :])[0])
        assert r > 0


class TestOneCodePath:
    @pytest.mark.parametrize("name,keys", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_builds_and_validates(self, bodies, name, keys):
        region = am.build_region([bodies[k] for k in keys], farfield=3.0)
        assert region.n_bodies == len(keys)
        assert region.euler_characteristic == 1 - len(keys)

    @pytest.mark.parametrize("name,keys", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_every_farfield_shape_serves_every_configuration(self, bodies, name, keys):
        """The truncation curve and the body set are independent choices."""
        group = [bodies[k] for k in keys]
        for outer in (am.circle_farfield(group, 3.0),
                      am.c_farfield(group, 3.0, 5.0),
                      am.box_farfield(group, 3.0, 5.0, 3.0),
                      am.offset_farfield(group, 3.0)):
            region = am.build_region(group, outer=outer)
            assert region.n_bodies == len(keys)
            assert region.euler_characteristic == 1 - len(keys)

    @pytest.mark.parametrize("name,keys", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_hole_points_land_inside_their_bodies(self, bodies, name, keys):
        region = am.build_region([bodies[k] for k in keys], farfield=3.0)
        hp = region.hole_points()
        assert len(hp) == len(keys)
        assert not region.contains(hp).any()          # inside a body is not fluid
        for body, p in zip(region.holes, hp):
            assert bool(body.loop.contains(p[None, :])[0])

    @pytest.mark.parametrize("name,keys", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_sample_is_consistent(self, bodies, name, keys):
        region = am.build_region([bodies[k] for k in keys], farfield=3.0)
        pts, loop_id, offsets = region.sample()
        assert len(pts) == len(loop_id) == offsets[-1]
        assert offsets[0] == 0
        assert len(offsets) == region.n_bodies + 2
        assert loop_id.max() == region.n_bodies


class TestCornerResolution:
    def test_a_blunted_section_carries_two_corners(self, bodies):
        region = am.build_region([bodies["main"]], farfield=3.0, te="blunt")
        assert len(region.holes[0].corners) == 2
        np.testing.assert_array_equal(region.holes[0].corner_elements, [3, 3])

    def test_a_bluff_body_carries_none(self, bodies):
        region = am.build_region([bodies["fuselage"]], farfield=3.0)
        assert len(region.holes[0].corners) == 0

    def test_the_default_circle_is_smooth(self, bodies):
        region = am.build_region([bodies["main"], bodies["flap"]], farfield=3.0)
        assert len(region.outer.corners) == 0

    def test_a_c_shape_contributes_exactly_two(self, bodies):
        group = [bodies["main"]]
        region = am.build_region(group, outer=am.c_farfield(group, 3.0, 5.0))
        assert len(region.outer.corners) == 2

    def test_a_prescribed_box_keeps_its_four_corners(self, bodies):
        region = am.build_region([bodies["main"]], outer=am.box_farfield(
            [bodies["main"]], 4.0, 6.0, 4.0))
        assert len(region.outer.corners) == 4
        np.testing.assert_allclose(
            np.degrees(region.outer.fluid_angles), 90.0, atol=1.0)

    def test_sharp_policy_leaves_the_cusp_alone(self, bodies):
        region = am.build_region([bodies["main"]], farfield=3.0, te="sharp")
        c = region.holes[0].corners
        assert len(c) == 1
        assert np.degrees(c.turn[0]) > 150.0       # still a cusp


class TestMultipleCusps:
    """A body with two sharp tips must get two faces, not one and a loop that
    reopens the first. Every cut renumbers the loop, so faces are tracked by
    position; and a face narrower than the detector's window reads as a single
    corner whose turn is the sum of its two, which looks exactly like a cusp.
    """

    @staticmethod
    def lens(n=1200):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return Loop.from_points(
            np.column_stack([np.cos(t), 0.25 * np.sin(t) ** 3]), "lens")

    def test_two_cusps_are_detected(self):
        from aeromesh.geometry.cusp import find_cusps
        assert len(find_cusps(self.lens().resample(n=2000))) == 2

    @pytest.mark.parametrize("thickness", [0.005, 0.01])
    def test_both_tips_are_opened(self, thickness):
        from aeromesh.geometry.region import prepare_boundary
        b = prepare_boundary(self.lens(), is_hole=True, te="blunt",
                             te_thickness=thickness)
        assert len(b.corners) == 4
        np.testing.assert_array_equal(b.corner_elements, [3, 3, 3, 3])

    @pytest.mark.parametrize("thickness", [0.001, 0.002, 0.005])
    def test_a_single_cusp_body_is_not_reopened(self, thickness, bodies):
        """Regression: the face must not read as a cusp on the next pass."""
        from aeromesh.geometry.region import prepare_boundary
        b = prepare_boundary(bodies["main"], is_hole=True, te="blunt",
                             te_thickness=thickness)
        assert len(b.corners) == 2
        p0, p1 = b.corners.positions(b.loop)
        assert np.linalg.norm(p1 - p0) == pytest.approx(thickness, rel=1e-3)


class TestValidation:
    def test_rejects_a_body_outside_the_outer_boundary(self, bodies):
        outer = am.rectangle(-1, -1, 1, 1)
        stray = bodies["main"].transform(dx=50.0)
        with pytest.raises(ValueError, match="not strictly inside"):
            Region.build(outer, [stray])

    def test_rejects_overlapping_bodies(self, bodies):
        a = bodies["main"]
        b = bodies["main"].transform(dx=0.1, dy=0.005, name="overlapping")
        with pytest.raises(ValueError, match="overlap"):
            Region.build(am.rectangle(-10, -10, 20, 10), [a, b])

    def test_rejects_a_self_intersecting_loop(self):
        bowtie = Loop.from_points([[0, 0], [1, 1], [1, 0], [0, 1]])
        with pytest.raises(ValueError, match="self-intersecting"):
            Region.build(am.rectangle(-5, -5, 5, 5), [bowtie])


class TestPolicyArguments:
    def test_rejects_an_unknown_te_policy(self, bodies):
        with pytest.raises(ValueError, match="blunt"):
            am.build_region([bodies["main"]], farfield=3.0, te="rounded")

    def test_rejects_an_empty_body_set(self):
        with pytest.raises(ValueError, match="at least one body"):
            am.build_region([])
