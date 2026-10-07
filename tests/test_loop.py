"""Invariant tests for the Loop primitive (S0).

These assert geometric facts, not array shapes. A shape assertion cannot tell
you a skeleton has branches of radius zero; an invariant can.
"""

import numpy as np
import pytest

from aeromesh.geometry.loop import Loop, drop_duplicates, signed_area


def circle(n=400, r=1.0, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]))


UNIT_SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1]]


class TestConstruction:
    def test_rewinds_clockwise_input_to_ccw(self):
        cw = Loop.from_points(UNIT_SQUARE[::-1])
        assert cw.area > 0

    def test_ccw_input_is_left_alone(self):
        ccw = Loop.from_points(UNIT_SQUARE)
        np.testing.assert_allclose(ccw.points, np.array(UNIT_SQUARE, dtype=float))

    def test_closing_point_is_not_stored(self):
        lp = Loop.from_points(UNIT_SQUARE + [[0, 0]])
        assert lp.n_points == 4

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            Loop.from_points([[0, 0], [1, 1]])
        with pytest.raises(ValueError):
            Loop.from_points(np.zeros((5, 3)))

    def test_drop_duplicates_handles_the_wrap(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        assert len(drop_duplicates(pts)) == 4


class TestMeasures:
    def test_square_area_perimeter_centroid(self):
        sq = Loop.from_points(UNIT_SQUARE)
        assert sq.area == pytest.approx(1.0)
        assert sq.perimeter == pytest.approx(4.0)
        np.testing.assert_allclose(sq.centroid, [0.5, 0.5], atol=1e-12)

    def test_circle_converges_to_pi(self):
        assert circle(2000).area == pytest.approx(np.pi, rel=1e-5)
        assert circle(2000).perimeter == pytest.approx(2 * np.pi, rel=1e-5)

    def test_centroid_is_area_weighted_not_sample_mean(self):
        """A shape densely sampled on one side must not drag its centroid."""
        dense = np.linspace(0, 1, 200)
        pts = np.vstack([
            np.column_stack([dense, np.zeros_like(dense)]),
            [[1, 1], [0, 1]],
        ])
        lp = Loop.from_points(pts)
        np.testing.assert_allclose(lp.centroid, [0.5, 0.5], atol=1e-3)
        assert abs(lp.points.mean(axis=0)[1] - 0.5) > 0.1  # the sample mean is fooled


class TestResampling:
    def test_preserves_perimeter_and_scale(self):
        """Corner preservation means a polygon keeps its perimeter exactly.

        Without it, uniform arc-length sampling chamfers every corner and the
        far-field box silently shrinks.
        """
        sq = Loop.from_points(UNIT_SQUARE)
        r = sq.resample(n=97)
        assert r.n_points == pytest.approx(97, abs=5)
        assert r.perimeter == pytest.approx(4.0, abs=1e-12)
        assert r.area == pytest.approx(1.0, abs=1e-12)
        assert r.scale == pytest.approx(sq.scale)

    def test_without_corner_preservation_a_polygon_is_chamfered(self):
        sq = Loop.from_points(UNIT_SQUARE)
        r = sq.resample(n=97, preserve_corners=False)
        assert r.n_points == 97
        assert r.perimeter < 4.0

    def test_smooth_curve_is_unaffected_by_corner_preservation(self):
        c = circle(500)
        a = c.resample(n=311, preserve_corners=True)
        b = c.resample(n=311, preserve_corners=False)
        np.testing.assert_allclose(a.points, b.points)

    def test_spacing_is_honoured(self):
        r = Loop.from_points(UNIT_SQUARE).resample(spacing=0.01)
        assert r.edge_lengths.max() == pytest.approx(0.01, rel=1e-9)

    def test_sharp_vertices_finds_polygon_corners_and_nothing_on_a_circle(self):
        assert Loop.from_points(UNIT_SQUARE).sharp_vertices().tolist() == [0, 1, 2, 3]
        assert circle(400).sharp_vertices().size == 0

    def test_requires_exactly_one_of_n_or_spacing(self):
        sq = Loop.from_points(UNIT_SQUARE)
        with pytest.raises(ValueError):
            sq.resample()
        with pytest.raises(ValueError):
            sq.resample(n=10, spacing=0.1)

    def test_arclength_is_monotone_and_starts_at_zero(self):
        s = circle(300).arclength
        assert s[0] == 0.0
        assert np.all(np.diff(s) > 0)


class TestContainment:
    def test_square_spot_checks(self):
        sq = Loop.from_points(UNIT_SQUARE)
        got = sq.contains([[0.5, 0.5], [1.5, 0.5], [0.5, -0.2], [-0.01, 0.5]])
        np.testing.assert_array_equal(got, [True, False, False, False])

    def test_agrees_with_matplotlib_on_a_nonconvex_shape(self):
        mpl_path = pytest.importorskip("matplotlib.path")
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        r = 1.0 + 0.4 * np.cos(5 * t)          # a five-lobed star
        lp = Loop.from_points(np.column_stack([r * np.cos(t), r * np.sin(t)]))
        rng = np.random.default_rng(12345)
        q = rng.uniform(-1.6, 1.6, size=(20000, 2))
        ref = mpl_path.Path(lp.closed_points).contains_points(q)
        np.testing.assert_array_equal(lp.contains(q), ref)


class TestTransform:
    def test_rigid_motion_preserves_area_and_perimeter(self):
        sq = Loop.from_points(UNIT_SQUARE).resample(n=200)
        moved = sq.transform(angle=0.7, dx=-3.0, dy=11.0)
        assert moved.area == pytest.approx(sq.area)
        assert moved.perimeter == pytest.approx(sq.perimeter)

    def test_scaling_is_quadratic_in_area(self):
        sq = Loop.from_points(UNIT_SQUARE).resample(n=200)
        assert sq.transform(scale=3.0).area == pytest.approx(9.0 * sq.area)

    def test_name_can_be_replaced(self):
        sq = Loop.from_points(UNIT_SQUARE, name="a")
        assert sq.transform(dx=1.0).name == "a"
        assert sq.transform(dx=1.0, name="b").name == "b"


def test_signed_area_sign_convention():
    assert signed_area(np.array(UNIT_SQUARE, dtype=float)) > 0
    assert signed_area(np.array(UNIT_SQUARE[::-1], dtype=float)) < 0
