"""Invariant tests for scale-invariant corner detection (S0).

The regression that matters: a NACA leading edge is high curvature, not a
corner. A fixed-threshold detector flags it, and every downstream stage then
inherits a feature that does not exist.
"""

import numpy as np
import pytest

from aeromesh.geometry.airfoil import load_airfoil
from aeromesh.geometry.corners import (
    detect_corners,
    fluid_interior_angle,
    vertex_turns,
)
from aeromesh.geometry.loop import Loop


def ellipse(a, b, n=800):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([a * np.cos(t), b * np.sin(t)]))


@pytest.fixture(params=["0012", "2412", "8412"])
def naca(request):
    return load_airfoil(request.param, n_points=400).resample(n=1200)


class TestTurningInvariant:
    @pytest.mark.parametrize("loop", [
        Loop.from_points([[0, 0], [1, 0], [1, 1], [0, 1]]).resample(n=400),
        ellipse(1.0, 1.0),
        ellipse(3.0, 1.0),
        load_airfoil("2412", n_points=300).resample(n=900),
    ])
    def test_total_turning_is_two_pi(self, loop):
        """Gauss-Bonnet for a simple closed curve. Holds for every loop, always."""
        assert vertex_turns(loop).sum() == pytest.approx(2 * np.pi, abs=1e-9)


class TestDiscrimination:
    def test_square_has_four_right_angle_corners(self):
        sq = Loop.from_points([[0, 0], [1, 0], [1, 1], [0, 1]]).resample(n=400)
        c = detect_corners(sq)
        assert len(c) == 4
        np.testing.assert_allclose(np.degrees(c.turn), 90.0, atol=0.5)
        np.testing.assert_allclose(c.residual, 0.0, atol=1e-6)

    def test_triangle_interior_angles_sum_to_pi(self):
        tri = Loop.from_points([[0, 0], [1, 0], [0.3, 0.8]]).resample(n=600)
        c = detect_corners(tri)
        assert len(c) == 3
        interior = fluid_interior_angle(c.turn, is_hole=False)
        assert interior.sum() == pytest.approx(np.pi, abs=1e-2)

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (3.0, 1.0), (1.05, 1.0), (1.0, 6.0)])
    def test_no_corners_on_any_ellipse(self, a, b):
        assert len(detect_corners(ellipse(a, b))) == 0

    def test_naca_has_exactly_one_corner_and_it_is_the_trailing_edge(self, naca):
        """The regression. High curvature at the LE must not read as a corner."""
        c = detect_corners(naca)
        assert len(c) == 1, f"expected only the TE, got {naca.points[c.index]}"
        x_te = naca.points[c.index[0], 0]
        assert x_te > 0.95, "the detected corner is not at the trailing edge"

    def test_leading_edge_curvature_would_fool_a_fixed_threshold(self, naca):
        """Documents *why* the extrapolation is needed, not just that it works."""
        from aeromesh.geometry.corners import _turning_interpolator

        phi = _turning_interpolator(naca)
        s_le = naca.arclength[int(np.argmin(naca.points[:, 0]))]
        w = 0.02 * naca.perimeter
        raw = abs(phi(np.array([s_le + w / 2]))[0] - phi(np.array([s_le - w / 2]))[0])
        assert np.degrees(raw) > 30.0        # a naive detector sees a big turn here
        assert s_le not in naca.arclength[detect_corners(naca).index]


class TestMetamorphic:
    """A rigid motion or a uniform scaling cannot change what is a corner."""

    @pytest.mark.parametrize("kw", [
        dict(angle=0.7),
        dict(dx=-12.0, dy=4.5),
        dict(scale=37.0),
        dict(scale=0.013, angle=-2.1, dx=3.0, dy=-9.0),
    ])
    def test_corner_set_is_invariant(self, kw):
        base = load_airfoil("2412", n_points=400).resample(n=1200)
        moved = base.transform(**kw)
        a, b = detect_corners(base), detect_corners(moved)
        assert len(a) == len(b)
        np.testing.assert_allclose(np.sort(a.turn), np.sort(b.turn), atol=1e-6)

    def test_mirroring_a_symmetric_section_reproduces_it(self):
        base = load_airfoil("0012", n_points=400).resample(n=1200)
        mirrored = Loop.from_points(base.points * np.array([1.0, -1.0]))
        a, b = detect_corners(base), detect_corners(mirrored)
        assert len(a) == len(b)
        np.testing.assert_allclose(np.sort(a.turn), np.sort(b.turn), atol=1e-3)


class TestFluidAngle:
    def test_hole_and_outer_conventions_are_complementary(self):
        turn = np.array([np.pi / 2])
        assert fluid_interior_angle(turn, is_hole=False)[0] == pytest.approx(np.pi / 2)
        assert fluid_interior_angle(turn, is_hole=True)[0] == pytest.approx(3 * np.pi / 2)

    def test_a_cusp_on_a_body_gives_a_full_turn_of_fluid(self):
        assert fluid_interior_angle(np.array([np.pi]), is_hole=True)[0] == \
            pytest.approx(2 * np.pi)
