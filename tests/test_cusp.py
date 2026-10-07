"""Invariant tests for cusp opening (S0).

A cusp is the degenerate case for a medial axis. Opening it into a finite face
is what makes the problem well posed, so the face must be exact and the two
corners it creates must be symmetric for a symmetric tip -- an asymmetric face
changes the derived element count at the corner.
"""

import numpy as np
import pytest

from aeromesh.geometry.airfoil import load_airfoil
from aeromesh.geometry.corners import fluid_interior_angle, vertex_turns
from aeromesh.geometry.cusp import find_cusps, open_cusp
from aeromesh.geometry.loop import Loop


@pytest.fixture(params=["0012", "2412", "8412"])
def naca(request):
    return load_airfoil(request.param, n_points=400).resample(n=2000)


class TestCuspDetection:
    def test_every_naca_section_has_exactly_one_cusp(self, naca):
        assert len(find_cusps(naca)) == 1

    def test_a_circle_has_no_cusp(self):
        t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
        assert len(find_cusps(Loop.from_points(
            np.column_stack([np.cos(t), np.sin(t)])))) == 0

    def test_a_right_angle_is_not_a_cusp(self):
        sq = Loop.from_points([[0, 0], [1, 0], [1, 1], [0, 1]]).resample(n=400)
        assert len(find_cusps(sq)) == 0


class TestOpening:
    @pytest.mark.parametrize("thickness", [0.001, 0.002, 0.005, 0.01])
    def test_face_length_is_exact(self, naca, thickness):
        out, face = open_cusp(naca, int(find_cusps(naca)[0]), thickness)
        p0, p1 = out.points[face]
        assert np.linalg.norm(p1 - p0) == pytest.approx(thickness, rel=1e-6)

    def test_face_corners_are_symmetric(self, naca):
        out, face = open_cusp(naca, int(find_cusps(naca)[0]), 0.002)
        t = vertex_turns(out)[face]
        assert abs(t[0] - t[1]) < np.deg2rad(1.0)

    def test_turning_is_conserved(self, naca):
        """Opening a cusp splits its turn between two corners; it creates none."""
        before = vertex_turns(naca).sum()
        out, face = open_cusp(naca, int(find_cusps(naca)[0]), 0.002)
        assert vertex_turns(out).sum() == pytest.approx(before, abs=1e-9)

    def test_blunt_trailing_edge_gives_three_elements_per_corner(self, naca):
        """Fogg's n_c at each face corner, from the fluid interior angle."""
        out, face = open_cusp(naca, int(find_cusps(naca)[0]), 0.002)
        ang = fluid_interior_angle(vertex_turns(out)[face], is_hole=True)
        np.testing.assert_array_equal(np.rint(ang / (np.pi / 2)).astype(int), [3, 3])

    def test_opening_barely_changes_the_section(self, naca):
        out, _ = open_cusp(naca, int(find_cusps(naca)[0]), 0.002)
        assert out.area == pytest.approx(naca.area, rel=2e-3)

    def test_the_result_is_still_a_valid_ccw_loop(self, naca):
        out, _ = open_cusp(naca, int(find_cusps(naca)[0]), 0.002)
        assert out.area > 0
        assert out.n_points > 100

    def test_refuses_a_thickness_the_feature_cannot_reach(self, naca):
        with pytest.raises(ValueError, match="not a cusp"):
            open_cusp(naca, int(find_cusps(naca)[0]), thickness=5.0)

    def test_rejects_nonsense_arguments(self, naca):
        i = int(find_cusps(naca)[0])
        with pytest.raises(ValueError):
            open_cusp(naca, i, thickness=0.0)
        with pytest.raises(ValueError):
            open_cusp(naca, i, thickness=0.002, n_face=1)
