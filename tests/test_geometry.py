"""Tests for aeromesh.geometry (Module ①)."""

import numpy as np
import pytest

from aeromesh.geometry.boundary import Boundary, load_airfoil, _signed_area


class TestLoadAirfoil:
    """Test the convenience loader."""

    def test_naca_4digit(self):
        bnd = load_airfoil("0012")
        assert isinstance(bnd, Boundary)
        assert bnd.name == "NACA 0012"

    def test_naca_5digit(self):
        bnd = load_airfoil("23012")
        assert isinstance(bnd, Boundary)
        assert bnd.name == "NACA 23012"

    def test_from_dat_file(self, sd7037):
        assert isinstance(sd7037, Boundary)
        assert sd7037.n_points > 20

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="Cannot interpret"):
            load_airfoil("not_a_valid_thing_123456")


class TestBoundaryProperties:
    """Test that boundaries satisfy all required invariants."""

    def test_shape(self, naca0012):
        assert naca0012.points.ndim == 2
        assert naca0012.points.shape[1] == 2
        assert naca0012.n_points > 50

    def test_closed(self, naca0012):
        """First and last points must be identical."""
        np.testing.assert_array_equal(
            naca0012.points[0], naca0012.points[-1]
        )

    def test_ccw_winding(self, naca0012):
        """Signed area must be positive (CCW)."""
        area = _signed_area(naca0012.points)
        assert area > 0, f"Expected CCW (positive area), got {area}"

    def test_unit_chord(self, naca0012):
        """Chord should be normalized to 1.0."""
        x_range = naca0012.points[:, 0].max() - naca0012.points[:, 0].min()
        assert abs(x_range - 1.0) < 1e-10

    def test_le_at_origin(self, naca0012):
        """Leading edge should be at x = 0."""
        assert abs(naca0012.le[0]) < 1e-10

    def test_no_consecutive_duplicates(self, naca0012):
        """No two consecutive points should be closer than 1e-12."""
        diffs = np.linalg.norm(np.diff(naca0012.points, axis=0), axis=1)
        assert np.all(diffs > 1e-12)

    def test_naca0012_symmetric(self, naca0012):
        """NACA 0012 should be symmetric about y = 0."""
        # CCW winding: upper_surface = TE→LE (y >= 0),
        #              lower_surface = LE→TE (y <= 0)
        upper = naca0012.upper_surface
        lower = naca0012.lower_surface

        # Upper surface (TE → LE) should have y >= 0
        assert upper[:, 1].max() > 0.04, "Upper surface should have positive y"
        # Lower surface (LE → TE) should have y <= 0
        assert lower[:, 1].min() < -0.04, "Lower surface should have negative y"

    def test_sd7037_loads_clean(self, sd7037):
        """SD7037 should load without issues and be valid."""
        assert sd7037.n_points > 30
        area = _signed_area(sd7037.points)
        assert area > 0  # CCW
        np.testing.assert_array_equal(sd7037.points[0], sd7037.points[-1])

    def test_le_index_valid(self, naca0012):
        """LE index should point to the minimum-x point."""
        min_x_idx = np.argmin(naca0012.points[:, 0])
        assert naca0012.le_index == min_x_idx
