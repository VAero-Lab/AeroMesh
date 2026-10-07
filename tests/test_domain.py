"""Tests for aeromesh.domain (Module ②)."""

import numpy as np
import pytest

from aeromesh.domain.outer import Domain, DomainType, build_domain, build_c_domain, build_o_domain, build_h_domain
from aeromesh.geometry.boundary import _signed_area


class TestCDomain:
    """Test the C-domain construction."""

    def test_builds_without_error(self, naca0012):
        domain = build_c_domain(naca0012, farfield=10.0, wake_length=3.0)
        assert isinstance(domain, Domain)
        assert domain.domain_type == DomainType.C

    def test_outer_boundary_has_points(self, naca0012):
        domain = build_c_domain(naca0012)
        assert len(domain.outer) > 20

    def test_outer_boundary_ccw(self, naca0012):
        domain = build_c_domain(naca0012)
        outer_closed = np.vstack([domain.outer, domain.outer[:1]])
        area = _signed_area(outer_closed)
        assert area > 0, f"Outer boundary should be CCW, got area={area}"

    def test_vertices_combined(self, naca0012):
        domain = build_c_domain(naca0012)
        n_air = naca0012.n_points - 1  # open polygon
        n_out = len(domain.outer)
        assert len(domain.vertices) == n_air + n_out

    def test_segments_close_both_loops(self, naca0012):
        domain = build_c_domain(naca0012)
        n_air = naca0012.n_points - 1
        n_out = len(domain.outer)
        # Total segments = airfoil loop + outer loop
        assert len(domain.segments) == n_air + n_out

    def test_markers_correct(self, naca0012):
        domain = build_c_domain(naca0012)
        n_air = naca0012.n_points - 1
        # Airfoil markers should be 10 (upper) or 11 (lower)
        air_markers = domain.markers[:n_air]
        assert np.all((air_markers == 10) | (air_markers == 11))
        # Outer markers should be >= 20
        assert np.all(domain.markers[n_air:] >= 20)

    def test_hole_point_inside_airfoil(self, naca0012):
        """The hole point should be inside the airfoil polygon."""
        from shapely.geometry import Point, Polygon

        domain = build_c_domain(naca0012)
        airfoil_poly = Polygon(naca0012.points)
        hp = Point(domain.hole_point[0], domain.hole_point[1])
        assert airfoil_poly.contains(hp)

    def test_triangle_input_format(self, naca0012):
        domain = build_c_domain(naca0012)
        tri_input = domain.to_triangle_input()
        assert "vertices" in tri_input
        assert "segments" in tri_input
        assert "holes" in tri_input
        assert tri_input["holes"].shape == (1, 2)

    def test_sd7037_domain(self, sd7037):
        """Domain should also work for cambered airfoils."""
        domain = build_c_domain(sd7037, farfield=12.0, wake_length=4.0)
        assert domain.domain_type == DomainType.C
        assert len(domain.vertices) > 50


class TestODomain:
    """Test the O-domain construction."""

    def test_builds_without_error(self, naca0012):
        domain = build_o_domain(naca0012, farfield=10.0)
        assert domain.domain_type == DomainType.O
        assert len(domain.outer) > 20

    def test_outer_boundary_ccw(self, naca0012):
        domain = build_o_domain(naca0012)
        outer_closed = np.vstack([domain.outer, domain.outer[:1]])
        assert _signed_area(outer_closed) > 0


class TestHDomain:
    """Test the H-domain construction."""

    def test_builds_without_error(self, naca0012):
        domain = build_h_domain(naca0012, farfield=10.0, wake_length=3.0)
        assert domain.domain_type == DomainType.H
        assert len(domain.outer) > 20

    def test_outer_boundary_ccw(self, naca0012):
        domain = build_h_domain(naca0012)
        outer_closed = np.vstack([domain.outer, domain.outer[:1]])
        assert _signed_area(outer_closed) > 0


class TestDomainDispatcher:
    """Test the generic build_domain function."""

    def test_dispatcher_c(self, naca0012):
        domain = build_domain(naca0012, domain_type="C")
        assert domain.domain_type == DomainType.C

    def test_dispatcher_o(self, naca0012):
        domain = build_domain(naca0012, domain_type=DomainType.O)
        assert domain.domain_type == DomainType.O

    def test_dispatcher_h(self, naca0012):
        domain = build_domain(naca0012, domain_type="h")
        assert domain.domain_type == DomainType.H

    def test_unsupported_type(self, naca0012):
        with pytest.raises(ValueError):
            build_domain(naca0012, domain_type="INVALID")
