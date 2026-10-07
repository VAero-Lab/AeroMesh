"""Tests for the superseded medial extraction in ``aeromesh.medial.cdt``.

Preserved verbatim from ``tests/test_medial.py``. That module is replaced by
``aeromesh.medial.axis`` (S1) and is retained only until the repository is under
version control; see decision D7 in PROJECT_TRACKER.md. The S1 tests are in
``test_medial_axis.py``.
"""

import numpy as np
import pytest

from aeromesh.domain.outer import build_c_domain
from aeromesh.medial.cdt import (
    triangulate_domain,
    compute_circumcentres,
    extract_medial_axis,
)
from aeromesh.medial.graph import MedialGraph


class TestCDT:
    """Test the Constrained Delaunay Triangulation."""

    def test_triangulates(self, naca0012):
        domain = build_c_domain(naca0012, farfield=10.0)
        result = triangulate_domain(domain)
        assert "vertices" in result
        assert "triangles" in result
        assert len(result["triangles"]) > 100

    def test_no_inverted_triangles(self, naca0012):
        """All triangles should have positive area."""
        domain = build_c_domain(naca0012, farfield=10.0)
        result = triangulate_domain(domain)
        verts = result["vertices"]
        tris = result["triangles"]

        # Compute signed area of each triangle
        A = verts[tris[:, 0]]
        B = verts[tris[:, 1]]
        C = verts[tris[:, 2]]
        areas = 0.5 * ((B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1])
                        - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1]))
        assert np.all(areas > -1e-15), "Found inverted triangles"


class TestCircumcentres:
    """Test circumcentre computation."""

    def test_circumcentre_count(self, naca0012):
        domain = build_c_domain(naca0012, farfield=10.0)
        result = triangulate_domain(domain)
        cc = compute_circumcentres(result["vertices"], result["triangles"])
        assert cc.shape == (len(result["triangles"]), 2)


class TestRawMedialAxis:
    """Test the raw medial axis extraction."""

    def test_extracts_skeleton(self, naca0012):
        domain = build_c_domain(naca0012, farfield=10.0)
        result = triangulate_domain(domain)
        mg = extract_medial_axis(result, domain)

        # The new skeleton should be very sparse!
        # TopMaker style: just a few nodes and edges, not thousands
        assert mg.n_nodes > 0
        assert mg.n_nodes < 50  # Pruned and collapsed
        assert mg.n_edges > 0
        assert mg.n_edges < 50

        positions = mg.positions()
        assert positions.ndim == 2
        assert positions.shape[1] == 2
