"""Tests for decomposition splits (S3).

A split is a curve of the medial coordinate system, never a hand-placed
polyline. The property that matters is orthogonality: a medial radius meets the
boundary perpendicularly by definition, so a constant-s split is orthogonal to
the wall at both ends by construction, not by later optimisation.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.splits import (
    Split,
    best_split,
    candidate_splits,
    radius_split,
    split_rank,
)


@pytest.fixture(scope="module")
def cases():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    out = {}
    for label, outer in (("box", am.box_farfield([main], 4.0, 6.0, 4.0)),
                         ("circle", am.circle_farfield([main], 4.0))):
        region = am.build_region([main], outer=outer)
        out[label] = (region, am.exterior_axis(region))
    return out


class TestRanking:
    def test_n_zero_beats_n_one(self):
        """A straight-through split leaves no concave corner, so it wins."""
        assert split_rank(np.pi) < split_rank(np.pi / 2)

    def test_within_a_class_the_ideal_angle_wins(self):
        assert split_rank(np.pi) < split_rank(np.deg2rad(160))
        assert split_rank(np.pi / 2) < split_rank(np.deg2rad(110))

    def test_a_thin_slot_is_ranked_worst(self):
        """theta_m near zero means the flow reverses; splitting there is poor."""
        assert split_rank(np.deg2rad(20)) > split_rank(np.pi / 2) > split_rank(np.pi)

    def test_the_ideal_angles_score_exactly(self):
        assert split_rank(np.pi) == pytest.approx(0.0)
        assert split_rank(np.pi / 2) == pytest.approx(10.0)


class TestSplitGeometry:
    def test_a_split_runs_wall_to_wall_through_the_medial_point(self, cases):
        region, axis = cases["box"]
        for u, v, k in axis.graph.edges(keys=True):
            s = radius_split(axis.graph.edges[u, v, k], 5, (u, v, k))
            d = axis.graph.edges[u, v, k]
            np.testing.assert_allclose(s.curve[0], d["t1"][5], atol=1e-9)
            np.testing.assert_allclose(s.curve[-1], d["t2"][5], atol=1e-9)
            assert np.min(np.linalg.norm(s.curve - d["polyline"][5], axis=1)) < 1e-9

    def test_a_split_meets_the_wall_orthogonally_at_both_ends(self, cases):
        """The property the medial parameterisation exists to give. Checked
        against the actual boundary polyline, not against the construction.

        Touches landing on a C0 corner are excluded: the wall has no tangent
        there, so orthogonality is not defined. A radius ending at a corner is a
        flare, which is a different object.
        """
        region, axis = cases["box"]
        pts, _, offsets = region.sample()
        corners = {int(i) + int(o)
                   for b, o in zip(region.boundaries, offsets[:-1])
                   for i in b.corners.index}

        worst, checked = 0.0, 0
        for u, v, k in axis.graph.edges(keys=True):
            d = axis.graph.edges[u, v, k]
            for i in (3, len(d["polyline"]) // 2, len(d["polyline"]) - 4):
                sp = radius_split(d, i, (u, v, k))
                for end, j in ((sp.curve[:2], sp.touch_index[0]),
                               (sp.curve[:-3:-1], sp.touch_index[1])):
                    if any(abs(j - c) <= 3 for c in corners):
                        continue
                    ray = end[1] - end[0]
                    ray /= np.linalg.norm(ray)
                    tangent = pts[(j + 3) % len(pts)] - pts[(j - 3) % len(pts)]
                    tangent /= np.linalg.norm(tangent)
                    worst = max(worst, abs(float(ray @ tangent)))
                    checked += 1
        assert checked > 20
        assert worst < 0.08, f"worst |cos| between split and wall tangent = {worst}"

    def test_the_curve_is_resampled_for_meshing(self, cases):
        region, axis = cases["box"]
        u, v, k = next(iter(axis.graph.edges(keys=True)))
        s = radius_split(axis.graph.edges[u, v, k], 5, (u, v, k), n_curve=41)
        assert 38 <= len(s.curve) <= 44
        assert np.all(np.linalg.norm(np.diff(s.curve, axis=0), axis=1) > 0)

    def test_touch_indices_address_the_boundary(self, cases):
        region, axis = cases["box"]
        pts, _, _ = region.sample()
        for u, v, k in axis.graph.edges(keys=True):
            s = radius_split(axis.graph.edges[u, v, k], 5, (u, v, k))
            for end, j in ((s.curve[0], s.touch_index[0]),
                           (s.curve[-1], s.touch_index[1])):
                assert np.linalg.norm(pts[j] - end) < 0.05


class TestCandidates:
    def test_candidates_come_back_best_first(self, cases):
        region, axis = cases["box"]
        for u, v, k in axis.graph.edges(keys=True):
            ranks = [c.rank for c in candidate_splits(axis, (u, v, k), stride=7)]
            assert ranks == sorted(ranks)

    def test_a_flare_into_a_corner_ranks_worse_than_a_ring_edge(self, cases):
        """A flare has theta_m = pi/2 throughout, so every split on it is n = 1."""
        region, axis = cases["box"]
        flares, primaries = [], []
        for u, v, k in axis.graph.edges(keys=True):
            best = best_split(axis, (u, v, k))
            (flares if axis.graph.edges[u, v, k]["kind"].name == "FLARE"
             else primaries).append(best.rank)
        assert flares and primaries
        assert min(flares) > max(primaries)

    def test_the_margin_keeps_splits_off_the_vertices(self, cases):
        region, axis = cases["box"]
        for u, v, k in axis.graph.edges(keys=True):
            d = axis.graph.edges[u, v, k]
            for c in candidate_splits(axis, (u, v, k), stride=3, margin=0.1):
                assert 0.09 * d["length"] <= c.s <= 0.91 * d["length"]

    def test_every_edge_offers_a_split(self, cases):
        for label in ("box", "circle"):
            region, axis = cases[label]
            for u, v, k in axis.graph.edges(keys=True):
                assert best_split(axis, (u, v, k)) is not None
