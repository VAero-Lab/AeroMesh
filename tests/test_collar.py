"""Tests for the near-wall collar -- zone 1 of a zoned blocking.

The collar exists because the general medial decomposition is not a
boundary-layer method. Asked for a near-wall band it returned four blocks of
unequal length with a 262 degree interior angle, an aspect ratio of 402 set by
the blocking rather than by Delta_y1, and interior block edges turning through
374 degrees.

Nothing here is selected. A thin band around a simple closed curve is an
annulus; an annulus is an O-grid because there is nothing else it could be. The
cuts are the grid lines a reflex wall corner demands, ``n_c - 1`` of them at the
cross directions, so a blunt trailing edge (two corners at 261.8 degrees,
n_c = 3) gives four blocks and a sharp one (a single corner at 343.5 degrees,
n_c = 4) gives three -- from the same code, with no branch on which it is.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.collar import (build_collar, cell_quality, mesh_collar,
                                      nearest_on_polyline, smooth_offset, tfi,
                                      validate_mesh)

DELTA, RHO_MAX = 0.02, 0.05


@pytest.fixture(scope="module")
def cases():
    out = {}
    for te in ("blunt", "sharp"):
        for digits in ("2412", "4412", "0012"):
            body = am.load_airfoil(digits, n_points=400).transform(name="main")
            region = am.build_region([body],
                                     outer=am.circle_farfield([body], 8.0),
                                     te=te)
            collar = build_collar(region, delta_target=DELTA, rho_max=RHO_MAX)
            out[(te, digits)] = (region, collar)
    return out


ALL = [(te, d) for te in ("blunt", "sharp") for d in ("2412", "4412", "0012")]


class TestTransfiniteInterpolation:
    """A bilinear patch must reproduce its own four corners. Two of them were
    wrong by exactly 1.0 on a unit test quad, because the correction terms at
    (1,1) and (0,1) were swapped -- `T` is the opposite side read backwards, so
    its *start* is the (1,1) corner, not its end."""

    @staticmethod
    def _quad(n=40):
        t = np.linspace(0, 1, n)
        return (np.column_stack([t, 0.3 * np.sin(np.pi * t)]),
                np.column_stack([np.ones_like(t), t]),
                np.column_stack([1 - t, 1 + 0.3 * np.sin(np.pi * t)]),
                np.column_stack([np.zeros_like(t), 1 - t]))

    def test_every_corner_is_reproduced(self):
        s = self._quad()
        g = tfi(s, 9, 9)
        np.testing.assert_allclose(g[0, 0], s[0][0], atol=1e-12)
        np.testing.assert_allclose(g[-1, 0], s[0][-1], atol=1e-12)
        np.testing.assert_allclose(g[-1, -1], s[1][-1], atol=1e-12)
        np.testing.assert_allclose(g[0, -1], s[2][-1], atol=1e-12)

    def test_every_edge_lies_on_its_side(self):
        s = self._quad()
        g = tfi(s, 21, 21)
        for row, side, rev in ((g[:, 0], s[0], False), (g[-1, :], s[1], False),
                               (g[:, -1], s[2], True), (g[0, :], s[3], True)):
            from shapely.geometry import LineString, Point
            line = LineString(side)
            assert max(Point(p).distance(line) for p in row) < 1e-9

    def test_a_rectangle_maps_to_itself(self):
        t = np.linspace(0, 1, 11)
        s = (np.column_stack([2 * t, np.zeros_like(t)]),
             np.column_stack([2 * np.ones_like(t), t]),
             np.column_stack([2 - 2 * t, np.ones_like(t)]),
             np.column_stack([np.zeros_like(t), 1 - t]))
        g = tfi(s, 5, 4)
        np.testing.assert_allclose(g[:, 0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(g[:, -1, 1], 1.0, atol=1e-12)
        np.testing.assert_allclose(g[0, :, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(g[-1, :, 0], 2.0, atol=1e-12)


class TestSmoothOffset:
    """Smoothing and offsetting fight each other and neither works alone:
    offsetting per sample gives a curve whose segments span orders of
    magnitude, and smoothing shortens it onto the wall."""

    def test_the_curve_is_at_the_distance_asked_for(self, cases):
        region, collar = cases[("blunt", "2412")]
        d = nearest_on_polyline(collar["outer"], collar["wall"])[1]
        assert abs(d.max() - DELTA) / DELTA < 0.02
        assert abs(d.min() - DELTA) / DELTA < 0.02

    def test_the_curve_is_evenly_sampled(self, cases):
        _, collar = cases[("blunt", "2412")]
        seg = np.linalg.norm(np.roll(collar["outer"], -1, axis=0)
                             - collar["outer"], axis=1)
        assert seg.max() / seg.min() < 1.5

    def test_it_is_smoother_than_the_raw_offset(self, cases):
        from aeromesh.blocking.collar import outward_normals
        _, collar = cases[("blunt", "2412")]
        wall, delta = collar["wall"], collar["delta"]

        def turning(c):
            v = np.roll(c, -1, axis=0) - c
            v /= np.maximum(np.linalg.norm(v, axis=1), 1e-300)[:, None]
            return np.degrees(np.arccos(np.clip(
                np.sum(np.roll(v, 1, axis=0) * v, axis=1), -1, 1))).max()

        raw = wall + delta[:, None] * outward_normals(wall)
        assert turning(collar["outer"]) < turning(raw)


class TestCollarStructure:
    @pytest.mark.parametrize("te,digits", ALL, ids=[f"{a} {b}" for a, b in ALL])
    def test_the_blocks_tile_the_annulus(self, cases, te, digits):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        _, collar = cases[(te, digits)]
        polys = []
        for sides in collar["blocks"]:
            o = np.vstack([sides[0]] + [s[1:] for s in sides[1:]])
            p = Polygon(o)
            polys.append(p if p.is_valid else p.buffer(0))
        annulus = (Polygon(collar["outer"])
                   .difference(Polygon(collar["wall"])))
        union = unary_union(polys)
        assert annulus.difference(union).area / annulus.area < 1e-9
        assert abs(sum(p.area for p in polys)
                   - union.area) / annulus.area < 1e-9

    def test_a_blunt_trailing_edge_gives_two_bands_and_two_wedges(self, cases):
        """Two corners at 261.8 degrees, n_c = 3, so two rays each."""
        region, collar = cases[("blunt", "2412")]
        assert len(region.boundaries[1].corners) == 2
        assert list(region.boundaries[1].corner_elements) == [3, 3]
        assert len(collar["rays"]) == 4
        assert collar["kinds"].count("band") == 2
        assert collar["kinds"].count("wedge") == 2

    def test_a_sharp_trailing_edge_gives_one_band_and_two_wedges(self, cases):
        """One corner at 343.5 degrees, n_c = 4, so three rays -- and the band
        wraps the whole aerofoil, which index comparison alone cannot tell from
        a wedge because every ray leaves the same point."""
        region, collar = cases[("sharp", "2412")]
        assert len(region.boundaries[1].corners) == 1
        assert list(region.boundaries[1].corner_elements) == [4]
        assert len(collar["rays"]) == 3
        assert collar["kinds"].count("band") == 1
        assert collar["kinds"].count("wedge") == 2

    @pytest.mark.parametrize("te,digits", ALL, ids=[f"{a} {b}" for a, b in ALL])
    def test_the_corner_rays_are_straight(self, cases, te, digits):
        _, collar = cases[(te, digits)]
        for sides, tags in zip(collar["blocks"], collar["labels"]):
            for side, tag in zip(sides, tags):
                if tag != "ray":
                    continue
                assert len(side) == 2

    @pytest.mark.parametrize("te,digits", ALL, ids=[f"{a} {b}" for a, b in ALL])
    def test_the_block_angles_are_near_right(self, cases, te, digits):
        _, collar = cases[(te, digits)]
        worst_lo, worst_hi = 360.0, 0.0
        for sides in collar["blocks"]:
            m = len(sides)
            for i in range(m):
                a, c = sides[i], sides[(i + 1) % m]
                u, v = a[-2] - a[-1], c[1] - c[0]
                ang = (-np.degrees(np.arctan2(
                    float(u[0] * v[1] - u[1] * v[0]),
                    float(np.dot(u, v))))) % 360.0
                worst_lo, worst_hi = min(worst_lo, ang), max(worst_hi, ang)
        assert worst_lo > 60.0
        assert worst_hi < 120.0


class TestCollarMesh:
    """The checks that were missing. Every per-patch measure passed on a mesh
    that was folded over itself and covered 204 per cent of the annulus."""

    @pytest.mark.parametrize("te,digits", ALL, ids=[f"{a} {b}" for a, b in ALL])
    def test_the_mesh_fills_the_collar_exactly_once(self, cases, te, digits):
        _, collar = cases[(te, digits)]
        grids = mesh_collar(collar, n_across=16, target_spacing=0.02,
                            first_cell=1e-4)
        v = validate_mesh(collar, grids)
        assert abs(v["coverage"] - 1.0) < 1e-3, v
        assert abs(v["overlap"]) < 1e-6, v
        assert v["outside"] < 1e-6, v
        assert v["ok"], v

    @pytest.mark.parametrize("te,digits", ALL, ids=[f"{a} {b}" for a, b in ALL])
    def test_every_cell_has_positive_area(self, cases, te, digits):
        _, collar = cases[(te, digits)]
        grids = mesh_collar(collar, n_across=16, target_spacing=0.02,
                            first_cell=1e-4)
        assert all(cell_quality(g)["all_positive"] for g in grids)

    def test_the_first_cell_height_is_what_was_asked_for(self, cases):
        _, collar = cases[("blunt", "2412")]
        for want in (1e-4, 1e-5):
            grids = mesh_collar(collar, n_across=24, target_spacing=0.01,
                                first_cell=want)
            got = min(float(np.min(nearest_on_polyline(g[:, 1, :],
                                                       collar["wall"])[1]))
                      for g in grids if g.shape[1] > 2)
            assert 0.5 * want < got < 2.0 * want, (want, got)

    def test_the_aspect_ratio_follows_delta_y1_not_the_blocking(self, cases):
        """The old general blocking produced an aspect ratio of 402 because of
        where its block boundaries happened to fall -- a property of the
        decomposition, which is the wrong thing for it to depend on. Here it is
        a consequence of the wall spacing.

        Asserted in the thin-cell regime, where the scaling is clean: a factor
        of five in the first cell gives a factor of five in the aspect ratio
        (4.2e-5 -> 8.4e-6 gives 200 -> 1001). For *thick* first cells the
        maximum is set by the along-wall spacing instead and the relation
        flattens -- 1.67e-4 to 8.4e-5 moves it only 108 to 132 -- which is the
        expected behaviour and not what this test is about.
        """
        _, collar = cases[("blunt", "2412")]
        ars = []
        for first in (5e-5, 1e-5):
            grids = mesh_collar(collar, n_across=24, target_spacing=0.01,
                                first_cell=first)
            ars.append(max(cell_quality(g)["max_aspect"] for g in grids))
        assert ars[1] / ars[0] > 3.5, ars
