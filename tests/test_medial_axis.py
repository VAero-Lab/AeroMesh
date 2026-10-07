"""Invariant tests for the medial engine (S1).

The headline gate is the retract identity: the medial axis deformation-retracts
onto its region, so the cycle count of the exterior axis must equal the number
of bodies, exactly, for every configuration. Everything else is accuracy against
closed-form skeletons and the vertex typing Rigby's rules are keyed on.

Note: this file replaces the S0-era tests of the superseded ``medial/cdt.py``,
which are preserved in ``test_medial_legacy.py``.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.geometry.loop import Loop
from aeromesh.geometry.region import Region, prepare_boundary
from aeromesh.medial.axis import EdgeKind, VertexKind, exterior_axis, interior_axis
from aeromesh.medial.fields import (
    optimum_flow_index,
    segment_distance,
    touch_groups,
)


def circle(r=1.0, n=1400, cx=0.0, cy=0.0, name="circle"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]), name)


def ellipse(a, b, cx=0.0, cy=0.0, n=3000, name="ellipse"):
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
        "fuselage": ellipse(0.55, 0.85, 0.45, -1.55, n=500, name="fuselage"),
    }


# ═══════════════════════════════════════════════════════════════════
#  Fields
# ═══════════════════════════════════════════════════════════════════

class TestFields:
    def test_segment_distance_beats_sample_distance(self):
        """Measuring to segments, not samples, is what keeps r_m within 1%."""
        from scipy.spatial import cKDTree

        c = circle(1.0, n=200)
        pts = c.points
        lid = np.zeros(len(pts), int)
        off = np.array([0, len(pts)])
        q = np.array([[0.0, 0.0]])
        d_seg = segment_distance(q, pts, lid, off)[0]
        assert d_seg == pytest.approx(np.cos(np.pi / 200), rel=1e-6)
        assert cKDTree(pts).query(q)[0][0] == pytest.approx(1.0)

    def test_touch_groups_merge_runs_and_the_wrap(self):
        lid = np.zeros(600, int)
        off = np.array([0, 600])
        g = np.array([0, 1, 2, 300, 301, 598, 599])
        groups = touch_groups(g, lid, off)
        assert len(groups) == 2
        assert sorted(len(x) for x in groups) == [2, 5]

    @pytest.mark.parametrize("theta_deg,expected", [
        (180, 0), (170, 0), (140, 0), (134, 1), (90, 1), (46, 1), (44, 2), (0, 2)])
    def test_optimum_flow_index(self, theta_deg, expected):
        assert int(optimum_flow_index(np.deg2rad(theta_deg))) == expected


# ═══════════════════════════════════════════════════════════════════
#  Closed-form skeletons
# ═══════════════════════════════════════════════════════════════════

class TestAnalytic:
    def test_annulus_radius_and_r_m(self):
        """The medial axis of an annulus is the circle of mean radius, with
        r_m = half the gap. Both closed form."""
        R1, R2 = 1.0, 3.0
        region = Region.build(circle(R2, 2400, name="outer"),
                              [circle(R1, 1400, name="inner")],
                              te="sharp", validate=False)
        ax = exterior_axis(region)
        pos = ax.sample_points()
        assert np.abs(np.hypot(pos[:, 0], pos[:, 1]) - 0.5 * (R1 + R2)).max() < 1e-4
        r_m = np.concatenate([d["r_m"] for _, _, d in ax.graph.edges(data=True)])
        assert np.abs(r_m - 0.5 * (R2 - R1)).max() / (0.5 * (R2 - R1)) < 0.01

    def test_ellipse_interior_axis_is_the_major_axis_segment(self):
        """For semi-axes a > b the axis is the segment |x| <= (a^2-b^2)/a on
        y = 0, with r_m = b at the centre."""
        a, b = 2.0, 1.0
        ax = interior_axis(ellipse(a, b))
        pos = ax.sample_points()
        assert np.abs(pos[:, 1]).max() < 1e-9
        extent = (a * a - b * b) / a
        assert abs(np.abs(pos[:, 0]).max() - extent) / extent < 0.01
        poly = np.concatenate([d["polyline"][:, 0] for _, _, d in ax.graph.edges(data=True)])
        r_m = np.concatenate([d["r_m"] for _, _, d in ax.graph.edges(data=True)])
        assert r_m[int(np.argmin(np.abs(poly)))] == pytest.approx(b, rel=1e-3)

    def test_a_circle_collapses_to_a_finite_contact_vertex(self):
        """Its medial axis is one point. A discrete Voronoi returns nothing, so
        the vertex is emitted explicitly with the angle Fogg's k consumes."""
        ax = interior_axis(circle(1.0, 1200))
        assert ax.n_vertices == 1 and ax.n_edges == 0
        d = next(iter(ax.graph.nodes(data=True)))[1]
        assert d["kind"] is VertexKind.FINITE_CONTACT
        assert d["r_m"] == pytest.approx(1.0, rel=1e-3)
        # At a finite-contact vertex theta_m is the *angular extent of contact*,
        # not the largest angle between two touches, and it is not capped at pi.
        # A full circle is in contact all the way round: 2*pi, giving k = -4,
        # the four superimposed negatives of the 0-sided template. A semicircular
        # end spans pi and gives k = -2 (Fogg, Fig. 10).
        assert d["theta_m"] == pytest.approx(2 * np.pi, abs=1e-6)
        assert -int(np.floor(d["theta_m"] / (np.pi / 2))) == -4


# ═══════════════════════════════════════════════════════════════════
#  The retract certificate
# ═══════════════════════════════════════════════════════════════════

CONFIGS = [
    ("1 body", ["main"]),
    ("2 bodies", ["main", "flap"]),
    ("3 bodies", ["main", "flap", "slat"]),
    ("airfoil + fuselage", ["main", "fuselage"]),
    ("bluff body alone", ["fuselage"]),
]
SHAPES = ["circle", "c", "box"]


def _outer(kind, group, scale=1.0):
    if kind == "circle":
        return am.circle_farfield(group, 4.0 * scale)
    if kind == "c":
        return am.c_farfield(group, 4.0 * scale, 6.0 * scale)
    return am.box_farfield(group, 4.0 * scale, 6.0 * scale, 4.0 * scale)


def _region(group, kind, scale=1.0):
    """Build a region whose length parameters scale with the bodies.

    ``farfield`` and ``te_thickness`` are absolute lengths, so a metamorphic
    test that scales the geometry has to scale them too -- otherwise it changes
    the problem rather than moving it.
    """
    return am.build_region(group, outer=_outer(kind, group, scale),
                           te_thickness=0.002 * scale)


class TestRetractCertificate:
    @pytest.mark.parametrize("kind", SHAPES)
    @pytest.mark.parametrize("name,keys", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_betti_one_equals_body_count(self, bodies, name, keys, kind):
        group = [bodies[k] for k in keys]
        region = am.build_region(group, outer=_outer(kind, group))
        ax = exterior_axis(region)
        cert = ax.certificate()
        assert cert["ok"], cert
        assert ax.betti_1 == len(keys)

    def test_an_interior_axis_has_no_cycles(self, bodies):
        for key in ("main", "flap", "fuselage"):
            ax = interior_axis(bodies[key].resample(n=1500))
            assert ax.betti_1 == 0


# ═══════════════════════════════════════════════════════════════════
#  Vertex and edge typing
# ═══════════════════════════════════════════════════════════════════

class TestTyping:
    @pytest.mark.parametrize("kind,n_flares", [("circle", 0), ("c", 2), ("box", 4)])
    def test_flare_count_follows_the_truncation_corners(self, bodies, kind, n_flares):
        """A corner of the fluid boundary generates a flare, and a flare will
        generate a block. This is the mechanism the far-field shape acts through."""
        group = [bodies["main"]]
        ax = exterior_axis(am.build_region(group, outer=_outer(kind, group)))
        assert ax.edge_kinds().get("FLARE", 0) == n_flares
        assert ax.kinds().get("CORNER", 0) == n_flares

    def test_a_smooth_farfield_around_one_body_is_a_bare_loop(self, bodies):
        """Rigby's doughnut: no medial vertices at all, hence one block wrapping
        on itself. Derived, not prescribed."""
        group = [bodies["main"]]
        ax = exterior_axis(am.build_region(group, outer=_outer("circle", group)))
        assert ax.edge_kinds() == {"LOOP": 1}
        assert ax.kinds() == {}, "a bare loop has no medial vertices"
        assert ax.n_real_vertices == 0
        assert ax.betti_1 == 1

    def test_flares_reach_radius_exactly_zero(self, bodies):
        group = [bodies["main"]]
        ax = exterior_axis(am.build_region(group, outer=_outer("box", group)))
        corners = [d for _, d in ax.graph.nodes(data=True)
                   if d["kind"] is VertexKind.CORNER]
        assert len(corners) == 4
        assert all(d["r_m"] == 0.0 for d in corners)

    def test_no_branch_of_radius_zero_anywhere_else(self, bodies):
        """The defect that made the first iteration unusable: a medial point
        whose maximal inscribed circle has zero radius is not on the axis."""
        for kind in SHAPES:
            group = [bodies["main"]]
            ax = exterior_axis(am.build_region(group, outer=_outer(kind, group)))
            n_zero = sum(int((d["r_m"] <= 1e-12).sum())
                         for _, _, d in ax.graph.edges(data=True))
            n_corner = ax.kinds().get("CORNER", 0)
            assert n_zero == n_corner

    def test_the_leading_edge_is_a_dangle_not_a_corner(self, bodies):
        b = prepare_boundary(bodies["main"], is_hole=True, te="blunt")
        ax = interior_axis(b.loop, corners=b.corners.positions(b.loop),
                           corner_idx=b.corners.index)
        dangles = [d for _, d in ax.graph.nodes(data=True)
                   if d["kind"] is VertexKind.DANGLE]
        assert len(dangles) == 1
        assert dangles[0]["pos"][0] < 0.1          # behind the leading edge
        assert dangles[0]["r_m"] > 0.01            # a real osculating radius

    def test_a_blunt_trailing_edge_gives_two_interior_flares(self, bodies):
        b = prepare_boundary(bodies["main"], is_hole=True, te="blunt")
        ax = interior_axis(b.loop, corners=b.corners.positions(b.loop),
                           corner_idx=b.corners.index)
        assert ax.kinds().get("CORNER", 0) == 2
        assert ax.edge_kinds().get("FLARE", 0) == 2


# ═══════════════════════════════════════════════════════════════════
#  Medial angle and the flow index
# ═══════════════════════════════════════════════════════════════════

class TestMedialAngle:
    @pytest.mark.parametrize("code", ["0012", "2412", "8412"])
    def test_interior_theta_spans_nose_to_midchord(self, code):
        """theta_m near 70 deg at the leading edge -- Fogg's n = 1, a nose
        singularity -- rising to ~180 deg over the mid-chord, where n = 0."""
        lp = am.load_airfoil(code, n_points=400).resample(n=1500)
        ax = interior_axis(lp)
        lo, hi = ax.theta_range()
        assert np.deg2rad(60) < lo < np.deg2rad(85)
        assert hi > np.deg2rad(175)
        hist = ax.flow_index_histogram()
        assert hist.get(1, 0) > 0 and hist.get(0, 0) > hist.get(1, 0)

    def test_the_exterior_loop_is_featureless_at_far_field_distance(self, bodies):
        """The measurement that motivates computing the interior axis at all."""
        group = [bodies["main"]]
        ax = exterior_axis(am.build_region(group, outer=am.circle_farfield(group, 15.0)))
        lo, hi = ax.theta_range()
        assert np.degrees(lo) > 170.0
        assert ax.flow_index_histogram() == {0: sum(
            len(d["theta_m"]) for _, _, d in ax.graph.edges(data=True))}

    def test_a_multi_element_slot_is_not(self, bodies):
        """theta_m collapses in the gap, so the flux balance will demand
        singularities there. Nothing is told a slot exists."""
        group = [bodies["main"], bodies["flap"]]
        ax = exterior_axis(am.build_region(group, outer=am.circle_farfield(group, 4.0)))
        lo, _ = ax.theta_range()
        assert np.degrees(lo) < 60.0
        assert sum(v for k, v in ax.flow_index_histogram().items() if k != 0) > 0


# ═══════════════════════════════════════════════════════════════════
#  Metamorphic
# ═══════════════════════════════════════════════════════════════════

class TestMetamorphic:
    """A rigid motion or a uniform scaling of the *whole problem* cannot change
    a skeleton's structure.

    The outer boundary has to be transformed along with the bodies. Rotating the
    bodies inside an axis-aligned box is not a rigid motion -- it is a different
    configuration, and its medial axis is legitimately wired differently.
    """

    @staticmethod
    def moved_region(base, kind, kw):
        from aeromesh.geometry.region import Region

        scale = kw.get("scale", 1.0)
        outer = _outer(kind, base)
        return Region.build(outer.transform(**kw),
                            [b.transform(**kw) for b in base],
                            te_thickness=0.002 * scale)

    @pytest.mark.parametrize("kw", [
        dict(angle=0.7), dict(dx=-12.0, dy=4.5), dict(scale=37.0),
        dict(scale=0.013, angle=-2.1, dx=3.0, dy=-9.0),
    ])
    @pytest.mark.parametrize("kind", ["circle", "box"])
    def test_structure_is_invariant(self, bodies, kw, kind):
        import networkx as nx

        base = [bodies["main"], bodies["flap"]]
        a = exterior_axis(_region(base, kind))
        b = exterior_axis(self.moved_region(base, kind, kw))
        assert a.certificate()["ok"] and b.certificate()["ok"]
        assert a.betti_1 == b.betti_1
        assert a.kinds() == b.kinds()
        assert a.edge_kinds() == b.edge_kinds()
        assert nx.is_isomorphic(a.graph, b.graph)

    @pytest.mark.parametrize("kw", [dict(scale=100.0), dict(scale=0.02)])
    def test_medial_angle_is_scale_invariant(self, bodies, kw):
        """theta_m is an angle: scaling the whole problem must not move it."""
        base = [bodies["main"]]
        a = exterior_axis(_region(base, "box"))
        b = exterior_axis(self.moved_region(base, "box", kw))
        np.testing.assert_allclose(a.theta_range(), b.theta_range(), atol=np.deg2rad(1))

    def test_flare_angle_is_exactly_the_corner_angle(self, bodies):
        """pi - alpha, taken from the corner rather than measured at the tip,
        where both touch points converge and the measurement is ill-conditioned."""
        base = [bodies["main"]]
        ax = exterior_axis(_region(base, "box"))
        tips = [d["theta_m"] for _, d in ax.graph.nodes(data=True)
                if d["kind"] is VertexKind.CORNER]
        assert len(tips) == 4
        np.testing.assert_allclose(np.degrees(tips), 90.0, atol=1e-9)
