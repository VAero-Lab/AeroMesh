"""Tests for the singularity solver and the index budget (S2).

What is verified here is the budget identity and the solver machinery. The flux
balance *at medial vertices* -- Fogg's second critical position class -- is not
yet solved, so the budget does not close on every configuration. That is
reported, not absorbed: the certificate is doing its job when it refuses.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.geometry.loop import Loop
from aeromesh.medial.axis import interior_axis
from aeromesh.topology.singularities import (
    Corner,
    Singularity,
    SingularityField,
    critical_crossings,
    finite_contact_singularities,
    flux_residual,
    merge,
    solve_singularities,
)


def ellipse(a, b, cx=0.0, cy=0.0, n=600, name="e"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


@pytest.fixture(scope="module")
def bodies():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    return {
        "main": main,
        "flap": main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02, dy=-0.10,
                               name="flap"),
        "fuselage": ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage"),
        "slat": main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02,
                               name="slat"),
    }


class TestBudgetIdentity:
    """sum_interior k = sum_corners (2 - n_c) - 4*chi.

    Derived from discrete Gauss-Bonnet for quad meshes, independently of the
    flux balance. Checked against configurations whose correct blocking is known.
    """

    @pytest.mark.parametrize("name,corner_nc,chi,expected", [
        ("square, regular grid", [1, 1, 1, 1], 1, 0),
        ("disk, 0-sided template", [], 1, -4),
        ("triangle, 3-sided template", [1, 1, 1], 1, -1),
        ("L-shape, submapped", [1] * 5 + [3], 1, 0),
        ("annulus, O-grid", [], 0, 0),
        ("regular hexagon", [1] * 6, 1, 2),
    ])
    def test_known_blockings(self, name, corner_nc, chi, expected):
        f = SingularityField(
            corners=[Corner(np.zeros(2), np.pi / 2, nc) for nc in corner_nc], chi=chi)
        assert f.k_required == expected, name

    def test_it_reproduces_foggs_hexagon(self):
        """Fogg states k = 2 at the centre of a regular hexagon. The identity was
        not derived from that case, so this is a genuine prediction."""
        f = SingularityField(corners=[Corner(np.zeros(2), 2 * np.pi / 3, 1)] * 6, chi=1)
        assert f.k_required == 2


class TestFluxResidual:
    def test_vanishes_where_the_flow_index_is_exact(self):
        for theta in (0.0, np.pi / 2, np.pi):
            assert flux_residual(theta) == pytest.approx(0.0, abs=1e-12)

    def test_is_bounded_by_a_quarter_turn(self):
        theta = np.linspace(0.0, np.pi, 2001)
        assert np.abs(flux_residual(theta)).max() <= np.pi / 4 + 1e-12

    def test_is_maximal_at_the_critical_angles(self):
        for c in (np.pi / 4, 3 * np.pi / 4):
            assert abs(flux_residual(c)) == pytest.approx(np.pi / 4, abs=1e-12)


class TestMerging:
    def test_a_cancelling_pair_disappears(self):
        a = Singularity(np.array([0.0, 0.0]), +1, "crossing", r_m=1.0)
        b = Singularity(np.array([0.1, 0.0]), -1, "crossing", r_m=1.0)
        assert merge([a, b], factor=0.75) == []

    def test_same_sign_singularities_combine(self):
        a = Singularity(np.array([0.0, 0.0]), -1, "crossing", r_m=1.0)
        b = Singularity(np.array([0.1, 0.0]), -1, "crossing", r_m=1.0)
        out = merge([a, b], factor=0.75)
        assert len(out) == 1 and out[0].k == -2 and out[0].n_merged == 2

    def test_distant_singularities_are_left_alone(self):
        a = Singularity(np.array([0.0, 0.0]), -1, "crossing", r_m=0.1)
        b = Singularity(np.array([5.0, 0.0]), -1, "crossing", r_m=0.1)
        assert len(merge([a, b], factor=0.75)) == 2

    def test_merging_conserves_the_total(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, size=(30, 2))
        ks = rng.choice([-1, 1], size=30)
        sings = [Singularity(p, int(k), "crossing", r_m=0.3) for p, k in zip(pts, ks)]
        assert sum(s.k for s in merge(sings, factor=0.75)) == int(ks.sum())


class TestFiniteContact:
    def test_a_circle_gives_four_negative_singularities(self):
        """Contact spans 2*pi, so k = -floor(2*pi / (pi/2)) = -4: the four
        superimposed negatives of the 0-sided template."""
        ax = interior_axis(ellipse(1.0, 1.0, n=1200, name="circle"))
        out = finite_contact_singularities(ax)
        assert len(out) == 1
        assert out[0].k == -4
        assert out[0].source == "finite_contact"

    def test_it_matches_the_budget_for_a_disk(self):
        ax = interior_axis(ellipse(1.0, 1.0, n=1200, name="circle"))
        f = SingularityField(singularities=finite_contact_singularities(ax),
                             corners=[], chi=1)
        assert f.budget()["ok"], f.budget()


class TestCrossingSign:
    """The sign at a critical crossing, pinned by two closed-form checks.

    The thin-sliver derivation suggests a signed ``k = -dn``, but ``dn`` reverses
    with the direction of travel along the edge. These two tests rule it out and
    select ``k = -|dn|``.
    """

    @staticmethod
    def dn_sequence(a):
        from aeromesh.medial.fields import optimum_flow_index

        ax = interior_axis(ellipse(a, 1.0, n=4000))
        dns = []
        for _, _, d in ax.graph.edges(data=True):
            n = optimum_flow_index(d["theta_m"])
            dns += [int(n[j + 1] - n[j]) for j in np.flatnonzero(np.diff(n) != 0)]
        return dns

    @pytest.mark.parametrize("a", [3.0, 2.0, 1.5, 1.2, 1.05, 1.01])
    def test_the_signed_rules_are_ruled_out_at_every_aspect_ratio(self, a):
        """The dn sequence is always [-1, -1, +1, +1], which sums to zero under
        either signed rule and to -4 under -|dn|. The budget requires -4."""
        dns = self.dn_sequence(a)
        assert sorted(dns) == [-1, -1, 1, 1]
        assert -sum(dns) == 0 and sum(dns) == 0          # both signed rules
        assert -sum(abs(x) for x in dns) == -4           # -|dn|

    @pytest.mark.parametrize("a", [3.0, 2.0, 1.5, 1.2, 1.05])
    def test_an_elongated_ellipse_is_resolved_by_crossings(self, a):
        ax = interior_axis(ellipse(a, 1.0, n=4000))
        found = critical_crossings(ax)
        assert len(found) == 4 and sum(s.k for s in found) == -4
        assert SingularityField(singularities=found, corners=[], chi=1).budget()["ok"]

    def test_a_circle_is_resolved_by_finite_contact_to_the_same_total(self):
        """Two entirely separate routes to the same -4."""
        assert sum(s.k for s in finite_contact_singularities(
            interior_axis(ellipse(1.0, 1.0, n=2000)))) == -4

    @pytest.mark.parametrize("a", [1.01])
    def test_the_transition_band_is_a_known_defect(self, a):
        """Documents an open gap rather than hiding it.

        Right at the circular limit both mechanisms are in play: the ends have
        enough contact to be promoted to finite-contact vertices, which guards
        their crossings out, but not enough extent for
        ``-floor(extent/(pi/2))`` to reach -2 apiece. The total comes to -2
        where the budget needs -4.

        The band is narrow -- every aspect ratio from 3:1 down to 1.05:1 is
        exact by crossings, and the circle itself is exact by finite contact.
        Only the handover between the two mechanisms is wrong.
        """
        from aeromesh.topology.singularities import vertex_singularities

        ax = interior_axis(ellipse(a, 1.0, n=4000))
        total = (sum(s.k for s in critical_crossings(ax))
                 + sum(s.k for s in finite_contact_singularities(ax))
                 + sum(s.k for s in vertex_singularities(ax)))
        assert total == -2 != -4


class TestSolver:
    def test_a_bluff_body_needs_and_finds_no_singularities(self, bodies):
        group = [bodies["fuselage"]]
        region = am.build_region(group, outer=am.circle_farfield(group, 4.0))
        f = solve_singularities(am.exterior_axis(region), region)
        assert f.k_required == 0
        assert f.singularities == []
        assert f.budget()["ok"]

    def test_a_multi_element_slot_produces_crossings(self, bodies):
        """theta_m collapsing in the gap drives class-1 crossings. The budget
        does not close here because the configuration has medial vertices."""
        group = [bodies["main"], bodies["flap"]]
        region = am.build_region(group, outer=am.circle_farfield(group, 4.0))
        axis = am.exterior_axis(region)
        f = solve_singularities(axis, region)
        assert any(s.source == "crossing" for s in f.singularities)
        assert axis.n_real_vertices == 2
        assert abs(f.budget()["residual"]) <= 2

    def test_a_singularity_lands_in_the_slot(self, bodies):
        """Nothing tells the solver a slot exists; theta_m collapsing there does.

        The slot singularity must be close to *both* elements -- that is what
        distinguishes it from one out on the far ring.
        """
        main, flap = bodies["main"], bodies["flap"]
        region = am.build_region([main, flap], outer=am.circle_farfield([main, flap], 4.0))
        f = solve_singularities(am.exterior_axis(region), region, do_merge=False)
        assert f.singularities
        d = [max(np.linalg.norm(main.points - s.position, axis=1).min(),
                 np.linalg.norm(flap.points - s.position, axis=1).min())
             for s in f.singularities]
        assert min(d) < 0.25, f"no singularity in the slot; closest {min(d):.3f}"

    def test_an_airfoil_in_a_smooth_far_field_balances(self, bodies):
        """Its blunt trailing edge has two corners at n_c = 3, so the budget
        requires sum k = -2, and the two concave-corner switches supply it.
        No medial vertices, so nothing is missing."""
        group = [bodies["main"]]
        region = am.build_region(group, outer=am.circle_farfield(group, 4.0))
        axis = am.exterior_axis(region)
        assert axis.n_real_vertices == 0
        f = solve_singularities(axis, region)
        assert f.budget()["k_required"] == -2
        assert f.budget()["ok"], f.budget()

    def test_the_certificate_reports_rather_than_absorbs(self, bodies):
        """A field that does not close must say so, with the shortfall stated.
        Absorbing it would certify nothing."""
        group = [bodies["main"]]
        region = am.build_region(group, outer=am.c_farfield(group, 4.0, 6.0))
        b = solve_singularities(am.exterior_axis(region), region).budget()
        assert not b["ok"]
        assert b["residual"] == b["k_total"] - b["k_required"]
        assert b["residual"] != 0

    def test_every_vertex_free_configuration_balances(self, bodies):
        """The whole remaining gap is medial vertices, and this pins that down."""
        for group in ([bodies["fuselage"]], [bodies["main"]]):
            region = am.build_region(group, outer=am.circle_farfield(group, 4.0))
            axis = am.exterior_axis(region)
            assert axis.n_real_vertices == 0
            assert solve_singularities(axis, region).budget()["ok"]

    def test_corner_records_carry_the_element_count(self, bodies):
        group = [bodies["main"]]
        region = am.build_region(group, outer=am.box_farfield(group, 4.0, 6.0, 4.0))
        f = solve_singularities(am.exterior_axis(region), region)
        nc = sorted(c.n_c for c in f.corners)
        assert nc == [1, 1, 1, 1, 3, 3]       # four box corners, two blunt-TE corners
        assert sum(c.index_contribution for c in f.corners) == 2


class TestVertexClass:
    """Class 2: flux balance at medial vertices.

    ``k_V = sum_j (2 - n_j) - 4`` over the incident medial edges. Each cut across
    an incident edge carries ``n_j`` quarter-turns of the cross field and enters
    the balance exactly as a boundary corner's ``n_c`` does, so the accounting
    that gives the global budget gives this for the disc around one vertex.

    It reduces to ``m - 4`` when every incident edge has ``n_j = 1``, which is
    why that simpler form worked on regular polygons and failed on a rectangle,
    whose central medial edge has ``n = 0``.
    """

    @staticmethod
    def poly(pts, n=3000):
        return Loop.from_points(np.asarray(pts, float), "poly").resample(n=n)

    @classmethod
    def regular(cls, k, n=3000):
        t = np.linspace(0, 2 * np.pi, k, endpoint=False)
        return cls.poly(np.column_stack([np.cos(t), np.sin(t)]), n)

    @staticmethod
    def stadium(n=4000):
        th = np.linspace(-np.pi / 2, np.pi / 2, 400)
        return Loop.from_points(np.vstack([
            np.column_stack([1 + np.cos(th), np.sin(th)]),
            np.column_stack([-1 - np.cos(th), -np.sin(th)])]),
            "stadium").resample(n=n)

    @staticmethod
    def solve_interior(loop):
        """Budget and solved total for a simply connected body, chi = 1."""
        from aeromesh.geometry.corners import detect_corners, fluid_interior_angle
        from aeromesh.medial.axis import interior_axis
        from aeromesh.topology.singularities import vertex_singularities

        c = detect_corners(loop)
        n_c = np.rint(fluid_interior_angle(c.turn, is_hole=False) / (np.pi / 2))
        required = int(sum(2 - x for x in n_c)) - 4
        ax = interior_axis(loop, corners=c.positions(loop) if len(c) else None,
                           corner_idx=c.index if len(c) else None)
        total = (sum(s.k for s in critical_crossings(
                     ax, c.positions(loop) if len(c) else np.empty((0, 2)),
                     tol=5.0 * loop.perimeter / loop.n_points))
                 + sum(s.k for s in finite_contact_singularities(ax))
                 + sum(s.k for s in vertex_singularities(ax)))
        return required, total

    def test_triangle(self):
        assert self.solve_interior(self.poly([[0, 0], [1, 0], [0.5, 0.9]])) == (-1, -1)

    def test_square(self):
        assert self.solve_interior(self.poly([[0, 0], [1, 0], [1, 1], [0, 1]])) == (0, 0)

    def test_pentagon(self):
        assert self.solve_interior(self.regular(5)) == (1, 1)

    def test_hexagon_reproduces_foggs_value(self):
        """Fogg states k = +2 at the centre of a regular hexagon. Here it comes
        out of the medial axis of an actual hexagon, not out of the identity."""
        required, total = self.solve_interior(self.regular(6))
        assert required == total == 2

    def test_rectangle(self):
        """The case that ruled out ``m - 4``: its central medial edge has n = 0,
        so two m = 3 vertices must give 0 rather than -1 each."""
        assert self.solve_interior(self.poly([[0, 0], [2, 0], [2, 1], [0, 1]])) == (0, 0)

    def test_stadium(self):
        """Fogg's Fig. 10 case: each semicircular end has contact over pi and
        gives k = -2, which is the whole budget."""
        assert self.solve_interior(self.stadium()) == (-4, -4)

    def test_l_shape_is_off_by_one(self):
        """Documents the remaining gap rather than hiding it. One of the three
        vertices reads an incident flow index of 0 where the balance needs 1."""
        required, total = self.solve_interior(
            self.poly([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]))
        assert required == 0
        assert total == 1


class TestConfigurations:
    """The budget on the configurations the library actually targets."""

    @staticmethod
    def field(bodies, outer):
        region = am.build_region(bodies, outer=outer)
        return solve_singularities(am.exterior_axis(region), region)

    def test_a_bluff_body_balances(self, bodies):
        g = [bodies["fuselage"]]
        assert self.field(g, am.circle_farfield(g, 4.0)).budget()["ok"]

    def test_an_airfoil_in_a_smooth_far_field_balances(self, bodies):
        g = [bodies["main"]]
        f = self.field(g, am.circle_farfield(g, 4.0))
        assert f.budget()["k_required"] == -2      # its blunt trailing edge
        assert f.budget()["ok"], f.budget()

    def test_an_airfoil_in_a_box_balances(self, bodies):
        """Four flares from the box corners plus the trailing edge, and the
        vertex balance closes it."""
        g = [bodies["main"]]
        f = self.field(g, am.box_farfield(g, 4.0, 6.0, 4.0))
        assert f.budget()["k_required"] == 2
        assert f.budget()["ok"], f.budget()

    def test_a_three_element_configuration_balances(self, bodies):
        g = [bodies["main"], bodies["flap"], bodies["slat"]]
        f = self.field(g, am.circle_farfield(g, 4.0))
        assert f.budget()["ok"], f.budget()

    @pytest.mark.parametrize("name,keys,outer", [
        ("C-shape", ["main"], "c"),
        ("main+flap circle", ["main", "flap"], "circle"),
        ("main+flap box", ["main", "flap"], "box"),
    ])
    def test_the_remaining_cases_are_close_but_open(self, bodies, name, keys, outer):
        """Documents what is still unbalanced, and by how little."""
        g = [bodies[k] for k in keys]
        o = {"circle": am.circle_farfield(g, 4.0),
             "c": am.c_farfield(g, 4.0, 6.0),
             "box": am.box_farfield(g, 4.0, 6.0, 4.0)}[outer]
        b = self.field(g, o).budget()
        assert not b["ok"]
        assert abs(b["residual"]) <= 2, f"{name}: residual {b['residual']}"
