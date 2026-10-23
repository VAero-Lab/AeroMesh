"""Tests for the realised index certificate, corner resolution and the
near-wall thickness law (S3 items D and C).

The certificate is the piece that makes the rest of these testable. For a
quadrilateral mesh,

    sum_interior (valence - 4) = sum_boundary (2 - q) - 4 * chi,

which is Euler's formula rearranged and therefore holds for *every* valid quad
mesh. That makes it useless as a prediction and valuable as a consistency test
of a blocking's own bookkeeping: anything that makes one block treat a point as
a corner while its neighbour runs past it shows up as a non-zero residual. It
resolves defects an area test cannot -- the crack it found on a fuselage with a
wing was 1e-7 of the region area, three orders below what the shapely tiling
check will flag.
"""

from collections import defaultdict

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.blocks import (clean_sides, local_spacing, roll_loop_edge)
from aeromesh.blocking.layers import medial_radius_on_boundary, wall_distance
from aeromesh.blocking.search import corner_cuts
from aeromesh.geometry.loop import Loop


def ellipse(a, b, cx=0.0, cy=0.0, n=400, name="e"):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Loop.from_points(
        np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)]), name)


@pytest.fixture(scope="module")
def bodies():
    main = am.load_airfoil("2412", n_points=400).transform(name="main")
    return {
        "main": main,
        "flap": main.transform(scale=0.30, angle=np.deg2rad(-25), dx=1.05,
                               dy=-0.08, name="flap"),
        "fuse": ellipse(1.6, 0.55, name="fuse"),
        "disc": ellipse(1.0, 1.0, name="disc"),
    }


CONFIGS = [
    ("aerofoil / box", ["main"], "box"),
    ("aerofoil / circle", ["main"], "circle"),
    ("aerofoil / C", ["main"], "c"),
    ("two aerofoils / box", ["main", "flap"], "box"),
    ("two aerofoils / circle", ["main", "flap"], "circle"),
    ("disc / box", ["disc"], "box"),
    ("fuselage / circle", ["fuse"], "circle"),
]


def build(group, kind):
    outer = {"circle": lambda g: am.circle_farfield(g, 8.0),
             "c": lambda g: am.c_farfield(g, 8.0, 14.0),
             "box": lambda g: am.box_farfield(g, 4.0, 6.0, 4.0)}[kind](group)
    region = am.build_region(group, outer=outer)
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    return region, axis, field, am.decompose(axis, region, field=field)


# ═══════════════════════════════════════════════════════════════════
#  The identity itself
# ═══════════════════════════════════════════════════════════════════

class TestGaussBonnet:
    """Ground truth: the identity on meshes built by hand, not by AeroMesh."""

    @staticmethod
    def _mesh(nx, ny, removed=()):
        V = {}
        for j in range(ny + 1):
            for i in range(nx + 1):
                V[(i, j)] = len(V)
        faces = [[V[(i, j)], V[(i + 1, j)], V[(i + 1, j + 1)], V[(i, j + 1)]]
                 for j in range(ny) for i in range(nx)
                 if (i, j) not in set(removed)]
        quads_at, edges = defaultdict(int), defaultdict(int)
        for f in faces:
            for a in f:
                quads_at[a] += 1
            for k in range(4):
                u, v = f[k], f[(k + 1) % 4]
                edges[(min(u, v), max(u, v))] += 1
        valence = defaultdict(int)
        for u, v in edges:
            valence[u] += 1
            valence[v] += 1
        boundary = {u for e, c in edges.items() if c == 1 for u in e}
        chi = len(V) - len(edges) + len(faces)
        interior = sum(valence[x] - 4 for x in V.values() if x not in boundary)
        bnd = sum(2 - quads_at[x] for x in boundary)
        return interior, bnd, chi

    def test_a_plain_grid_is_a_disc(self):
        interior, bnd, chi = self._mesh(5, 5)
        assert chi == 1
        assert interior == bnd - 4 * chi == 0

    def test_one_square_hole_makes_an_annulus(self):
        interior, bnd, chi = self._mesh(5, 5, removed=[(2, 2)])
        assert chi == 0
        assert interior == bnd - 4 * chi

    def test_two_holes(self):
        interior, bnd, chi = self._mesh(5, 5, removed=[(1, 1), (3, 3)])
        assert chi == -1
        assert interior == bnd - 4 * chi == 0

    def test_three_holes(self):
        interior, bnd, chi = self._mesh(7, 7, removed=[(1, 1), (3, 3), (5, 5)])
        assert chi == -2
        assert interior == bnd - 4 * chi


# ═══════════════════════════════════════════════════════════════════
#  The certificate on real blockings
# ═══════════════════════════════════════════════════════════════════

class TestCertificate:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_blocking_balances_its_own_index_budget(self, bodies, name, keys, kind):
        _, _, _, system = build([bodies[k] for k in keys], kind)
        cert = system.index_certificate()
        assert cert["ok"], (name, cert)

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_no_hanging_nodes(self, bodies, name, keys, kind):
        _, _, _, system = build([bodies[k] for k in keys], kind)
        assert system.hanging_nodes() == []

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_certificate_agrees_with_the_tiling_checks(self, bodies, name, keys, kind):
        _, _, _, system = build([bodies[k] for k in keys], kind)
        assert system.validate()["tiles"]
        assert system.validate_strict()["ok"]
        assert system.interface_report()["conformal"]
        assert system.index_certificate()["ok"]

    def test_a_ring_carries_no_index(self, bodies):
        _, _, _, system = build([bodies["fuse"]], "circle")
        assert system.kinds() == {"ring": 1}
        cert = system.index_certificate()
        assert cert["k_realised"] == cert["k_required"] == 0

    def test_corner_valences_put_cut_feet_on_the_boundary(self, bodies):
        """A cut ends at an exact perpendicular foot, which lies *between* two
        boundary samples. Measured to the nearest sample it sits half a spacing
        off the wall, and every wall corner of every block is then counted as an
        interior node -- which breaks the certificate by several units per block
        while the blocking is correct."""
        _, _, _, system = build([bodies["main"]], "box")
        corners = system.corner_valences()
        assert corners
        assert all(on_boundary for _, _, on_boundary in corners.values())


# ═══════════════════════════════════════════════════════════════════
#  Pinned endpoints
# ═══════════════════════════════════════════════════════════════════

class TestPinnedEndpoints:
    """A cut's feet are analytic and shared exactly by the blocks either side.
    An arc's are traced through samples. Closing the gap between them at their
    midpoint moves the one that was right, by an amount that depends on the
    other sides of *this* block -- so the neighbour does not follow."""

    def test_a_pinned_end_does_not_move(self):
        cut = np.array([[0.0, 0.0], [1.0, 0.0]])
        arc = np.array([[1.0, 1e-4], [2.0, 1.0]])
        sides = clean_sides((cut, arc), snap_tol=1e-2,
                            pinned=(True, False))
        np.testing.assert_allclose(sides[0][-1], [1.0, 0.0], atol=0.0)
        np.testing.assert_allclose(sides[1][0], [1.0, 0.0], atol=0.0)

    def test_an_unpinned_join_still_meets_in_the_middle(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[1.0, 1e-4], [2.0, 1.0]])
        sides = clean_sides((a, b), snap_tol=1e-2, pinned=(False, False))
        np.testing.assert_allclose(sides[0][-1], [1.0, 5e-5])
        np.testing.assert_allclose(sides[1][0], [1.0, 5e-5])

    def test_both_pinned_is_left_alone_at_the_midpoint(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([[1.0, 1e-4], [2.0, 1.0]])
        sides = clean_sides((a, b), snap_tol=1e-2, pinned=(True, True))
        np.testing.assert_allclose(sides[0][-1], [1.0, 5e-5])

    def test_the_fuselage_and_wing_crack_is_closed(self, bodies):
        """The case that motivated it: a lens 0.24 long and 7e-4 wide between
        two vertex regions, 5e-7 of the region area -- which passed the shapely
        tiling check and failed the certificate by two."""
        group = [bodies["fuse"],
                 am.load_airfoil("0012", n_points=300).transform(
                     scale=0.9, dx=0.2, dy=1.2, name="wing")]
        region = am.build_region(group, outer=am.box_farfield([bodies["fuse"]],
                                                             5.0, 8.0, 5.0))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        system = am.decompose(axis, region, field=field)
        assert system.index_certificate()["ok"]
        assert system.validate_strict()["gap_fraction"] < 1e-15


# ═══════════════════════════════════════════════════════════════════
#  Local tolerance
# ═══════════════════════════════════════════════════════════════════

class TestLocalSpacing:
    def test_the_tolerance_is_read_per_loop_not_globally(self, bodies):
        """A box 40 across sampled at 720 points has a spacing of 0.055. Twice
        that as a single tolerance erases real sides on a 0.3-chord flap."""
        group = [bodies["main"], bodies["flap"]]
        region = am.build_region(group, outer=am.box_farfield(group, 4., 6., 4.))
        tol_at = local_spacing(region)
        on_box = tol_at(region.outer.loop.points[0])
        on_flap = tol_at(region.holes[1].loop.points[0])
        assert on_box > 5.0 * on_flap

    def test_a_point_off_the_boundary_is_never_snapped(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        tol_at = local_spacing(region)
        assert tol_at(np.array([2.0, 2.0])) == 0.0


# ═══════════════════════════════════════════════════════════════════
#  Corner resolution
# ═══════════════════════════════════════════════════════════════════

class TestCornerResolution:
    def test_a_blunt_trailing_edge_is_never_buried(self, bodies):
        for kind in ("box", "circle", "c"):
            _, _, _, system = build([bodies["main"]], kind)
            assert system.buried_corners() == [], kind

    def test_without_the_second_pass_a_ring_buries_them(self, bodies):
        region, axis, field, _ = build([bodies["main"]], "circle")
        raw = am.decompose(axis, region, field=field, resolve_corners=False)
        assert raw.kinds() == {"ring": 1}
        assert len(raw.buried_corners()) == 2

    def test_only_the_corners_that_need_it_are_cut(self, bodies):
        """On a box a flare already runs out to each trailing-edge corner, so
        the corner is a block corner already. Cutting anyway turns a 90 degree
        block angle into a 3.6 degree one."""
        region, axis, field, system = build([bodies["main"]], "box")
        raw = am.decompose(axis, region, field=field, resolve_corners=False)
        assert raw.buried_corners() == []
        assert system.n_blocks == raw.n_blocks
        assert system.quality()["min_angle"] > 45.0

    def test_corner_cuts_are_medial_radii_that_touch_the_corner(self, bodies):
        region, axis, field, _ = build([bodies["main"]], "circle")
        raw = am.decompose(axis, region, field=field, resolve_corners=False)
        cuts = corner_cuts(axis, region, only=raw.buried_corners())
        assert cuts
        _, _, offsets = region.sample()
        for (u, v, key), idx in cuts.items():
            d = axis.graph.edges[u, v, key]
            for i in idx:
                touches = {int(d["i1"][i]), int(d["i2"][i])}
                assert touches & set(raw.buried_corners())

    def test_a_smooth_body_needs_no_corner_cut(self, bodies):
        region, axis, field, _ = build([bodies["fuse"]], "circle")
        assert corner_cuts(axis, region) == {}


class TestQuality:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_no_block_side_collapses(self, bodies, name, keys, kind):
        _, _, _, system = build([bodies[k] for k in keys], kind)
        assert system.quality()["min_side"] > 0.0

    def test_quality_reports_what_a_corner_is_short_of(self, bodies):
        """A blunt trailing edge of 261.8 degrees wants three elements, so two
        grid lines leaving it. The blocking supplies fewer; that is a known and
        reported limitation, not a silent one."""
        _, _, _, system = build([bodies["main"]], "box")
        q = system.quality()
        assert "unresolved_corners" in q
        assert all(v >= 1 for v in q["unresolved_corners"].values())


# ═══════════════════════════════════════════════════════════════════
#  Closed medial edges
# ═══════════════════════════════════════════════════════════════════

class TestRollLoopEdge:
    def test_rolling_keeps_the_edge_closed_and_the_same_length(self, bodies):
        region = am.build_region([bodies["fuse"]],
                                 outer=am.circle_farfield([bodies["fuse"]], 8.0))
        axis = am.exterior_axis(region)
        loops = [d for _, _, d in axis.graph.edges(data=True)
                 if d["kind"].name == "LOOP"]
        assert loops
        d = loops[0]
        rolled = roll_loop_edge(d, 37)
        np.testing.assert_allclose(rolled["polyline"][0], rolled["polyline"][-1])
        assert rolled["polyline"].shape == d["polyline"].shape
        assert rolled["length"] == pytest.approx(d["length"], rel=1e-12)
        np.testing.assert_allclose(rolled["polyline"][0], d["polyline"][37])

    def test_rolling_by_zero_changes_nothing(self, bodies):
        region = am.build_region([bodies["fuse"]],
                                 outer=am.circle_farfield([bodies["fuse"]], 8.0))
        axis = am.exterior_axis(region)
        d = [x for _, _, x in axis.graph.edges(data=True)
             if x["kind"].name == "LOOP"][0]
        np.testing.assert_allclose(roll_loop_edge(d, 0)["polyline"], d["polyline"])


# ═══════════════════════════════════════════════════════════════════
#  The thickness law
# ═══════════════════════════════════════════════════════════════════

class TestThicknessLaw:
    def test_a_constant_rho_layer_follows_the_medial_radius(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        r_m = medial_radius_on_boundary(axis, region)[1]
        delta = wall_distance(axis, region, rho_max=0.05)[1]
        assert delta.max() / delta.min() == pytest.approx(
            r_m.max() / r_m.min(), rel=0.2)

    def test_the_cap_makes_the_layer_uniform_where_there_is_room(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        delta = wall_distance(axis, region, delta_target=0.02, rho_max=0.05)[1]
        np.testing.assert_allclose(delta, 0.02)

    def test_the_cap_binds_only_where_the_geometry_forces_it(self, bodies):
        """Put a flap next to the aerofoil and r_m collapses in the slot by a
        factor of ~50. A constant fraction of it collapses with it; the capped
        law thins only there."""
        group = [bodies["main"], bodies["flap"]]
        region = am.build_region(group, outer=am.box_farfield(group, 4., 6., 4.))
        axis = am.exterior_axis(region)
        plain = wall_distance(axis, region, rho_max=0.05)[1]
        capped = wall_distance(axis, region, delta_target=0.02, rho_max=0.05)[1]
        assert plain.max() / plain.min() > 20.0
        assert capped.max() / capped.min() < 0.5 * (plain.max() / plain.min())
        assert capped.max() == pytest.approx(0.02)
        assert np.all(capped <= plain + 1e-15)

    def test_the_cap_never_exceeds_the_available_room(self, bodies):
        group = [bodies["main"], bodies["flap"]]
        region = am.build_region(group, outer=am.box_farfield(group, 4., 6., 4.))
        axis = am.exterior_axis(region)
        for li, delta in wall_distance(axis, region, delta_target=10.0,
                                       rho_max=0.05).items():
            r_m = medial_radius_on_boundary(axis, region)[li]
            assert np.all(delta <= 0.05 * r_m.max() + 1e-12)

    def test_a_bad_rho_max_is_rejected(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        with pytest.raises(ValueError):
            wall_distance(axis, region, rho_max=1.5)
        with pytest.raises(ValueError):
            wall_distance(axis, region, delta_target=-1.0)


# ═══════════════════════════════════════════════════════════════════
#  Defects found by review, each with the reproduction that found it
# ═══════════════════════════════════════════════════════════════════

class TestSideKeyIsReversalInvariant:
    """Two blocks sharing a side walk it in opposite directions, so the key
    that pairs them must not depend on which way round it is read."""

    def test_an_even_length_side_pairs_with_its_reverse(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        system = am.decompose(axis, region, field=field)
        tol = system.coincidence_tol()
        # An even number of points is the case that used to fail: index
        # len//2 gives 59 one way round and 58 the other.
        side = np.column_stack([np.linspace(0.0, 1.0, 118),
                                np.linspace(0.0, 0.3, 118) ** 2])
        assert system._side_key(side, tol) == system._side_key(side[::-1], tol)

    def test_a_two_point_side_pairs_with_its_reverse(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        tol = system.coincidence_tol()
        side = np.array([[0.0, 0.0], [0.02, 0.0]])
        assert system._side_key(side, tol) == system._side_key(side[::-1], tol)

    def test_a_layered_blocking_has_no_stranded_sides(self, bodies):
        """The defect this invariant protects: ten of the layer's offset and
        cut-end sides paired with nothing, were filed as boundary sides, and
        the old ``accounted`` tautology could not see it."""
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        layered = am.insert_layer(system, delta_target=0.02, rho_max=0.05)
        assert layered.interface_report()["stranded"] == []
        assert layered.interface_report()["conformal"]


class TestStrandedSides:
    """``paired + boundary == n_sides`` is a tautology once the sides claimed
    three times are taken out, so a side that pairs with nobody is filed as a
    boundary side and the count still balances. Checking the *name* is what
    catches it."""

    def test_a_side_in_mid_air_is_not_a_boundary_side(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        assert system.interface_report()["conformal"]

        from aeromesh.blocking.blocks import Block, BlockSystem
        victim = next(b for b in system.blocks if b.n_sides >= 4)
        sides = [np.asarray(s, dtype=float).copy() for s in victim.sides]
        sides[1] = sides[1] + np.array([0.01, 0.01])      # unglue one side
        hurt = BlockSystem(
            blocks=[Block(id=b.id, sides=(tuple(sides) if b.id == victim.id
                                          else b.sides),
                          kind=b.kind, source=b.source, wraps=b.wraps)
                    for b in system.blocks],
            region=region, axis=axis)
        assert hurt.interface_report()["stranded"]
        assert not hurt.interface_report()["conformal"]


class TestCanonicalSides:
    """``n_sides`` must be the logical degree of the polygon, because that is
    what every consumer reads it as."""

    def test_flat_joins_are_merged(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        raw = am.decompose(axis, region, field=field, canonical=False)
        canon = am.decompose(axis, region, field=field)
        assert sum(b.n_sides for b in canon.blocks) < \
               sum(b.n_sides for b in raw.blocks)
        assert canon.is_sound()

    def test_no_flat_corner_survives_canonicalisation(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        canon = am.decompose(axis, region,
                             field=am.solve_singularities(axis, region))
        flat = [a for a in canon.corner_angles().values() if abs(a - 180.0) < 5.0]
        assert not flat, flat

    def test_a_real_boundary_corner_is_never_merged_away(self, bodies):
        """The trailing edge is a 261.8 degree corner, not a flat join, and
        merging it would re-bury it."""
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        canon = am.decompose(axis, region,
                             field=am.solve_singularities(axis, region))
        assert canon.buried_corners() == []

    def test_canonicalising_cannot_make_a_blocking_unsound(self, bodies):
        for kind in ("box", "circle", "c"):
            _, _, _, system = build([bodies["main"]], kind)
            assert system.is_sound(), kind
            assert system.canonicalise().is_sound(), kind


class TestDesignIndex:
    """What the blocking asks for, from its corner angles, as against what
    midpoint subdivision actually delivers."""

    def test_a_flat_corner_asks_for_nothing(self):
        from aeromesh.blocking.blocks import Block, BlockSystem
        # a square with one extra point mid-side: five sides, four real corners
        sq = [np.array([[0.0, 0.0], [1.0, 0.0]]),
              np.array([[1.0, 0.0], [1.0, 1.0]]),
              np.array([[1.0, 1.0], [0.5, 1.0]]),
              np.array([[0.5, 1.0], [0.0, 1.0]]),
              np.array([[0.0, 1.0], [0.0, 0.0]])]
        blk = Block(id=0, sides=tuple(sq), kind="span", source=())
        ang = sorted(round(a) for a in BlockSystem(
            blocks=[blk], region=None, axis=None).corner_angles().values())
        assert ang == [90, 90, 90, 90, 180]

    def test_the_design_index_is_blind_to_a_flat_join(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        raw = am.decompose(axis, region, field=field, canonical=False)
        canon = am.decompose(axis, region, field=field)
        assert raw.design_index()["k_design"] == canon.design_index()["k_design"]

    def test_canonicalising_narrows_the_gap_to_the_realised_index(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        raw = am.decompose(axis, region, field=field, canonical=False)
        canon = am.decompose(axis, region, field=field)
        assert abs(canon.quality()["design_gap"]) < abs(raw.quality()["design_gap"])

    def test_a_smooth_body_in_a_smooth_far_field_asks_for_nothing(self, bodies):
        _, _, _, system = build([bodies["fuse"]], "circle")
        d = system.design_index()
        assert d["k_design"] == 0
        assert d["gap_to_realised"] == 0


class TestCornerElementDeterminism:
    """n_c must be a property of the geometry, not of the last bit of the
    angle. A regular octagon's fluid angle is exactly 2.5 quarter-turns at
    every corner, so every corner is a tie."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
    def test_a_regular_polygon_gives_the_same_answer_at_any_rotation(self, n):
        terms = set()
        for rot in (0.0, 0.1, 0.25, 0.37, 0.5, 1.234):
            t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
            g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                                 name="g").resample(400)
            region = am.build_region([g], outer=am.circle_farfield([g], 6.0))
            b = region.boundaries[1]
            terms.add(int(np.sum(2 - b.corner_elements)))
        assert len(terms) == 1, terms

    def test_n_c_is_never_zero(self):
        """A corner sharper than a quarter-turn would otherwise ask for no
        elements at a mesh node at all."""
        spike = Loop.from_points(np.array(
            [[0.0, 0.0], [1.0, 0.03], [2.0, 0.0], [1.0, 1.0]]), name="spike")
        region = am.build_region([spike],
                                 outer=am.circle_farfield([spike], 8.0))
        for b in region.boundaries:
            if len(b.corners):
                assert np.all(b.corner_elements >= 1)

    def test_a_blunt_trailing_edge_is_nowhere_near_a_tie(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        x = region.boundaries[1].fluid_angles / (np.pi / 2)
        assert np.all(np.abs(x - (np.floor(x) + 0.5)) > 0.3)
        assert np.all(region.boundaries[1].corner_elements == 3)


class TestCornerCutsMatchTheFoot:
    """A corner-resolving cut has to have its foot *at* the corner. Matching on
    the touch index admits the sample at each end of the run, where the foot has
    already moved onto the adjacent edge."""

    def test_every_corner_cut_has_its_foot_on_the_corner(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.circle_farfield([bodies["main"]], 8.0))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        raw = am.decompose(axis, region, field=field, resolve_corners=False)
        buried = raw.buried_corners()
        assert buried
        cuts = corner_cuts(axis, region, only=buried)
        assert cuts
        _, _, offsets = region.sample()
        pts = region.sample()[0]
        tol = 1e-7 * region.outer.loop.scale
        for (u, v, key), idx in cuts.items():
            d = axis.graph.edges[u, v, key]
            for i in idx:
                feet = np.vstack([d["t1"][i], d["t2"][i]])
                assert np.min(np.linalg.norm(feet - pts[buried], axis=1)
                              .reshape(len(feet), -1).min(axis=1)) <= tol \
                    or np.min([np.linalg.norm(f - pts[g])
                               for f in feet for g in buried]) <= tol

    def test_resolution_actually_unburies_them(self, bodies):
        for kind in ("box", "circle", "c"):
            _, _, _, system = build([bodies["main"]], kind)
            assert system.buried_corners() == [], kind

    def test_what_cannot_be_resolved_is_reported(self):
        """A regular pentagon body keeps some corners buried. That has to show
        up in ``notes``, not pass silently."""
        t = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.13
        g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                             name="p5").resample(600)
        region = am.build_region([g], outer=am.circle_farfield([g], 6.0))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        # It used to keep corners buried, and it no longer does: the detected
        # corner index was one sample off the vertex carrying the turn, so the
        # foot matching found nothing and `buried_corners` reported a kink that
        # was in fact resolved. All five are resolved now, with no notes -- so
        # what this test pins is that there is nothing left to report.
        assert system.buried_corners() == []
        assert system.notes == []
        assert system.is_sound()


class TestResolutionNeverBreaksTheBlocking:
    """Corner resolution is a refinement, and a refinement that breaks the
    decomposition is not an improvement."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
    @pytest.mark.parametrize("kind", ["circle", "box"])
    def test_a_polygonal_body_is_always_sound(self, n, kind):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.13
        g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                             name=f"p{n}").resample(600)
        outer = (am.circle_farfield([g], 6.0) if kind == "circle"
                 else am.box_farfield([g], 4.0, 6.0, 4.0))
        region = am.build_region([g], outer=outer)
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        assert system.is_sound()


# ═══════════════════════════════════════════════════════════════════
#  Defects found by the second audit round
# ═══════════════════════════════════════════════════════════════════

class TestCornerIndexLandsOnTheVertex:
    """At a polyline corner the extrapolated turn is a *plateau* -- on a unit
    triangle two adjacent samples both read exactly 120.000 degrees -- so the
    non-maximum suppression's stable tie order kept the lower index, whose own
    vertex turn was 0.00 degrees. The angle was right and the position was one
    sample out, and everything downstream matches corners by position."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
    def test_a_polygon_corner_sits_on_the_turning_vertex(self, n):
        from aeromesh.geometry.corners import detect_corners, vertex_turns
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.13
        g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                             name="g").resample(600)
        corners = detect_corners(g)
        assert len(corners) == n
        own = np.abs(vertex_turns(g))[corners.index]
        want = 2 * np.pi / n
        np.testing.assert_allclose(own, want, rtol=1e-6)

    def test_a_smooth_feature_is_not_snapped(self):
        """The snap is only for a turn concentrated on one vertex. A cusped
        trailing edge spreads the same total over tens of samples, and moving
        the corner to that window's maximum would be moving it for no reason."""
        from aeromesh.geometry.corners import detect_corners
        af = am.load_airfoil("2412", n_points=400)
        before = detect_corners(af).index.copy()
        again = detect_corners(af).index
        np.testing.assert_array_equal(before, again)
        assert len(before) == 1          # the cusped TE, and nothing else

    def test_the_blunt_trailing_edge_corners_stay_where_they_were(self, bodies):
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        b = region.boundaries[1]
        pos = b.corners.positions(b.loop)
        assert len(pos) == 2
        np.testing.assert_allclose(np.degrees(b.fluid_angles), 261.8, atol=0.2)
        assert np.all(b.corner_elements == 3)

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
    def test_a_polygonal_body_no_longer_reports_a_buried_corner(self, n):
        """Ten of ten corners previously reported as buried on polygonal bodies
        were this off-by-one: the kink *was* resolved, and the check was looking
        a sample away from it."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.13
        g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                             name=f"p{n}").resample(600)
        region = am.build_region([g], outer=am.box_farfield([g], 4., 6., 4.))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        assert system.buried_corners() == []
        assert system.is_sound()
        assert system.notes == []


class TestCanonicaliseMergesOnlyBoundaryArcs:
    """A cut's foot at a *reflex* corner can leave almost tangentially -- 179.6
    degrees was measured -- so a cut-into-wall join can read as flat. Merging it
    destroyed a real 120 degree block corner on a triangular body and cost that
    blocking its conformality."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
    def test_a_polygonal_body_canonicalises_and_stays_sound(self, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.13
        g = Loop.from_points(np.column_stack([np.cos(t), np.sin(t)]),
                             name=f"p{n}").resample(600)
        region = am.build_region([g], outer=am.box_farfield([g], 4., 6., 4.))
        axis = am.exterior_axis(region)
        field = am.solve_singularities(axis, region)
        raw = am.decompose(axis, region, field=field, canonical=False)
        canon = am.decompose(axis, region, field=field)
        assert canon.is_sound()
        assert canon.interface_report()["conformal"]
        assert sum(b.n_sides for b in canon.blocks) < \
               sum(b.n_sides for b in raw.blocks)

    def test_a_rejected_canonicalisation_is_reported(self, bodies):
        """If the merge has to be rolled back, `n_sides` overstates the logical
        degree and `m_gons` and `design_gap` are computed on a representation
        with spurious sides. That must be said, not left silent."""
        for kind in ("box", "circle", "c"):
            _, _, _, system = build([bodies["main"]], kind)
            rejected = [n for n in system.notes if "canonicalisation" in n]
            if rejected:
                assert "spurious" in rejected[0]


class TestLayerNormalsPointIntoTheFluid:
    """The probe that decides which side is fluid has to be shorter than the
    body is thick. A 0.3-chord flap is 0.036 thick while the layer reaches 0.194
    on its outboard side, so a probe scaled to the *largest* thickness on the
    arc stepped clean through the body, found fluid on both sides at every
    sample, and the tie built the whole near-wall layer inside the flap -- with
    the area check still passing at a relative error of 4.5e-4 and nothing in
    the notes."""

    LAYOUTS = [("aerofoil / box", ["main"], "box"),
               ("aerofoil / C", ["main"], "c"),
               ("two aerofoils / box", ["main", "flap"], "box"),
               ("two aerofoils / circle", ["main", "flap"], "circle")]

    @pytest.mark.parametrize("name,keys,kind", LAYOUTS,
                             ids=[c[0] for c in LAYOUTS])
    @pytest.mark.parametrize("rho_max", [0.05, 0.3])
    def test_the_default_thickness_law_is_sound(self, bodies, name, keys,
                                                kind, rho_max):
        """With no ``delta_target`` -- the documented default -- and over the
        range of ``rho_max`` the optimisation stage is meant to vary."""
        _, _, _, system = build([bodies[k] for k in keys], kind)
        layered = am.insert_layer(system, rho_max=rho_max)
        assert layered.validate()["tiles"], name
        assert layered.validate_strict()["ok"], name
        assert layered.interface_report()["conformal"], name

    def test_the_layer_is_outside_every_body(self, bodies):
        group = [bodies["main"], bodies["flap"]]
        region = am.build_region(group, outer=am.circle_farfield(group, 8.0))
        axis = am.exterior_axis(region)
        system = am.decompose(axis, region,
                              field=am.solve_singularities(axis, region))
        layered = am.insert_layer(system, rho_max=0.3)
        # Tested on each block's own interior point, not on its outline: a
        # layer block's wall side lies *on* the body, where point-in-polygon is
        # ambiguous by construction, so an outline test reports hundreds of
        # "inside" points for a block that is perfectly placed.
        from aeromesh.blocking.templates import centre_of
        seen = 0
        for b in layered.blocks:
            if b.kind != "layer":
                continue
            seen += 1
            c = centre_of(b)
            assert bool(region.contains(c[None, :])[0]), (b.id, c)
        assert seen

    def test_the_vote_ignores_a_probe_that_crossed_something(self, bodies):
        """A sample where both sides read as fluid has stepped across the body
        and carries no information; counting it is what produced the tie."""
        from aeromesh.blocking.layers import _normals_along
        region = am.build_region([bodies["main"]],
                                 outer=am.box_farfield([bodies["main"]], 4., 6., 4.))
        arc = bodies["main"].points
        n = _normals_along(arc, region, 10.0)      # absurdly long probe
        step = 1e-4
        assert region.contains(arc + step * n).mean() > 0.95


class TestDesignIndexIsAnIdentity:
    """Charging every block corner with the elements its angle asks for gives a
    second reading of the same budget, and it has to balance identically. The
    first version differenced two different models -- design counts in the block
    term, one-element-per-corner in the required term -- and dropped the
    interior term, which is not optional because ``round`` is not additive."""

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_design_budget_balances(self, bodies, name, keys, kind):
        _, _, _, system = build([bodies[k] for k in keys], kind)
        d = system.design_index()
        assert d["ok"], (name, d)
        assert d["residual"] == 0

    def test_the_interior_term_is_not_optional(self, bodies):
        """Three corners of 100, 130 and 130 degrees at one node ask for
        1 + 1 + 1 = 3 elements, not the 4 that 360 degrees would."""
        assert max(1, int(np.floor(100.0 / 90.0 + 0.5))) == 1
        assert max(1, int(np.floor(130.0 / 90.0 + 0.5))) == 1
        _, _, _, system = build([bodies["main"]], "box")
        assert "k_interior" in system.design_index()

    def test_a_smooth_body_asks_for_what_it_gets(self, bodies):
        _, _, _, system = build([bodies["fuse"]], "circle")
        d = system.design_index()
        assert d["k_design"] == d["k_realised"] == 0
        assert d["gap_to_realised"] == 0

    def test_the_gap_is_the_unresolved_reflex_corners(self, bodies):
        """Midpoint subdivision puts one element at every corner, so wherever a
        corner asks for more the delivered mesh carries index the design would
        have placed elsewhere."""
        _, _, _, system = build([bodies["main"]], "box")
        assert system.design_index()["gap_to_realised"] > 0
        assert system.quality()["unresolved_corners"]
