"""Tests for block connectivity and the uncut-flare rule (S3 items A and B).

Connectivity is what S4 needs in order to mesh conformally: two blocks sharing a
side have to agree on how many points lie along it. It is also a check no area
test can make -- a side claimed by three blocks, or one that should pair and does
not, is a non-conformal decomposition that tiles perfectly well.
"""

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.blocks import orient_ccw
from aeromesh.blocking.search import select_splits
from aeromesh.geometry.loop import Loop


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
        "slat": main.transform(scale=0.22, angle=np.deg2rad(22), dx=-0.20, dy=0.02,
                               name="slat"),
        "fuselage": ellipse(0.55, 0.85, 0.45, -1.55, name="fuselage"),
    }


def setup(group, kind):
    outer = {"circle": lambda g: am.circle_farfield(g, 4.0),
             "c": lambda g: am.c_farfield(g, 4.0, 6.0),
             "box": lambda g: am.box_farfield(g, 4.0, 6.0, 4.0)}[kind](group)
    region = am.build_region(group, outer=outer)
    axis = am.exterior_axis(region)
    field = am.solve_singularities(axis, region)
    return region, axis, field, am.decompose(axis, region, field=field)


CONFIGS = [
    ("1 body / circle", ["main"], "circle"),
    ("1 body / C-shape", ["main"], "c"),
    ("1 body / box", ["main"], "box"),
    ("2 bodies / circle", ["main", "flap"], "circle"),
    ("2 bodies / box", ["main", "flap"], "box"),
    ("3 bodies / circle", ["main", "flap", "slat"], "circle"),
    ("3 bodies / box", ["main", "flap", "slat"], "box"),
    ("bluff body / box", ["fuselage"], "box"),
]


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TestOrientation:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_every_block_is_wound_counter_clockwise(self, bodies, name, keys, kind):
        """Blocks are assembled by walking the medial graph, which fixes no
        handedness, so some come out clockwise unless normalised. Consistency
        matters for interfaces and for transfinite interpolation."""
        _, _, _, system = setup([bodies[k] for k in keys], kind)
        for b in system.blocks:
            if b.wraps:
                continue
            assert signed_area(b.outline()) > 0, f"{b} is clockwise"

    def test_orient_ccw_flips_a_clockwise_block(self):
        sides = (np.array([[0.0, 0.0], [0.0, 1.0]]),
                 np.array([[0.0, 1.0], [1.0, 1.0]]),
                 np.array([[1.0, 1.0], [1.0, 0.0]]),
                 np.array([[1.0, 0.0], [0.0, 0.0]]))
        out = orient_ccw(sides)
        pts = np.vstack([out[0]] + [s[1:] for s in out[1:]])
        assert signed_area(pts) > 0

    def test_orient_ccw_leaves_a_counter_clockwise_block_alone(self):
        sides = (np.array([[0.0, 0.0], [1.0, 0.0]]),
                 np.array([[1.0, 0.0], [1.0, 1.0]]),
                 np.array([[1.0, 1.0], [0.0, 1.0]]),
                 np.array([[0.0, 1.0], [0.0, 0.0]]))
        out = orient_ccw(sides)
        for a, b in zip(out, sides):
            np.testing.assert_allclose(a, b)


class TestConnectivity:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_decomposition_is_conformal(self, bodies, name, keys, kind):
        """Every side is either on the region boundary or shared with exactly
        one other block."""
        _, _, _, system = setup([bodies[k] for k in keys], kind)
        rep = system.interface_report()
        assert rep["over_shared"] == []
        assert rep["accounted"], rep
        assert rep["conformal"], rep

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_an_interface_is_traversed_oppositely_by_its_two_blocks(
            self, bodies, name, keys, kind):
        """Both blocks walk their own outline counter-clockwise, so a side they
        share must be walked one way by one and the other way by the other."""
        _, _, _, system = setup([bodies[k] for k in keys], kind)
        for i in system.connectivity()["interfaces"]:
            assert i.flipped

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_interface_sides_are_geometrically_identical(
            self, bodies, name, keys, kind):
        """Adjacent blocks are built from the same split, so a shared side is the
        same curve in both -- which is what makes matching point counts across it
        well defined."""
        _, _, _, system = setup([bodies[k] for k in keys], kind)
        by_id = {b.id: b for b in system.blocks}
        for i in system.connectivity()["interfaces"]:
            a = by_id[i.block_a].sides[i.side_a]
            b = by_id[i.block_b].sides[i.side_b]
            assert len(a) == len(b)
            np.testing.assert_allclose(a, b[::-1], atol=1e-9)

    def test_there_is_at_least_one_interface_whenever_there_is_more_than_one_block(
            self, bodies):
        for name, keys, kind in CONFIGS:
            _, _, _, system = setup([bodies[k] for k in keys], kind)
            if system.n_blocks > 1:
                assert system.interface_report()["n_interfaces"] > 0


class TestUncutFlares:
    """A flare carrying no singularity is left uncut, and its wedge merges into
    the neighbouring vertex region."""

    def test_a_flare_without_a_singularity_gets_no_cut(self, bodies):
        region, axis, field, _ = setup([bodies["main"]], "box")
        chosen = select_splits(axis, region, field)
        flares = [(u, v, k) for u, v, k in axis.graph.edges(keys=True)
                  if axis.graph.edges[u, v, k]["kind"].name == "FLARE"]
        assert flares
        for ek in flares:
            assert len(chosen[ek]) == 0

    def test_no_corner_wedge_survives_when_its_flare_is_uncut(self, bodies):
        """The corner becomes a vertex of the enclosing region instead."""
        _, _, _, system = setup([bodies["main"]], "box")
        assert "corner" not in system.kinds()

    def test_the_corner_itself_is_still_on_the_boundary_of_some_block(self, bodies):
        """Merging the wedge must not drop the corner from the decomposition."""
        region, _, _, system = setup([bodies["main"]], "box")
        corners = region.outer.corners.positions(region.outer.loop)
        assert len(corners) == 4
        for c in corners:
            assert any(np.linalg.norm(b.outline() - c, axis=1).min() < 1e-6
                       for b in system.blocks), f"corner {c} is on no block"

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_leaving_flares_uncut_keeps_the_tiling(self, bodies, name, keys, kind):
        _, _, _, system = setup([bodies[k] for k in keys], kind)
        assert system.validate()["tiles"]
        assert system.validate_strict()["ok"], system.validate_strict()

    def test_it_reduces_the_block_count(self, bodies):
        """The point of the change: no cut is spent where no criterion can
        choose one."""
        region, axis, field, ranked = setup([bodies["main"]], "box")
        even = am.decompose(axis, region)
        assert ranked.n_blocks < even.n_blocks
        assert ranked.n_blocks == 4
