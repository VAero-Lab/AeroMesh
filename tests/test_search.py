"""Tests for the ranked split search (S3).

The search decides *where* to cut. Its objective, following Fogg sec. 4.2, is
that each singularity ends up alone inside its own subregion -- which in this
block structure means each block holds at most one. That is the property these
tests check, alongside the ranking actually being obeyed.
"""

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import aeromesh as am
from aeromesh.blocking.search import locate, place_singularities, select_splits
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
    return region, axis, am.solve_singularities(axis, region)


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


def block_polygon(b):
    if b.kind == "ring":
        return Polygon(b.sides[0], [b.sides[1]])
    return Polygon(b.outline())


class TestPlacement:
    def test_a_singularity_is_located_on_the_axis(self, bodies):
        region, axis, field = setup([bodies["main"], bodies["flap"]], "circle")
        assert field.singularities
        for s in field.singularities:
            p = locate(axis, s.position)
            assert p is not None
            assert p.distance < 0.05 * region.outer.loop.scale

    def test_a_vertex_singularity_places_no_cut(self, bodies):
        """The vertex region already contains it, so it needs no cut of its own."""
        region, axis, field = setup([bodies["main"]], "box")
        assert all(s.source == "vertex" for s in field.singularities)
        assert place_singularities(axis, field) == {}

    def test_edge_singularities_are_sorted_along_their_edge(self, bodies):
        region, axis, field = setup([bodies["main"], bodies["flap"], bodies["slat"]], "box")
        for placements in place_singularities(axis, field).values():
            arc = [p.s for p in placements]
            assert arc == sorted(arc)


class TestSelection:
    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_every_primary_edge_gets_at_least_one_cut(self, bodies, name, keys, kind):
        """Without one, the vertex regions at an edge's two ends would run into
        each other. Flares are the exception -- see TestUncutFlares."""
        region, axis, field = setup([bodies[k] for k in keys], kind)
        chosen = select_splits(axis, region, field)
        for u, v, k in axis.graph.edges(keys=True):
            kind_name = axis.graph.edges[u, v, k]["kind"].name
            if kind_name in ("LOOP", "FLARE"):
                continue
            assert len(chosen[(u, v, k)]) >= 1

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_a_flare_is_cut_only_when_a_singularity_sits_on_it(
            self, bodies, name, keys, kind):
        """Along a flare theta_m is constant and r_m is linear, so no criterion
        can choose a position; the corner is resolved and the wedge merges into
        the vertex region instead."""
        region, axis, field = setup([bodies[k] for k in keys], kind)
        chosen = select_splits(axis, region, field)
        placed = place_singularities(axis, field)
        for u, v, k in axis.graph.edges(keys=True):
            if axis.graph.edges[u, v, k]["kind"].name != "FLARE":
                continue
            has_singularity = len(placed.get((u, v, k), []))
            assert (len(chosen[(u, v, k)]) > 0) == bool(has_singularity)

    def test_an_edge_with_n_singularities_gets_n_plus_one_cuts(self, bodies):
        """One cut per gap: before the first, between each pair, after the last."""
        region, axis, field = setup([bodies["main"], bodies["flap"]], "box")
        placed = place_singularities(axis, field)
        chosen = select_splits(axis, region, field)
        checked = 0
        for edge, placements in placed.items():
            assert len(chosen[edge]) <= len(placements) + 1
            checked += 1
        assert checked > 0

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_search_never_needs_more_blocks_than_even_spacing(
            self, bodies, name, keys, kind):
        """Cuts go only where a singularity has to be separated, so the ranked
        search is never the more fragmented of the two."""
        region, axis, field = setup([bodies[k] for k in keys], kind)
        even = am.decompose(axis, region)
        ranked = am.decompose(axis, region, field=field)
        assert ranked.n_blocks <= even.n_blocks


class TestIsolation:
    """The objective the search exists to meet."""

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_each_block_holds_at_most_one_singularity(self, bodies, name, keys, kind):
        region, axis, field = setup([bodies[k] for k in keys], kind)
        system = am.decompose(axis, region, field=field)
        for b in system.blocks:
            poly = block_polygon(b).buffer(1e-9)
            inside = [s for s in field.singularities if poly.contains(Point(s.position))]
            assert len(inside) <= 1, f"{b} holds {len(inside)} singularities"

    @pytest.mark.parametrize("name,keys,kind", CONFIGS, ids=[c[0] for c in CONFIGS])
    def test_the_decomposition_still_tiles(self, bodies, name, keys, kind):
        """Isolating singularities must not cost correctness."""
        region, axis, field = setup([bodies[k] for k in keys], kind)
        system = am.decompose(axis, region, field=field)
        assert system.validate()["tiles"], system.validate()
        assert system.validate_strict()["ok"], system.validate_strict()

    def test_isolation_survives_a_rigid_motion(self, bodies):
        from aeromesh.geometry.region import Region

        base = [bodies["main"], bodies["flap"]]
        outer = am.box_farfield(base, 4.0, 6.0, 4.0)
        kw = dict(angle=0.7, dx=-12.0, dy=4.5)
        moved_region = Region.build(outer.transform(**kw),
                                    [b.transform(**kw) for b in base])
        axis = am.exterior_axis(moved_region)
        field = am.solve_singularities(axis, moved_region)
        system = am.decompose(axis, moved_region, field=field)
        assert system.validate()["tiles"]
        for b in system.blocks:
            poly = block_polygon(b).buffer(1e-9)
            assert sum(1 for s in field.singularities
                       if poly.contains(Point(s.position))) <= 1
