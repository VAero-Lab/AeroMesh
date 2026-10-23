"""Along-wall spacing, the wedge split, and the smoother that was rejected.

Each of these pins a *measured* result rather than an argued one, because in
every case the reasoning and the measurement disagreed.
"""

from __future__ import annotations

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.collar import (build_collar, curvature_spacing,
                                      mesh_collar, nearest_on_polyline,
                                      smooth_assembly, validate_mesh)


@pytest.fixture(scope="module")
def collar():
    body = am.load_airfoil("2412", n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    return build_collar(region, delta_target=0.02, rho_max=0.05, n_outer=800)


def min_angles(grid):
    a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    q = np.stack([a, b, c, d], axis=2)
    e = np.roll(q, -1, axis=2) - q
    L = np.linalg.norm(e, axis=3)
    prev = np.roll(e, 1, axis=2)
    cos = (np.einsum("ijkl,ijkl->ijk", -prev, e)
           / np.maximum(np.linalg.norm(prev, axis=3) * L, 1e-300))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).min(axis=2)


# ── curvature_spacing ──────────────────────────────────────────────

def test_zero_strength_is_uniform():
    t = np.linspace(0.0, 2.0 * np.pi, 400)
    side = np.column_stack([np.cos(t), np.sin(t)])
    got = curvature_spacing(side, 50, 0.0)
    assert np.allclose(got, np.linspace(0.0, 1.0, 50))


def test_endpoints_and_monotonicity():
    t = np.linspace(0.0, np.pi, 400)
    side = np.column_stack([np.cos(t), 0.1 * np.sin(t)])
    for strength in (0.0, 0.5, 2.0, 8.0):
        got = curvature_spacing(side, 60, strength)
        assert got[0] == pytest.approx(0.0)
        assert got[-1] == pytest.approx(1.0)
        assert np.all(np.diff(got) > 0.0)


def test_a_circle_is_unchanged_because_curvature_is_constant():
    """The weight is uniform when |kappa| is, so a circle must not move."""
    t = np.linspace(0.0, 1.9 * np.pi, 600)
    side = np.column_stack([np.cos(t), np.sin(t)])
    got = curvature_spacing(side, 40, 4.0)
    assert np.allclose(got, np.linspace(0.0, 1.0, 40), atol=2e-3)


def test_points_move_towards_the_curved_end():
    """A curve that is sharp at one end and straight at the other."""
    t = np.linspace(0.0, 1.0, 800)
    # a quarter circle of radius 0.05 followed by a long straight run
    arc = np.column_stack([0.05 * np.cos(np.pi * (1 - t[:200] / t[199]) / 2),
                           0.05 * np.sin(np.pi * (1 - t[:200] / t[199]) / 2)])
    straight = np.column_stack([np.linspace(0.05, 2.0, 600),
                                np.zeros(600)])
    side = np.vstack([arc, straight[1:]])

    n = 40
    uniform = curvature_spacing(side, n, 0.0)
    adapted = curvature_spacing(side, n, 4.0)
    # The arc occupies a small fraction of the arc length, so adapting must
    # push parameters towards it -- i.e. the adapted values are smaller.
    assert np.all(adapted[1:-1] <= uniform[1:-1] + 1e-12)
    assert float(np.max(uniform[1:-1] - adapted[1:-1])) > 0.01


def test_degenerate_inputs_fall_back_to_uniform():
    assert np.allclose(curvature_spacing(np.zeros((3, 2)), 10, 2.0),
                       np.linspace(0.0, 1.0, 10))
    assert len(curvature_spacing(np.zeros((40, 2)), 7, 2.0)) == 7


def test_curvature_improves_the_band_and_coverage(collar):
    """The reason it is on by default."""
    plain = mesh_collar(collar, n_across=25, target_spacing=0.01,
                        first_cell=1e-4, curvature=0.0)
    adapted = mesh_collar(collar, n_across=25, target_spacing=0.01,
                          first_cell=1e-4, curvature=2.0)
    assert min_angles(adapted[0]).min() > min_angles(plain[0]).min()
    v0, v2 = validate_mesh(collar, plain), validate_mesh(collar, adapted)
    assert v2["coverage"] >= v0["coverage"]
    assert v2["all_positive"] and v0["all_positive"]


# ── the wedge split ────────────────────────────────────────────────

def test_node_index_split_beats_every_length_split(collar):
    """Pins the measurement that overruled the argument.

    The node-index midpoint sits at 18 per cent of the ray's length under wall
    clustering, which makes the three sub-blocks visibly unequal. Dividing by
    length instead is worse at every value tried.
    """
    def wedge_min(split):
        grids = mesh_collar(collar, n_across=25, target_spacing=0.01,
                            first_cell=1e-4, wedge_split=split)
        i, worst = 0, np.inf
        for kind in collar["kinds"]:
            n = 1 if kind == "band" else 3
            if kind != "band":
                worst = min(worst, min(min_angles(grids[k]).min()
                                       for k in range(i, i + n)))
            i += n
        return worst

    default = wedge_min(None)
    assert default > wedge_min(0.5)
    assert default > wedge_min(0.8)
    assert default == pytest.approx(59.0, abs=1.5)


# ── the smoother that was measured and rejected ────────────────────

def test_assembly_smoothing_of_a_wedge_is_harmful(collar):
    """`smooth_assembly` exists and works; applying it here does not.

    Laplacian smoothing equalises spacing and the wedges are deliberately
    graded, so relaxing their interiors destroys the angles it was meant to
    improve. Kept as a test so the function is not quietly switched on.
    """
    grids = mesh_collar(collar, n_across=25, target_spacing=0.01,
                        first_cell=1e-4)
    groups, i = [], 0
    for sides, kind in zip(collar["blocks"], collar["kinds"]):
        if kind == "band":
            i += 1
        else:
            groups.append((list(range(i, i + 3)), list(sides)))
            i += 3
    assert groups, "expected at least one wedge"

    idx, sides = groups[0]
    before = min(min_angles(grids[k]).min() for k in idx)
    after_grids = smooth_assembly([grids[k] for k in idx], sides, sweeps=60)
    after = min(min_angles(g).min() for g in after_grids)
    assert after < before, "smoothing helped -- revisit the default"


def test_assembly_smoothing_holds_the_fixed_curves(collar):
    """Whatever it does to the interior, it must not move a shared side.

    This is the test that found the bug in the first version: it matched held
    nodes against the fixed sides' own sample positions, and the mesh nodes are
    resampled along each side and never land on them, so nothing was held and
    the boundary drifted. Held nodes are now found by distance to the curve.
    """
    grids = mesh_collar(collar, n_across=25, target_spacing=0.01,
                        first_cell=1e-4)
    i = 0
    checked = 0
    for sides, kind in zip(collar["blocks"], collar["kinds"]):
        n = 1 if kind == "band" else 3
        if kind != "band":
            idx = list(range(i, i + n))
            out = smooth_assembly([grids[k] for k in idx], list(sides),
                                  sweeps=40)
            for before, after in zip((grids[k] for k in idx), out):
                on_side = np.zeros(before.shape[:2], dtype=bool)
                for side in sides:
                    q = np.asarray(side, dtype=float)
                    d = nearest_on_polyline(before.reshape(-1, 2), q)[1]
                    on_side |= (d <= 1e-9).reshape(before.shape[:2])
                assert on_side.any(), "no node recognised as held"
                moved = np.linalg.norm(after[on_side] - before[on_side],
                                       axis=1)
                assert float(np.max(moved)) < 1e-12
            checked += 1
        i += n
    assert checked > 0, "expected at least one wedge"


def test_wall_spacing_survives_assembly_smoothing(collar):
    """The anisotropy is the whole point of the collar, so smoothing must not
    be able to eat it even when it is applied."""
    grids = mesh_collar(collar, n_across=25, target_spacing=0.01,
                        first_cell=1e-4)
    first = min(float(np.min(nearest_on_polyline(g[:, 1, :],
                                                 collar["wall"])[1]))
                for g in grids if g.shape[1] > 2)
    assert first == pytest.approx(1e-4, rel=0.05)
