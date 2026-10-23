"""One degenerate medial sample must not zero the thickness field.

``medial_radius_on_boundary`` takes the *minimum* medial radius over the medial
samples touching each boundary sample, which is right -- a larger circle through
the same point is not the maximal one there -- but it makes the field maximally
sensitive to a single bad sample. The medial axis of a polygon approximating a
smooth curve grows a spurious branch at essentially every vertex, so bad samples
exist, and one of them is enough.
"""

from __future__ import annotations

import numpy as np
import pytest

import aeromesh as am
from aeromesh.blocking.layers import medial_radius_on_boundary, wall_distance
from aeromesh.medial.axis import exterior_axis


@pytest.fixture(scope="module")
def cambered():
    """NACA 6409: the geometry that exposed it."""
    body = am.load_airfoil("6409", n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    return region, exterior_axis(region)


def test_the_degenerate_sample_is_actually_present(cambered):
    """Pin the cause, so a fix upstream in the axis does not silently make
    this file test nothing."""
    _, axis = cambered
    r = np.concatenate([np.asarray(d["r_m"], dtype=float)
                        for _, _, d in axis.graph.edges(data=True)])
    th = np.concatenate([np.asarray(d["theta_m"], dtype=float)
                         for _, _, d in axis.graph.edges(data=True)])
    bad = (r <= 1e-9) & (th <= 1e-9)
    assert bad.sum() >= 1, "the degenerate medial sample is gone -- retarget"
    assert bad.sum() < 0.01 * len(r), "many degenerate samples: a worse problem"


@pytest.mark.parametrize("name", ["0012", "2412", "4415", "6409", "6412",
                                  "23012", "63012", "4412"])
def test_boundary_radius_is_positive_everywhere(name):
    body = am.load_airfoil(name, n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    axis = exterior_axis(region)
    r_m = medial_radius_on_boundary(axis, region)[1]
    assert np.all(np.isfinite(r_m))
    assert float(np.min(r_m)) > 0.0
    # Nothing below the sampling can be resolved, so nothing should be reported
    # there either.
    pts = region.boundaries[1].loop.points
    spacing = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    assert float(np.min(r_m)) > spacing


@pytest.mark.parametrize("name", ["6409", "6412"])
def test_thickness_is_not_cancelled_by_the_cap(name):
    """The consequence that mattered: delta went to zero, so the collar's
    outer curve touched the wall."""
    body = am.load_airfoil(name, n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    axis = exterior_axis(region)
    delta = wall_distance(axis, region, delta_target=0.02, rho_max=0.05)[1]
    assert float(np.min(delta)) == pytest.approx(0.02, rel=1e-9), (
        "the medial cap is biting on a lone aerofoil, where there is no "
        "neighbour to make room for")


def test_a_genuinely_small_radius_is_still_reported():
    """The filter must not swallow a real minimum.

    Two bodies close together have a true medial radius minimum in the slot,
    orders of magnitude above the point spacing. If the degeneracy filter were
    a blanket floor on r_m rather than a resolution floor, this would move.
    """
    main = am.load_airfoil("2412", n_points=300).transform(name="main")
    flap = am.load_airfoil("2412", n_points=300).transform(
        scale=0.3, angle=-25.0, dx=1.02, dy=-0.08, name="flap")
    region = am.build_region([main, flap],
                             outer=am.circle_farfield([main, flap], 10.0),
                             te="blunt")
    axis = exterior_axis(region)
    radii = medial_radius_on_boundary(axis, region)

    body_min = min(float(np.min(radii[li])) for li in (1, 2))
    far_max = float(np.max(radii[1]))
    assert body_min > 0.0
    # The slot really does collapse r_m: that is the behaviour the capped
    # thickness law exists to handle, and it must survive the filter.
    assert far_max / body_min > 10.0
