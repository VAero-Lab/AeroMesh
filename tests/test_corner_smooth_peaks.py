"""A smooth peak of high curvature is not a corner.

The detector extrapolates total turning to zero window width on the model
T(w) = a + c*w. That model has no saturation in the continuum, but on a
*sampled* curve it saturates as soon as the window is wide compared with the
local radius of curvature, and then the fit reports the saturated total as a
corner angle. These tests pin the two statistics that separate the cases and,
more importantly, pin the convergence: refine the sampling and the false
positive must go away on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

import aeromesh as am
from aeromesh.geometry.corners import detect_corners
from aeromesh.geometry.loop import Loop


def polygon(n_vertices: int, n_samples: int = 1200) -> Loop:
    th = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    P = np.column_stack([np.cos(th), np.sin(th)])
    P = np.vstack([P, P[:1]])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    c = np.concatenate(([0.0], np.cumsum(d)))
    u = np.linspace(0.0, c[-1], n_samples, endpoint=False)
    return Loop(np.column_stack([np.interp(u, c, P[:, 0]),
                                 np.interp(u, c, P[:, 1])]))


def ellipse(aspect: float, n_samples: int = 1200) -> Loop:
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    return Loop(np.column_stack([np.cos(t), np.sin(t) / aspect]))


@pytest.mark.parametrize("n_vertices", [3, 4, 5, 6, 8, 12])
@pytest.mark.parametrize("n_samples", [600, 1200, 2400])
def test_polygon_corners_all_survive(n_vertices, n_samples):
    """The rejection must not cost a single genuine corner."""
    corners = detect_corners(polygon(n_vertices, n_samples))
    assert len(corners) == n_vertices
    expected = 2.0 * np.pi / n_vertices
    assert np.allclose(np.abs(corners.turn), expected, atol=np.deg2rad(0.5))
    # A true C0 corner fits the model exactly.
    assert float(np.max(corners.residual)) < np.deg2rad(0.05)


@pytest.mark.parametrize("aspect", [1.0, 2.0, 4.0, 10.0])
@pytest.mark.parametrize("n_samples", [600, 1200, 2400])
def test_smooth_ellipse_has_no_corners(aspect, n_samples):
    assert len(detect_corners(ellipse(aspect, n_samples))) == 0


def test_thin_ellipse_false_positive_vanishes_under_refinement():
    """The one honest guarantee: it is resolution, not a threshold.

    A 20:1 ellipse has a nose radius of 1/400 against a mean spacing of
    1/230 at 600 samples, so the polyline genuinely does turn through 131
    degrees in a few segments and no statistic computed from it can know the
    nose is round. That is not a detector failure, it is what the data says.
    What must hold is that refining removes it, and that it never comes back
    -- which a tuned threshold could not promise.
    """
    found = [len(detect_corners(ellipse(20.0, n)))
             for n in (600, 1200, 2400, 4800)]
    assert found[0] > 0, "expected the under-resolved nose to be flagged"
    assert found[-1] == 0
    assert found == sorted(found, reverse=True), found


def test_detection_is_monotone_in_sampling_for_a_thinner_ellipse():
    """Same statement one step thinner, where it takes longer to resolve."""
    found = [len(detect_corners(ellipse(50.0, n)))
             for n in (1200, 9600, 19200)]
    assert found[0] > 0 and found[-1] == 0, found


def test_the_gate_is_applied_after_suppression_not_before():
    """Rejecting a candidate must remove the feature, not shift it.

    Judged before non-maximum suppression, a rejected peak simply lets the
    next flagged sample along the same nose win instead, and the detection
    survives one sample over having learnt nothing. Measured on exactly this
    ellipse: the gate fired and the count stayed at 2.
    """
    assert len(detect_corners(ellipse(10.0, 600))) == 0


@pytest.mark.parametrize("name", ["0012", "2412", "4415", "6409", "6412",
                                  "23012", "63012", "4412"])
@pytest.mark.parametrize("te,expected", [("blunt", 2), ("sharp", 1)])
def test_aerofoil_corners_are_the_trailing_edge_only(name, te, expected):
    """The regression this was written for.

    NACA 6409 -- 6 per cent camber on 9 per cent thickness -- had its
    *leading* edge detected as a 20.5 degree corner, which put three extra
    rays and two extra wedges into the collar and produced a folded mesh:
    minimum angle 0.5 degrees, aspect ratio 4779, negative Jacobians. No
    other aerofoil in the battery showed it, so nothing caught it until the
    battery was widened.
    """
    body = am.load_airfoil(name, n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te=te)
    hole = region.holes[0]
    corners = hole.corners
    assert len(corners) == expected
    x = hole.loop.points[corners.index, 0]
    assert np.all(x > 0.5), f"corner away from the trailing edge at x={x}"


def test_two_detected_corners_close_together_both_survive():
    """The case that forbids rejecting on window dependence alone.

    When two genuine corners sit closer than the widest window, the wide half
    of the band straddles both and sums them, so the lower and upper halves
    disagree and the stability ratio falls well below one -- on corners that
    are unambiguously real. What saves them is the residual: straddling leaves
    the turning profile piecewise linear, so the fit stays exact, while a
    saturating nose bends it. Rejection therefore requires *both* failures.

    Built as a wedge with a flat back, the back's length set to a few mean
    spacings so the straddle actually happens.
    """
    spacing = 2.2 / 1200.0
    gap = 18.0 * spacing          # separable, but inside the widest window
    P = np.array([[0.0, 0.0], [1.0, 0.5 * gap], [1.0, -0.5 * gap]])
    Q = np.vstack([P, P[:1]])
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    c = np.concatenate(([0.0], np.cumsum(d)))
    u = np.linspace(0.0, c[-1], 1200, endpoint=False)
    loop = Loop(np.column_stack([np.interp(u, c, Q[:, 0]),
                                 np.interp(u, c, Q[:, 1])]))

    corners = detect_corners(loop)
    assert len(corners) == 3, np.degrees(np.abs(corners.turn))
    # The two back corners are the right angles; the apex is the sharp one.
    angles = np.sort(np.degrees(np.abs(corners.turn)))
    assert angles[0] == pytest.approx(angles[1], abs=1.0)
    assert angles[0] == pytest.approx(90.0, abs=4.0)
    assert float(np.max(corners.residual)) < np.deg2rad(0.05)


def test_aerofoil_trailing_edge_corners_are_constructed_not_detected():
    """Why the aerofoil cases cannot stand in for the test above.

    ``_merge_corners`` injects the trailing-edge corners from the geometry that
    built them and lets the detector contribute only corners far from those, so
    a blunt trailing edge's corners carry residual 0.0 by assignment rather than
    by measurement. The rejection rule is therefore never exercised on them --
    which is exactly why it needs its own geometry, above, and why the aerofoil
    tests here only pin that no *extra* corner survives.
    """
    body = am.load_airfoil("2412", n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    hole = region.holes[0]
    assert len(hole.corners) == 2
    assert np.allclose(hole.corners.residual, 0.0)
    # The raw detector, with no injection, finds the trailing edge as one
    # feature -- the two corners are closer than it can separate.
    assert len(detect_corners(hole.loop)) == 1


def test_a_real_crease_in_a_far_field_is_not_rejected():
    """The case that set the threshold, and the one that caught it wrong.

    Offsetting each body separately rather than their hull leaves a genuine
    crease where the two offsets meet -- shallow, around 20 to 24 degrees, and
    slightly rounded by the contouring that produced the level set, so it looks
    statistically like a saturating nose. It is not one, and the far-field
    tests exist to document that it is there.

    With the threshold at 0.9 this crease was thrown away: it measures 0.824 to
    0.890 on stability, which is *inside* the range real corners occupy. The
    gap in the measured data is between 0.608 and 0.824.
    """
    from aeromesh.domain.farfield import offset_farfield

    main = am.load_airfoil("2412", n_points=300).transform(name="main")
    flap = main.transform(scale=0.32, angle=np.deg2rad(-28), dx=1.02,
                          dy=-0.10, name="flap")
    for distance in (2.0, 3.0):
        ff = offset_farfield([main, flap], distance, hull=False)
        found = detect_corners(ff.resample(spacing=ff.scale / 500))
        assert len(found) > 0, f"crease lost at distance {distance}"


def test_rejection_can_be_switched_off():
    """``stability_tol=0`` restores the previous, permissive behaviour."""
    body = am.load_airfoil("6409", n_points=400).transform(name="m")
    region = am.build_region([body],
                             outer=am.circle_farfield([body], 8.0), te="blunt")
    loop = region.holes[0].loop

    kept = detect_corners(loop)
    assert np.all(loop.points[kept.index, 0] > 0.5), "leading edge not rejected"

    permissive = detect_corners(loop, stability_tol=0.0)
    assert len(permissive) == len(kept) + 1
    x = loop.points[permissive.index, 0]
    assert np.any(x < 0.5), "expected the spurious leading-edge corner back"
