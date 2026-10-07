"""Opening a cusp into a finite face.

A cusp -- a boundary corner whose solid interior angle is near zero -- is the
degenerate case for a medial axis: the branch that should emanate from it is
tangent to both surfaces and numerically ill-conditioned. A finite face is the
well-posed case.

This module cuts a loop with a line perpendicular to the cusp's own axis and
closes the gap with a straight face. Everything is stated in terms of corners,
arc length and local tangents, so it applies to any sharp tip -- an airfoil
trailing edge, a slot lip, a fin -- and contains nothing airfoil-specific.
"""

from __future__ import annotations

import numpy as np

from aeromesh.geometry.corners import Corners, detect_corners, vertex_turns
from aeromesh.geometry.loop import Loop


def cusp_axis(loop: Loop, s_tip: float, probe: float | None = None) -> np.ndarray:
    """Unit vector pointing from a cusp tip into the body it terminates.

    Taken as the direction from the tip to the midpoint of two points an equal
    arc length away on either side -- the bisector of the two tangents, which
    for a cusp is the axis of the sliver.
    """
    if probe is None:
        probe = 0.01 * loop.perimeter
    tip = loop.interpolate(np.array([s_tip]))[0]
    a = loop.interpolate(np.array([s_tip - probe]))[0]
    b = loop.interpolate(np.array([s_tip + probe]))[0]
    axis = 0.5 * (a + b) - tip
    n = np.linalg.norm(axis)
    if n < 1e-14:
        raise ValueError("Degenerate cusp axis; the two sides are coincident.")
    return axis / n


def _cut_points(loop: Loop, s_tip: float, axis: np.ndarray, depth: float,
                max_arc: float) -> tuple[float, float] | None:
    """Arc positions where each side of the cusp first reaches ``depth`` along
    ``axis``, measured from the tip. Returns ``(s_back, s_fwd)`` or None."""
    tip = loop.interpolate(np.array([s_tip]))[0]
    n = 4000
    out = []
    for sign in (-1.0, +1.0):
        s = s_tip + sign * np.linspace(0.0, max_arc, n)
        proj = (loop.interpolate(s) - tip) @ axis
        hit = np.flatnonzero(proj >= depth)
        if hit.size == 0:
            return None
        k = int(hit[0])
        if k == 0:
            out.append(float(s[0]))
            continue
        t = (depth - proj[k - 1]) / (proj[k] - proj[k - 1])
        out.append(float(s[k - 1] + t * (s[k] - s[k - 1])))
    return out[0], out[1]


def open_cusp(
    loop: Loop,
    index: int,
    thickness: float,
    n: int | None = None,
    n_face: int = 9,
    max_trim_frac: float = 0.25,
) -> tuple[Loop, np.ndarray]:
    """Cut a loop back from the cusp at ``index`` and close it with a face.

    The cut is perpendicular to the cusp's axis, so the two corners it creates
    are symmetric for a symmetric tip and the face length is exactly
    ``thickness``.

    Parameters
    ----------
    loop : Loop
        The curve to modify.
    index : int
        Vertex index of the cusp, as reported by ``detect_corners``.
    thickness : float
        Face length, in the same units as the loop's coordinates.
    n : int, optional
        Point count of the returned loop, excluding the face. Defaults to
        keeping the input loop's spacing.
    n_face : int
        Number of points on the new face, endpoints included.
    max_trim_frac : float
        Safety limit on how much of the perimeter may be trimmed per side.

    Returns
    -------
    loop : Loop
        The trimmed loop with the face inserted, wound CCW.
    face_index : np.ndarray, shape (2,)
        Vertex indices of the two ends of the face. These are exact C0 corners
        by construction; they do not need to be re-detected, and for a thin
        face they may be too close together for detection to separate them.

    Raises
    ------
    ValueError
        If ``thickness`` cannot be reached within the trim limit, meaning the
        feature is not a cusp at that scale.
    """
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    if n_face < 2:
        raise ValueError("n_face must be at least 2.")

    L = loop.perimeter
    s_tip = loop.arclength[index]
    axis = cusp_axis(loop, s_tip)
    max_arc = max_trim_frac * L

    tip = loop.interpolate(np.array([s_tip]))[0]
    # Both sides must be able to reach the cut depth, so take the smaller of
    # the two per-side maxima -- not the maximum over both.
    reach = min(
        float(np.max((loop.interpolate(
            s_tip + sign * np.linspace(0.0, max_arc, 2001)) - tip) @ axis))
        for sign in (-1.0, +1.0)
    )
    if reach <= 0.0:
        raise ValueError("Cusp axis points away from the boundary; not a cusp.")

    lo, hi = 0.0, reach * (1.0 - 1e-9)
    widest = _cut_points(loop, s_tip, axis, hi, max_arc)
    if widest is None:
        raise ValueError("Cusp axis does not intersect the boundary; not a cusp.")
    gap_hi = float(np.linalg.norm(loop.interpolate(np.array([widest[1]]))[0]
                                  - loop.interpolate(np.array([widest[0]]))[0]))
    if gap_hi < thickness:
        raise ValueError(
            f"Cannot open a face of {thickness:.4g} within {max_trim_frac:.0%} of "
            f"the perimeter; the widest reachable gap is {gap_hi:.4g}. "
            f"This feature is not a cusp at that scale."
        )

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cp = _cut_points(loop, s_tip, axis, mid, max_arc)
        if cp is None:
            hi = mid
            continue
        p0 = loop.interpolate(np.array([cp[0]]))[0]
        p1 = loop.interpolate(np.array([cp[1]]))[0]
        if float(np.linalg.norm(p1 - p0)) < thickness:
            lo = mid
        else:
            hi = mid
    s_back, s_fwd = _cut_points(loop, s_tip, axis, 0.5 * (lo + hi), max_arc)

    keep_len = (s_back + L) - s_fwd
    if n is None:
        n = loop.n_points
    n_keep = max(8, int(round(n * keep_len / L)))
    sample = np.linspace(0.0, keep_len, n_keep)

    # Preserve any sharp vertices already on the kept arc -- the faces of cusps
    # opened earlier on the same loop. Without this, a second cut resamples the
    # first face's corners away and they stop being exact.
    sharp = loop.arclength[loop.sharp_vertices()]
    rel = np.mod(sharp - s_fwd, L)
    rel = rel[(rel > 1e-12) & (rel < keep_len - 1e-12)]
    if rel.size:
        step = keep_len / max(n_keep - 1, 1)
        gap = np.min(np.abs(sample[:, None] - rel[None, :]), axis=1)
        sample = np.sort(np.concatenate([sample[gap > 0.5 * step], rel]))

    body = loop.interpolate(s_fwd + sample)
    t = np.linspace(0.0, 1.0, n_face)[1:-1]
    face_pts = body[-1] + np.outer(t, body[0] - body[-1])

    pts = np.vstack([body, face_pts])
    out = Loop.from_points(pts, name=loop.name)
    if out.n_points != len(pts):
        raise RuntimeError("Cusp face collapsed under de-duplication; "
                           "increase thickness or reduce n_face.")
    return out, np.array([0, len(body) - 1], dtype=int)


def find_cusps(
    loop: Loop,
    corners: Corners | None = None,
    max_solid_angle: float = np.deg2rad(40.0),
    exclude: np.ndarray | None = None,
    exclude_radius: float = 0.0,
) -> np.ndarray:
    """Indices of corners sharp enough to count as cusps.

    The solid interior angle at a corner of a CCW loop is ``pi - turn``, so a
    near-zero solid angle means a turn close to pi.

    Parameters
    ----------
    loop : Loop
        The curve to inspect.
    corners : Corners, optional
        Pre-computed corners; detected if omitted.
    max_solid_angle : float
        Largest solid angle still counted as a cusp.
    exclude : np.ndarray, shape (M, 2), optional
        Points near which candidates are ignored. Use this to hide faces that
        have already been opened: when a face is narrower than the detector's
        smallest window its two corners cannot be separated, and it is then
        reported as one corner whose turn is their sum -- which is
        indistinguishable from a cusp. The two cases cannot be told apart from
        the geometry at that sampling, so the caller, which knows what it built,
        has to say.
    exclude_radius : float
        Radius of that exclusion, in the loop's own units.

    Returns
    -------
    np.ndarray
        Vertex indices of the cusps, in loop order.
    """
    if corners is None:
        corners = detect_corners(loop)
    if len(corners) == 0:
        return np.empty(0, dtype=int)

    idx = corners.index[(np.pi - corners.turn) < max_solid_angle]
    if exclude is None or len(exclude) == 0 or idx.size == 0:
        return idx

    d = np.linalg.norm(loop.points[idx][:, None, :] - np.asarray(exclude)[None, :, :],
                       axis=2)
    return idx[d.min(axis=1) > exclude_radius]
