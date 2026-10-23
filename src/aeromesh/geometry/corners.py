"""Scale-invariant C0 corner detection.

A corner is a point where the tangent direction jumps. High curvature is not
a corner: a NACA 0012 leading edge turns through tens of degrees over a short
stencil, which any fixed-threshold detector flags, wrongly.

The discriminator is the *total turning* accumulated in a window of arc length
w centred on the point. Total turning is the integral of curvature plus any
tangent jumps inside the window, so

    smooth point of mean curvature k :  T(w) = k*w          ->  0
    C0 corner of exterior angle a    :  T(w) = a + k*w      ->  a

exactly, with no saturation at any w. A least-squares fit of ``T(w) = a + c*w``
and an extrapolation to w = 0 therefore separates the two cases and returns
the corner angle as a by-product.

The window range is derived from the loop's own sampling, so the detector has
no absolute length scale in it and no knowledge of what the loop represents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeromesh.geometry.loop import Loop


@dataclass(frozen=True)
class Corners:
    """Detected C0 corners on a loop.

    Attributes
    ----------
    index : np.ndarray, shape (K,)
        Vertex index of each corner in the loop's point array.
    turn : np.ndarray, shape (K,)
        Extrapolated signed exterior turn angle (radians). Positive turns
        left, i.e. convex with respect to the region the CCW loop encloses.
    residual : np.ndarray, shape (K,)
        RMS residual of the linear fit, in radians. A large value means the
        window straddled more than one feature; treat the angle as soft.
    """

    index: np.ndarray
    turn: np.ndarray
    residual: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    def positions(self, loop: Loop) -> np.ndarray:
        return loop.points[self.index]


def vertex_turns(loop: Loop) -> np.ndarray:
    """Signed turn angle at each vertex, between its incoming and outgoing edge.

    Sums to 2*pi over a simple CCW loop.
    """
    p = loop.points
    v_out = np.roll(p, -1, axis=0) - p
    v_in = p - np.roll(p, 1, axis=0)
    return np.arctan2(
        v_in[:, 0] * v_out[:, 1] - v_in[:, 1] * v_out[:, 0],
        v_in[:, 0] * v_out[:, 0] + v_in[:, 1] * v_out[:, 1],
    )


def _turning_interpolator(loop: Loop):
    """Return ``Phi(s)``: total turning accumulated from s = 0 to s.

    Defined for any real s, with ``Phi(s + L) = Phi(s) + 2*pi``.
    """
    phi = vertex_turns(loop)
    s = loop.arclength
    L = loop.perimeter
    # Turning is a step function; place each step at its vertex and read the
    # value just after it, so a window strictly containing a vertex sees it.
    cum = np.cumsum(phi)
    knots = np.concatenate(([0.0], s[1:], [L]))
    vals = np.concatenate(([cum[0]], cum[1:], [cum[-1]]))

    def phi_of(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        wraps = np.floor(q / L)
        return np.interp(q - wraps * L, knots, vals) + wraps * 2.0 * np.pi

    return phi_of


def detect_corners(
    loop: Loop,
    angle_tol: float = np.deg2rad(20.0),
    n_scales: int = 5,
    w_min_spacings: float = 3.0,
    w_max_spacings: float = 12.0,
    stability_tol: float = 0.7,
    resid_tol: float = 0.02,
) -> Corners:
    """Find C0 corners by extrapolating total turning to zero window width.

    Parameters
    ----------
    loop : Loop
        The curve to inspect. It should already be uniformly resampled; the
        window widths are derived from its mean point spacing.
    angle_tol : float
        Minimum extrapolated turn, in radians, to call a vertex a corner.
    n_scales : int
        Number of window widths used in the fit.
    w_min_spacings, w_max_spacings : float
        Window widths as multiples of the mean point spacing. The smallest
        window must be narrower than the arc-length gap between two distinct
        corners, or they merge into one detection with the summed angle.
        Resample more finely to resolve closely spaced corners.
    stability_tol : float
        A detection is window dependent if the angle fitted on the lower half
        of the window band is less than this fraction of the angle fitted on
        the upper half. Measured, real corners run 0.824 to 1.000 and smooth
        curvature peaks 0.265 to 0.608, so the default sits in the gap between
        them. Set to 0 to disable the check.
    resid_tol : float
        A detection is poorly fitted if its RMS residual exceeds this fraction
        of the mean total turning over the band. A true C0 corner fits exactly,
        so this is zero for one. A detection must be *both* window dependent
        and poorly fitted to be rejected as a smooth curvature peak.

    Returns
    -------
    Corners
        One entry per corner. Vertices flagged within one maximum window of a
        stronger neighbour are suppressed, so a corner yields one detection.
    """
    if n_scales < 2:
        raise ValueError("n_scales must be at least 2.")
    L = loop.perimeter
    delta = L / loop.n_points
    w = np.linspace(w_min_spacings * delta, w_max_spacings * delta, n_scales)
    if w[0] <= 0 or w[-1] >= 0.5 * L:
        raise ValueError("Window widths must be positive and well below the perimeter.")

    phi_of = _turning_interpolator(loop)
    s = loop.arclength
    T = np.array([phi_of(s + 0.5 * wk) - phi_of(s - 0.5 * wk) for wk in w])

    def _fit(widths, values):
        wm = widths.mean()
        dw = widths - wm
        c = (dw @ (values - values.mean(axis=0))) / float(np.sum(dw * dw))
        a = values.mean(axis=0) - c * wm
        r = np.sqrt(np.mean((values - (a + np.outer(widths, c))) ** 2, axis=0))
        return a, c, r

    a, c, resid = _fit(w, T)

    # A large extrapolated angle is not enough, because the model behind it --
    # T(w) = a + c*w, with no saturation at any w -- stops holding once the
    # window is wide compared with the local radius of curvature. Then all the
    # turning the nose has to give is already inside the narrowest window, T
    # flattens, and the fit reports the saturated total as an intercept. A 10:1
    # ellipse, which has no corner anywhere, is detected as a 38 degree corner
    # at 1200 samples; a 50:1 ellipse as 154 degrees. The docstring's claim of
    # no saturation is true of the continuum and false of a sampled curve.
    #
    # What separates the two is that a real corner's angle does not depend on
    # the window. Refining a square's sampling 16-fold leaves 90.0 degrees
    # untouched, while the 10:1 ellipse decays 74.7 -> 38.0 -> 12.4 -> 2.5 ->
    # 0.5. So fit the lower and upper halves of the band separately and require
    # agreement.
    #
    # On its own that would reject a genuine corner too. The two corners of a
    # blunt trailing edge sit a few samples apart, so the wider half of the
    # band straddles both and sums them -- 116.9 against 163.6, a ratio of
    # 0.71, on an aerofoil whose trailing edge is certainly a corner. The
    # discriminator there is the residual: straddling keeps T piecewise linear,
    # so the fit stays exact, while a saturating nose bends it. Measured on the
    # cambered aerofoil whose leading edge is the false positive this guards
    # against: ratio 0.28 with 14 per cent relative residual, against ratio
    # 0.71 with 0.0 per cent for the trailing edge it must not reject.
    #
    # Hence a detection is rejected only when it fails *both* -- window
    # dependent *and* poorly fitted.
    #
    # Where the thresholds come from, measured at the indices that survive
    # suppression rather than at their shoulders, which is a distinction that
    # cost a wrong first answer:
    #
    #     real   polygons, 3 to 12 sides, 600-2400 samples   1.000   0.0000
    #     real   two corners inside one window               1.000   0.0000
    #     real   crease in a per-body offset far field       0.824   0.0347
    #     FALSE  cambered aerofoil nose (6409, 6412)         0.265   0.0500
    #     FALSE  10:1 ellipse, 600 and 1200 samples          0.402   0.0551
    #     FALSE  20:1 ellipse, 2400 samples                  0.608   0.0464
    #
    # so the gap is between 0.608 and 0.824 and the threshold belongs in it.
    # 0.9 does not: it sits *inside* the real range and threw away the far-field
    # crease -- a corner the far-field tests exist to document -- while still
    # admitting coarser ellipses. That was caught by a test, not by reasoning.
    #
    # The thin ellipses that remain detected at coarse sampling are not a
    # failure of the rule. A 20:1 ellipse at 600 samples has a nose radius well
    # below its own point spacing, so no statistic computed from that polyline
    # can know the nose is round; the polyline really does turn through 131
    # degrees in a few segments. Refine it and the detection goes away on its
    # own, which is the only guarantee worth making here.
    mid = len(w) // 2
    a_lo = _fit(w[:mid + 1], T[:mid + 1])[0]
    a_hi = _fit(w[mid:], T[mid:])[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        stability = np.where(np.abs(a_hi) > 1e-12, a_lo / a_hi, 1.0)
        rel_resid = resid / np.maximum(np.mean(np.abs(T), axis=0), 1e-12)
    smooth_peak = (stability < stability_tol) & (rel_resid > resid_tol)

    # Judged *after* suppression, never before. Rejecting a candidate here lets
    # the next flagged sample along the same feature win the suppression
    # instead, so the feature survives with a shoulder standing in for its peak
    # and the rejection does nothing -- measured on a 10:1 ellipse, whose nose
    # was rejected at its peak and detected one sample over.
    flagged = np.flatnonzero(np.abs(a) > angle_tol)
    if flagged.size == 0:
        e = np.empty(0)
        return Corners(e.astype(int), e, e)

    # Non-maximum suppression over one maximum window of arc length.
    keep: list[int] = []
    for i in flagged[np.argsort(-np.abs(a[flagged]))]:
        ds = np.abs(s[i] - s[keep]) if keep else np.empty(0)
        if keep and np.min(np.minimum(ds, L - ds)) < w[-1]:
            continue
        keep.append(int(i))

    keep = [i for i in keep if not smooth_peak[i]]
    if not keep:
        e = np.empty(0)
        return Corners(e.astype(int), e, e)

    # Snap each kept index onto the vertex that actually carries the turn.
    #
    # The extrapolated turn is a *plateau* at a polygon corner: on a unit
    # triangle a[377] and a[378] are both exactly 120.000 degrees, because the
    # window straddles the vertex either way. The suppression above sorts by
    # |a| and numpy's sort is stable, so the tie goes to the lower index -- and
    # on that triangle index 377 has a vertex turn of 0.00 degrees while 378
    # carries the whole 120. The *angle* returned was right and the *position*
    # was one sample out, which is worse than it sounds: everything downstream
    # matches corners by position, so `corner_cuts` found no medial foot within
    # tolerance and gave up silently, `buried_corners` reported a corner that
    # was in fact resolved, and `canonicalise`'s guard failed to protect a real
    # 120 degree wall corner and merged it away. Ten of ten corners reported as
    # buried on polygonal bodies were this artefact.
    #
    # The search window is a few samples only. The offset is one sample, and a
    # blunt trailing edge has two genuine corners a handful of samples apart
    # that must not collapse onto each other.
    # Only where the turn is genuinely concentrated on one vertex. On a
    # polyline corner it is -- the whole 120 degrees sits on a single sample and
    # its neighbours carry nothing -- but on a smooth nose or a cusped trailing
    # edge the same total is spread over tens of samples, and there the window's
    # maximum is not a landmark. Snapping that would move the corner by up to
    # the window width for no reason.
    turns = np.abs(vertex_turns(loop))
    n = len(turns)
    taken: set = set()
    snapped: list[int] = []
    for i in sorted(keep):
        near = [(i + k) % n for k in range(-4, 5)]
        local = turns[np.asarray(near)]
        concentrated = float(np.max(local)) > 5.0 * float(np.median(local))
        if not concentrated:
            pick = int(i)
        else:
            order = sorted(near, key=lambda j: (-turns[j],
                                                abs((j - i + n // 2) % n
                                                    - n // 2)))
            pick = next((j for j in order if j not in taken), int(i))
        while pick in taken:
            pick = (pick + 1) % n
        taken.add(pick)
        snapped.append((pick, int(i)))

    keep_arr = np.array([pick for pick, _ in snapped], dtype=int)
    src = np.array([src_i for _, src_i in snapped], dtype=int)
    order = np.argsort(keep_arr)
    keep_arr, src = keep_arr[order], src[order]
    # The angle and the residual stay the extrapolated ones -- those are the
    # scale-invariant measurement; only the index moves.
    return Corners(index=keep_arr, turn=a[src], residual=resid[src])


def fluid_interior_angle(turn: np.ndarray | float, is_hole: bool) -> np.ndarray:
    """Interior angle of the *fluid* at a corner, from the loop's turn angle.

    For a CCW loop the enclosed region has interior angle ``pi - turn``. When
    the loop is a hole -- a solid body, with fluid outside it -- the fluid
    angle is ``pi + turn`` instead.

    A cusped trailing edge (turn = +pi) therefore has a fluid interior angle
    of 2*pi, and a right-angled far-field corner (turn = +pi/2, not a hole)
    has pi/2.
    """
    turn = np.asarray(turn, dtype=np.float64)
    return np.pi + turn if is_hole else np.pi - turn
