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

    wm = w.mean()
    dw = w - wm
    c = (dw @ (T - T.mean(axis=0))) / float(np.sum(dw * dw))
    a = T.mean(axis=0) - c * wm
    resid = np.sqrt(np.mean((T - (a + np.outer(w, c))) ** 2, axis=0))

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

    keep_arr = np.array(sorted(keep), dtype=int)
    return Corners(index=keep_arr, turn=a[keep_arr], residual=resid[keep_arr])


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
