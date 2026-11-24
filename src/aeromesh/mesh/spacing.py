"""One-dimensional node distributions.

Everything here answers one question: *where do the nodes go along a curve*.
Two rules are enforced by construction rather than checked afterwards.

**Adjacent spacing is graded.** A size field ``h(s)`` is built from what the
geometry asks for (curvature, prescribed end spacings, a ceiling) and then
limited so that ``|dh/ds| <= ratio - 1``. Nodes placed at equal increments of
``integral ds / h`` then have neighbouring cells within ``ratio`` of each
other. The October audit measured 36x jumps at the trailing edge that came from
a per-block node count with a floor of 8; a size field has no such floor.

**The first cell height follows from the flow.** ``first_cell_height`` is the
flat-plate estimate used in the proposal; it is the only place y+ enters.
"""

from __future__ import annotations

import numpy as np


def first_cell_height(reynolds: float, yplus: float = 1.0,
                      chord: float = 1.0) -> float:
    """Wall-normal height of the first cell for a target y+.

    Uses ``Cf = 0.026 Re^(-1/7)`` and ``u_tau = U sqrt(Cf / 2)``, so that
    ``y1 = y+ * chord / (Re * sqrt(Cf / 2))``. Note that ``y+`` here is at the
    first *node* off the wall; a cell-centred solver sees half of it.
    """
    cf = 0.026 * float(reynolds) ** (-1.0 / 7.0)
    return float(yplus) * float(chord) / (float(reynolds) * np.sqrt(cf / 2.0))


def arclength(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0),
                                                           axis=1))))


def curvature(points: np.ndarray) -> np.ndarray:
    """Discrete curvature of an open polyline, one value per point."""
    p = np.asarray(points, dtype=float)
    seg = np.diff(p, axis=0)
    ds = np.maximum(np.linalg.norm(seg, axis=1), 1e-300)
    t = seg / ds[:, None]
    cross = t[:-1, 0] * t[1:, 1] - t[:-1, 1] * t[1:, 0]
    dot = np.einsum("ij,ij->i", t[:-1], t[1:])
    turn = np.abs(np.arctan2(cross, dot))
    k = turn / (0.5 * (ds[:-1] + ds[1:]))
    return np.concatenate(([k[0]], k, [k[-1]])) if len(k) else np.zeros(len(p))


def limit_gradation(h: np.ndarray, s: np.ndarray, ratio: float) -> np.ndarray:
    """Limit a size field so neighbouring cells differ by at most ``ratio``.

    For geometric growth ``h_{k+1} = ratio * h_k`` the size rises by
    ``(ratio - 1) * h`` over a distance ``h``, i.e. ``dh/ds = ratio - 1``.
    """
    h = np.array(h, dtype=float)
    beta = float(ratio) - 1.0
    ds = np.diff(s)
    for i in range(1, len(h)):
        h[i] = min(h[i], h[i - 1] + beta * ds[i - 1])
    for i in range(len(h) - 2, -1, -1):
        h[i] = min(h[i], h[i + 1] + beta * ds[i])
    return h


def graded_parameters(points: np.ndarray, h_max: float,
                      h_start: float | None = None,
                      h_end: float | None = None,
                      ratio: float = 1.2, turn_per_cell: float | None = None,
                      n: int | None = None) -> np.ndarray:
    """Arc-length positions of graded nodes along an open polyline.

    Parameters
    ----------
    points : (m, 2) fine polyline; its ends become nodes.
    h_max : ceiling on the spacing.
    h_start, h_end : spacing wanted at the two ends, if any.
    ratio : largest allowed ratio between neighbouring cells.
    turn_per_cell : if given (radians), the spacing is also capped at
        ``turn_per_cell / curvature`` so no cell turns the wall by more.
    n : number of cells. Derived from the size field when ``None``; when
        pinned, the field is scaled uniformly to give that count.

    Returns
    -------
    (n + 1,) arc lengths, from 0 to the polyline's length.
    """
    p = np.asarray(points, dtype=float)
    s_in = arclength(p)
    total = s_in[-1]
    if total <= 0.0:
        return np.zeros(2)
    h_in = np.full(len(p), float(h_max))
    if turn_per_cell is not None:
        k = curvature(p)
        # light smoothing: curvature of a sampled curve is noisy
        if len(k) > 7:
            k = np.convolve(np.pad(k, 3, mode="edge"), np.ones(7) / 7.0,
                            mode="valid")
        h_in = np.minimum(h_in, float(turn_per_cell) / np.maximum(k, 1e-300))

    # The size field is limited and integrated on its own grid, refined
    # geometrically towards any end that asks for a small spacing. Doing it on
    # the input polyline was wrong whenever that polyline was coarser than the
    # spacing requested: the limiter then acts between samples several cells
    # apart, and a 1e-4 end spacing on 7e-4 samples came out with neighbouring
    # cells in a ratio of 1.77 against a limit of 1.15.
    parts = [np.linspace(0.0, total, 20001)]
    for h_e, flip in ((h_start, False), (h_end, True)):
        if h_e is not None and h_e < total:
            g = np.geomspace(min(float(h_e), total) / 50.0, total, 2000)
            parts.append(total - g if flip else g)
    s = np.unique(np.clip(np.concatenate(parts), 0.0, total))
    h = np.interp(s, s_in, h_in)
    if h_start is not None:
        h[0] = min(h[0], float(h_start))
    if h_end is not None:
        h[-1] = min(h[-1], float(h_end))
    h = limit_gradation(h, s, ratio)

    inv = 1.0 / h
    m = np.concatenate(([0.0], np.cumsum(0.5 * (inv[:-1] + inv[1:])
                                         * np.diff(s))))
    cells = max(1, int(round(m[-1]))) if n is None else max(1, int(n))
    return np.interp(np.linspace(0.0, m[-1], cells + 1), m, s)


def layer_heights(first, n_layers: int, growth: float = 1.15,
                  distance: float | None = None,
                  max_growth: float = 1.3) -> np.ndarray:
    """Wall-normal cell heights, shape ``(len(first), n_layers)``.

    A single global progression ``H_j = y1 * growth^j`` is shared by every
    station, where ``y1`` is the smallest first height. A station asking for a
    larger first height ``f_i`` simply adds the excess to every layer:
    ``h_ij = H_j + (f_i - y1)``. The wall gets the exact geometric law and a
    wake cut starts coarser and blends into it.

    The blend is deliberately the slowest one available. A sharp one (a
    fourth-power smooth maximum was tried first) makes a station's height
    switch from its own ``f_i`` to the shared ``H_j`` within a few layers, at a
    different layer for each station; neighbouring stations then taper
    differently from one layer to the next, and on wake cells of aspect ratio
    in the hundreds that is finite-volume non-orthogonality -- 82 degrees
    measured, by ``checkMesh`` and by ``fv_nonorthogonality`` alike.

    When ``distance`` is given the growth of the outer layers is raised (up to
    ``max_growth``) so the total reaches it in exactly ``n_layers``; the first
    layers, where the boundary layer lives, keep ``growth`` untouched.

    """
    f = np.atleast_1d(np.asarray(first, dtype=float))
    y1 = float(f.min())
    n = int(n_layers)

    def progression(g_out: float, n_bl: int) -> np.ndarray:
        j = np.arange(n)
        g = np.where(j < n_bl, growth, g_out)
        return y1 * np.concatenate(([1.0], np.cumprod(g[1:])))

    H = progression(growth, n)
    if distance is not None and H.sum() < distance:
        n_bl = max(2, int(0.35 * n))
        lo, hi = float(growth), float(max_growth)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if progression(mid, n_bl).sum() < distance:
                lo = mid
            else:
                hi = mid
        H = progression(hi, n_bl)
    elif distance is not None:
        # more layers than the distance needs: cut the growth of the outer
        # part. The boundary-layer law is kept only as far as a physical
        # thickness (5 % of the distance scale, at most 0.05 chord), not for a
        # fixed share of the layers: with 300 layers, 35 % of them at 1.15
        # already reach ~40 chords and no outer law could bring the total back
        # to 500 -- the level-3 mesh landed at 889 chords.
        H0 = progression(growth, n)
        n_bl = int(np.searchsorted(np.cumsum(H0), min(0.05, 0.05 * distance)))
        n_bl = max(2, min(n_bl, int(0.35 * n)))
        lo, hi = 1.0, float(growth)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if progression(mid, n_bl).sum() < distance:
                lo = mid
            else:
                hi = mid
        H = progression(hi, n_bl)
    return H[None, :] + (f[:, None] - y1)
