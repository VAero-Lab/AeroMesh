"""Hyperbolic grid generation: a mesh marched from a curve.

Solves, layer by layer, the two equations a boundary-layer mesh actually wants
(Steger & Chaussee 1980; Chan & Steger 1992):

    r_xi . r_eta = 0                         orthogonality
    x_xi y_eta - x_eta y_xi = V              prescribed cell area

``xi`` runs along the curve and ``eta`` away from it. Linearised about the
known layer and written for the increment ``dr = r_{j+1} - r_j``:

    [ I + (1 + theta) C delta_xi - eps_i Lap_xi ] dr = g + eps_e Lap_xi r_j

with ``C = B^-1 A``, ``g = (V / |r_xi|^2) (-y_xi, x_xi)`` -- the normal step --
and second-difference dissipation. The system is a periodic (or open)
block-tridiagonal one in ``xi`` and is solved directly.

Why this and not the explicit normal march of ``blocking.collar.march_collar``:
the explicit march has no area equation, so where the wall is concave its lines
converge and nothing stops them crossing. Here the cell area is an equation,
and a converging pair of lines is pushed apart by it.

Conventions
-----------
The curve is traversed with the region to be meshed on its **left**. For a
body that means clockwise. The returned grid has shape ``(n, n_layers + 1, 2)``
and its quads ``(i, j) -> (i+1, j) -> (i+1, j+1) -> (i, j+1)`` are
counter-clockwise, i.e. have positive area.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


def _xi_diff(r: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return 0.5 * (np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
    d = np.empty_like(r)
    d[1:-1] = 0.5 * (r[2:] - r[:-2])
    d[0] = r[1] - r[0]
    d[-1] = r[-1] - r[-2]
    return d


def _lap(r: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        return np.roll(r, -1, axis=0) - 2.0 * r + np.roll(r, 1, axis=0)
    d = np.zeros_like(r)
    d[1:-1] = r[2:] - 2.0 * r[1:-1] + r[:-2]
    return d


def _smooth(v: np.ndarray, passes: int, periodic: bool,
            nu: float = 0.25) -> np.ndarray:
    v = np.array(v, dtype=float)
    for _ in range(int(passes)):
        if periodic:
            v = v + nu * (np.roll(v, -1) - 2.0 * v + np.roll(v, 1))
        else:
            w = v.copy()
            w[1:-1] = v[1:-1] + nu * (v[2:] - 2.0 * v[1:-1] + v[:-2])
            v = w
    return v


def march(curve: np.ndarray, heights: np.ndarray, periodic: bool = True,
          theta: float = 1.0, eps: float = 0.5, eps_wall: float = 0.0,
          volume_smoothing: float = 0.3, end_directions=None) -> np.ndarray:
    """March a grid outward from ``curve``.

    Parameters
    ----------
    curve : (n, 2) nodes of the starting line, region to be meshed on the left.
        For ``periodic=True`` the first node is not repeated.
    heights : (n, n_layers) cell height wanted at each node in each layer.
    periodic : closed curve (O-type) or open curve (C-type).
    theta : implicitness of the ``xi`` coupling (0 = Crank-Nicolson-like,
        1 = fully implicit plus one). Larger is more robust.
    eps : dissipation coefficient far from the wall.
    eps_wall : dissipation coefficient in the first layer. Zero keeps the first
        cell height exact; the coefficient ramps to ``eps`` as
        ``sqrt(j / n_layers)`` and is scaled by the local cell aspect ratio, so
        it is negligible where cells are thin and strong where lines could
        cross.
    volume_smoothing : how strongly the along-curve cell size is averaged as
        the march proceeds (0 = never). This is what lets a clustered wall
        distribution relax to an even one in the far field.
    end_directions : for an open curve, the unit vectors along which the two
        end nodes slide, shape (2, 2). The ends keep their step length and are
        confined to these lines (an outflow plane, for a C-mesh).

    Returns
    -------
    (n, n_layers + 1, 2) grid.
    """
    r = np.asarray(curve, dtype=float).copy()
    heights = np.asarray(heights, dtype=float)
    n, n_layers = heights.shape
    if len(r) != n:
        raise ValueError("heights must have one row per curve node")

    idx = np.arange(n)
    ip = (idx + 1) % n if periodic else np.minimum(idx + 1, n - 1)
    im = (idx - 1) % n if periodic else np.maximum(idx - 1, 0)

    rows = [r.copy()]
    for j in range(n_layers):
        h = heights[:, j]
        r_xi = _xi_diff(r, periodic)
        gamma = np.maximum(np.einsum("ij,ij->i", r_xi, r_xi), 1e-300)
        ds = np.sqrt(gamma)

        # --- cell area: local along-curve size times the wanted height. Far
        # from the wall the along-curve size is progressively averaged, which
        # is what relaxes a clustered wall distribution to an even one.
        #
        # The averaging is weighted by the cell's own aspect, h / ds, and that
        # weighting is not cosmetic. Averaging ds changes the height actually
        # marched, h * ds_v / ds, differently at neighbouring stations wherever
        # the wall spacing is graded, so cells taper; and on a thin cell a
        # taper moves the centroid along the cell by more than the cell is
        # tall. Applied uniformly it gave 35 degrees of finite-volume
        # non-orthogonality at j = 10 over a leading edge, inside the boundary
        # layer, on cells whose corners were all within 5 degrees of square.
        # Thin cells are therefore left exactly as prescribed.
        frac = (j + 1) / n_layers
        passes = int(round(volume_smoothing * 60.0 * frac ** 1.5))
        lam = np.clip(h / ds, 0.0, 1.0) ** 2
        ds_v = ds + lam * (_smooth(ds, passes, periodic) - ds)
        vol = ds_v * h
        # The area equation fixes the cell's *area*, and on a curved level the
        # cell is a sector, not a rectangle: its far side is longer (convex)
        # or shorter (concave) than its near side by the factor 1 + kappa h.
        # Asking for area ds * h therefore gives a height below h on a convex
        # wall -- measured on a circle, 0.9804 of the request at h / R = 0.01
        # with theta = 1, and 2.5 per cent at a leading edge for a 1e-4 first
        # cell. The linearised implicit scheme shrinks the step by
        # 1 / (1 + (1 + theta) kappa h), so the area is asked for with that
        # factor put back and the height comes out as prescribed.
        t_hat = r_xi / ds[:, None]
        dt = _xi_diff(t_hat, periodic)
        kappa = -(t_hat[:, 0] * dt[:, 1] - t_hat[:, 1] * dt[:, 0]) / ds
        # Smoothed and bounded: the correction feeds the level's own curvature
        # back into its step, and unbounded that is an instability -- the
        # first attempt, clipped at +0.5, sent the outer boundary to x = -1e11.
        kappa = _smooth(kappa, 4, periodic)
        # It is also confined to thin cells (the same weight that switches the
        # averaging off): bounded at 5 per cent everywhere it still crossed
        # lines in the far field above a trailing edge, 377 inverted cells,
        # and the far field has no first-cell height to protect.
        vol = vol * (1.0 + (1.0 - lam) * (1.0 + theta)
                     * np.clip(kappa * h, -0.05, 0.05))

        x_xi, y_xi = r_xi[:, 0], r_xi[:, 1]
        x_eta = -y_xi * vol / gamma
        y_eta = x_xi * vol / gamma

        # C = B^-1 A
        c11 = (x_xi * x_eta - y_xi * y_eta) / gamma
        c12 = (x_xi * y_eta + y_xi * x_eta) / gamma
        c21 = c12
        c22 = -c11

        g = np.column_stack([x_eta, y_eta])

        # --- dissipation, scaled by cell aspect (step / along-curve size) and
        # ramped from the wall.
        aspect = np.clip(h / ds, 0.0, 1.0)
        ramp = eps_wall + (eps - eps_wall) * np.sqrt(j / max(n_layers - 1, 1))
        e_e = ramp * aspect
        e_i = 2.0 * e_e
        rhs = g + e_e[:, None] * _lap(r, periodic)

        # --- assemble the 2n x 2n system
        a = 0.5 * (1.0 + theta)
        R, Cc, D = [], [], []

        def put(row, col, val):
            R.append(row); Cc.append(col); D.append(val)

        interior = np.ones(n, dtype=bool)
        if not periodic:
            interior[[0, -1]] = False
        ii = idx[interior]
        for comp_r, (ca, cb) in enumerate(((c11, c12), (c21, c22))):
            row = 2 * ii + comp_r
            # diagonal
            put(row, 2 * ii + comp_r, 1.0 + 2.0 * e_i[ii])
            # +1 neighbour
            put(row, 2 * ip[ii] + 0, a * ca[ii] - (e_i[ii] if comp_r == 0 else 0.0))
            put(row, 2 * ip[ii] + 1, a * cb[ii] - (e_i[ii] if comp_r == 1 else 0.0))
            # -1 neighbour
            put(row, 2 * im[ii] + 0, -a * ca[ii] - (e_i[ii] if comp_r == 0 else 0.0))
            put(row, 2 * im[ii] + 1, -a * cb[ii] - (e_i[ii] if comp_r == 1 else 0.0))

        b = rhs.reshape(-1).copy()
        if not periodic:
            ends = ((0, 1), (n - 1, n - 2))
            dirs = (np.asarray(end_directions, dtype=float)
                    if end_directions is not None else None)
            for k, (e, nb) in enumerate(ends):
                if dirs is None:
                    d = g[e] / max(np.linalg.norm(g[e]), 1e-300)
                else:
                    d = dirs[k] / np.linalg.norm(dirs[k])
                # the end node moves along d by the height of its own layer
                put(np.array([2 * e]), np.array([2 * e]), 1.0)
                put(np.array([2 * e + 1]), np.array([2 * e + 1]), 1.0)
                b[2 * e] = h[e] * d[0]
                b[2 * e + 1] = h[e] * d[1]

        A = csc_matrix((np.concatenate([np.broadcast_to(v, np.shape(rw)).ravel()
                                        for v, rw in zip(D, R)]),
                        (np.concatenate([np.ravel(x) for x in R]),
                         np.concatenate([np.ravel(x) for x in Cc]))),
                       shape=(2 * n, 2 * n))
        dr = spsolve(A, b).reshape(n, 2)
        r = r + dr
        rows.append(r.copy())

    return np.stack(rows, axis=1)
