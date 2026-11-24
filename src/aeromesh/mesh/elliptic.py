"""Elliptic smoothing of a structured block (Thompson, Thames & Mastin 1974).

Solves, by point relaxation with every boundary node held,

    a (r_xixi + P r_xi) - 2 b r_xieta + c (r_etaeta + Q r_eta) = 0,
    a = |r_eta|^2,  b = r_xi . r_eta,  c = |r_xi|^2,

with ``P = Q = 0`` (Winslow) or with control functions taken from the grid
itself, ``P = -r_xi.r_xixi / |r_xi|^2``, ``Q = -r_eta.r_etaeta / |r_eta|^2``,
frozen at the start ("grid-preserving" sources in the spirit of
Thomas & Middlecoff 1980): the input grid then satisfies the equations
exactly along each line direction, so the smoother changes only what the
sources do not pin -- the cross-coupling -- and keeps the spacing.

Kept as a measured experiment: on the marched C-mesh it does not beat the
march near the wall (see ``tests`` and the report). ``region`` limits the
smoothing to nodes inside a box, with a weight blended to zero at its edge.
"""
from __future__ import annotations

import numpy as np


def smooth(grid: np.ndarray, iters: int = 200, omega: float = 0.5,
           sources: bool = True, region=None, hold_rows: int = 1
           ) -> np.ndarray:
    """Return a smoothed copy of ``grid`` (``(ni, nj, 2)``).

    ``hold_rows`` rows next to ``j = 0`` are held as well as the boundary
    (``1`` holds only the wall row).
    """
    g = np.array(grid, dtype=float)
    ni, nj = g.shape[:2]
    if sources:
        rx = 0.5 * (g[2:, 1:-1] - g[:-2, 1:-1])
        ry = 0.5 * (g[1:-1, 2:] - g[1:-1, :-2])
        rxx = g[2:, 1:-1] - 2 * g[1:-1, 1:-1] + g[:-2, 1:-1]
        ryy = g[1:-1, 2:] - 2 * g[1:-1, 1:-1] + g[1:-1, :-2]
        P = -np.einsum("ijk,ijk->ij", rx, rxx) / np.maximum(
            np.einsum("ijk,ijk->ij", rx, rx), 1e-300)
        Q = -np.einsum("ijk,ijk->ij", ry, ryy) / np.maximum(
            np.einsum("ijk,ijk->ij", ry, ry), 1e-300)
    else:
        P = Q = 0.0
    w = np.ones((ni - 2, nj - 2))
    if region is not None:
        (x0, x1), (y0, y1) = region
        c = g[1:-1, 1:-1]
        dx = np.minimum(c[..., 0] - x0, x1 - c[..., 0]) / (x1 - x0)
        dy = np.minimum(c[..., 1] - y0, y1 - c[..., 1]) / (y1 - y0)
        w = np.clip(4.0 * np.minimum(dx, dy), 0.0, 1.0)
    if hold_rows > 1:
        w[:, :hold_rows - 1] = 0.0
    for _ in range(int(iters)):
        rx = 0.5 * (g[2:, 1:-1] - g[:-2, 1:-1])
        ry = 0.5 * (g[1:-1, 2:] - g[1:-1, :-2])
        a = np.einsum("ijk,ijk->ij", ry, ry)
        b = np.einsum("ijk,ijk->ij", rx, ry)
        c = np.einsum("ijk,ijk->ij", rx, rx)
        cross = 0.25 * (g[2:, 2:] - g[2:, :-2] - g[:-2, 2:] + g[:-2, :-2])
        num = (a[..., None] * (g[2:, 1:-1] + g[:-2, 1:-1]
                               + (P[..., None] if sources else 0.0) * rx)
               + c[..., None] * (g[1:-1, 2:] + g[1:-1, :-2]
                                 + (Q[..., None] if sources else 0.0) * ry)
               - 2.0 * b[..., None] * cross)
        new = num / (2.0 * (a + c))[..., None]
        g[1:-1, 1:-1] += omega * w[..., None] * (new - g[1:-1, 1:-1])
    return g
