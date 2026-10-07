"""Decomposition splits.

Every candidate split is a curve of the medial coordinate system, never a
hand-placed polyline:

    constant s   a medial radius pair, running T1 -> p(s) -> T2
    constant rho  a level set of rho = d_wall / r_m(s), offset from the wall

A medial radius meets the boundary perpendicularly by definition, so a
constant-s split is orthogonal to the wall at both ends by construction. That
property is the reason the parameterisation is worth having, and it is not
something the optimiser later has to discover.

Candidates are ranked the way Fogg ranks them (sec. 4.1): an ``n = 0`` split is
preferred over ``n = 1`` because it does not create a concave corner, and within
each, medial angles nearer pi and pi/2 respectively are the more favourable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aeromesh.medial.fields import optimum_flow_index


@dataclass(frozen=True)
class Split:
    """One decomposition split across a medial edge.

    Attributes
    ----------
    edge : tuple
        ``(u, v, key)`` of the medial edge it cuts.
    index : int
        Sample index along that edge's polyline.
    s : float
        Arc length along the edge.
    curve : np.ndarray, shape (M, 2)
        The split polyline, from one touch point through the medial point to the
        other.
    touch_index : tuple[int, int]
        Boundary sample index of each end, so the blocks either side can walk
        the arc between consecutive splits.
    theta_m : float
    n : int
        Fogg's optimum flow index across the split.
    rank : float
        Lower is better. See :func:`split_rank`.
    """

    edge: tuple
    index: int
    s: float
    curve: np.ndarray
    touch_index: tuple
    theta_m: float
    n: int
    rank: float


def split_rank(theta_m: float) -> float:
    """Fogg's preference, as a number where lower is better.

    ``n = 0`` splits come first because they leave a logically straight edge
    rather than a concave corner; within a class, the closer ``theta_m`` is to
    the ideal for that class -- pi for ``n = 0``, pi/2 for ``n = 1`` -- the
    better. ``n = 2`` is a thin slot, where a split is a poor idea at all.
    """
    n = int(optimum_flow_index(theta_m))
    ideal = {0: np.pi, 1: np.pi / 2}.get(n)
    if ideal is None:
        return 100.0 + abs(theta_m)
    return n * 10.0 + abs(theta_m - ideal)


def radius_split(edge_data: dict, index: int, edge_key: tuple,
                 n_curve: int = 33) -> Split:
    """The constant-s split at one sample of a medial edge.

    Two straight runs, ``T1 -> p`` and ``p -> T2``, resampled so the curve has a
    usable point count for meshing. Both meet the wall at a right angle.
    """
    i = int(np.clip(index, 0, len(edge_data["polyline"]) - 1))
    p = edge_data["polyline"][i]
    a, b = edge_data["t1"][i], edge_data["t2"][i]
    half = max(2, n_curve // 2)
    ta = np.linspace(0.0, 1.0, half, endpoint=False)[:, None]
    tb = np.linspace(0.0, 1.0, half + 1)[:, None]
    curve = np.vstack([a + ta * (p - a), p + tb * (b - p)])
    theta = float(edge_data["theta_m"][i])
    return Split(edge=edge_key, index=i, s=float(edge_data["s"][i]), curve=curve,
                 touch_index=(int(edge_data["i1"][i]), int(edge_data["i2"][i])),
                 theta_m=theta, n=int(optimum_flow_index(theta)),
                 rank=split_rank(theta))


def candidate_splits(axis, edge_key: tuple, stride: int = 1,
                     margin: float = 0.0) -> list[Split]:
    """Every admissible constant-s split on one medial edge, best first.

    ``margin`` keeps candidates clear of the edge's ends as a fraction of its
    length, so a split is not proposed on top of a medial vertex, where the
    radius pair is degenerate and the vertex's own balance applies instead.
    """
    u, v, key = edge_key
    d = axis.graph.edges[u, v, key]
    n = len(d["polyline"])
    lo = int(np.ceil(margin * (n - 1)))
    hi = int(np.floor((1.0 - margin) * (n - 1)))
    out = [radius_split(d, i, edge_key) for i in range(lo, hi + 1, max(1, stride))]
    return sorted(out, key=lambda s: s.rank)


def best_split(axis, edge_key: tuple, near_s: float | None = None,
               margin: float = 0.05, window: float | None = None) -> Split | None:
    """The best-ranked split on an edge, optionally restricted to near ``near_s``.

    Returns None when the edge offers no admissible candidate, which is the
    condition under which Fogg fixes a singularity into a block corner instead.
    """
    cands = candidate_splits(axis, edge_key, margin=margin)
    if near_s is not None and window is not None:
        cands = [c for c in cands if abs(c.s - near_s) <= window] or cands
    return cands[0] if cands else None
